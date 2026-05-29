import asyncio
import os
import time
import numpy as np
import torch
import random
from typing import Dict, Any, List

import tinker
from tinker.types import AdamParams, ModelInput, SamplingParams
import wandb
import requests
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer
from tenacity import retry, stop_after_attempt, wait_exponential

from tinker_atropos.types import (
    GenerateRequest,
    GenerateResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
)
from tinker_atropos.config import TinkerAtroposConfig


class TinkerAtroposTrainer:
    """
    Trainer that handles both RL training and inference through Tinker API.
    Connects to Atropos Trajectory API to coordinate environment interaciton.
    """

    def __init__(self, config: TinkerAtroposConfig):
        self.config = config

        # Model and training config
        self.base_model = config.base_model
        self.lora_rank = config.lora_rank
        self.learning_rate = config.learning_rate
        self.atropos_api_url = config.atropos_api_url
        self.num_steps = config.num_steps

        # Tinker clients
        self.service_client = None
        self.training_client = None
        self.current_sampling_client = None

        self.tokenizer = None

        # Atropos registration
        self.trainer_id = None
        self.group_mean_rewards = []
        self.wandb_group = None

    async def setup(self):
        print("Setting up Tinker-Atropos Trainer...")

        # Create single ServiceClient for both training and inference
        print(f"Creating ServiceClient for {self.base_model}...")
        self.service_client = tinker.ServiceClient()

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        print(f"Loaded tokenizer for {self.base_model}")

        # Create LoRA training client
        print("Creating training client...")
        self.training_client = await self.service_client.create_lora_training_client_async(
            base_model=self.base_model,
            rank=self.lora_rank,
        )
        print("Training client created")

        # Save initial weights and create sampling client
        print("Saving initial weights...")
        initial_path = self.training_client.save_weights_for_sampler(name="step_0").result().path
        self.current_sampling_client = self.service_client.create_sampling_client(
            model_path=initial_path
        )
        print(f"Initial sampling client created: {initial_path}")

        self.wandb_group = self.config.wandb_group or wandb.sdk.lib.runid.generate_id()

        print("Registering with Atropos API...")
        self.trainer_id = await self._register_trainer()
        print(f"Registered as trainer: {self.trainer_id}")

        if self.config.use_wandb:
            try:
                wandb.init(
                    project=self.config.wandb_project,
                    name=f"{self.config.wandb_run_name}-trainer-{self.config.wandb_run_suffix}",
                    group=self.wandb_group,
                    tags=["trainer"],
                )
                print(f"Wandb initialized (trainer): {wandb.run.name} in group: {self.wandb_group}")
            except Exception as e:
                print(f"Error initializing wandb: {e}")
                self.config.use_wandb = False

    async def _register_trainer(self) -> str:
        """Register this trainer with the Atropos API server."""
        url = f"{self.atropos_api_url}/register"

        payload = {
            "wandb_project": self.config.wandb_project,
            "wandb_group": self.wandb_group,
            "batch_size": self.config.batch_size,
            "max_token_len": self.config.max_token_trainer_length,
            "starting_step": 0,
            "checkpoint_dir": self.config.checkpoint_dir,
            "save_checkpoint_interval": self.config.save_checkpoint_interval,
            "num_steps": self.num_steps,
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        result = response.json()
        return result.get("uuid")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
    def get_batch(self):
        """Fetch a batch of rollouts from Atropos API with retry logic."""
        data = requests.get(f"{self.atropos_api_url}/batch", timeout=10).json()
        return data

    def pad_data_to_good_offset(
        self, data: Dict[str, Any]
    ) -> tuple[List[tinker.Datum], List[float]]:
        """
        Convert Atropos batch into Tinker Datums for training.

        Pads logprobs and advantages to align with token sequences:
        - Prompt tokens get 0.0 for logprobs and advantages (no gradient)
        - Generated tokens get actual logprobs and advantages
        """
        batch = data["batch"]

        datums = []
        group_mean_rewards = []
        all_reference_logprobs = []
        all_advantages = []
        skipped_count = 0

        for item in batch:
            # Calculate advantages
            scores = np.array(item["scores"])
            original_mean = np.mean(scores)
            advantages = scores - original_mean

            group_mean_rewards.append(original_mean)

            # Skip groups where all advantages are zero
            if len(scores) > 1 and np.all(advantages == 0.0):
                skipped_count += 1
                continue

            # Apply advantage overrides
            if item.get("overrides") is not None:
                for i in range(len(item["overrides"])):
                    if item["overrides"][i].get("set_advantage_to_zero", False):
                        advantages[i] = 0.0

            for i in range(len(item["tokens"])):
                tokens = item["tokens"][i]
                trajectory_logprobs = item["inference_logprobs"][i]
                advantage = advantages[i]

                all_advantages.append(advantage)

                input_tokens = tokens[:-1]
                target_tokens = tokens[1:]

                all_logprobs = trajectory_logprobs[1:]  # Shift right to align with targets

                all_advantages_padded = [0.0 if lp == 1.0 else advantage for lp in all_logprobs]

                all_reference_logprobs.extend(all_logprobs)

                assert (
                    len(input_tokens)
                    == len(target_tokens)
                    == len(all_logprobs)
                    == len(all_advantages_padded)
                ), f"Length mismatch: input={len(input_tokens)}, target={len(target_tokens)}, logprobs={len(all_logprobs)}, advantages={len(all_advantages_padded)}"

                datum = tinker.Datum(
                    model_input=tinker.ModelInput.from_ints(tokens=input_tokens),
                    loss_fn_inputs={
                        "target_tokens": tinker.TensorData.from_torch(
                            torch.tensor(target_tokens, dtype=torch.int64)
                        ),
                        "logprobs": tinker.TensorData.from_torch(
                            torch.tensor(all_logprobs, dtype=torch.float32)
                        ),
                        "advantages": tinker.TensorData.from_torch(
                            torch.tensor(all_advantages_padded, dtype=torch.float32)
                        ),
                    },
                )
                datums.append(datum)

        # Calculate logprob stats
        if all_reference_logprobs:
            logprob_array = np.array(all_reference_logprobs)
            # Filter out both 0.0 and 1.0 (1.0 are placeholder values for prompt tokens)
            logprob_array_actual = logprob_array[(logprob_array != 0.0) & (logprob_array != 1.0)]
            if len(logprob_array_actual) > 0:
                self.logprob_stats = {
                    "logprobs/mean": float(np.mean(logprob_array_actual)),
                    "logprobs/std": float(np.std(logprob_array_actual)),
                    "logprobs/min": float(np.min(logprob_array_actual)),
                    "logprobs/p50": float(np.percentile(logprob_array_actual, 50)),
                }
            else:
                self.logprob_stats = {}
        else:
            self.logprob_stats = {}

        # Calculate advantage stats
        if all_advantages:
            advantages_array = np.array(all_advantages)
            if np.std(advantages_array) > 1e-6:
                self.advantage_stats = {
                    "advantages/mean": float(np.mean(advantages_array)),
                    "advantages/std": float(np.std(advantages_array)),
                    "advantages/sum": float(np.sum(advantages_array)),
                }
            else:
                self.advantage_stats = {}
        else:
            self.advantage_stats = {}

        if skipped_count > 0:
            print(f"Skipped {skipped_count} groups with zero advantages")

        return datums, group_mean_rewards

    def get_data(self) -> List[tinker.Datum]:
        """
        Poll Atropos for a batch of rollouts and convert to Tinker Datums.
        Waits until a batch is available.
        """
        import time
        import json

        while True:
            data = self.get_batch()

            if data.get("batch") is not None:
                with open("temp.json", "w", encoding="utf-8") as f:
                    json.dump(data, f)

                datums, group_mean_rewards = self.pad_data_to_good_offset(data)
                self.group_mean_rewards = group_mean_rewards
                return datums
            else:
                time.sleep(1)

    async def train_step(self, step: int) -> Dict[str, Any]:
        """Execute one training step: fetch batch, forward-backward, optimizer step."""
        print(f"\n{'='*60}")
        print(f"Step {step}/{self.num_steps}")
        print(f"{'='*60}")

        step_start = time.time()
        metrics = {"step": step}

        # Fetch batch from Atropos
        print("Fetching data from Atropos...")
        data = self.get_data()
        print(f"Got {len(data)} Datum objects")

        # Forward-backward pass with importance sampling loss
        print("Running forward-backward pass...")
        fwd_bwd_result = await self.training_client.forward_backward_async(
            data, loss_fn="importance_sampling"
        )

        # Optimizer step
        print("Running optimizer step...")
        adam_params = AdamParams(learning_rate=self.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
        optim_result = await self.training_client.optim_step_async(adam_params)

        fwd_bwd_result = await fwd_bwd_result.result_async()
        optim_result = await optim_result.result_async()

        loss_val = (
            fwd_bwd_result.metrics["loss:sum"] if "loss:sum" in fwd_bwd_result.metrics else 0.0
        )

        print(f"Loss: {loss_val}")

        # Calculate training logprob stats
        training_logprobs_all = []
        for datum, output in zip(data, fwd_bwd_result.loss_fn_outputs):
            training_logprobs = output["logprobs"].to_torch()
            advantages = datum.loss_fn_inputs["advantages"].to_torch()
            mask = advantages != 0.0
            training_lp_masked = training_logprobs[mask]
            training_logprobs_all.extend(training_lp_masked.cpu().numpy().tolist())

        if training_logprobs_all:
            training_lp_array = np.array(training_logprobs_all)
            self.training_logprob_stats = {
                "logprobs/mean_training": float(np.mean(training_lp_array)),
                "logprobs/std_training": float(np.std(training_lp_array)),
                "logprobs/min_training": float(np.min(training_lp_array)),
                "logprobs/p50_training": float(np.percentile(training_lp_array, 50)),
            }

            # Calculate logprob drift
            if hasattr(self, "logprob_stats") and "logprobs/mean" in self.logprob_stats:
                ref_mean = self.logprob_stats["logprobs/mean"]
                train_mean = float(np.mean(training_lp_array))
                self.training_logprob_stats["logprobs/diff"] = ref_mean - train_mean
        else:
            self.training_logprob_stats = {}

        # Update sampling client with new weights
        print("Saving checkpoint and updating sampling client...")
        new_path = (
            self.training_client.save_weights_for_sampler(name=f"step_{step+1}").result().path
        )
        self.current_sampling_client = self.service_client.create_sampling_client(
            model_path=new_path
        )
        print(f"Sampling client updated: {new_path}")

        step_time = time.time() - step_start
        metrics["step_time"] = step_time
        metrics["learning_rate"] = self.learning_rate
        metrics["loss"] = loss_val

        if self.group_mean_rewards:
            metrics["reward/mean"] = np.mean(self.group_mean_rewards)
            print(f"Reward/mean: {metrics['reward/mean']:.4f}")

        if self.config.use_wandb:
            wandb_metrics = {
                "train/loss": loss_val,
                "train/learning_rate": self.learning_rate,
                "reward/mean": metrics["reward/mean"],
            }

            if hasattr(self, "logprob_stats"):
                wandb_metrics.update(self.logprob_stats)
            if hasattr(self, "training_logprob_stats"):
                wandb_metrics.update(self.training_logprob_stats)
            if hasattr(self, "advantage_stats"):
                wandb_metrics.update(self.advantage_stats)

            wandb.log(wandb_metrics, step=step + 1)

        return metrics

    async def run(self):
        """Main training loop."""
        print("\n" + "=" * 60)
        print("Starting Tinker-Atropos Training")
        print("=" * 60 + "\n")

        await self.setup()

        for step in range(self.num_steps):
            try:
                metrics = await self.train_step(step)
                print(f"\nStep {step} complete - Loss: {metrics.get('loss', 'N/A')}")
            except Exception as e:
                print(f"Error in step {step}: {e}")
                import traceback

                traceback.print_exc()
                break

        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60 + "\n")

        print(
            f"Final weights are available here: tinker://{str(self.training_client.model_id)}/sampler_weights/final"
        )


trainer: TinkerAtroposTrainer | None = None

# FastAPI server for Atropos environment to call for inference
app = FastAPI(title="Tinker-Atropos Service")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "trainer_initialized": trainer is not None,
    }


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest):
    """
    OpenAI-compatible completions endpoint.
    Called by inference server wrapper for regular completions (non-chat).
    """
    if trainer is None:
        raise HTTPException(status_code=503, detail="Trainer not initialized")

    try:
        # Handle single prompt (string) or batch (list of strings)
        if isinstance(request.prompt, str):
            prompts = [request.prompt]
        else:
            prompts = request.prompt

        all_choices = []
        choice_index = 0

        for prompt in prompts:
            # Tokenize prompt
            prompt_tokens = trainer.tokenizer.encode(prompt, add_special_tokens=False)
            model_input = ModelInput.from_ints(prompt_tokens)

            # Generate using Tinker sampling client
            sampling_params = SamplingParams(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stop=request.stop if request.stop else [],
            )

            result = await trainer.current_sampling_client.sample_async(
                prompt=model_input,
                sampling_params=sampling_params,
                num_samples=request.n,
            )

            # Format choices
            for sequence in result.sequences:
                output_text = trainer.tokenizer.decode(sequence.tokens, skip_special_tokens=True)
                all_choices.append(
                    {
                        "text": output_text,
                        "index": choice_index,
                        "finish_reason": "stop",
                    }
                )
                choice_index += 1

        return CompletionResponse(
            id=f"cmpl-{random.randint(0, 999999)}",
            choices=all_choices,
            created=int(time.time()),
            model=trainer.base_model,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Completion failed: {str(e)}")


@app.get("/wandb_info")
async def wandb_info():
    if trainer is None:
        raise HTTPException(status_code=503, detail="Trainer not initialized")

    return {
        "group": trainer.wandb_group,
        "project": trainer.config.wandb_project,
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    Called by inference server wrapper for chat completions.
    """
    if trainer is None:
        raise HTTPException(status_code=503, detail="Trainer not initialized")

    try:
        messages_dict = [{"role": msg.role, "content": msg.content} for msg in request.messages]

        # Apply chat template and tokenize
        prompt_text = trainer.tokenizer.apply_chat_template(
            messages_dict, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = trainer.tokenizer.encode(prompt_text, add_special_tokens=False)
        model_input = ModelInput.from_ints(prompt_tokens)

        # Generate using Tinker sampling client
        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=request.stop if request.stop else [],
        )

        result = await trainer.current_sampling_client.sample_async(
            prompt=model_input,
            sampling_params=sampling_params,
            num_samples=request.n,
        )

        # Format as OpenAI response
        choices = []
        for i, sequence in enumerate(result.sequences):
            output_text = trainer.tokenizer.decode(sequence.tokens, skip_special_tokens=True)
            choices.append(
                {
                    "message": {
                        "role": "assistant",
                        "content": output_text,
                    },
                    "index": i,
                    "finish_reason": "stop",
                }
            )

        return ChatCompletionResponse(
            id=f"chatcmpl-{random.randint(0, 999999)}",
            choices=choices,
            created=int(time.time()),
            model=trainer.base_model,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")


@app.post("/generate", response_model=GenerateResponse | List[GenerateResponse])
async def generate(request: GenerateRequest):
    """
    /generate endpoint for ManagedServer.
    Called by ManagedServer with tokenized input_ids.
    Returns GenerateResponse for single completion (n=1) or List[GenerateResponse] for multiple (n>1).
    """
    if trainer is None:
        raise HTTPException(status_code=503, detail="Trainer not initialized")

    try:
        # Extract input_ids (ManagedServer sends tokenized input)
        if request.input_ids is None:
            raise HTTPException(status_code=400, detail="input_ids is required")

        prompt_tokens = request.input_ids

        # Extract sampling params
        sampling_params = request.sampling_params or {}
        n = sampling_params.get("n", 1)
        max_tokens = sampling_params.get("max_new_tokens", 256)
        temperature = sampling_params.get("temperature", 1.0)
        stop = sampling_params.get("stop", [])

        # Generate using Tinker sampling client
        model_input = ModelInput.from_ints(prompt_tokens)
        tinker_sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop if isinstance(stop, list) else [stop],
        )

        result = await trainer.current_sampling_client.sample_async(
            prompt=model_input,
            sampling_params=tinker_sampling_params,
            num_samples=n,
        )

        if n == 1:
            sequence = result.sequences[0]
            output_tokens = sequence.tokens
            output_logprobs = sequence.logprobs if sequence.logprobs else []
            output_text = trainer.tokenizer.decode(output_tokens, skip_special_tokens=True)

            output_token_logprobs = []
            for token_id, logprob in zip(output_tokens, output_logprobs):
                token_text = trainer.tokenizer.decode([token_id])
                output_token_logprobs.append((logprob, token_id, token_text))

            return GenerateResponse(
                text=output_text,
                meta_info={
                    "prompt_tokens": len(prompt_tokens),
                    "completion_tokens": len(output_tokens),
                    "finish_reason": "stop",
                    "output_token_logprobs": output_token_logprobs,
                },
            )
        else:
            # Multiple completions - return list of GenerateResponse objects
            results = []
            for sequence in result.sequences:
                output_tokens = sequence.tokens
                output_logprobs = sequence.logprobs if sequence.logprobs else []
                output_text = trainer.tokenizer.decode(output_tokens, skip_special_tokens=True)

                # Format logprobs for response
                output_token_logprobs = []
                for token_id, logprob in zip(output_tokens, output_logprobs):
                    token_text = trainer.tokenizer.decode([token_id])
                    output_token_logprobs.append((logprob, token_id, token_text))

                results.append(
                    GenerateResponse(
                        text=output_text,
                        meta_info={
                            "prompt_tokens": len(prompt_tokens),
                            "completion_tokens": len(output_tokens),
                            "finish_reason": "stop",
                            "output_token_logprobs": output_token_logprobs,
                        },
                    )
                )

            return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


def run_fastapi_server():
    """Run FastAPI server in background thread."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")


async def main():
    global trainer

    config = TinkerAtroposConfig(
        lora_rank=int(os.getenv("LORA_RANK", "32")),
        learning_rate=float(os.getenv("LEARNING_RATE", "4e-5")),
        num_steps=50,
    )

    print(f"Using wandb run: {config.wandb_run_name}")

    trainer = TinkerAtroposTrainer(config)

    # Start FastAPI server in background thread for Atropos to call
    import threading

    server_thread = threading.Thread(target=run_fastapi_server, daemon=True)
    server_thread.start()

    print("Waiting for FastAPI server to start...")
    await asyncio.sleep(3)

    await trainer.run()


if __name__ == "__main__":
    asyncio.run(main())
