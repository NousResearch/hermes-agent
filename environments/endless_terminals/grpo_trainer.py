"""
GRPO trainer for Endless Terminals using obiwan96/qwen3-8b-openthinker-sft-endless-terminals.

Full fine-tuning (no LoRA) on top of the paper's SFT model.
Talks to atropos rollout API at :8000, serves inference via vLLM at :9001.
ManagedServer (in the env) handles tool call parsing client-side.

Usage:
  Terminal 1: run-api   (atropos API server)
  Terminal 2: python environments/endless_terminals/grpo_trainer.py
  Terminal 3: python environments/endless_terminals/endless_terminals_env.py serve \
                --config environments/endless_terminals/tinker_qwen.yaml
"""

import atexit
import json
import math
import os
import shutil
import subprocess
import time
from typing import List, Optional, Tuple

import numpy as np
import requests
import torch
import torch.nn.functional as F
import wandb
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "obiwan96/qwen3-8b-openthinker-sft-endless-terminals"
ATROPOS_URL = "http://localhost:8000"
VLLM_PORT = 9001

vllm_process = None


def cleanup_vllm():
    global vllm_process
    if vllm_process:
        print("\nTerminating vLLM process...")
        vllm_process.terminate()
        try:
            vllm_process.wait(timeout=10)
            print("vLLM process terminated.")
        except subprocess.TimeoutExpired:
            vllm_process.kill()
            vllm_process.wait()
            print("vLLM process killed.")
        vllm_process = None


atexit.register(cleanup_vllm)


class TrainingConfig(BaseModel):
    model_name: str = MODEL_NAME
    lr: float = 1e-6
    training_steps: int = 500
    batch_size: int = 2
    seq_len: int = 16384          # Match paper's 16k context window
    gradient_accumulation_steps: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "./checkpoints-openthinker"
    vllm_restart_interval: int = 1  # Restart vLLM every step (always on-policy)
    vllm_port: int = VLLM_PORT
    use_wandb: bool = True
    wandb_project: str = "endless-terminals"
    wandb_run_name: str = "openthinker-sft-grpo"


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=15))
def register_trainer(config: TrainingConfig):
    requests.post(
        f"{ATROPOS_URL}/register",
        json={
            "wandb_group": config.wandb_run_name,
            "wandb_project": config.wandb_project,
            "batch_size": config.batch_size * config.gradient_accumulation_steps,
            "max_token_len": config.seq_len,
            "starting_step": 0,
            "checkpoint_dir": config.save_path,
            "save_checkpoint_interval": config.training_steps,
            "num_steps": config.training_steps,
        },
        timeout=10,
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def get_batch():
    return requests.get(f"{ATROPOS_URL}/batch", timeout=10).json()


def pad_data(data, batch_size: int):
    max_token_len = max(
        max(len(x) for x in item["tokens"]) for item in data["batch"]
    )
    good_multiple = 64
    if (max_token_len - 1) % good_multiple != 0:
        max_token_len = math.ceil((max_token_len - 1) / good_multiple) * good_multiple
        token_setup_len = max_token_len + 1
    else:
        token_setup_len = max_token_len
        max_token_len = max_token_len - 1

    input_ids, labels, advantages, temperatures = [], [], [], []

    for item in data["batch"]:
        scores = np.array(item["scores"], dtype=np.float32)
        if len(scores) > 1:
            scores = scores - scores.mean()
            scores = scores / max(scores.std(), 1e-8)

        if item.get("overrides"):
            for i, ov in enumerate(item["overrides"]):
                if ov and ov.get("set_advantage_to_zero"):
                    scores[i] = 0.0

        for i, tokens in enumerate(item["tokens"]):
            n_gen = sum(1 for m in item["masks"][i] if m != -100)

            label_item = np.concatenate([
                np.array(item["masks"][i]),
                np.full(max(0, token_setup_len - len(tokens)), -100, dtype=np.int32),
            ])
            tokens_padded = np.concatenate([
                np.array(tokens),
                np.zeros(max(0, token_setup_len - len(tokens)), dtype=np.int32),
            ])

            input_ids.append(tokens_padded[:-1])
            labels.append(label_item[1:])

            # Sequence-level averaging: divide advantage by n_gen tokens
            seq_advantage = scores[i] / max(n_gen, 1)
            advantages.append(seq_advantage)

            t = 1.0
            if item.get("overrides") and i < len(item["overrides"]):
                ov = item["overrides"][i]
                if isinstance(ov, dict) and "temperature" in ov:
                    t = float(ov["temperature"])
            elif item.get("generation_params") and "temperature" in item["generation_params"]:
                t = float(item["generation_params"]["temperature"])
            temperatures.append(t)

    token_batches, label_batches, adv_batches, temp_batches = [], [], [], []
    for i in range(len(input_ids) // batch_size):
        s, e = i * batch_size, (i + 1) * batch_size
        token_batches.append(torch.tensor(np.stack(input_ids[s:e])))
        label_batches.append(torch.tensor(np.stack(labels[s:e])))
        adv_batches.append(torch.tensor(np.array(advantages[s:e], dtype=np.float32)).view(-1, 1))
        temp_batches.append(torch.tensor(np.array(temperatures[s:e], dtype=np.float32)).view(-1, 1, 1))

    return token_batches, label_batches, adv_batches, temp_batches


def get_data(batch_size: int, seq_len: int):
    while True:
        data = get_batch()
        if data.get("batch") is not None:
            with open("temp.json", "w") as f:
                json.dump(data, f)
            return pad_data(data, batch_size)
        time.sleep(1)


def launch_vllm(model_path: str, port: int):
    global vllm_process
    cleanup_vllm()
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--dtype", "bfloat16",
        "--gpu-memory-utilization", "0.45",
        "--disable-log-requests",
        "--served-model-name", MODEL_NAME,
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
    ]
    print(f"Launching vLLM: {' '.join(cmd)}")
    vllm_process = subprocess.Popen(cmd)
    print(f"vLLM PID: {vllm_process.pid}")
    # Give vLLM time to start
    time.sleep(30)


def train(config: TrainingConfig):
    global vllm_process

    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=config.dict(),
        )

    print(f"Loading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16)
    model.to(config.device)
    model.gradient_checkpointing_enable()
    model.train()

    optimizer = AdamW(model.parameters(), lr=config.lr, betas=(0.9, 0.95), eps=1e-8)

    os.makedirs(config.save_path, exist_ok=True)
    register_trainer(config)

    # Save initial weights and launch vLLM
    initial_path = os.path.join(config.save_path, "step_0")
    print(f"Saving initial weights to {initial_path}...")
    model.save_pretrained(initial_path)
    tokenizer.save_pretrained(initial_path)
    launch_vllm(initial_path, config.vllm_port)

    for step in range(config.training_steps):
        print(f"\n{'='*50}\nStep {step+1}/{config.training_steps}\n{'='*50}")

        token_batches, label_batches, adv_batches, temp_batches = get_data(
            config.batch_size, config.seq_len
        )

        total_loss = 0.0
        optimizer.zero_grad()

        for tokens, labels, advs, temps in zip(token_batches, label_batches, adv_batches, temp_batches):
            tokens = tokens.to(config.device)
            labels = labels.to(config.device)
            advs = advs.to(config.device)

            outputs = model(tokens)
            logits = outputs.logits

            t = temps.to(logits.device, logits.dtype)
            t = torch.where(t <= 0, torch.ones_like(t), t)
            logits = logits / t

            logp_per_token = -F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="none",
                ignore_index=-100,
            ).view(labels.shape)

            mask = (labels != -100).float()
            grpo_loss = (
                ((-torch.exp(logp_per_token - logp_per_token.detach()) * mask).sum(-1) / mask.sum(-1).clamp_min(1))
                * advs.squeeze(-1)
            ).mean() / config.gradient_accumulation_steps

            grpo_loss.backward()
            total_loss += grpo_loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        print(f"Loss: {total_loss:.4f} | Grad norm: {grad_norm:.4f}")

        if config.use_wandb:
            wandb.log({"train/loss": total_loss, "train/grad_norm": grad_norm.item()}, step=step + 1)

        # Save checkpoint and restart vLLM with updated weights
        checkpoint_path = os.path.join(config.save_path, f"step_{step+1}")
        print(f"Saving checkpoint to {checkpoint_path}...")
        if os.path.exists(checkpoint_path):
            shutil.rmtree(checkpoint_path)
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        launch_vllm(checkpoint_path, config.vllm_port)

    print("Training complete.")
    if config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    config = TrainingConfig()
    train(config)
