import random
import time
from typing import Dict, List, Optional, Tuple, TypedDict, Union

from datasets import load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Item
from tinker_atropos.config import TinkerAtroposConfig

CONFIG_PATH = "configs/default.yaml"

question_suffix = " Provide a numerical answer without units, written inside \\boxed{}."

convo_prefix = [
    {
        "role": "user",
        "content": "How many r's are in strawberry?" + question_suffix,
    },
    {
        "role": "assistant",
        "content": "Let's spell the word out and number all the letters: 1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. We have r's at positions 3, 8, and 9. \\boxed{3}",
    },
]


class GSM8kRow(TypedDict):
    question: str
    answer: str


class GSM8kEnv(BaseEnv):
    """
    Atropos environment for GSM8k math problems.
    Generates rollouts, scores them, and provides training batches.
    """

    name = "gsm8k"

    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = list()
        self.eval_metrics = list()
        # Add tracking for wandb visualizations
        self.rollouts_for_wandb = []
        self.completion_lengths = []

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        config = (
            TinkerAtroposConfig.from_yaml(CONFIG_PATH) if CONFIG_PATH else TinkerAtroposConfig()
        )

        env_config = BaseEnvConfig(
            tokenizer_name=config.base_model,
            group_size=config.group_size,
            use_wandb=config.use_wandb,
            rollout_server_url=config.atropos_api_url,
            total_steps=config.num_steps,
            batch_size=config.batch_size,
            steps_per_eval=config.steps_per_eval,
            max_token_length=config.max_token_env_length,
            max_num_workers=config.max_num_workers,
            max_batches_offpolicy=config.max_batches_offpolicy,
            wandb_name=f"{config.wandb_run_name}-env",
            ensure_scores_are_not_same=config.ensure_scores_are_not_same,
        )
        server_configs = [
            APIServerConfig(
                model_name=config.base_model,
                base_url=config.inference_api_url + "/v1",
                api_key="x",
                server_type="sglang",
                num_requests_for_eval=config.num_requests_for_eval,
            ),
        ]

        return env_config, server_configs

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        try:
            wandb_metrics["train/percent_correct"] = sum(self.percent_correct_buffer) / len(
                self.percent_correct_buffer
            )
        except ZeroDivisionError:
            # Skip if buffer is empty
            pass

        self.percent_correct_buffer = list()
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()
        # Call the parent method to handle the server metrics
        await super().wandb_log(wandb_metrics)

    async def setup(self):
        # Ensure tokenizer has a chat template
        if self.tokenizer.chat_template is None:
            # Set default Llama-style chat template
            self.tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                "{{ '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
                "{% elif message['role'] == 'user' %}"
                "{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
                "{% elif message['role'] == 'assistant' %}"
                "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
                "{% endif %}"
                "{% if loop.last and message['role'] != 'assistant' %}"
                "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
                "{% endif %}"
                "{% endfor %}"
            )

        self.train = load_dataset("gsm8k", "main", split="train").shuffle(seed=42)
        test_data = load_dataset("gsm8k", "main", split="test").shuffle(seed=42)
        self.test = list()
        for item in test_data:
            self.test.append(
                {
                    "question": item["question"],
                    "gold_answer": item["answer"].split("#")[-1].strip().replace(",", ""),
                }
            )
        self.iter = 0

    def save_checkpoint(self, step, data=None):
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    async def rollout_and_score_eval(self, question: str, answer: str) -> dict:
        """
        Generate and score a single evaluation rollout.
        """
        completion = await self.server.chat_completion(
            messages=[
                *convo_prefix,
                {"role": "user", "content": question + question_suffix},
            ],
            n=1,
            max_tokens=self.config.max_token_length,
            temperature=0.0,
            split="eval",
        )

        response_content = completion.choices[0].message.content

        # Parse gold answer
        gold_parsed = parse(
            "\\boxed{" + answer + "}",
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )

        # Parse model answer
        answer_parsed = parse(
            response_content.split("</think>")[-1],
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )

        score = 1 if verify(answer_parsed, gold_parsed) else 0

        sample = {
            "messages": [
                *convo_prefix,
                {"role": "user", "content": question + question_suffix},
                {"role": "assistant", "content": response_content},
            ],
            "question": question,
            "gold_answer": answer,
            "gold_parsed": str(gold_parsed) if gold_parsed else None,
            "model_parsed": str(answer_parsed) if answer_parsed else None,
            "score": int(score),
            "correct": bool(score),
            "finish_reason": completion.choices[0].finish_reason,
            "response_after_think": (
                response_content.split("</think>")[-1]
                if "</think>" in response_content
                else response_content
            ),
        }

        return {"score": score, "sample": sample}

    async def evaluate(self, *args, **kwargs):
        """
        Top level evaluation method call with metrics and logging
        """
        start_time = time.time()

        # Generate and score all test examples
        eval_tasks = []
        for item in self.test:
            eval_tasks.append(self.rollout_and_score_eval(item["question"], item["gold_answer"]))
        results = await tqdm_asyncio.gather(*eval_tasks)

        # Extract scores and samples
        scores = [result["score"] for result in results]
        samples = [result["sample"] for result in results]

        percent_correct = sum(scores) / len(scores)

        end_time = time.time()

        # Add to existing metrics for wandb
        self.eval_metrics.append(("eval/percent_correct", percent_correct))

        # Log evaluation results
        eval_metrics = {
            "eval/percent_correct": percent_correct,
        }

        await self.evaluate_log(
            metrics=eval_metrics,
            samples=samples,
            start_time=start_time,
            end_time=end_time,
            generation_parameters={
                "temperature": 0.0,
                "max_tokens": self.config.max_token_length,
            },
        )

    async def collect_trajectories(self, item: GSM8kRow) -> Tuple[ScoredDataGroup, list[Item]]:
        """
        Generate rollouts for a single question.
        Returns scored data.
        """
        user_message = {"role": "user", "content": item["question"] + question_suffix}
        gold_answer = "\\boxed{" + item["answer"].split("#")[-1].strip().replace(",", "") + "}"

        # Generate group_size rollouts with logprobs
        messages = [*convo_prefix, user_message]

        async with self.server.managed_server(tokenizer=self.tokenizer) as managed:
            chat_completion = await managed.chat_completion(
                messages=messages,
                n=self.config.group_size,
                max_tokens=self.config.max_token_length,
                temperature=1.0,
                stop=[self.tokenizer.eos_token_id],
            )

            state = managed.get_state()
            nodes = state["nodes"]

        to_score = list()
        to_backlog = list()
        for choice, node in zip(chat_completion.choices, nodes):
            completion_messages = (
                *convo_prefix,
                user_message,
                {"role": "assistant", "content": choice.message.content},
            )
            to_score.append(
                {
                    "messages": completion_messages,
                    "gold_answer": gold_answer,
                    "finish_reason": choice.finish_reason,
                    "tokens": node.tokens,
                    "masked_tokens": node.masked_tokens,
                    "logprobs": node.logprobs,
                }
            )
        to_postprocess = await self.score(to_score)
        return to_postprocess, to_backlog

    async def score(
        self, rollout_group_data
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        """
        Score a group of rollouts.
        """
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()
        scores["inference_logprobs"] = list()

        # Parse gold answer
        gold_parsed = parse(
            rollout_group_data[0]["gold_answer"],
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )

        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            random.shuffle(rollout_group_data)
            for item in rollout_group_data:
                # Parse model answer
                answer_parsed = parse(
                    item["messages"][-1]["content"].split("</think>")[-1],
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                boxed="all",
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                # Binary reward: 1 if correct, 0 otherwise
                reward = verify(answer_parsed, gold_parsed)

                # Use pre-computed tokens and masks from ManagedServer
                tokens = item["tokens"]  # Full unmasked tokens
                masked_tokens = item["masked_tokens"]  # Pre-masked (-100 for prompt)
                logprobs = item["logprobs"]  # Pre-masked (1.0 for prompt)

                # Skip very short completions (less than 10 generated tokens)
                if len([1 for i in masked_tokens if i != -100]) < 10:
                    continue

                scores["tokens"].append(tokens)
                scores["masks"].append(masked_tokens)
                scores["inference_logprobs"].append(logprobs)
                scores["scores"].append(1.0 if reward else 0.0)

                if len(scores["tokens"]) >= self.config.group_size:
                    break

            for score in scores["scores"]:
                self.percent_correct_buffer.append(max(score, 0))
            return scores
        else:
            # If the gold solution is not parseable, we return None
            return None

    async def get_next_item(self) -> GSM8kRow:
        """
        Atropos specific method for getting the next item from the dataset
        """
        next_item = self.train[self.iter % len(self.train)]
        self.iter += 1
        return next_item


if __name__ == "__main__":
    GSM8kEnv.cli()
