"""
Tau-bench evaluation runner for Hermes Agent.

Runs the tau-bench retail or airline evaluation using HermesAgent backed by
atroposlib's OpenAIServer — the same server abstraction used by HermesAgentLoop
in our training environments.

Usage:
    # Against OpenRouter (auto-detects OPENROUTER_API_KEY, sets required headers)
    python environments/benchmarks/taubench/run_eval.py \\
        --model anthropic/claude-sonnet-4-5 \\
        --base-url openrouter \\
        --env retail

    # Against OpenAI
    python environments/benchmarks/taubench/run_eval.py \\
        --model gpt-4o \\
        --base-url https://api.openai.com/v1 \\
        --env retail

    # Against a local vLLM server
    python environments/benchmarks/taubench/run_eval.py \\
        --model NousResearch/Hermes-3-Llama-3.1-70B \\
        --base-url http://localhost:8000/v1 \\
        --env retail \\
        --num-trials 3

    # Specific tasks only
    python environments/benchmarks/taubench/run_eval.py \\
        --model anthropic/claude-sonnet-4-5 \\
        --base-url openrouter \\
        --env retail \\
        --task-ids 0 1 2 5 10

Results are saved to results/taubench/ as JSON.

Dependencies:
    pip install -e ".[taubench]"
    (tau-bench: pip install git+https://github.com/sierra-research/tau-bench)
"""

import argparse
import json
import logging
import os
import random
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from math import comb
from pathlib import Path
from typing import Dict, List, Optional

# Ensure hermes-agent repo root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from atroposlib.envs.server_handling.openai_server import OpenAIServer
from atroposlib.envs.server_handling.server_manager import APIServerConfig

from tau_bench.envs import get_env
from tau_bench.types import EnvRunResult

from environments.benchmarks.taubench.hermes_agent import HermesAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def build_server(model: str, base_url: str, api_key: Optional[str]) -> OpenAIServer:
    """
    Build an atroposlib OpenAIServer for the given endpoint.

    For OpenRouter, injects HTTP-Referer and X-Title headers into the
    underlying openai client after construction (APIServerConfig has no
    default_headers field).
    """
    is_openrouter = "openrouter" in base_url.lower()

    if is_openrouter:
        resolved_key = (
            api_key
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY", "EMPTY")
        )
    else:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")

    config = APIServerConfig(
        model_name=model,
        base_url=base_url,
        api_key=resolved_key,
    )
    server = OpenAIServer(config=config)

    if is_openrouter:
        server.openai = server.openai.with_options(
            default_headers={
                "HTTP-Referer": "https://hermes-agent.nousresearch.com",
                "X-Title": "Hermes Agent",
            }
        )

    return server


def display_metrics(results: List[EnvRunResult]) -> None:
    if not results:
        print("No results to display.")
        return

    avg_reward = sum(r.reward for r in results) / len(results)
    print(f"\n{'='*50}")
    print(f"Results ({len(results)} tasks):")
    print(f"  Average reward: {avg_reward:.4f}")

    task_results: Dict[int, List[float]] = {}
    for r in results:
        task_results.setdefault(r.task_id, []).append(r.reward)

    num_tasks = len(task_results)
    total_trials = len(results) // num_tasks if num_tasks > 0 else 1

    for k in range(1, min(total_trials + 1, 5)):
        successes = sum(
            1 for rewards in task_results.values() if any(r >= 1.0 for r in rewards[:k])
        )
        p_k = successes / num_tasks if num_tasks > 0 else 0.0
        print(f"  Pass@{k}: {p_k:.4f}  ({successes}/{num_tasks} tasks)")

    print(f"{'='*50}\n")


def run_eval(
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    env_name: str,
    task_split: str,
    user_model: str,
    user_base_url: Optional[str],
    user_api_key: Optional[str],
    num_trials: int,
    max_concurrency: int,
    max_num_steps: int,
    temperature: float,
    max_tokens: Optional[int],
    task_ids: Optional[List[int]],
    start_index: int,
    end_index: int,
    log_dir: str,
    seed: int,
    shuffle: bool,
    system_prompt: Optional[str],
) -> List[EnvRunResult]:
    random.seed(seed)

    # Expand "openrouter" shorthand to full URL
    if base_url and base_url.strip().lower() == "openrouter":
        base_url = OPENROUTER_BASE_URL
    if not base_url:
        base_url = "https://api.openai.com/v1"

    os.makedirs(log_dir, exist_ok=True)
    time_str = datetime.now().strftime("%m%d%H%M%S")
    ckpt_path = os.path.join(
        log_dir,
        f"hermes-{model.split('/')[-1]}-{temperature}_{env_name}_{task_split}_{time_str}.json",
    )

    # Resolve litellm provider + model for the tau-bench user simulator.
    # tau-bench's user sim calls litellm.completion(model=..., custom_llm_provider=...).
    user_effective_base = user_base_url or base_url
    user_is_openrouter = "openrouter" in user_effective_base.lower()

    if user_is_openrouter:
        user_provider = "openrouter"
        if user_model == "gpt-4o":
            user_model = model  # mirror the agent model
        if "/" not in user_model:
            user_model = f"openai/{user_model}"
        if not os.environ.get("OPENROUTER_API_KEY") and (api_key or user_api_key):
            os.environ["OPENROUTER_API_KEY"] = user_api_key or api_key
    else:
        user_provider = "openai"

    ref_env = get_env(
        env_name,
        user_strategy="llm",
        user_model=user_model,
        task_split=task_split,
        user_provider=user_provider,
    )
    total_tasks = len(ref_env.tasks)

    if task_ids and len(task_ids) > 0:
        idxs = task_ids
    else:
        end = total_tasks if end_index == -1 else min(end_index, total_tasks)
        idxs = list(range(start_index, end))

    logger.info(
        "Running %s eval on %s (%d tasks, %d trials, concurrency=%d)",
        env_name, model, len(idxs), num_trials, max_concurrency,
    )
    logger.info("Results checkpoint: %s", ckpt_path)

    results: List[EnvRunResult] = []

    def _run_task(idx: int, trial: int) -> EnvRunResult:
        isolated_env = get_env(
            env_name,
            user_strategy="llm",
            user_model=user_model,
            task_split=task_split,
            user_provider=user_provider,
            task_index=idx,
        )
        # Build a fresh server per task — each worker runs its own asyncio event
        # loop via asyncio.run(), and AsyncOpenAI clients are bound to the loop
        # they were created in. Sharing one client across loops causes hangs.
        task_server = build_server(model=model, base_url=base_url, api_key=api_key)
        agent = HermesAgent(
            server=task_server,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        logger.info("Trial %d | Task %d", trial + 1, idx)
        try:
            solve_result = agent.solve(
                env=isolated_env,
                task_index=idx,
                max_num_steps=max_num_steps,
            )
            return EnvRunResult(
                task_id=idx,
                reward=solve_result.reward,
                info=solve_result.info,
                traj=solve_result.messages,
                trial=trial,
            )
        except Exception as e:
            logger.error("Task %d trial %d failed: %s", idx, trial, traceback.format_exc())
            return EnvRunResult(
                task_id=idx,
                reward=0.0,
                info={"error": str(e)},
                traj=[],
                trial=trial,
            )

    for trial in range(num_trials):
        trial_idxs = list(idxs)
        if shuffle:
            random.shuffle(trial_idxs)

        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = [executor.submit(_run_task, idx, trial) for idx in trial_idxs]
            for future in futures:
                result = future.result()
                results.append(result)
                with open(ckpt_path, "w") as f:
                    json.dump([r.model_dump() for r in results], f, indent=2)

    display_metrics(results)
    logger.info("Final results saved to %s", ckpt_path)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run tau-bench evaluation with Hermes Agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Model name (e.g., anthropic/claude-sonnet-4-5)")
    parser.add_argument(
        "--base-url", default=None,
        help="OpenAI-compatible API base URL. Use 'openrouter' as shorthand for "
             "https://openrouter.ai/api/v1 (auto-detects OPENROUTER_API_KEY).",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API key. OpenRouter: falls back to OPENROUTER_API_KEY. Others: OPENAI_API_KEY.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument(
        "--user-model", default="gpt-4o",
        help="Model for the tau-bench user simulator. Defaults to the agent model when on OpenRouter.",
    )
    parser.add_argument("--user-base-url", default=None)
    parser.add_argument("--user-api-key", default=None)
    parser.add_argument("--env", default="retail", choices=["retail", "airline"])
    parser.add_argument("--task-split", default="test", choices=["train", "test", "dev"])
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--max-num-steps", type=int, default=30)
    parser.add_argument("--task-ids", type=int, nargs="*", default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--log-dir", default="results/taubench")
    parser.add_argument("--system-prompt", default=None)

    args = parser.parse_args()

    run_eval(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        env_name=args.env,
        task_split=args.task_split,
        user_model=args.user_model,
        user_base_url=args.user_base_url,
        user_api_key=args.user_api_key,
        num_trials=args.num_trials,
        max_concurrency=args.max_concurrency,
        max_num_steps=args.max_num_steps,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        task_ids=args.task_ids,
        start_index=args.start_index,
        end_index=args.end_index,
        log_dir=args.log_dir,
        seed=args.seed,
        shuffle=args.shuffle,
        system_prompt=args.system_prompt,
    )


if __name__ == "__main__":
    main()
