"""
tau2-bench evaluation runner for Hermes Agent.

Runs the tau2-bench retail, airline, telecom, or banking_knowledge evaluation
using HermesAgent backed by litellm — the same inference path used across the
rest of the Hermes Agent codebase.

Usage:
    # Against OpenRouter (auto-detects OPENROUTER_API_KEY)
    python environments/benchmarks/taubench/run_eval.py \\
        --model openrouter/anthropic/claude-sonnet-4-5 \\
        --base-url openrouter \\
        --env retail

    # Against OpenAI directly
    python environments/benchmarks/taubench/run_eval.py \\
        --model gpt-4o \\
        --env retail

    # Local vLLM
    python environments/benchmarks/taubench/run_eval.py \\
        --model openai/NousResearch/Hermes-3-Llama-3.1-70B \\
        --base-url http://localhost:8000/v1 \\
        --env retail \\
        --num-trials 3

    # Specific tasks only
    python environments/benchmarks/taubench/run_eval.py \\
        --model openrouter/anthropic/claude-sonnet-4-5 \\
        --base-url openrouter \\
        --env retail \\
        --task-ids task_1 task_2 task_5

Results are saved to results/tau2bench/ as JSON.

Dependencies (requires Python 3.12+):
    pip install "tau2 @ git+https://github.com/sierra-research/tau2-bench.git"
    # or: pip install -e ".[tau2bench]"
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

_repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from tau2.data_model.simulation import Results, TextRunConfig
from tau2.evaluator.evaluator import EvaluationType
from tau2.registry import registry
from tau2.runner.batch import run_tasks
from tau2.runner.helpers import get_tasks

from environments.benchmarks.taubench.hermes_agent import create_hermes_agent

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
AGENT_NAME = "hermes_agent"


def _register_agent(
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    temperature: float,
    top_p: Optional[float],
    max_tokens: Optional[int],
    thinking: bool,
    tool_parser: Optional[str],
) -> None:
    """Register the HermesAgent factory with the tau2 registry (idempotent)."""
    if registry.get_agent_factory(AGENT_NAME) is not None:
        return

    def factory(tools, domain_policy, **kwargs):
        return create_hermes_agent(
            tools=tools,
            domain_policy=domain_policy,
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            thinking=thinking,
            tool_parser=tool_parser,
        )

    registry.register_agent_factory(factory=factory, name=AGENT_NAME)
    logger.info("Registered agent factory: %s (model=%s, thinking=%s, tool_parser=%s)", AGENT_NAME, model, thinking, tool_parser)


def run_eval(
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    user_model: str,
    env_name: str,
    task_split: Optional[str],
    num_trials: int,
    max_concurrency: int,
    max_steps: int,
    temperature: float,
    top_p: Optional[float],
    max_tokens: Optional[int],
    thinking: bool,
    tool_parser: Optional[str],
    task_ids: Optional[list],
    start_index: int,
    end_index: int,
    log_dir: str,
    seed: int,
) -> Results:
    # Resolve OpenRouter shorthand
    if base_url and base_url.strip().lower() == "openrouter":
        base_url = OPENROUTER_BASE_URL

    is_openrouter = base_url and "openrouter" in base_url.lower()

    # litellm requires the "openrouter/" prefix to route correctly
    if is_openrouter and not model.startswith("openrouter/"):
        model = f"openrouter/{model}"
    if is_openrouter and not user_model.startswith("openrouter/"):
        user_model = f"openrouter/{user_model}"

    # Resolve API key for agent
    if is_openrouter:
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if api_key and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = api_key
    else:
        api_key = api_key or os.environ.get("OPENAI_API_KEY")

    # Resolve user sim auth — user sim may use OpenRouter even when agent is local.
    # Check if user_model has openrouter/ prefix or OPENROUTER_API_KEY is set.
    user_model_is_openrouter = user_model.startswith("openrouter/")
    if not user_model_is_openrouter and "openrouter" in user_model.lower():
        user_model = f"openrouter/{user_model}"
        user_model_is_openrouter = True

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")

    _register_agent(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        thinking=thinking,
        tool_parser=tool_parser,
    )

    # Load tasks — task_ids in tau2 are strings like "task_1"
    tasks = get_tasks(
        task_set_name=env_name,
        task_split_name=task_split,
        task_ids=[str(i) for i in task_ids] if task_ids else None,
    )

    if not task_ids and (end_index != -1 or start_index != 0):
        end = end_index if end_index != -1 else len(tasks)
        tasks = tasks[start_index:end]

    logger.info(
        "Running tau2-%s eval: %d tasks, %d trial(s), concurrency=%d",
        env_name, len(tasks), num_trials, max_concurrency,
    )

    save_path = Path(log_dir) / f"tau2-{env_name}-{model.split('/')[-1]}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Pass auth to user sim. User sim may be on OpenRouter even if agent is local.
    user_llm_args: dict = {}
    if user_model_is_openrouter and openrouter_key:
        user_llm_args["api_key"] = openrouter_key
        user_llm_args["base_url"] = OPENROUTER_BASE_URL
    elif is_openrouter and api_key:
        user_llm_args["api_key"] = api_key
        user_llm_args["base_url"] = base_url

    config = TextRunConfig(
        domain=env_name,
        agent=AGENT_NAME,
        user="user_simulator",
        llm_agent=model,
        llm_args_agent={},
        llm_user=user_model,
        llm_args_user=user_llm_args,
        num_trials=num_trials,
        max_steps=max_steps,
        max_concurrency=max_concurrency,
        seed=seed,
    )

    results = run_tasks(
        config,
        tasks,
        save_path=save_path,
        console_display=True,
        # ALL: respects each task's reward_basis. NL assertions are skipped
        # gracefully (scored as pass) rather than raising an error, so tasks
        # are evaluated only on their actual basis components (DB, ACTION, etc.)
        evaluation_type=EvaluationType.ALL,
    )

    logger.info("Results saved to %s", save_path)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run tau2-bench evaluation with Hermes Agent (requires Python 3.12+)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True,
        help="litellm model string, e.g. 'openrouter/anthropic/claude-sonnet-4-5' or 'gpt-4o'",
    )
    parser.add_argument(
        "--base-url", default=None,
        help="API base URL. Use 'openrouter' as shorthand for https://openrouter.ai/api/v1.",
    )
    parser.add_argument("--api-key", default=None, help="API key (falls back to OPENROUTER_API_KEY / OPENAI_API_KEY)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature. NVIDIA used 1.0 for nemotron-super.")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Nucleus sampling. NVIDIA used 0.95 for nemotron-super.")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--thinking", action="store_true", default=False,
                        help="Enable reasoning/thinking mode (use_reasoning=true). "
                             "Required to match NVIDIA's reported nemotron-super scores.")
    parser.add_argument("--tool-parser", default=None,
                        help="Tool call parser to use (e.g. 'qwen3_coder'). When set, tools are "
                             "embedded in the system prompt as <tools> XML and responses are parsed "
                             "from raw text instead of using OpenAI function calling format.")
    parser.add_argument(
        "--user-model", default="qwen/qwen3-235b-a22b-2507:nitro",
        help="litellm model string for the tau2 user simulator. "
             "Defaults to qwen/qwen3-235b-a22b-2507:nitro (instruct, non-thinking) to match NVIDIA's eval setup. "
             "When using --base-url openrouter the openrouter/ prefix is added automatically.",
    )
    parser.add_argument(
        "--env", default="retail",
        choices=["retail", "airline", "telecom", "banking_knowledge", "mock"],
    )
    parser.add_argument(
        "--task-split", default=None,
        help="Task split name (e.g. 'base'). Defaults to the domain default.",
    )
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument(
        "--task-ids", nargs="*", default=None,
        help="Specific task IDs to run (tau2 task IDs are strings like 'task_1')",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--log-dir", default="results/tau2bench")

    args = parser.parse_args()

    run_eval(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        user_model=args.user_model,
        env_name=args.env,
        task_split=args.task_split,
        num_trials=args.num_trials,
        max_concurrency=args.max_concurrency,
        max_steps=args.max_steps,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        thinking=args.thinking,
        tool_parser=args.tool_parser,
        task_ids=args.task_ids,
        start_index=args.start_index,
        end_index=args.end_index,
        log_dir=args.log_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
