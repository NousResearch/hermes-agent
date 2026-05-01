"""
OpenThoughts-TBLite Evaluation Environment

A lighter, faster alternative to Terminal-Bench 2.0 for iterating on terminal
agents. Uses the same evaluation logic as TerminalBench2EvalEnv but defaults
to the NousResearch/openthoughts-tblite dataset (100 difficulty-calibrated
tasks vs TB2's 89 harder tasks).

TBLite tasks are a curated subset of TB2 with a difficulty distribution
designed to give meaningful signal even for smaller models:
  - Easy (40 tasks):   >= 70% pass rate with Claude Haiku 4.5
  - Medium (26 tasks): 40-69% pass rate
  - Hard (26 tasks):   10-39% pass rate
  - Extreme (8 tasks): < 10% pass rate

Usage:
    python environments/benchmarks/tblite/tblite_env.py evaluate

    # Filter to specific tasks:
    python environments/benchmarks/tblite/tblite_env.py evaluate \\
        --env.task_filter "broken-python,pandas-etl"
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

_repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from pydantic import Field

from atroposlib.envs.base import EvalHandlingEnum
from atroposlib.envs.server_handling.server_manager import APIServerConfig

from environments.benchmarks.terminalbench_2.terminalbench2_env import (
    TerminalBench2EvalConfig,
    TerminalBench2EvalEnv,
)


class TBLiteEvalConfig(TerminalBench2EvalConfig):
    """Configuration for the OpenThoughts-TBLite evaluation environment.

    Inherits all TB2 config fields. Only the dataset default and task timeout
    differ -- TBLite tasks are calibrated to be faster.
    """

    dataset_name: str = Field(
        default="NousResearch/openthoughts-tblite",
        description="HuggingFace dataset containing TBLite tasks.",
    )

    task_timeout: int = Field(
        default=1200,
        description="Maximum wall-clock seconds per task. TBLite tasks are "
        "generally faster than TB2, so 20 minutes is usually sufficient.",
    )


class TBLiteEvalEnv(TerminalBench2EvalEnv):
    """OpenThoughts-TBLite evaluation environment.

    Inherits all evaluation logic from TerminalBench2EvalEnv (agent loop,
    test verification, Docker image resolution, metrics, wandb logging).
    Only the default configuration differs.
    """

    name = "openthoughts-tblite"
    env_config_cls = TBLiteEvalConfig

    @classmethod
    def config_init(cls) -> Tuple[TBLiteEvalConfig, List[APIServerConfig]]:
        env_config = TBLiteEvalConfig(
            enabled_toolsets=["terminal", "file"],
            disabled_toolsets=None,
            distribution=None,

            max_agent_turns=60,
            max_token_length=16000,
            agent_temperature=0.6,
            system_prompt=None,

            terminal_backend="modal",
            terminal_timeout=300,

            test_timeout=180,

            # 100 tasks in parallel
            tool_pool_size=128,

            eval_handling=EvalHandlingEnum.STOP_TRAIN,
            group_size=1,
            steps_per_eval=1,
            total_steps=1,

            tokenizer_name="NousResearch/Hermes-3-Llama-3.1-8B",
            use_wandb=True,
            wandb_name="openthoughts-tblite",
            ensure_scores_are_not_same=False,
        )

        server_configs = [
            APIServerConfig(
                base_url="https://openrouter.ai/api/v1",
                model_name="anthropic/claude-sonnet-4",
                server_type="openai",
                api_key=os.getenv("OPENROUTER_API_KEY", ""),
                health_check=False,
            )
        ]

        return env_config, server_configs


if __name__ == "__main__":
    TBLiteEvalEnv.cli()
