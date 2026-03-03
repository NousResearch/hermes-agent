"""
YCBenchEvalEnv -- YC-Bench Long-Horizon Agent Benchmark Environment

Evaluates agentic LLMs on YC-Bench: a deterministic, long-horizon benchmark
where the agent acts as CEO of an AI startup over a simulated 1-3 year run.
The agent manages cash flow, employees, tasks, and prestige across 7 domains,
interacting exclusively via CLI subprocess calls against a SQLite-backed
discrete-event simulation.

Unlike TerminalBench2 (per-task binary pass/fail), YC-Bench measures sustained
multi-turn strategic coherence -- whether an agent can manage compounding
decisions over hundreds of turns without going bankrupt.

This is an eval-only environment. Run via:

    python environments/benchmarks/yc_bench/yc_bench_env.py evaluate \
        --config environments/benchmarks/yc_bench/default.yaml

The evaluate flow:
    1. setup()     -- Builds eval matrix (preset x seed combinations)
    2. evaluate()  -- Iterates over all runs through:
        a. rollout_and_score_eval()  -- Per-run agent loop
            - Initialises a fresh yc-bench simulation (preset + seed)
            - Runs HermesAgentLoop with terminal tool only
            - Reads final SQLite DB to extract score
            - Returns survival (0/1) + normalised funds score
        b. Aggregates per-preset and overall metrics
        c. Logs results via evaluate_log() and wandb

Key features:
  - CLI-only interface: agent calls yc-bench subcommands via terminal tool
  - Deterministic: same seed + preset = same world (SHA256-based RNG)
  - Multi-dimensional scoring: survival + normalised final funds
  - Per-preset difficulty breakdown in results
  - Isolated SQLite DB per run (no cross-run state leakage)
"""

import asyncio
import json
import logging
import math
import os
import sqlite3
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from pydantic import Field

from atroposlib.envs.base import EvalHandlingEnum
from atroposlib.envs.server_handling.server_manager import APIServerConfig

from environments.agent_loop import AgentResult, HermesAgentLoop
from environments.hermes_base_env import HermesAgentBaseEnv, HermesAgentEnvConfig

logger = logging.getLogger(__name__)

# =============================================================================
# System prompt
# =============================================================================

YC_BENCH_SYSTEM_PROMPT = """You are the CEO of an early-stage AI startup. You manage the company
exclusively through the `yc-bench` CLI tool. Your goal is to survive until the horizon end
without going bankrupt, while maximising final funds.

## Core loop
1. Call `yc-bench sim resume` to advance time to the next event.
2. Read the wake events and company status.
3. Decide: accept tasks from the market, assign employees, dispatch work, cancel if needed.
4. Repeat until bankruptcy or horizon end.

## Key commands
- `yc-bench company status`                          -- funds, prestige, runway, payroll date
- `yc-bench employee list`                           -- skills, salary, active task count
- `yc-bench market browse`                           -- available tasks
- `yc-bench task list`                               -- your accepted/active/completed tasks
- `yc-bench task inspect --task-id UUID`             -- progress %, deadline, assignments
- `yc-bench task accept --task-id UUID`              -- pull from market (sets deadline)
- `yc-bench task assign --task-id UUID --employee-id UUID`
- `yc-bench task dispatch --task-id UUID`            -- start work (needs >= 1 assignment)
- `yc-bench task cancel --task-id UUID --reason ""`  -- 2x prestige penalty, use sparingly
- `yc-bench sim resume`                              -- advance simulation clock
- `yc-bench scratchpad write/append/read`            -- persistent notes

## Survival rules
- Payroll fires on the first business day of each month. Running out of funds = bankruptcy.
- Missing a task deadline costs 1.4x prestige. Cancelling costs 2.0x.
- Assigning one employee to too many tasks splits their effective rate -- focus beats breadth.
- Higher prestige tasks pay much more. Specialise in 2-3 domains to unlock them.
- Use the scratchpad to track your strategy, deadlines, and employee assignments.

Always call `yc-bench company status` and `yc-bench task list` at the start of each turn
to orient yourself. Think step by step before acting."""

# Starting funds in cents ($250,000)
INITIAL_FUNDS_CENTS = 25_000_000


# =============================================================================
# Configuration
# =============================================================================

class YCBenchEvalConfig(HermesAgentEnvConfig):
    """
    Configuration for the YC-Bench evaluation environment.

    Extends HermesAgentEnvConfig with YC-Bench-specific settings for
    preset selection, seed control, and scoring.
    """

    presets: List[str] = Field(
        default=["fast_test", "medium", "hard"],
        description="YC-Bench preset names to evaluate.",
    )
    seeds: List[int] = Field(
        default=[1, 2, 3],
        description="Random seeds -- each preset x seed = one run.",
    )
    run_timeout: int = Field(
        default=3600,
        description="Maximum wall-clock seconds per run. Default 60 minutes.",
    )
    survival_weight: float = Field(
        default=0.5,
        description="Weight of survival (0/1) in composite score.",
    )
    funds_weight: float = Field(
        default=0.5,
        description="Weight of normalised final funds in composite score.",
    )
    db_dir: str = Field(
        default="/tmp/yc_bench_dbs",
        description="Directory for per-run SQLite databases.",
    )


# =============================================================================
# Scoring helpers
# =============================================================================

def _read_final_score(db_path: str) -> Dict[str, Any]:
    """
    Read final game state from a YC-Bench SQLite database.

    Returns dict with final_funds_cents (int), survived (bool),
    terminal_reason (str).
    """
    if not os.path.exists(db_path):
        logger.warning("DB not found at %s", db_path)
        return {"final_funds_cents": 0, "survived": False, "terminal_reason": "db_missing"}

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute("SELECT funds_cents FROM company LIMIT 1")
        row = cur.fetchone()
        funds = row[0] if row else 0

        terminal_reason = "horizon_end"
        try:
            cur.execute(
                "SELECT event_type FROM simulation_log ORDER BY id DESC LIMIT 1"
            )
            log_row = cur.fetchone()
            if log_row:
                terminal_reason = log_row[0]
        except sqlite3.OperationalError:
            pass

        conn.close()
        return {
            "final_funds_cents": funds,
            "survived": funds >= 0,
            "terminal_reason": terminal_reason,
        }

    except Exception as e:
        logger.error("Failed to read DB %s: %s", db_path, e)
        return {"final_funds_cents": 0, "survived": False, "terminal_reason": f"db_error: {e}"}


def _compute_composite_score(
    final_funds_cents: int,
    survived: bool,
    survival_weight: float = 0.5,
    funds_weight: float = 0.5,
    initial_funds_cents: int = INITIAL_FUNDS_CENTS,
) -> float:
    """
    Compute composite score from survival and final funds.

    Score = survival_weight * survival_score
          + funds_weight * normalised_funds_score

    Normalised funds uses log-scale relative to initial capital:
    - funds <= 0:          0.0
    - funds == initial:   ~0.5
    - funds == 100x:       1.0
    """
    survival_score = 1.0 if survived else 0.0

    if final_funds_cents <= 0:
        funds_score = 0.0
    else:
        max_ratio = 100.0
        ratio = final_funds_cents / max(initial_funds_cents, 1)
        funds_score = min(math.log1p(ratio) / math.log1p(max_ratio), 1.0)

    return survival_weight * survival_score + funds_weight * funds_score


# =============================================================================
# Main Environment
# =============================================================================

class YCBenchEvalEnv(HermesAgentBaseEnv):
    """
    YC-Bench long-horizon agent benchmark environment (eval-only).

    Each eval item is a (preset, seed) pair. The agent interacts with the
    simulation entirely via the terminal tool, calling yc-bench CLI commands.
    After the agent loop ends, the SQLite DB is read to extract the final score.

    Scoring:
      composite = 0.5 * survival + 0.5 * normalised_funds
    """

    name = "yc-bench"
    env_config_cls = YCBenchEvalConfig

    @classmethod
    def config_init(cls) -> Tuple[YCBenchEvalConfig, List[APIServerConfig]]:
        env_config = YCBenchEvalConfig(
            enabled_toolsets=["terminal"],
            disabled_toolsets=None,
            distribution=None,
            max_agent_turns=200,
            max_token_length=32000,
            agent_temperature=0.0,
            system_prompt=YC_BENCH_SYSTEM_PROMPT,
            terminal_backend="local",
            terminal_timeout=60,
            presets=["fast_test", "medium", "hard"],
            seeds=[1, 2, 3],
            run_timeout=3600,
            survival_weight=0.5,
            funds_weight=0.5,
            db_dir="/tmp/yc_bench_dbs",
            eval_handling=EvalHandlingEnum.STOP_TRAIN,
            group_size=1,
            steps_per_eval=1,
            total_steps=1,
            tokenizer_name="NousResearch/Hermes-3-Llama-3.1-8B",
            use_wandb=True,
            wandb_name="yc-bench",
            ensure_scores_are_not_same=False,
        )

        server_configs = [
            APIServerConfig(
                base_url="https://openrouter.ai/api/v1",
                model_name="anthropic/claude-sonnet-4-6",
                server_type="openai",
                api_key=os.getenv("OPENROUTER_API_KEY", ""),
                health_check=False,
            )
        ]

        return env_config, server_configs

    # =========================================================================
    # Setup
    # =========================================================================

    async def setup(self):
        """Build the eval matrix and verify yc-bench is installed."""
        import subprocess

        result = subprocess.run(["yc-bench", "--help"], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "yc-bench CLI not found. Install with:\n"
                "  pip install yc-bench\n"
                "Or: git clone https://github.com/collinear-ai/yc-bench "
                "&& cd yc-bench && pip install -e ."
            )
        print("yc-bench CLI verified.")

        self.all_eval_items = [
            {"preset": preset, "seed": seed}
            for preset in self.config.presets
            for seed in self.config.seeds
        ]
        self.iter = 0

        os.makedirs(self.config.db_dir, exist_ok=True)
        self.eval_metrics: List[Tuple[str, float]] = []

        import datetime
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(log_dir, exist_ok=True)
        run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._streaming_path = os.path.join(log_dir, f"samples_{run_ts}.jsonl")
        self._streaming_file = open(self._streaming_path, "w")
        self._streaming_lock = __import__("threading").Lock()

        print(f"\nYC-Bench eval matrix: {len(self.all_eval_items)} runs")
        for item in self.all_eval_items:
            print(f"  preset={item['preset']!r}  seed={item['seed']}")
        print(f"Streaming results to: {self._streaming_path}\n")

    def _save_result(self, result: Dict[str, Any]):
        if not hasattr(self, "_streaming_file") or self._streaming_file.closed:
            return
        with self._streaming_lock:
            self._streaming_file.write(
                json.dumps(result, ensure_ascii=False, default=str) + "\n"
            )
            self._streaming_file.flush()

    # =========================================================================
    # Training pipeline stubs
    # =========================================================================

    async def get_next_item(self):
        item = self.all_eval_items[self.iter % len(self.all_eval_items)]
        self.iter += 1
        return item

    def format_prompt(self, item: Dict[str, Any]) -> str:
        preset = item["preset"]
        seed = item["seed"]
        return (
            f"Start a new YC-Bench simulation with preset='{preset}' and seed={seed}.\n\n"
            f"Run: `yc-bench run --config {preset} --seed {seed} --no-live`\n\n"
            "Wait for the simulation to initialise, then begin managing the company. "
            "Call `yc-bench company status` to see your starting position, then enter "
            "the core loop: sim resume -> observe -> decide -> act -> repeat."
        )

    async def compute_reward(self, item, result, ctx) -> float:
        return 0.0

    async def collect_trajectories(self, item):
        return None, []

    async def score(self, rollout_group_data):
        return None

    # =========================================================================
    # Per-run evaluation
    # =========================================================================

    async def rollout_and_score_eval(self, eval_item: Dict[str, Any]) -> Dict:
        """
        Evaluate a single (preset, seed) run.

        1. Sets DATABASE_URL env var to an isolated DB path for this run
        2. Runs HermesAgentLoop with terminal tool
        3. Reads SQLite DB to compute final score
        4. Returns result dict with survival, funds, and composite score
        """
        preset = eval_item["preset"]
        seed = eval_item["seed"]
        run_id = str(uuid.uuid4())[:8]
        run_key = f"{preset}_seed{seed}_{run_id}"

        from tqdm import tqdm
        tqdm.write(f"  [START] preset={preset!r} seed={seed} (run_id={run_id})")
        run_start = time.time()

        db_path = os.path.join(self.config.db_dir, f"yc_bench_{run_key}.db")
        os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
        os.environ["YC_BENCH_EXPERIMENT"] = preset

        try:
            tools, valid_names = self._resolve_tools_for_group()

            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": YC_BENCH_SYSTEM_PROMPT},
                {"role": "user", "content": self.format_prompt(eval_item)},
            ]

            agent = HermesAgentLoop(
                server=self.server,
                tool_schemas=tools,
                valid_tool_names=valid_names,
                max_turns=self.config.max_agent_turns,
                task_id=run_id,
                temperature=self.config.agent_temperature,
                max_tokens=self.config.max_token_length,
                extra_body=self.config.extra_body,
            )
            result = await agent.run(messages)

            score_data = _read_final_score(db_path)
            final_funds = score_data["final_funds_cents"]
            survived = score_data["survived"]
            terminal_reason = score_data["terminal_reason"]

            composite = _compute_composite_score(
                final_funds_cents=final_funds,
                survived=survived,
                survival_weight=self.config.survival_weight,
                funds_weight=self.config.funds_weight,
            )

            elapsed = time.time() - run_start
            status = "SURVIVED" if survived else "BANKRUPT"
            funds_str = f"${final_funds / 100:,.0f}" if final_funds >= 0 else f"-${abs(final_funds) / 100:,.0f}"
            tqdm.write(
                f"  [{status}] preset={preset!r} seed={seed} "
                f"funds={funds_str} score={composite:.3f} "
                f"turns={result.turns_used} ({elapsed:.0f}s)"
            )

            out = {
                "preset": preset,
                "seed": seed,
                "survived": survived,
                "final_funds_cents": final_funds,
                "final_funds_usd": final_funds / 100,
                "terminal_reason": terminal_reason,
                "composite_score": composite,
                "turns_used": result.turns_used,
                "finished_naturally": result.finished_naturally,
                "elapsed_seconds": elapsed,
                "db_path": db_path,
                "messages": result.messages,
            }
            self._save_result(out)
            return out

        except Exception as e:
            elapsed = time.time() - run_start
            logger.error("Run %s failed: %s", run_key, e, exc_info=True)
            tqdm.write(f"  [ERROR] preset={preset!r} seed={seed}: {e} ({elapsed:.0f}s)")
            out = {
                "preset": preset, "seed": seed,
                "survived": False, "final_funds_cents": 0,
                "final_funds_usd": 0.0,
                "terminal_reason": f"error: {e}",
                "composite_score": 0.0, "turns_used": 0,
                "error": str(e), "elapsed_seconds": elapsed,
            }
            self._save_result(out)
            return out

    # =========================================================================
    # Evaluate
    # =========================================================================

    async def _run_with_timeout(self, item: Dict[str, Any]) -> Dict:
        preset = item["preset"]
        seed = item["seed"]
        try:
            return await asyncio.wait_for(
                self.rollout_and_score_eval(item),
                timeout=self.config.run_timeout,
            )
        except asyncio.TimeoutError:
            from tqdm import tqdm
            tqdm.write(f"  [TIMEOUT] preset={preset!r} seed={seed} (exceeded {self.config.run_timeout}s)")
            out = {
                "preset": preset, "seed": seed,
                "survived": False, "final_funds_cents": 0,
                "final_funds_usd": 0.0,
                "terminal_reason": f"timeout ({self.config.run_timeout}s)",
                "composite_score": 0.0, "turns_used": 0, "error": "timeout",
            }
            self._save_result(out)
            return out

    async def evaluate(self, *args, **kwargs) -> None:
        """
        Run YC-Bench evaluation over all (preset, seed) combinations.

        Runs sequentially -- each run is 100-500 turns, parallelising would
        be prohibitively expensive and cause DB path conflicts.
        """
        start_time = time.time()
        from tqdm import tqdm

        print(f"\n{'='*60}")
        print("Starting YC-Bench Evaluation")
        print(f"{'='*60}")
        print(f"  Presets: {self.config.presets}")
        print(f"  Seeds: {self.config.seeds}")
        print(f"  Total runs: {len(self.all_eval_items)}")
        print(f"  Max turns/run: {self.config.max_agent_turns}")
        print(f"  Run timeout: {self.config.run_timeout}s")
        print(f"{'='*60}\n")

        results = []
        pbar = tqdm(total=len(self.all_eval_items), desc="YC-Bench", dynamic_ncols=True)

        for item in self.all_eval_items:
            result = await self._run_with_timeout(item)
            results.append(result)
            survived_count = sum(1 for r in results if r.get("survived"))
            pbar.set_postfix_str(f"survived={survived_count}/{len(results)}")
            pbar.update(1)

        pbar.close()
        end_time = time.time()

        valid = [r for r in results if r is not None]
        if not valid:
            print("Warning: No valid results.")
            return

        total = len(valid)
        survived_total = sum(1 for r in valid if r.get("survived"))
        survival_rate = survived_total / total if total else 0.0
        avg_score = sum(r.get("composite_score", 0) for r in valid) / total if total else 0.0

        preset_results: Dict[str, List[Dict]] = defaultdict(list)
        for r in valid:
            preset_results[r["preset"]].append(r)

        eval_metrics = {
            "eval/survival_rate": survival_rate,
            "eval/avg_composite_score": avg_score,
            "eval/total_runs": total,
            "eval/survived_runs": survived_total,
            "eval/evaluation_time_seconds": end_time - start_time,
        }

        for preset, items in sorted(preset_results.items()):
            ps = sum(1 for r in items if r.get("survived"))
            pt = len(items)
            pa = sum(r.get("composite_score", 0) for r in items) / pt if pt else 0
            key = preset.replace("-", "_")
            eval_metrics[f"eval/survival_rate_{key}"] = ps / pt if pt else 0
            eval_metrics[f"eval/avg_score_{key}"] = pa

        self.eval_metrics = [(k, v) for k, v in eval_metrics.items()]

        print(f"\n{'='*60}")
        print("YC-Bench Evaluation Results")
        print(f"{'='*60}")
        print(f"Overall survival rate: {survival_rate:.1%} ({survived_total}/{total})")
        print(f"Average composite score: {avg_score:.4f}")
        print(f"Evaluation time: {end_time - start_time:.1f}s")

        print("\nPer-preset breakdown:")
        for preset, items in sorted(preset_results.items()):
            ps = sum(1 for r in items if r.get("survived"))
            pt = len(items)
            pa = sum(r.get("composite_score", 0) for r in items) / pt if pt else 0
            print(f"  {preset}: {ps}/{pt} survived  avg_score={pa:.4f}")
            for r in items:
                status = "SURVIVED" if r.get("survived") else "BANKRUPT"
                funds = r.get("final_funds_usd", 0)
                print(f"    seed={r['seed']}  [{status}]  ${funds:,.0f}  score={r.get('composite_score', 0):.3f}")

        print(f"{'='*60}\n")

        samples = [{k: v for k, v in r.items() if k != "messages"} for r in valid]

        try:
            await self.evaluate_log(
                metrics=eval_metrics,
                samples=samples,
                start_time=start_time,
                end_time=end_time,
                generation_parameters={
                    "temperature": self.config.agent_temperature,
                    "max_tokens": self.config.max_token_length,
                    "max_agent_turns": self.config.max_agent_turns,
                },
            )
        except Exception as e:
            print(f"Error logging results: {e}")

        if hasattr(self, "_streaming_file") and not self._streaming_file.closed:
            self._streaming_file.close()
            print(f"Results saved to: {self._streaming_path}")

    # =========================================================================
    # Wandb logging
    # =========================================================================

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}
        for k, v in self.eval_metrics:
            wandb_metrics[k] = v
        self.eval_metrics = []
        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    YCBenchEvalEnv.cli()
