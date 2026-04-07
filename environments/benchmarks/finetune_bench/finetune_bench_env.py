"""
FinetuneBenchEnv -- Fine-tune evaluation benchmark.

Runs a structured prompt bank against the agent loop, scoring tool selection,
execution quality, and end-to-end task completion. This is the evaluation gate
referenced in the hermes-finetune design spec.

Usage:
    python finetune_bench_env.py evaluate \\
        --config environments/benchmarks/finetune_bench/default.yaml

    python finetune_bench_env.py process \\
        --env.data_path_to_save_groups results.jsonl
"""

import json
import logging
import os
import re
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure repo root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import yaml
from pydantic import Field

from atroposlib.envs.base import EvalHandlingEnum
from atroposlib.envs.server_handling.server_manager import APIServerConfig

from environments.agent_loop import AgentResult, HermesAgentLoop
from environments.hermes_base_env import HermesAgentBaseEnv, HermesAgentEnvConfig
from environments.tool_context import ToolContext

logger = logging.getLogger(__name__)


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class CaseResult:
    case_id: str
    tier: int
    category: str
    tags: list
    tool_selection_correct: bool = False
    tool_args_valid: bool = False
    task_completed: bool = False
    format_valid: bool = True
    tool_call_parseable: bool = True
    turns_used: int = 0
    tool_errors: int = 0
    reward: float = 0.0
    is_canary: bool = False


# =============================================================================
# Configuration
# =============================================================================

class FinetuneBenchConfig(HermesAgentEnvConfig):
    """Configuration for the finetune benchmark environment."""

    prompt_bank_path: str = Field(
        default="environments/benchmarks/finetune_bench/prompt_bank.yaml",
        description="Path to the prompt bank YAML file.",
    )
    custom_cases_dir: str = Field(
        default="~/.hermes/finetune/bench/custom",
        description="Directory for user-provided custom test cases.",
    )
    baseline_results_path: Optional[str] = Field(
        default=None,
        description="Path to previous eval results for comparison.",
    )
    regression_threshold_tool_selection: float = Field(
        default=0.03,
        description="Maximum allowed regression in tool selection accuracy.",
    )
    regression_threshold_execution: float = Field(
        default=0.05,
        description="Maximum allowed regression in tool execution success.",
    )
    regression_threshold_completion: float = Field(
        default=0.05,
        description="Maximum allowed regression in task completion rate.",
    )
    format_compliance_minimum: float = Field(
        default=0.95,
        description="Minimum required format compliance rate.",
    )


# =============================================================================
# Environment
# =============================================================================

class FinetuneBenchEnv(HermesAgentBaseEnv):
    """
    Fine-tune evaluation benchmark environment.

    Runs test cases from a prompt bank and scores tool selection,
    execution quality, and end-to-end task completion.
    """

    name = "finetune-bench"
    env_config_cls = FinetuneBenchConfig

    @classmethod
    def config_init(cls):
        env_config = FinetuneBenchConfig(
            enabled_toolsets=["terminal", "file", "web"],
            # Docker backend is REQUIRED — see the comment in default.yaml.
            terminal_backend="docker",
            max_agent_turns=15,
            eval_handling="STOP_TRAIN",
            steps_per_eval=1,
            total_steps=1,
        )
        server_configs = [
            APIServerConfig(
                model_name="carnice",
                base_url="http://localhost:8008/v1",
                api_key="none",
                num_requests_for_eval=1,
            )
        ]
        return env_config, server_configs

    async def setup(self):
        """Load prompt bank, custom cases, and configure docker sandbox mount."""
        # Suppress the per-container disk-quota warning that fires for every
        # case on systems without overlay2-on-XFS-with-pquota. It's harmless
        # informational noise and would print 243 times in a full run.
        class _DropDiskQuotaWarning(logging.Filter):
            def filter(self, record):
                return "per-container disk limits" not in record.getMessage()

        logging.getLogger("tools.environments.docker").addFilter(_DropDiskQuotaWarning())

        # Bind-mount the host scratch dir into every container the bench
        # spawns so per-case working dirs (created by _rollout_case) are
        # visible inside the sandbox at the same path. This is set as an
        # env var because docker_volumes isn't supported in the per-task
        # override dict, only in global config.
        host_scratch = Path("/tmp/finetune-bench")
        host_scratch.mkdir(parents=True, exist_ok=True)
        existing = os.environ.get("TERMINAL_DOCKER_VOLUMES", "")
        mount_spec = f"{host_scratch}:{host_scratch}"
        if mount_spec not in existing:
            try:
                current = json.loads(existing) if existing else []
            except json.JSONDecodeError:
                current = []
            if mount_spec not in current:
                current.append(mount_spec)
            os.environ["TERMINAL_DOCKER_VOLUMES"] = json.dumps(current)
            logger.info("Mounted %s into bench containers", mount_spec)

        bank_path = Path(self.config.prompt_bank_path)
        if not bank_path.is_absolute():
            bank_path = _repo_root / bank_path

        if bank_path.exists():
            with open(bank_path) as f:
                data = yaml.safe_load(f)
                self.prompt_bank = data.get("cases", [])
        else:
            logger.warning("Prompt bank not found: %s", bank_path)
            self.prompt_bank = []

        # Merge custom cases
        custom_dir = Path(self.config.custom_cases_dir).expanduser()
        if custom_dir.exists():
            for custom_file in sorted(custom_dir.glob("*.yaml")):
                with open(custom_file) as f:
                    custom = yaml.safe_load(f)
                    if custom and "cases" in custom:
                        self.prompt_bank.extend(custom["cases"])

        # Load baseline for comparison
        self.baseline = None
        if self.config.baseline_results_path:
            bp = Path(self.config.baseline_results_path)
            if bp.exists():
                with open(bp) as f:
                    self.baseline = json.load(f)

        self.results: List[CaseResult] = []
        self.iter = 0

        logger.info("Loaded %d test cases", len(self.prompt_bank))

    async def get_next_item(self):
        if self.iter >= len(self.prompt_bank):
            return None
        item = self.prompt_bank[self.iter]
        self.iter += 1
        return item

    def format_prompt(self, item):
        return item["prompt"]

    async def compute_reward(self, item, result, ctx):
        """Score a single test case."""
        case = CaseResult(
            case_id=item["id"],
            tier=item["tier"],
            category=item["category"],
            tags=item.get("tags", []),
            is_canary=item.get("canary", False),
            turns_used=result.turns_used,
            tool_errors=len(result.tool_errors) if result.tool_errors else 0,
        )

        messages = result.messages

        # Format compliance
        case.format_valid = self._check_format(messages)
        case.tool_call_parseable = self._check_tool_parse(messages)

        # Tier 1: Tool selection
        if item["tier"] >= 1:
            expected = item.get("expected", {})
            actual_tools = self._extract_tool_calls(messages)

            if expected.get("should_call_tool") is False:
                case.tool_selection_correct = len(actual_tools) == 0
            elif expected.get("tool_name"):
                case.tool_selection_correct = any(
                    t["name"] == expected["tool_name"] for t in actual_tools
                )

        # Tier 2: Tool execution
        if item["tier"] >= 2 and item.get("verification"):
            v = item["verification"]
            if v["method"] == "output_match":
                case.tool_args_valid = self._verify_output(messages, v)
            elif v["method"] == "functional_test":
                case.task_completed = await self._verify_functional(ctx, item, v)
                case.tool_args_valid = case.task_completed

        # Tier 3: End-to-end
        if item["tier"] == 3 and item.get("verification"):
            v = item["verification"]
            case.task_completed = await self._verify_functional(ctx, item, v)

        # Composite reward
        case.reward = self._compute_composite(case, item["tier"])
        self.results.append(case)
        return case.reward

    def _compute_composite(self, case: CaseResult, tier: int) -> float:
        if not case.format_valid or not case.tool_call_parseable:
            return 0.0

        if tier == 1:
            return 1.0 if case.tool_selection_correct else 0.0
        elif tier == 2:
            selection = 0.4 if case.tool_selection_correct else 0.0
            execution = 0.6 if case.tool_args_valid else 0.0
            return selection + execution
        else:
            selection = 0.2 if case.tool_selection_correct else 0.0
            execution = 0.3 if case.tool_args_valid else 0.0
            completion = 0.5 if case.task_completed else 0.0
            return selection + execution + completion

    def _check_format(self, messages: List[Dict]) -> bool:
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if content is None and not msg.get("tool_calls"):
                    return False
        return True

    def _check_tool_parse(self, messages: List[Dict]) -> bool:
        for msg in messages:
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    try:
                        args = tc.get("function", {}).get("arguments")
                        if isinstance(args, str):
                            json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        return False
        return True

    def _extract_tool_calls(self, messages: List[Dict]) -> List[Dict]:
        tools = []
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    tools.append({
                        "name": tc.get("function", {}).get("name"),
                        "arguments": tc.get("function", {}).get("arguments"),
                    })
        return tools

    def _verify_output(self, messages: List[Dict], verification: Dict) -> bool:
        expected = verification.get("expected_value", "")
        for msg in messages:
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if expected in content:
                    return True
        return False

    async def _verify_functional(self, ctx: ToolContext, item: Dict, verification: Dict) -> bool:
        checks = verification.get("checks", [])
        commands = verification.get("test_commands", [])

        outputs = []
        for cmd in commands:
            try:
                result = ctx.terminal(cmd, timeout=30)
                outputs.append(result)
            except Exception as e:
                outputs.append({"output": str(e), "exit_code": -1})

        passed = 0
        for check in checks:
            idx = check.get("command_index", 0)
            if idx >= len(outputs):
                continue

            output = outputs[idx]
            check_type = check["type"]

            if check_type == "exit_code":
                if output.get("exit_code") == check["expected"]:
                    passed += 1
            elif check_type == "output_contains":
                content = output.get("output", "")
                if all(s in content for s in check["expected"]):
                    passed += 1
            elif check_type == "output_regex":
                content = output.get("output", "")
                if re.search(check["pattern"], content):
                    passed += 1
            elif check_type == "file_exists":
                try:
                    result = ctx.terminal(f"test -f {check['path']} && echo EXISTS", timeout=10)
                    if "EXISTS" in result.get("output", ""):
                        passed += 1
                except Exception:
                    pass

        return passed == len(checks) if checks else False

    # =========================================================================
    # Per-case rollout (called by evaluate)
    # =========================================================================

    async def _rollout_case(self, item: Dict) -> CaseResult:
        """Run one test case end-to-end: setup → agent loop → score."""
        import asyncio
        import shutil
        import uuid as _uuid

        from tools.terminal_tool import (
            register_task_env_overrides,
            clear_task_env_overrides,
        )

        task_id = str(_uuid.uuid4())

        # --- Setup phase: every case gets an isolated working directory ---
        # If the case specifies setup.working_dir, use that. Otherwise mint a
        # per-case temp dir so Tier 1/2 cases can't scribble in the cwd.
        setup_cfg = item.get("setup") or {}
        working_dir = setup_cfg.get("working_dir")
        if not working_dir:
            working_dir = f"/tmp/finetune-bench/{item.get('id', task_id[:8])}"

        wd = Path(working_dir).expanduser()
        if wd.exists():
            shutil.rmtree(wd, ignore_errors=True)
        wd.mkdir(parents=True, exist_ok=True)

        # Seed files specified in setup.files
        for fname, content in (setup_cfg.get("files") or {}).items():
            fpath = wd / fname
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(content)

        # Pin the terminal sandbox cwd for this rollout. The local terminal
        # backend honors this override so all commands the agent runs land
        # inside the per-case dir, not the repo root.
        register_task_env_overrides(task_id, {"cwd": str(wd)})

        # --- Resolve tools (uses HermesAgentBaseEnv helper) ---
        tools, valid_names = self._resolve_tools_for_group()

        # --- Build messages ---
        # Tell the model its working directory so it doesn't try to use
        # absolute paths or assume it's somewhere else.
        prompt_text = (
            f"[Working directory: {wd}]\n\n{self.format_prompt(item)}"
        )
        messages: List[Dict[str, Any]] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": prompt_text})

        # --- Run agent loop (always direct server for OpenAI-compatible llama.cpp) ---
        agent = HermesAgentLoop(
            server=self.server,
            tool_schemas=tools,
            valid_tool_names=valid_names,
            max_turns=item.get("max_turns") or self.config.max_agent_turns,
            task_id=task_id,
            temperature=self.config.agent_temperature,
            max_tokens=self.config.max_token_length,
            extra_body=self.config.extra_body,
        )

        try:
            try:
                result = await agent.run(messages)
            except Exception as e:
                logger.error("Case %s rollout failed: %s", item.get("id"), e)
                case = CaseResult(
                    case_id=item.get("id", "?"),
                    tier=item.get("tier", 1),
                    category=item.get("category", "unknown"),
                    tags=item.get("tags", []),
                    is_canary=item.get("canary", False),
                    format_valid=False,
                    tool_call_parseable=False,
                    reward=0.0,
                )
                self.results.append(case)
                return case

            # --- Score via compute_reward ---
            ctx = ToolContext(task_id)
            try:
                await self.compute_reward(item, result, ctx)
            except Exception as e:
                logger.error("Case %s scoring failed: %s", item.get("id"), e)
            finally:
                try:
                    ctx.cleanup()
                except Exception:
                    pass

            return self.results[-1] if self.results else None
        finally:
            # Always release the per-task cwd override so it doesn't leak
            # into the next case (or any concurrent local-terminal users).
            try:
                clear_task_env_overrides(task_id)
            except Exception:
                pass

    # =========================================================================
    # Evaluation & reporting
    # =========================================================================

    async def evaluate(self, *args, **kwargs):
        """Iterate the prompt bank, score each case, then aggregate."""
        import asyncio

        # Atropos may not have called setup() — defensive load.
        if not hasattr(self, "prompt_bank") or not self.prompt_bank:
            await self.setup()

        if not self.prompt_bank:
            print("[finetune-bench] No test cases loaded — check prompt_bank_path")
            return

        from tqdm import tqdm

        total = len(self.prompt_bank)
        print(f"\n{'='*60}")
        print(f"  FINETUNE BENCH — running {total} test cases")
        print(f"{'='*60}\n")

        pbar = tqdm(total=total, desc="Evaluating", dynamic_ncols=True)
        for item in self.prompt_bank:
            try:
                await self._rollout_case(item)
            except Exception as e:
                logger.error("Case %s failed in evaluate loop: %s", item.get("id"), e)
            passed = sum(1 for r in self.results if r.reward >= 0.5)
            done = len(self.results)
            pct = (passed / done * 100) if done else 0.0
            pbar.set_postfix_str(f"pass={passed}/{done} ({pct:.1f}%)")
            pbar.update(1)
        pbar.close()

        metrics = self._aggregate_metrics()

        # Save results
        results_dir = Path("~/.hermes/finetune/bench/results").expanduser()
        results_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = results_dir / f"bench_{ts}.json"
        with open(result_path, "w") as f:
            json.dump({
                "metrics": metrics,
                "cases": [asdict(r) for r in self.results],
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)

        logger.info("Results saved to %s", result_path)

        # Print report
        if self.baseline:
            baseline_metrics = self.baseline.get("metrics", {})
            comparison = self._compare(metrics, baseline_metrics)
            checks = self._verdict(comparison)
            self._print_report(metrics, baseline_metrics, comparison, checks)
        else:
            self._print_report(metrics)

    def _aggregate_metrics(self) -> Dict[str, float]:
        tier1 = [r for r in self.results if r.tier == 1]
        tier2 = [r for r in self.results if r.tier == 2]
        tier3 = [r for r in self.results if r.tier == 3]
        canary = [r for r in self.results if r.is_canary]
        no_tool = [r for r in self.results
                   if r.tier == 1 and r.category == "no_tool_needed"]
        all_cases = self.results

        def safe_ratio(numerator, denominator):
            return numerator / denominator if denominator else 0.0

        return {
            "tool_selection_accuracy": safe_ratio(
                sum(1 for r in tier1 if r.tool_selection_correct), len(tier1)),
            "tool_execution_success": safe_ratio(
                sum(1 for r in tier2 if r.tool_args_valid), len(tier2)),
            "task_completion_rate": safe_ratio(
                sum(1 for r in tier3 if r.task_completed), len(tier3)),
            "format_compliance": safe_ratio(
                sum(1 for r in all_cases if r.format_valid and r.tool_call_parseable),
                len(all_cases)),
            "no_tool_accuracy": safe_ratio(
                sum(1 for r in no_tool if r.tool_selection_correct), len(no_tool)),
            "hallucination_rate": safe_ratio(
                sum(1 for r in all_cases if not r.tool_call_parseable),
                len(all_cases)),
            "mean_turns": safe_ratio(
                sum(r.turns_used for r in all_cases), len(all_cases)),
            "mean_errors": safe_ratio(
                sum(r.tool_errors for r in all_cases), len(all_cases)),
            "canary_pass_rate": safe_ratio(
                sum(1 for r in canary if r.reward > 0.5), len(canary)),
            "total_cases": len(all_cases),
        }

    def _compare(self, current: Dict, baseline: Dict) -> Dict:
        comparison = {}
        for key in current:
            if key in baseline and isinstance(current[key], (int, float)):
                comparison[key] = {
                    "baseline": baseline[key],
                    "candidate": current[key],
                    "delta": current[key] - baseline[key],
                }
        return comparison

    def _verdict(self, comparison: Dict) -> Dict:
        cfg = self.config
        checks = {}

        ts = comparison.get("tool_selection_accuracy", {})
        checks["tool_selection"] = ts.get("delta", 0) >= -cfg.regression_threshold_tool_selection

        te = comparison.get("tool_execution_success", {})
        checks["tool_execution"] = te.get("delta", 0) >= -cfg.regression_threshold_execution

        tc = comparison.get("task_completion_rate", {})
        checks["task_completion"] = tc.get("delta", 0) >= -cfg.regression_threshold_completion

        fc = comparison.get("format_compliance", {})
        checks["format_compliance"] = fc.get("candidate", 0) >= cfg.format_compliance_minimum

        hr = comparison.get("hallucination_rate", {})
        checks["no_hallucinations"] = hr.get("candidate", 0) == 0.0

        cr = comparison.get("canary_pass_rate", {})
        checks["canary"] = cr.get("delta", 0) >= -0.05

        checks["overall"] = all(v for k, v in checks.items() if k != "overall")
        return checks

    def _print_report(
        self,
        current: Dict,
        baseline: Dict = None,
        comparison: Dict = None,
        checks: Dict = None,
    ):
        w = 62
        print()
        print("+" + "=" * w + "+")

        if baseline and comparison:
            print(f"|{'FINETUNE BENCH — Comparison Report':^{w}}|")
            print("+" + "=" * w + "+")

            metrics = [
                ("Tool Selection Acc.", "tool_selection_accuracy", True),
                ("Tool Execution Succ.", "tool_execution_success", True),
                ("Task Completion Rate", "task_completion_rate", True),
                ("Format Compliance", "format_compliance", True),
                ("No-Tool Accuracy", "no_tool_accuracy", True),
                ("Hallucination Rate", "hallucination_rate", True),
                ("Mean Turns/Task", "mean_turns", False),
                ("Mean Errors/Task", "mean_errors", False),
                ("Canary Pass Rate", "canary_pass_rate", True),
            ]

            print(f"| {'Metric':<22} {'Baseline':>9} {'Candidate':>10} {'Delta':>10} |")
            print("+" + "-" * w + "+")

            for label, key, is_pct in metrics:
                if key not in comparison:
                    continue
                c = comparison[key]
                if is_pct:
                    b_s = f"{c['baseline']*100:.1f}%"
                    c_s = f"{c['candidate']*100:.1f}%"
                    d_s = f"{c['delta']*100:+.1f}%"
                else:
                    b_s = f"{c['baseline']:.1f}"
                    c_s = f"{c['candidate']:.1f}"
                    d_s = f"{c['delta']:+.1f}"
                print(f"| {label:<22} {b_s:>9} {c_s:>10} {d_s:>10} |")

            print("+" + "-" * w + "+")

            if checks:
                passed = sum(1 for k, v in checks.items() if k != "overall" and v)
                total = sum(1 for k in checks if k != "overall")
                verdict_str = "PASS" if checks["overall"] else "FAIL"
                print(f"| VERDICT: {verdict_str} ({passed}/{total} checks pass)")
                for k, v in checks.items():
                    if k != "overall" and not v:
                        print(f"| FAIL: {k}")
        else:
            print(f"|{'FINETUNE BENCH — Results':^{w}}|")
            print("+" + "=" * w + "+")
            for key, val in current.items():
                if isinstance(val, float):
                    print(f"| {key:<40} {val:>10.4f}    |")
                else:
                    print(f"| {key:<40} {str(val):>10}    |")

        print("+" + "=" * w + "+")
        print()


if __name__ == "__main__":
    FinetuneBenchEnv.cli()
