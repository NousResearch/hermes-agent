"""
FinetuneBenchEnv -- Fine-tune evaluation benchmark.

Runs a structured prompt bank against the hermes agent loop, scoring tool
selection, execution quality, and end-to-end task completion. This is the
evaluation gate referenced in the hermes-finetune design spec.

Standalone: drives the production agent (``run_agent.AIAgent``) directly
against an OpenAI-compatible endpoint. It has no dependency on the removed
Atropos RL environment framework, so it works both from a repo checkout and
from an installed skill (``<hermes-home>/skills/mlops/finetune/bench/``) —
the ``run_agent`` / ``tools`` / ``model_tools`` modules it drives are part
of the hermes-agent installation itself.

Usage:
    python finetune_bench_env.py evaluate \\
        --config default.yaml

    python finetune_bench_env.py evaluate \\
        --config default.yaml --env.prompt_bank_path /path/to/bank.yaml
"""

import argparse
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

BENCH_DIR = Path(__file__).resolve().parent

# Shared path constants live in the sibling scripts/ package of the skill.
_scripts_dir = str(BENCH_DIR.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from common import BENCH_DIR as BENCH_STATE_DIR  # noqa: E402  (~/.hermes/finetune/bench)

logger = logging.getLogger(__name__)

# Shared host scratch root. Each run works in a unique subdirectory
# (``run-<timestamp>-<uuid>``) so artifacts can never leak between runs even
# when cleanup fails (e.g. root-owned files created by the Docker container
# in the bind mount).
RUN_ROOT = Path("/tmp/finetune-bench")


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
    # The rollout failed for infrastructure reasons (endpoint timeout,
    # connection reset, ...). Such cases say nothing about model quality and
    # are excluded from every quality denominator.
    infra_error: bool = False
    # The model called a tool whose name is not in the served tool list —
    # a true hallucination (as opposed to unparseable tool-call JSON, which
    # is tracked separately as a parse failure).
    hallucinated_tool: bool = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class FinetuneBenchConfig:
    """Configuration for the finetune benchmark environment.

    Populated from the ``env:`` section of the config YAML; the ``setup:``
    section supplies the model endpoint. Unknown keys are ignored (with a
    debug log) so configs written for older revisions keep loading.
    """

    # Agent / sandbox
    enabled_toolsets: List[str] = field(default_factory=lambda: ["terminal", "file"])
    terminal_backend: str = "docker"
    # The bench executes agent-generated commands. Any backend other than
    # docker runs them directly on the host — refuse unless this flag (or
    # the FINETUNE_BENCH_ALLOW_UNSANDBOXED=1 env var) makes that explicit.
    allow_unsandboxed: bool = False
    max_agent_turns: int = 15
    agent_temperature: float = 0.1
    terminal_timeout: int = 60
    terminal_lifetime: int = 600
    max_token_length: int = 4096
    system_prompt: str = ""
    extra_body: Optional[Dict[str, Any]] = None
    # Fixed sampling seed sent with every request. Without it, run-to-run
    # sampling noise at low temperature is on the same order as the 3%
    # regression threshold, so the promotion gate flaps.
    seed: Optional[int] = 1234

    # Benchmark-specific
    prompt_bank_path: str = "prompt_bank.yaml"
    custom_cases_dir: str = "~/.hermes/finetune/bench/custom"
    baseline_results_path: Optional[str] = None
    regression_threshold_tool_selection: float = 0.03
    regression_threshold_execution: float = 0.05
    regression_threshold_completion: float = 0.05
    format_compliance_minimum: float = 0.95

    # Endpoint (from the ``setup:`` section)
    model_name: str = "carnice"
    base_url: str = "http://localhost:8008/v1"
    api_key: str = "none"

    @classmethod
    def load(cls, config_path: Path, overrides: Optional[Dict[str, str]] = None) -> "FinetuneBenchConfig":
        cfg = cls()
        data = {}
        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
        else:
            logger.warning("Config not found: %s (using defaults)", config_path)

        valid = set(cls.__dataclass_fields__)
        for key, value in (data.get("env") or {}).items():
            if key in valid:
                setattr(cfg, key, value)
            else:
                logger.debug("Ignoring unknown env config key: %s", key)

        setup = data.get("setup") or []
        if setup and isinstance(setup, list) and isinstance(setup[0], dict):
            first = setup[0]
            cfg.model_name = first.get("model_name", cfg.model_name)
            cfg.base_url = first.get("base_url", cfg.base_url)
            cfg.api_key = first.get("api_key", cfg.api_key)

        for key, value in (overrides or {}).items():
            if key not in valid:
                logger.warning("Ignoring unknown --env override: %s", key)
                continue
            current = getattr(cfg, key)
            if isinstance(current, bool):
                value = str(value).lower() in {"1", "true", "yes"}
            elif isinstance(current, int):
                value = int(value)
            elif isinstance(current, float):
                value = float(value)
            setattr(cfg, key, value)

        return cfg


# =============================================================================
# Verification context
# =============================================================================

class VerifyContext:
    """Run verification commands in the same sandbox the agent used.

    Sharing the rollout's ``task_id`` means terminal commands see the same
    container/cwd state (files, processes) the model produced.
    """

    def __init__(self, task_id: str):
        self.task_id = task_id

    def terminal(self, command: str, timeout: int = 180) -> Dict[str, Any]:
        """Run a command; returns a dict with 'exit_code' and 'output'."""
        import asyncio
        import concurrent.futures

        from model_tools import handle_function_call

        def _call() -> str:
            return handle_function_call(
                "terminal",
                {"command": command, "timeout": timeout},
                task_id=self.task_id,
            )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            result = _call()
        else:
            # Container backends use asyncio.run() internally — hop to a
            # worker thread so they get a clean event loop.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(_call).result(timeout=300)

        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"exit_code": -1, "output": result}

    def cleanup(self) -> None:
        """Tear down the rollout's sandbox (containers, sessions)."""
        try:
            from tools.terminal_tool import cleanup_vm
            cleanup_vm(self.task_id)
        except Exception as e:
            logger.debug("Sandbox cleanup failed for %s: %s", self.task_id[:8], e)


# =============================================================================
# Environment
# =============================================================================

class FinetuneBenchEnv:
    """
    Fine-tune evaluation benchmark environment.

    Runs test cases from a prompt bank and scores tool selection,
    execution quality, and end-to-end task completion.
    """

    name = "finetune-bench"

    # A run whose infra-error fraction exceeds this is invalid: too many
    # cases were skipped for the metrics to be trustworthy.
    MAX_INFRA_ERROR_FRACTION = 0.05

    # Process exit codes (also documented in the bench spec):
    #   1 — baseline comparison verdict is FAIL
    #   2 — configured LLM endpoint unreachable
    #   3 — run invalid (too many infrastructure errors)
    #   4 — sandbox unavailable (Docker daemon down) or an unsandboxed
    #       terminal backend was configured without an explicit override
    EXIT_VERDICT_FAIL = 1
    EXIT_ENDPOINT_UNREACHABLE = 2
    EXIT_RUN_INVALID = 3
    EXIT_SANDBOX_UNAVAILABLE = 4

    def __init__(self, config: FinetuneBenchConfig):
        self.config = config
        self.prompt_bank: List[Dict[str, Any]] = []
        self.baseline: Optional[Dict[str, Any]] = None
        self.results: List[CaseResult] = []
        # Case ids with authoring errors (e.g. an output_match verification
        # without an expected_value/expected_regex, or a working_dir outside
        # the bench scratch root). Malformed cases are scored as failed
        # checks — never as vacuous passes — and their count is surfaced in
        # the metrics so they can't silently inflate anything.
        self.malformed_case_ids: set = set()
        # Unique per-run scratch base: every case works under this directory,
        # and case-authored /tmp/finetune-bench/... paths are remapped into it,
        # so stale artifacts from a previous run can never satisfy this run's
        # file_exists / functional checks.
        self.run_base = RUN_ROOT / (
            f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        )

    def _remap_case_path(self, value: str) -> str:
        """Rewrite the shared scratch root in case-authored strings (working
        dirs, test commands, file_exists paths) to this run's unique base."""
        if not isinstance(value, str):
            return value
        return value.replace(str(RUN_ROOT), str(self.run_base))

    def setup(self):
        """Load prompt bank, custom cases, and configure the sandbox."""
        # Force non-interactive mode for any tool calls the agent loop makes
        # during the rollout. The bench env is definitionally non-interactive
        # — there's no user to enter a sudo password, click an approval
        # button, or answer a clarify question. When the bench is launched
        # from inside hermes, HERMES_INTERACTIVE is inherited from the
        # parent, which causes the agent's terminal tool to pop a 45-second
        # sudo password prompt that blocks the rollout. HERMES_YOLO_MODE
        # bypasses the dangerous-command approval flow for the same reason.
        os.environ.pop("HERMES_INTERACTIVE", None)
        os.environ["HERMES_YOLO_MODE"] = "1"
        os.environ.setdefault("SUDO_PASSWORD", "")  # ensures no prompt is shown

        # Map sandbox config onto the terminal tool's environment knobs.
        # This process is a dedicated bench subprocess, so mutating our own
        # environment doesn't leak into the dispatching hermes session.
        if self.config.terminal_backend:
            os.environ["TERMINAL_ENV"] = self.config.terminal_backend
        os.environ["TERMINAL_TIMEOUT"] = str(self.config.terminal_timeout)
        os.environ["TERMINAL_LIFETIME_SECONDS"] = str(self.config.terminal_lifetime)

        # Enforce the sandbox contract BEFORE any rollout. "Docker is
        # required" must be more than a config default: with the daemon
        # down, every case would fail as a bogus QUALITY failure and the
        # run would "validly" poison the baseline.
        self._enforce_sandbox_backend()

        # Pre-flight: confirm the configured LLM endpoint is reachable BEFORE
        # we start the rollout loop. Without this check, a dead llama-server
        # causes the bench to hang silently on its first request. Fail loud
        # and fast instead.
        self._preflight_health_check()

        # Suppress the per-container disk-quota warning that fires for every
        # case on systems without overlay2-on-XFS-with-pquota. It's harmless
        # informational noise and would print 243 times in a full run.
        class _DropDiskQuotaWarning(logging.Filter):
            def filter(self, record):
                return "per-container disk limits" not in record.getMessage()

        logging.getLogger("tools.environments.docker").addFilter(_DropDiskQuotaWarning())

        # Bind-mount the host scratch root into every container the bench
        # spawns so per-case working dirs (created by _rollout_case under
        # the unique per-run base) are visible inside the sandbox at the
        # same path.
        host_scratch = RUN_ROOT
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

        bank_path = Path(self.config.prompt_bank_path).expanduser()
        if not bank_path.is_absolute():
            bank_path = BENCH_DIR / bank_path

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
            bp = Path(self.config.baseline_results_path).expanduser()
            if bp.exists():
                with open(bp) as f:
                    self.baseline = json.load(f)

        self.results = []

        logger.info("Loaded %d test cases", len(self.prompt_bank))

    def _preflight_health_check(self):
        """
        Verify the configured LLM server is reachable before starting the
        rollout loop. Raises SystemExit with a clear message if not, so
        the failure is loud and immediate instead of a silent hang.
        """
        import urllib.request
        import urllib.error

        base_url = self.config.base_url
        if not base_url:
            print("[finetune-bench] ⚠ no base_url configured — "
                  "skipping pre-flight health check")
            return

        # Construct the health-check URL. /v1/models is the standard
        # OpenAI-compatible endpoint and is what llama-server, vLLM,
        # SGLang, and OpenRouter all expose.
        health_url = base_url.rstrip("/")
        if not health_url.endswith("/v1/models"):
            if health_url.endswith("/v1"):
                health_url = health_url + "/models"
            else:
                health_url = health_url + "/v1/models"

        print(f"[finetune-bench] pre-flight: GET {health_url}")
        request = urllib.request.Request(health_url)
        api_key = (self.config.api_key or "").strip()
        if api_key and api_key.lower() != "none":
            # Auth-gated endpoints 401 an anonymous /v1/models probe even
            # when they're perfectly healthy — send the configured key.
            request.add_header("Authorization", f"Bearer {api_key}")
        try:
            # urlopen raises HTTPError for any non-2xx status, so reaching
            # the body of this ``with`` means the server answered 2xx.
            with urllib.request.urlopen(request, timeout=5):
                print("[finetune-bench] pre-flight: ✓ server is responsive")
                return
        except urllib.error.HTTPError as e:
            print(
                f"[finetune-bench] pre-flight: ✗ server returned HTTP {e.code}"
            )
            if e.code in (401, 403):
                print(
                    "  The endpoint rejected the configured api_key — check "
                    "the bench config's setup.api_key."
                )
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            print(
                f"[finetune-bench] pre-flight: ✗ could not reach LLM server at {base_url}"
            )
            print(f"  reason: {e}")
            print(
                "  Start your local model (e.g. llama-server) before running the bench, "
                "or update the bench default.yaml's setup.base_url to point at a "
                "different endpoint."
            )

        # If we reach here, the server is unreachable. Exit with a non-zero
        # code so the parent dispatcher (manage.py / cli.py) sees a real
        # failure instead of waiting hours for a hung run.
        raise SystemExit(self.EXIT_ENDPOINT_UNREACHABLE)

    def _enforce_sandbox_backend(self) -> None:
        """Verify the sandbox contract before any case runs.

        docker backend  → the daemon must actually answer ``docker info``.
        anything else   → refuse unless FINETUNE_BENCH_ALLOW_UNSANDBOXED=1
                          (or config ``allow_unsandboxed: true``) explicitly
                          accepts running agent-generated commands on the
                          host.
        """
        backend = (os.environ.get("TERMINAL_ENV") or "local").strip().lower()
        if backend == "docker":
            self._preflight_docker_check()
            return

        allowed = (
            os.environ.get("FINETUNE_BENCH_ALLOW_UNSANDBOXED") == "1"
            or self.config.allow_unsandboxed
        )
        if not allowed:
            print(
                f"[finetune-bench] ✗ refusing terminal_backend={backend!r}: "
                "the bench executes agent-generated shell commands, and any "
                "non-docker backend runs them DIRECTLY ON THIS HOST."
            )
            print(
                "  If you really want that, set FINETUNE_BENCH_ALLOW_UNSANDBOXED=1 "
                "(or allow_unsandboxed: true in the bench config) and re-run."
            )
            raise SystemExit(self.EXIT_SANDBOX_UNAVAILABLE)
        print(
            f"[finetune-bench] ⚠ UNSANDBOXED RUN: terminal_backend={backend!r} "
            "executes agent-generated commands directly on this host. "
            "You accepted this via FINETUNE_BENCH_ALLOW_UNSANDBOXED/allow_unsandboxed."
        )

    def _preflight_docker_check(self) -> None:
        """Hard-fail (distinct exit code) when the Docker daemon is unusable."""
        try:
            proc = subprocess.run(
                ["docker", "info"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=30,
            )
        except FileNotFoundError:
            print(
                "[finetune-bench] ✗ Docker backend is configured but the "
                "'docker' CLI was not found on PATH. Install Docker or point "
                "the bench at a different sandbox."
            )
            raise SystemExit(self.EXIT_SANDBOX_UNAVAILABLE)
        except (subprocess.TimeoutExpired, OSError) as e:
            print(f"[finetune-bench] ✗ 'docker info' failed to run: {e}")
            raise SystemExit(self.EXIT_SANDBOX_UNAVAILABLE)

        if proc.returncode != 0:
            stderr = (proc.stderr or b"").decode("utf-8", "replace").strip()
            print(
                "[finetune-bench] ✗ the Docker daemon is not usable "
                f"(docker info exited {proc.returncode}). Start it and re-run."
            )
            if stderr:
                print(f"  docker info: {stderr.splitlines()[-1]}")
            raise SystemExit(self.EXIT_SANDBOX_UNAVAILABLE)

        print("[finetune-bench] pre-flight: ✓ Docker daemon is responsive")

    def format_prompt(self, item):
        return item["prompt"]

    # =========================================================================
    # Scoring
    # =========================================================================

    def compute_reward(self, item, messages: List[Dict], turns_used: int,
                       tool_errors: int, ctx: VerifyContext,
                       available_tools: Optional[set] = None) -> CaseResult:
        """Score a single test case."""
        case = CaseResult(
            case_id=item["id"],
            tier=item["tier"],
            category=item["category"],
            tags=item.get("tags", []),
            is_canary=item.get("canary", False),
            turns_used=turns_used,
            tool_errors=tool_errors,
        )

        # A tool result showing the bench's own Docker daemon went away
        # mid-run is an infrastructure failure, not model quality. Record
        # it as infra (excluded from quality denominators, counted toward
        # run invalidation) instead of a bogus quality zero.
        if self._docker_infra_failure(messages):
            logger.error(
                "Case %s: Docker daemon failure in tool results — "
                "recording as infra error", item.get("id"),
            )
            case.infra_error = True
            self.results.append(case)
            return case

        # Format compliance
        case.format_valid = self._check_format(messages)
        case.tool_call_parseable = self._check_tool_parse(messages)

        actual_tools = self._extract_tool_calls(messages)

        # True hallucination: a call to a tool name the server never offered.
        if available_tools:
            case.hallucinated_tool = any(
                t.get("name") not in available_tools for t in actual_tools
            )

        # Tier 1: Tool selection
        if item["tier"] >= 1:
            expected = item.get("expected") or {}

            if expected.get("should_call_tool") is False:
                case.tool_selection_correct = len(actual_tools) == 0
            elif expected.get("tool_name"):
                case.tool_selection_correct = any(
                    t["name"] == expected["tool_name"] for t in actual_tools
                )
            elif expected.get("should_call_tool") is True:
                # Cases (e.g. skill_invocation) that only assert *a* tool is
                # used, without pinning which one.
                case.tool_selection_correct = len(actual_tools) > 0

        # Tier 2/3: Execution quality and end-to-end completion.
        # Verification runs exactly ONCE per case — tier-3 functional cases
        # reuse the same result for both tool_args_valid and task_completed
        # (re-running test_commands would double wall time and re-execute
        # mutating commands, letting the two metrics disagree).
        if item["tier"] >= 2 and item.get("verification"):
            v = item["verification"]
            if v.get("method") == "functional_test" and v.get("checks"):
                ok = self._verify_functional(ctx, item, v)
                case.tool_args_valid = ok
                case.task_completed = ok
            else:
                # output_match — or a functional_test with no checks, which
                # can only be scored against the transcript.
                ok = self._verify_output(messages, v, case_id=item.get("id"))
                case.tool_args_valid = ok
                if item["tier"] >= 3:
                    case.task_completed = ok

        # Composite reward
        case.reward = self._compute_composite(case, item["tier"])
        self.results.append(case)
        return case

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

    # Heuristic markers of a failed tool execution in plain-text output.
    _TOOL_ERROR_MARKERS = (
        "traceback (most recent call last)",
        "command not found",
        "no such file or directory",
        "permission denied",
        "segmentation fault",
        "is not recognized as an internal",
        "fatal:",
    )

    # Markers of the BENCH's own Docker sandbox dying mid-run. These come
    # from the host-side docker client, never from commands the agent runs
    # inside the container (there is no docker CLI in the sandbox image, so
    # an in-container `docker ...` prints "command not found" instead).
    _DOCKER_INFRA_MARKERS = (
        "cannot connect to the docker daemon",
        "is the docker daemon running",
        "error during connect",
        "docker daemon is not running",
        "docker is not available",
    )

    @classmethod
    def _docker_infra_failure(cls, messages: List[Dict]) -> bool:
        """True if any tool result shows the bench's Docker backend failed."""
        for msg in messages:
            if msg.get("role") == "tool" and isinstance(msg.get("content"), str):
                low = msg["content"].lower()
                if any(m in low for m in cls._DOCKER_INFRA_MARKERS):
                    return True
        return False

    def _tool_output_ok(self, content: Any) -> bool:
        """True if a tool result looks like a successful execution: non-empty
        and free of error signatures (nonzero exit codes, tracebacks, ...)."""
        if not isinstance(content, str) or not content.strip():
            return False

        # Terminal tool results are JSON: {"exit_code": N, "output": "..."}.
        # An explicit exit code is authoritative.
        parsed = None
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            pass
        if isinstance(parsed, dict):
            exit_code = parsed.get("exit_code")
            if isinstance(exit_code, (int, float)) and not isinstance(exit_code, bool):
                return int(exit_code) == 0
            out = parsed.get("output")
            if isinstance(out, str):
                content = out

        low = content.lower()
        if low.lstrip().startswith("error"):
            return False
        if any(marker in low for marker in self._TOOL_ERROR_MARKERS):
            return False
        # Plain-text nonzero-exit markers, e.g. "exit code: 127".
        m = re.search(r"exit[\s_-]?code\D{0,3}(-?\d+)", low)
        if m and m.group(1) != "0":
            return False
        return True

    @staticmethod
    def _final_assistant_text(messages: List[Dict]) -> str:
        """The last non-empty assistant text — the model's final answer."""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content
        return ""

    @staticmethod
    def _is_short_expectation(expected: str) -> bool:
        """Numeric or very short values substring-match everywhere (a bare
        "7" appears in timestamps, sizes, ...) — they need word boundaries."""
        return bool(re.fullmatch(r"-?\d+(\.\d+)?", expected)) or len(expected) <= 3

    def _verify_output(self, messages: List[Dict], verification: Dict,
                       case_id: Optional[str] = None) -> bool:
        """Score an ``output_match`` verification against the transcript.

        Verification fields:
          expected_value   substring (word-boundary matched when short/numeric)
          expected_regex   regex alternative/complement to expected_value
          match_scope      "transcript" (default: final answer + successful
                           tool results) or "final_answer" (the model's
                           conclusion must state the value itself)
          expect_failure   error-recovery semantics — execution credit
                           requires (a) at least one FAILED tool call and
                           (b) the final assistant answer acknowledging /
                           explaining the failure (matched case-insensitively
                           via expected_regex/expected_value)

        A case that asserts nothing (no expected_value and no expected_regex)
        is a case-authoring error: it is logged, counted in the
        ``malformed_cases`` metric, and scored as FAILED — it can never be a
        vacuous pass.
        """
        expected = verification.get("expected_value", "")
        expected = "" if expected is None else str(expected).strip()
        expected_regex = str(verification.get("expected_regex") or "").strip()

        if not expected and not expected_regex:
            logger.error(
                "Case %s: output_match verification without expected_value/"
                "expected_regex — malformed case, check scored as FAILED",
                case_id,
            )
            if case_id:
                self.malformed_case_ids.add(str(case_id))
            return False

        tool_texts: List[str] = []
        ok_tool_texts: List[str] = []
        failed_tool_texts: List[str] = []
        for msg in messages:
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    tool_texts.append(content)
                    if self._tool_output_ok(content):
                        ok_tool_texts.append(content)
                    else:
                        failed_tool_texts.append(content)

        final_answer = self._final_assistant_text(messages)

        if verification.get("expect_failure"):
            # The correct trajectory INCLUDES a failing command. Nonzero
            # exits must not be penalized here — they are the point.
            if not failed_tool_texts:
                return False  # nothing failed: the scenario never happened
            if expected_regex:
                return bool(re.search(expected_regex, final_answer, re.IGNORECASE))
            return expected.lower() in final_answer.lower()

        final_only = verification.get("match_scope") == "final_answer"

        if expected_regex:
            candidates = [final_answer] if final_only else [final_answer] + ok_tool_texts
            return any(re.search(expected_regex, c) for c in candidates if c)

        if self._is_short_expectation(expected):
            # Word-boundary match, restricted to the final assistant answer
            # or the last successful tool result — never "any message".
            pattern = re.compile(
                r"(?<![\w.])" + re.escape(expected) + r"(?![\w.])"
            )
            candidates = [final_answer] if final_only else (
                [final_answer] + (ok_tool_texts[-1:] if ok_tool_texts else [])
            )
            return any(pattern.search(c) for c in candidates if c)

        # Long values: substring over the final answer plus SUCCESSFUL tool
        # results only — error output must not satisfy a content assertion.
        candidates = [final_answer] if final_only else [final_answer] + ok_tool_texts
        return any(expected in c for c in candidates if c)

    def _verify_functional(self, ctx: VerifyContext, item: Dict, verification: Dict) -> bool:
        checks = verification.get("checks", [])
        # Case-authored commands/paths reference the shared scratch root —
        # remap them into this run's unique base dir.
        commands = [
            self._remap_case_path(c) for c in verification.get("test_commands", [])
        ]

        outputs = []
        for cmd in commands:
            try:
                result = ctx.terminal(cmd, timeout=30)
                outputs.append(result)
            except Exception as e:
                outputs.append({"output": str(e), "exit_code": -1})

        passed = 0
        for check in checks:
            check_type = check.get("type")

            # file_exists runs its own probe — it does not consume a
            # test_commands output, so it must work with zero test_commands.
            if check_type == "file_exists":
                try:
                    path = self._remap_case_path(str(check.get("path", "")))
                    result = ctx.terminal(
                        f"test -f {shlex.quote(path)} && echo EXISTS", timeout=10
                    )
                    if "EXISTS" in result.get("output", ""):
                        passed += 1
                except Exception:
                    pass
                continue

            idx = check.get("command_index", 0)
            if idx >= len(outputs):
                # Authoring error: the check references a command that was
                # never defined. Count it as FAILED (loudly), never skip it.
                logger.warning(
                    "Case %s: check %r references command_index %d but only "
                    "%d test command(s) exist — counting the check as failed",
                    item.get("id"), check_type, idx, len(outputs),
                )
                continue

            output = outputs[idx]

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

        return passed == len(checks) if checks else False

    # =========================================================================
    # Per-case rollout (called by evaluate)
    # =========================================================================

    @staticmethod
    def _is_infra_error(exc: BaseException) -> bool:
        """Classify a rollout exception as infrastructure/transport (endpoint
        timeout, connection reset, ...) rather than model quality."""
        if isinstance(exc, (ConnectionError, TimeoutError)):
            return True
        name = type(exc).__name__.lower()
        if "timeout" in name or "connection" in name:
            return True
        msg = str(exc).lower()
        markers = (
            "timed out",
            "timeout",
            "connection reset",
            "connection refused",
            "connection aborted",
            "connection error",
            "remote end closed",
            "temporarily unavailable",
            "name or service not known",
            "bad gateway",
            "service unavailable",
            "gateway timeout",
            # The bench's own Docker sandbox went away mid-run.
            "cannot connect to the docker daemon",
            "is the docker daemon running",
            "docker daemon is not running",
            "docker is not available",
        )
        return any(m in msg for m in markers)

    def _failed_case_result(self, item: Dict, *, infra: bool = False,
                            turns_used: int = 0, tool_errors: int = 0,
                            quality_failure: bool = False) -> CaseResult:
        """Record a zero-reward result for THIS case (never dropped, never a
        duplicate of the previous case)."""
        case = CaseResult(
            case_id=item.get("id", "?"),
            tier=item.get("tier", 1),
            category=item.get("category", "unknown"),
            tags=item.get("tags", []),
            is_canary=item.get("canary", False),
            turns_used=turns_used,
            tool_errors=tool_errors,
            # Only a genuine model-side failure counts against format /
            # parseability; infra errors and scorer bugs stay neutral.
            format_valid=not quality_failure,
            tool_call_parseable=not quality_failure,
            infra_error=infra,
            reward=0.0,
        )
        self.results.append(case)
        return case

    def _rollout_case(self, item: Dict) -> Optional[CaseResult]:
        """Run one test case end-to-end: setup → agent loop → score."""
        from run_agent import AIAgent
        from tools.terminal_tool import (
            register_task_env_overrides,
            clear_task_env_overrides,
        )

        task_id = str(uuid.uuid4())

        # --- Setup phase: every case gets an isolated working directory ---
        # Case-specified working_dirs are remapped from the shared scratch
        # root into this run's unique base; otherwise mint a per-case dir
        # under it. Unique per-run bases make cross-run leakage impossible
        # even when cleanup fails (e.g. root-owned container artifacts).
        setup_cfg = item.get("setup") or {}
        working_dir = setup_cfg.get("working_dir")
        if working_dir:
            working_dir = self._remap_case_path(working_dir)
        else:
            working_dir = str(self.run_base / str(item.get("id", task_id[:8])))

        wd = Path(working_dir).expanduser()

        # Safety guard: the setup phase rmtree-s the working dir. Only paths
        # under this run's scratch base are legitimate — a custom case with
        # e.g. ``working_dir: ~/projects`` must never be wiped. Skip such
        # cases loudly instead of touching the path.
        try:
            resolved = wd.resolve()
            base = self.run_base.resolve()
            inside_base = resolved == base or base in resolved.parents
        except OSError as e:
            logger.error("Case %s: cannot resolve working_dir %s: %s",
                         item.get("id"), wd, e)
            inside_base = False
        if not inside_base:
            logger.error(
                "Case %s: working_dir %s does not resolve under the bench "
                "scratch root %s — SKIPPING case (authoring error). Use a "
                "path under %s in custom cases.",
                item.get("id"), wd, self.run_base, RUN_ROOT,
            )
            self.malformed_case_ids.add(str(item.get("id")))
            return None

        if wd.exists():
            try:
                shutil.rmtree(wd)
            except OSError as e:
                logger.warning("Could not fully clean working dir %s: %s", wd, e)
        wd.mkdir(parents=True, exist_ok=True)

        # Seed files specified in setup.files
        for fname, content in (setup_cfg.get("files") or {}).items():
            fpath = wd / fname
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(content)

        # Pin the terminal sandbox cwd for this rollout so all commands the
        # agent runs land inside the per-case dir, not the host cwd.
        register_task_env_overrides(task_id, {"cwd": str(wd)})

        # --- Build prompt ---
        # Tell the model its working directory so it doesn't try to use
        # absolute paths or assume it's somewhere else.
        prompt_text = (
            f"[Working directory: {wd}]\n\n{self.format_prompt(item)}"
        )

        # --- Run the production agent loop against the configured endpoint ---
        request_overrides: Dict[str, Any] = {
            "temperature": self.config.agent_temperature,
        }
        if self.config.seed is not None:
            # Fixed seed on every request: keeps run-to-run noise below the
            # regression thresholds so the promotion gate doesn't flap.
            request_overrides["seed"] = int(self.config.seed)
        if self.config.extra_body:
            request_overrides["extra_body"] = dict(self.config.extra_body)

        agent = AIAgent(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            model=self.config.model_name,
            max_iterations=item.get("max_turns") or self.config.max_agent_turns,
            enabled_toolsets=list(self.config.enabled_toolsets),
            quiet_mode=True,
            max_tokens=self.config.max_token_length,
            request_overrides=request_overrides,
            session_id=f"finetune-bench-{task_id[:8]}",
        )

        # Tool names actually served to the model — used to flag true
        # hallucinations (calls to tools that were never offered).
        served_tool_names = getattr(agent, "valid_tool_names", None)

        try:
            try:
                result = agent.run_conversation(
                    prompt_text,
                    system_message=self.config.system_prompt or None,
                    task_id=task_id,
                )
                messages = result.get("messages", [])
            except Exception as e:
                infra = self._is_infra_error(e)
                logger.error(
                    "Case %s rollout failed (%s): %s",
                    item.get("id"),
                    "infrastructure" if infra else "quality",
                    e,
                )
                # Infra/transport failures say nothing about the model —
                # they're excluded from quality denominators. Anything else
                # counts against format/parseability as before.
                return self._failed_case_result(
                    item, infra=infra, quality_failure=not infra
                )

            turns_used = sum(
                1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant"
            )
            tool_errors = sum(
                1 for m in messages
                if isinstance(m, dict)
                and m.get("role") == "tool"
                and isinstance(m.get("content"), str)
                and m["content"].lstrip().lower().startswith("error")
            )

            # --- Score ---
            ctx = VerifyContext(task_id)
            try:
                return self.compute_reward(
                    item, messages, turns_used, tool_errors, ctx,
                    available_tools=served_tool_names,
                )
            except Exception as e:
                # A scorer bug (e.g. malformed check dict) must record a
                # failed result for THIS case — never drop it or return the
                # previous case's result, which would skew denominators.
                logger.error("Case %s scoring failed: %s", item.get("id"), e)
                return self._failed_case_result(
                    item, turns_used=turns_used, tool_errors=tool_errors
                )
            finally:
                try:
                    ctx.cleanup()
                except Exception:
                    pass
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

    def evaluate(self) -> int:
        """Iterate the prompt bank, score each case, then aggregate.

        Returns a process exit code: 0 on success (or when there is no
        baseline to compare against), EXIT_VERDICT_FAIL when the baseline
        comparison verdict is FAIL. Invalid runs raise SystemExit instead.
        """
        if not self.prompt_bank:
            self.setup()

        if not self.prompt_bank:
            print("[finetune-bench] No test cases loaded — check prompt_bank_path")
            return self.EXIT_VERDICT_FAIL

        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        total = len(self.prompt_bank)
        print(f"\n{'='*60}")
        print(f"  FINETUNE BENCH — running {total} test cases")
        print(f"{'='*60}\n")

        pbar = tqdm(total=total, desc="Evaluating", dynamic_ncols=True) if tqdm else None
        for i, item in enumerate(self.prompt_bank, 1):
            try:
                self._rollout_case(item)
            except Exception as e:
                logger.error("Case %s failed in evaluate loop: %s", item.get("id"), e)
            passed = sum(1 for r in self.results if r.reward >= 0.5)
            done = len(self.results)
            pct = (passed / done * 100) if done else 0.0
            if pbar:
                pbar.set_postfix_str(f"pass={passed}/{done} ({pct:.1f}%)")
                pbar.update(1)
            else:
                print(f"  [{i}/{total}] pass={passed}/{done} ({pct:.1f}%)")
        if pbar:
            pbar.close()

        metrics = self._aggregate_metrics()

        # Compare against the baseline BEFORE saving so the verdict lands
        # in the results JSON (additive key — consumers of the metrics/cases
        # keys are unaffected).
        baseline_metrics = comparison = checks = None
        if self.baseline:
            baseline_metrics = self.baseline.get("metrics", {})
            comparison = self._compare(metrics, baseline_metrics)
            checks = self._verdict(comparison)

        # Save results
        results_dir = BENCH_STATE_DIR / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = results_dir / f"bench_{ts}.json"
        payload = {
            "metrics": metrics,
            "cases": [asdict(r) for r in self.results],
            "timestamp": datetime.now().isoformat(),
        }
        if checks is not None:
            payload["verdict"] = checks
        with open(result_path, "w") as f:
            json.dump(payload, f, indent=2)

        logger.info("Results saved to %s", result_path)

        # Clean this run's scratch tree. Failures are logged, not ignored —
        # but a leak is harmless because the next run uses a fresh base dir.
        if self.run_base.exists():
            try:
                shutil.rmtree(self.run_base)
            except OSError as e:
                logger.warning(
                    "Could not remove run scratch dir %s: %s", self.run_base, e
                )

        # Too many infra errors → results untrustworthy; fail loudly.
        self._validate_run(metrics)

        # Print report
        if checks is not None:
            self._print_report(metrics, baseline_metrics, comparison, checks)
            if not checks.get("overall", False):
                return self.EXIT_VERDICT_FAIL
        else:
            self._print_report(metrics)
        return 0

    def _aggregate_metrics(self) -> Dict[str, float]:
        # Infra-error cases (endpoint timeouts, connection resets) say
        # nothing about model quality — exclude them from every quality
        # denominator and report their count separately.
        scored = [r for r in self.results if not r.infra_error]
        infra_errors = len(self.results) - len(scored)

        tier1 = [r for r in scored if r.tier == 1]
        tier2 = [r for r in scored if r.tier == 2]
        tier3 = [r for r in scored if r.tier == 3]
        canary = [r for r in scored if r.is_canary]
        no_tool = [r for r in scored
                   if r.tier == 1 and r.category == "no_tool_needed"]
        all_cases = scored

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
            # True hallucinations: calls to tool names the server never
            # offered. Unparseable tool-call JSON is a separate metric.
            "hallucination_rate": safe_ratio(
                sum(1 for r in all_cases if r.hallucinated_tool),
                len(all_cases)),
            "tool_call_parse_failure_rate": safe_ratio(
                sum(1 for r in all_cases if not r.tool_call_parseable),
                len(all_cases)),
            "mean_turns": safe_ratio(
                sum(r.turns_used for r in all_cases), len(all_cases)),
            "mean_errors": safe_ratio(
                sum(r.tool_errors for r in all_cases), len(all_cases)),
            "canary_pass_rate": safe_ratio(
                sum(1 for r in canary if r.reward > 0.5), len(canary)),
            "total_cases": len(self.results),
            "scored_cases": len(scored),
            "infra_errors": infra_errors,
            "infra_error_rate": safe_ratio(infra_errors, len(self.results)),
            # Case-authoring errors (empty output_match assertions, unsafe
            # working_dirs). These score as failures/skips, never as passes.
            "malformed_cases": len(self.malformed_case_ids),
        }

    def _validate_run(self, metrics: Dict[str, float]) -> None:
        """Invalidate the run (clear error, nonzero exit) when too many cases
        hit infrastructure errors for the metrics to be trustworthy."""
        frac = metrics.get("infra_error_rate", 0.0)
        if frac > self.MAX_INFRA_ERROR_FRACTION:
            n = metrics.get("infra_errors", 0)
            print(
                f"[finetune-bench] ✗ RUN INVALID: {n} case(s) "
                f"({frac * 100:.1f}%) failed with infrastructure errors "
                f"(limit {self.MAX_INFRA_ERROR_FRACTION * 100:.0f}%). "
                "The metrics are not trustworthy — fix the endpoint/network "
                "and re-run the benchmark."
            )
            raise SystemExit(self.EXIT_RUN_INVALID)

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
        """Gate the run on baseline comparison — fail CLOSED.

        Consistent with ``eval.py``: a gate only applies to metrics present
        in the candidate/baseline intersection (``comparison``). A missing
        gate metric is skipped with a printed warning — it never auto-passes.
        If NO gate metric is comparable at all, the verdict is FAIL: an
        empty intersection means the baseline says nothing about this run.
        """
        cfg = self.config

        # (metric key, check name, pass predicate over the comparison entry)
        gates = [
            ("tool_selection_accuracy", "tool_selection",
             lambda c: c["delta"] >= -cfg.regression_threshold_tool_selection),
            ("tool_execution_success", "tool_execution",
             lambda c: c["delta"] >= -cfg.regression_threshold_execution),
            ("task_completion_rate", "task_completion",
             lambda c: c["delta"] >= -cfg.regression_threshold_completion),
            ("format_compliance", "format_compliance",
             lambda c: c["candidate"] >= cfg.format_compliance_minimum),
            # Small tolerance instead of an exact-zero gate: a single flaky
            # case must not auto-FAIL the whole run.
            ("hallucination_rate", "no_hallucinations",
             lambda c: c["candidate"] <= max(0.01, c["baseline"] + 0.01)),
            ("canary_pass_rate", "canary",
             lambda c: c["delta"] >= -0.05),
        ]

        checks = {}
        for metric, name, predicate in gates:
            if metric in comparison:
                checks[name] = bool(predicate(comparison[metric]))
            else:
                print(
                    f"[finetune-bench] ⚠ gate '{name}' skipped: metric "
                    f"'{metric}' missing from candidate/baseline intersection"
                )

        if not checks:
            print(
                "[finetune-bench] ✗ no gate metric is comparable between "
                "candidate and baseline — verdict FAIL (empty intersection)"
            )
            checks["overall"] = False
            return checks

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
                ("Parse Failure Rate", "tool_call_parse_failure_rate", True),
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


# =============================================================================
# CLI
# =============================================================================

def _parse_env_overrides(unknown_args: List[str]) -> Dict[str, str]:
    """Parse ``--env.<key> <value>`` (or ``--env.<key>=<value>``) pairs."""
    overrides: Dict[str, str] = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith("--env."):
            key = arg[len("--env."):]
            if "=" in key:
                key, _, value = key.partition("=")
                overrides[key] = value
            elif i + 1 < len(unknown_args):
                overrides[key] = unknown_args[i + 1]
                i += 1
            else:
                logger.warning("Missing value for override %s", arg)
        else:
            logger.warning("Ignoring unknown argument: %s", arg)
        i += 1
    return overrides


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Finetune evaluation benchmark")
    parser.add_argument("command", choices=["evaluate"], help="Action to run")
    parser.add_argument(
        "--config",
        default=str(BENCH_DIR / "default.yaml"),
        help="Path to the bench config YAML",
    )
    args, unknown = parser.parse_known_args(argv)
    overrides = _parse_env_overrides(unknown)

    config = FinetuneBenchConfig.load(Path(args.config).expanduser(), overrides)
    env = FinetuneBenchEnv(config)
    env.setup()
    # A FAIL verdict against the configured baseline is a real failure —
    # propagate it as a nonzero exit code so dispatchers can't miss it.
    return int(env.evaluate() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
