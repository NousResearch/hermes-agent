# Design: Fine-Tune Evaluation Benchmark (`finetune-bench`)

**Status:** Proposal (companion to `hermes-finetune` design spec)
**Target:** `optional-skills/mlops/finetune/bench/`
**Framework:** standalone — drives the production agent (`run_agent.AIAgent`) directly. (Originally targeted the Atropos `HermesAgentBaseEnv` framework, which was removed upstream in #26106; code snippets below predate that rewrite and are kept for design rationale only — see `optional-skills/mlops/finetune/bench/finetune_bench_env.py` for the current implementation.)

---

## Purpose

Answer one question: **did the fine-tune help?**

This benchmark runs before and after every adapter training cycle, producing a quantitative comparison across tool calling, response quality, and format compliance. It is the evaluation gate referenced in the `hermes-finetune` design spec (§5) — no adapter is promoted without passing this benchmark.

Unlike TerminalBench2 (which tests general coding ability) or YC-Bench (which tests strategic coherence), this benchmark tests **personalized agent behavior** — the specific capabilities the fine-tune is supposed to improve.

---

## Architecture

The benchmark is a standalone harness around the production agent:

```
FinetuneBenchEnv
            ├── setup()           → load prompt bank + ground truth
            ├── get_next_item()   → return next test case
            ├── format_prompt()   → convert to user message
            ├── compute_reward()  → multi-axis scoring via ToolContext
            └── evaluate()        → aggregate metrics, produce comparison report
```

It runs via the standard CLI:

```bash
# Evaluate base model (before fine-tune)
python optional-skills/mlops/finetune/bench/finetune_bench_env.py evaluate \
    --config optional-skills/mlops/finetune/bench/default.yaml \
    --openai.model_name base-model

# Evaluate fine-tuned model (after fine-tune)  
python optional-skills/mlops/finetune/bench/finetune_bench_env.py evaluate \
    --config optional-skills/mlops/finetune/bench/default.yaml \
    --openai.model_name finetuned-model

# Or via the skill
/finetune eval --cluster c-a7f3e2 --version v2
```

---

## Test Case Taxonomy

Test cases are organized into three tiers. Each tier targets a different failure mode of fine-tuning.

### Tier 1: Tool Selection (automated scoring, ~100 cases)

Does the model pick the correct tool? This is binary, cheap to run, and catches the most common fine-tune regression: the model starts reaching for tools when it shouldn't, or stops using tools when it should.

**Categories:**

| Category | Count | What it tests | Example |
|---|---|---|---|
| Correct tool, simple | 25 | Basic tool routing | "List files in the current directory" → `terminal` with `ls` |
| Correct tool, ambiguous | 15 | Disambiguation, incl. recognizing missing capabilities | "Which version of tar is installed?" → check, don't guess; "What's the weather in Tokyo?" (no web access) → no tool, say so |
| No tool needed | 20 | Knowing when NOT to call a tool | "What is a monad?" → text response, no tool call |
| Multi-tool sequence | 15 | Chaining tools in the right order | "Find the largest file in /tmp and show its contents" → `terminal` × 2 |
| Tool with complex args | 15 | Argument construction quality | "Create a Python virtualenv named 'test' and install requests" → correct bash |
| Skill invocation | 10 | Recognizing when a skill should be loaded | "Help me fine-tune my model" → `/finetune` skill activation |

The bench is **offline and hermetic**: the only toolsets served are those in
the bench config (`terminal`, `file` by default). No case may expect a tool
outside that set — a case expecting e.g. `web_search` would be unwinnable and
would count a correct model answer as a hallucination. This invariant is
pinned by `tests/test_finetune_bench.py::TestPromptBank::test_no_case_expects_an_unserved_tool`.
A small minority of tier-1 cases deliberately state "you have no internet
access" and expect the model to abstain and say so.

**Ground truth format:**

```json
{
  "id": "ts-001",
  "tier": 1,
  "category": "correct_tool_simple",
  "prompt": "List all Python files in the current directory",
  "expected": {
    "tool_name": "terminal",
    "tool_args_pattern": ".*\\.py.*",
    "should_call_tool": true
  },
  "tags": ["terminal", "file-operations"]
}
```

**Scoring:** Exact match on tool name (1.0 or 0.0). Partial credit (0.5) if the right tool is called but with incorrect arguments. 0.0 if a tool is called when none was needed, or vice versa.

### Tier 2: Tool Execution Quality (semi-automated scoring, ~80 cases)

The model picked the right tool — but does it use it well? This tier checks argument correctness, error handling, and output interpretation.

**Categories:**

| Category | Count | What it tests | Example |
|---|---|---|---|
| Bash correctness | 20 | Valid, idiomatic shell commands | "Sum the sizes of all .log files" over seeded fixtures → exact total |
| Error recovery | 15 | Handling tool failures gracefully | Command fails by design → model diagnoses and explains (see `expect_failure` below) |
| Output interpretation | 16 | Reading tool output and drawing correct conclusions | Seeded `ps aux` snapshot → model names the right process |
| Docs lookup | 10 | Interrogating local docs/config/metadata (offline replacement for the old "web search quality" set) | "Which class does pathlib.Path inherit from?" → `pydoc`, answer `PurePath` |
| File operations | 10 | Read/write/edit file sequences | "Add a docstring to the main function in app.py" → verified by importing the module |
| Multi-turn coherence | 10 | Maintaining context across tool calls | 3-step task where each step depends on the previous result |

Every tier-2 case carries a **real, deterministic assertion**. Cases either
seed fixture files (`setup.files`) whose contents pin the correct answer
(often with values a model cannot guess from the prompt, e.g. sentinel
tokens or counterfactual data), assert patterns that are stable inside the
sandbox image (`python3 --version` → `Python 3\.`), or use
`functional_test` checks against artifacts the task creates. An
`output_match` verification with no `expected_value`/`expected_regex` is a
**case-authoring error**: the scorer logs it, scores the check as failed,
and reports the case in the `malformed_cases` metric — it can never be a
vacuous pass.

**Ground truth format:**

```json
{
  "id": "te-001",
  "tier": 2,
  "category": "bash_correctness",
  "prompt": "Count the number of lines in all .py files in the current directory",
  "setup": {
    "files": {
      "app.py": "def main():\n    pass\n",
      "utils.py": "import os\nimport sys\n\ndef helper():\n    return True\n"
    }
  },
  "expected": {
    "tool_name": "terminal",
    "output_contains": ["7", "lines"],
    "exit_code": 0
  },
  "verification": {
    "method": "output_match",
    "expected_value": "7"
  },
  "tags": ["terminal", "text-processing"]
}
```

**Scoring:** Uses `ToolContext` to verify actual outcomes in the sandbox. The reward function runs the model's command, checks exit code, and validates output against expected patterns. Scored 0.0–1.0 with partial credit for correct approach but wrong result.

#### `output_match` verification semantics

An `output_match` verification supports these fields:

| Field | Meaning |
|---|---|
| `expected_value` | Substring assertion. Short/numeric values (≤3 chars or pure numbers) are word-boundary matched against the final assistant answer and the *last successful* tool result only; longer values substring-match the final answer plus *successful* tool results (error output never satisfies a content assertion). |
| `expected_regex` | Regex alternative/complement to `expected_value` (e.g. `Python 3\.[0-9]+\.[0-9]+`, or a random-hex shape check). |
| `match_scope` | `transcript` (default) or `final_answer`. With `final_answer`, only the model's concluding message is searched — used for interpretation tasks where merely `cat`-ing a fixture file must not count as answering. |
| `expect_failure` | Error-recovery semantics; see below. |

At least one of `expected_value` / `expected_regex` is **required**. A case
with neither is malformed: logged, scored as failed, counted in
`malformed_cases`.

#### Error-recovery cases: `expect_failure`

For 15 error-recovery cases the *correct* trajectory includes a failing
command ("try to write to /proc/sys/…, explain why it fails"). Naive
exit-code scoring inverts the incentive: a perfect run would score worse
than padding with an irrelevant successful command. With
`expect_failure: true`, execution credit instead requires **both**:

1. at least one tool call in the transcript that FAILED (nonzero exit /
   error output), and
2. the final assistant answer acknowledging/explaining the failure —
   matched case-insensitively against the case's `expected_regex`
   (or `expected_value`).

A run that only executes successful commands scores zero execution credit
on these cases, and a failing command with no explanation also scores zero.
Note: the sandbox runs as root, so the failures these cases provoke are
chosen to be user-independent (read-only mounts, nonexistent
commands/paths/packages, corrupt inputs) rather than classic EACCES
scenarios.

### Tier 3: End-to-End Task Completion (judge-scored, ~40 cases)

Multi-turn scenarios that test the full agent loop. These are the most realistic but also the most expensive to run.

**Categories:**

| Category | Count | What it tests | Example |
|---|---|---|---|
| Project scaffolding | 8 | Creating structured output from requirements | "Create a Flask app with two endpoints and a SQLite backend" |
| Debug and fix | 8 | Diagnosing and repairing broken code | Pre-seeded buggy file, model must find and fix the issue |
| Research and summarize | 8 | Web search → synthesis → structured output | "Find the top 3 Python web frameworks by GitHub stars and compare them" |
| Refactor | 8 | Reading existing code and restructuring it | "Refactor this 200-line script into separate modules" |
| Workflow automation | 8 | Multi-tool orchestration | "Set up a git repo, create a branch, add files, and make a commit" |

**Ground truth format:**

```json
{
  "id": "e2e-001",
  "tier": 3,
  "category": "project_scaffolding",
  "prompt": "Create a Python CLI tool that takes a URL and outputs the word count of the page content. Use argparse for argument parsing.",
  "setup": {
    "working_dir": "/workspace/wordcount"
  },
  "verification": {
    "method": "functional_test",
    "test_commands": [
      "python wordcount.py --help",
      "python wordcount.py https://example.com"
    ],
    "checks": [
      {"type": "exit_code", "command_index": 0, "expected": 0},
      {"type": "output_regex", "command_index": 1, "pattern": "\\d+"},
      {"type": "file_exists", "path": "/workspace/wordcount/wordcount.py"}
    ]
  },
  "max_turns": 15,
  "tags": ["scaffolding", "cli", "web"]
}
```

**Scoring:** Functional verification via `ToolContext` — runs test commands in the same sandbox the model used. Binary pass/fail per check, composite score across checks. For subjective quality (code style, documentation), an optional judge model scores on a 1–5 scale.

---

## Scoring System

### Per-Case Scoring

Every test case produces a `CaseResult`:

```python
@dataclass
class CaseResult:
    case_id: str
    tier: int
    category: str
    tags: list[str]

    # Core metrics
    tool_selection_correct: bool       # Tier 1: right tool chosen
    tool_args_valid: bool              # Tier 2: arguments are correct
    task_completed: bool               # Tier 3: end state matches expected
    
    # Format health
    format_valid: bool                 # ChatML structure well-formed
    tool_call_parseable: bool          # Tool call JSON valid
    
    # Efficiency
    turns_used: int                    # How many LLM calls
    tool_errors: int                   # Errors during tool execution
    
    # Composite
    reward: float                      # 0.0–1.0 composite score
    
    # Diagnostics
    messages: list[dict]               # Full conversation trace
    reasoning: list[str]               # Extracted reasoning per turn
```

### Aggregate Metrics

The `evaluate()` method computes aggregate metrics that map directly to the promotion criteria in the finetune design spec:

| Metric | Computation | Promotion threshold |
|---|---|---|
| **Tool Selection Accuracy** | % of Tier 1 cases with correct tool | Must not regress > 3% vs. previous |
| **Tool Execution Success** | % of Tier 2 cases passing verification | Must not regress > 5% |
| **Task Completion Rate** | % of Tier 3 cases passing all checks | Must not regress > 5% |
| **Format Compliance** | % of all cases with valid ChatML + parseable tool calls | ≥ 95% |
| **No-Tool Accuracy** | % of "no tool needed" cases where model correctly abstained | Must not regress > 5% |
| **Efficiency** | Mean turns used normalized by task complexity | Informational (no gate) |
| **Error Rate** | Mean tool errors per case | Informational (no gate) |
| **Hallucination Rate** | % of cases where model called a tool that was never served | ≤ max(1%, baseline + 1%) — a single flaky case must not auto-FAIL the run |
| **Infra Error Rate** | % of cases lost to infrastructure (endpoint/Docker daemon failures) | > 5% invalidates the whole run (exit 3) |
| **Malformed Cases** | Count of case-authoring errors (empty assertions, unsafe working_dirs) | Informational; malformed checks always score as failed |

### Verdict semantics (fail closed)

The verdict gates only apply to metrics present in **both** candidate and
baseline (the intersection — same semantics as `eval.py`). A gate whose
metric is missing is skipped with a printed warning; it never auto-passes.
If *no* gate metric is comparable at all, the verdict is FAIL: an empty
intersection means the baseline says nothing about the run.

### Results JSON

Each run writes `~/.hermes/finetune/bench/results/bench_<ts>.json` with the
stable keys `metrics`, `cases`, and `timestamp`. When a baseline comparison
was performed, the additive key `verdict` records the per-gate booleans and
`overall`. `metrics` additionally carries `scored_cases`, `infra_errors`,
`infra_error_rate`, and `malformed_cases` alongside the headline rates.

### Sandbox enforcement and exit codes

The Docker backend is required and **verified**, not merely defaulted:

- `setup()` preflights the daemon with `docker info` and hard-exits when it
  is unusable — with the daemon down, every case would otherwise fail as a
  bogus *quality* zero and the run would silently poison the baseline.
- Any non-docker terminal backend executes agent-generated commands
  directly on the host. The bench refuses to start unless the operator
  opts in explicitly via `FINETUNE_BENCH_ALLOW_UNSANDBOXED=1` (or
  `allow_unsandboxed: true` in the config), and prints a loud warning when
  they do.
- Tool results showing the daemon died mid-run ("Cannot connect to the
  Docker daemon…") are classified as **infrastructure errors** for that
  case — excluded from quality denominators and counted toward run
  invalidation — never as model-quality failures.
- Case `working_dir`s are wiped between runs; a working dir that does not
  resolve under the bench scratch root (`/tmp/finetune-bench`) is refused
  and the case is skipped with a loud warning, so a custom case with
  `working_dir: ~/projects` can never delete user data.

Process exit codes:

| Exit code | Meaning |
|---|---|
| 0 | Run completed; verdict PASS or no baseline configured |
| 1 | Run completed but the baseline-comparison verdict is FAIL |
| 2 | Configured LLM endpoint unreachable (preflight) |
| 3 | Run invalid: infra error rate above 5% — metrics untrustworthy |
| 4 | Sandbox unavailable: Docker daemon down/missing, or unsandboxed backend refused |

### Comparison Report

When two eval runs exist (baseline vs. candidate), the benchmark produces a comparison:

```
╔══════════════════════════════════════════════════════════════╗
║            FINETUNE BENCH — Comparison Report               ║
╠══════════════════════════════════════════════════════════════╣
║ Baseline:  qwen3-8b-q5km          (2026-04-06 12:00)       ║
║ Candidate: qwen3-8b-q5km+c-a7f3e2-v2  (2026-04-06 14:30)  ║
╠═══════════════════════╦═══════════╦═══════════╦═════════════╣
║ Metric                ║ Baseline  ║ Candidate ║ Delta       ║
╠═══════════════════════╬═══════════╬═══════════╬═════════════╣
║ Tool Selection Acc.   ║   82.0%   ║   86.0%   ║  +4.0% ✓   ║
║ Tool Execution Succ.  ║   71.3%   ║   73.8%   ║  +2.5% ✓   ║
║ Task Completion Rate  ║   62.5%   ║   65.0%   ║  +2.5% ✓   ║
║ Format Compliance     ║   98.0%   ║   97.5%   ║  -0.5% ✓   ║
║ No-Tool Accuracy      ║   90.0%   ║   85.0%   ║  -5.0% ⚠   ║
║ Hallucination Rate    ║    0.0%   ║    0.0%   ║   0.0% ✓   ║
║ Mean Turns/Task       ║    4.2    ║    3.8    ║  -0.4  ✓   ║
║ Mean Errors/Task      ║    0.3    ║    0.2    ║  -0.1  ✓   ║
╠═══════════════════════╩═══════════╩═══════════╩═════════════╣
║ VERDICT: PASS (7/8 metrics pass, 1 warning)                ║
║ WARNING: No-Tool Accuracy regressed 5.0% — at threshold    ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Canary Set

A subset of test cases (20–30) is designated as the **canary set**. These are frozen across all adapter versions and never change. They test baseline capabilities that should never degrade regardless of what the fine-tune targets:

- Basic arithmetic and reasoning (no tools)
- Simple file operations
- Standard bash commands
- Correct refusal of impossible requests
- Format compliance on edge cases (empty responses, very long outputs, Unicode)

Canary regression is an automatic promotion blocker, even if all other metrics pass. This catches catastrophic forgetting.

---

## Custom Test Cases

Users can add their own test cases to the prompt bank. This is essential — the generic cases test general agent competence, but the user-specific cases test whether the fine-tune improved the *specific things they care about*.

Custom cases are added to `~/.hermes/finetune/bench/custom/`:

```yaml
# ~/.hermes/finetune/bench/custom/my-tests.yaml
cases:
  - id: custom-001
    tier: 2
    category: "rust-toolchain"
    prompt: "Create a new Rust project with tokio and serde as dependencies"
    setup:
      # Must live under /tmp/finetune-bench — the bench wipes the case
      # working dir between runs and refuses (skips the case) any path
      # outside its scratch root.
      working_dir: "/tmp/finetune-bench/rust-test"
    verification:
      method: functional_test
      test_commands:
        - "cat Cargo.toml"
        - "cargo check 2>&1"
      checks:
        - type: output_contains
          command_index: 0
          expected: ["tokio", "serde"]
        - type: exit_code
          command_index: 1
          expected: 0
    tags: ["rust", "project-setup"]
```

Custom cases are merged into the prompt bank at runtime and appear in the evaluation report alongside the standard cases, tagged distinctly so the user can see performance on their specific workloads.

---

## Implementation: Environment Class

```python
# optional-skills/mlops/finetune/bench/finetune_bench_env.py

"""
Fine-tune evaluation benchmark.

Runs a structured prompt bank against the agent loop, scoring
tool selection, execution quality, and end-to-end task completion.

Usage:
    python finetune_bench_env.py evaluate --config default.yaml
    python finetune_bench_env.py process --env.data_path_to_save_groups results.jsonl
"""

import json
import re
import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

from environments.hermes_base_env import HermesAgentBaseEnv, HermesAgentEnvConfig


@dataclass
class CaseResult:
    case_id: str
    tier: int
    category: str
    tags: list[str]
    tool_selection_correct: bool = False
    tool_args_valid: bool = False
    task_completed: bool = False
    format_valid: bool = True
    tool_call_parseable: bool = True
    turns_used: int = 0
    tool_errors: int = 0
    reward: float = 0.0
    is_canary: bool = False


class FinetuneBenchConfig(HermesAgentEnvConfig):
    prompt_bank_path: str = "prompt_bank.yaml"  # relative to the bench dir
    custom_cases_dir: str = "~/.hermes/finetune/bench/custom"
    baseline_results_path: Optional[str] = None  # path to previous eval for comparison
    regression_threshold_tool_selection: float = 0.03
    regression_threshold_execution: float = 0.05
    regression_threshold_completion: float = 0.05
    format_compliance_minimum: float = 0.95


class FinetuneBenchEnv(HermesAgentBaseEnv):
    name = "finetune-bench"
    env_config_cls = FinetuneBenchConfig

    @classmethod
    def config_init(cls):
        env_config = FinetuneBenchConfig(
            enabled_toolsets=["terminal", "file", "web"],
            terminal_backend="local",
            max_agent_turns=15,
            eval_handling="STOP_TRAIN",
            steps_per_eval=1,
            total_steps=1,
        )
        server_configs = [...]  # configured via YAML
        return env_config, server_configs

    async def setup(self):
        """Load prompt bank and custom cases."""
        bank_path = Path(self.env_config.prompt_bank_path)
        with open(bank_path) as f:
            self.prompt_bank = yaml.safe_load(f)["cases"]

        # Merge custom cases
        custom_dir = Path(self.env_config.custom_cases_dir).expanduser()
        if custom_dir.exists():
            for custom_file in custom_dir.glob("*.yaml"):
                with open(custom_file) as f:
                    custom = yaml.safe_load(f)
                    self.prompt_bank.extend(custom.get("cases", []))

        # Load baseline for comparison if available
        self.baseline = None
        if self.env_config.baseline_results_path:
            bp = Path(self.env_config.baseline_results_path)
            if bp.exists():
                with open(bp) as f:
                    self.baseline = json.load(f)

        self.results: list[CaseResult] = []
        self.iter = 0

    async def get_next_item(self):
        if self.iter >= len(self.prompt_bank):
            return None
        item = self.prompt_bank[self.iter]
        self.iter += 1
        return item

    def format_prompt(self, item):
        return item["prompt"]

    async def compute_reward(self, item, result, ctx):
        """Score a single test case using the agent result and ToolContext."""
        case = CaseResult(
            case_id=item["id"],
            tier=item["tier"],
            category=item["category"],
            tags=item.get("tags", []),
            is_canary=item.get("canary", False),
            turns_used=result.turns_used,
            tool_errors=len(result.tool_errors),
        )

        messages = result.messages

        # --- Format compliance ---
        case.format_valid = self._check_format(messages)
        case.tool_call_parseable = self._check_tool_parse(messages)

        # --- Tier 1: Tool selection ---
        if item["tier"] >= 1:
            expected = item.get("expected", {})
            actual_tools = self._extract_tool_calls(messages)

            if expected.get("should_call_tool") is False:
                case.tool_selection_correct = len(actual_tools) == 0
            elif expected.get("tool_name"):
                case.tool_selection_correct = any(
                    t["name"] == expected["tool_name"] for t in actual_tools
                )

        # --- Tier 2: Tool execution ---
        if item["tier"] >= 2 and item.get("verification"):
            v = item["verification"]
            if v["method"] == "output_match":
                case.tool_args_valid = await self._verify_output(
                    ctx, messages, v
                )
            elif v["method"] == "functional_test":
                case.task_completed = await self._verify_functional(
                    ctx, item, v
                )
                case.tool_args_valid = case.task_completed

        # --- Tier 3: End-to-end ---
        if item["tier"] == 3 and item.get("verification"):
            v = item["verification"]
            case.task_completed = await self._verify_functional(ctx, item, v)

        # --- Composite reward ---
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
        else:  # tier 3
            selection = 0.2 if case.tool_selection_correct else 0.0
            execution = 0.3 if case.tool_args_valid else 0.0
            completion = 0.5 if case.task_completed else 0.0
            return selection + execution + completion

    def _check_format(self, messages: list[dict]) -> bool:
        """Verify all assistant messages have valid ChatML structure."""
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if content is None and not msg.get("tool_calls"):
                    return False
        return True

    def _check_tool_parse(self, messages: list[dict]) -> bool:
        """Verify all tool calls are parseable JSON."""
        for msg in messages:
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    try:
                        if isinstance(tc.get("function", {}).get("arguments"), str):
                            json.loads(tc["function"]["arguments"])
                    except (json.JSONDecodeError, TypeError):
                        return False
        return True

    def _extract_tool_calls(self, messages: list[dict]) -> list[dict]:
        """Extract all tool calls from assistant messages."""
        tools = []
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    tools.append({
                        "name": tc.get("function", {}).get("name"),
                        "arguments": tc.get("function", {}).get("arguments"),
                    })
        return tools

    async def _verify_output(self, ctx, messages, verification) -> bool:
        """Check if tool output matches expected value."""
        for msg in messages:
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                expected = verification.get("expected_value", "")
                if expected in content:
                    return True
        return False

    async def _verify_functional(self, ctx, item, verification) -> bool:
        """Run test commands in the sandbox and check results."""
        checks = verification.get("checks", [])
        commands = verification.get("test_commands", [])
        
        # Run all test commands
        outputs = []
        for cmd in commands:
            result = ctx.terminal(cmd, timeout=30)
            outputs.append(result)

        # Evaluate checks
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
                result = ctx.terminal(f"test -f {check['path']} && echo EXISTS")
                if "EXISTS" in result.get("output", ""):
                    passed += 1

        return passed == len(checks) if checks else False

    async def evaluate(self, *args, **kwargs):
        """Aggregate results, compare to baseline, produce report."""
        metrics = self._aggregate_metrics()
        
        # Save results
        results_path = Path(f"~/.hermes/finetune/bench/results/{self._run_id()}.json").expanduser()
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump({
                "metrics": metrics,
                "cases": [asdict(r) for r in self.results],
            }, f, indent=2)

        # Compare to baseline
        if self.baseline:
            comparison = self._compare(metrics, self.baseline["metrics"])
            verdict = self._verdict(comparison)
            self._print_report(metrics, self.baseline["metrics"], comparison, verdict)
        else:
            self._print_report(metrics)

    def _aggregate_metrics(self) -> dict:
        tier1 = [r for r in self.results if r.tier == 1]
        tier2 = [r for r in self.results if r.tier == 2]
        tier3 = [r for r in self.results if r.tier == 3]
        canary = [r for r in self.results if r.is_canary]
        no_tool = [r for r in self.results 
                   if r.tier == 1 and r.category == "no_tool_needed"]
        all_cases = self.results

        return {
            "tool_selection_accuracy": (
                sum(1 for r in tier1 if r.tool_selection_correct) / len(tier1)
                if tier1 else 0.0
            ),
            "tool_execution_success": (
                sum(1 for r in tier2 if r.tool_args_valid) / len(tier2)
                if tier2 else 0.0
            ),
            "task_completion_rate": (
                sum(1 for r in tier3 if r.task_completed) / len(tier3)
                if tier3 else 0.0
            ),
            "format_compliance": (
                sum(1 for r in all_cases if r.format_valid and r.tool_call_parseable)
                / len(all_cases) if all_cases else 0.0
            ),
            "no_tool_accuracy": (
                sum(1 for r in no_tool if r.tool_selection_correct) / len(no_tool)
                if no_tool else 0.0
            ),
            "hallucination_rate": (
                sum(1 for r in all_cases if not r.tool_call_parseable)
                / len(all_cases) if all_cases else 0.0
            ),
            "mean_turns": (
                sum(r.turns_used for r in all_cases) / len(all_cases)
                if all_cases else 0.0
            ),
            "mean_errors": (
                sum(r.tool_errors for r in all_cases) / len(all_cases)
                if all_cases else 0.0
            ),
            "canary_pass_rate": (
                sum(1 for r in canary if r.reward > 0.5) / len(canary)
                if canary else 0.0
            ),
            "total_cases": len(all_cases),
        }

    def _compare(self, current: dict, baseline: dict) -> dict:
        comparison = {}
        for key in current:
            if key in baseline and isinstance(current[key], (int, float)):
                comparison[key] = {
                    "baseline": baseline[key],
                    "candidate": current[key],
                    "delta": current[key] - baseline[key],
                }
        return comparison

    def _verdict(self, comparison: dict) -> dict:
        cfg = self.env_config
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

        checks["overall"] = all(checks.values())
        return checks


if __name__ == "__main__":
    FinetuneBenchEnv.cli()
```

---

## Prompt Bank Structure

```yaml
# optional-skills/mlops/finetune/bench/prompt_bank.yaml

cases:
  # ============================================================
  # TIER 1: Tool Selection
  # ============================================================

  # -- Correct tool, simple --
  - id: ts-001
    tier: 1
    category: correct_tool_simple
    prompt: "List all files in the current directory"
    expected:
      tool_name: terminal
      should_call_tool: true
    tags: [terminal, basics]

  - id: ts-002
    tier: 1
    category: correct_tool_simple
    prompt: "What is the current date and time?"
    expected:
      tool_name: terminal
      should_call_tool: true
    tags: [terminal, basics]

  # -- No tool needed --
  - id: ts-020
    tier: 1
    category: no_tool_needed
    prompt: "Explain the difference between a stack and a queue"
    expected:
      should_call_tool: false
    tags: [knowledge, no-tool]

  - id: ts-021
    tier: 1
    category: no_tool_needed
    prompt: "What does the HTTP 404 status code mean?"
    expected:
      should_call_tool: false
    tags: [knowledge, no-tool]
    canary: true

  # -- Ambiguous --
  - id: ts-040
    tier: 1
    category: correct_tool_ambiguous
    prompt: "What version of Python is installed?"
    expected:
      tool_name: terminal
      tool_args_pattern: "python.*--version"
      should_call_tool: true
    tags: [terminal, version-check]

  # ============================================================
  # TIER 2: Tool Execution Quality
  # ============================================================

  - id: te-001
    tier: 2
    category: bash_correctness
    prompt: "Count the number of lines in all .py files in the current directory"
    setup:
      files:
        app.py: "def main():\n    pass\n"
        utils.py: "import os\nimport sys\n\ndef helper():\n    return True\n"
    expected:
      tool_name: terminal
      exit_code: 0
    verification:
      method: output_match
      expected_value: "7"
    tags: [terminal, text-processing]

  - id: te-010
    tier: 2
    category: error_recovery
    prompt: "Install the Python package 'nonexistent-pkg-xyz' and if it fails, tell me what happened"
    expected:
      tool_name: terminal
      should_call_tool: true
    verification:
      method: output_match
      # The correct trajectory FAILS: credit requires a failed tool call
      # plus a final answer explaining it (matched case-insensitively).
      expect_failure: true
      expected_regex: '(no matching|not (be )?found|could ?n[o'']t|unable|fail)'
    tags: [terminal, error-handling]

  # ============================================================
  # TIER 3: End-to-End
  # ============================================================

  - id: e2e-001
    tier: 3
    category: project_scaffolding
    prompt: "Create a Python script called greet.py that takes a name as a command-line argument and prints 'Hello, {name}!'"
    setup:
      working_dir: /workspace/greet-test
    verification:
      method: functional_test
      test_commands:
        - "python greet.py World"
        - "python greet.py Alice"
      checks:
        - type: output_contains
          command_index: 0
          expected: ["Hello, World!"]
        - type: output_contains
          command_index: 1
          expected: ["Hello, Alice!"]
        - type: file_exists
          path: /workspace/greet-test/greet.py
    max_turns: 10
    tags: [scaffolding, cli, python]

  - id: e2e-010
    tier: 3
    category: debug_and_fix
    prompt: "The file buggy.py has a bug. Find and fix it."
    setup:
      working_dir: /workspace/debug-test
      files:
        buggy.py: |
          def fibonacci(n):
              if n <= 1:
                  return n
              return fibonacci(n - 1) + fibonacci(n - 3)  # bug: should be n-2
          
          if __name__ == "__main__":
              for i in range(10):
                  print(f"fib({i}) = {fibonacci(i)}")
    verification:
      method: functional_test
      test_commands:
        - "python buggy.py"
      checks:
        - type: output_contains
          command_index: 0
          expected: ["fib(6) = 8", "fib(7) = 13"]
    max_turns: 10
    tags: [debug, python]

  # ... (full bank would contain ~220 cases)
```

---

## Integration with the Fine-Tune Pipeline

The benchmark plugs into the pipeline at two points:

**Pre-training baseline:** Before any adapter training, run the benchmark against the base model (or current active adapter). This result is saved as the baseline.

**Post-training gate:** After training completes, run the benchmark against the candidate adapter. Compare to baseline. If the verdict is PASS, the adapter is eligible for promotion. If FAIL, it's rejected and the failure report is logged.

The `/finetune eval` command orchestrates both runs automatically:

```
/finetune eval --cluster c-a7f3e2 --version v2

# This:
# 1. Loads the baseline results for cluster c-a7f3e2 (or runs baseline if none exists)
# 2. Runs finetune-bench against the v2 adapter
# 3. Compares and produces the verdict
# 4. Stores results at ~/.hermes/finetune/bench/results/
```

Results are also available via `/finetune status`, which shows the last eval scores per cluster.

---

## Key Indicators of Improvement

To summarize: when reviewing eval results, these are the signals that tell you the fine-tune is working:

**Clear wins:** Tool selection accuracy goes up (the model better understands when to use tools and which ones). No-tool accuracy stays stable or improves (the model hasn't learned to over-use tools). Task completion rate increases on Tier 3 cases that match the user's domain.

**Clear regressions:** Format compliance drops (the fine-tune corrupted the model's output structure — this is fatal). Hallucination rate rises past the ~1% flake tolerance (the model is inventing tools). No-tool accuracy drops significantly (the model is over-indexing on tool use from training data). Canary set regresses (catastrophic forgetting).

**Ambiguous signals:** Small fluctuations (±1–2%) in any metric are noise, not signal. Efficiency improvements (fewer turns) are nice but not sufficient alone. Improved performance on custom cases but regression on standard cases suggests overfitting.

The single most important metric is **format compliance**. If that drops, nothing else matters — the agent loop will break. Everything else is relative improvement measured against a stable baseline.
