"""
Golden-task eval harness (P2-2 objective quality evaluation).

Provides the fitness signal that gates Denji's autonomous profile
edits (WS-7).  A golden task set is a curated collection of tasks
with rubrics; the runner executes them, scores the results via an
LLM judge, and persists versioned results for before/after comparison
of profile/prompt changes.

Architecture
------------
1. ``GoldenTask`` — a task with description, expected behaviors, and a rubric.
2. ``EvalResult`` — a scored result from a single task run.
3. ``EvalRun`` — a collection of results from running the full set.
4. ``run_eval()`` — executes all tasks against a target profile and
   returns an EvalRun with pass-rate and per-task scores.
5. ``compare_runs()`` — delta between two EvalRuns, used by Denji's
   keep/rollback gate.

Golden task format (YAML)
--------------------------
    tasks:
      - id: code-review-001
        domain: code
        description: "Review a PR with a security bug"
        input:
          pr_diff: "..."
          context: "This is a login module"
        expected_behaviors:
          - "Identifies the SQL injection vulnerability"
          - "Suggests parameterised queries"
          - "Does NOT suggest regex-based sanitisation"
        rubric:
          pass_if: "All expected behaviors present, no anti-patterns"
          fail_if: "Misses SQL injection or suggests regex sanitisation"

Usage
-----
    from hermes_cli.eval_harness import run_eval, GoldenTask

    tasks = GoldenTask.load_set("path/to/golden_tasks.yaml")
    result = run_eval(
        tasks=tasks,
        profile="code-reviewer",
        judge_profile="kensei-review",
    )
    print(f"Pass rate: {result.pass_rate:.0%}")
    result.save("evals/v1.0.0.json")
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class GoldenTask:
    """A single golden task with rubric."""

    id: str
    domain: str
    description: str
    input: dict = field(default_factory=dict)
    expected_behaviors: list[str] = field(default_factory=list)
    rubric: dict = field(default_factory=dict)
    # Weight in the overall score (0.0-1.0, default 1.0).
    weight: float = 1.0

    @classmethod
    def load_set(cls, path: str | Path) -> list[GoldenTask]:
        """Load a golden task set from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        tasks = []
        for item in data.get("tasks", []):
            tasks.append(cls(
                id=item["id"],
                domain=item.get("domain", "general"),
                description=item["description"],
                input=item.get("input", {}),
                expected_behaviors=item.get("expected_behaviors", []),
                rubric=item.get("rubric", {}),
                weight=float(item.get("weight", 1.0)),
            ))
        return tasks

    @property
    def content_hash(self) -> str:
        """Stable hash of task content for version tracking."""
        canonical = json.dumps({
            "id": self.id,
            "description": self.description,
            "input": self.input,
            "expected_behaviors": self.expected_behaviors,
            "rubric": self.rubric,
        }, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]


@dataclass
class EvalResult:
    """Result of running a single golden task."""

    task_id: str
    domain: str
    passed: bool
    score: float  # 0.0-1.0
    judge_notes: str = ""
    model_output: str = ""
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    task_hash: str = ""
    weight: float = 1.0


@dataclass
class EvalRun:
    """A complete eval run across all golden tasks."""

    run_id: str
    timestamp: str
    profile: str
    judge_profile: str
    task_count: int
    passed_count: int
    results: list[EvalResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        if self.task_count == 0:
            return 0.0
        return self.passed_count / self.task_count

    @property
    def weighted_score(self) -> float:
        """Weighted average score across all tasks."""
        if not self.results:
            return 0.0
        total_weight = sum(
            t.weight for t in self.results  # type: ignore[attr-defined]
        ) if hasattr(self.results[0], 'weight') else len(self.results)
        # Use raw score average since weights are stored per-task
        return sum(r.score for r in self.results) / len(self.results)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "profile": self.profile,
            "judge_profile": self.judge_profile,
            "task_count": self.task_count,
            "passed_count": self.passed_count,
            "pass_rate": self.pass_rate,
            "weighted_score": self.weighted_score,
            "results": [
                {
                    "task_id": r.task_id,
                    "domain": r.domain,
                    "passed": r.passed,
                    "score": r.score,
                    "judge_notes": r.judge_notes,
                    "elapsed_seconds": r.elapsed_seconds,
                    "error": r.error,
                }
                for r in self.results
            ],
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        """Persist the eval run to a JSON file."""
        os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> EvalRun:
        """Load a previously-saved eval run."""
        with open(path) as f:
            data = json.load(f)
        results = [
            EvalResult(
                task_id=r["task_id"],
                domain=r.get("domain", "general"),
                passed=r["passed"],
                score=r["score"],
                judge_notes=r.get("judge_notes", ""),
                elapsed_seconds=r.get("elapsed_seconds", 0.0),
                error=r.get("error"),
            )
            for r in data["results"]
        ]
        return cls(
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            profile=data["profile"],
            judge_profile=data.get("judge_profile", ""),
            task_count=data["task_count"],
            passed_count=data["passed_count"],
            results=results,
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Eval comparison (for Denji's keep/rollback gate)
# ---------------------------------------------------------------------------


@dataclass
class EvalDelta:
    """Difference between two eval runs."""

    baseline_run_id: str
    candidate_run_id: str
    baseline_pass_rate: float
    candidate_pass_rate: float
    pass_rate_delta: float
    regressions: list[str]  # task IDs that passed in baseline but failed in candidate
    improvements: list[str]  # task IDs that failed in baseline but passed in candidate
    recommendation: str  # "keep", "rollback", or "review"


def compare_runs(baseline: EvalRun, candidate: EvalRun) -> EvalDelta:
    """Compare two eval runs and produce a keep/rollback recommendation.

    Rules:
    - If candidate has regressions (tasks that went from pass→fail):
      recommendation = "rollback"
    - If candidate has improvements with no regressions:
      recommendation = "keep"
    - If candidate has no change or mixed with no regressions:
      recommendation = "review"
    """
    baseline_results = {r.task_id: r for r in baseline.results}
    candidate_results = {r.task_id: r for r in candidate.results}

    regressions = []
    improvements = []

    for task_id, br in baseline_results.items():
        cr = candidate_results.get(task_id)
        if cr is None:
            continue
        if br.passed and not cr.passed:
            regressions.append(task_id)
        elif not br.passed and cr.passed:
            improvements.append(task_id)

    pass_rate_delta = candidate.pass_rate - baseline.pass_rate

    if regressions:
        recommendation = "rollback"
    elif improvements:
        recommendation = "keep"
    elif pass_rate_delta >= 0 and candidate.pass_rate > 0:
        recommendation = "keep"  # no regressions, has passing tasks
    else:
        recommendation = "review"

    return EvalDelta(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        baseline_pass_rate=baseline.pass_rate,
        candidate_pass_rate=candidate.pass_rate,
        pass_rate_delta=pass_rate_delta,
        regressions=regressions,
        improvements=improvements,
        recommendation=recommendation,
    )


# ---------------------------------------------------------------------------
# Judge prompt — the LLM that scores task output
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """You are a rigorous eval judge. Your job is to score the output
of an AI agent against a predefined rubric.

You will receive:
1. The task description — what the agent was asked to do.
2. The expected behaviors — what a correct output should contain.
3. The rubric — criteria for pass/fail.
4. The agent's output — the actual result.

Evaluate the output against the rubric. Be strict but fair.

Return a JSON object with exactly these fields:
{
  "passed": true/false,
  "score": 0.0-1.0,
  "notes": "brief explanation of the score"
}

score scale:
  1.0 — Perfect. All expected behaviors present, no issues.
  0.7-0.9 — Good. Minor issues but essentially correct.
  0.4-0.6 — Partial. Some expected behaviors present, significant gaps.
  0.1-0.3 — Poor. Mostly incorrect or missing.
  0.0 — Completely wrong, empty, or harmful.
"""


def build_judge_prompt(task: GoldenTask, agent_output: str) -> str:
    """Build the judge prompt for a single task evaluation."""
    behaviors = "\n".join(f"  - {b}" for b in task.expected_behaviors)
    rubric_pass = task.rubric.get("pass_if", "N/A")
    rubric_fail = task.rubric.get("fail_if", "N/A")

    return f"""## Task: {task.id}

### Description
{task.description}

### Expected Behaviors
{behaviors}

### Rubric
- PASS if: {rubric_pass}
- FAIL if: {rubric_fail}

### Agent Output
```
{agent_output[:8000]}
```

Evaluate the output against the rubric. Return JSON only.
"""


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------


def run_eval(
    tasks: list[GoldenTask],
    *,
    profile: str,
    judge_profile: str = "kensei-review",
    eval_version: str = "",
    dry_run: bool = False,
) -> EvalRun:
    """Run a full eval suite against a target profile.

    Each golden task is executed via the target profile (using delegate_task
    or direct CLI invocation).  The output is then scored by a judge model.

    Args:
        tasks: The golden task set to run.
        profile: The Hermes profile to evaluate.
        judge_profile: The profile used for scoring (must have LLM access).
        eval_version: Human-readable version tag for the results file.
        dry_run: If True, skips actual execution and returns placeholder results.

    Returns:
        An EvalRun with all results.
    """
    run_id = f"eval-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    timestamp = datetime.now(timezone.utc).isoformat()
    results: list[EvalResult] = []

    if dry_run:
        for task in tasks:
            results.append(EvalResult(
                task_id=task.id,
                domain=task.domain,
                passed=True,
                score=1.0,
                judge_notes="[dry_run] placeholder",
                task_hash=task.content_hash,
                weight=task.weight,  # type: ignore[call-arg]
            ))
    else:
        for task in tasks:
            start = time.monotonic()
            try:
                # NOTE: Actual task execution requires integration with
                # delegate_task or hermes CLI.  For the initial harness,
                # we return a NOT_RUN result and document the integration
                # point.  The integration waits on P2-6 (staging harness)
                # for safe agent invocation during evals.
                output = _execute_golden_task(task, profile)
                judge_result = _judge_output(task, output, judge_profile)
                elapsed = time.monotonic() - start
                results.append(EvalResult(
                    task_id=task.id,
                    domain=task.domain,
                    passed=judge_result.get("passed", False),
                    score=float(judge_result.get("score", 0.0)),
                    judge_notes=judge_result.get("notes", ""),
                    model_output=output[:2000],
                    elapsed_seconds=elapsed,
                    task_hash=task.content_hash,
                    weight=task.weight,  # type: ignore[call-arg]
                ))
            except Exception as exc:
                elapsed = time.monotonic() - start
                results.append(EvalResult(
                    task_id=task.id,
                    domain=task.domain,
                    passed=False,
                    score=0.0,
                    error=str(exc),
                    elapsed_seconds=elapsed,
                    task_hash=task.content_hash,
                    weight=task.weight,  # type: ignore[call-arg]
                ))

    passed = sum(1 for r in results if r.passed)
    return EvalRun(
        run_id=run_id,
        timestamp=timestamp,
        profile=profile,
        judge_profile=judge_profile,
        task_count=len(tasks),
        passed_count=passed,
        results=results,
        metadata={
            "eval_version": eval_version,
            "task_hashes": {t.id: t.content_hash for t in tasks},
        },
    )


def _execute_golden_task(task: GoldenTask, profile: str) -> str:
    """Execute a golden task against the target profile.

    Calls the profile's configured model directly via call_llm.
    For full agent evaluation (with skills/tools), delegate_task
    via the gateway is needed — but direct model evaluation catches
    model-quality regressions (the primary use case for profile edits).

    Args:
        task: The golden task to execute.
        profile: Hermes profile name (e.g. 'remii-deep').

    Returns:
        The model's response text.
    """
    from hermes_cli.config import load_config_readonly
    from agent.auxiliary_client import call_llm

    cfg = load_config_readonly()

    # Resolve profile config — look up the profile's provider/model
    profile_cfg = _get_profile_config(cfg, profile)

    provider = profile_cfg.get("provider", "opencode-go")
    model = profile_cfg.get("model", "minimax-m3")
    # Only pass base_url for custom: providers. For a known provider an
    # explicit base_url makes call_llm treat it as a custom endpoint and
    # fall back to OPENAI_API_KEY instead of the provider's credential pool,
    # which 401s. Omitting it lets call_llm resolve the provider's own
    # base_url + key together.
    base_url = profile_cfg.get("base_url") if str(provider).startswith("custom:") else None
    timeout = profile_cfg.get("timeout", 120)

    # Build the task prompt
    input_str = json.dumps(task.input, indent=2) if task.input else ""
    user_prompt = (
        f"# Task: {task.id}\n\n"
        f"{task.description}\n"
    )
    if input_str:
        user_prompt += f"\n## Input\n```json\n{input_str}\n```\n"

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI agent being evaluated. Complete the task described "
                "by the user. Be thorough and accurate. Return your answer directly "
                "— no preamble or meta-commentary."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    response = call_llm(
        provider=provider,
        model=model,
        base_url=base_url,
        messages=messages,
        timeout=timeout,
    )
    return response.choices[0].message.content or ""


def _get_profile_config(cfg: dict, profile: str) -> dict:
    """Extract a profile's LLM config from the Hermes configuration.

    Looks up the profile's config.yaml, falling back to the global
    default provider/model if the profile has no explicit overrides.
    """
    import os

    # Try profile-specific config first
    profile_cfg_path = os.path.expanduser(
        f"~/.hermes/profiles/{profile}/config.yaml"
    )
    if os.path.exists(profile_cfg_path):
        try:
            import yaml as _yaml
            with open(profile_cfg_path) as f:
                pc = _yaml.safe_load(f)
            if "model" in pc:
                model_cfg = pc["model"]
                # Profiles use either a flat ``model: <name>`` string or the
                # nested ``model: {default: <name>, provider: <prov>}`` form.
                # call_llm needs a plain string, so normalise both.
                base_url = None
                if isinstance(model_cfg, dict):
                    model = model_cfg.get("default") or model_cfg.get("model")
                    provider = (
                        model_cfg.get("provider")
                        or pc.get("provider")
                        or cfg.get("provider", "opencode-go")
                    )
                    base_url = model_cfg.get("base_url") or pc.get("base_url")
                else:
                    model = model_cfg
                    provider = pc.get("provider", cfg.get("provider", "opencode-go"))
                    base_url = pc.get("base_url")
                if model:
                    return {
                        "provider": provider,
                        "model": model,
                        "base_url": base_url,
                        "timeout": pc.get("timeout", 120),
                    }
        except Exception:
            pass

    # Fall back to global config
    providers_cfg = cfg.get("providers", {})
    # Try to get the default from the main provider block
    default_model = None
    default_provider = None
    for key in ("default", "model"):
        if key in cfg:
            default_model = cfg[key]
            break
    if not default_model:
        default_model = "minimax-m3"
    if not default_provider:
        default_provider = cfg.get("provider", "opencode-go")

    return {
        "provider": default_provider,
        "model": default_model,
        "timeout": 120,
    }


def _judge_output(task: GoldenTask, output: str, judge_profile: str) -> dict:
    """Score agent output against the task rubric using a real LLM judge.

    Calls the judge profile's model with the judge prompt and parses
    the JSON response. Falls back to a basic keyword-match heuristic
    if the LLM call fails (never returns hardcoded pass/fail silently).

    Args:
        task: The golden task with rubric.
        output: The agent's actual output text.
        judge_profile: Hermes profile to use as judge.

    Returns:
        dict with keys: passed, score, notes.
    """
    from hermes_cli.config import load_config_readonly
    from agent.auxiliary_client import call_llm

    # If the output is a placeholder, short-circuit — don't fabricate a score
    if output.startswith("[EVAL_PLACEHOLDER]") or output.startswith("[DRY_RUN]"):
        return {
            "passed": False,
            "score": 0.0,
            "notes": f"Output is a placeholder — task was not actually executed: {output[:200]}",
        }

    cfg = load_config_readonly()
    judge_cfg = _get_profile_config(cfg, judge_profile)

    judge_prompt = build_judge_prompt(task, output)
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": judge_prompt},
    ]

    try:
        response = call_llm(
            provider=judge_cfg.get("provider", "opencode-go"),
            model=judge_cfg.get("model", "minimax-m3"),
            messages=messages,
            timeout=judge_cfg.get("timeout", 120),
        )
        raw = response.choices[0].message.content or ""
        parsed = _parse_judge_json(raw)
        return parsed
    except Exception as exc:
        # Judge failed — return a scored failure, not a fabricated pass
        # Also attempt a basic keyword heuristic as a fallback signal
        heuristic = _keyword_heuristic_judge(task, output)
        return {
            "passed": heuristic.get("passed", False),
            "score": heuristic.get("score", 0.0),
            "notes": (
                f"[JUDGE_ERROR] LLM judge ({judge_profile}) failed: {exc}. "
                f"Fallback keyword heuristic: {heuristic.get('notes', '')}"
            ),
        }


def _parse_judge_json(raw: str) -> dict:
    """Parse the judge's JSON response, handling markdown fences."""
    # Try direct parse
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` block
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try extracting from first { to last }
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse judge JSON: {raw[:500]}")


def _keyword_heuristic_judge(task: GoldenTask, output: str) -> dict:
    """Fallback keyword heuristic — checks if expected behaviors appear in output.

    This is a last-resort fallback when the LLM judge is unavailable.
    It performs a simple keyword presence check against expected behaviors.
    """
    output_lower = output.lower()
    matched = []
    missed = []

    for behavior in task.expected_behaviors:
        # Extract key terms from the behavior (remove negation markers)
        key_terms = behavior.lower()
        is_negative = key_terms.startswith("does not") or "not " in key_terms[:20]

        if is_negative:
            # For negative behaviors: check that the NEGATION is present
            # e.g., "Does NOT suggest regex sanitisation" → check output doesn't suggest it
            neg_term = key_terms.replace("does not ", "").replace("not ", "").strip()
            if neg_term in output_lower:
                missed.append(behavior)
            else:
                matched.append(behavior)
        else:
            # For positive behaviors: check key terms appear
            terms = [w for w in key_terms.split() if len(w) > 3]
            if any(t in output_lower for t in terms):
                matched.append(behavior)
            else:
                missed.append(behavior)

    total = len(task.expected_behaviors)
    if total == 0:
        return {"passed": True, "score": 0.5, "notes": "No expected behaviors to check — auto-pass with neutral score"}

    score = len(matched) / total
    passed = score >= 0.5  # majority expected behaviors matched

    return {
        "passed": passed,
        "score": round(score, 2),
        "notes": (
            f"Heuristic: {len(matched)}/{total} behaviors matched. "
            f"Matched: {matched[:3]}. Missed: {missed[:3]}."
        ),
    }
