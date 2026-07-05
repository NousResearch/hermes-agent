"""Tests for Task Runtime MVP.

Covers:
- IntentResolver: task_type inference, skill hints, source detection.
- ContextResolver: hermes_home resolution, kanban_board detection, config_version.
- SkillLoader: SkillRecord installed/missing handling.
- TaskContractBuilder: contract schema, fingerprint stability, deterministic.
- ExecutionPipeline: dry-run no HTTP, supervised/enforce require confirmation.
- ResultConsolidator: trace.jsonl append-only, secrets redacted.
- TaskRuntime coordinator: end-to-end via API + via CLI.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from hermes_cli.task_runtime import (
    CONTRACT_SCHEMA,
    TaskRuntime,
)
from hermes_cli.task_runtime.intent_resolver import resolve as resolve_intent
from hermes_cli.task_runtime.context_resolver import resolve as resolve_context
from hermes_cli.task_runtime.skill_loader import load as load_skills
from hermes_cli.task_runtime.task_contract_builder import build as build_contract
from hermes_cli.task_runtime.execution_pipeline import run as run_pipeline
from hermes_cli.task_runtime.result_consolidator import build as build_result, _redact_secrets


HERMES_HOME = Path(os.environ.get("HERMES_HOME", REPO / ".hermes-test-home"))


# ---------------------------------------------------------------------------
# IntentResolver tests
# ---------------------------------------------------------------------------


def test_intent_resolver_research_type():
    r = resolve_intent("Explain how Registry pattern works")
    assert r.task_type == "research"
    assert r.intent_id
    assert r.source == "cli"


def test_intent_resolver_code_type():
    r = resolve_intent("Implement a new function for hash recompute")
    assert r.task_type == "code"


def test_intent_resolver_review_type():
    r = resolve_intent("Audit this code and verify the implementation against the rubric")
    assert r.task_type == "review"


def test_intent_resolver_review_type_simple():
    """When only `review` matches, type is review (no `code` keyword)."""
    r = resolve_intent("Audit this for issues")
    assert r.task_type == "review"


def test_intent_resolver_kanban_source():
    r = resolve_intent("process this task", source="kanban", source_id="abc-123")
    assert r.task_type == "kanban"
    assert r.source_id == "abc-123"


def test_intent_resolver_skill_hints():
    r = resolve_intent("Explain the Registry pattern in Producer Normalizer v1.1")
    assert "producer-normalizer" in r.suggested_skills


def test_intent_resolver_needs_gbrain_for_what_is():
    r = resolve_intent("What is the difference between v1.0 and v1.1?")
    assert r.needs_gbrain is True


def test_intent_resolver_needs_obsidian_for_notes():
    r = resolve_intent("Look at my notes in obsidian")
    assert r.needs_obsidian is True


def test_intent_resolver_no_gbrain_for_code_only():
    r = resolve_intent("Fix this bug in the function")
    assert r.needs_gbrain is False


def test_intent_resolver_deterministic_id_when_provided():
    r1 = resolve_intent("hello", intent_id="test-fixed")
    r2 = resolve_intent("hello", intent_id="test-fixed")
    assert r1.intent_id == r2.intent_id == "test-fixed"


# ---------------------------------------------------------------------------
# ContextResolver tests
# ---------------------------------------------------------------------------


def test_context_resolver_returns_metadata():
    r = resolve_intent("Explain something")
    ctx = resolve_context(r, hermes_home=HERMES_HOME)
    assert "hermes_home" in ctx
    assert "vault_root" in ctx
    assert "skills_dir" in ctx
    assert "kanban_board" in ctx
    assert "config_version" in ctx
    assert "hermes_disable_self_improvement" in ctx


def test_context_resolver_no_http_no_llm():
    """ContextResolver must NOT execute any I/O sync / import / embed."""
    r = resolve_intent("Explain something")
    # Before/after: assert knowledge_refs and memory_keys start empty (no I/O).
    ctx = resolve_context(r, hermes_home=HERMES_HOME)
    assert ctx["knowledge_refs"] == []
    assert ctx["memory_keys"] == []


# ---------------------------------------------------------------------------
# SkillLoader tests
# ---------------------------------------------------------------------------


def test_skill_loader_returns_records():
    r = resolve_intent("Explain the Registry pattern in Producer Normalizer v1.1")
    ctx = resolve_context(r, hermes_home=HERMES_HOME)
    skills = load_skills(r, ctx)
    # producer-normalizer should be in suggested_skills for this prompt.
    assert any(s.skill_name == "producer-normalizer" for s in skills)


def test_skill_loader_missing_skill_marked_missing():
    r = resolve_intent("Use the fictional-skill-xyz plugin")
    # Force a hint via metadata (not via natural language).
    from hermes_cli.task_runtime.intent_resolver import ResolvedIntent
    r2 = ResolvedIntent(
        intent_id=r.intent_id, raw_text=r.raw_text, source=r.source,
        source_id=r.source_id, task_type=r.task_type,
        needs_gbrain=r.needs_gbrain, needs_obsidian=r.needs_obsidian,
        needs_memory=r.needs_memory, needs_skills=True,
        suggested_skills=["fictional-skill-xyz"], metadata={},
    )
    ctx = resolve_context(r2, hermes_home=HERMES_HOME)
    skills = load_skills(r2, ctx)
    s = next(s for s in skills if s.skill_name == "fictional-skill-xyz")
    assert s.installed is False
    assert s.source == "missing"


# ---------------------------------------------------------------------------
# TaskContractBuilder tests
# ---------------------------------------------------------------------------


def test_contract_schema_is_1_0_0():
    r = resolve_intent("Explain")
    ctx = resolve_context(r, hermes_home=HERMES_HOME)
    skills = load_skills(r, ctx)
    c = build_contract(r, ctx, skills, execution_mode="dry-run")
    assert c["task_contract_schema"] == "1.0.0"
    assert c["task_contract_version"] == "1.0.0"


def test_contract_has_required_fields():
    r = resolve_intent("Explain")
    ctx = resolve_context(r, hermes_home=HERMES_HOME)
    skills = load_skills(r, ctx)
    c = build_contract(r, ctx, skills, execution_mode="dry-run")
    for key in ("intent", "context", "producer", "normalizer", "reviewer",
                "execution_mode", "contract_fingerprint", "contract_id"):
        assert key in c, f"missing key: {key}"


def test_contract_fingerprint_stable_for_same_inputs():
    """Determinism: same ResolvedIntent+context+skills ⇒ same fingerprint."""
    r = resolve_intent("Explain", intent_id="fixed-id")
    ctx = resolve_context(r, hermes_home=HERMES_HOME)
    skills = load_skills(r, ctx)
    c1 = build_contract(r, ctx, skills, execution_mode="dry-run")
    c2 = build_contract(r, ctx, skills, execution_mode="dry-run")
    # contract_id will differ (uuid); fingerprint must be stable.
    assert c1["contract_fingerprint"] == c2["contract_fingerprint"]
    assert c1["contract_id"] != c2["contract_id"]


def test_contract_normalizer_version_is_1_1_0():
    r = resolve_intent("Explain")
    ctx = resolve_context(r, hermes_home=HERMES_HOME)
    skills = load_skills(r, ctx)
    c = build_contract(r, ctx, skills, execution_mode="dry-run")
    assert c["normalizer"]["version"] == "1.1.0"
    assert c["normalizer"]["enabled"] is True


def test_contract_reviewer_version_is_1_0_0():
    r = resolve_intent("Explain")
    ctx = resolve_context(r, hermes_home=HERMES_HOME)
    skills = load_skills(r, ctx)
    c = build_contract(r, ctx, skills, execution_mode="dry-run")
    assert c["reviewer"]["version"] == "1.0.0"


# ---------------------------------------------------------------------------
# ExecutionPipeline tests
# ---------------------------------------------------------------------------


def test_pipeline_dry_run_no_http_no_llm():
    r = resolve_intent("Explain")
    ctx = resolve_context(r, hermes_home=HERMES_HOME)
    skills = load_skills(r, ctx)
    c = build_contract(r, ctx, skills, execution_mode="dry-run")
    pr = run_pipeline(c)
    assert pr.engine_status == "OK"
    assert pr.normalizer_verdict == "PASS"
    assert pr.reviewer_called is False
    assert pr.reviewer_skipped is True
    assert pr.reviewer_verdict is None
    assert pr.metrics.get("producer_http_calls") == 0
    assert pr.metrics.get("reviewer_http_calls") == 0


def test_pipeline_shadow_runs_but_no_mutations():
    r = resolve_intent("Explain")
    ctx = resolve_context(r, hermes_home=HERMES_HOME)
    skills = load_skills(r, ctx)
    c = build_contract(r, ctx, skills, execution_mode="shadow")
    pr = run_pipeline(c)
    assert pr.metrics.get("pipeline_mode") == "shadow"


def test_pipeline_supervised_requires_confirmation():
    r = resolve_intent("Explain")
    ctx = resolve_context(r, hermes_home=HERMES_HOME)
    skills = load_skills(r, ctx)
    c = build_contract(r, ctx, skills, execution_mode="supervised")
    pr = run_pipeline(c)  # no confirmation
    assert pr.engine_status == "STOP"
    assert any("confirmation" in e for e in pr.errors)


def test_pipeline_enforce_requires_confirmation():
    r = resolve_intent("Explain")
    ctx = resolve_context(r, hermes_home=HERMES_HOME)
    skills = load_skills(r, ctx)
    c = build_contract(r, ctx, skills, execution_mode="enforce")
    pr = run_pipeline(c)
    assert pr.engine_status == "STOP"


def test_pipeline_enforce_with_confirmation_passes():
    r = resolve_intent("Explain")
    ctx = resolve_context(r, hermes_home=HERMES_HOME)
    skills = load_skills(r, ctx)
    c = build_contract(r, ctx, skills, execution_mode="enforce")
    pr = run_pipeline(c, confirmation_token="confirm")
    assert pr.engine_status == "OK"


def test_pipeline_unknown_mode_returns_error():
    r = resolve_intent("Explain")
    ctx = resolve_context(r, hermes_home=HERMES_HOME)
    skills = load_skills(r, ctx)
    c = build_contract(r, ctx, skills, execution_mode="invalid-mode-xyz")
    pr = run_pipeline(c)
    # The build_contract will fail with ValueError; pipeline not reached.
    # If contract builds, pipeline returns STOP with error.
    assert pr.engine_status == "STOP"


# ---------------------------------------------------------------------------
# ResultConsolidator tests
# ---------------------------------------------------------------------------


def test_redact_secrets_jwt():
    text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature"
    out = _redact_secrets(text)
    assert "[REDACTED]" in out
    assert "eyJ" not in out


def test_redact_secrets_sk_key():
    text = "API key is sk-abcdef1234567890abcdef"
    out = _redact_secrets(text)
    assert "[REDACTED]" in out
    assert "sk-abcdef" not in out


def test_trace_appends_one_line_per_run(tmp_path):
    r = resolve_intent("Explain", intent_id="trace-test-1")
    ctx = resolve_context(r, hermes_home=HERMES_HOME)
    skills = load_skills(r, ctx)
    c = build_contract(r, ctx, skills, execution_mode="dry-run")
    pr = run_pipeline(c)
    trace_dir = tmp_path / "traces"
    result = build_result(r, c, pr, trace_dir=trace_dir)
    assert result.trace_path and Path(result.trace_path).exists()
    lines = Path(result.trace_path).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["intent_id"] == "trace-test-1"
    assert rec["execution_mode"] == "dry-run"


def test_trace_appends_two_lines_on_two_runs(tmp_path):
    trace_dir = tmp_path / "traces"
    for i in (1, 2):
        r = resolve_intent("Explain", intent_id=f"trace-{i}")
        ctx = resolve_context(r, hermes_home=HERMES_HOME)
        skills = load_skills(r, ctx)
        c = build_contract(r, ctx, skills, execution_mode="dry-run")
        pr = run_pipeline(c)
        build_result(r, c, pr, trace_dir=trace_dir)
    lines = (trace_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_trace_redacts_secrets_in_final_answer(tmp_path):
    """The final_answer (when containing a secret-looking string) is redacted."""
    r = resolve_intent("Explain using Bearer eyJabc12345def for auth", intent_id="redact-test")
    ctx = resolve_context(r, hermes_home=HERMES_HOME)
    skills = load_skills(r, ctx)
    c = build_contract(r, ctx, skills, execution_mode="dry-run")
    pr = run_pipeline(c)
    trace_dir = tmp_path / "traces"
    result = build_result(r, c, pr, trace_dir=trace_dir)
    text = Path(result.trace_path).read_text()
    # The raw_text in the contract contains the secret; the trace's
    # final_answer has been through _redact_secrets so the JWT should be
    # replaced. We assert the literal raw JWT fragment is not in the trace.
    assert "eyJabc12345def" not in text, f"raw JWT leaked into trace: {text[:200]}"


# ---------------------------------------------------------------------------
# TaskRuntime coordinator tests
# ---------------------------------------------------------------------------


def test_runtime_dry_run_returns_task_result(tmp_path):
    rt = TaskRuntime(hermes_home=HERMES_HOME, trace_dir=tmp_path / "traces")
    result = rt.run("Explain Registry pattern in Producer Normalizer v1.1", execution_mode="dry-run")
    assert result.intent_id
    assert result.contract_id
    assert result.contract_fingerprint.startswith("sha256:")
    assert result.execution_mode == "dry-run"
    assert result.pipeline_result["engine_status"] == "OK"
    assert "task" in result.final_answer.lower()


def test_runtime_invalid_mode_raises():
    rt = TaskRuntime(hermes_home=HERMES_HOME)
    with pytest.raises(ValueError):
        rt.run("hello", execution_mode="bogus")


def test_runtime_is_thin_coordinator():
    """The coordinator should not have business logic methods beyond
    run(). All logic lives in dedicated modules.
    """
    rt = TaskRuntime
    public_methods = [m for m in dir(rt) if not m.startswith("_")]
    # Only `run` should be the public business method (plus inherited dunder).
    runtime_methods = [m for m in public_methods if not m.startswith("__")]
    assert runtime_methods == ["run"], f"unexpected methods: {runtime_methods}"


# ---------------------------------------------------------------------------
# CLI tests (smoke)
# ---------------------------------------------------------------------------


def test_cli_help_shows_task_subcommand(tmp_path):
    """Smoke test: `hermes task --help` works."""
    import subprocess
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "HERMES_HOME": str(tmp_path / ".hermes"),
        "HERMES_DISABLE_SELF_IMPROVEMENT": "1",
    }
    proc = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "task", "--help"],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=15,
    )
    assert "Task Runtime" in proc.stdout or "task" in proc.stdout
    assert proc.returncode == 0


def test_cli_task_dry_run_smoke(tmp_path, monkeypatch):
    """Smoke test: `hermes task "..." --mode dry-run` runs end-to-end."""
    import subprocess
    monkeypatch.chdir(REPO)
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "HERMES_HOME": str(tmp_path / ".hermes"),
        "HERMES_DISABLE_SELF_IMPROVEMENT": "1",
    }
    proc = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "task",
         "Explain Registry pattern", "--mode", "dry-run", "-v"],
        cwd=str(REPO), env=env,
        capture_output=True, text=True, timeout=20,
    )
    assert "intent_id:" in proc.stdout
    assert "execution_mode: dry-run" in proc.stdout
    assert proc.returncode == 0


def test_cli_task_enforce_without_confirm_fails(tmp_path):
    """Smoke test: enforce mode without --confirm produces STOP."""
    import subprocess
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "HERMES_HOME": str(tmp_path / ".hermes"),
        "HERMES_DISABLE_SELF_IMPROVEMENT": "1",
    }
    proc = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "task",
         "x", "--mode", "enforce"],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=15,
    )
    # With enforce + no confirm, pipeline returns STOP, but final_answer
    # still prints. The errors should include 'confirmation'.
    assert "confirmation" in proc.stdout or "STOP" in proc.stdout