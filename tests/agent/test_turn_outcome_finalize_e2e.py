"""E2E: the full ``finalize_turn`` path produces a Layer 0 outcome.

Drives the real decision chain — turn-finalizer → ``evaluate_turn_outcome`` →
``tools.skill_verify`` (real subprocess) → ``skill_usage.bump_outcome`` sidecar
— with a used skill whose mechanical verifier FAILs. Only the aux-model seam is
stubbed (no client → None, which the down-only rule makes irrelevant when a
verifier FAIL is on the table).
"""

from __future__ import annotations

import json

import pytest

from agent.turn_finalizer import finalize_turn


class _StubBudget:
    used = 5
    max_total = 3
    remaining = 0


class _StubCompressor:
    last_prompt_tokens = 0


class _StubAgent:
    """Minimal agent surface ``finalize_turn`` reads from."""

    def __init__(self, *, used_skills, failed_file_mutations):
        self._turn_used_skills = set(used_skills)
        self._turn_failed_file_mutations = dict(failed_file_mutations)
        self.max_iterations = 3
        self.iteration_budget = _StubBudget()
        self.context_compressor = _StubCompressor()
        self.model = "stub/model"
        self.provider = "stub"
        self.base_url = "http://stub"
        self.session_id = "sess-1"
        self.quiet_mode = True
        self.platform = "cli"
        self._interrupt_requested = False
        self._interrupt_message = None
        self._tool_guardrail_halt_decision = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        for attr in (
            "session_input_tokens",
            "session_output_tokens",
            "session_cache_read_tokens",
            "session_cache_write_tokens",
            "session_reasoning_tokens",
            "session_prompt_tokens",
            "session_completion_tokens",
            "session_total_tokens",
            "session_estimated_cost_usd",
        ):
            setattr(self, attr, 0)
        self.session_cost_status = "ok"
        self.session_cost_source = "stub"

    # --- fallible cleanup surfaces -------------------------------------
    def _save_trajectory(self, *a, **k):
        pass

    def _cleanup_task_resources(self, *a, **k):
        pass

    def _drop_trailing_empty_response_scaffolding(self, *a, **k):
        pass

    def _persist_session(self, *a, **k):
        pass

    # --- harmless no-ops ------------------------------------------------
    def _emit_status(self, *a, **k):
        pass

    def _safe_print(self, *a, **k):
        pass

    def _handle_max_iterations(self, messages, n):
        return "done"

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **k):
        pass


def _write_skill_with_failing_verifier(home, name):
    """A curation-eligible skill whose verifier mechanically FAILs."""
    skills_dir = home / "skills"
    d = skills_dir / name
    d.mkdir(parents=True)
    (d / "scripts").mkdir()
    (d / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: e2e verifier skill\n"
        "version: 1.0.0\n"
        "metadata:\n"
        "  hermes:\n"
        "    verify: {run: \"scripts/verify.py\", timeout_seconds: 30}\n"
        "---\n"
        f"# {name}\n",
        encoding="utf-8",
    )
    payload = json.dumps({"success": False, "reason": "e2e mechanical fail"})
    (d / "scripts" / "verify.py").write_text(
        "print(" + repr(payload) + ")\n", encoding="utf-8"
    )
    return d


def _write_plain_skill(home, name):
    """A curation-eligible skill with no verifier (unverified residue)."""
    skills_dir = home / "skills"
    d = skills_dir / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: e2e plain skill\n"
        "version: 1.0.0\n"
        "---\n"
        f"# {name}\n",
        encoding="utf-8",
    )
    return d


def _run_turn(agent, *, final_response="all done"):
    messages = [
        {"role": "user", "content": "do a thing"},
        {"role": "assistant", "content": "working"},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "assistant", "content": final_response},
    ]
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=None,
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="do a thing",
        original_user_message="do a thing",
        _should_review_memory=False,
        _turn_exit_reason="text_response(Done)",
    )


@pytest.fixture
def outcome_home(tmp_path, monkeypatch):
    """HERMES_HOME already points at a per-test tempdir (autouse fixture); the
    fixture returns the tempdir so tests can write skills into ``skills/``."""
    import os

    home = tmp_path / "hermes_test"
    return home


def test_finalize_turn_records_verifier_fail_and_attaches_outcome(
    outcome_home, monkeypatch
):
    from tools.skill_usage import get_record, set_verify_enabled

    skill = "e2e_skill"
    _write_skill_with_failing_verifier(outcome_home, skill)
    set_verify_enabled(skill, True)

    # Enable the pipeline and stub ONLY the aux seam (no client → None; the
    # down-only rule records the mechanical FAIL regardless).
    monkeypatch.setattr(
        "agent.turn_outcome._default_outcome_config",
        lambda: {"enabled": True, "run": "auto"},
    )
    monkeypatch.setattr(
        "agent.turn_outcome._default_aux_eval", lambda prompt: None
    )

    agent = _StubAgent(
        used_skills=[skill], failed_file_mutations={}
    )
    result = _run_turn(agent)

    # The seam is on the result dict.
    assert result["skills_used"] == [skill]
    assert result["outcome"] is not None
    assert result["outcome"]["task_succeeded"] is False
    assert result["outcome"]["failure_points"] == [skill]
    assert "e2e mechanical fail" in result["outcome"]["reason"]

    # The mechanical FAIL landed on the skill's sidecar.
    assert get_record(skill)["recent_outcomes"] == [False]


def test_finalize_turn_feeds_tool_errors_into_evidence_catalog(
    outcome_home, monkeypatch
):
    """The real finalize path enumerates this turn's tool errors into the
    evidence catalog, and a confident judge citation against them lands as a
    hard False (gated tier: tool-error existence + confidence floor)."""
    from tools.skill_usage import get_record

    skill = "e2e_tool"
    _write_plain_skill(outcome_home, skill)

    monkeypatch.setattr(
        "agent.turn_outcome._default_outcome_config",
        lambda: {"enabled": True, "run": "auto"},
    )

    seen_prompt = {}

    def _aux(prompt):
        seen_prompt["text"] = prompt
        # [1] is the tool_error evidence item for this turn's failing tool.
        return {
            "task_succeeded": False,
            "confidence": 0.9,
            "failure_points": [{"skill": skill, "evidence": [1]}],
            "reason": "the write failed",
        }

    monkeypatch.setattr("agent.turn_outcome._default_aux_eval", _aux)

    agent = _StubAgent(used_skills=[skill], failed_file_mutations={})
    messages = [
        {"role": "user", "content": "do a thing"},
        {"role": "assistant", "content": "writing"},
        {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "write_file",
            "content": json.dumps({"error": "permission denied"}),
        },
        {"role": "assistant", "content": "all done"},
    ]
    result = finalize_turn(
        agent,
        final_response="all done",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=None,
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="do a thing",
        original_user_message="do a thing",
        _should_review_memory=False,
        _turn_exit_reason="text_response(Done)",
    )

    # The evidence catalog reached the judge's prompt with the tool error.
    assert "tool_error(write_file)" in seen_prompt.get("text", "")
    assert result["outcome"] is not None
    assert result["outcome"]["failure_points"] == [skill]
    assert get_record(skill)["recent_outcomes"] == [False]


def test_disabled_pipeline_is_inert_through_finalize(outcome_home, monkeypatch):
    """Default config (enabled: false) → outcome None, no sidecar write, and
    no trajectory change — the dormant surface costs nothing."""
    from tools.skill_usage import get_record, set_verify_enabled

    skill = "e2e_off"
    _write_skill_with_failing_verifier(outcome_home, skill)
    set_verify_enabled(skill, True)

    monkeypatch.setattr(
        "agent.turn_outcome._default_outcome_config",
        lambda: {"enabled": False},
    )
    # A stub that would FAIL if it were ever called (it must not be).
    def _boom(prompt):
        raise AssertionError("aux eval must not run when the pipeline is off")

    monkeypatch.setattr("agent.turn_outcome._default_aux_eval", _boom)

    agent = _StubAgent(used_skills=[skill], failed_file_mutations={})
    result = _run_turn(agent)

    assert result["skills_used"] == [skill]
    assert result["outcome"] is None
    assert get_record(skill)["recent_outcomes"] == []


def test_interrupted_turn_produces_no_outcome(outcome_home, monkeypatch):
    """User-stopped turns are not work failures — nothing is recorded."""
    from tools.skill_usage import get_record, set_verify_enabled

    skill = "e2e_stop"
    _write_skill_with_failing_verifier(outcome_home, skill)
    set_verify_enabled(skill, True)

    monkeypatch.setattr(
        "agent.turn_outcome._default_outcome_config",
        lambda: {"enabled": True},
    )

    agent = _StubAgent(used_skills=[skill], failed_file_mutations={})
    messages = [{"role": "user", "content": "do a thing"}]
    result = finalize_turn(
        agent,
        final_response=None,
        api_call_count=1,
        interrupted=True,
        failed=False,
        messages=messages,
        conversation_history=None,
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="do a thing",
        original_user_message="do a thing",
        _should_review_memory=False,
        _turn_exit_reason="user_stopped",
    )

    assert result["outcome"] is None
    assert get_record(skill)["recent_outcomes"] == []
