"""Compression retry diagnostics are not the identity of a failure episode."""
import copy
from types import SimpleNamespace

import pytest

from agent import conversation_compression as cc


def _error(elapsed):
    return f"Codex auxiliary Responses stream stalled: no new output for 60.0s ({elapsed}s elapsed)"


def _agent():
    warnings = []
    return SimpleNamespace(
        context_compressor=SimpleNamespace(_last_compress_aborted=True),
        session_id="one", _emit_warning=warnings.append, warnings=warnings,
    )


def _abort(agent, error):
    agent.context_compressor._last_summary_error = error
    messages = [{"role": "user", "content": "keep this"}]
    before = copy.deepcopy(messages)
    assert cc._candidate_rejected(
        agent, messages, messages, before, attempt_generation=None, attempt_started_at=0,
    )
    assert messages == before


@pytest.mark.parametrize("errors,expected", [
    ([_error("79.9"), _error("79.9")], 1),
    ([_error("79.9"), _error("83.7")], 1),
    ([_error("79.9"), "HTTP 401", "HTTP 403"], 3),
    (["other service (79.9s elapsed)", "other service (83.7s elapsed)"], 2),
    ([None, None], 1),
])
def test_abort_warning_identity(monkeypatch, errors, expected):
    attempts = []
    monkeypatch.setattr(cc, "_emit_aborted_attempt_telemetry", lambda *args: attempts.append(args))
    agent = _agent()
    for error in errors:
        _abort(agent, error)
    assert len(agent.warnings) == expected
    assert len(attempts) == len(errors)
    assert agent._last_compression_summary_warning == (errors[-1] or "unknown error")
    assert (errors[0] or "unknown error") in agent.warnings[0]


def test_new_session_rearms_warning(monkeypatch):
    monkeypatch.setattr(cc, "_emit_aborted_attempt_telemetry", lambda *args: None)
    agent = _agent()
    _abort(agent, _error("79.9"))
    agent.session_id = "two"
    _abort(agent, _error("79.9"))
    assert len(agent.warnings) == 2


def test_changed_attempt_route_rearms_warning(monkeypatch):
    monkeypatch.setattr(cc, "_emit_aborted_attempt_telemetry", lambda *args: None)
    agent = _agent()
    for attempt, provider, model in [("a", "provider-a", "model-a"), ("b", "provider-a", "model-b"),
                                     ("c", "provider-b", "model-b")]:
        agent._compression_attempt_id = attempt
        agent.context_compressor._last_compression_telemetry = {
            "attempt_id": attempt, "aux_provider": provider, "aux_model": model,
        }
        _abort(agent, _error("79.9"))
    assert len(agent.warnings) == 3


@pytest.mark.parametrize("committed,progress,fallback,expected", [
    (True, True, False, 2), (False, True, False, 1),
    (True, False, False, 1), (True, True, True, 1),
])
def test_only_committed_recovery_rearms_warning(monkeypatch, committed, progress, fallback, expected):
    monkeypatch.setattr(cc, "_emit_aborted_attempt_telemetry", lambda *args: None)
    monkeypatch.setattr(cc, "_reset_read_dedup_caches", lambda *args, **kwargs: None)
    agent = _agent()
    agent.tools = []
    agent.context_compressor.compression_count = 1
    _abort(agent, _error("79.9"))
    cc._finish_compaction_boundary(
        agent, [{"role": "user", "content": "summary"}], new_system_prompt="",
        old_session_id=None, in_place=False, compacted_in_place=False,
        session_commit_succeeded=committed, defer_context_engine_notification=False,
        compression_made_progress=progress, compression_used_fallback=fallback,
        compression_feasibility_skip=False, task_id="test",
    )
    _abort(agent, _error("79.9"))
    assert len(agent.warnings) == expected


def test_fallback_outcome_still_notifies(monkeypatch):
    monkeypatch.setattr(cc, "_emit_aborted_attempt_telemetry", lambda *args: None)
    agent = _agent()
    _abort(agent, _error("79.9"))
    agent.context_compressor._last_summary_error = _error("79.9")
    cc._warn_summary_or_aux_fallback(agent)
    cc._warn_summary_or_aux_fallback(agent)
    assert len(agent.warnings) == 2
    assert "fallback context marker" in agent.warnings[-1]
