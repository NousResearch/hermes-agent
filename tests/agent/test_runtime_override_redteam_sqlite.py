"""RED-to-GREEN acceptance: temporary projection never clears durable session state."""

import json
import threading
from types import SimpleNamespace

import pytest

from agent.context_compressor import PROACTIVE_PRUNE_REARM_MODEL_CONFIG_KEY as PRUNE_KEY
from agent.runtime_override import apply_runtime_override, validate_runtime_override
from hermes_state import SessionDB
from tests.agent.test_runtime_override import (
    _OVERRIDE_MODEL,
    _build_model_owned_agent,
    _patch_model_owned_resolution,
)


def bound_agent(monkeypatch, tmp_path):
    _patch_model_owned_resolution(monkeypatch)
    db = SessionDB(tmp_path / "state.db")
    sid = "S1"
    db.create_session(sid, "test")
    db.set_compression_fallback_streak(sid, 5)
    db.set_compression_ineffective_count(sid, 3)
    db.set_compression_overload_streak(sid, 4)
    db.patch_session_model_config(sid, {PRUNE_KEY: 7, "other": "keep"})
    agent = _build_model_owned_agent()
    agent.context_compressor.bind_session_state(db, sid)
    return db, sid, agent


def read_state(db, sid):
    return (
        db.get_compression_fallback_streak(sid),
        db.get_compression_ineffective_count(sid),
        db.get_compression_overload_streak(sid),
        db.get_session_model_config_value(sid, PRUNE_KEY, "ABSENT"),
    )


def test_projection_and_normal_exit_do_not_write_durable_rows(monkeypatch, tmp_path):
    db, sid, agent = bound_agent(monkeypatch, tmp_path)
    before = read_state(db, sid)
    try:
        with apply_runtime_override(agent, {"model": _OVERRIDE_MODEL}):
            assert read_state(db, sid) == before
            assert agent.context_compressor._session_db is db
            assert agent.context_compressor._session_id == sid
        assert read_state(db, sid) == before
    finally:
        db.close()


def test_normal_exit_does_not_require_compensating_write(monkeypatch, tmp_path):
    db, sid, agent = bound_agent(monkeypatch, tmp_path)
    try:
        with apply_runtime_override(agent, {"model": _OVERRIDE_MODEL}):
            # A DB outage after projection cannot corrupt a row that the
            # projection never wrote; no compensating setter on scope exit.
            def outage(*args):
                raise RuntimeError("INJECTED_DB_OUTAGE")
            monkeypatch.setattr(db, "set_compression_overload_streak", outage)
        assert db.get_compression_overload_streak(sid) == 4
    finally:
        db.close()


def test_external_overload_write_survives_scope_exit(monkeypatch, tmp_path):
    db, sid, agent = bound_agent(monkeypatch, tmp_path)
    try:
        with apply_runtime_override(agent, {"model": _OVERRIDE_MODEL}):
            thread = threading.Thread(
                target=lambda: db.set_compression_overload_streak(sid, 7))
            thread.start()
            thread.join(timeout=10)
            assert not thread.is_alive()
        assert read_state(db, sid) == (5, 3, 7, 7)
    finally:
        db.close()


def test_one_new_write_does_not_cancel_unrelated_protections(monkeypatch, tmp_path):
    db, sid, agent = bound_agent(monkeypatch, tmp_path)
    try:
        with apply_runtime_override(agent, {"model": _OVERRIDE_MODEL}):
            db.set_compression_overload_streak(sid, 9)
        assert read_state(db, sid) == (5, 3, 9, 7)
    finally:
        db.close()


def test_raw_json_null_is_not_rewritten_as_absence(monkeypatch, tmp_path):
    db, sid, agent = bound_agent(monkeypatch, tmp_path)
    try:
        db._write_sql(
            "UPDATE sessions SET model_config = ? WHERE id = ?",
            (json.dumps({PRUNE_KEY: None, "other": 9}), sid),
        )
        with apply_runtime_override(agent, {"model": _OVERRIDE_MODEL}):
            assert json.loads(db.get_session(sid)["model_config"])[PRUNE_KEY] is None
        parsed = json.loads(db.get_session(sid)["model_config"])
        assert PRUNE_KEY in parsed and parsed[PRUNE_KEY] is None
        assert parsed["other"] == 9
    finally:
        db.close()


def test_in_turn_legitimate_compressor_write_still_persists(monkeypatch, tmp_path):
    db, sid, agent = bound_agent(monkeypatch, tmp_path)
    try:
        with apply_runtime_override(agent, {"model": _OVERRIDE_MODEL}):
            cc = agent.context_compressor
            assert cc._session_db is db
            cc._fallback_compression_streak = 11
            cc._persist_fallback_compression_streak()
            assert db.get_compression_fallback_streak(sid) == 11
        assert db.get_compression_fallback_streak(sid) == 11
    finally:
        db.close()


def test_in_memory_route_restore_failure_must_raise(monkeypatch):
    _patch_model_owned_resolution(monkeypatch)
    agent = _build_model_owned_agent()
    original_class = type(agent)

    class BrokenSetter(original_class):
        def __setattr__(self, name, value):
            if name == "model" and value == "orig-model" and getattr(self, "break_restore", False):
                raise RuntimeError("REFUSE_RESTORE")
            super().__setattr__(name, value)

    agent.__class__ = BrokenSetter
    with pytest.raises(RuntimeError, match="restore incomplete"):
        with apply_runtime_override(agent, {"model": _OVERRIDE_MODEL}):
            agent.break_restore = True
    assert agent._runtime_override == {}


def test_hook_payload_cannot_override_provider_or_credentials():
    assert validate_runtime_override({
        "model": "other", "api_key": "secret",
        "provider": "attacker", "base_url": "https://invalid.example",
        "api_mode": "responses",
    }) == {"model": "other"}


def test_concurrent_writer_keeps_binding_during_projection(monkeypatch, tmp_path):
    """Another thread's real write must not inherit projection-only suppression."""
    db, sid, agent = bound_agent(monkeypatch, tmp_path)
    from agent import agent_runtime_helpers
    original = agent_runtime_helpers._apply_model_owned_state
    observed = []

    def interleaved(agent_arg, new_model, snapshot=None, **kwargs):
        compressor = agent_arg.context_compressor
        assert compressor._session_db is db
        assert compressor._session_id == sid

        def legitimate_external_writer():
            observed.append(compressor._durable_write(
                "set_compression_overload_streak", "concurrent overload", 12
            ))
        worker = threading.Thread(target=legitimate_external_writer)
        worker.start()
        worker.join(timeout=10)
        assert not worker.is_alive()
        return original(agent_arg, new_model, snapshot=snapshot, **kwargs)

    try:
        monkeypatch.setattr(agent_runtime_helpers, "_apply_model_owned_state", interleaved)
        with apply_runtime_override(agent, {"model": _OVERRIDE_MODEL}):
            assert observed == [True]
            assert db.get_compression_overload_streak(sid) == 12
        assert db.get_compression_overload_streak(sid) == 12
        assert db.get_compression_fallback_streak(sid) == 5
    finally:
        db.close()


def test_real_fallback_persists_reset_even_after_ephemeral_zero(monkeypatch, tmp_path):
    """An actual switch writes its reset even if the temporary copy already has zero."""
    from agent.runtime_override import consume_runtime_override

    db, sid, agent = bound_agent(monkeypatch, tmp_path)
    try:
        assert db.get_compression_ineffective_count(sid) == 3
        with apply_runtime_override(agent, {"model": _OVERRIDE_MODEL}):
            temporary = agent.context_compressor
            assert temporary._ineffective_compression_count == 0
            assert db.get_compression_ineffective_count(sid) == 3
            temporary.update_model(
                model="fallback-model", context_length=96_000,
                provider="openai", api_mode="chat_completions",
                base_url="https://api.openai.com/v1",
            )
            consume_runtime_override(agent)
            assert db.get_compression_ineffective_count(sid) == 0
        assert db.get_compression_ineffective_count(sid) == 0
    finally:
        db.close()


def test_projection_suppression_expires_after_exception(monkeypatch, tmp_path):
    """The scope-local write guard must never outlive an exception."""
    from agent.ephemeral_model_projection import suppress_ephemeral_projection_writes

    db, sid, agent = bound_agent(monkeypatch, tmp_path)
    try:
        compressor = agent.context_compressor
        with pytest.raises(RuntimeError, match="injected"):
            with suppress_ephemeral_projection_writes(compressor):
                assert compressor._durable_write(
                    "set_compression_overload_streak", "projection reset", 0
                ) is False
                raise RuntimeError("injected")
        assert db.get_compression_overload_streak(sid) == 4
        assert compressor._durable_write(
            "set_compression_overload_streak", "real event", 13
        ) is True
        assert db.get_compression_overload_streak(sid) == 13
    finally:
        db.close()
