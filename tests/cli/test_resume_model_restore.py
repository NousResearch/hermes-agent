"""Tests for CLI resume model restoration and /model session persistence.

Covers _restore_session_model, _persist_model_switch_to_session (cli.py) and
SessionDB.session_gateway_runtime (hermes_state.py) — the round trip that
makes `hermes --resume` reopen a session on the model/provider it actually
used instead of the ambient config default (#57588-class, #79536).
"""

import json

import pytest

import cli as cli_mod
from hermes_state import SessionDB


def _make_stub(**overrides):
    """Bare HermesCLI the way resume paths see it (no __init__)."""
    stub = object.__new__(cli_mod.HermesCLI)
    stub.model = "ambient-model"
    stub.provider = "openrouter"
    stub.requested_provider = "openrouter"
    stub.base_url = "https://openrouter.ai/api/v1"
    stub.api_key = "ambient-key"
    stub.api_mode = ""
    stub.agent = None
    stub._console_print = lambda s: None
    for key, value in overrides.items():
        setattr(stub, key, value)
    return stub


def _row(model="glm-4.7", model_config=None):
    return {
        "model": model,
        "model_config": json.dumps(model_config) if model_config else None,
    }


# ── SessionDB.session_gateway_runtime ───────────────────────────────


def test_session_gateway_runtime_prefers_nested_key():
    meta = _row(model_config={
        "gateway_runtime": {"provider": "custom:feather", "base_url": "https://f/v1"},
        "provider": "openrouter",
    })
    runtime = SessionDB.session_gateway_runtime(meta)
    assert runtime["provider"] == "custom:feather"
    assert runtime["base_url"] == "https://f/v1"


def test_session_gateway_runtime_falls_back_to_top_level_keys():
    # The TUI gateway's _runtime_model_config writes top-level keys only.
    meta = _row(model_config={"provider": "nous", "api_mode": "chat_completions"})
    runtime = SessionDB.session_gateway_runtime(meta)
    assert runtime == {"provider": "nous", "api_mode": "chat_completions"}


def test_session_gateway_runtime_tolerates_garbage():
    assert SessionDB.session_gateway_runtime(None) == {}
    assert SessionDB.session_gateway_runtime({}) == {}
    assert SessionDB.session_gateway_runtime({"model_config": "{not json"}) == {}
    assert SessionDB.session_gateway_runtime({"model_config": json.dumps([1, 2])}) == {}


# ── _restore_session_model ──────────────────────────────────────────


def test_restore_session_model_restores_model_and_provider():
    stub = _make_stub()
    stub._restore_session_model(_row(model_config={
        "gateway_runtime": {"provider": "custom:feather", "base_url": "https://f/v1"},
    }))
    assert stub.model == "glm-4.7"
    assert stub.provider == "custom:feather"
    assert stub.requested_provider == "custom:feather"
    assert stub.base_url == "https://f/v1"
    # Stale launch-time explicit overrides must not leak into the restored
    # provider's credential resolution.
    assert stub._explicit_api_key is None
    assert stub._explicit_base_url == "https://f/v1"


def test_restore_session_model_explicit_cli_flag_wins():
    stub = _make_stub(model="cli-flag-model", _explicit_model_override=True)
    stub._restore_session_model(_row())
    assert stub.model == "cli-flag-model"
    assert stub.provider == "openrouter"


def test_restore_session_model_no_stored_model_is_noop():
    stub = _make_stub()
    stub._restore_session_model(_row(model=None))
    assert stub.model == "ambient-model"


def test_restore_session_model_matching_state_is_silent_noop():
    notes = []
    stub = _make_stub(model="glm-4.7", provider="custom:feather",
                      requested_provider="custom:feather",
                      _console_print=lambda s: notes.append(s))
    stub._restore_session_model(_row(model_config={
        "gateway_runtime": {"provider": "custom:feather"},
    }))
    assert not notes


def test_restore_session_model_swaps_running_agent_in_place():
    calls = {}

    class _Agent:
        def switch_model(self, **kwargs):
            calls.update(kwargs)

    stub = _make_stub(agent=_Agent())
    stub._restore_session_model(_row())
    assert calls["new_model"] == "glm-4.7"


# ── _persist_model_switch_to_session ────────────────────────────────


class _Result:
    new_model = "deepseek-v4-flash-free"
    target_provider = "custom:opencode-zen"
    base_url = "https://oz/v1"
    api_mode = ""


def test_persist_model_switch_writes_model_and_both_route_shapes():
    written = {}

    class _DB:
        def update_session_model(self, sid, model):
            written["model"] = (sid, model)

        def patch_session_model_config(self, sid, patch):
            written["patch"] = (sid, patch)

    stub = _make_stub(_session_db=_DB(), session_id="s1")
    stub._persist_model_switch_to_session(_Result())
    assert written["model"] == ("s1", "deepseek-v4-flash-free")
    sid, patch = written["patch"]
    # Nested shape for the CLI reader...
    assert patch["gateway_runtime"]["provider"] == "custom:opencode-zen"
    # ...and top-level for the TUI gateway's _stored_session_runtime_overrides.
    assert patch["provider"] == "custom:opencode-zen"
    assert patch["base_url"] == "https://oz/v1"
    assert "api_mode" not in patch["gateway_runtime"]  # empty values dropped


def test_persist_model_switch_noop_without_db_or_session():
    stub = _make_stub()  # no _session_db / session_id attributes at all
    stub._persist_model_switch_to_session(_Result())  # must not raise


def test_persist_model_switch_swallows_db_errors():
    class _DB:
        def update_session_model(self, *a):
            raise RuntimeError("disk full")

    stub = _make_stub(_session_db=_DB(), session_id="s1")
    stub._persist_model_switch_to_session(_Result())  # must not raise


def test_persist_model_switch_heals_bare_custom(monkeypatch):
    """Bare 'custom' is not routable — heal to custom:<name> or drop (C1)."""
    written = {}

    class _DB:
        def update_session_model(self, sid, model):
            written["model"] = model

        def patch_session_model_config(self, sid, patch):
            written["patch"] = patch

    class _BareResult:
        new_model = "qwen3.6-plus"
        target_provider = "custom"
        base_url = "https://my-endpoint/v1"
        api_mode = ""

    import hermes_cli.runtime_provider as rp
    monkeypatch.setattr(rp, "canonical_custom_identity",
                        lambda base_url=None, model=None: "custom:myendpoint")
    stub = _make_stub(_session_db=_DB(), session_id="s1")
    stub._persist_model_switch_to_session(_BareResult())
    assert written["patch"]["provider"] == "custom:myendpoint"

    # Healing fails -> provider dropped entirely, not persisted bare.
    monkeypatch.setattr(rp, "canonical_custom_identity",
                        lambda base_url=None, model=None: None)
    written.clear()
    stub._persist_model_switch_to_session(_BareResult())
    assert "provider" not in written["patch"]
    assert "provider" not in written["patch"]["gateway_runtime"]


def test_restore_session_model_heals_bare_custom_stored_rows(monkeypatch):
    """Rows persisted by older builds may carry bare 'custom' — heal or drop."""
    import hermes_cli.runtime_provider as rp
    monkeypatch.setattr(rp, "canonical_custom_identity",
                        lambda base_url=None, model=None: None)
    stub = _make_stub()
    stub._restore_session_model(_row(model_config={
        "gateway_runtime": {"provider": "custom"},
    }))
    # Provider dropped -> model restored but provider stays ambient.
    assert stub.model == "glm-4.7"
    assert stub.provider == "openrouter"


# ── round trip: persist → get_session shape → restore ───────────────


def test_round_trip_persist_then_restore(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="rt1", source="cli", model="ambient-model")

    stub = _make_stub(_session_db=db, session_id="rt1")
    stub._persist_model_switch_to_session(_Result())

    meta = db.get_session("rt1")
    restored = _make_stub()
    restored._restore_session_model(meta)
    assert restored.model == "deepseek-v4-flash-free"
    assert restored.provider == "custom:opencode-zen"
    assert restored.base_url == "https://oz/v1"
