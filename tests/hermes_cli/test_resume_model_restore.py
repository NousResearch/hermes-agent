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
    # Both shapes use or-None so stale keys are deleted (not merely omitted)
    # in BOTH gateway_runtime and top-level — the asymmetry that caused the
    # original stale-key bug.
    assert patch["gateway_runtime"]["api_mode"] is None
    assert patch["api_mode"] is None


def test_persist_model_switch_clears_stale_route_keys(tmp_path, monkeypatch):
    """A later switch must not inherit the previous switch's api_mode/base_url.

    patch_session_model_config merges key-level and only deletes on explicit
    None — dropping falsy values from the patch left the FIRST switch's
    api_mode (e.g. anthropic_messages) alive under the SECOND switch's
    provider, corrupting the wire protocol on TUI/desktop resume.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="stale1", source="cli", model="m0")
    stub = _make_stub(_session_db=db, session_id="stale1")

    class _First:
        new_model = "claude-x"
        target_provider = "custom:feather"
        base_url = "https://feather/v1"
        api_mode = "anthropic_messages"

    class _Second:
        new_model = "gpt-5.4"
        target_provider = "openrouter"
        base_url = "https://openrouter.ai/api/v1"
        api_mode = ""  # openrouter default — must ERASE the anthropic mode

    stub._persist_model_switch_to_session(_First())
    stub._persist_model_switch_to_session(_Second())

    meta = db.get_session("stale1")
    config = json.loads(meta["model_config"])
    # Top-level keys: stale values deleted.
    assert config["provider"] == "openrouter"
    assert "api_mode" not in config, config  # stale anthropic_messages deleted
    # Nested gateway_runtime: stale values replaced with None (the merge
    # replaces the entire gateway_runtime dict, not deep-merging its keys).
    # The reader's `or None` / `if v` filtering treats None the same as
    # absent, so stale values are effectively erased.
    gw = config.get("gateway_runtime", {})
    assert gw.get("provider") == "openrouter"
    assert gw.get("api_mode") is None  # stale anthropic_messages erased
    runtime = SessionDB.session_gateway_runtime(meta)
    assert runtime["provider"] == "openrouter"
    assert "api_mode" not in runtime


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

    # Healing fails -> provider dropped (explicit None deletes any stale
    # persisted provider), never persisted bare.
    monkeypatch.setattr(rp, "canonical_custom_identity",
                        lambda base_url=None, model=None: None)
    written.clear()
    stub._persist_model_switch_to_session(_BareResult())
    assert written["patch"]["provider"] is None
    assert written["patch"]["gateway_runtime"]["provider"] is None


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


# ── update_session_model provider persistence (#79536) ──────────────


def test_update_session_model_persists_provider(tmp_path, monkeypatch):
    """update_session_model writes $.model + $.provider into model_config."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="s1", source="cli", model="m0")
    db.update_session_model("s1", "claude-x", provider="custom:feather")
    meta = db.get_session("s1")
    assert meta["model"] == "claude-x"
    config = json.loads(meta["model_config"])
    assert config["model"] == "claude-x"
    assert config["provider"] == "custom:feather"


def test_update_session_model_without_provider_preserves_existing(tmp_path, monkeypatch):
    """Without provider, existing $.provider is left untouched."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="s2", source="cli", model="m0")
    db.update_session_model("s2", "claude-x", provider="custom:feather")
    db.update_session_model("s2", "gpt-5.4")  # no provider
    meta = db.get_session("s2")
    config = json.loads(meta["model_config"])
    assert config["model"] == "gpt-5.4"
    assert config["provider"] == "custom:feather"  # preserved


def test_update_session_model_null_model_config_with_provider(tmp_path, monkeypatch):
    """Provider persistence works when model_config starts as NULL."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="s3", source="cli", model="m0")
    # model_config is NULL at creation — update_session_model must create it
    db.update_session_model("s3", "claude-x", provider="minimax")
    meta = db.get_session("s3")
    config = json.loads(meta["model_config"])
    assert config["model"] == "claude-x"
    assert config["provider"] == "minimax"


# ── session_gateway_runtime billing_provider fallback (#85721) ─────


def test_session_gateway_runtime_falls_back_to_billing_provider():
    """Sessions that never ran /model have only billing_provider."""
    meta = {
        "model": "glm-4.7",
        "model_config": None,
        "billing_provider": "minimax",
    }
    runtime = SessionDB.session_gateway_runtime(meta)
    assert runtime == {"provider": "minimax"}


def test_session_gateway_runtime_billing_provider_bare_bucket_ignored():
    """Bare billing buckets (auto/custom) are not routable — skip them."""
    for bare in ("auto", "custom"):
        meta = {
            "model": "m",
            "model_config": None,
            "billing_provider": bare,
        }
        assert SessionDB.session_gateway_runtime(meta) == {}


def test_session_gateway_runtime_explicit_provider_wins_over_billing():
    """Explicit model_config provider takes precedence over billing_provider."""
    meta = _row(model_config={"provider": "nous"})
    meta["billing_provider"] = "minimax"
    runtime = SessionDB.session_gateway_runtime(meta)
    assert runtime == {"provider": "nous"}


def test_restore_session_model_restores_billing_provider_fallback():
    """End-to-end: _restore_session_model uses billing_provider fallback."""
    stub = _make_stub()
    stub._restore_session_model({
        "model": "glm-4.7",
        "model_config": None,
        "billing_provider": "minimax",
    })
    assert stub.model == "glm-4.7"
    assert stub.provider == "minimax"
