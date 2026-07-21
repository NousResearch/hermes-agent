"""gateway.persist_model_override: false makes /model session-only.

When ``gateway.persist_model_override`` is false, an in-chat /model switch
must still take effect for the current process lifetime (the in-memory
``_session_model_overrides`` map is updated by the caller), but the override
is never written to the session store and therefore never rehydrated after
a restart. This lets operators who manage models exclusively via config.yaml
/ `hermes model` (SSH) avoid being silently pinned to a stale in-chat
override.

Covers:
  - set_model_override(non-None) is a no-op on disk when the flag is off
  - set_model_override(None) (clearing) still works even with the flag off,
    so an operator can wipe a previously-persisted override after flipping
    the flag
  - the runner rehydrate path skips rehydration when the store has the flag
    off, even if a persisted override exists on disk (from before the flag
    was flipped)
  - default behavior (flag True) is unchanged by these tests
"""
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionEntry, SessionSource, SessionStore

OVERRIDE = {
    "model": "gpt-5o",
    "provider": "openai",
    "api_key": "«redacted:sk-…»",
    "base_url": "https://api.openai.example/v1",
    "api_mode": "responses",
}


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _no_persist_config() -> GatewayConfig:
    cfg = GatewayConfig()
    cfg.persist_model_override = False
    return cfg


@pytest.fixture
def store_factory(tmp_path, monkeypatch):
    """Build SessionStores over a shared sessions dir, without SQLite."""

    def _raise():
        raise RuntimeError("SQLite disabled in test")

    import hermes_state

    monkeypatch.setattr(hermes_state, "SessionDB", _raise)

    def _make(config: GatewayConfig = None) -> SessionStore:
        store = SessionStore(
            sessions_dir=tmp_path,
            config=config if config is not None else GatewayConfig(),
        )
        assert store._db is None
        return store

    return _make


def test_flag_off_set_override_is_not_persisted(store_factory):
    """With persist_model_override=False, set_model_override(non-None) is
    skipped on the persistence side — get_model_override returns None."""
    store = store_factory(config=_no_persist_config())
    entry = store.get_or_create_session(_make_source())
    session_key = entry.session_key

    store.set_model_override(session_key, OVERRIDE)

    assert store.get_model_override(session_key) is None


def test_flag_off_override_does_not_survive_restart(store_factory):
    """A second store instance (simulated restart) with the flag off must
    not see the override that a flag-off store 'set' earlier."""
    store = store_factory(config=_no_persist_config())
    entry = store.get_or_create_session(_make_source())
    session_key = entry.session_key
    store.set_model_override(session_key, OVERRIDE)

    store2 = store_factory(config=_no_persist_config())
    assert store2.get_model_override(session_key) is None


def test_flag_off_clear_still_works_to_wipe_prior_persisted(store_factory):
    """An operator may flip the flag off and then clear a pre-existing
    persisted override. set_model_override(None) must still write through
    even when persist_model_override is False."""
    # First: persist an override with the flag ON (default).
    store_persist = store_factory(config=GatewayConfig())
    entry = store_persist.get_or_create_session(_make_source())
    session_key = entry.session_key
    store_persist.set_model_override(session_key, OVERRIDE)
    assert store_persist.get_model_override(session_key) is not None

    # Now flip the flag off on a fresh store and clear the override.
    store_clear = store_factory(config=_no_persist_config())
    store_clear.set_model_override(session_key, None)

    # A fresh store with the flag off must see nothing.
    store_check = store_factory(config=_no_persist_config())
    assert store_check.get_model_override(session_key) is None


def test_flag_off_runner_does_not_rehydrate_persisted_override(store_factory):
    """Even if an override was persisted to disk before the flag was flipped
    off, the runner rehydrate path must skip it when the store has the flag
    off — sessions fall back to config.yaml's model.default."""
    # Persist with flag ON so something is on disk.
    store_persist = store_factory(config=GatewayConfig())
    entry = store_persist.get_or_create_session(_make_source())
    session_key = entry.session_key
    store_persist.set_model_override(session_key, OVERRIDE)

    # Fresh store with flag OFF, then a runner with an empty in-memory map.
    from gateway.run import GatewayRunner

    store = store_factory(config=_no_persist_config())
    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {}
    runner.session_store = store

    with patch(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        return_value={
            "api_key": "«redacted:sk-…»",
            "api_mode": "responses",
            "base_url": "https://api.openai.example/v1",
            "provider": "openai",
        },
    ):
        runner._rehydrate_session_model_override(session_key)

    # Nothing rehydrated — sessions will use config defaults.
    assert runner._session_model_overrides == {}


def test_flag_on_default_still_persists_and_rehydrates(store_factory):
    """Sanity: default behavior (flag True) is unchanged by this feature —
    the override persists and rehydrates as before."""
    store = store_factory(config=GatewayConfig())
    entry = store.get_or_create_session(_make_source())
    session_key = entry.session_key
    store.set_model_override(session_key, OVERRIDE)

    store2 = store_factory(config=GatewayConfig())
    persisted = store2.get_model_override(session_key)
    assert persisted == {
        "model": "gpt-5o",
        "provider": "openai",
        "base_url": "https://api.openai.example/v1",
    }


def test_flag_off_does_not_block_live_in_memory_override(store_factory):
    """The feature only gates persistence + rehydration. The in-memory
    _session_model_overrides map is owned by the caller (slash command path),
    so a live override still applies during the current process lifetime
    even when persistence is off. This test documents that contract by
    showing that a manually-set in-memory override is returned by
    _apply_session_model_override regardless of the flag."""
    from gateway.run import GatewayRunner

    store = store_factory(config=_no_persist_config())
    entry = store.get_or_create_session(_make_source())
    session_key = entry.session_key

    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {}
    runner.session_store = store

    # Caller simulates the slash-command path: set in-memory directly.
    runner._session_model_overrides[session_key] = {
        "model": "live-model",
        "provider": "openai",
        "api_key": "«redacted:sk-…»",
        "base_url": "https://api.openai.example/v1",
        "api_mode": "responses",
    }

    base_model = "config-default-model"
    base_kwargs = {"provider": "config-default-provider"}
    applied_model, applied_kwargs = runner._apply_session_model_override(
        session_key, base_model, base_kwargs
    )
    assert applied_model == "live-model"
    assert applied_kwargs["provider"] == "openai"


def test_gateway_config_round_trips_flag():
    """to_dict / from_dict must preserve persist_model_override."""
    cfg = GatewayConfig()
    cfg.persist_model_override = False
    as_dict = cfg.to_dict()
    assert as_dict["persist_model_override"] is False

    restored = GatewayConfig.from_dict(as_dict)
    assert restored.persist_model_override is False

    # Default when key absent.
    default_restored = GatewayConfig.from_dict({})
    assert default_restored.persist_model_override is True
