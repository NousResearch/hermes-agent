"""Per-session /reasoning overrides must survive gateway restarts.

The gateway persists the *model* half of the session-override pair
(``SessionEntry.model_override``) but not the *reasoning* half, so a restart
silently reverted ``/reasoning high`` to the config default while a ``/model``
switch in the same session survived — asymmetric and surprising.

``SessionEntry.reasoning_override`` closes that gap: the runner writes the
override through on every set/clear and lazily rehydrates it on first use.
Unlike the model override it carries no credential, so nothing is re-resolved.

Covers:
  - the override survives a simulated restart (a second SessionStore instance
    reading the same sessions dir, and a fresh runner rehydrating from it)
  - clearing the override (the /new, /reset and ``/reasoning reset`` path)
    nulls BOTH the in-memory field and the persisted one
  - live in-memory state wins over the persisted value
  - a sessions.json written before this field existed loads without error
  - only the known-safe keys are ever serialized
"""
import json

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import (
    SessionSource,
    SessionStore,
    sanitize_reasoning_override,
)

OVERRIDE = {"enabled": True, "effort": "high"}


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


@pytest.fixture
def store_factory(tmp_path, monkeypatch):
    """Build SessionStores over a shared sessions dir, without SQLite."""

    def _raise():
        raise RuntimeError("SQLite disabled in test")

    import hermes_state

    monkeypatch.setattr(hermes_state, "SessionDB", _raise)

    def _make() -> SessionStore:
        store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig())
        assert store._db is None
        return store

    return _make


def _sessions_json(tmp_path) -> dict:
    return json.loads((tmp_path / "sessions.json").read_text(encoding="utf-8"))


def _make_runner(store):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._session_reasoning_overrides = {}
    runner.session_store = store
    return runner


def test_override_persists_and_survives_restart(store_factory):
    store = store_factory()
    entry = store.get_or_create_session(_make_source())
    session_key = entry.session_key

    store.set_reasoning_override(session_key, OVERRIDE)

    # Simulated restart: a brand-new store instance reads the same dir.
    store2 = store_factory()
    assert store2.get_reasoning_override(session_key) == OVERRIDE


def test_runner_write_through_survives_restart(store_factory):
    """The runner's set/read pair round-trips through disk end to end."""
    store = store_factory()
    entry = store.get_or_create_session(_make_source())
    session_key = entry.session_key

    _make_runner(store)._set_session_reasoning_override(session_key, OVERRIDE)

    # Simulated restart: fresh store + fresh runner with no in-memory state.
    runner = _make_runner(store_factory())
    runner._rehydrate_session_reasoning_override(session_key)

    assert runner._session_reasoning_overrides[session_key] == OVERRIDE


def test_clearing_nulls_memory_and_disk(store_factory, tmp_path):
    """The clear path (/new, /reset, /reasoning reset) must drop both copies."""
    store = store_factory()
    entry = store.get_or_create_session(_make_source())
    session_key = entry.session_key

    runner = _make_runner(store)
    runner._set_session_reasoning_override(session_key, OVERRIDE)
    assert store.get_reasoning_override(session_key) == OVERRIDE

    runner._set_session_reasoning_override(session_key, None)

    assert session_key not in runner._session_reasoning_overrides
    assert store.get_reasoning_override(session_key) is None
    # And a post-restart rehydrate cannot resurrect it.
    assert store_factory().get_reasoning_override(session_key) is None
    assert "reasoning_override" not in _sessions_json(tmp_path)[session_key]


def test_expiry_finalization_drops_the_persisted_override(store_factory):
    """Session finalization is a conversation boundary — drop the override."""
    store = store_factory()
    entry = store.get_or_create_session(_make_source())
    store.set_reasoning_override(entry.session_key, OVERRIDE)

    store.set_expiry_finalized(entry)

    assert store.get_reasoning_override(entry.session_key) is None


def test_live_in_memory_override_wins_over_persisted(store_factory):
    store = store_factory()
    entry = store.get_or_create_session(_make_source())
    session_key = entry.session_key
    store.set_reasoning_override(session_key, {"enabled": True, "effort": "low"})

    runner = _make_runner(store)
    runner._session_reasoning_overrides[session_key] = OVERRIDE
    runner._rehydrate_session_reasoning_override(session_key)

    assert runner._session_reasoning_overrides[session_key] == OVERRIDE


def test_rehydrate_is_a_noop_with_nothing_persisted(store_factory):
    store = store_factory()
    session_key = store.get_or_create_session(_make_source()).session_key

    runner = _make_runner(store)
    runner._rehydrate_session_reasoning_override(session_key)

    assert session_key not in runner._session_reasoning_overrides


def test_resolve_reads_the_persisted_override_after_restart(
    store_factory, tmp_path, monkeypatch
):
    """The real read path — not just the rehydrate helper — must see disk.

    ``_resolve_session_reasoning_config`` is what every turn calls to decide
    the effort actually sent to the model. Wiring persistence into the store
    but not into this resolver would leave the restart bug fully intact while
    the round-trip tests stayed green.
    """
    import gateway.run as gateway_run

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "agent:\n  reasoning_effort: low\n", encoding="utf-8"
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

    store = store_factory()
    source = _make_source()
    session_key = store.get_or_create_session(source).session_key
    _make_runner(store)._set_session_reasoning_override(session_key, OVERRIDE)

    # Simulated restart: fresh store + fresh runner, no in-memory override.
    runner = _make_runner(store_factory())
    resolved = runner._resolve_session_reasoning_config(session_key=session_key)

    # The config default is "low"; the persisted session override must win.
    assert resolved == OVERRIDE


def test_resolve_falls_back_to_config_when_nothing_persisted(
    store_factory, tmp_path, monkeypatch
):
    """The mirror of the test above — no persisted override, no interference."""
    import gateway.run as gateway_run

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "agent:\n  reasoning_effort: low\n", encoding="utf-8"
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

    store = store_factory()
    session_key = store.get_or_create_session(_make_source()).session_key

    runner = _make_runner(store)
    assert runner._resolve_session_reasoning_config(session_key=session_key) == {
        "enabled": True,
        "effort": "low",
    }


def test_legacy_sessions_json_without_the_key_loads(store_factory, tmp_path):
    """A sessions.json written before this field existed must load clean."""
    store = store_factory()
    session_key = store.get_or_create_session(_make_source()).session_key

    raw = _sessions_json(tmp_path)
    assert "reasoning_override" not in raw[session_key]
    # Belt and braces: strip it even if a future default starts emitting one.
    raw[session_key].pop("reasoning_override", None)
    (tmp_path / "sessions.json").write_text(json.dumps(raw), encoding="utf-8")

    reloaded = store_factory()
    assert reloaded.get_reasoning_override(session_key) is None
    assert reloaded.get_or_create_session(_make_source()).session_key == session_key


def test_only_safe_keys_are_serialized(store_factory, tmp_path):
    """Nothing outside {enabled, effort} may reach sessions.json."""
    store = store_factory()
    session_key = store.get_or_create_session(_make_source()).session_key

    store.set_reasoning_override(
        session_key,
        {"enabled": True, "effort": "high", "api_key": "sk-should-never-persist"},
    )

    persisted = _sessions_json(tmp_path)[session_key]["reasoning_override"]
    assert persisted == OVERRIDE
    assert "sk-should-never-persist" not in (tmp_path / "sessions.json").read_text(
        encoding="utf-8"
    )


def test_sanitize_reasoning_override():
    assert sanitize_reasoning_override(None) is None
    assert sanitize_reasoning_override({}) is None
    assert sanitize_reasoning_override("high") is None  # type: ignore[arg-type]
    assert sanitize_reasoning_override({"api_key": "sk-x"}) is None
    assert sanitize_reasoning_override(OVERRIDE) == OVERRIDE
    # "reasoning disabled for this session" is a real override, not an absence.
    assert sanitize_reasoning_override({"enabled": False}) == {"enabled": False}
    assert sanitize_reasoning_override({"enabled": True, "effort": ""}) == {
        "enabled": True
    }
