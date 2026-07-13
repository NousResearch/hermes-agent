"""Manual reset preserves deliberate route preferences, and only those."""

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource, SessionStore, build_session_key


MODEL_IDENTITY = {
    "model": "gpt-5.6-sol",
    "provider": "openai-codex",
    "api_mode": "codex_responses",
}
REASONING_NONE = {"enabled": False, "effort": "none"}


def _source(user_id="u1", chat_id="c1"):
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id=user_id,
        chat_id=chat_id,
        chat_type="dm",
    )


@pytest.fixture
def store(tmp_path, monkeypatch):
    def _no_db():
        raise RuntimeError("SQLite disabled in focused routing test")

    import hermes_state

    monkeypatch.setattr(hermes_state, "SessionDB", _no_db)
    return SessionStore(tmp_path / "sessions", GatewayConfig())


def _seed_preferences(store, source=None):
    entry = store.get_or_create_session(source or _source())
    with store._lock:
        entry.model_override_identity = dict(MODEL_IDENTITY)
        entry.reasoning_override = dict(REASONING_NONE)
        entry.last_served_identity = {
            "model": "temporary-fallback",
            "provider": "openrouter",
        }
        store._save()
    return entry


def _write_reset_policy(home, value):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump(
            {"session_reset": {"preserve_route_preferences_on_manual_reset": value}}
        ),
        encoding="utf-8",
    )


def test_session_store_manual_reset_rotates_transcript_and_preserves_exact_preferences(store):
    old = _seed_preferences(store)

    new = store.reset_session(
        old.session_key,
        preserve_route_preferences=True,
    )

    assert new.session_id != old.session_id
    assert new.is_fresh_reset is True
    assert new.model_override_identity == MODEL_IDENTITY
    assert new.reasoning_override == REASONING_NONE
    assert new.last_served_identity is None
    assert store.load_transcript(new.session_id) == []

    persisted = json.loads((store.sessions_dir / "sessions.json").read_text())
    route_payload = persisted[old.session_key]
    assert route_payload["model_override_identity"] == MODEL_IDENTITY
    assert route_payload["reasoning_override"] == REASONING_NONE
    raw = json.dumps(route_payload)
    assert "api_key" not in raw
    assert "access_token" not in raw


def test_automatic_reset_default_still_clears_route_preferences(store):
    old = _seed_preferences(store)

    new = store.reset_session(old.session_key)

    assert new.model_override_identity is None
    assert new.reasoning_override is None
    assert new.last_served_identity is None


def test_real_sqlite_transcript_stays_on_old_session(tmp_path, monkeypatch):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    monkeypatch.setattr("hermes_state.SessionDB", lambda: db)
    session_store = SessionStore(tmp_path / "sessions", GatewayConfig())
    old = session_store.get_or_create_session(_source())
    db.append_message(old.session_id, "user", "old transcript sentinel")
    old.model_override_identity = dict(MODEL_IDENTITY)
    old.reasoning_override = dict(REASONING_NONE)
    session_store.persist()

    new = session_store.reset_session(
        old.session_key,
        preserve_route_preferences=True,
    )

    assert session_store.load_transcript(old.session_id)[0]["content"] == (
        "old transcript sentinel"
    )
    assert session_store.load_transcript(new.session_id) == []
    assert new.model_override_identity == MODEL_IDENTITY
    assert new.reasoning_override == REASONING_NONE
    db.close()


def test_route_preference_carryover_is_scoped_to_existing_routing_lane(store):
    first = _source(user_id="u1", chat_id="c1")
    second = _source(user_id="u2", chat_id="c2")
    first_entry = store.get_or_create_session(first)
    second_entry = store.get_or_create_session(second)
    assert first_entry.session_key != second_entry.session_key
    first_entry.model_override_identity = dict(MODEL_IDENTITY)
    store.persist()

    store.reset_session(first_entry.session_key, preserve_route_preferences=True)

    assert store.entry_for(first_entry.session_key).model_override_identity == MODEL_IDENTITY
    assert store.entry_for(second_entry.session_key).model_override_identity is None

    shared_a = SessionSource(
        platform=Platform.DISCORD,
        user_id="alice",
        chat_id="thread-1",
        thread_id="thread-1",
        chat_type="thread",
    )
    shared_b = SessionSource(
        platform=Platform.DISCORD,
        user_id="bob",
        chat_id="thread-1",
        thread_id="thread-1",
        chat_type="thread",
    )
    assert build_session_key(shared_a) == build_session_key(shared_b)


@pytest.mark.parametrize(
    ("model_identity", "reasoning", "expected_model", "expected_reasoning"),
    [
        (MODEL_IDENTITY, None, MODEL_IDENTITY, None),
        (None, REASONING_NONE, None, REASONING_NONE),
        (None, None, None, None),
    ],
)
def test_manual_reset_copies_each_preference_independently(
    store, model_identity, reasoning, expected_model, expected_reasoning
):
    old = store.get_or_create_session(_source())
    with store._lock:
        old.model_override_identity = model_identity
        old.reasoning_override = reasoning
        store._save()

    new = store.reset_session(old.session_key, preserve_route_preferences=True)

    assert new.model_override_identity == expected_model
    assert new.reasoning_override == expected_reasoning


def test_explicit_preference_clear_is_not_resurrected(store):
    old = _seed_preferences(store)
    with store._lock:
        old.model_override_identity = None
        old.reasoning_override = None
        store._save()

    new = store.reset_session(old.session_key, preserve_route_preferences=True)

    assert new.model_override_identity is None
    assert new.reasoning_override is None


def test_manual_reset_strips_tampered_secrets_from_model_identity(store):
    old = store.get_or_create_session(_source())
    with store._lock:
        old.model_override_identity = {
            **MODEL_IDENTITY,
            "api_key": "sk-must-not-survive",
            "access_token": "oauth-must-not-survive",
            "base_url": "https://user:pass@example.invalid/v1",
        }
        store._save()

    new = store.reset_session(old.session_key, preserve_route_preferences=True)

    assert new.model_override_identity == MODEL_IDENTITY
    raw = (store.sessions_dir / "sessions.json").read_text(encoding="utf-8")
    assert "sk-must-not-survive" not in raw
    assert "oauth-must-not-survive" not in raw
    assert "user:pass" not in raw


def test_manual_reset_preserves_malformed_identity_fail_closed_across_restart(store):
    old = store.get_or_create_session(_source())
    payload = json.loads((store.sessions_dir / "sessions.json").read_text())
    payload[old.session_key]["model_override_identity"] = {"model": "missing-provider"}
    (store.sessions_dir / "sessions.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    reloaded = SessionStore(store.sessions_dir, GatewayConfig())
    reloaded._ensure_loaded()
    malformed = reloaded.entry_for(old.session_key)
    assert malformed._model_override_identity_invalid is True

    reset = reloaded.reset_session(
        old.session_key, preserve_route_preferences=True
    )
    assert reset._model_override_identity_invalid is True
    persisted = json.loads((store.sessions_dir / "sessions.json").read_text())
    assert persisted[old.session_key]["model_override_identity"] == {}

    restarted = SessionStore(store.sessions_dir, GatewayConfig())
    restarted._ensure_loaded()
    lookup_runner = object.__new__(gateway_run.GatewayRunner)
    lookup_runner.session_store = restarted
    lookup = lookup_runner._persisted_session_route_identity(old.session_key)
    assert lookup.state == "unavailable"


def test_manual_reset_migrates_legacy_only_identity_without_secrets(store):
    old = store.get_or_create_session(_source())
    payload = json.loads((store.sessions_dir / "sessions.json").read_text())
    route = payload[old.session_key]
    route.pop("model_override_identity", None)
    route["model_override"] = {
        "model": "legacy-model",
        "provider": "legacy-provider",
        "api_mode": "chat_completions",
        "base_url": "https://user:password@example.invalid/v1",
        "api_key": "sk-must-not-survive",
        "access_token": "oauth-must-not-survive",
    }
    (store.sessions_dir / "sessions.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    reloaded = SessionStore(store.sessions_dir, GatewayConfig())
    reloaded._ensure_loaded()
    reset = reloaded.reset_session(
        old.session_key, preserve_route_preferences=True
    )

    expected = {
        "model": "legacy-model",
        "provider": "legacy-provider",
        "api_mode": "chat_completions",
    }
    assert reset.model_override_identity == expected
    assert reset.model_override is None
    raw = (store.sessions_dir / "sessions.json").read_text(encoding="utf-8")
    assert "model_override\"" not in raw
    assert "sk-must-not-survive" not in raw
    assert "oauth-must-not-survive" not in raw
    assert "user:password" not in raw

    restarted = SessionStore(store.sessions_dir, GatewayConfig())
    restarted._ensure_loaded()
    durable = restarted.entry_for(old.session_key)
    assert durable.model_override_identity == expected
    assert durable.model_override is None


def _runner_for_manual_reset(store):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {}
    runner.session_store = store
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    runner._pending_model_notes = {}
    runner._last_resolved_model = {}
    runner._queued_events = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._running_agents = {}
    runner._pending_approvals = {}
    runner._background_tasks = set()
    runner._session_db = None
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._invalidate_session_run_generation = MagicMock()
    runner._release_running_agent_state = MagicMock()
    runner._evict_cached_agent = MagicMock()
    runner._clear_session_boundary_security_state = MagicMock()
    runner._reset_notice_session_info = MagicMock(return_value="")
    runner._telegram_topic_new_header = MagicMock(return_value="")
    runner._is_telegram_topic_lane = MagicMock(return_value=False)
    runner._reresolve_model_override_credentials = MagicMock(
        return_value={
            **MODEL_IDENTITY,
            "api_key": "fresh-runtime-secret",
            "base_url": "https://chatgpt.com/backend-api/codex",
        }
    )
    return runner


@pytest.mark.asyncio
async def test_manual_reset_persisted_entry_wins_map_divergence(store, monkeypatch):
    old = _seed_preferences(store)
    runner = _runner_for_manual_reset(store)
    runner._session_model_overrides[old.session_key] = {
        "model": "stale-map-model",
        "provider": "openrouter",
        "api_mode": "chat_completions",
        "api_key": "stale-secret",
    }
    runner._session_reasoning_overrides[old.session_key] = {
        "enabled": True,
        "effort": "low",
    }
    home = store.sessions_dir.parent
    _write_reset_policy(home, True)
    monkeypatch.setattr(gateway_run, "_gateway_config_home", lambda: home)

    event = MessageEvent(text="/new", source=_source(), message_id="m1")
    reply = await runner._handle_reset_command(event)

    new = store.entry_for(old.session_key)
    assert new.session_id != old.session_id
    assert new.model_override_identity == MODEL_IDENTITY
    assert new.reasoning_override == REASONING_NONE
    assert runner._session_model_overrides[old.session_key]["model"] == "gpt-5.6-sol"
    assert runner._session_model_overrides[old.session_key]["api_key"] == "fresh-runtime-secret"
    assert runner._session_reasoning_overrides[old.session_key] == REASONING_NONE
    assert old.session_key not in runner._pending_model_notes
    assert "Preserved explicit model and reasoning preferences" in str(reply)
    assert "/model reset" in str(reply)
    assert "/reasoning reset" in str(reply)


@pytest.mark.asyncio
async def test_manual_reset_preserves_but_marks_unavailable_model_preference(
    store, monkeypatch
):
    old = _seed_preferences(store)
    runner = _runner_for_manual_reset(store)
    runner._reresolve_model_override_credentials.return_value = None
    home = store.sessions_dir.parent
    _write_reset_policy(home, True)
    monkeypatch.setattr(gateway_run, "_gateway_config_home", lambda: home)

    reply = await runner._handle_reset_command(
        MessageEvent(text="/new", source=_source(), message_id="m-unavailable")
    )

    assert store.entry_for(old.session_key).model_override_identity == MODEL_IDENTITY
    assert old.session_key in runner._session_model_override_unavailable
    assert "model preference" in str(reply)
    assert "currently unavailable" in str(reply)


@pytest.mark.asyncio
async def test_manual_reset_surfaces_invalid_persisted_model_preference(
    store, monkeypatch
):
    old = store.get_or_create_session(_source())
    with store._lock:
        old.model_override_identity = None
        old._model_override_identity_invalid = True
        store._save()
    runner = _runner_for_manual_reset(store)
    home = store.sessions_dir.parent
    _write_reset_policy(home, True)
    monkeypatch.setattr(gateway_run, "_gateway_config_home", lambda: home)

    reply = await runner._handle_reset_command(
        MessageEvent(text="/new", source=_source(), message_id="m-invalid")
    )

    new = store.entry_for(old.session_key)
    assert new._model_override_identity_invalid is True
    assert "model preference" in str(reply)
    assert "currently unavailable" in str(reply)


@pytest.mark.asyncio
async def test_runtime_kill_switch_false_restores_legacy_clear(store, monkeypatch):
    old = _seed_preferences(store)
    runner = _runner_for_manual_reset(store)
    runner._session_model_overrides[old.session_key] = {
        **MODEL_IDENTITY,
        "api_key": "runtime-only",
    }
    runner._session_reasoning_overrides[old.session_key] = dict(REASONING_NONE)
    home = store.sessions_dir.parent
    _write_reset_policy(home, False)
    monkeypatch.setattr(gateway_run, "_gateway_config_home", lambda: home)

    await runner._handle_reset_command(
        MessageEvent(text="/reset", source=_source(), message_id="m2")
    )

    new = store.entry_for(old.session_key)
    assert new.model_override_identity is None
    assert new.reasoning_override is None
    assert old.session_key not in runner._session_model_overrides
    assert old.session_key not in runner._session_reasoning_overrides


@pytest.mark.asyncio
async def test_unreadable_kill_switch_fails_to_legacy_clear(
    store, monkeypatch, tmp_path
):
    old = _seed_preferences(store)
    runner = _runner_for_manual_reset(store)
    home = tmp_path / "unreadable"
    home.mkdir()
    (home / "config.yaml").mkdir()
    monkeypatch.setattr(gateway_run, "_gateway_config_home", lambda: home)

    await runner._handle_reset_command(
        MessageEvent(text="/new", source=_source(), message_id="m3")
    )

    new = store.entry_for(old.session_key)
    assert new.session_id != old.session_id
    assert new.model_override_identity is None
    assert new.reasoning_override is None


def test_default_config_enables_manual_reset_route_preference_carryover():
    from hermes_cli.config import DEFAULT_CONFIG

    assert (
        DEFAULT_CONFIG["session_reset"][
            "preserve_route_preferences_on_manual_reset"
        ]
        is True
    )


def test_missing_session_reset_section_uses_new_default(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("agent: {}\n", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_gateway_config_home", lambda: tmp_path)

    assert gateway_run.GatewayRunner._preserve_route_preferences_on_manual_reset()


@pytest.mark.parametrize(
    "config",
    [
        [],
        {"session_reset": []},
        {"session_reset": {"preserve_route_preferences_on_manual_reset": 7}},
    ],
)
def test_malformed_runtime_config_fails_to_legacy_false(
    config, tmp_path, monkeypatch
):
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_gateway_config_home", lambda: tmp_path)

    assert not gateway_run.GatewayRunner._preserve_route_preferences_on_manual_reset()


def test_malformed_yaml_defaults_false_and_logs_warning(tmp_path, monkeypatch, caplog):
    sentinel = "TOP_SECRET_SENTINEL"
    (tmp_path / "config.yaml").write_text(
        f"session_reset: [unterminated-{sentinel}", encoding="utf-8"
    )
    monkeypatch.setattr(gateway_run, "_gateway_config_home", lambda: tmp_path)

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        preserved = gateway_run.GatewayRunner._preserve_route_preferences_on_manual_reset()

    assert preserved is False
    assert "config" in caplog.text.lower()
    assert "warning" in caplog.text.lower() or "could not" in caplog.text.lower()
    assert sentinel not in caplog.text


def test_reset_policy_reads_and_parses_one_snapshot(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "session_reset:\n  preserve_route_preferences_on_manual_reset: false\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_gateway_config_home", lambda: tmp_path)
    original = type(config_path).read_text
    reads = 0

    def counted_read(path, *args, **kwargs):
        nonlocal reads
        reads += 1
        return original(path, *args, **kwargs)

    monkeypatch.setattr(type(config_path), "read_text", counted_read)

    assert not gateway_run.GatewayRunner._preserve_route_preferences_on_manual_reset()
    assert reads == 1


def test_gateway_config_parse_warning_does_not_log_source_snippet(
    tmp_path, monkeypatch, caplog
):
    sentinel = "CONFIG_SECRET_SENTINEL"
    (tmp_path / "config.yaml").write_text(
        f"providers: [broken-{sentinel}", encoding="utf-8"
    )
    monkeypatch.setattr(gateway_run, "_gateway_config_home", lambda: tmp_path)
    monkeypatch.setattr(
        "hermes_cli.config.get_config_path", lambda: tmp_path / "different.yaml"
    )

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        assert gateway_run._load_gateway_config() == {}

    assert "error=ParserError" in caplog.text
    assert sentinel not in caplog.text
