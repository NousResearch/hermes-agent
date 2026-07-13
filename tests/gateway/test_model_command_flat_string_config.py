"""Regression tests for gateway /model --global persistence when config.yaml
has a flat-string ``model:`` value instead of a nested dict.

Before fix: ``cfg.setdefault("model", {})`` returned the existing string and
the next assignment raised ``TypeError: 'str' object does not support item
assignment``, so every ``/model X --global`` from Telegram/Discord crashed
silently and the user-visible result was "switch failed" with no persist.

After fix: the persist block coerces a scalar ``model:`` into a nested dict
before mutation, so ``--global`` succeeds and the config is rewritten in
the proper ``model: {default: ..., provider: ...}`` form.
"""

import yaml
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.config import GatewayConfig
from gateway.session import SessionSource, SessionStore


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
    return runner


def _make_event(text):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"),
    )


def _fake_switch_result():
    """Build a successful ModelSwitchResult that bypasses real provider resolution."""
    from hermes_cli.model_switch import ModelSwitchResult

    return ModelSwitchResult(
        success=True,
        new_model="gpt-5.5",
        target_provider="openrouter",
        provider_changed=True,
        api_key="sk-test",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
        provider_label="OpenRouter",
        is_global=True,
    )


def _setup_isolated_home(tmp_path, monkeypatch, model_yaml_value):
    """Write a config.yaml with the given ``model:`` value and stub the heavy bits."""
    import gateway.run as gateway_run

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    cfg_path = hermes_home / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"model": model_yaml_value, "providers": {}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **kw: _fake_switch_result(),
    )
    # save_config writes to ``get_hermes_home() / config.yaml`` — point it here.
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: hermes_home)
    return cfg_path


@pytest.mark.asyncio
async def test_model_global_persists_when_config_has_flat_string_model(tmp_path, monkeypatch):
    """Regression: ``model: deepseek-v4-flash`` (flat string) used to crash
    the gateway ``/model X --global`` persist branch with TypeError. After
    the fix, the flat string is coerced to ``{"default": ...}`` and the new
    model+provider are persisted on top.
    """
    cfg_path = _setup_isolated_home(tmp_path, monkeypatch, "deepseek-v4-flash")

    result = await _make_runner()._handle_model_command(
        _make_event("/model gpt-5.5 --global")
    )

    # Sanity: the handler returned a success-looking message (not a crash log).
    assert result is not None
    assert "gpt-5.5" in result

    # The persist block must have rewritten config.yaml as a nested dict.
    written = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert isinstance(written["model"], dict), (
        "model: should be coerced to a dict, got %r" % (written["model"],)
    )
    assert written["model"]["default"] == "gpt-5.5"
    assert written["model"]["provider"] == "openrouter"
    assert "base_url" not in written["model"]


@pytest.mark.asyncio
async def test_model_global_persists_when_config_has_missing_model(tmp_path, monkeypatch):
    """Companion case: ``model:`` key absent entirely. setdefault would have
    worked here, but the coercion branch also has to handle this cleanly.
    """
    import gateway.run as gateway_run

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    cfg_path = hermes_home / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"providers": {}}), encoding="utf-8")

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **kw: _fake_switch_result(),
    )
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: hermes_home)

    result = await _make_runner()._handle_model_command(
        _make_event("/model gpt-5.5 --global")
    )

    assert result is not None
    written = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert isinstance(written["model"], dict)
    assert written["model"]["default"] == "gpt-5.5"
    assert written["model"]["provider"] == "openrouter"


@pytest.mark.asyncio
async def test_model_global_persists_when_config_has_proper_dict_model(tmp_path, monkeypatch):
    """Already-correct nested dict must still work — no regression on the
    common case.
    """
    cfg_path = _setup_isolated_home(
        tmp_path,
        monkeypatch,
        {"default": "old-model", "provider": "openai-codex"},
    )

    result = await _make_runner()._handle_model_command(
        _make_event("/model gpt-5.5 --global")
    )

    assert result is not None
    written = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert written["model"]["default"] == "gpt-5.5"
    assert written["model"]["provider"] == "openrouter"


@pytest.mark.asyncio
async def test_model_no_flag_persists_by_default(tmp_path, monkeypatch):
    """A plain ``/model X`` (no --global) now persists to config.yaml.

    This is the user-facing fix: switching models in one session survives
    into the next without re-typing the switch every time.
    """
    cfg_path = _setup_isolated_home(
        tmp_path,
        monkeypatch,
        {"default": "old-model", "provider": "openai-codex"},
    )

    result = await _make_runner()._handle_model_command(
        _make_event("/model gpt-5.5")
    )

    assert result is not None
    assert "gpt-5.5" in result
    written = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert written["model"]["default"] == "gpt-5.5"


@pytest.mark.asyncio
async def test_model_session_flag_does_not_persist(tmp_path, monkeypatch):
    """``/model X --session`` opts out of persistence even under the new default."""
    cfg_path = _setup_isolated_home(
        tmp_path,
        monkeypatch,
        {"default": "old-model", "provider": "openai-codex"},
    )

    result = await _make_runner()._handle_model_command(
        _make_event("/model gpt-5.5 --session")
    )

    assert result is not None
    assert "gpt-5.5" in result
    written = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    # Config untouched — the session override is in-memory only.
    assert written["model"]["default"] == "old-model"


@pytest.mark.asyncio
async def test_model_reset_clears_session_override_without_switching_global(
    tmp_path, monkeypatch
):
    cfg_path = _setup_isolated_home(
        tmp_path,
        monkeypatch,
        {"default": "global-model", "provider": "openai-api"},
    )
    runner = _make_runner()
    event = _make_event("/model reset")
    session_key = runner._session_key_for_source(event.source)
    runner._session_model_overrides[session_key] = {
        "model": "session-model",
        "provider": "openai-codex",
        "api_mode": "codex_responses",
        "api_key": "runtime-only",
    }

    result = await runner._handle_model_command(event)

    assert session_key not in runner._session_model_overrides
    assert "cleared" in result.lower()
    written = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert written["model"]["default"] == "global-model"
    assert written["model"]["provider"] == "openai-api"


@pytest.mark.asyncio
async def test_model_reset_persistence_failure_never_claims_success(
    tmp_path, monkeypatch
):
    _setup_isolated_home(
        tmp_path,
        monkeypatch,
        {"default": "global-model", "provider": "openai-api"},
    )
    runner = _make_runner()
    event = _make_event("/model reset")
    session_key = runner._session_key_for_source(event.source)
    original = {
        "model": "session-model",
        "provider": "openai-codex",
        "api_mode": "codex_responses",
        "api_key": "runtime-only",
    }
    runner._session_model_overrides[session_key] = dict(original)
    runner.session_store = SimpleNamespace(
        clear_model_route_override=MagicMock(
            side_effect=OSError("atomic route save failed")
        )
    )

    result = await runner._handle_model_command(event)

    assert "was not cleared" in result
    assert "No route change was made" in result
    assert "cleared; using" not in result
    assert runner._session_model_overrides[session_key] == original


@pytest.mark.asyncio
async def test_model_reset_db_commit_survives_json_mirror_failure(
    tmp_path, monkeypatch, caplog
):
    from hermes_state import SessionDB

    _setup_isolated_home(
        tmp_path,
        monkeypatch,
        {"default": "global-model", "provider": "openai-api"},
    )
    db = SessionDB(tmp_path / "state.db")
    monkeypatch.setattr("hermes_state.SessionDB", lambda: db)
    sessions_dir = tmp_path / "gateway-sessions"
    store = SessionStore(sessions_dir, GatewayConfig())
    runner = _make_runner()
    event = _make_event("/model reset")
    entry = store.get_or_create_session(event.source)
    identity = {
        "model": "session-model",
        "provider": "openai-codex",
        "api_mode": "codex_responses",
    }
    with store._lock:
        entry.model_override_identity = dict(identity)
        entry.model_override = dict(identity)
        store._save()
    runner.session_store = store
    runner._session_model_overrides[entry.session_key] = {
        **identity,
        "api_key": "runtime-only",
    }
    monkeypatch.setattr(
        store,
        "_save_sessions_json",
        MagicMock(side_effect=OSError("mirror is read-only")),
    )

    result = await runner._handle_model_command(event)

    assert "cleared; using" in result
    assert entry.session_key not in runner._session_model_overrides
    assert entry.model_override_identity is None
    assert entry.model_override is None
    assert "after state.db commit" in caplog.text

    restarted = SessionStore(sessions_dir, GatewayConfig())
    restarted._ensure_loaded()
    durable = restarted.entry_for(entry.session_key)
    assert durable.model_override_identity is None
    assert durable.model_override is None
    db.close()


@pytest.mark.asyncio
async def test_model_reset_db_failure_rolls_back_memory_and_reports_failure(
    tmp_path, monkeypatch
):
    from hermes_state import SessionDB

    _setup_isolated_home(
        tmp_path,
        monkeypatch,
        {"default": "global-model", "provider": "openai-api"},
    )
    db = SessionDB(tmp_path / "state.db")
    monkeypatch.setattr("hermes_state.SessionDB", lambda: db)
    sessions_dir = tmp_path / "gateway-sessions"
    store = SessionStore(sessions_dir, GatewayConfig())
    runner = _make_runner()
    event = _make_event("/model reset")
    entry = store.get_or_create_session(event.source)
    identity = {
        "model": "session-model",
        "provider": "openai-codex",
        "api_mode": "codex_responses",
    }
    with store._lock:
        entry.model_override_identity = dict(identity)
        entry.model_override = dict(identity)
        store._save()
    runtime = {**identity, "api_key": "runtime-only"}
    runner.session_store = store
    runner._session_model_overrides[entry.session_key] = dict(runtime)
    monkeypatch.setattr(
        store._db,
        "replace_gateway_routing_entries",
        MagicMock(side_effect=OSError("primary commit failed")),
    )

    result = await runner._handle_model_command(event)

    assert "was not cleared" in result
    assert "No route change was made" in result
    assert runner._session_model_overrides[entry.session_key] == runtime
    assert entry.model_override_identity == identity
    assert entry.model_override == identity

    restarted = SessionStore(sessions_dir, GatewayConfig())
    restarted._ensure_loaded()
    durable = restarted.entry_for(entry.session_key)
    assert durable.model_override_identity == identity
    assert durable.model_override == {
        "model": identity["model"],
        "provider": identity["provider"],
    }
    db.close()


@pytest.mark.asyncio
async def test_model_switch_confirmation_shows_reasoning_from_config(tmp_path, monkeypatch):
    """The /model confirmation surfaces the effective reasoning effort, read
    from config.yaml (agent.reasoning_effort) when no session override is set.
    """
    cfg_path = _setup_isolated_home(
        tmp_path,
        monkeypatch,
        {"default": "old-model", "provider": "openai-codex"},
    )
    # Add an agent.reasoning_effort to the isolated config the handler reads.
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["agent"] = {"reasoning_effort": "xhigh"}
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    runner = _make_runner()
    runner._session_reasoning_overrides = {}
    result = await runner._handle_model_command(_make_event("/model gpt-5.5"))

    assert result is not None
    assert "Reasoning: xhigh" in result


@pytest.mark.asyncio
async def test_model_switch_confirmation_honors_session_reasoning_override(tmp_path, monkeypatch):
    """A /model switch does NOT clear a /reasoning session override, so the
    confirmation must reflect the override — not the global config value.
    """
    cfg_path = _setup_isolated_home(
        tmp_path,
        monkeypatch,
        {"default": "old-model", "provider": "openai-codex"},
    )
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["agent"] = {"reasoning_effort": "low"}  # global default
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    runner = _make_runner()
    # Seed a session override keyed to the event's session — high beats the
    # config's "low".
    ev = _make_event("/model gpt-5.5")
    session_key = runner._session_key_for_source(ev.source)
    runner._session_reasoning_overrides = {session_key: {"enabled": True, "effort": "high"}}

    result = await runner._handle_model_command(ev)

    assert result is not None
    assert "Reasoning: high" in result
    assert "Reasoning: low" not in result
