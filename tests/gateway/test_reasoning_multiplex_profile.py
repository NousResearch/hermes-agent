"""Routed multiplex /reasoning and /fast must use the routed profile config.

Regression for #87939: slash-command dispatch stays in the process/default
profile scope, so status reads and --global writes hit the default
config.yaml instead of the profile that actually runs the turn.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="/reasoning", platform=Platform.TELEGRAM, user_id="123", chat_id="routed-chat"):
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _make_runner(profile_home):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner.config = SimpleNamespace(multiplex_profiles=True)
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._session_reasoning_overrides = {}
    runner._session_model_overrides = {}
    runner._show_reasoning = False
    runner._service_tier = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    runner._session_db = None
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    runner._resolve_profile_home_for_source = lambda _source: profile_home
    return runner


def _write_homes(tmp_path):
    default_home = tmp_path / "default"
    routed_home = tmp_path / "profiles" / "beta"
    default_home.mkdir()
    routed_home.mkdir(parents=True)
    (default_home / "config.yaml").write_text(
        "agent:\n  reasoning_effort: medium\n  service_tier: normal\n",
        encoding="utf-8",
    )
    (routed_home / "config.yaml").write_text(
        "agent:\n  reasoning_effort: none\n  service_tier: normal\n",
        encoding="utf-8",
    )
    return default_home, routed_home


def _read_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_reasoning_status_reads_routed_profile_config(tmp_path, monkeypatch):
    default_home, routed_home = _write_homes(tmp_path)
    monkeypatch.setattr(gateway_run, "_hermes_home", default_home)
    default_before = (default_home / "config.yaml").read_bytes()

    runner = _make_runner(routed_home)
    result = await runner._handle_reasoning_command(_make_event("/reasoning"))

    assert result is not None
    assert "`none" in result
    assert "**Effort:**" in result
    assert (default_home / "config.yaml").read_bytes() == default_before


@pytest.mark.asyncio
async def test_reasoning_global_write_updates_only_routed_profile(tmp_path, monkeypatch):
    default_home, routed_home = _write_homes(tmp_path)
    monkeypatch.setattr(gateway_run, "_hermes_home", default_home)
    default_before = (default_home / "config.yaml").read_bytes()

    runner = _make_runner(routed_home)
    result = await runner._handle_reasoning_command(_make_event("/reasoning high --global"))

    assert result is not None
    assert _read_yaml(routed_home / "config.yaml")["agent"]["reasoning_effort"] == "high"
    assert (default_home / "config.yaml").read_bytes() == default_before


@pytest.mark.asyncio
async def test_reasoning_picker_callback_writes_routed_profile(tmp_path, monkeypatch):
    default_home, routed_home = _write_homes(tmp_path)
    monkeypatch.setattr(gateway_run, "_hermes_home", default_home)
    default_before = (default_home / "config.yaml").read_bytes()

    captured = {}

    async def _capture_picker(*_args, on_choice_selected=None, **_kwargs):
        captured["on_choice"] = on_choice_selected
        return True

    runner = _make_runner(routed_home)
    runner._try_send_choice_picker = _capture_picker

    result = await runner._handle_reasoning_command(_make_event("/reasoning"))
    assert result is None
    assert captured["on_choice"] is not None

    await captured["on_choice"]("routed-chat", "show")

    routed = _read_yaml(routed_home / "config.yaml")
    assert routed["display"]["platforms"]["telegram"]["show_reasoning"] is True
    assert (default_home / "config.yaml").read_bytes() == default_before


@pytest.mark.asyncio
async def test_fast_global_write_updates_only_routed_profile(tmp_path, monkeypatch):
    default_home, routed_home = _write_homes(tmp_path)
    monkeypatch.setattr(gateway_run, "_hermes_home", default_home)
    default_before = (default_home / "config.yaml").read_bytes()
    monkeypatch.setattr(
        "hermes_cli.models.model_supports_fast_mode",
        lambda _model: True,
    )

    runner = _make_runner(routed_home)
    result = await runner._handle_fast_command(_make_event("/fast on --global"))

    assert result is not None
    assert _read_yaml(routed_home / "config.yaml")["agent"]["service_tier"] == "fast"
    assert (default_home / "config.yaml").read_bytes() == default_before

    follow = await runner._handle_fast_command(_make_event("/fast"))
    assert follow is not None
    assert "fast" in follow.lower() or "priority" in follow.lower()


@pytest.mark.asyncio
async def test_reasoning_global_then_status_stays_on_routed_profile(tmp_path, monkeypatch):
    default_home, routed_home = _write_homes(tmp_path)
    monkeypatch.setattr(gateway_run, "_hermes_home", default_home)
    default_before = (default_home / "config.yaml").read_bytes()

    runner = _make_runner(routed_home)
    await runner._handle_reasoning_command(_make_event("/reasoning high --global"))
    result = await runner._handle_reasoning_command(_make_event("/reasoning"))

    assert result is not None
    assert "`high`" in result
    assert (default_home / "config.yaml").read_bytes() == default_before
