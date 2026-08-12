"""Fork migration: gateway.system_messages custom-text override.

Locks the fork's ``gateway.system_messages.<key>`` override layer grafted onto
upstream's i18n engine (agent/i18n.py) after the upstream/main rebase. Upstream's
own i18n behaviour is covered by tests/agent/test_i18n.py (unchanged, still
passing). Category suppression is a separate feature (see its own test module).
"""
from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import pytest

import agent.i18n as i18n


def _cfg(**system_messages):
    return {"gateway": {"system_messages": system_messages}, "display": {"language": "ru"}}


# ---- custom-text override (gateway.system_messages.<key>) -------------------

def test_override_replaces_catalog_value():
    with patch.object(i18n, "_load_config_dict", return_value=_cfg(draining="СВОЙ {count}")):
        i18n.reset_language_cache()
        assert i18n.t("gateway.draining", count=3) == "СВОЙ 3"
    i18n.reset_language_cache()


def test_override_missing_placeholder_stays_literal():
    with patch.object(i18n, "_load_config_dict", return_value=_cfg(draining="X {count} {bogus}")):
        i18n.reset_language_cache()
        out = i18n.t("gateway.draining", count=1)
        assert "{bogus}" in out and "1" in out  # safe formatter degrades
    i18n.reset_language_cache()


def test_override_compound_missing_placeholders_stay_literal():
    template = "X {count} {missing.attr} {missing[key]}"
    with patch.object(i18n, "_load_config_dict", return_value=_cfg(draining=template)):
        i18n.reset_language_cache()
        assert i18n.t("gateway.draining", count=1) == "X 1 {missing.attr} {missing[key]}"
    i18n.reset_language_cache()


def test_name_autofill_uses_configured_custom_skin(tmp_path, monkeypatch):
    from hermes_cli import skin_engine

    skins_dir = tmp_path / "skins"
    skins_dir.mkdir()
    (skins_dir / "reviewer.yaml").write_text(
        "name: reviewer\nbranding:\n  agent_name: Гермес\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(skin_engine, "get_hermes_home", lambda: tmp_path)

    with patch.object(i18n, "_load_config_dict",
                      return_value={"gateway": {"system_messages": {"draining": "{name} drains {count}"}},
                                    "display": {"language": "ru", "skin": "reviewer"}}):
        i18n.reset_language_cache()
        assert i18n.t("gateway.draining", count=2) == "Гермес drains 2"
    i18n.reset_language_cache()


def test_gateway_system_messages_is_registered_in_default_config():
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["gateway"]["system_messages"] == {}


# ---- current-main semantic integration ------------------------------------

def test_current_main_persistence_recovery_routes_through_i18n():
    from gateway.run import _normalize_empty_agent_response

    result = {
        "failed": True,
        "failure_reason": "session_persistence_failed:disk",
        "error": "disk full",
    }
    with patch("gateway.run.t", return_value="LOCALIZED-PERSISTENCE") as translate:
        assert _normalize_empty_agent_response(result, "") == "LOCALIZED-PERSISTENCE"
    translate.assert_called_once_with("gateway.session_storage_unavailable_disk")


@pytest.mark.asyncio
async def test_current_main_pause_reply_routes_through_i18n():
    from gateway.run import GatewayRunner

    event = SimpleNamespace(get_command_args=lambda: "off")
    with (
        patch("agent.estop.disengage", return_value=True),
        patch("gateway.run.t", return_value="LOCALIZED-RESUME") as translate,
    ):
        result = await GatewayRunner._handle_pause_command(object.__new__(GatewayRunner), event)
    assert result == "LOCALIZED-RESUME"
    translate.assert_called_once_with("gateway.pause_resumed")


@pytest.mark.asyncio
async def test_current_main_heartbeat_unavailable_routes_through_i18n():
    from gateway.slash_commands import GatewaySlashCommandsMixin

    runner = SimpleNamespace()

    async def _get_manager(_event):
        return None, None

    runner._get_heartbeat_manager_for_event = _get_manager
    event = SimpleNamespace(get_command_args=lambda: "", source=None)
    with patch("gateway.slash_commands.t", return_value="LOCALIZED-HEARTBEAT") as translate:
        result = await GatewaySlashCommandsMixin._handle_heartbeat_command(runner, event)
    assert result == "LOCALIZED-HEARTBEAT"
    translate.assert_called_once_with("gateway.heartbeat_unavailable")


@pytest.mark.asyncio
async def test_current_main_goal_gate_reply_routes_through_i18n():
    from gateway.slash_commands import GatewaySlashCommandsMixin

    gate = SimpleNamespace(command="pytest -q", max_retries=3, timeout_seconds=60)
    manager = SimpleNamespace(add_gate=lambda _command: gate)
    runner = SimpleNamespace()

    async def _get_manager(_event):
        return manager, None

    runner._get_goal_manager_for_event = _get_manager
    event = SimpleNamespace(get_command_args=lambda: "gate add pytest -q")
    with patch("gateway.slash_commands.t", return_value="LOCALIZED-GATE") as translate:
        result = await GatewaySlashCommandsMixin._handle_goal_command(runner, event)
    assert result == "LOCALIZED-GATE"
    translate.assert_called_once_with(
        "gateway.goal_gate_added",
        command="pytest -q",
        max_retries=3,
        timeout_seconds=60,
    )


@pytest.mark.asyncio
async def test_current_main_refine_reply_routes_through_i18n():
    from gateway.slash_commands import GatewaySlashCommandsMixin

    runner = SimpleNamespace(_session_key_for_source=lambda _source: None)
    event = SimpleNamespace(get_command_args=lambda: "", source=None)
    with patch("gateway.slash_commands.t", return_value="LOCALIZED-REFINE") as translate:
        result = await GatewaySlashCommandsMixin._handle_refine_command(runner, event)
    assert result == "LOCALIZED-REFINE"
    translate.assert_called_once_with("gateway.refine_unavailable")


def test_current_main_dns_hint_routes_through_i18n():
    from run_agent import AIAgent

    error = OSError(-3, "Temporary failure in name resolution")
    with patch("run_agent.t", return_value="LOCALIZED-OFFLINE") as translate:
        assert AIAgent._summarize_api_error(error) == "LOCALIZED-OFFLINE"
    translate.assert_called_once_with("gateway.provider_unreachable_offline")


def test_current_main_compaction_handoff_final_routes_through_i18n():
    import agent.conversation_loop as loop

    with patch.object(loop, "t", return_value="LOCALIZED-HANDOFF") as translate:
        assert loop._handoff_skip_final_response() == "LOCALIZED-HANDOFF"
    translate.assert_called_once_with("gateway.compaction_handoff_waiting")


# ---- marker-coupling preserved ---------------------------------------------

def test_provider_auth_envelope_stays_english_literal():
    """The raw 'Provider authentication failed: {exc}' envelope must remain a
    literal in run.py so _GATEWAY_PROVIDER_ERROR_SHAPE_RE still rewrites it."""
    src = Path("gateway/run.py").read_text(encoding="utf-8")
    assert 'f"⚠️ Provider authentication failed: {exc}"' in src


# ---- sample net-new keys resolve in Russian --------------------------------

@pytest.mark.parametrize("key,kwargs", [
    ("gateway.long_running", {"minutes": 5, "status_detail": ""}),
    ("gateway.kanban_done", {"tag": "", "task_id": "t1", "title": "X", "handoff": ""}),
    (
        "gateway.codex_gpt55_autoraise_notice",
        {"model": "gpt-5.6-terra", "cap": "272K", "from_pct": 80, "to_pct": 85},
    ),
    ("gateway.subgoal_cleared_many", {"count": 3}),
    ("gateway.tool_guardrail_halted", {"tool": "exec", "code": "E1"}),
])
def test_sample_keys_resolve_russian(key, kwargs):
    i18n.reset_language_cache()
    ru = i18n.t(key, lang="ru", **kwargs)
    en = i18n.t(key, lang="en", **kwargs)
    assert ru and ru != en
    for name in kwargs:
        assert "{" + name + "}" not in ru
