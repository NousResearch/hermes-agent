from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from hermes_cli.bot_catalog import resolve_bot_catalog_entry
from hermes_cli.bot_setup import _plugin_ready, _toolset_ready, bot_setup_status


def _bot_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes" / "profiles" / "bot"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "model:\n  provider: custom\n  default: local/model\n"
        "platform_toolsets:\n  cli: [hermes-cli, web]\n  desktop: [hermes-desktop, web]\n",
        encoding="utf-8",
    )
    (home / "profile.yaml").write_text(
        yaml.safe_dump({
            "provenance": {"kind": "bot-catalog", "catalog_name": "research-analyst"},
            "setup_state": "needs_setup",
            "ui_meta": {"hermes-bots": {"starter_prompt": "keep me"}},
        }),
        encoding="utf-8",
    )
    return home


def test_no_auth_local_runtime_is_accepted_but_first_task_is_required(tmp_path, monkeypatch):
    home = _bot_home(tmp_path, monkeypatch)
    entry = resolve_bot_catalog_entry("research-analyst", include_live=False)

    status = bot_setup_status(entry, runtime={
        "ok": True,
        "provider": "custom:local",
        "model": "local/model",
        "source": "no-key-required",
    })

    assert status["runtime"]["ok"] is True
    assert status["setup_state"] == "needs_setup"
    assert status["starter_prompt"] == entry.profile.starter_prompt
    assert status["first_task"]["status"] == "pending"
    assert status["can_start_first_task"] is True
    assert yaml.safe_load((home / "profile.yaml").read_text())["setup_state"] == "needs_setup"

    from hermes_state import SessionDB

    db = SessionDB(db_path=home / "state.db")
    db.create_session("bot-chat", source="desktop")
    db.set_session_title("bot-chat", "Bot Chat")
    # A greeting or an unfinished assistant row is not proof that the starter task completed.
    db.append_message("bot-chat", "assistant", "Hello!", finish_reason="stop")
    db.append_message("bot-chat", "user", entry.profile.starter_prompt)
    db.append_message("bot-chat", "assistant", "Working on it")
    db.close()

    still_pending = bot_setup_status(entry, runtime={
        "ok": True,
        "provider": "custom:local",
        "model": "local/model",
        "source": "no-key-required",
    })
    assert still_pending["setup_state"] == "needs_setup"
    saved_proof = yaml.safe_load((home / "profile.yaml").read_text())["first_task"]
    assert saved_proof["request_message_id"] is not None

    db = SessionDB(db_path=home / "state.db")
    db.append_message("bot-chat", "assistant", "Here is the completed research brief.", finish_reason="stop")
    db.close()

    ready = bot_setup_status(entry, runtime={
        "ok": True,
        "provider": "custom:local",
        "model": "local/model",
        "source": "no-key-required",
    })
    assert ready["setup_state"] == "ready"
    assert ready["starter_prompt"] is None
    assert ready["can_activate_routines"] is True


def test_enabled_toolset_requires_runtime_exposable_tools(tmp_path, monkeypatch):
    _bot_home(tmp_path, monkeypatch)
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **_kwargs: [])

    ready, detail = _toolset_ready("web")

    assert ready is False
    assert "runtime" in str(detail).lower()


def test_plugin_requirement_uses_canonical_list_plugins_key(monkeypatch):
    rows = [{"key": "web/ddgs", "name": "ddgs", "enabled": True, "error": None}]
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_manager",
        lambda: SimpleNamespace(list_plugins=lambda: rows),
    )

    assert _plugin_ready("web/ddgs") == (True, None)
    assert _plugin_ready("ddgs")[0] is False


def test_existing_codex_sign_in_is_reported_as_reused(tmp_path, monkeypatch):
    _bot_home(tmp_path, monkeypatch)
    entry = resolve_bot_catalog_entry("research-analyst", include_live=False)

    status = bot_setup_status(entry, runtime={
        "ok": True,
        "provider": "openai-codex",
        "model": "gpt-5-codex",
        "source": "credential_pool",
    })

    assert status["runtime"]["reused_sign_in"] is True
    assert status["runtime"]["action"] is None
    assert status["can_start_first_task"] is True
    assert status["setup_state"] == "needs_setup"


def test_runtime_failure_never_becomes_ready(tmp_path, monkeypatch):
    _bot_home(tmp_path, monkeypatch)
    entry = resolve_bot_catalog_entry("research-analyst", include_live=False)

    status = bot_setup_status(entry, runtime={"ok": False, "error": "No usable credentials found"})

    assert status["setup_state"] == "needs_setup"
    assert status["runtime"]["error"] == "No usable credentials found"
    assert status["runtime"]["action"] == "model"
    assert status["can_start_first_task"] is False
