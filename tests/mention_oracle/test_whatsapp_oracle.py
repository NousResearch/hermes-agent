"""Owner contract Oracle — the WhatsApp/Telegram personal @mention path (2026-11-23).

Y's WhatsApp intake is the ``personal-auto-reply`` plugin (wa-reader ``messages.jsonl`` →
``whatsapp_event`` → ``OwnerCommands`` → frozen prompt → ``send_command_reply``), not a platform
adapter — so the real payload boundary is the plugin itself, loaded here by file location exactly
as the plugin's own test suite does (``reply_fixture`` loader). The only substitutions are
external edges: the reader file and the transport client. Expected values are independent
literals (``CANARY_DM_SECRET``).

Skips with a loud reason when the installed plugin directory is absent; the location is
``PERSONAL_AUTO_REPLY_DIR`` so a checkout can point the Oracle at a staged copy.
"""
import asyncio
import importlib.util
import json
import os
import sqlite3
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

PLUGIN_DIR = Path(os.environ.get(
    "PERSONAL_AUTO_REPLY_DIR",
    "/home/yqecea/.hermes/profiles/yqecea/plugins/personal-auto-reply"))

CANARY_DM_SECRET = "CANARY_DM_SECRET_9481"
GROUP_CHAT = "120363423327214990@g.us"
DM_CHAT = "77001234567@s.whatsapp.net"

pytestmark = pytest.mark.skipif(not PLUGIN_DIR.is_dir(),
                                reason=f"personal-auto-reply plugin not installed at {PLUGIN_DIR}")


@pytest.fixture(scope="module")
def plugin():
    """Load the installed plugin as a package under ``reply_fixture`` (the plugin tests' own
    mechanism), so ``from . import …`` inside its modules resolves."""
    pkg = types.ModuleType("reply_fixture")
    pkg.__path__ = [str(PLUGIN_DIR)]
    sys.modules["reply_fixture"] = pkg
    loaded = {}
    for name in ("policy", "media", "commands", "triage", "engine", "native", "adapter"):
        spec = importlib.util.spec_from_file_location(f"reply_fixture.{name}", PLUGIN_DIR / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return SimpleNamespace(**loaded)


_ROW_SEQ = iter(range(1, 10_000))


def _row(chat, text, **kw):
    """A real-shaped wa-reader row (the payload boundary the plugin consumes).
    Every row gets a distinct id — the trigger filter drops rows sharing it."""
    row = {
        "chat": chat, "id": f"3EB0FAKE{next(_ROW_SEQ):06d}", "ts": time.time(),
        "text": text, "fromMe": True, "isBot": False,
        "addressed": False, "addressed_question": False, "forwarded": False,
        "source": "notify", "reader_instance": "wa-1", "chatName": chat,
        "sender": "77001234567@s.whatsapp.net",
    }
    row.update(kw)
    return row


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── Allowed matrix: @H/@Y/@S trigger rows admit through the real path ──────────

@pytest.mark.parametrize("trigger,profile", [("@Y ping", "yqecea"), ("@S ping", "sqecea"),
                                             ("@H ping", "default")])
def test_group_trigger_row_admits_with_typed_scope(plugin, trigger, profile):
    event = plugin.adapter.whatsapp_event(_row(GROUP_CHAT, trigger))
    assert event["group"] is True
    assert event["scope"] == "group", "group intake must be typed at the adapter boundary"
    parsed = plugin.commands.parse_command(event["text"])
    assert parsed is not None, "a valid @mention must parse as a command"
    assert plugin.commands.PROFILE_FOR[parsed[0]] == profile


def test_dm_trigger_row_typed_dm_scope(plugin, monkeypatch, tmp_path):
    monkeypatch.setattr(plugin.native, "WA", tmp_path / "wa-reader")
    event = plugin.adapter.whatsapp_event(_row(DM_CHAT, "@Y ping"))
    assert event["group"] is False
    assert event["scope"] == "dm"


# ── Canary isolation: group prompt must not contain DM rows ────────────────────

def test_group_command_prompt_excludes_dm_canary(plugin, monkeypatch, tmp_path):
    """The frozen admission payload for a group trigger carries the group's own history,
    an explicit chat_scope=group marker, group metadata, and zero DM-canary bytes."""
    wa_dir = tmp_path / "wa-reader"
    trigger = _row(GROUP_CHAT, "@Y status?")
    _write_jsonl(wa_dir / "messages.jsonl", [
        _row(DM_CHAT, f"private note {CANARY_DM_SECRET}", ts=time.time() - 60),
        _row(GROUP_CHAT, "group earlier message", ts=time.time() - 30, fromMe=False),
        trigger,
    ])
    monkeypatch.setattr(plugin.native, "WA", wa_dir)

    event = plugin.adapter.whatsapp_event(trigger)
    history = asyncio.run(plugin.native.command_context(event))
    rendered = plugin.native.render_command_prompt(event, "@Y status?", history)

    assert CANARY_DM_SECRET not in rendered, (
        "DM chat rows leaked into a group command prompt")
    assert "group earlier message" in rendered, (
        "the group's own history must reach the prompt")
    assert '"chat_scope"' in rendered and '"group"' in rendered, (
        "the frozen route must carry an explicit chat_scope=group marker")
    assert '"chat_name"' in rendered, (
        "group metadata (the multi-user chat's name) must reach the prompt")


def test_dm_command_prompt_keeps_dm_rows(plugin, monkeypatch, tmp_path):
    """Control arm: the canary in the DM trigger's own chat stays visible in a DM turn."""
    wa_dir = tmp_path / "wa-reader"
    trigger = _row(DM_CHAT, "@Y status?")
    _write_jsonl(wa_dir / "messages.jsonl", [
        _row(DM_CHAT, f"private note {CANARY_DM_SECRET}", ts=time.time() - 60),
        trigger,
    ])
    monkeypatch.setattr(plugin.native, "WA", wa_dir)
    event = plugin.adapter.whatsapp_event(trigger)
    history = asyncio.run(plugin.native.command_context(event))
    rendered = plugin.native.render_command_prompt(event, "@Y status?", history)
    assert CANARY_DM_SECRET in rendered
    assert '"chat_scope"' in rendered and '"dm"' in rendered


# ── Replay smoke: the payloads the feature actually failed on (sanitized) ──────

def test_replay_failed_group_trigger_rows(plugin, monkeypatch, tmp_path):
    """Real failed rows, 2026-11-22: `@Y` and `@H` from a WhatsApp group died downstream
    (run blocked on model quota) — the prompt path must still admit and freeze them."""
    wa_dir = tmp_path / "wa-reader"
    rows = [
        _row(GROUP_CHAT, "@Y приём, что там по задачам?", id="AC6FC9A10001"),
        _row(GROUP_CHAT, "@H собери статус по проектам", id="AC5DDDA10002"),
    ]
    _write_jsonl(wa_dir / "messages.jsonl", rows)
    monkeypatch.setattr(plugin.native, "WA", wa_dir)

    for row in rows:
        event = plugin.adapter.whatsapp_event(row)
        assert event["scope"] == "group"
        assert plugin.commands.parse_command(event["text"]) is not None
        history = asyncio.run(plugin.native.command_context(event))
        rendered = plugin.native.render_command_prompt(event, event["text"], history)
        assert row["text"] in rendered and '"chat_scope"' in rendered


# ── Watchdog: mention_received without reply_sent ──────────────────────────────

class _DeadTransport:
    """External edge: the session never answers and no reply is ever sent."""

    async def command_run_results(self, refs):
        return {}


def _store(tmp_path):
    db = sqlite3.connect(tmp_path / "state.sqlite")
    db.row_factory = sqlite3.Row
    return SimpleNamespace(db=db, is_automatic=lambda event: False)


def test_stale_owner_command_logs_mention_unanswered(plugin, tmp_path, caplog):
    """A command admitted but never answered past the timeout raises a structured alert."""
    import logging

    oc = plugin.commands.OwnerCommands(_store(tmp_path), _DeadTransport())
    event = {"platform": "whatsapp", "chat_id": GROUP_CHAT, "id": "3EB0DEAD", "scope": "group",
             "_command": "@Y status?", "ts": time.time() - 9999}
    oc.store.db.execute(
        "INSERT INTO owner_commands(key,event,profile,state,run_ref,created_at,updated_at) "
        "VALUES('k-stale',?, 'yqecea','running','ref-1', ?, ?)",
        (json.dumps(event), time.time() - 9999, time.time() - 9999))
    oc.store.db.commit()

    with caplog.at_level(logging.ERROR):
        asyncio.run(oc.reconcile())
    assert any("mention_unanswered" in rec.message for rec in caplog.records), (
        "an unanswered mention past the timeout must log mention_unanswered")


def test_fresh_owner_command_stays_quiet(plugin, tmp_path, caplog):
    """Control arm: an in-flight command inside the timeout must not alert."""
    import logging

    oc = plugin.commands.OwnerCommands(_store(tmp_path), _DeadTransport())
    event = {"platform": "whatsapp", "chat_id": GROUP_CHAT, "id": "3EB0FRESH", "scope": "group",
             "_command": "@Y status?", "ts": time.time()}
    oc.store.db.execute(
        "INSERT INTO owner_commands(key,event,profile,state,run_ref,created_at,updated_at) "
        "VALUES('k-fresh',?, 'yqecea','running','ref-2', ?, ?)",
        (json.dumps(event), time.time(), time.time()))
    oc.store.db.commit()

    with caplog.at_level(logging.ERROR):
        asyncio.run(oc.reconcile())
    assert not any("mention_unanswered" in rec.message for rec in caplog.records)
