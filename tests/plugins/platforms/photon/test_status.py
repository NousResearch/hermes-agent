"""Photon CLI status state-summary tests."""
from __future__ import annotations

import json
from pathlib import Path

from plugins.platforms.photon import cli
from plugins.platforms.photon.state import PhotonStateStore


def test_status_prints_state_counts(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    store = PhotonStateStore()
    store.load()
    store.record_sent_message("msg-1", chat_key="+1", space_id="space")
    store.record_last_inbound("+1", "inbound-1", space_id="space")
    store.record_reaction_added("space", "inbound-1", "like", "reaction-1")
    store.record_audit(action="send", status="succeeded", chat_key="+1")
    rendered: list[str] = []

    cli._print_state_summary(rendered.append)

    output = "\n".join(rendered)
    assert "state schema        : v1" in output
    assert "1 sent, 1 inbound chats, 1 active reactions, 1 audit entries" in output
    assert str(home / "plugins" / "photon" / "state.json") in output


def test_status_handles_corrupt_state(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    path = home / "plugins" / "photon" / "state.json"
    path.parent.mkdir(parents=True)
    path.write_text("{bad", encoding="utf-8")
    rendered: list[str] = []

    cli._print_state_summary(rendered.append)

    assert any("unavailable/corrupt" in line for line in rendered)


def test_status_does_not_print_stored_message_text(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    path = home / "plugins" / "photon" / "state.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({
            "schema_version": 1,
            "updated_at": "2026-06-30T00:00:00Z",
            "sent_messages": {
                "msg-1": {
                    "chat_key": "+1",
                    "space_id": "space",
                    "sent_at": "2026-06-30T00:00:00Z",
                    "kind": "text",
                    "text": "do not show this",
                }
            },
            "last_inbound_by_chat": {},
            "reactions": {},
            "audit": [],
        }),
        encoding="utf-8",
    )
    rendered: list[str] = []

    cli._print_state_summary(rendered.append)

    output = "\n".join(rendered)
    assert "do not show this" not in output
    assert "msg-1" not in output
