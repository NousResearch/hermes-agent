"""CLI `create --notify-chat`: explicit notify subscription for CLI/subprocess
creates.

The in-gateway `kanban` tool auto-subscribes the calling session off its
ContextVars (HERMES_SESSION_PLATFORM/CHAT_ID). A bare CLI or subprocess create
has no session channel, so the auto path no-ops and a review-required `blocked`
hand-back would arrive with no subscriber to notify. `--notify-chat` /
`--notify-platform` close that gap explicitly.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _created_id(out: str) -> str:
    m = re.search(r"(t_[a-f0-9]+)", out)
    assert m, f"no task id in output: {out!r}"
    return m.group(1)


def test_create_notify_chat_writes_subscription(kanban_home):
    out = kc.run_slash(
        "create 'watched' --notify-chat 12345 --notify-platform discord "
        "--notify-thread 67890"
    )
    tid = _created_id(out)
    with kb.connect_closing() as conn:
        subs = kb.list_notify_subs(conn, tid)
    assert len(subs) == 1
    assert subs[0]["platform"] == "discord"
    assert subs[0]["chat_id"] == "12345"
    assert subs[0]["thread_id"] == "67890"


def test_create_without_notify_flags_writes_no_subscription(kanban_home):
    # A bare CLI create has no session channel; auto-subscribe no-ops and we
    # must not invent a target. Zero rows is the correct, documented behaviour.
    out = kc.run_slash("create 'unwatched'")
    tid = _created_id(out)
    with kb.connect_closing() as conn:
        subs = kb.list_notify_subs(conn, tid)
    assert subs == []


def test_create_half_notify_target_is_rejected(kanban_home):
    # Only one of the pair given -> hard error, not a silent no-op (silently
    # dropping it recreates the exact hand-back-goes-unseen failure the flag
    # exists to prevent).
    out_chat_only = kc.run_slash("create 'x' --notify-chat 12345")
    assert (
        "must be given" in out_chat_only.lower()
        or "together" in out_chat_only.lower()
    )
    out_plat_only = kc.run_slash("create 'y' --notify-platform discord")
    assert (
        "must be given" in out_plat_only.lower()
        or "together" in out_plat_only.lower()
    )
