import argparse
import asyncio
import pytest

from pathlib import Path
from types import SimpleNamespace
from hermes_cli import kanban_db as kb
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Allow the kanban notifier path-validator to upload artifacts the
    # tests write under ``tmp_path``. Without this, every artifact-delivery
    # test silently drops files because ``tmp_path`` isn't inside the
    # default ``MEDIA_DELIVERY_SAFE_ROOTS`` cache dirs.
    monkeypatch.setenv("HERMES_MEDIA_ALLOW_DIRS", str(tmp_path))
    kb.init_db()
    return home


def _assert_inherited_notify_sub(subs: list[dict]) -> None:
    assert len(subs) == 1
    assert subs[0]["platform"] == "telegram"
    assert subs[0]["chat_id"] == "chat1"
    assert subs[0]["thread_id"] == "topic1"
    assert subs[0]["user_id"] == "user1"
    assert subs[0]["notifier_profile"] == "default"


def _parse_create(*argv: str) -> argparse.Namespace:
    """Build a real CLI create namespace instead of hand-copying defaults."""
    from hermes_cli import kanban

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    kanban.build_parser(subparsers)
    return parser.parse_args(["kanban", "create", *argv])


def test_cli_create_autosubscribes_current_desktop_session(
    kanban_home, monkeypatch, capsys
):
    """Desktop tool shells must preserve the TUI poller's return address.

    Desktop agent turns commonly create cards through ``hermes kanban create``
    in the terminal tool.  That path must be delivery-equivalent to the
    model-facing ``kanban_create`` tool: stamp the durable session id and add a
    subscription that the Desktop/TUI notification poller can consume.
    """
    from gateway.session_context import reset_session_vars
    from hermes_cli import kanban

    reset_session_vars()
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "desktop")
    monkeypatch.setenv("HERMES_SESSION_ID", "desktop-session-1")
    monkeypatch.setenv("HERMES_SESSION_KEY", "desktop-poller-key-1")
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)

    args = _parse_create("desktop proof", "--assignee", "default", "--json")
    assert kanban._cmd_create(args) == 0
    capsys.readouterr()

    conn = kb.connect()
    try:
        task = kb.list_tasks(conn)[0]
        subs = kb.list_notify_subs(conn, task.id)
    finally:
        conn.close()

    assert task.session_id == "desktop-session-1"
    assert len(subs) == 1
    assert subs[0]["platform"] == "tui"
    assert subs[0]["chat_id"] == "desktop-poller-key-1"
    assert subs[0]["notifier_profile"] == "default"

    # Prove the row is addressed to the identity the live Desktop/TUI poller
    # actually consumes, not merely present in the database.
    conn = kb.connect()
    try:
        kb.complete_task(conn, task.id, summary="desktop delivery proof")
    finally:
        conn.close()
    from tui_gateway.server import _collect_kanban_notifications

    texts = _collect_kanban_notifications(
        {"session_key": "desktop-poller-key-1"}
    )
    assert len(texts) == 1
    assert task.id in texts[0]
    assert "desktop delivery proof" in texts[0]


def test_cli_create_autosubscribes_exact_buzz_thread(
    kanban_home, monkeypatch, capsys
):
    """A Buzz-origin terminal card retains its channel, thread, and owner."""
    from gateway.session_context import reset_session_vars
    from hermes_cli import kanban

    reset_session_vars()
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "buzz")
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "buzz")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "buzz-channel-1")
    monkeypatch.setenv("HERMES_SESSION_THREAD_ID", "buzz-thread-7")
    monkeypatch.setenv("HERMES_SESSION_CHAT_TYPE", "group")
    monkeypatch.setenv("HERMES_SESSION_USER_ID", "synthetic-user")
    monkeypatch.setenv("HERMES_SESSION_ID", "buzz-session-1")
    monkeypatch.setenv("HERMES_PROFILE", "buzzobserver")

    args = _parse_create("buzz proof", "--assignee", "default", "--json")
    assert kanban._cmd_create(args) == 0
    capsys.readouterr()

    conn = kb.connect()
    try:
        task = kb.list_tasks(conn)[0]
        subs = kb.list_notify_subs(conn, task.id)
    finally:
        conn.close()

    assert task.session_id == "buzz-session-1"
    assert len(subs) == 1
    assert subs[0]["platform"] == "buzz"
    assert subs[0]["chat_id"] == "buzz-channel-1"
    assert subs[0]["thread_id"] == "buzz-thread-7"
    assert subs[0]["chat_type"] == "group"
    assert subs[0]["user_id"] == "synthetic-user"
    assert subs[0]["notifier_profile"] == "buzzobserver"










# ---------------------------------------------------------------------------
# Regression: gateway watchers must not double-init the kanban DB.
#
# Both the notifier watcher (`_kanban_notifier_watcher`) and the dispatcher
# tick (`_tick_once_for_board`) used to call `_kb.connect(board=slug)`
# immediately followed by `_kb.init_db(board=slug)`. Since `connect()`
# already runs the schema + idempotent migration on first open per process,
# the explicit `init_db()` was redundant — and worse, `init_db()`
# deliberately busts the per-process cache and re-runs the migration on a
# *second* connection, which races the first.  On legacy DBs this surfaced
# as `duplicate column name: <col>` (now tolerated by
# `_add_column_if_missing`) and intermittent `database is locked` errors
# (issue #21378).
#
# The fix removes the `init_db()` calls in both watchers; this regression
# test pins that behaviour so we don't reintroduce them.
# ---------------------------------------------------------------------------




@pytest.mark.asyncio
async def test_gateway_create_autosubscribes_on_explicit_board(kanban_home):
    """`/kanban --board <slug> create ...` must subscribe on that board.

    The gateway handler currently auto-subscribes after `/kanban create`,
    but the create detection must still work when the shared `--board`
    flag appears before the subcommand, and the subscription must land in
    that board's DB rather than the ambient/default board.
    """
    from gateway.run import GatewayRunner
    from gateway.config import Platform

    kb.create_board("projx")

    runner = object.__new__(GatewayRunner)
    source = SimpleNamespace(
        platform=Platform.TELEGRAM,
        chat_id="chat1",
        chat_type="dm",
        thread_id="20197",
        user_id="u1",
    )
    event = SimpleNamespace(
        text='/kanban --board projx create "hello" --assignee alice',
        source=source,
        message_id="462",
        reply_to_message_id=None,
    )

    out = await GatewayRunner._handle_kanban_command(runner, event)

    assert "subscribed" in out.lower()

    conn = kb.connect(board="projx")
    try:
        subs = kb.list_notify_subs(conn)
        tasks = kb.list_tasks(conn)
    finally:
        conn.close()

    assert [t.title for t in tasks] == ["hello"]
    assert len(subs) == 1
    assert subs[0]["chat_id"] == "chat1"
    assert subs[0]["thread_id"] == "20197"
    assert subs[0]["delivery_metadata"] == {
        "chat_type": "dm",
        "direct_messages_topic_id": "20197",
        "telegram_dm_topic_reply_fallback": True,
        "telegram_reply_to_message_id": "462",
        "thread_id": "20197",
    }

    conn = kb.connect(board="default")
    try:
        assert kb.list_notify_subs(conn) == []
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_notifier_artifact_delivery_skips_missing_files(kanban_home, tmp_path, monkeypatch):
    """Missing artifact paths are silently skipped — they may have been
    referenced by name only. The notifier must not crash and must still
    deliver any artifacts that do exist."""
    import hermes_cli.kanban_db as kb
    from gateway.run import GatewayRunner
    from gateway.config import Platform
    from tools import kanban_tools as kt

    # Allow ``tmp_path`` through the media-delivery safety filter. See the
    # companion test for the full explanation.
    monkeypatch.setenv("HERMES_MEDIA_ALLOW_DIRS", str(tmp_path))

    real_pdf = tmp_path / "real.pdf"
    real_pdf.write_bytes(b"%PDF-fake")

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="worker1")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat1")
    finally:
        conn.close()

    import os
    os.environ["HERMES_KANBAN_TASK"] = tid
    try:
        kt._handle_complete({
            "summary": "one real, one ghost",
            "artifacts": [str(real_pdf), "/tmp/definitely-does-not-exist.pdf"],
        })
    finally:
        os.environ.pop("HERMES_KANBAN_TASK", None)

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_sub_fail_counts = {}
    runner._kanban_dispatcher_lock_handle = object()

    fake_adapter = MagicMock()
    fake_adapter.name = "telegram"

    documents_uploaded: list = []

    async def _send(chat_id, msg, metadata=None):
        runner._running = False

    async def _send_document(chat_id, file_path, metadata=None, **_kw):
        documents_uploaded.append(file_path)

    fake_adapter.send = AsyncMock(side_effect=_send)
    fake_adapter.send_document = AsyncMock(side_effect=_send_document)
    fake_adapter.send_multiple_images = AsyncMock()
    from gateway.platforms.base import BasePlatformAdapter
    fake_adapter.extract_local_files = BasePlatformAdapter.extract_local_files

    runner.adapters = {Platform.TELEGRAM: fake_adapter}

    _orig_sleep = asyncio.sleep

    async def _fast_sleep(_):
        await _orig_sleep(0)

    with patch("gateway.run.asyncio.sleep", side_effect=_fast_sleep):
        await asyncio.wait_for(
            runner._kanban_notifier_watcher(interval=1),
            timeout=10.0,
        )

    # Only the real file was uploaded.
    assert len(documents_uploaded) == 1
    assert "real.pdf" in documents_uploaded[0]
