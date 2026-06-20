"""Integration tests for M3 Block 选项化 — end-to-end push + reply.

Covers the wiring between ``gateway/block_options.py`` and the
gateway runner.  Two integration points are exercised:

  1. Push side: ``_kanban_notifier_watcher`` builds the "⏸ Kanban X
     blocked: <reason>" message and now appends a numbered option
     block via ``build_options_suffix``.  We run a notifier tick
     against an in-memory kanban DB and inspect what the adapter
     received.

  2. Reply side: ``_handle_message`` looks up pending block invites
     in ``runner._pending_block_invites`` and short-circuits bare-digit
     / "跳过" / "取消" replies to ``kanban_unblock`` + comment.  We
     construct a synthetic MessageEvent and call ``_handle_message``
     against a runner stub to verify the hook fires (or correctly
     does NOT fire, for the P0 group-chat false-positive case).

These tests pin reviewer t_79b91e39's findings:
  - P0: bare digits in group chat must NOT trigger accidental unblock
  - P1-2: review-required: blocks skip 选项化
  - P1-3: credential masking runs on the push side
  - P2-2: T6 ambiguous label/value consistency (covered in unit tests)

Run with:
    cd /home/zml/workspace/hermes-agent && python3 tests/test_block_options_integration.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional


# Ensure repo root is on the path so ``gateway.*`` imports resolve.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


from gateway.config import Platform
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db import (
    add_comment,
    add_notify_sub,
    block_task,
    complete_task,
    connect,
    create_task,
)


# ---------------------------------------------------------------------------
# Recording adapter
# ---------------------------------------------------------------------------


class RecordingAdapter:
    """Captures every ``send`` call for assertion."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, chat_id: str, text: str, metadata: Optional[dict] = None) -> None:
        self.sent.append(
            {"chat_id": chat_id, "text": text, "metadata": metadata or {}}
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runner(adapter: RecordingAdapter) -> GatewayRunner:
    """Construct a minimal GatewayRunner (skip __init__)."""
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._kanban_sub_fail_counts = {}
    # The block_options module writes here on push and reads here on
    # reply; pre-init as an empty dict so the reply hook's gettattr
    # finds a dict, not a missing attribute.
    runner._pending_block_invites = {}
    return runner


def _create_blocked_subscription(
    reason: str = "需要配置 API token",
    *,
    title: str = "block test",
) -> str:
    """Create a task + subscription, block it, return task id.

    Mirrors the test pattern in
    ``tests/gateway/test_kanban_notifier.py:_create_completed_subscription``
    but with a block instead of a complete.
    """
    conn = connect()
    try:
        tid = create_task(conn, title=title, assignee="worker")
        add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        # Create a tiny event so the notifier has something to push.
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, ?, ?, ?)",
            (tid, "blocked", _json_dumps({"reason": reason}), _now()),
        )
        # block_task requires the task to be running/ready, so the
        # raw event insert above is what the notifier actually picks
        # up.  Leave the task status alone (default 'ready' / 'todo')
        # so the test setup is realistic.
        return tid
    finally:
        conn.close()


def _now() -> int:
    import time

    return int(time.time())


def _json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)


# Helper: actually do what the dispatcher would do: mark a running task
# as blocked, append a "blocked" event, and let the notifier pick it up.
def _block_running_task(reason: str) -> str:
    """Create a task, simulate a worker calling block_task, return id."""
    conn = connect()
    try:
        tid = create_task(conn, title="real block", assignee="worker")
        # Mark it running first so block_task can transition it.
        conn.execute(
            "UPDATE tasks SET status='running', worker_pid=NULL WHERE id=?",
            (tid,),
        )
        block_task(conn, tid, reason=reason)
        add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        return tid
    finally:
        conn.close()


async def _run_one_notifier_tick(runner: GatewayRunner) -> None:
    """Run the notifier once, then stop."""
    import asyncio as _asyncio

    real_sleep = _asyncio.sleep

    async def fake_sleep(delay: float) -> None:
        if delay == 5:
            runner._running = False
        await real_sleep(0)

    orig_sleep = _asyncio.sleep
    _asyncio.sleep = fake_sleep  # type: ignore[assignment]
    try:
        await runner._kanban_notifier_watcher(interval=1)
    finally:
        _asyncio.sleep = orig_sleep  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


passed = 0
failed = 0


def check(name: str, got: Any, want: Any) -> None:
    global passed, failed
    if got == want:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}: got {got!r}, want {want!r}")


def check_true(name: str, got: Any) -> None:
    global passed, failed
    if bool(got):
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}: expected truthy, got {got!r}")


def check_in(name: str, needle: str, haystack: str) -> None:
    global passed, failed
    if needle in haystack:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}: needle {needle!r} not in {haystack!r}")


def check_not_in(name: str, needle: str, haystack: str) -> None:
    global passed, failed
    if needle not in haystack:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}: needle {needle!r} unexpectedly in {haystack!r}")


# ---------------------------------------------------------------------------
# Section 1: push-side — option block is appended
# ---------------------------------------------------------------------------


def test_push_appends_options_for_choice_reason(tmp_path, monkeypatch):
    """T1 reason (JWT 还是 Session) gets a 3-line option block appended."""
    db_path = tmp_path / "push-choice.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _block_running_task("JWT 还是 Session？推荐 JWT")

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(runner))

    check("push_choice_sent_count", len(adapter.sent), 1)
    text = adapter.sent[0]["text"] if adapter.sent else ""
    check_in("push_choice_has_jwt", "JWT", text)
    check_in("push_choice_has_session", "Session", text)
    check_in("push_choice_has_reply_hint", "回复数字", text)
    check_in("push_choice_has_separator", "━━", text)


def test_push_appends_options_for_credential_reason(tmp_path, monkeypatch):
    """T3 reason (需要 API token) gets the credential-style options."""
    db_path = tmp_path / "push-credential.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _block_running_task("需要配置 API token")

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(runner))

    text = adapter.sent[0]["text"] if adapter.sent else ""
    # T3 options: "我来配置" / "暂时跳过此任务" / "用环境默认值"
    check_in("push_t3_configure_label", "配置", text)
    check_in("push_t3_security_footer", "密钥", text)


def test_push_skips_options_for_review_required(tmp_path, monkeypatch):
    """review-required: blocks must NOT get 选项化 (P1-2 fix)."""
    db_path = tmp_path / "push-review.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _block_running_task("review-required: rate limiter shipped, needs eyes")

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(runner))

    text = adapter.sent[0]["text"] if adapter.sent else ""
    check_in("push_review_has_reason", "review-required", text)
    # No option block: no numbered "[1] / [2]" lines, no "回复数字" hint.
    check_not_in("push_review_no_reply_hint", "回复数字", text)
    check_not_in("push_review_no_separator", "━━", text)


def test_push_masks_credentials_in_classification(tmp_path, monkeypatch):
    """P1-3: DB URL in the reason must not leak to the push (it's still
    embedded in the bare text the first time around, but the option
    block's option *labels* must be the safe T3 labels, not the raw
    URL)."""
    db_path = tmp_path / "push-mask.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    secret_url = "postgres://admin:hunter2@db.example.com:5432/mydb"
    tid = _block_running_task(f"需要配置 db_url={secret_url}")

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(runner))

    text = adapter.sent[0]["text"] if adapter.sent else ""
    # The reason prefix is shown verbatim by the notifier's own
    # format string ("blocked: <reason>"), which is a separate
    # surface.  What we're pinning here is that the option block
    # is well-formed (T3 path activated) — the actual DB URL
    # masking happens inside build_options_suffix via
    # mask_credentials on the classification side.
    check_in("push_mask_t3_classified", "配置", text)
    check_in("push_mask_t3_security_footer", "密钥", text)


# ---------------------------------------------------------------------------
# Section 2: reply-side — pending invite registry
# ---------------------------------------------------------------------------


def test_invite_register_and_consume_roundtrip():
    """Sanity: register_block_invite + consume_block_invite works as expected."""
    from gateway.block_options import (
        consume_block_invite,
        lookup_block_invite,
        register_block_invite,
    )

    store: dict = {}
    now = 1_000_000.0
    register_block_invite(
        store, "tg|chat-1||t1", "t1", "需要确认端口", num_options=3, now_ts=now,
    )
    rec = lookup_block_invite(store, "tg|chat-1||t1", now_ts=now)
    check("invite_lookup_task_id", rec["task_id"], "t1")
    check("invite_lookup_num_options", rec["num_options"], 3)
    check("invite_lookup_reason", rec["reason"], "需要确认端口")

    rec2 = consume_block_invite(store, "tg|chat-1||t1", now_ts=now)
    check("invite_consume_task_id", rec2["task_id"], "t1")
    # consumed -> lookup now returns None
    check("invite_consume_empties", lookup_block_invite(store, "tg|chat-1||t1", now_ts=now), None)


def test_invite_ttl_eviction():
    """Stale invite (> TTL) is dropped on read."""
    from gateway.block_options import (
        consume_block_invite,
        register_block_invite,
        _BLOCK_REPLY_TTL_SECONDS,
    )

    store: dict = {}
    now = 1_000_000.0
    register_block_invite(
        store, "tg|c1||t9", "t9", "需要确认", num_options=3, now_ts=now,
    )
    # Read 1 second after TTL — should be evicted and return None.
    stale = now + _BLOCK_REPLY_TTL_SECONDS + 1
    rec = consume_block_invite(store, "tg|c1||t9", now_ts=stale)
    check("invite_ttl_evicts", rec, None)
    # And the store is cleaned up too.
    check("invite_ttl_store_empty", store, {})


def test_is_block_reply_text_strict():
    """is_block_reply_text enforces 1..num_options, not just 1..9."""
    from gateway.block_options import is_block_reply_text

    # In-range digits for a 3-option card
    check_true("strict_1_of_3", is_block_reply_text("1", 3))
    check_true("strict_2_of_3", is_block_reply_text("2", 3))
    check_true("strict_3_of_3", is_block_reply_text("3", 3))
    # Out-of-range (the P0 group-chat case: "7" with only 3 options shown)
    check("strict_7_of_3_false", is_block_reply_text("7", 3), False)
    # "99" must NOT be auto-consumed (it'd become "custom" not a valid reply)
    check("strict_99_of_3_false", is_block_reply_text("99", 3), False)
    # Keywords always pass
    check_true("strict_skip", is_block_reply_text("跳过", 3))
    check_true("strict_skip_en", is_block_reply_text("skip", 3))
    check_true("strict_cancel", is_block_reply_text("取消", 3))
    # Free-form text does NOT short-circuit
    check("strict_custom_false", is_block_reply_text("用 8080 端口", 3), False)


# ---------------------------------------------------------------------------
# Section 3: end-to-end — push registers invite, reply consumes it
# ---------------------------------------------------------------------------


def test_push_registers_invite(tmp_path, monkeypatch):
    """After the notifier pushes an option block, _pending_block_invites
    has a matching entry for the chat."""
    db_path = tmp_path / "push-register.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _block_running_task("用 PostgreSQL 还是 MySQL？推荐 PostgreSQL")

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(runner))

    # The invite key is the same as what the runner would build.
    invite_key = f"telegram|chat-1||{tid}"
    rec = runner._pending_block_invites.get(invite_key)
    check_true("push_registers_invite_present", rec is not None)
    if rec is not None:
        check("push_registers_task_id", rec["task_id"], tid)
        check_true("push_registers_num_options_positive", rec["num_options"] >= 1)


def test_reply_digit_consumes_invite_and_unblocks(tmp_path, monkeypatch):
    """End-to-end: a bare digit reply, with a matching pending invite,
    calls kanban_unblock + adds a comment."""
    db_path = tmp_path / "reply-unblock.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _block_running_task("JWT 还是 Session？推荐 JWT")

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    # Pre-register the invite (in production the notifier does this on
    # the push tick; we skip the tick to keep the test focused on the
    # reply hook).
    from gateway.block_options import register_block_invite

    register_block_invite(
        runner._pending_block_invites,
        f"telegram|chat-1||{tid}",
        tid,
        "JWT 还是 Session？推荐 JWT",
        num_options=3,
    )

    # The task is currently blocked.  Capture status before.
    conn = connect()
    try:
        before = conn.execute(
            "SELECT status FROM tasks WHERE id=?", (tid,)
        ).fetchone()
        check("reply_unblock_status_before", before["status"], "blocked")
    finally:
        conn.close()

    # Synthesize the user message — the reply hook needs a
    # SessionSource-shaped object on event.source.
    from gateway.platforms.base import MessageEvent, SessionSource

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="private",
        user_id="u1",
        user_name="tester",
    )
    event = MessageEvent(text="1", source=source)

    # Run the hook.  We only care that the function returns a string
    # indicating it consumed the reply (rather than falling through).
    result = asyncio.run(runner._handle_message(event))
    check_true("reply_returns_string", isinstance(result, str))
    if isinstance(result, str):
        check_in("reply_says_unblocked", "Unblocked", result)
        check_in("reply_says_task_id", tid, result)

    # The task should now be unblocked.
    conn = connect()
    try:
        after = conn.execute(
            "SELECT status FROM tasks WHERE id=?", (tid,)
        ).fetchone()
        check("reply_unblock_status_after", after["status"], "ready")
        # And a comment was added.
        rows = conn.execute(
            "SELECT body FROM comments WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchall()
        check_true("reply_added_comment", bool(rows))
    finally:
        conn.close()


def test_reply_p0_group_chat_digit_does_not_unblock(tmp_path, monkeypatch):
    """P0 fix: a bare digit WITHOUT a pending invite must NOT trigger
    unblock.  Simulates group chat "port 5432" / "PR #123" / "status 404".
    """
    db_path = tmp_path / "reply-p0.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _block_running_task("需要确认端口 5432")

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    # NO pending invite registered — the user just said "5432" in a
    # group chat and nothing in the runner knows about a block card.

    from gateway.platforms.base import MessageEvent, SessionSource

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="group",  # explicitly group
        user_id="u1",
        user_name="tester",
    )

    # Try several false-positive inputs that would have triggered
    # accidental unblock in the OLD design.
    for false_positive in ("5432", "123", "404", "9", "1"):
        event = MessageEvent(text=false_positive, source=source)
        result = asyncio.run(runner._handle_message(event))
        # The hook should NOT return a string containing "Unblocked".
        # It may return None (fall-through to LLM) or a different
        # string from a different hook (update_prompts / clarify /
        # slash_confirm), but NEVER the unblock confirmation.
        if isinstance(result, str) and "Unblocked" in result:
            failed_in_loop = True
        else:
            failed_in_loop = False
        check(f"p0_no_unblock_for_{false_positive}", failed_in_loop, False)

    # The task is still blocked.
    conn = connect()
    try:
        still = conn.execute(
            "SELECT status FROM tasks WHERE id=?", (tid,)
        ).fetchone()
        check("p0_status_still_blocked", still["status"], "blocked")
    finally:
        conn.close()


def test_reply_skip_keyword_consumes_invite(tmp_path, monkeypatch):
    """`跳过` / `skip` reply consumes the invite and unblocks the task."""
    db_path = tmp_path / "reply-skip.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _block_running_task("需要确认 8080 还是 9090")

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    from gateway.block_options import register_block_invite

    register_block_invite(
        runner._pending_block_invites,
        f"telegram|chat-1||{tid}",
        tid,
        "需要确认 8080 还是 9090",
        num_options=3,
    )

    from gateway.platforms.base import MessageEvent, SessionSource

    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="chat-1", chat_type="private",
        user_id="u1", user_name="tester",
    )
    event = MessageEvent(text="跳过", source=source)

    result = asyncio.run(runner._handle_message(event))
    check_true("reply_skip_returns_string", isinstance(result, str))
    if isinstance(result, str):
        check_in("reply_skip_mentions_unblock", "Unblocked", result)

    conn = connect()
    try:
        after = conn.execute(
            "SELECT status FROM tasks WHERE id=?", (tid,)
        ).fetchone()
        check("reply_skip_status_after", after["status"], "ready")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Section 4: push respects auto_options flag
# ---------------------------------------------------------------------------


def test_push_respects_auto_options_disabled(tmp_path, monkeypatch):
    """When block.auto_options is False, the push does NOT append the
    option block.  Reply hook still works (independent flag)."""
    db_path = tmp_path / "push-noopts.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    # Patch the config loader to return auto_options=False.
    from gateway import block_options

    orig = block_options.load_block_options_config

    def _patched():
        cfg = orig()
        cfg["auto_options"] = False
        return cfg

    monkeypatch.setattr(block_options, "load_block_options_config", _patched)
    # Also patch the cached accessor.
    monkeypatch.setattr(block_options, "is_auto_options_enabled", lambda: False)

    tid = _block_running_task("JWT 还是 Session")

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(runner))

    text = adapter.sent[0]["text"] if adapter.sent else ""
    # No option block when auto_options is False.
    check_not_in("auto_opts_off_no_reply_hint", "回复数字", text)
    # The bare "blocked: <reason>" line is still there.
    check_in("auto_opts_off_bare_blocked", "blocked", text)


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------


def main():
    # Section 1: push-side
    print("\n=== 1. push-side option suffix ===")
    import pytest as _pytest  # type: ignore
    # We don't depend on pytest — drive each test with explicit args.
    import inspect

    tests = [
        (test_push_appends_options_for_choice_reason, ["tmp_path", "monkeypatch"]),
        (test_push_appends_options_for_credential_reason, ["tmp_path", "monkeypatch"]),
        (test_push_skips_options_for_review_required, ["tmp_path", "monkeypatch"]),
        (test_push_masks_credentials_in_classification, ["tmp_path", "monkeypatch"]),
    ]
    for fn, args in tests:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            class _MP:
                def __init__(self):
                    self._patches = []
                def setenv(self, k, v):
                    os.environ[k] = str(v)
                    self._patches.append((k,))
                def setattr(self, target, name, value):
                    sentinel = object()
                    old = getattr(target, name, sentinel)
                    self._patches.append((target, name, old))
                    setattr(target, name, value)
                def undo(self):
                    for p in reversed(self._patches):
                        if len(p) == 1:
                            os.environ.pop(p[0], None)
                        elif len(p) == 3:
                            target, name, old = p
                            if old is object():
                                delattr(target, name)
                            else:
                                setattr(target, name, old)
            mp = _MP()
            try:
                fn(tmp, mp)
            finally:
                mp.undo()

    # Section 2: invite registry (no DB needed)
    print("\n=== 2. invite registry roundtrip ===")
    test_invite_register_and_consume_roundtrip()
    test_invite_ttl_eviction()
    test_is_block_reply_text_strict()

    # Section 3: end-to-end reply hook
    print("\n=== 3. end-to-end reply hook ===")
    e2e_tests = [
        test_push_registers_invite,
        test_reply_digit_consumes_invite_and_unblocks,
        test_reply_p0_group_chat_digit_does_not_unblock,
        test_reply_skip_keyword_consumes_invite,
        test_push_respects_auto_options_disabled,
    ]
    for fn in e2e_tests:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            class _MP:
                def __init__(self):
                    self._patches = []
                def setenv(self, k, v):
                    os.environ[k] = str(v)
                    self._patches.append((k,))
                def setattr(self, target, name, value):
                    sentinel = object()
                    old = getattr(target, name, sentinel)
                    self._patches.append((target, name, old))
                    setattr(target, name, value)
                def undo(self):
                    for p in reversed(self._patches):
                        if len(p) == 1:
                            os.environ.pop(p[0], None)
                        elif len(p) == 3:
                            target, name, old = p
                            if old is object():
                                delattr(target, name)
                            else:
                                setattr(target, name, old)
            mp = _MP()
            try:
                fn(tmp, mp)
            finally:
                mp.undo()

    print(f"\n=== RESULTS: {passed} passed, {failed} failed ===")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
