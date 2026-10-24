"""``transform_persisted_row``: an embedder may project each session-db row at flush."""

import types

from agent.session_persistence import _db_flush_row
from hermes_cli.plugins import get_plugin_manager


def test_persisted_row_hook_sees_each_row_once_with_its_index(monkeypatch):
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    seen = []

    def _project(agent, message, row, msg_idx):
        seen.append(msg_idx)
        return {**row, "content": f"[{msg_idx}] {row['content']}"}

    mgr._hooks.setdefault("transform_persisted_row", []).append(_project)
    try:
        row = _db_flush_row(types.SimpleNamespace(), {"role": "assistant", "content": "hi"}, False, 3)
    finally:
        mgr._hooks = saved
    assert seen == [3]
    assert row["content"] == "[3] hi"


def _with_hook(callback):
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    mgr._hooks.setdefault("transform_persisted_row", []).append(callback)
    return mgr, saved


def test_persisted_row_hook_gets_a_copy_of_the_message():
    """``VALID_HOOKS`` promises callbacks copies: a callback scribbling on ``message`` must not
    rewrite the live transcript dict."""
    msg = {"role": "assistant", "content": "hi"}

    def _scribble(agent, message, row, msg_idx):
        message["content"] = "scribbled"
        row["content"] = "scribbled"

    mgr, saved = _with_hook(_scribble)
    try:
        row = _db_flush_row(types.SimpleNamespace(), msg, False, 0)
    finally:
        mgr._hooks = saved
    assert msg["content"] == "hi"
    assert row["content"] == "hi"


def test_hung_persisted_row_hook_is_bounded_and_skipped(monkeypatch):
    """A hung transform must not stall the session-db flush: past ``hook_callback_timeout`` it is
    skipped and the untransformed row is written."""
    import threading
    import time

    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.2)
    release = threading.Event()

    def _hung(agent, message, row, msg_idx):
        release.wait(timeout=5.0)
        return {**row, "content": "late"}

    mgr, saved = _with_hook(_hung)
    started = time.monotonic()
    try:
        row = _db_flush_row(types.SimpleNamespace(), {"role": "assistant", "content": "hi"}, False, 0)
    finally:
        release.set()
        mgr._hooks = saved
    assert row["content"] == "hi"
    assert time.monotonic() - started < 3.0
