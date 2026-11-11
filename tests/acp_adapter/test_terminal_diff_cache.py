"""Terminal diffs, the cross-turn snapshot cache, and ACP session lineage.

A shell command mutates files with no structured result of its own and no success flag to
gate on, so the ACP diff has to be assembled from files the model observed earlier —
possibly in an earlier API round.  ``SessionState.read_snapshots_cache`` carries that
observation across turns; ``new_session`` carries ``parent_session_id`` so an editor can
branch a session and have the lineage land in the session row.
"""

from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from acp_adapter.events import make_step_cb, make_tool_progress_cb
from acp_adapter.session import SessionManager
from hermes_state import SessionDB


@pytest.fixture()
def loop():
    event_loop = asyncio.new_event_loop()
    yield event_loop
    event_loop.close()


@pytest.fixture()
def conn():
    import acp

    return MagicMock(spec=acp.Client)


def _start(progress, name, args):
    progress("tool.started", name, None, args)


def _complete(progress, name, result=None):
    progress("tool.completed", name, None, None, result=result)


# ---------------------------------------------------------------------------
# Cross-turn snapshot cache
# ---------------------------------------------------------------------------


def test_read_file_then_terminal_renders_the_edit_as_a_diff(tmp_path, loop, conn):
    """`read_file` in one round, a shell edit in a later one: the ACP diff for the command
    is built from the observation the read left behind."""
    target = tmp_path / "app.py"
    target.write_text("print('before')\n", encoding="utf-8")
    key = str(target.resolve())
    cache: dict[str, str | None] = {}
    ids, meta = {}, {}

    progress = make_tool_progress_cb(conn, "s", loop, ids, meta, cache)
    with patch("acp_adapter.events._send_update"):
        _start(progress, "read_file", {"path": str(target)})
    assert cache == {key: "print('before')\n"}

    # The command rewrites the file; its own result is opaque text.
    target.write_text("print('after')\n", encoding="utf-8")
    with patch("acp_adapter.events._send_update"):
        _start(progress, "terminal", {"command": f"sed -i s/before/after/ {target}"})

    snapshot = meta[list(meta)[-1]]["snapshot"]
    assert snapshot is not None
    assert snapshot.before[key] == "print('before')\n"

    from agent.display import extract_edit_diff

    diff = extract_edit_diff("terminal", "done", snapshot=snapshot)
    assert diff is not None
    assert "-print('before')" in diff
    assert "+print('after')" in diff

    # Completing the command re-baselines the cache, so the NEXT diff starts post-command.
    with patch("acp_adapter.events._send_update"):
        _complete(progress, "terminal", result="done")
    assert cache == {key: "print('after')\n"}

    with patch("acp_adapter.events._send_update"):
        _start(progress, "terminal", {"command": "true"})
    assert meta[list(meta)[-1]]["snapshot"].before[key] == "print('after')\n"


def test_unchanged_files_produce_no_terminal_diff(tmp_path, loop, conn):
    """A command that touched nothing must not fabricate a diff."""
    target = tmp_path / "app.py"
    target.write_text("same\n", encoding="utf-8")
    cache: dict[str, str | None] = {str(target.resolve()): "same\n"}
    ids, meta = {}, {}

    progress = make_tool_progress_cb(conn, "s", loop, ids, meta, cache)
    with patch("acp_adapter.events._send_update"):
        _start(progress, "terminal", {"command": "ls"})

    from agent.display import extract_edit_diff

    assert extract_edit_diff("terminal", "listing", snapshot=meta[list(meta)[-1]]["snapshot"]) is None


def test_write_then_terminal_diff_starts_from_post_write_content(tmp_path, loop, conn):
    """A write tool records its before-state and re-baselines on completion, so a later
    command's diff reports only the command's own change — not the earlier write."""
    target = tmp_path / "app.py"
    target.write_text("v1\n", encoding="utf-8")
    key = str(target.resolve())
    cache: dict[str, str | None] = {}
    ids, meta = {}, {}

    progress = make_tool_progress_cb(conn, "s", loop, ids, meta, cache)
    with patch("acp_adapter.events._send_update"), patch(
        "agent.display.capture_local_edit_snapshot",
        return_value=type("S", (), {"paths": [], "before": {key: "v1\n"}})(),
    ):
        _start(progress, "write_file", {"path": str(target), "content": "v2\n"})
    assert cache == {key: "v1\n"}

    target.write_text("v2\n", encoding="utf-8")
    with patch("acp_adapter.events._send_update"):
        _complete(progress, "write_file", result="{}")
    assert cache == {key: "v2\n"}  # re-baselined after the write

    target.write_text("v3\n", encoding="utf-8")
    with patch("acp_adapter.events._send_update"):
        _start(progress, "terminal", {"command": "bump"})

    from agent.display import extract_edit_diff

    diff = extract_edit_diff("terminal", "ok", snapshot=meta[list(meta)[-1]]["snapshot"])
    assert diff is not None
    assert "-v2" in diff and "+v3" in diff
    assert "v1" not in diff


@pytest.mark.parametrize("bad_path", [1, None, "", "   ", {"nested": 1}, ["a"]])
def test_malformed_read_path_is_ignored_not_raised(bad_path, loop, conn):
    """A model can emit a non-string path; resolving it inside this (swallowed) callback
    used to raise and silently kill the completion path."""
    cache: dict[str, str | None] = {}
    progress = make_tool_progress_cb(conn, "s", loop, {}, {}, cache)
    with patch("acp_adapter.events._send_update"):
        _start(progress, "read_file", {"path": bad_path})
    assert cache == {}


# ---------------------------------------------------------------------------
# Diff rendering
# ---------------------------------------------------------------------------


def test_terminal_completion_keeps_output_and_appends_the_diff(tmp_path):
    """The command's own output must survive: the diff is added alongside it, not instead."""
    from acp_adapter.tools import _build_tool_complete_content
    from agent.display import LocalEditSnapshot

    target = tmp_path / "app.py"
    snapshot = LocalEditSnapshot(paths=[target], before={str(target.resolve()): "before\n"})
    target.write_text("after\n", encoding="utf-8")

    content = _build_tool_complete_content(
        "terminal", "command output", function_args={"command": "sed -i s/before/after/ app.py"},
        snapshot=snapshot,
    )
    types = [getattr(block, "type", None) for block in content]
    assert "diff" in types, types
    texts = [getattr(getattr(block, "content", None), "text", "") or "" for block in content]
    assert any("command output" in t for t in texts), texts


def test_terminal_without_a_snapshot_falls_back_to_plain_text():
    from acp_adapter.tools import _build_tool_complete_content

    content = _build_tool_complete_content("terminal", "just output", function_args={"command": "ls"})
    assert [getattr(block, "type", None) for block in content] == ["content"]


# ---------------------------------------------------------------------------
# None-cache guard (the reviewer's deref case)
# ---------------------------------------------------------------------------


def test_step_callback_without_a_cache_does_not_raise(loop, conn):
    """``make_step_cb(..., read_snapshots_cache=None)`` must not dereference the cache."""
    cb = make_step_cb(conn, "s", loop, {"terminal": deque(["tc-1"])}, {}, None, None)
    with patch("acp_adapter.events._send_update"):
        cb(1, [{"name": "terminal", "result": "ok", "arguments": '{"command": "ls"}'}])


def test_step_fallback_refreshes_baselines_for_runtimes_without_completions(tmp_path, loop, conn):
    """No ``tool.completed`` projected: the step closer must still re-baseline after a
    mutating tool, or the next diff would report a change that is already applied."""
    target = tmp_path / "app.py"
    key = str(target.resolve())
    target.write_text("v1\n", encoding="utf-8")
    cache: dict[str, str | None] = {key: "v1\n"}

    cb = make_step_cb(conn, "s", loop, {"terminal": deque(["tc-1"])}, {}, None, cache)
    target.write_text("v2\n", encoding="utf-8")
    with patch("acp_adapter.events._send_update"):
        cb(1, [{"name": "terminal", "result": "ok", "arguments": '{"command": "bump"}'}])
    assert cache == {key: "v2\n"}


# ---------------------------------------------------------------------------
# Session lineage
# ---------------------------------------------------------------------------


def _manager(db):
    return SessionManager(db=db, agent_factory=lambda: SimpleNamespace(model="fixture"))


def _materialize(manager, state):
    """An empty session is ephemeral by design; content is what mints the row."""
    state.history.append({"role": "user", "content": "hello"})
    manager.save_session(state.session_id)
    return state


def test_branch_session_records_its_parent(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    manager = _manager(db)
    db.create_session(session_id="parent-1", source="acp", model="fixture")

    state = _materialize(manager, manager.create_session(cwd=str(workspace), parent_session_id="parent-1"))

    assert state.parent_session_id == "parent-1"
    assert db.get_session(state.session_id)["parent_session_id"] == "parent-1"
    db.close()


def test_root_session_leaves_the_parent_column_null(tmp_path):
    """No parent must be NULL, never an empty string — the column has a FK on ``sessions.id``
    and ``''`` violates it, which silently failed every session-row write."""
    db = SessionDB(tmp_path / "state.db")
    manager = _manager(db)

    state = _materialize(manager, manager.create_session(cwd=str(tmp_path)))

    assert state.parent_session_id is None
    assert db.get_session(state.session_id)["parent_session_id"] in (None, "")
    db.close()


def test_new_session_forwards_the_parent_from_kwargs(loop):
    """ACP has no schema field for a branch parent, so it arrives as an extra kwarg from the
    editor and must reach ``create_session``."""
    import acp_adapter.server as server_mod

    captured = {}

    class _Manager:
        def create_session(self, cwd=".", parent_session_id=None):
            captured["cwd"], captured["parent"] = cwd, parent_session_id
            return SimpleNamespace(session_id="sid", model="m", history=[])

    agent = server_mod.HermesACPAgent(session_manager=_Manager())
    agent._attach_session_mcp = _noop_async
    agent._session_response_fields = _noop_response_fields

    import asyncio as _asyncio

    _asyncio.run(agent.new_session(cwd="/work", parent_session_id="parent-9"))
    assert captured == {"cwd": "/work", "parent": "parent-9"}


async def _noop_async(*args, **kwargs):
    return None


async def _noop_response_fields(*args, **kwargs):
    return {}


def test_new_session_without_parent_passes_none(loop):
    import asyncio as _asyncio

    import acp_adapter.server as server_mod

    captured = {}

    class _Manager:
        def create_session(self, cwd=".", parent_session_id=None):
            captured["parent"] = parent_session_id
            return SimpleNamespace(session_id="sid", model="m", history=[])

    agent = server_mod.HermesACPAgent(session_manager=_Manager())
    agent._attach_session_mcp = _noop_async
    agent._session_response_fields = _noop_response_fields

    _asyncio.run(agent.new_session(cwd="/work"))
    assert captured["parent"] is None
