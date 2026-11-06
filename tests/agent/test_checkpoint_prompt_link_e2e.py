"""Real-path E2E: file checkpoints link to the prompt that caused them.

A real ``AIAgent`` with the real tool executor, a real ``CheckpointManager`` and a real ``SessionDB``,
in a temp ``HERMES_HOME`` and a temp git working directory; only the model is scripted. Pins what a
mocked manager cannot: that the uid stamped on the live user message at the first tool iteration is
the ``message_uid`` the row is later flushed with (so the link resolves in ``state.db``), that
``/rollback`` quotes the prompt, that the ``rollback.for_prompt`` RPC answers from real checkpoints
and real rows, and that the link keeps resolving after the transcript is compacted.
"""

import os
import sqlite3
import subprocess
from pathlib import Path

import pytest

from tests.e2e.core.history._helpers import NO_BACKGROUND_REVIEW, OFFLINE_CONFIG, big
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall, write_hermes_home


def _git(work, *args):
    subprocess.run(["git", "-C", str(work), *args], check=True, capture_output=True)


class _Session:
    """A real agent driven the way the classic CLI drives it (it owns ``history``)."""

    def __init__(self, base_url: str, hermes_home: Path, sid: str, *, checkpoints: bool = True) -> None:
        from hermes_cli.config import load_config
        from hermes_cli.tools_config import _get_platform_tools
        from hermes_state import SessionDB
        from run_agent import AIAgent

        self.home = hermes_home
        self.db = SessionDB(db_path=hermes_home / "state.db")
        self.agent = AIAgent(
            provider="custom", base_url=base_url, api_key="sk-fake-e2e", model="fake-model",
            session_db=self.db, session_id=sid, quiet_mode=True, platform="cli",
            enabled_toolsets=sorted(_get_platform_tools(load_config(), "cli")),
            checkpoints_enabled=checkpoints,
        )
        self.history: list[dict] = []

    @property
    def sid(self) -> str:
        return self.agent.session_id

    def turn(self, text: str) -> dict:
        result = self.agent.run_conversation(text, conversation_history=self.history, task_id=self.sid)
        self.history = result.get("messages") or self.history
        assert result.get("final_response") is not None, result
        return result

    def user_rows(self) -> list[tuple[int, str, str, int]]:
        """``(row_id, message_uid, content, active)`` of every stored user message, oldest first."""
        conn = sqlite3.connect(f"file:{self.home / 'state.db'}?mode=ro", uri=True)
        try:
            return conn.execute(
                "SELECT id, message_uid, content, active FROM messages WHERE session_id = ? AND role = 'user' "
                "ORDER BY id", (self.sid,)).fetchall()
        finally:
            conn.close()

    def close(self) -> None:
        try:
            self.agent.close()
        finally:
            self.db.close()


@pytest.fixture()
def world(tmp_path, monkeypatch):
    hermes_home = Path(os.environ["HERMES_HOME"])  # the hermetic conftest's per-test home
    home = tmp_path / "home"
    home.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    for name in ("a.txt", "b.txt", "c.txt"):  # the project the turns then add files to
        (work / name).write_text(f"{name} v0\n")
    _git(work, "init", "-q")
    _git(work, "-c", "user.email=t@example.invalid", "-c", "user.name=t", "add", "-A")
    _git(work, "-c", "user.email=t@example.invalid", "-c", "user.name=t", "commit", "-qm", "init")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("TERMINAL_CWD", str(work))
    monkeypatch.chdir(work)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-e2e")
    opened: list[_Session] = []

    def make(script, *, checkpoints=True, sid="e2e-link"):
        srv = FakeLLMServer(script)
        srv.__enter__()
        servers.append(srv)
        write_hermes_home(hermes_home, srv.base_url, extra_config=OFFLINE_CONFIG + NO_BACKGROUND_REVIEW)
        session = _Session(srv.base_url, hermes_home, sid, checkpoints=checkpoints)
        session.srv = srv
        opened.append(session)
        return session

    servers: list[FakeLLMServer] = []
    yield {"make": make, "work": work, "hermes_home": hermes_home}
    for session in opened:
        session.close()
    for srv in servers:
        srv.__exit__(None, None, None)


def _write(work, name, text):
    return ToolCall("write_file", {"path": str(work / name), "content": text})


def test_two_prompts_link_their_checkpoints_and_the_lookup_answers(world):
    work = world["work"]
    session = world["make"]([
        _write(work, "n1.txt", "n1.txt v1\n"), Text("first done"),
        _write(work, "n2.txt", "n2.txt v1\n"), _write(work, "n3.txt", "n3.txt v1\n"), Text("second done"),
    ])
    session.turn("rewrite a please")
    session.turn("now rewrite b and c")
    mgr = session.agent._checkpoint_mgr

    first, second = session.user_rows()
    checkpoints = mgr.list_checkpoints(str(work))  # newest first: second prompt's two, then the first's

    # The uid stamped on the live message at the first tool iteration is the one the row was flushed with.
    assert [(cp["session_id"], cp["prompt_uid"]) for cp in checkpoints] == [
        (session.sid, second[1]), (session.sid, second[1]), (session.sid, first[1])]
    assert (work / "n1.txt").read_text() == "n1.txt v1\n"

    from tools.checkpoint_prompts import checkpoints_from_prompt

    from_first = checkpoints_from_prompt(mgr, session.sid, first[0])
    from_second = checkpoints_from_prompt(mgr, session.sid, second[0])
    assert [row["files_changed_since"] for row in from_first] == [1, 2, 3]   # c; b+c; a+b+c
    assert [row["files_changed_since"] for row in from_second] == [1, 2]
    assert {row["workdir"] for row in from_first} == {str(work.resolve())}
    assert all(row["message"] == "before write_file" for row in from_first)

    # A row that is not a user message of this session is refused, not guessed at.
    assert checkpoints_from_prompt(mgr, session.sid, 10**9) is None
    assert checkpoints_from_prompt(mgr, "some-other-session", first[0]) is None


def test_rollback_lists_each_checkpoint_under_the_prompt_that_caused_it(world):
    work = world["work"]
    session = world["make"]([_write(work, "n1.txt", "n1.txt v1\n"), Text("done")])
    session.turn("rewrite a please")

    from tools.checkpoint_manager import format_checkpoint_list

    text = format_checkpoint_list(session.agent._checkpoint_mgr.list_checkpoints(str(work)), str(work))

    assert "before write_file" in text
    assert "↳ prompt: “rewrite a please”" in text


def test_rpc_answers_from_real_checkpoints_and_rows(world):
    import threading

    from tui_gateway import server

    work = world["work"]
    session = world["make"]([_write(work, "n1.txt", "n1.txt v1\n"), Text("done")])
    session.turn("rewrite a please")
    (row,) = session.user_rows()
    server._sessions["sid"] = {
        "agent": session.agent, "session_key": session.sid, "history": [], "history_lock": threading.Lock(),
        "history_version": 0, "running": False, "attached_images": [], "image_counter": 0, "cols": 80,
        "slash_worker": None, "show_reasoning": False, "tool_progress_mode": "all", "cwd": str(work),
    }
    try:
        ok = server.handle_request({"id": "1", "method": "rollback.for_prompt",
                                    "params": {"session_id": "sid", "message_id": row[0]}})
        bad = server.handle_request({"id": "2", "method": "rollback.for_prompt",
                                     "params": {"session_id": "sid", "message_id": 10**9}})
    finally:
        server._sessions.pop("sid", None)

    assert ok["result"]["enabled"] is True
    (cp,) = ok["result"]["checkpoints"]
    assert cp["prompt_uid"] == row[1] and cp["files_changed_since"] == 1 and cp["workdir"] == str(work.resolve())
    assert "result" not in bad and bad["error"]


def test_checkpoints_off_touches_no_message_and_records_no_link(world):
    work = world["work"]
    session = world["make"]([_write(work, "n1.txt", "n1.txt v1\n"), Text("done")], checkpoints=False)
    session.turn("rewrite a please")

    mgr = session.agent._checkpoint_mgr
    assert mgr.enabled is False and mgr._prompt_link is None
    assert mgr.list_checkpoints(str(work)) == []
    # The live user message was never stamped by the checkpoint hook: its uid is the store's own.
    live_user = next(m for m in session.history if m.get("role") == "user")
    (row,) = session.user_rows()
    assert live_user.get("message_uid") in (None, row[1])


def test_the_link_keeps_resolving_after_the_transcript_is_compacted(world):
    from agent.conversation_compression_manual import compress_now, parse_compress_args

    work = world["work"]
    labels = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot")
    # The two newest prompts edit files; they sit in the window compaction keeps.
    session = world["make"]([
        Text("ok"), Text("ok"), Text("ok"), Text("ok"),
        _write(work, "n1.txt", "n1.txt v1\n"), Text("done a"),
        _write(work, "n2.txt", "n2.txt v1\n"), Text("done b"),
    ])
    for label in labels[:4]:
        session.turn(big(label))
    session.turn("edit a now")
    session.turn("edit b now")
    mgr = session.agent._checkpoint_mgr
    before = {content: (row_id, uid) for row_id, uid, content, _ in session.user_rows()}

    result = compress_now(session.agent, session.history, parse_compress_args(""), task_id=session.sid)
    assert result.status == "compressed", result.summary
    session.history = result.after_messages

    from tools.checkpoint_prompts import checkpoints_from_prompt

    active = [(row_id, uid, content) for row_id, uid, content, is_active in session.user_rows() if is_active]
    edit_a = next(r for r in active if "edit a now" in r[2])
    # The uid is the same logical message before and after; the compaction copy may carry a new row id.
    assert edit_a[1] == before["edit a now"][1]
    rows = checkpoints_from_prompt(mgr, session.sid, edit_a[0])
    assert rows is not None and [r["prompt_uid"] for r in rows][-1] == edit_a[1]
    assert len(rows) == 2  # edit a's checkpoint and edit b's


def test_what_the_model_is_sent_does_not_change_with_checkpoints_on(world):
    """The prompt cache stays byte-stable: the uid the checkpoint hook stamps is persistence-only."""
    sent = {}
    for checkpoints, sid in ((False, "e2e-off"), (True, "e2e-on")):
        session = world["make"]([Text("hello back"), Text("fine")], checkpoints=checkpoints, sid=sid)
        session.turn("say hello")
        session.turn("and again")
        sent[checkpoints] = session.srv.main_requests()

    assert len(sent[True]) == len(sent[False]) == 2
    for on, off in zip(sent[True], sent[False]):
        assert "message_uid" not in str(on)
        assert on["tools"] == off["tools"]
        # Everything after the system message (which carries session-specific text) is identical.
        assert on["messages"][1:] == off["messages"][1:]
