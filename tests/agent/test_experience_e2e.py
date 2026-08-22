"""End-to-end: the experience loop through a real agent turn.

Everything else about this feature is tested against fakes — a
``SimpleNamespace`` agent, a monkeypatched recorder, a hand-built row. Those
prove the pieces. They cannot prove the loop actually closes inside a live
``run_conversation``, which is where the wiring bugs live: a hook attached to
the wrong chokepoint, a context that never reaches the request, an exception
swallowed so quietly that recording silently does nothing.

So this file drives the real ``AIAgent`` against an in-process mock provider
(the harness pattern from ``test_api_content_sidecar``) and asserts on the
**bytes captured off the wire**:

* turn 1 does real work → a row exists in the real store, scoped to the
  project root;
* turn 2 asks the same thing in different words, from a SUBDIRECTORY → the
  fenced block is present in the user message the provider actually received;
* the block never leaks into the persisted transcript content;
* a correction supersedes a stored row, and the next turn stops seeing it.

Cost discipline: a live turn costs ~4 s alone and ~70 s inside the suite's
16-worker pool. Profiling one turn puts ~90 ms of that in this feature
(``build_turn_context`` 51 ms + ``finalize_turn`` 36 ms) and ~4 s in tool
execution middleware — so the price here buys wiring proof, nothing else.
Hence two tests and four turns, each asserting as much as one turn allows.

Note on outcomes: turns here record ``partial`` — the scripted ``read_file``
comes back as an error in this sandbox. That is a property of the tool
sandbox, not of the feature; outcome classification is covered
deterministically in ``test_experience.py::TestExtraction``. The test that
needs a stored ``success`` seeds it through the real store API rather than
pretending a tool succeeded.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from hermes_state import SessionDB


class _MockHandler(BaseHTTPRequestHandler):
    captured_requests: list = []
    response_queue: list = []

    def do_POST(self):  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length).decode())
        type(self).captured_requests.append(req)
        # The model context-length probe posts here too. Only a real
        # chat-completions request may consume the queued script, or the probe
        # silently eats the first scripted turn.
        is_chat = "messages" in req
        if is_chat and type(self).response_queue:
            resp = type(self).response_queue.pop(0)
        else:
            resp = _text_resp("DONE")
        if req.get("stream") is True:
            self._write_sse(resp)
            return
        body = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_sse(self, resp: dict) -> None:
        """Replay the response as an SSE stream.

        The agent requests ``stream: true`` by default; a handler that only
        speaks JSON makes every call look like an empty stream, which the
        retry ladder then spends ~30 s failing on.
        """
        msg = resp["choices"][0]["message"]
        content = msg.get("content") or ""
        tcs = msg.get("tool_calls")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        chunks = [{"id": "m", "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": ""},
             "finish_reason": None}]}]
        if content:
            chunks.append({"id": "m", "choices": [
                {"index": 0, "delta": {"content": content}, "finish_reason": None}]})
        for ti, tc in enumerate(tcs or []):
            chunks.append({"id": "m", "choices": [{"index": 0, "delta": {
                "tool_calls": [{
                    "index": ti, "id": tc["id"], "type": "function",
                    "function": {"name": tc["function"]["name"],
                                 "arguments": tc["function"]["arguments"]}}]},
                "finish_reason": None}]})
        chunks.append({"id": "m", "choices": [
            {"index": 0, "delta": {},
             "finish_reason": "tool_calls" if tcs else "stop"}]})
        for c in chunks:
            self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, *a, **kw):
        pass


def _tc_resp(name: str, args: str = "{}") -> dict:
    return {
        "id": "m",
        "choices": [{"index": 0, "message": {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "call_1", "type": "function",
                            "function": {"name": name, "arguments": args}}]},
            "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    }


def _text_resp(text: str) -> dict:
    return {
        "id": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    }


@pytest.fixture()
def live(monkeypatch):
    """Mock provider + isolated HERMES_HOME + a shared SessionDB.

    ``make_agent()`` builds a fresh ``AIAgent`` bound to the same DB/session,
    so successive calls model successive turns of one conversation the way the
    gateway does (a new agent per message, history reloaded from the store).
    """
    _MockHandler.captured_requests = []
    _MockHandler.response_queue = []
    srv = HTTPServer(("127.0.0.1", 0), _MockHandler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    test_home = tempfile.mkdtemp(prefix="hermes_exp_e2e_")
    hermes_home = os.path.join(test_home, ".hermes")
    os.makedirs(hermes_home)
    prev_home = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = hermes_home

    # Pointing HERMES_HOME at a bare directory means `load_config_readonly()`
    # falls back to shipped defaults — which turn LSP ON. A file read then
    # tries to bring up a language server for the sandbox project and each
    # tool call costs seconds. This file already runs real agents inside a
    # 16-worker pool; left alone it starves every other test file.
    #
    # None of these subsystems are under test here, so switch them off
    # explicitly rather than inheriting whatever the defaults happen to be.
    with open(os.path.join(hermes_home, "config.yaml"), "w", encoding="utf-8") as fh:
        fh.write(
            "lsp:\n"
            "  enabled: false\n"
            "compression:\n"
            "  enabled: false\n"
            "memory:\n"
            "  memory_enabled: false\n"
            "  user_profile_enabled: false\n"
        )

    # Make the sandbox a real project root. Without it the root resolver walks
    # UP out of the temp directory and lands on whatever marker it finds first
    # — on Windows, tempdirs live under the user's home, so the workspace key
    # would be the home directory and the scoping assertions below would be
    # measuring the host machine rather than the feature.
    project = os.path.join(test_home, "project")
    os.makedirs(os.path.join(project, ".git"))
    subdir = os.path.join(project, "services", "api")
    os.makedirs(subdir)
    readable = os.path.join(project, "README.md")
    with open(readable, "w", encoding="utf-8") as fh:
        fh.write("# project\n")

    # TERMINAL_CWD is how Hermes is actually told where to work:
    # ``runtime_cwd.resolve_agent_cwd()`` reads it, and both the file tools and
    # the experience scoping key resolve through that. Pinning it here is what
    # makes the sandbox the agent's real working directory — without it the
    # tools resolve against the process cwd and every file read misses.
    prev_terminal_cwd = os.environ.get("TERMINAL_CWD")
    os.environ["TERMINAL_CWD"] = project
    monkeypatch.delenv("HERMES_EXPERIENCE", raising=False)
    monkeypatch.delenv("HERMES_EXPERIENCE_RETRIEVAL", raising=False)

    from run_agent import AIAgent

    db = SessionDB(db_path=Path(test_home) / "state.db")
    sid = "sess-exp-e2e"

    def make_agent(cwd: str | None = None):
        agent = AIAgent(
            api_key="test-key", base_url=f"http://127.0.0.1:{port}/v1",
            provider="openai-compat", model="test-model",
            max_iterations=10, enabled_toolsets=["file"],
            quiet_mode=True, skip_context_files=True, skip_memory=True,
            save_trajectories=False, platform="cli",
            session_db=db, session_id=sid,
        )
        agent.valid_tool_names = {"read_file"}
        # Pin the working directory so scoping is deterministic and
        # independent of wherever pytest happens to be running from.
        agent.session_cwd = cwd or project
        return agent

    try:
        yield make_agent, _MockHandler, db, sid, project, subdir, readable
    finally:
        srv.shutdown()
        db.close()
        shutil.rmtree(test_home, ignore_errors=True)
        if prev_terminal_cwd is None:
            os.environ.pop("TERMINAL_CWD", None)
        else:
            os.environ["TERMINAL_CWD"] = prev_terminal_cwd
        if prev_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = prev_home


def _chat_requests(handler) -> list:
    # The model context-length probe also hits the mock; keep only
    # chat-completions payloads.
    return [r for r in handler.captured_requests if "messages" in r]


def _last_user_content(handler) -> str:
    reqs = _chat_requests(handler)
    assert reqs, "no chat request reached the provider"
    users = [m for m in reqs[-1].get("messages", []) if m.get("role") == "user"]
    assert users, "no user message in the last request"
    return users[-1].get("content") or ""


def _work_turn(make_agent, handler, prompt: str, history=None, task_id="t",
               cwd=None, read_path="README.md"):
    """One turn that calls a tool and then answers.

    ``read_path`` decides the recorded outcome: a readable file makes the tool
    succeed (``success``), a bogus path makes it fail (``partial``).

    The argument is ``path`` — ``read_file`` reads ``args["path"]``, and any
    other key silently resolves to the empty string, which sends the tool into
    a "did you mean" directory scan that costs seconds per call.
    """
    handler.response_queue.append(
        _tc_resp("read_file", json.dumps({"path": read_path}))
    )
    handler.response_queue.append(_text_resp("All done."))
    agent = make_agent(cwd)
    return agent.run_conversation(
        prompt, conversation_history=history if history is not None else [],
        task_id=task_id,
    )


class TestLiveLoop:
    """Kept to two tests, and to four real turns between them.

    Each turn here drives a real ``AIAgent``: ~4 s alone, and ~70 s inside the
    16-worker pool the suite runs under, where the cost is not the feature
    (measured at ~90 ms of a 4.2 s turn — the rest is tool-execution
    middleware) but the machinery around it. A file that holds a worker for
    minutes starves the other heavy files, so this covers only what CANNOT be
    established without a live turn, and each test proves as much as it can
    per turn it spends.

    Outcome classification, relevance scoring, config gating, redaction and
    failure isolation are all proven deterministically in ``test_experience.py``
    and ``test_experience_wiring.py``; none of that is repeated here.
    """

    def test_record_then_retrieve_from_a_subdirectory(self, live):
        """The whole loop, plus P2's payoff, in two turns.

        Turn 1 works in the project root; turn 2 asks a paraphrase from a
        SUBDIRECTORY. One pass therefore proves: a real turn records; the row
        is scoped to the project root; retrieval matches a paraphrase; the
        scoping key survives a directory change; the fenced block reaches the
        provider; and it never lands in the stored transcript.
        """
        make_agent, handler, db, sid, project, subdir, readable = live

        # ── Turn 1: real work, in the project root ──
        _work_turn(make_agent, handler, "rebuild the payment ledger index",
                   task_id="t1", cwd=project)

        rows = db.export_experiences()
        assert len(rows) == 1, "finalize_turn did not reach the recorder"
        row = rows[0]
        assert "payment ledger index" in row["task"]
        assert "read_file" in row["tools"]
        assert row["workspace"] == project, (
            "the scoping key must be the project root, not the raw cwd"
        )
        assert row["verification"], "the evidence axis was never populated"

        # ── Turn 2: a paraphrase, from a subdirectory ──
        handler.captured_requests = []
        history = db.get_messages_as_conversation(sid)
        os.environ["TERMINAL_CWD"] = subdir
        _work_turn(make_agent, handler,
                   "the payment ledger index needs rebuilding again",
                   history=history, task_id="t2", cwd=subdir)

        sent = _last_user_content(handler)
        assert "<experience-context>" in sent, (
            "a turn run from a subdirectory lost the project's experience"
        )
        assert "rebuild the payment ledger index" in sent
        assert "not instructions" in sent, "the data-boundary note was dropped"

        # The injection is API-copy only.
        for msg in db.get_messages(sid):
            assert "<experience-context>" not in (msg["content"] or ""), (
                "stored content was polluted; only the api_content sidecar "
                "may carry the injection"
            )

    def test_a_correction_supersedes_and_stops_retrieval(self, live):
        """The user-correction hook, end to end through the real prologue.

        The prior experience is seeded through the store API: only a `success`
        is superseded by a correction, and this harness cannot produce one (see
        the module docstring). Everything after the seed is live.

        The "is it served before the correction" check goes through
        ``retrieve_experience_context`` rather than a full turn on purpose — a
        turn on the same task would MERGE into the seeded row and rewrite its
        outcome, destroying the precondition (and cost another live turn).
        """
        from agent.experience import Experience, normalize_task, task_fingerprint
        from agent.experience_runtime import (
            retrieve_experience_context,
            workspace_key,
        )

        make_agent, handler, db, sid, project, subdir, readable = live
        agent = make_agent()

        task = "migrate the invoice numbering scheme"
        norm = normalize_task(task)
        seeded = db.record_experience(Experience(
            task=task, task_norm=norm, task_hash=task_fingerprint(norm),
            outcome="success", strategy="used patch", tools=["patch"],
            session_id=sid, cwd=project, workspace=workspace_key(agent),
        ).to_row())
        assert seeded
        assert "used patch" in retrieve_experience_context(agent, task)

        # A live correction turn supersedes it.
        _work_turn(make_agent, handler, "no, that's the wrong approach",
                   task_id="t2")
        row = db.get_experience(seeded)
        assert row["correction_count"] == 1, "the correction hook never fired"
        assert row["superseded"] == 1, "a corrected success must stop being served"

        # And the next real turn's request no longer carries it.
        handler.captured_requests = []
        history = db.get_messages_as_conversation(sid)
        _work_turn(make_agent, handler, task, history=history, task_id="t3")
        sent = _last_user_content(handler)
        assert task in sent
        assert "used patch" not in sent, "a superseded experience came back"
