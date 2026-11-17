"""Worker-side board contracts through the real dispatcher and a real ``hermes chat -q`` worker.

* ``kanban_complete(artifacts=[...])``: a declared file inside the task workspace is staged as a
  ``task_attachments`` row; a declared file OUTSIDE the workspace must be reported (attached or the
  completion refused), never silently dropped while the card reads as delivered (#120647).
* ``--skill`` pins: a pin that resolves reaches the model on the worker's first request; a pin that
  no longer resolves must not kill the dispatcher-owned worker before its session starts (#119619).

Verdicts come from kanban.db rows, files on disk and the provider's request log.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import pytest

from tests.e2e.core._pending_fixes import known_gate
from tests.e2e.core.kanban._helpers import Board, wait_until
from tests.fakes.fake_llm_provider import FakeLLMServer, Text, ToolCall

pytestmark = [
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc worker liveness"),
    pytest.mark.live_system_guard_bypass,  # teardown SIGKILLs this board's reparented workers
]

KNOWN: dict[str, tuple[str, str]] = {
    "outside": (r"card done with its declared outside-workspace artifact never attached",
                "#120647 kanban_complete artifact outside the workspace is silently never attached"),
    "test_worker_with_unresolvable_pinned_skill_still_starts_its_session": (
        r"worker died before its session \(exits=",
        "#119619 dispatcher-owned worker with a stale --skill pin exits rc=1 before its session"),
}

_TASK_RE = re.compile(r"work kanban task (t_[0-9a-f]+)")
_EXIT_RE = re.compile(r"\[kanban-worker-exit\] rc=(-?\d+)")
_REFUSAL_KIND_RE = re.compile(r"block|refus|reject|artifact|violation")
SKILL_MARK = "E2E_PINNED_SKILL_BODY_7f3a"


class KnownGap(AssertionError):
    """The tracked bug's own assertion, the only type ``known_gate`` accepts; any harness failure
    stays red."""


def _task_id(rec: dict) -> str:
    for m in rec["body"]["messages"]:
        hit = _TASK_RE.search(str(m.get("content")))
        if hit:
            return hit.group(1)
    raise AssertionError("worker prompt did not name its task")


def _run_one_card(board: Board, tid: str) -> None:
    board.dispatch()
    pid = board.task(tid)["worker_pid"]
    assert pid, board.diag(tid)
    board.wait_worker_exit(tid, int(pid))


# artifacts ------------------------------------------------------------------------------------


def _artifact_location(board: Board, tid: str, where: str) -> Path:
    workspace = board.hermes_home / "kanban" / "workspaces" / tid
    return {"inside": workspace / "report.md",
            "outside": board.hermes_home / "scripts" / f"{tid}-deliverable.md"}[where]


@pytest.mark.parametrize("where", ["inside", "outside"])
def test_declared_artifact_is_attached_or_reported(tmp_path, where: str) -> None:
    board_ref: dict[str, Board] = {}
    blocked: dict[str, bool] = {}
    payload = f"DELIVERABLE_{where.upper()}_c0ffee\n"

    def responder(rec: dict):
        last = rec["body"]["messages"][-1]
        if last.get("role") == "tool":
            # A refused completion is answered the way a careful worker would: park the card.
            if '"error"' in str(last.get("content")) and not blocked.get("sent"):
                blocked["sent"] = True
                return ToolCall("kanban_block", {"reason": "artifact could not be delivered"})
            return Text("delivered")
        tid = _task_id(rec)
        path = _artifact_location(board_ref["b"], tid, where)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")  # the file exists before the call declares it
        return ToolCall("kanban_complete", {"summary": "report written", "artifacts": [str(path)]})

    with FakeLLMServer(responder) as srv:
        board = board_ref["b"] = Board(tmp_path, srv.base_url)
        try:
            tid = board.create(f"artifact card ({where})")
            _run_one_card(board, tid)
            status = board.task(tid)["status"]
            attached = board._q("SELECT * FROM task_attachments WHERE task_id = ?", (tid,))
            stored = [Path(a["stored_path"]) for a in attached]
            with known_gate(KNOWN, where, raises=KnownGap):
                if status == "done" and not attached:
                    raise KnownGap(f"card done with its declared {where}-workspace artifact never attached\n"
                                   f"{board.diag(tid)}")
            assert status == "done" or where == "outside", board.diag(tid)
            if status != "done":
                _assert_visible_refusal(board, srv, tid, _artifact_location(board, tid, where))
            if attached:
                assert len(attached) == 1 and stored[0].read_text(encoding="utf-8") == payload, attached
                assert not stored[0].is_relative_to(board.hermes_home / "kanban" / "workspaces"), stored
        finally:
            board.kill_workers()


def _assert_visible_refusal(board: Board, srv: FakeLLMServer, tid: str, artifact: Path) -> None:
    """A completion that did not land must say why, naming the artifact, somewhere a human or the
    worker sees it; and the refusal must not turn into a crash loop."""
    board.dispatch("--max", "0")  # reap the exited worker so its run is booked
    name = artifact.name
    tool_errors = [str(m.get("content")) for body in srv.main_requests() for m in body["messages"]
                   if m.get("role") == "tool" and '"error"' in str(m.get("content"))]
    traces = [c for c in tool_errors if name in c]
    traces += [e["kind"] for e in board.events(tid)
               if _REFUSAL_KIND_RE.search(e["kind"]) and name in json.dumps(e["payload"])]
    traces += [r["error"] for r in board.runs(tid) if name in (r["error"] or "")]
    assert traces, f"card left {board.task(tid)['status']} with no refusal naming {name}\n{board.diag(tid)}"
    runs = board.runs(tid)
    assert len(runs) == 1 and runs[0]["outcome"] != "crashed", board.diag(tid)
    assert not board.events(tid, "gave_up") and board.task(tid)["consecutive_failures"] <= 1, board.diag(tid)


# pinned skills --------------------------------------------------------------------------------


def _seed_skill(board: Board, name: str) -> None:
    skill = board.hermes_home / "skills" / name
    skill.mkdir(parents=True, exist_ok=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Use when running the kanban e2e pin probe.\n---\n\n"
        f"# {name}\n\nAlways follow {SKILL_MARK}.\n", encoding="utf-8")


def _completing_responder(rec: dict):
    if rec["body"]["messages"][-1].get("role") == "tool":
        return Text("pinned run done")
    return ToolCall("kanban_complete", {"summary": "pinned skill run"})


def test_worker_with_resolvable_pinned_skill_sees_it_on_first_request(tmp_path) -> None:
    with FakeLLMServer(_completing_responder) as srv:
        board = Board(tmp_path, srv.base_url)
        try:
            _seed_skill(board, "e2e-pinned")
            tid = board.create("pinned skill card", "--skill", "e2e-pinned")
            _run_one_card(board, tid)
            first = srv.main_requests()[0]
            assert SKILL_MARK in str(first["messages"]), "pinned skill body never reached the model"
            assert board.task(tid)["status"] == "done", board.diag(tid)
            # The terminal call is the first answer; at most one closing turn follows it.
            assert 1 <= len(srv.main_requests()) <= 2, len(srv.main_requests())
        finally:
            board.kill_workers()


def test_worker_with_unresolvable_pinned_skill_still_starts_its_session(tmp_path) -> None:
    with FakeLLMServer(_completing_responder) as srv:
        board = Board(tmp_path, srv.base_url)
        try:
            _seed_skill(board, "e2e-archived")
            tid = board.create("stale pin card", "--skill", "e2e-archived")
            # The operator removes the skill after the pin was written.
            shutil.rmtree(board.hermes_home / "skills" / "e2e-archived")
            _run_one_card(board, tid)
            log = wait_until(lambda: board.worker_log(tid), 10, "worker log")
            exits = [int(rc) for rc in _EXIT_RE.findall(log)]
            # The bug's own signature: no model call at all, and the worker died of the stale pin
            # (nonzero exit trailer, or its log names the missing skill). Anything else stays red.
            with known_gate(KNOWN, "test_worker_with_unresolvable_pinned_skill_still_starts_its_session",
                            raises=KnownGap):
                if not srv.main_requests() and ((exits and exits[-1] != 0) or "e2e-archived" in log):
                    raise KnownGap(f"worker died before its session (exits={exits}); board:\n{board.diag(tid)}")
            assert SKILL_MARK not in str(srv.main_requests()[0]["messages"]), "removed skill still loaded"
            assert board.task(tid)["status"] == "done", board.diag(tid)
        finally:
            board.kill_workers()
