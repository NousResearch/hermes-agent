"""Full-read audit: stop gate on verified whole-folder reads (#124875)."""

from __future__ import annotations

import json
import zipfile
from types import SimpleNamespace

import pytest

from agent.full_read_audit import (
    arm_full_read_audit,
    audit_requested,
    build_full_read_nudge,
    finalizer_refusal,
    missing_read_paths,
)
from agent.turn_stop_gates import apply_stop_gates
from tools import file_tools as ft
from tools import terminal_tool as tt
from tools.file_state import get_registry
from tools.file_tools_read_tracking import _read_tracker, has_complete_read
from tools.file_tools_read_tracking import _mark_full_write_baseline


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    task_id = "fullread-124875"
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta\n", encoding="utf-8")
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    # Fake file ops that really read tmp files (see
    # tests/tools/test_file_read_guards.py): no environment setup, so the
    # home-I/O guard never trips. Session cwd only anchors the audit inventory.
    monkeypatch.setattr(ft, "_get_file_ops", lambda task_id="default": _HostFileOps())
    tt.record_session_cwd(task_id, str(tmp_path))
    yield tmp_path, task_id
    tt.clear_session_cwd(task_id)
    _read_tracker.pop(task_id, None)
    get_registry().forget_task(task_id)


@pytest.fixture
def quiet_gates(monkeypatch):
    import agent.turn_stop_gates as gates

    monkeypatch.setattr(gates, "_verify_on_stop_nudge", lambda agent: None)
    monkeypatch.setattr(gates, "_pre_verify_nudge", lambda agent, final_response, attempt: None)
    monkeypatch.setattr(gates, "_kanban_stop_nudge", lambda agent, messages: None)


def _agent():
    return SimpleNamespace(_full_read_audit=None, _full_read_nudges=0)


def _read(path, task_id, **kw):
    import os

    if not os.path.isabs(path):
        path = os.path.join(tt.get_session_cwd(task_id) or "", path)
    out = json.loads(ft.read_file_tool(path, task_id=task_id, **kw))
    assert not out.get("error"), out
    return out


class _FakeReadResult:
    """Mirrors FileOperations.read_file's surface over real tmp files."""

    def __init__(self, content, total_lines, file_size, truncated):
        self.content = content
        self._total_lines = total_lines
        self._file_size = file_size
        self._truncated = truncated

    def to_dict(self):
        return {
            "content": self.content,
            "total_lines": self._total_lines,
            "file_size": self._file_size,
            "truncated": self._truncated,
        }


class _HostFileOps:
    """Host-style ops (``env = None``) that really read the file."""

    env = None

    def read_file(self, path, offset=1, limit=2000):
        import os

        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.read().splitlines()
        total = len(lines)
        page = lines[offset - 1:offset - 1 + limit]
        return _FakeReadResult(
            "\n".join(page), total, os.path.getsize(path),
            truncated=(offset - 1 + limit < total),
        )

    def read_file_bytes(self, path, max_bytes=None):
        import base64
        from types import SimpleNamespace

        with open(path, "rb") as handle:
            raw = handle.read() if max_bytes is None else handle.read(max_bytes + 1)
        return SimpleNamespace(error=None, base64_content=base64.b64encode(raw).decode("ascii"),
                               file_size=len(raw))

    def _add_line_numbers(self, content, start_line=1):
        if content.endswith("\n"):
            content = content[:-1]
        return "\n".join(
            "%d|%s" % (i, line)
            for i, line in enumerate(content.split("\n"), start=start_line)
        )


def _read_all(agent, root, task_id, skip=()):
    """Read every inventoried file (the env scaffold may add its own)."""
    import os

    for path in agent._full_read_audit["paths"]:
        rel = os.path.relpath(path, str(root))
        if rel in skip:
            continue
        _read(rel, task_id)
    return [os.path.relpath(p, str(root)) for p in agent._full_read_audit["paths"]]


# -- admission trigger ------------------------------------------------------

TRIGGER_PHRASES = [
    "please read all files and get up to speed",
    "Read everything in this folder",
    "review all of it before deciding",
    "scan all documents",
    "go through everything",
    "read the whole directory",
]

UNRELATED = [
    "read a.txt and fix the typo",
    "what does b.txt say?",
    "summarize the project",
    "read the docs online",
]


@pytest.mark.parametrize("text", TRIGGER_PHRASES)
def test_trigger_phrases_arm(text):
    assert audit_requested(text) is True


@pytest.mark.parametrize("text", UNRELATED)
def test_unrelated_requests_do_not_arm(text):
    assert audit_requested(text) is False


def test_arm_snapshots_inventory(workspace):
    root, task_id = workspace
    agent = _agent()
    arm_full_read_audit(agent, "read all files", task_id)
    assert agent._full_read_audit is not None
    assert {str(root / "a.txt"), str(root / "b.txt")} <= set(agent._full_read_audit["paths"])


def test_arm_disarmed_without_trigger(workspace):
    _, task_id = workspace
    agent = _agent()
    arm_full_read_audit(agent, "read a.txt please", task_id)
    assert agent._full_read_audit is None


# -- stop gate ---------------------------------------------------------------

def test_gate_blocks_unread_and_names_file(workspace, quiet_gates):
    root, task_id = workspace
    agent = _agent()
    arm_full_read_audit(agent, "read all files", task_id)
    _read("a.txt", task_id)

    messages = [{"role": "user", "content": "read all files"}]
    final_msg = {"role": "assistant", "content": "Summary: alpha and beta."}
    verdict = apply_stop_gates(
        agent, final_msg, final_response="Summary: alpha and beta.",
        messages=messages, conversation_history=None,
        pending_verification_response=None,
        pending_verification_response_previewed=False,
    )
    assert verdict.continue_turn is True
    assert verdict.final_response is None
    # Discarded, never previewed, no fallback.
    assert verdict.pending_verification_response is None
    assert verdict.pending_verification_response_previewed is False
    assert "b.txt" in messages[-1]["content"]
    assert messages[-1].get("_full_read_synthetic") is True
    # The blocked candidate was never emitted as interim: only user rows added.
    assert [m["role"] for m in messages] == ["user", "user"]


def test_gate_passes_when_all_read(workspace, quiet_gates):
    root, task_id = workspace
    agent = _agent()
    arm_full_read_audit(agent, "read all files", task_id)
    _read_all(agent, root, task_id)

    messages = [{"role": "user", "content": "read all files"}]
    final_msg = {"role": "assistant", "content": "done"}
    verdict = apply_stop_gates(
        agent, final_msg, final_response="done",
        messages=messages, conversation_history=None,
        pending_verification_response=None,
        pending_verification_response_previewed=False,
    )
    assert verdict.continue_turn is False
    assert verdict.final_response == "done"
    assert len(messages) == 1


def test_decisive_file_blocks_synthesis(workspace, quiet_gates):
    root, task_id = workspace
    (root / "decisive.txt").write_text("the launch code is 0000, not 1234\n", encoding="utf-8")
    agent = _agent()
    arm_full_read_audit(agent, "read everything", task_id)
    _read_all(agent, root, task_id, skip=("decisive.txt",))

    assert build_full_read_nudge(agent) is not None
    assert "decisive.txt" in build_full_read_nudge(agent)
    _read("decisive.txt", task_id)
    assert missing_read_paths(agent) == []


def test_partial_page_is_not_complete(workspace):
    root, task_id = workspace
    (root / "multi.txt").write_text("".join("line %d\n" % i for i in range(10)), encoding="utf-8")
    resolved = str(root / "multi.txt")
    assert has_complete_read(resolved, task_id) is False
    _read("multi.txt", task_id, offset=1, limit=5)
    assert has_complete_read(resolved, task_id) is False
    _read("multi.txt", task_id, offset=6, limit=5)
    assert has_complete_read(resolved, task_id) is True


def test_write_is_not_a_read(workspace):
    root, task_id = workspace
    agent = _agent()
    # Exactly what a successful write_file leaves behind (write baseline) but
    # no read: the file exists at the same version, never paged through.
    (root / "written.txt").write_text("fresh words\n", encoding="utf-8")
    _mark_full_write_baseline(str(root / "written.txt"), task_id)
    arm_full_read_audit(agent, "read all files", task_id)
    # The file was written (write baseline exists) but never read.
    assert has_complete_read(str(root / "written.txt"), task_id) is False
    missing = missing_read_paths(agent)
    assert str(root / "written.txt") in missing


def _minimal_docx(path):
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>hello docx</w:t></w:r></w:p></w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("word/document.xml", document)


def test_extracted_document_counts_as_full_read(workspace):
    root, task_id = workspace
    _minimal_docx(str(root / "note.docx"))
    out = json.loads(ft.read_file_tool("note.docx", task_id=task_id))
    assert "hello docx" in json.dumps(out), out
    assert has_complete_read(str(root / "note.docx"), task_id) is True


# -- finalizer backstop --------------------------------------------------------

def test_finalizer_refuses_incomplete(workspace):
    _, task_id = workspace
    agent = _agent()
    arm_full_read_audit(agent, "read all files", task_id)
    _read("a.txt", task_id)
    refusal = finalizer_refusal(agent)
    assert refusal is not None
    assert "b.txt" in refusal
    assert "continue" in refusal


def test_finalizer_silent_when_complete(workspace):
    root, task_id = workspace
    agent = _agent()
    arm_full_read_audit(agent, "read all files", task_id)
    _read_all(agent, root, task_id)
    assert finalizer_refusal(agent) is None
    assert missing_read_paths(agent) == []


# -- finalizer backstop end-to-end ---------------------------------------------

class _StubBudget:
    used = 0
    max_total = 99
    remaining = 99


class _StubCompressor:
    last_prompt_tokens = 0


class _StubFinalizerAgent:
    """Minimal agent surface that ``finalize_turn`` reads from."""

    def __init__(self):
        self.max_iterations = 3
        self.iteration_budget = _StubBudget()
        self.context_compressor = _StubCompressor()
        self.model = "stub/model"
        self.provider = "stub"
        self.base_url = "http://stub"
        self.session_id = "sess-1"
        self.quiet_mode = True
        self.platform = "cli"
        self._interrupt_requested = False
        self._interrupt_message = None
        self._tool_guardrail_halt_decision = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self._full_read_audit = None
        self._persist_disabled = False
        for attr in (
            "session_input_tokens",
            "session_output_tokens",
            "session_cache_read_tokens",
            "session_cache_write_tokens",
            "session_reasoning_tokens",
            "session_prompt_tokens",
            "session_completion_tokens",
            "session_total_tokens",
            "session_estimated_cost_usd",
        ):
            setattr(self, attr, 0)
        self.session_cost_status = "ok"
        self.session_cost_source = "stub"

    def _save_trajectory(self, *a, **k):
        pass

    def _cleanup_task_resources(self, *a, **k):
        pass

    def _drop_trailing_empty_response_scaffolding(self, *a, **k):
        pass

    def _persist_session(self, *a, **k):
        pass

    def _emit_status(self, *a, **k):
        pass

    def _safe_print(self, *a, **k):
        pass

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **k):
        pass


def test_finalize_turn_refuses_unverified_synthesis(workspace):
    from agent.turn_finalizer import finalize_turn

    _, task_id = workspace
    agent = _StubFinalizerAgent()
    arm_full_read_audit(agent, "read all files", task_id)
    _read("a.txt", task_id)
    result = finalize_turn(
        agent,
        final_response="Summary of everything.",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=[{"role": "user", "content": "read all files"}],
        conversation_history=None,
        effective_task_id=task_id,
        turn_id="turn-1",
        user_message="read all files",
        original_user_message="read all files",
        _should_review_memory=False,
        _turn_exit_reason="text_response(stop)",
    )
    assert result["failed"] is True
    assert "b.txt" in result["final_response"]
    assert "Summary of everything" not in result["final_response"]


def test_finalize_turn_delivers_verified_synthesis(workspace):
    from agent.turn_finalizer import finalize_turn

    root, task_id = workspace
    agent = _StubFinalizerAgent()
    arm_full_read_audit(agent, "read all files", task_id)
    _read_all(agent, root, task_id)
    result = finalize_turn(
        agent,
        final_response="Summary of everything.",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=[{"role": "user", "content": "read all files"}],
        conversation_history=None,
        effective_task_id=task_id,
        turn_id="turn-1",
        user_message="read all files",
        original_user_message="read all files",
        _should_review_memory=False,
        _turn_exit_reason="text_response(stop)",
    )
    assert result["failed"] is False
    assert result["final_response"] == "Summary of everything."


# -- the refusal's promise survives the turn that produced it (#124875) --------


def test_continue_turn_is_still_audited(workspace):
    """The refusal says "send `continue`"; that follow-up carries no completeness
    quantifier, so it must not deliver the synthesis that was just refused."""
    root, task_id = workspace
    agent = _agent()
    arm_full_read_audit(agent, "read all files", task_id)
    _read("a.txt", task_id)
    assert finalizer_refusal(agent) is not None
    assert str(root / "b.txt") in agent._full_read_audit_pending["paths"]

    for follow_up in ("continue", "ok thanks", "go on"):
        agent._full_read_audit = None  # turn start clears it before arming
        arm_full_read_audit(agent, follow_up, task_id)
        assert agent._full_read_audit is not None, follow_up
        nudge = build_full_read_nudge(agent)
        assert nudge is not None and "b.txt" in nudge, follow_up


def test_rearm_stops_once_the_read_is_complete(workspace):
    root, task_id = workspace
    agent = _agent()
    arm_full_read_audit(agent, "read all files", task_id)
    _read("a.txt", task_id)
    finalizer_refusal(agent)
    _read_all(agent, root, task_id)

    agent._full_read_audit = None
    arm_full_read_audit(agent, "continue", task_id)
    assert missing_read_paths(agent) == []
    assert agent._full_read_audit_pending is None
    # A later unrelated turn is not gated by the spent audit.
    arm_full_read_audit(agent, "ok thanks", task_id)
    assert agent._full_read_audit is None


def test_rearm_is_bounded_for_an_abandoned_folder(workspace):
    from agent.full_read_audit import _PENDING_TURNS

    _, task_id = workspace
    agent = _agent()
    arm_full_read_audit(agent, "read all files", task_id)
    _read("a.txt", task_id)
    finalizer_refusal(agent)
    for _ in range(_PENDING_TURNS):
        agent._full_read_audit = None
        arm_full_read_audit(agent, "continue", task_id)
        assert agent._full_read_audit is not None
    agent._full_read_audit = None
    arm_full_read_audit(agent, "continue", task_id)
    assert agent._full_read_audit is None


def test_gate_gives_up_after_the_nudge_cap(workspace, quiet_gates):
    """An unsatisfiable audit must not burn every iteration: capped like the sibling gates."""
    from agent.turn_stop_gates import _MAX_FULL_READ_NUDGES

    _, task_id = workspace
    agent = _agent()
    arm_full_read_audit(agent, "read all files", task_id)
    _read("a.txt", task_id)
    agent._full_read_nudges = _MAX_FULL_READ_NUDGES

    messages = [{"role": "user", "content": "read all files"}]
    final_msg = {"role": "assistant", "content": "Summary: alpha."}
    verdict = apply_stop_gates(
        agent, final_msg, final_response="Summary: alpha.",
        messages=messages, conversation_history=None,
        pending_verification_response=None,
        pending_verification_response_previewed=False,
    )
    assert verdict.continue_turn is False
    assert verdict.final_response == "Summary: alpha."
    assert len(messages) == 1
