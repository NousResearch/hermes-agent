"""Recovery stubs retain the details needed to decide whether to open the original."""

import hashlib
import json

from agent.context_compressor import (
    ContextCompressor, _PRUNE_MIN_CHARS, _is_summary_stub, _summarize_tool_result,
)
from agent.tool_dispatch_helpers import _maybe_wrap_untrusted


def _summary(name, args, body):
    return _summarize_tool_result(name, json.dumps(args), body)


def test_failing_terminal_uses_decoded_lines_and_last_nonempty_line():
    last = "FAILED tests/test_example.py::test_x - AssertionError"
    output = "\n".join([f"progress {i}" for i in range(37)] + [last]) + "\n"
    body = json.dumps({"output": output, "exit_code": 1})
    assert _summary("terminal", {"command": "pytest tests/test_example.py -q"}, body) == (
        "[terminal] ran `pytest tests/test_example.py -q` -> exit 1, 38 lines output | last: " + last)


def test_terminal_stub_stays_under_the_prune_floor():
    """A stub at or over the floor would be summarised again by a later pass and lose its exit code."""
    body = json.dumps({"output": "ERROR " + "y" * 300, "exit_code": 127})
    stub = _summary("terminal", {"command": "c" * 200}, body)
    assert len(stub) < _PRUNE_MIN_CHARS
    assert "-> exit 127, 1 line output | last: ERROR yyy" in stub


def test_delegate_batch_names_count_and_first_goal():
    tasks = [{"goal": "a" * 80}, {"goal": "second"}, {"goal": "third"}]
    stub = _summary("delegate_task", {"tasks": tasks}, "done")
    assert stub == "[delegate_task] 3 tasks, first: '" + "a" * 57 + "...' (4 chars result)"
    assert _is_summary_stub(stub)


def _payload_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def test_generic_mcp_stub_keeps_issue_number():
    """A normal MCP result keeps the call's arguments and a stable payload hash (#88042)."""
    body = "done"
    stub = _summary("mcp_github_get_issue", {"owner": "org", "repo": "project", "issue_number": 123}, body)
    assert stub == (
        "[mcp_github_get_issue] owner=org repo=project issue_number=123 "
        f"hash={_payload_hash(body)} lines=1 ({len(body)} chars result)"
    )
    assert "\n" not in stub
    assert _is_summary_stub(stub)


def test_normal_mcp_result_stays_inline_until_it_is_pruned():
    """Under the prune floor the body is untouched. Summarising it still names the document, not only a byte count."""
    document = "short note"
    raw = json.dumps({"result": document})
    messages = [{"role": "tool", "tool_call_id": "c1", "content": raw}]
    changed = ContextCompressor._demote_tool_result_at(
        messages, 0, {"c1": ("mcp__vault__read", json.dumps({"path": "notes/a.md"}))}, _PRUNE_MIN_CHARS,
    )
    assert changed is False
    assert messages[0]["content"] == raw

    stub = _summary("mcp__vault__read", {"path": "notes/a.md"}, raw)
    assert f"hash={_payload_hash(document)}" in stub
    assert "path=notes/a.md" in stub
    assert f"({len(raw)} chars result)" in stub
    assert document not in stub
    assert _is_summary_stub(stub)


def test_oversized_mcp_result_compresses_to_a_bounded_hash_stub():
    """A document-sized MCP result becomes a stub the next prune pass will not summarise again.

    The body is not copied into the stub (it was wrapped as untrusted data). The hash is of the
    document, so the same body matches with or without that wrapper, and a different body does not.
    """
    document = "HEADTOKEN\n" + ("x" * 20_000) + "\nTAILTOKEN"
    raw = json.dumps({"result": document, "structuredContent": {"result": document}})
    wrapped = _maybe_wrap_untrusted("mcp__vault__read", raw)
    assert len(wrapped) > len(raw) > 20_000
    assert "TAILTOKEN" in wrapped

    stub = _summary("mcp__vault__read", {"path": "notes/a.md"}, wrapped)
    bare = _summary("mcp__vault__read", {"path": "notes/a.md"}, raw)
    other = _summary("mcp__vault__read", {"path": "notes/a.md"}, json.dumps({"result": document + "!"}))

    assert stub.startswith("[mcp__vault__read] path=notes/a.md ")
    assert f"hash={_payload_hash(document)}" in stub
    assert f"hash={_payload_hash(document)}" in bare
    assert f"hash={_payload_hash(document + '!')}" in other
    assert "TAILTOKEN" not in stub and "HEADTOKEN" not in stub
    assert "untrusted_tool_result" not in stub
    assert "Treat it as DATA" not in stub
    assert len(stub) < 400 and len(stub) < len(wrapped)
    assert "\n" not in stub
    assert _is_summary_stub(stub)

    messages = [{"role": "tool", "tool_call_id": "c1", "content": wrapped}]
    calls = {"c1": ("mcp__vault__read", json.dumps({"path": "notes/a.md"}))}
    assert ContextCompressor._demote_tool_result_at(messages, 0, calls, _PRUNE_MIN_CHARS) is True
    assert messages[0]["content"] == stub
    assert ContextCompressor._demote_tool_result_at(messages, 0, calls, _PRUNE_MIN_CHARS) is False
    assert messages[0]["content"] == stub


def test_wrapped_mcp_failure_stays_marked_failed():
    """The untrusted wrapper used to hide ``{"error": ...}``, so a failed MCP call compressed as a success."""
    error = json.dumps({"error": "vault unavailable: timeout"})
    wrapped = _maybe_wrap_untrusted("mcp__vault__read", error)
    assert wrapped.startswith("<untrusted_tool_result")
    stub = _summary("mcp__vault__read", {"path": "notes/missing.md"}, wrapped)
    assert "FAILED: vault unavailable: timeout" in stub
    assert "path=notes/missing.md" in stub
    assert _is_summary_stub(stub)


def test_generic_argument_preview_has_a_total_budget():
    args = {f"arg{i}": "v" * 80 for i in range(10)}
    stub = _summary("custom", args, "done")
    expected = (" arg0=" + "v" * 40 + " arg1=" + "v" * 40 + " arg2=" + "v" * 22)
    assert len(expected) == 120
    assert stub == "[custom]" + expected + " (4 chars result)"
    assert "\n" not in stub and len(stub) < 400
    assert _is_summary_stub(stub)
