"""Regression tests for the #83714 imitation class.

Models replay their own compressed tool-call leaves — a 200-char head of the original
content plus the compression marker — verbatim as NEW tool calls. The stub then EXECUTES
(head only; the rest silently dropped) and renders into platform progress displays.

Contract: whatever ``_truncate_tool_call_args_json`` emits, in the shapes it is stored and
replayed in (verbatim, wire-escaped, embedded mid-leaf, nested payloads), must be refused
before dispatch with recovery tool results that preserve role alternation and never quote
the marker back (#47967 anti-priming). Leaves that merely quote the marker PREFIX (grep
patterns, compressor maintenance, fixtures) must still dispatch.
"""

import json
from types import SimpleNamespace

import pytest

from agent.context_compressor import (
    _truncate_tool_call_args_json,
    json_args_contain_compression_marker_copy,
)
from agent.message_sanitization import uniquify_tool_call_ids
from agent.turn_tool_validation import validate_tool_calls


class _FakeAgent:
    def __init__(self):
        self.valid_tool_names = {"terminal", "read_file"}
        self._invalid_tool_retries = 0
        self._invalid_json_retries = 0
        self.log_prefix = ""
        self.buffered = []

    def _uniquify_tool_call_ids(self, tool_calls):
        uniquify_tool_call_ids(tool_calls)

    def _repair_tool_call(self, name):
        return None

    def _vprint(self, *args, **kwargs):
        pass

    def _buffer_vprint(self, msg):
        self.buffered.append(msg)

    def _build_assistant_message(self, assistant_message, finish_reason):
        return {
            "role": "assistant",
            "content": assistant_message.content or "",
            "tool_calls": assistant_message.tool_calls,
        }


def _call(name, arguments, call_id="call_1"):
    return SimpleNamespace(
        id=call_id, call_id=call_id, type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _run(arguments, extra_calls=()):
    agent = _FakeAgent()
    messages = [{"role": "user", "content": "do the thing"}]
    assistant_message = SimpleNamespace(
        content="", tool_calls=[_call("terminal", arguments), *extra_calls]
    )
    verdict = validate_tool_calls(
        agent, assistant_message, "tool_calls", messages=messages,
        conversation_history=None, api_call_count=1, effective_task_id=None,
    )
    return verdict, messages


_LONG_CMD = (
    "timeout 30 ipmitool -I lanplus -H 192.168.1.248 -U admin -P admin chassis power cycle "
    "&& sleep 175 && timeout 150 ssh -o ConnectTimeout=20 krpa 'grep -E \"pcie-cap|pcie-speed\" "
    "/var/log/cmp-vf-chain.log | tail -5; echo GREP-DONE-SENTINEL && lspci -vvv -s 81:00.0 "
    "| grep -E \"LnkCap|LnkSta\" && cat /sys/bus/pci/devices/0000:81:00.0/current_link_speed "
    "/sys/bus/pci/devices/0000:81:00.0/current_link_width && dmesg | tail -40 "
    "&& nvidia-smi --query-gpu=pcie.link.gen.current --format=csv && echo ALL-DONE'"
)


def _compressed_command():
    """Round-trip through the REAL compressor shrink — the gate must refuse exactly what
    the compressor emits, not a hand-built approximation."""
    original = json.dumps({"command": _LONG_CMD, "timeout": 500})
    compressed = _truncate_tool_call_args_json(original)
    assert compressed != original, "precondition: the leaf must actually shrink"
    return json.loads(compressed)["command"]


def _shapes():
    compressed = _compressed_command()
    return {
        # The live incident: the compressed leaf replayed verbatim as a new call.
        "verbatim": json.dumps({"command": compressed, "timeout": 500}),
        # Providers may wire non-ASCII as \uXXXX (the incident stored \u27ea on the wire);
        # the detector must parse before matching.
        "wire_escaped": json.dumps({"command": compressed}, ensure_ascii=True),
        # Second live incident: the copy pasted INTO a heredoc script — marker mid-leaf.
        "mid_leaf_heredoc": json.dumps(
            {"command": "python3 - <<'EOF'\nold = " + compressed + "\nprint(old)\nEOF"}
        ),
        # Nested payload (delegate_task-shaped): the copy rides inside a list of dicts.
        "nested_payload": json.dumps({"tasks": [{"context": compressed}]}),
    }


@pytest.mark.parametrize("shape", sorted(_shapes()))
def test_compressed_leaf_replay_is_refused(shape):
    verdict, messages = _run(_shapes()[shape])
    assert verdict.action == "continue"
    # Role alternation: the refused batch is answered assistant → tool, never dispatched.
    assert messages[-2]["role"] == "assistant"
    refusal = messages[-1]
    assert refusal["role"] == "tool"
    assert refusal["tool_call_id"] == "call_1"
    assert refusal["content"].startswith("Error: this tool call was REFUSED")
    # #47967 anti-priming: the corrective must not teach the literal marker back.
    assert "HERMES-CONTEXT-COMPRESSION" not in refusal["content"]


def test_mixed_batch_results_every_call():
    clean = _call("read_file", json.dumps({"path": "/etc/hosts"}), call_id="ok_1")
    verdict, messages = _run(
        _shapes()["verbatim"],
        extra_calls=[clean],
    )
    # The clean call does not execute either (the batch re-issues), but every tool_call
    # keeps a matching result so the transcript stays provider-legal.
    assert verdict.action == "continue"
    # assistant row + one result per call
    assert [m["role"] for m in messages[-2:]] == ["tool", "tool"]
    by_id = {m["tool_call_id"]: m["content"] for m in messages[-2:]}
    assert by_id["call_1"].startswith("Error: this tool call was REFUSED")
    assert by_id["ok_1"].startswith("Skipped:")


@pytest.mark.parametrize("command", [
    # Debugging/maintenance quotes the PREFIX without the counts+sentence shape.
    "grep -c 'HERMES-CONTEXT-COMPRESSION' agent.log",
    "sqlite3 state.db \"SELECT id FROM messages WHERE tool_calls LIKE '%428 of 628%'\"",
    # The marker's fixed wording without the prefix/counts is ordinary prose.
    "echo 'chars omitted here by some other tool'",
])
def test_legitimate_quotes_still_dispatch(command):
    verdict, messages = _run(json.dumps({"command": command}))
    assert verdict.action == "ok"
    assert messages[-1]["role"] == "user"  # nothing appended
    assert not json_args_contain_compression_marker_copy(command)


def test_detector_scans_unparseable_text_raw():
    # Args that never reach the gate as JSON (defensive path) are still scanned.
    assert json_args_contain_compression_marker_copy("not json {" + _compressed_command())
