"""Tests for the read-only tool loop MoA advisors can use.

The advisor may look at the machine (read_file / list_dir / view_image) but can
never change it, so these tests pin the dispatch surface, the path guards the
reader already enforces, and the loop's bounds.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def _response(content: str = "", tool_calls=None, usage=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def _tool_call(name: str, arguments: dict, call_id: str = "call_1"):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


class TestExecuteReferenceTool:
    def test_read_file_delegates_to_the_guarded_reader(self, tmp_path):
        from unittest.mock import patch

        from agent.moa_reference_tools import execute_reference_tool

        target = tmp_path / "note.txt"
        target.write_text("alpha\nbeta\n", encoding="utf-8")
        seen: dict = {}

        def fake_read(path, offset=1, limit=2000, task_id="default"):
            seen.update({"path": path, "offset": offset, "limit": limit, "task_id": task_id})
            return '{"content": "1|alpha\\n2|beta", "is_image": false}'

        with patch("tools.file_tools.read_file_tool", side_effect=fake_read):
            out = execute_reference_tool("read_file", json.dumps({"path": str(target), "limit": 20}))

        assert "alpha" in out
        assert seen["path"] == str(target)
        assert seen["limit"] == 20
        assert seen["task_id"] == "moa-reference"

    def test_credential_denial_passes_through(self, tmp_path):
        from unittest.mock import patch

        from agent.moa_reference_tools import execute_reference_tool

        denial = '{"error": "Access denied: .env is a Hermes credential store and cannot be read directly."}'

        with patch("tools.file_tools.read_file_tool", return_value=denial):
            out = execute_reference_tool("read_file", json.dumps({"path": "C:/hermes/.env"}))

        assert "credential store" in out
        assert out == denial

    def test_image_read_is_redirected_to_view_image(self, tmp_path):
        from unittest.mock import patch

        from agent.moa_reference_tools import execute_reference_tool

        with patch("tools.file_tools.read_file_tool", return_value='{"content": "", "is_image": true}'):
            out = execute_reference_tool("read_file", json.dumps({"path": str(tmp_path / "shot.png")}))

        assert "view_image" in out

    def test_unknown_tool_is_refused(self):
        from agent.moa_reference_tools import execute_reference_tool

        out = execute_reference_tool("write_file", json.dumps({"path": "x", "content": "y"}))

        assert out.startswith("error: unknown tool")
        assert "write_file" in out

    def test_only_read_tools_are_exposed(self):
        from agent.moa_reference_tools import REFERENCE_TOOL_SCHEMAS

        names = {schema["function"]["name"] for schema in REFERENCE_TOOL_SCHEMAS}

        assert names == {"read_file", "list_dir", "view_image"}

    def test_list_dir_lists_entries(self, tmp_path):
        from agent.moa_reference_tools import execute_reference_tool

        (tmp_path / "one.txt").write_text("x", encoding="utf-8")
        (tmp_path / "sub").mkdir()

        out = execute_reference_tool("list_dir", json.dumps({"path": str(tmp_path)}))

        assert "one.txt" in out and "dir sub" in out

    def test_bad_arguments_are_reported(self):
        from agent.moa_reference_tools import execute_reference_tool

        assert execute_reference_tool("read_file", "not json").startswith("error:")


class TestRunReferenceToolLoop:
    def test_tool_round_then_answer(self, tmp_path):
        from unittest.mock import patch

        from agent.moa_reference_tools import run_reference_tool_loop

        target = tmp_path / "note.txt"
        target.write_text("the answer is 42\n", encoding="utf-8")
        sent: list = []

        def send(messages, tools):
            sent.append({"messages": [dict(m) for m in messages], "tools": tools})
            if len(sent) == 1:
                return _response(tool_calls=[_tool_call("read_file", {"path": str(target)})])
            return _response(content="advice from the file")

        with patch("tools.file_tools.read_file_tool", return_value='{"content": "1|the answer is 42"}'):
            responses, convo = run_reference_tool_loop(send, [{"role": "user", "content": "look"}])

        assert len(responses) == 2
        assert sent[0]["tools"] and sent[1]["tools"]
        tool_messages = [m for m in convo if m.get("role") == "tool"]
        assert tool_messages and "the answer is 42" in tool_messages[0]["content"]
        assert convo[-1]["role"] == "tool"

    def test_plain_answer_uses_one_call(self):
        from agent.moa_reference_tools import run_reference_tool_loop

        calls: list = []

        def send(messages, tools):
            calls.append(tools)
            return _response(content="just advice")

        responses, _ = run_reference_tool_loop(send, [{"role": "user", "content": "hi"}])

        assert len(responses) == 1
        assert calls[0] is not None

    def test_iteration_cap_withdraws_tools(self):
        from agent.moa_reference_tools import run_reference_tool_loop

        calls: list = []

        def send(messages, tools):
            calls.append(tools)
            if tools is not None:
                return _response(tool_calls=[_tool_call("list_dir", {"path": "."})])
            return _response(content="final advice")

        responses, _ = run_reference_tool_loop(
            send, [{"role": "user", "content": "loop"}], max_iterations=2,
        )

        assert len(responses) == 3  # two tool rounds, then the tools-withdrawn answer
        assert calls[0] and calls[1] and calls[2] is None
        assert responses[-1].choices[0].message.content == "final advice"

    def test_expired_deadline_skips_tools(self):
        from agent.moa_reference_tools import run_reference_tool_loop

        calls: list = []

        def send(messages, tools):
            calls.append(tools)
            return _response(content="straight answer")

        run_reference_tool_loop(send, [{"role": "user", "content": "hi"}], deadline=0.0)

        assert calls == [None]

    def test_tool_failure_is_text_not_exception(self):
        from unittest.mock import patch

        from agent.moa_reference_tools import run_reference_tool_loop

        def send(messages, tools):
            if tools is not None:
                return _response(tool_calls=[_tool_call("read_file", {"path": "C:/definitely/missing.txt"})])
            return _response(content="advised anyway")

        with patch("tools.file_tools.read_file_tool", side_effect=RuntimeError("disk on fire")):
            responses, convo = run_reference_tool_loop(send, [{"role": "user", "content": "hi"}])

        assert responses[-1].choices[0].message.content == "advised anyway"
        tool_messages = [m for m in convo if m.get("role") == "tool"]
        assert tool_messages and "disk on fire" in tool_messages[0]["content"]


class TestRunReferenceWithTools:
    def test_system_prompt_and_loop_are_used(self, tmp_path):
        from unittest.mock import patch

        from agent.moa_loop import _run_reference

        target = tmp_path / "data.txt"
        target.write_text("value=7\n", encoding="utf-8")
        calls: list = []

        def fake_call_llm(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return _response(tool_calls=[_tool_call("read_file", {"path": str(target)})])
            return _response(content="the file says value=7")

        slot = {"provider": "nous", "model": "tencent/hy4-flash:free"}

        with patch("agent.moa_loop._slot_runtime", return_value={"provider": "nous", "model": slot["model"]}), \
             patch("agent.moa_loop.call_llm", side_effect=fake_call_llm), \
             patch("agent.moa_loop._slot_request_headers", return_value=None), \
             patch("tools.file_tools.read_file_tool", return_value='{"content": "1|value=7"}'), \
             patch("agent.moa_loop._maybe_apply_moa_cache_control", side_effect=lambda msgs, rt, **kw: msgs):
            _label, text, _acct = _run_reference(
                slot, [{"role": "user", "content": "what is in data.txt?"}],
                reference_tools=True,
            )

        assert text == "the file says value=7"
        assert len(calls) == 2 and calls[0]["tools"] and calls[1]["tools"]
        assert "read_file" in calls[0]["messages"][0]["content"]
        assert "cannot call tools" not in calls[0]["messages"][0]["content"]

    def test_without_tools_one_plain_call(self):
        from unittest.mock import patch

        from agent.moa_loop import _run_reference

        calls: list = []

        def fake_call_llm(**kwargs):
            calls.append(kwargs)
            return _response(content="plain advice")

        slot = {"provider": "nous", "model": "tencent/hy4-flash:free"}

        with patch("agent.moa_loop._slot_runtime", return_value={"provider": "nous", "model": slot["model"]}), \
             patch("agent.moa_loop.call_llm", side_effect=fake_call_llm), \
             patch("agent.moa_loop._slot_request_headers", return_value=None), \
             patch("agent.moa_loop._maybe_apply_moa_cache_control", side_effect=lambda msgs, rt, **kw: msgs):
            _, text, _ = _run_reference(slot, [{"role": "user", "content": "hi"}])

        assert text == "plain advice"
        assert len(calls) == 1
        assert "tools" not in calls[0]


class TestRealPath:
    """No mocks: the advisor's reader is the agent's own, guards included."""

    def test_real_read_returns_file_content(self, tmp_path):
        from agent.moa_reference_tools import execute_reference_tool

        target = tmp_path / "note.txt"
        target.write_text("codename BLUE-HORSE-7731\n", encoding="utf-8")

        out = execute_reference_tool("read_file", json.dumps({"path": str(target)}))

        assert "BLUE-HORSE-7731" in out

    def test_real_env_file_is_refused_without_leaking(self, tmp_path):
        from agent.moa_reference_tools import execute_reference_tool

        target = tmp_path / ".env"
        target.write_text("ANTHROPIC_API_KEY=sk-not-a-real-key\n", encoding="utf-8")

        out = execute_reference_tool("read_file", json.dumps({"path": str(target)}))

        assert "cannot be read" in out
        assert "sk-not-a-real-key" not in out

    def test_real_list_dir_shows_entries(self, tmp_path):
        from agent.moa_reference_tools import execute_reference_tool

        (tmp_path / "one.txt").write_text("x", encoding="utf-8")

        out = execute_reference_tool("list_dir", json.dumps({"path": str(tmp_path)}))

        assert "one.txt" in out


class TestPresetSchema:
    def test_reference_tools_normalizes_and_defaults_off(self):
        from hermes_cli.moa_config import normalize_moa_config

        preset = normalize_moa_config({})["presets"]["default"]
        assert preset["reference_tools"] is False

        enabled = normalize_moa_config({"reference_tools": True})["presets"]["default"]
        assert enabled["reference_tools"] is True

        stringly = normalize_moa_config({"presets": {"default": {"reference_tools": "yes"}}})
        assert stringly["presets"]["default"]["reference_tools"] is True

    def test_reference_tools_survives_the_flat_view(self):
        from hermes_cli.moa_config import normalize_moa_config

        cfg = normalize_moa_config({"presets": {"default": {"reference_tools": True}}})

        assert cfg["reference_tools"] is True
