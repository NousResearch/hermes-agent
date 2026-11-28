"""A shell hook's output is bounded before it reaches memory or the model.

A hook that prints without end (``yes``, a log tail) used to be buffered whole by
``_spawn`` until its timeout: one 6-second hook grew the agent past 50 GB RSS. Text a
hook hands the model (context, block reasons) is also shortened so one oversized answer
cannot overflow the context every turn. Real subprocesses, no mocks.
"""

import json
import sys
import time

import pytest

from agent import shell_hooks
from agent.shell_hooks import ShellHookSpec, _evaluate_result, _spawn

pytestmark = pytest.mark.platforms("posix")


def _script(tmp_path, body: str):
    path = tmp_path / "hook.py"
    path.write_text("import sys, json\nsys.stdin.read()\n" + body, encoding="utf-8")
    return f"{sys.executable} {path}"


def test_runaway_output_is_stopped_at_once_and_fails_closed(tmp_path):
    command = _script(tmp_path, "while True:\n    sys.stdout.write('y' * 65536)\n")
    spec = ShellHookSpec(event="pre_tool_call", command=command, timeout=30, fail_closed=True)
    started = time.monotonic()
    result = _spawn(spec, "{}")
    assert time.monotonic() - started < 10  # killed on overflow, not at the 30 s timeout
    assert "1 MiB" in result["error"] and len(result["stdout"]) <= shell_hooks._MAX_HOOK_STREAM_BYTES
    verdict = _evaluate_result(spec, result)
    assert verdict["action"] == "block" and "1 MiB" in verdict["message"]


def test_model_bound_context_is_shortened_with_a_note(tmp_path):
    size = shell_hooks._MAX_HOOK_TEXT_CHARS * 4
    command = _script(tmp_path, f"print(json.dumps({{'context': 'z' * {size}}}))\n")
    spec = ShellHookSpec(event="pre_llm_call", command=command, timeout=30)
    context = _evaluate_result(spec, _spawn(spec, "{}"))["context"]
    assert context.startswith("z" * 100)
    assert len(context) < shell_hooks._MAX_HOOK_TEXT_CHARS + 200
    assert f"{size:,} characters" in context


def test_a_quiet_background_helper_does_not_leave_reader_threads_behind(tmp_path):
    import threading
    helper = tmp_path / "helper.py"
    helper.write_text("import time\ntime.sleep(8)\n", encoding="utf-8")
    command = _script(tmp_path, f"import subprocess\nsubprocess.Popen([sys.executable, {str(helper)!r}])\n"
                                "print(json.dumps({'context': 'ok'}))\n")
    spec = ShellHookSpec(event="pre_llm_call", command=command, timeout=30)
    started = time.monotonic()
    assert _evaluate_result(spec, _spawn(spec, "{}"))["context"] == "ok"
    assert time.monotonic() - started < 10
    deadline = time.monotonic() + 3
    while any(t.name.startswith("hook-std") for t in threading.enumerate()) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not [t.name for t in threading.enumerate() if t.name.startswith("hook-std")]
