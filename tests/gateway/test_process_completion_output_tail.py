"""Completion wake-up output tail for long process output (#156760)."""

import json

from gateway.run import GatewayRunner
from tools.process_registry import ProcessSession


def _completion_output(output: str) -> str:
    proc = ProcessSession(id="proc_tailtest", command="runner", output_buffer=output, started_at=1.0)
    proc.exit_code = 0
    event = GatewayRunner._build_process_completion_event({"session_id": proc.id}, proc, proc.id)
    return event["output"]


def test_single_long_line_keeps_its_tail():
    line = json.dumps({"outcome": "sent", "reply": "x" * 3000, "status": "ok"}) + "\n"
    out = _completion_output(line)
    assert "showing last 0 chars" not in out
    header, body = out.split("\n", 1)
    assert header == "[… output truncated — showing last 2000 chars]"
    assert body == line[-2000:]
    assert body.rstrip().endswith('"status": "ok"}')


def test_long_line_then_short_last_line_keeps_the_long_line():
    output = "y" * 3000 + "\ndone\n"
    out = _completion_output(output)
    assert "y" * 1000 in out
    assert out.endswith("done\n")


def test_many_lines_still_snap_to_a_line_start():
    output = "".join(f"line {i:04d} of output\n" for i in range(300))
    out = _completion_output(output)
    header, body = out.split("\n", 1)
    assert body.startswith("line ")
    assert body.endswith("line 0299 of output\n")
    assert header == f"[… output truncated — showing last {len(body)} chars]"


def test_short_output_is_unchanged():
    assert _completion_output("READY\n") == "READY\n"
