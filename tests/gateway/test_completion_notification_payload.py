"""The completion notification must never deliver an empty payload.

A process whose output IS the payload — the ``message_agent`` delivery runner prints one
``json.dumps({...})`` line carrying a teammate's reply — used to be cut to nothing: the
builder kept the last 2000 chars and snapped forward to the first newline in that window,
and a single-line payload longer than 2000 chars has exactly one newline (the last one).
The sender received ``showing last 0 chars`` and concluded its teammate never answered
(#151053, #150617, #152075).

The transform/redact seams are stubbed out so the assertions describe the truncation rule
itself, not what a redactor happens to do to a given payload.
"""

import json
import types

from gateway.run_notifications import GatewayNotificationsMixin
from tools.process_registry import COMPLETION_OUTPUT_CHARS


class _RawOutput:
    """Make the builder see the process output verbatim (no plugin transform, no redaction)."""

    def __enter__(self):
        import agent.redact
        import tools.process_registry as registry
        self._restore = (agent.redact.redact_terminal_output, registry.transform_process_output)
        agent.redact.redact_terminal_output = lambda output, command="": output
        registry.transform_process_output = lambda output, **kwargs: output
        return self

    def __exit__(self, *exc):
        import agent.redact
        import tools.process_registry as registry
        agent.redact.redact_terminal_output, registry.transform_process_output = self._restore
        return False


def _session(output: str, *, completion_output_chars: int = 0):
    return types.SimpleNamespace(
        command="python3 bot_mode_dm.py --run-delivery query-file dm.txt",
        exit_code=0,
        task_id="",
        output_buffer=output,
        completion_output_chars=completion_output_chars,
    )


def _build(session) -> str:
    with _RawOutput():
        return GatewayNotificationsMixin._build_process_completion_event({}, session, "proc_test")["output"]


def _body(output: str) -> str:
    """The payload a notification carries, without its cut marker."""
    if output.startswith("[… output truncated"):
        return output.split("chars]\n", 1)[1]
    return output


REPLY = "關羽，複驗完成。\n" + "| 判準 | 結果 |\n" * 200 + "END-OF-REPORT"
ONE_LINE = json.dumps({"outcome": "@huangzhong answered.", "reply": REPLY, "status": "settled"},
                      ensure_ascii=False) + "\n"


def test_single_line_payload_survives_the_default_window():
    """Regression: a one-line payload longer than the window used to arrive as 0 chars."""
    assert len(ONE_LINE) > COMPLETION_OUTPUT_CHARS
    body = _body(_build(_session(ONE_LINE)))
    assert body.strip(), "completion notification carried no payload"
    assert "END-OF-REPORT" in body, "the tail must carry the end of the teammate's reply"


def test_single_line_payload_sized_by_the_caller():
    """A bot DM asks for a message-sized window; the whole reply must fit."""
    out = _build(_session(ONE_LINE, completion_output_chars=18000))
    assert out == ONE_LINE, "a payload that fits the requested window must not be marked as cut"
    assert json.loads(out)["reply"] == REPLY


def test_multiline_output_still_snaps_to_a_line_boundary():
    """The #23284 behaviour (never start mid-line) is unchanged for ordinary logs."""
    body = _body(_build(_session("".join(f"line {i} ok\n" for i in range(600)))))
    assert body, "multi-line tail must not be empty"
    assert body.startswith("line "), body[:40]
    assert len(body) <= COMPLETION_OUTPUT_CHARS


def test_short_output_passes_through_untouched():
    assert _build(_session('{"a": 1}\n')) == '{"a": 1}\n'
