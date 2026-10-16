"""#131786: completion-event tail must not be emptied by its own trailing newline.

The 2000-char tail window snaps to the nearest *preceding* newline so
notifications never start mid-line (#23284). When the only newline inside the
window is the terminating one (single-line JSON / single-line results), the
snap dropped every content byte and announced "showing last 0 chars".
"""

from types import SimpleNamespace

from gateway.run_notifications import GatewayNotificationsMixin


def _session(output_buffer: str) -> SimpleNamespace:
    return SimpleNamespace(
        output_buffer=output_buffer,
        exit_code=0,
        command="python gen.py",
        task_id="",
        started_at=None,
        parent_session_id="",
    )


def _completion_event(output_buffer: str) -> dict:
    return GatewayNotificationsMixin._build_process_completion_event(
        {"session_id": "proc_tail"}, _session(output_buffer), "proc_tail"
    )


def _marker_and_body(output: str) -> tuple[str, str]:
    marker, _, body = output.partition("\n")
    return marker, body


def test_single_line_tail_with_trailing_newline_keeps_content():
    """Gap: one long line whose only newline is the terminator keeps the window."""
    line = '{"result":"' + ("x" * 2200) + '"}\n'
    output = _completion_event(line)["output"]
    marker, body = _marker_and_body(output)
    assert "showing last 0 chars" not in marker
    assert body.strip(), "content bytes must survive the newline snap"
    assert "x" * 64 in body
    assert f"showing last {len(body)} chars" in marker


def test_whitespace_only_rest_after_first_newline_keeps_window():
    """Gap: a newline followed only by blank padding must not discard the line."""
    raw = ("A" * 1900) + "\n" + (" " * 120)
    output = _completion_event(raw)["output"]
    marker, body = _marker_and_body(output)
    assert body.strip(), "window reduced to blank padding"
    assert "A" * 64 in body
    assert f"showing last {len(body)} chars" in marker


def test_multiline_tail_still_snaps_to_preceding_newline():
    """Guard (#23284): multi-line output still starts on a clean line boundary."""
    raw = ("A" * 1500) + "\n" + ("B" * 600) + "\n" + ("C" * 50)
    window = raw[-2000:]
    expected_tail = window[window.find("\n") + 1:]
    output = _completion_event(raw)["output"]
    marker, body = _marker_and_body(output)
    assert body == expected_tail
    assert f"showing last {len(expected_tail)} chars" in marker


def test_single_line_without_trailing_newline_keeps_full_window():
    """Guard: no newline in the window keeps the whole 2000-char tail."""
    raw = "A" * 2500
    output = _completion_event(raw)["output"]
    marker, body = _marker_and_body(output)
    assert body == "A" * 2000
    assert "showing last 2000 chars" in marker


def test_short_output_passes_through_without_marker():
    """Guard: output under the limit is never wrapped in a truncation marker."""
    output = _completion_event("hello\nworld\n")["output"]
    assert output == "hello\nworld\n"
