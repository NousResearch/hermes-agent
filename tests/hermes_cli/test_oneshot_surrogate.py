"""Regression tests for the lone-surrogate output class (#80366).

Model text can carry lone UTF-16 surrogates (e.g. an OpenAI-compatible
provider returning a raw ``\\ud800`` escape). Any UTF-8 sink then raises
UnicodeEncodeError. Fragment fixes #80429 (@hudsonwa), #80374
(@rainbowgore) and #80409 (@RelaxJonh) each scrubbed the oneshot stdout
sink; the boundary fix sanitizes every runtime's result once at
``run_conversation`` egress, with the sink keeping a defensive scrub.
"""

import io

from agent.message_sanitization import _sanitize_result_egress
from hermes_cli.oneshot import _emit_response


def _utf8_stream():
    """A stream that actually UTF-8-encodes, like real stdout."""
    return io.TextIOWrapper(io.BytesIO(), encoding="utf-8")


def _read(stream):
    stream.flush()
    return stream.buffer.getvalue().decode("utf-8")


class TestCrashMechanism:
    def test_raw_write_of_lone_surrogate_crashes_utf8_stream(self):
        # Documents the failure mode the class is about: this is what
        # oneshot did on main, minus the scrub.
        stream = _utf8_stream()
        try:
            stream.write("hello \ud800 world")
            stream.flush()
        except UnicodeEncodeError:
            return
        raise AssertionError("expected UnicodeEncodeError from raw write")


class TestEmitResponse:
    def test_lone_high_surrogate_prints_replacement_char(self):
        stream = _utf8_stream()
        _emit_response(stream, "hello \ud800 world")
        assert _read(stream) == "hello \ufffd world\n"

    def test_lone_low_surrogate_prints_replacement_char(self):
        stream = _utf8_stream()
        _emit_response(stream, "text \udc00 end")
        assert _read(stream) == "text \ufffd end\n"

    def test_valid_text_and_astral_emoji_unchanged(self):
        stream = _utf8_stream()
        _emit_response(stream, "emoji: \U0001f600\n")
        assert _read(stream) == "emoji: \U0001f600\n"

    def test_trailing_newline_added_once(self):
        stream = _utf8_stream()
        _emit_response(stream, "no newline")
        assert _read(stream) == "no newline\n"


class TestResultEgressBoundary:
    def test_scrubs_final_response_error_and_interrupt_message(self):
        result = {
            "final_response": "ok \ud800",
            "error": "bad \udfff turn",
            "interrupt_message": "\ud800",
            "messages": [{"role": "assistant", "content": "untouched \ud800"}],
        }
        out = _sanitize_result_egress(result)
        assert out is result
        assert result["final_response"] == "ok \ufffd"
        assert result["error"] == "bad \ufffd turn"
        assert result["interrupt_message"] == "\ufffd"
        # messages persistence has its own sanitization path; egress only
        # touches the text fields handed to sinks.
        assert result["messages"][0]["content"] == "untouched \ud800"

    def test_codex_runtime_shaped_result(self):
        # codex_runtime returns final_text verbatim with no ingestion scrub;
        # the boundary must cover it.
        result = {"final_response": "turn \ud800 text", "completed": True}
        assert _sanitize_result_egress(result)["final_response"] == "turn \ufffd text"

    def test_clean_result_untouched_and_fast(self):
        result = {"final_response": "all good", "error": None}
        assert _sanitize_result_egress(result)["final_response"] == "all good"
        assert result["error"] is None

    def test_non_dict_and_missing_fields_pass_through(self):
        assert _sanitize_result_egress(None) is None
        assert _sanitize_result_egress("s") == "s"
        assert _sanitize_result_egress({}) == {}
