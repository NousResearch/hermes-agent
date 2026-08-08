"""The zero-connected discovery warning must name why each server is out (#80938).

`get_mcp_status()` already records why every configured server is not
connected — a per-server ``status`` plus a recorded ``error`` for failures —
but the background-discovery warning threw all of it away and logged one
generic line. Operators could not tell an auth failure from a server that
never started, or from one that was simply disabled.
"""
from unittest.mock import patch

from hermes_cli.mcp_startup import (
    _CAUSE_MAX_CHARS,
    _CAUSE_MAX_SERVERS,
    _format_zero_connected_causes,
)


def _entry(name, status, *, error=None, connected=False):
    e = {"name": name, "status": status, "connected": connected, "tools": 0}
    if error is not None:
        e["error"] = error
    return e


class TestZeroConnectedCauses:
    def test_names_each_server_and_its_cause(self):
        out = _format_zero_connected_causes([
            _entry("calendar", "failed", error="401 Unauthorized"),
            _entry("maps", "connecting"),
            _entry("docs", "disabled"),
        ])

        assert "calendar [failed]: 401 Unauthorized" in out
        assert "maps [connecting]" in out
        assert "docs [disabled]" in out

    def test_distinguishes_failure_from_never_started(self):
        """The whole point: 'failed' and 'configured' must not read alike."""
        out = _format_zero_connected_causes([
            _entry("a", "failed", error="handshake timed out"),
            _entry("b", "configured"),
        ])

        assert "a [failed]: handshake timed out" in out
        assert "b [configured]" in out
        assert "b [configured]:" not in out  # no empty cause tail

    def test_connected_entries_are_skipped(self):
        out = _format_zero_connected_causes([
            _entry("live", "connected", connected=True),
            _entry("dead", "failed", error="refused"),
        ])

        assert "live" not in out
        assert "dead [failed]: refused" in out

    def test_returns_empty_when_nothing_to_add(self):
        """An empty add-on keeps the caller's original message intact."""
        assert _format_zero_connected_causes([]) == ""
        assert _format_zero_connected_causes(None) == ""
        assert _format_zero_connected_causes([_entry("x", "connected", connected=True)]) == ""

    def test_bounds_a_pathological_error_body(self):
        out = _format_zero_connected_causes([_entry("x", "failed", error="E" * 5000)])

        assert len(out) < _CAUSE_MAX_CHARS + 80
        assert "…" in out

    def test_bounds_the_server_count(self):
        entries = [_entry(f"s{i}", "failed", error="boom") for i in range(_CAUSE_MAX_SERVERS + 5)]

        out = _format_zero_connected_causes(entries)

        assert "(+5 more)" in out
        assert out.count("[failed]") == _CAUSE_MAX_SERVERS

    def test_error_bodies_are_redacted(self):
        """Causes are provider text — they must not carry a secret into logs."""
        with patch("agent.redact.redact_sensitive_text", return_value="[redacted]") as red:
            out = _format_zero_connected_causes(
                [_entry("x", "failed", error="Bearer sk-live-supersecret rejected")]
            )

        assert red.called
        assert "sk-live-supersecret" not in out
        assert "[redacted]" in out

    def test_withholds_the_body_when_redaction_is_unavailable(self):
        """Fail closed: no redactor means no error body, not a raw dump."""
        with patch("agent.redact.redact_sensitive_text", side_effect=RuntimeError("gone")):
            out = _format_zero_connected_causes(
                [_entry("x", "failed", error="token sk-live-supersecret")]
            )

        assert "sk-live-supersecret" not in out
        assert "redaction unavailable" in out

    def test_never_raises_on_malformed_status(self):
        """A diagnostics helper must not be able to break startup."""
        assert _format_zero_connected_causes(["not-a-dict", None, 42]) == ""
        assert _format_zero_connected_causes([{"connected": False}]) == " — ? [unknown]"

    def test_newlines_are_flattened(self):
        out = _format_zero_connected_causes([_entry("x", "failed", error="line1\nline2")])

        assert "\n" not in out
        assert "line1 line2" in out
