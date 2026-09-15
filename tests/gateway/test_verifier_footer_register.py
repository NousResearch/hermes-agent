"""Per-surface register for the file-mutation verifier footer (#97109).

Messaging channels get one neutral honesty line (count preserved; no paths,
tool names, or shell commands); developer surfaces (CLI transcript, logs, raw
gateway surfaces) keep the full footer byte-identical. The neutral line keeps
the ``File-mutation verifier:`` header shape so the TTS strip (#40772) still
silences it on voice paths.
"""

import pytest

from agent.turn_explainers import TurnExplainersMixin
from gateway.config import Platform
from gateway.run import _sanitize_gateway_final_response
from tools.tts_text_normalize import (
    downgrade_verifier_footer_for_messaging,
    strip_nonspoken_blocks,
)

FAILED = {
    "path/to/file": {"tool": "patch", "error_preview": "Could not find old_string xyz"},
    "/etc/secret.conf": {"tool": "write_file", "error_preview": "denied"},
}
BODY = "Done, I updated the files as requested."


def _full_text():
    footer = TurnExplainersMixin._format_file_mutation_failure_footer(FAILED)
    assert footer  # the fixture must actually carry a footer
    return BODY + "\n\n" + footer


class TestDowngradeHelper:
    def test_neutral_line_keeps_count_drops_detail(self):
        out = downgrade_verifier_footer_for_messaging(_full_text())
        assert BODY in out
        assert "2 file change(s) described above did not actually take effect." in out
        for leaked in ("path/to/file", "/etc/secret.conf", "[patch]", "[write_file]",
                        "git status", "read_file", "Could not find old_string", "\u2022"):
            assert leaked not in out

    def test_no_footer_passthrough(self):
        assert downgrade_verifier_footer_for_messaging(BODY) == BODY
        assert downgrade_verifier_footer_for_messaging("") == ""

    def test_idempotent(self):
        once = downgrade_verifier_footer_for_messaging(_full_text())
        assert downgrade_verifier_footer_for_messaging(once) == once

    def test_neutral_line_still_silenced_for_tts(self):
        out = downgrade_verifier_footer_for_messaging(_full_text())
        spoken = strip_nonspoken_blocks(out)
        assert "File-mutation verifier" not in spoken
        assert BODY in spoken


class TestSanitizePerSurface:
    @pytest.mark.parametrize("platform", ["telegram", "whatsapp", "slack", Platform.SIGNAL])
    def test_messaging_gets_neutral_register(self, platform):
        out = _sanitize_gateway_final_response(platform, _full_text())
        assert "2 file change(s) described above did not actually take effect." in out
        for leaked in ("path/to/file", "git status", "[patch]", "\u2022"):
            assert leaked not in out

    @pytest.mark.parametrize("platform", ["local", "api_server", "webhook"])
    def test_developer_surfaces_keep_full_footer(self, platform):
        raw = _full_text()
        assert _sanitize_gateway_final_response(platform, raw) == raw


class TestStreamFinalizeDowngrade:
    def _runner_with_source(self, platform):
        from types import SimpleNamespace

        from gateway.run_turn_runner import TurnRunner
        from gateway.turn_context import TurnContext

        ctx = TurnContext()
        ctx.source = SimpleNamespace(platform=platform)
        return TurnRunner(SimpleNamespace(), ctx)

    def test_messaging_stream_result_downgraded(self):
        runner = self._runner_with_source(Platform.TELEGRAM)
        result = {"final_response": _full_text(), "messages": [], "completed": True}
        runner._finish_stream_consumer(result, [], None)
        assert "2 file change(s) described above did not actually take effect." in result["final_response"]
        assert "git status" not in result["final_response"]

    def test_local_stream_result_untouched(self):
        runner = self._runner_with_source(Platform.LOCAL)
        raw = _full_text()
        result = {"final_response": raw, "messages": [], "completed": True}
        runner._finish_stream_consumer(result, [], None)
        assert result["final_response"] == raw
