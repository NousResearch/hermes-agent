"""A failed cron run's on-disk output doc must carry the full traceback, not just the one-liner.

Regression for #104538: a run that died with ``RuntimeError: Connection error.`` wrote only that
bare one-line message under ``## Error`` in ``~/.hermes/cron/output/<id>/*.md`` — no traceback, so
an operator could not tell which connection failed or where. The traceback belongs in the on-disk
output doc only; the delivered alert stays shaped from the one-line error (verified separately by
``_compose_run_delivery``), so this must not change the run tuple's error field.
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cron.scheduler import _run_error_section


def _captured_traceback(exc_factory):
    """Return the ``traceback.format_exc()`` text for a raised-and-caught exception, mirroring the
    ``except Exception`` block in ``run_job`` (which formats the live exception context)."""
    try:
        raise exc_factory()
    except Exception:
        return traceback.format_exc()


class TestRunErrorSection:
    def test_includes_full_traceback_not_just_one_liner(self):
        def _boom():
            # An intermediate frame so the trace has real depth to preserve.
            raise RuntimeError("Connection error.")

        detail = _captured_traceback(_boom)
        section = _run_error_section("RuntimeError: Connection error.", detail)

        assert section.startswith("## Error\n\n```\n")
        assert section.endswith("```\n")
        # The whole point: the frames that identify *where* the connection failed are present,
        # not merely the final message line.
        assert "Traceback (most recent call last)" in section
        assert "_boom" in section
        assert "RuntimeError: Connection error." in section

    def test_falls_back_to_one_liner_when_no_detail(self):
        # traceback.format_exc() outside any handler is the sentinel "NoneType: None"; the helper
        # is also called defensively with empty detail — either way it degrades to the one-liner.
        for empty in ("", "   ", "\n"):
            section = _run_error_section("RuntimeError: Connection error.", empty)
            assert section == "## Error\n\n```\nRuntimeError: Connection error.\n```\n"

    def test_detail_is_stripped(self):
        section = _run_error_section("Boom", "\n\nTraceback...\nBoom\n\n")
        assert section == "## Error\n\n```\nTraceback...\nBoom\n```\n"
