"""Oneshot stdout must survive lone UTF-16 surrogates in model text (#80366)."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


def test_oneshot_replaces_lone_surrogate_and_exits_zero():
    """hermes -z must print U+FFFD and exit 0 when the model returns U+D800."""
    program = textwrap.dedent(
        """
        import hermes_cli.oneshot as oneshot

        dirty = "answer \\ud800 here"
        oneshot._run_agent = lambda *args, **kwargs: (
            dirty,
            {"final_response": dirty, "failed": False, "partial": False, "completed": True},
        )
        raise SystemExit(oneshot.run_oneshot("hello"))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    # U+FFFD as UTF-8; no raw surrogate bytes
    assert "\ufffd".encode("utf-8") in result.stdout
    assert b"answer " in result.stdout
    assert b" here\n" in result.stdout
    assert b"Traceback" not in result.stderr
