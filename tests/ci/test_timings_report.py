"""Tests for scripts/ci/timings_report.py.

The CI timing report is an *observability* job: its own contract is that "a
missing report must never redden the PR". A fork PR gets no repo secrets, so
the workflow's ``AUTOFIX_BOT_PAT`` is empty and the job can receive an unset
``GITHUB_TOKEN``. The script must then degrade gracefully (exit 0 with a
placeholder report/summary), not crash with "missing environment variable
GITHUB_TOKEN" and fail every fork PR.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "timings_report.py"
_spec = importlib.util.spec_from_file_location("timings_report", _PATH)
if _spec is None or _spec.loader is None:
    raise ImportError("Failed to load timings_report.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _run_main(monkeypatch, tmp_path, *, token):
    summary = tmp_path / "summary.md"
    output = tmp_path / "report.html"
    json_out = tmp_path / "timings.json"
    if token is None:
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    else:
        monkeypatch.setenv("GITHUB_TOKEN", token)
    monkeypatch.setenv("GITHUB_REPOSITORY", "org/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_SHA", "deadbeef")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "timings_report.py",
            "--summary-out", str(summary),
            "--output", str(output),
            "--json-out", str(json_out),
        ],
    )
    return summary, output, json_out


def test_missing_github_token_degrades_instead_of_crashing(monkeypatch, tmp_path):
    """Empty GITHUB_TOKEN (fork PR) must exit 0 with a placeholder report, not
    raise ``ValueError: missing environment variable GITHUB_TOKEN``."""
    summary, output, json_out = _run_main(monkeypatch, tmp_path, token=None)

    with pytest.raises(SystemExit) as exc_info:
        _mod.main()

    assert exc_info.value.code == 0
    # A placeholder HTML report + a degraded summary are emitted...
    assert output.exists()
    assert "unavailable" in summary.read_text(encoding="utf-8").lower()
    # ...but NO JSON, so an empty run can never be cached as the main baseline.
    assert not json_out.exists()


def test_empty_string_github_token_also_degrades(monkeypatch, tmp_path):
    """A present-but-empty GITHUB_TOKEN (``AUTOFIX_BOT_PAT`` resolving to '')
    is treated the same as unset."""
    summary, output, _ = _run_main(monkeypatch, tmp_path, token="")

    with pytest.raises(SystemExit) as exc_info:
        _mod.main()

    assert exc_info.value.code == 0
    assert output.exists()
