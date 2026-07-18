"""Tests for scripts/ci/timings_report.py.

The timing report is an observability job: per its own contract ("a missing
report must never redden the PR"), any unavailable-timings condition must
degrade to a placeholder report and exit 0. The load-bearing case is a
fork-originated PR run, where the AUTOFIX_BOT_PAT-backed GITHUB_TOKEN secret
is never provided — that must not crash.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "timings_report.py"
_spec = importlib.util.spec_from_file_location("timings_report", _PATH)
if _spec is None or _spec.loader is None:
    raise ImportError("Failed to load timings_report.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _run_main(tmp_path, monkeypatch, argv_extra=()):
    out_html = tmp_path / "report.html"
    out_json = tmp_path / "timings.json"
    out_summary = tmp_path / "summary.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "timings_report.py",
            "--baseline", str(tmp_path / "baseline.json"),
            "--output", str(out_html),
            "--json-out", str(out_json),
            "--summary-out", str(out_summary),
            *argv_extra,
        ],
    )
    return out_html, out_json, out_summary


class TestForkPrTokenAbsence:
    def test_missing_token_degrades_and_exits_zero(self, tmp_path, monkeypatch):
        # Fork PR shape: repo/run/sha present, token secret absent.
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.setenv("GITHUB_REPOSITORY", "NousResearch/hermes-agent")
        monkeypatch.setenv("GITHUB_RUN_ID", "123")
        monkeypatch.setenv("GITHUB_SHA", "deadbeef")
        out_html, out_json, out_summary = _run_main(tmp_path, monkeypatch)

        with pytest.raises(SystemExit) as excinfo:
            _mod.main()

        assert excinfo.value.code == 0, (
            "observability job must never redden the PR when the token secret "
            "is unavailable (the normal case for fork PR runs)"
        )
        # Degraded artifacts are written so the workflow's upload steps succeed.
        assert "unavailable" in out_html.read_text(encoding="utf-8")
        assert "unavailable" in out_summary.read_text(encoding="utf-8")
        # No JSON on purpose: an empty timings file must never be cached as
        # the main baseline.
        assert not out_json.exists()

    def test_empty_token_treated_like_missing(self, tmp_path, monkeypatch):
        # The workflow env-wires the secret even when empty: GITHUB_TOKEN="".
        monkeypatch.setenv("GITHUB_TOKEN", "")
        monkeypatch.setenv("GITHUB_REPOSITORY", "NousResearch/hermes-agent")
        monkeypatch.setenv("GITHUB_RUN_ID", "123")
        monkeypatch.setenv("GITHUB_SHA", "deadbeef")
        _run_main(tmp_path, monkeypatch)

        with pytest.raises(SystemExit) as excinfo:
            _mod.main()

        assert excinfo.value.code == 0

    def test_other_missing_env_still_raises(self, tmp_path, monkeypatch):
        # Only the token is expected to be absent on forks; a missing
        # GITHUB_REPOSITORY is a real configuration error and must stay loud.
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
        monkeypatch.setenv("GITHUB_RUN_ID", "123")
        monkeypatch.setenv("GITHUB_SHA", "deadbeef")
        _run_main(tmp_path, monkeypatch)

        with pytest.raises(ValueError, match="GITHUB_REPOSITORY"):
            _mod.main()
