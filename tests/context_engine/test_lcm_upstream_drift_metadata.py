from __future__ import annotations

from pathlib import Path

import pytest

from scripts import check_lcm_upstream_drift as drift

CURRENT_METADATA = Path("plugins/context_engine/lcm/VENDORED_FROM.txt")


def _without_field(source: Path, field: str) -> str:
    lines = source.read_text(encoding="utf-8").splitlines()
    needle = f"  {field}:"
    return "\n".join(line for line in lines if not line.startswith(needle)) + "\n"


def test_parse_current_lcm_metadata_validates_required_provenance_fields() -> None:
    metadata = drift.parse_metadata(CURRENT_METADATA)

    assert metadata.source_repository == "github.com/stephenschoettler/hermes-lcm"
    assert metadata.source_commit == "03b74f84440be99164ce3e2cd929917bc9550bfe"
    assert metadata.vendored_commit == "03b74f84440be99164ce3e2cd929917bc9550bfe"
    assert metadata.vendored_version == "v0.16.2"
    assert metadata.ingest_audit_verdict == "PASS"
    assert metadata.last_upstream_security_check == "2026-06-16"
    assert metadata.checked_by.startswith("Apollo")
    assert metadata.next_check_due == "2026-09-16"


def test_metadata_parser_fails_when_required_field_is_missing(tmp_path: Path) -> None:
    broken = tmp_path / "VENDORED_FROM.txt"
    broken.write_text(_without_field(CURRENT_METADATA, "next_check_due"), encoding="utf-8")

    with pytest.raises(ValueError, match="next_check_due"):
        drift.parse_metadata(broken)


def test_offline_checker_passes_current_metadata_shape(capsys: pytest.CaptureFixture[str]) -> None:
    code = drift.main(["--metadata", str(CURRENT_METADATA), "--offline"])

    captured = capsys.readouterr()
    assert code == 0
    assert "PASS" in captured.out
    assert "offline metadata validation" in captured.out
    assert "FAIL" not in captured.out + captured.err


def test_checker_exits_nonzero_and_prints_fail_on_missing_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    broken = tmp_path / "VENDORED_FROM.txt"
    broken.write_text(_without_field(CURRENT_METADATA, "checked_by"), encoding="utf-8")

    code = drift.main(["--metadata", str(broken), "--offline"])

    captured = capsys.readouterr()
    assert code == 2
    assert "FAIL" in captured.err
    assert "checked_by" in captured.err


def test_live_checker_warns_on_upstream_drift_and_security_relevant_commits(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    upstream_report = drift.UpstreamReport(
        head_commit="ffffffffffffffffffffffffffffffffffffffff",
        tag_commit="03b74f84440be99164ce3e2cd929917bc9550bfe",
        commits=[
            drift.UpstreamCommit(
                sha="ffffffffffffffffffffffffffffffffffffffff",
                message="security: tighten credential redaction",
            )
        ],
        lookup_warnings=[],
    )
    monkeypatch.setattr(drift, "query_upstream", lambda _metadata: upstream_report)

    code = drift.main(["--metadata", str(CURRENT_METADATA)])

    captured = capsys.readouterr()
    assert code == 0
    assert "WARN" in captured.out
    assert "upstream HEAD differs from vendored commit" in captured.out
    assert "security-relevant upstream commit" in captured.out
    assert "ffffffffffff" in captured.out
