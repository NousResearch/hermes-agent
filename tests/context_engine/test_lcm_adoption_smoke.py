"""Repeatable fork LCM isolated adoption smoke gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import probe_hermes_lcm_isolated as probe


REQUIRED_CHECKS = [
    "load+identity",
    "normal-chat/tool-ingestion",
    "threshold-compaction",
    "grep/describe/expand-byte-exact-recall",
    "bad-id-loud-error",
    "reset-semantics",
    "fail-open",
]


def test_isolated_smoke_runs_all_required_checks(tmp_path, monkeypatch) -> None:
    profile_dir = tmp_path / "temp-profile"
    monkeypatch.setenv("HERMES_HOME", str(profile_dir))

    checks = probe.run_smoke(probe.FORK_PLUGIN_DIR, profile_dir)

    assert [check.name for check in checks] == REQUIRED_CHECKS
    assert all(check.ok for check in checks), "\n".join(check.line() for check in checks)


def test_probe_exits_zero_and_writes_report_when_all_checks_pass(tmp_path, monkeypatch) -> None:
    profile_dir = tmp_path / "temp-profile"
    out_path = tmp_path / "reports" / "hermes-lcm-adoption-smoke.md"
    monkeypatch.setenv("HERMES_HOME", str(profile_dir))

    rc = probe.main(["--profile-dir", str(profile_dir), "--out", str(out_path)])

    assert rc == 0
    report = out_path.read_text(encoding="utf-8")
    assert "GO (isolated smoke clean)" in report
    assert "7/7 checks passed" in report
    for name in REQUIRED_CHECKS:
        assert name in report


def test_probe_exits_nonzero_if_any_smoke_check_fails(tmp_path, monkeypatch) -> None:
    profile_dir = tmp_path / "temp-profile"
    out_path = tmp_path / "reports" / "failed.md"
    monkeypatch.setenv("HERMES_HOME", str(profile_dir))
    monkeypatch.setattr(
        probe,
        "run_smoke",
        lambda plugin_dir, profile_dir: [probe.Check("forced-failure", False, "synthetic red")],
    )

    rc = probe.main(["--profile-dir", str(profile_dir), "--out", str(out_path)])

    assert rc == 1
    report = out_path.read_text(encoding="utf-8")
    assert "BLOCKED (smoke failure)" in report
    assert "0/1 checks passed" in report


@pytest.mark.parametrize("root_name", ["plugins", "profiles"])
def test_probe_refuses_live_plugin_and_profile_paths(root_name) -> None:
    hermes_roots = probe._hermes_roots()
    assert hermes_roots
    live_path = hermes_roots[0] / root_name / "not-a-temp-profile"

    with pytest.raises(RuntimeError, match="REFUSING live profile/plugin path"):
        probe._guard_isolated_profile(live_path)


def test_probe_allows_explicit_staging_profile_under_worktree() -> None:
    staging_profile = probe.WORKTREE_ROOT / "staging" / "lcm-profile"

    probe._guard_isolated_profile(staging_profile)
