"""Regression coverage for cross-profile gateway credential diagnostics (#118388)."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import hermes_constants
from hermes_cli import doctor_state, profile_channels


def _profile_home(root: Path, name: str, token: str) -> Path:
    home = root if name == "default" else root / "profiles" / name
    home.mkdir(parents=True, exist_ok=True)
    (home / ".env").write_text(f"TELEGRAM_BOT_TOKEN={token}\n", encoding="utf-8")
    return home


def test_scan_reports_duplicate_platform_credential_with_paths_not_secret(monkeypatch, tmp_path):
    root = tmp_path / ".hermes"
    default = _profile_home(root, "default", "123:shared-secret")
    worker = _profile_home(root, "worker", "123:shared-secret")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)

    report = profile_channels.scan_local_profile_credential_collisions()

    assert [(collision.platform, collision.paths) for collision in report.collisions] == [
        ("telegram", (default, worker)),
    ]
    assert "shared-secret" not in report.format_for_display()
    assert str(default) in report.format_for_display()
    assert str(worker) in report.format_for_display()


def test_scan_ignores_distinct_platform_credentials(monkeypatch, tmp_path):
    root = tmp_path / ".hermes"
    _profile_home(root, "default", "123:default-secret")
    _profile_home(root, "worker", "123:worker-secret")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)

    report = profile_channels.scan_local_profile_credential_collisions()

    assert report.collisions == ()
    assert report.unreadable_paths == ()


def test_scan_reports_unreadable_profile_without_hiding_other_collisions(monkeypatch, tmp_path):
    default = tmp_path / ".hermes"
    worker = default / "profiles" / "worker"
    broken = default / "profiles" / "broken"
    monkeypatch.setattr(
        profile_channels,
        "_local_profile_homes",
        lambda: (("default", default), ("worker", worker), ("broken", broken)),
    )
    monkeypatch.setattr(
        profile_channels,
        "_profile_credential_claims",
        lambda home: (
            {("telegram", "fingerprint")} if home != broken else (_ for _ in ()).throw(OSError("denied"))
        ),
    )

    report = profile_channels.scan_local_profile_credential_collisions()

    assert report.collisions[0].platform == "telegram"
    assert report.unreadable_paths == (broken,)


def test_doctor_and_gateway_status_surface_collision_without_secret(monkeypatch, tmp_path):
    default = tmp_path / ".hermes"
    worker = default / "profiles" / "worker"
    report = profile_channels.LocalProfileCredentialCollisionReport(
        collisions=(profile_channels.ProfileCredentialCollision("telegram", (default, worker)),),
    )
    monkeypatch.setattr(profile_channels, "scan_local_profile_credential_collisions", lambda: report)

    doctor_output = io.StringIO()
    with redirect_stdout(doctor_output):
        findings = doctor_state._check_cross_profile_gateway_credentials(False)
    assert "Duplicate platform credentials" in doctor_output.getvalue()
    assert str(worker) in doctor_output.getvalue()
    assert findings.manual_issues

    from hermes_cli import gateway

    status_output = io.StringIO()
    with redirect_stdout(status_output):
        gateway._print_cross_profile_credential_warnings()
    assert "Duplicate platform credentials" in status_output.getvalue()
    assert "shared-secret" not in status_output.getvalue()
