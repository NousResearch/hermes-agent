"""Tests for the structured Hermes system doctor dashboard."""

from __future__ import annotations

from argparse import Namespace
from datetime import datetime, timezone
import json
from types import SimpleNamespace

import pytest


FIXED_NOW = datetime(2026, 5, 12, 0, 0, tzinfo=timezone.utc)


def _entry_by_name(report, name):
    matches = [entry for entry in report.entries if entry.name == name]
    assert matches, f"missing entry {name!r}"
    return matches[0]


def test_report_counts_runtime_stores_without_live_services(tmp_path, monkeypatch):
    from hermes_cli import system_doctor

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(system_doctor.shutil, "which", lambda cmd: "/usr/bin/codex" if cmd == "codex" else None)

    vault_rules = tmp_path / "Obsidian Vault" / "90. setting" / "Vault 운영 원칙.md"
    vault_rules.parent.mkdir(parents=True)
    vault_rules.write_text("# rules\n", encoding="utf-8")
    (home / "config.yaml").write_text(
        "obsidian:\n"
        f"  vault_path: {json.dumps(str(tmp_path / 'Obsidian Vault'))}\n",
        encoding="utf-8",
    )

    pending_dir = home / "pending_interactions"
    pending_dir.mkdir()
    (pending_dir / "records.json").write_text(
        json.dumps(
            {
                "records": [
                    {"status": "open", "expires_at": "2026-05-13T00:00:00Z"},
                    {"status": "open", "expires_at": "2026-05-11T00:00:00Z"},
                    {"status": "expired", "expires_at": "2026-05-10T00:00:00Z"},
                ]
            }
        ),
        encoding="utf-8",
    )

    governance_dir = home / "memory_governance"
    governance_dir.mkdir()
    (governance_dir / "review_queue.json").write_text(
        json.dumps({"items": [{"status": "pending_review"}, {"status": "approved"}]}),
        encoding="utf-8",
    )

    cron_dir = home / "cron"
    cron_dir.mkdir()
    (cron_dir / "jobs.json").write_text(
        json.dumps({"jobs": [{"enabled": True}, {"enabled": False}]}),
        encoding="utf-8",
    )

    codex_config = tmp_path / "codex-config.toml"
    codex_config.write_text("[features]\ngoals = true\n", encoding="utf-8")

    report = system_doctor.build_system_doctor_report(
        hermes_home=home,
        codex_config_path=codex_config,
        now=FIXED_NOW,
    )

    assert _entry_by_name(report, "config.yaml").status == system_doctor.STATUS_OK
    assert _entry_by_name(report, "Obsidian vault rules").status == system_doctor.STATUS_OK
    assert _entry_by_name(report, "Pending interaction store").detail == "total=3 active=1 expired=2"
    assert _entry_by_name(report, "Memory governance queue").detail == "total=2 pending_review=1"
    assert _entry_by_name(report, "Cron storage").detail == "total=2 enabled=1 disabled=1"
    assert _entry_by_name(report, "codex command").status == system_doctor.STATUS_OK
    assert _entry_by_name(report, "Codex goals feature flag").detail == "goals flag present"


def test_report_marks_corrupt_json_stores_as_fail(tmp_path, monkeypatch):
    from hermes_cli import system_doctor

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    for relative in (
        "pending_interactions/records.json",
        "memory_governance/review_queue.json",
        "cron/jobs.json",
    ):
        path = home / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")

    report = system_doctor.build_system_doctor_report(
        hermes_home=home,
        codex_config_path=tmp_path / "missing-codex.toml",
        now=FIXED_NOW,
    )

    assert _entry_by_name(report, "Pending interaction store").status == system_doctor.STATUS_FAIL
    assert _entry_by_name(report, "Memory governance queue").status == system_doctor.STATUS_FAIL
    assert _entry_by_name(report, "Cron storage").status == system_doctor.STATUS_FAIL
    assert report.exit_code(fail_on_fail=False) == 0
    assert report.exit_code(fail_on_fail=True) == 1


def test_dashboard_renderer_redacts_secret_like_values():
    from hermes_cli.system_doctor import (
        STATUS_FAIL,
        SystemDoctorEntry,
        SystemDoctorReport,
        render_system_doctor_dashboard,
    )

    report = SystemDoctorReport(
        generated_at="2026-05-12T00:00:00Z",
        hermes_home="/tmp/.hermes",
        entries=(
            SystemDoctorEntry(
                name="Secret-bearing check",
                status=STATUS_FAIL,
                detail="OPENAI_API_KEY=sk-test-secret-value and Bearer abcdefghijklmnop",
                remediation="Remove PASSWORD=hunter2 from logs",
                category="Secrets",
            ),
        ),
    )

    out = render_system_doctor_dashboard(report)

    assert "sk-test-secret-value" not in out
    assert "Bearer abcdefghijklmnop" not in out
    assert "hunter2" not in out
    assert "[REDACTED]" in out


def test_honcho_reachability_is_opt_in(tmp_path, monkeypatch):
    from hermes_cli import system_doctor
    from plugins.memory.honcho import client as honcho_client

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")

    fake_config = SimpleNamespace(
        enabled=True,
        api_key="configured-but-not-rendered",
        base_url=None,
        workspace_id="hermes-test",
        recall_mode="hybrid",
        write_frequency="async",
    )
    monkeypatch.setattr(honcho_client.HonchoClientConfig, "from_global_config", lambda: fake_config)
    monkeypatch.setattr(honcho_client, "resolve_config_path", lambda: home / "honcho.json")

    def fail_if_called(_config):
        raise AssertionError("reachability should not be checked by default")

    monkeypatch.setattr(honcho_client, "get_honcho_client", fail_if_called)

    report = system_doctor.build_system_doctor_report(
        hermes_home=home,
        codex_config_path=tmp_path / "missing-codex.toml",
        now=FIXED_NOW,
    )

    assert _entry_by_name(report, "Honcho configured").status == system_doctor.STATUS_OK
    reachable = _entry_by_name(report, "Honcho reachable")
    assert reachable.status == system_doctor.STATUS_WARN
    assert "not checked" in reachable.detail


def test_doctor_dashboard_only_can_exit_nonzero(monkeypatch, capsys):
    from hermes_cli import doctor as doctor_mod
    from hermes_cli import system_doctor

    def fake_dashboard(*, fail_on_fail, check_honcho_reachability):
        assert fail_on_fail is True
        assert check_honcho_reachability is False
        print("# fake dashboard")
        return 1

    monkeypatch.setattr(system_doctor, "print_system_doctor_dashboard", fake_dashboard)

    with pytest.raises(SystemExit) as exc:
        doctor_mod.run_doctor(
            Namespace(
                fix=False,
                dashboard_only=True,
                dashboard_fail_exit=True,
                check_dashboard_reachability=False,
            )
        )

    assert exc.value.code == 1
    assert "# fake dashboard" in capsys.readouterr().out
