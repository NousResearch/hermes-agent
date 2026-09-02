"""Setup-time disclosure for Kanban's dispatcher runtime contract."""

from argparse import Namespace
from pathlib import Path

from hermes_cli import kanban as kanban_cli


def test_init_discloses_automatic_worker_authority_and_uncapped_dispatch(
    monkeypatch, capsys, tmp_path: Path
):
    monkeypatch.setattr(kanban_cli.kb, "init_db", lambda: tmp_path / "kanban.db")
    monkeypatch.setattr(kanban_cli.kb, "list_profiles_on_disk", lambda: ["dev"])
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "kanban": {
                "dispatch_in_gateway": True,
                "auto_decompose": True,
            }
        },
    )

    assert kanban_cli._cmd_init(Namespace()) == 0

    output = capsys.readouterr().out
    assert "automatically claims Ready and Review tasks" in output
    assert "auto-decomposes Triage tasks" in output
    assert "one OS worker process per claim" in output
    assert "No worker concurrency cap is configured" in output
    assert "No per-tick spawn limit is configured" in output
    assert "enabled tools and approval policy" in output


def test_init_reports_manual_modes_and_effective_concurrency_caps(
    monkeypatch, capsys, tmp_path: Path
):
    monkeypatch.setattr(kanban_cli.kb, "init_db", lambda: tmp_path / "kanban.db")
    monkeypatch.setattr(kanban_cli.kb, "list_profiles_on_disk", lambda: [])
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "kanban": {
                "dispatch_in_gateway": False,
                "auto_decompose": False,
                "max_in_progress": "2",
                "max_in_progress_per_profile": 1,
                "max_spawn": "4",
            }
        },
    )

    assert kanban_cli._cmd_init(Namespace()) == 0

    output = capsys.readouterr().out
    assert "Gateway dispatch: disabled" in output
    assert "Triage orchestration: manual" in output
    assert "Running-worker concurrency caps: board=2, per-profile=1" in output
    assert "Per-tick spawn limit: 4 (not a running-worker cap)" in output


def test_init_honors_gateway_dispatch_environment_override(
    monkeypatch, capsys, tmp_path: Path
):
    monkeypatch.setattr(kanban_cli.kb, "init_db", lambda: tmp_path / "kanban.db")
    monkeypatch.setattr(kanban_cli.kb, "list_profiles_on_disk", lambda: [])
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "kanban": {
                "dispatch_in_gateway": True,
                "review_dispatch": False,
                "auto_decompose": True,
            }
        },
    )
    monkeypatch.setenv("HERMES_KANBAN_DISPATCH_IN_GATEWAY", "off")

    assert kanban_cli._cmd_init(Namespace()) == 0

    output = capsys.readouterr().out
    assert "Gateway dispatch: disabled" in output
    assert "Ready tasks wait for a manual dispatch pass" in output
    assert "Review tasks remain parked for human review" in output
    assert "configured but inactive while gateway dispatch is disabled" in output


def test_init_does_not_claim_dispatch_is_enabled_when_config_load_fails(
    monkeypatch, capsys, tmp_path: Path
):
    monkeypatch.setattr(kanban_cli.kb, "init_db", lambda: tmp_path / "kanban.db")
    monkeypatch.setattr(kanban_cli.kb, "list_profiles_on_disk", lambda: [])

    def fail_config_load():
        raise RuntimeError("cannot read config")

    monkeypatch.setattr("hermes_cli.config.load_config", fail_config_load)

    assert kanban_cli._cmd_init(Namespace()) == 0

    output = capsys.readouterr().out
    assert "Operational contract unavailable" in output
    assert "gateway dispatcher will remain disabled" in output
    assert "Resolve the config error before starting the gateway dispatcher" in output
    assert "Gateway dispatch: enabled" not in output
