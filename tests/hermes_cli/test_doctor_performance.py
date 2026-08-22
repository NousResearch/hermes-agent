"""Behavior checks for the response-performance doctor section."""

import json

from hermes_cli import doctor


def _patch_config(monkeypatch, config):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    monkeypatch.setattr(
        "hermes_cli.prompt_size.compute_prompt_breakdown",
        lambda _platform: {
            "system_prompt": {"bytes": 40_000},
            "tools": {"json_bytes": 30_000},
        },
    )


def test_performance_check_flags_long_timeout_nonstreaming_and_overlap(tmp_path, monkeypatch, capsys):
    home = tmp_path / ".hermes"
    cron_dir = home / "cron"
    cron_dir.mkdir(parents=True)
    (cron_dir / "jobs.json").write_text(
        json.dumps({
            "jobs": [{
                "id": "mail",
                "name": "Mail monitor",
                "enabled": True,
                "no_agent": False,
                "schedule": {"kind": "interval", "minutes": 10},
            }],
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(doctor, "HERMES_HOME", home)
    _patch_config(monkeypatch, {
        "agent": {"gateway_timeout": 1800},
        "streaming": {"enabled": False},
    })

    issues = []
    doctor._check_performance_risks(issues)

    out = capsys.readouterr().out
    assert "idle timeout 30m" in out
    assert "Streaming is disabled" in out
    assert "Mail monitor" in out
    assert len(issues) == 3


def test_performance_check_accepts_bounded_nonoverlapping_configuration(tmp_path, monkeypatch, capsys):
    home = tmp_path / ".hermes"
    cron_dir = home / "cron"
    cron_dir.mkdir(parents=True)
    (cron_dir / "jobs.json").write_text(
        json.dumps({
            "jobs": [{
                "id": "mail",
                "enabled": True,
                "no_agent": False,
                "schedule": {"kind": "interval", "minutes": 10},
            }],
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(doctor, "HERMES_HOME", home)
    _patch_config(monkeypatch, {
        "agent": {"gateway_timeout": 300},
        "streaming": {"enabled": True},
    })

    issues = []
    doctor._check_performance_risks(issues)

    out = capsys.readouterr().out
    assert "Gateway idle timeout" in out
    assert "Response streaming enabled" in out
    assert "No interval cron jobs can overlap" in out
    assert issues == []


def test_performance_check_flags_large_fixed_prompt(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(doctor, "HERMES_HOME", tmp_path / ".hermes")
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"agent": {"gateway_timeout": 300}, "streaming": {"enabled": True}},
    )
    monkeypatch.setattr(
        "hermes_cli.prompt_size.compute_prompt_breakdown",
        lambda _platform: {
            "system_prompt": {"bytes": 100_000},
            "tools": {"json_bytes": 60_000},
        },
    )

    issues = []
    doctor._check_performance_risks(issues)

    assert "large fixed prompt" in capsys.readouterr().out
    assert any("prompt-size" in issue for issue in issues)
