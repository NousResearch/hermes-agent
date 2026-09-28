"""Global CLI flags must not hide a dashboard's launchd supervisor."""
import os
import plistlib

import pytest

from hermes_cli import main_dashboard


@pytest.mark.parametrize("command,expected", [
    ("python -m hermes_cli.main --profile research --tui dashboard --no-open --host 127.0.0.1 --port 9119", ("dashboard", "127.0.0.1", 9119)),
    ("python -m hermes_cli.main -p default dashboard --port 9120 --host 192.0.2.10 --open-profile research --no-open", ("dashboard", "192.0.2.10", 9120)),
    ("hermes --profile=research serve --port=9120", ("serve", "127.0.0.1", 9120)),
    ("hermes dashboard", ("dashboard", "127.0.0.1", 9119)),
    ("hermes gateway run", None),
    ("python unrelated.py dashboard", None),
])
def test_dashboard_runtime_with_global_profile_flags(command, expected):
    assert main_dashboard._parse_dashboard_runtime(command) == expected


@pytest.mark.platforms("macos")
@pytest.mark.parametrize("profile_flags", [["--profile", "research", "--tui"], ["-p", "default"]])
def test_loaded_launchd_job_with_profile_flags(tmp_path, monkeypatch, profile_flags):
    from hermes_cli import gateway

    label = "example.hermes.dashboard"
    argv = ["python", "-m", "hermes_cli.main", *profile_flags,
            "dashboard", "--host", "127.0.0.1", "--port", "9119"]
    (tmp_path / "dashboard.plist").write_bytes(plistlib.dumps({
        "Label": label, "ProgramArguments": argv,
    }))
    domain = f"gui/{os.getuid()}"
    probes = []

    def loaded_service(candidate_domain, candidate_label):
        probes.append((candidate_domain, candidate_label))
        return (True, 12345)

    monkeypatch.setattr(gateway, "_launchd_print_service_pid", loaded_service)
    jobs = main_dashboard._loaded_launchd_backend_jobs([("agent", tmp_path)])
    assert jobs == [(domain, label, argv, 12345)]
    assert probes == [(domain, label)]
