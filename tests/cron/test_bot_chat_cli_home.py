"""The unowned CLI lane executes only at the home used for owner discovery."""
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from cron import scheduler_delivery as delivery
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.mark.parametrize("profile", ["beta", "default", ""])
def test_cli_keeps_discovered_home_when_launch_selection_changes(tmp_path, monkeypatch, profile):
    root = tmp_path / "custom"
    home = root / "profiles" / "beta" if profile == "beta" else root
    home.mkdir(parents=True)
    other = root / "profiles" / "other"
    other.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(root))
    token = set_hermes_home_override(str(root))
    real_run = subprocess.run
    seen = []

    def discover(target):
        assert target == home
        (root / "active_profile").write_text("other", encoding="utf-8")
        return None

    def run(argv, **kwargs):
        # Exercise the actual startup resolver with the production child env/flags.
        code = ('import json,sys; sys.argv=["hermes"]+json.loads(sys.argv[1]); '
                'import hermes_cli.main; from hermes_constants import get_hermes_home; '
                'print(json.dumps(str(get_hermes_home())))')
        result = real_run([sys.executable, "-c", code, json.dumps(argv[1:])],
                          env=kwargs["env"], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        seen.append(Path(json.loads(result.stdout.strip().splitlines()[-1])))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr("tools.bot_live_delivery.find_canonical_live_owner", discover)
    monkeypatch.setattr(delivery.shutil, "which", lambda _: "/bin/hermes")
    monkeypatch.setattr(delivery.subprocess, "run", run)
    try:
        assert delivery._deliver_to_bot_chat({"id": "job"}, "output", profile) is None
        assert seen == [home]
        assert not (tmp_path / ".hermes").exists()
    finally:
        reset_hermes_home_override(token)


@pytest.mark.parametrize("removed_during_discovery", [False, True])
def test_missing_destination_never_launches_or_recreates(tmp_path, monkeypatch, removed_during_discovery):
    root = tmp_path / "custom"
    root.mkdir()
    home = root / "profiles" / "beta"
    if removed_during_discovery:
        home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    token = set_hermes_home_override(str(root))
    run = Mock(return_value=subprocess.CompletedProcess([], 0, "", ""))

    def discover(target):
        if home.exists():
            home.rmdir()
        return None

    monkeypatch.setattr("tools.bot_live_delivery.find_canonical_live_owner", discover)
    monkeypatch.setattr(delivery.subprocess, "run", run)
    try:
        error = delivery._deliver_to_bot_chat({"id": "job"}, "output", "beta")
        assert error is not None and str(home) in error
        run.assert_not_called()
        assert not home.exists()
    finally:
        reset_hermes_home_override(token)
