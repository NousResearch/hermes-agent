from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[3]
DEPLOY_HELPER = ROOT / "ops" / "muncho" / "runtime" / "muncho-auto-deploy-release"


def _executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


@pytest.mark.parametrize(
    ("source_sha", "expected_returncode", "expected_status"),
    [
        ("b" * 40, 0, "deploy_rolled_back"),
        ("c" * 40, 1, "deploy_rollback_failed"),
    ],
)
def test_failed_activation_restores_only_the_exact_previous_release(
    tmp_path, source_sha, expected_returncode, expected_status
):
    releases = tmp_path / "releases"
    previous = releases / "hermes-agent-bbbbbbbbbbbb"
    target = releases / "hermes-agent-222222222222"
    for release in (previous, target):
        interpreter = release / ".venv" / "bin" / "python"
        interpreter.parent.mkdir(parents=True)
        _executable(interpreter, "#!/bin/sh\nexit 0\n")
    (previous / ".codex-source-commit").write_text(source_sha, encoding="ascii")
    active = tmp_path / "active"
    active.symlink_to(target, target_is_directory=True)
    state = tmp_path / "state"
    reports = tmp_path / "reports"
    hermes_home = tmp_path / "home"
    (hermes_home / "scripts").mkdir(parents=True)
    planned = hermes_home / "scripts" / "planned_gateway_restart.sh"
    _executable(planned, "#!/bin/sh\necho planned-stop-ok\n")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _executable(
        fake_bin / "systemctl",
        "#!/bin/sh\n"
        "if [ \"$1\" = restart ]; then exit 0; fi\n"
        "if [ \"$1\" = is-active ]; then echo active; exit 0; fi\n"
        "exit 2\n",
    )
    _executable(fake_bin / "date", "#!/bin/sh\necho 2026-07-14T10:30:00+03:00\n")
    _executable(fake_bin / "sudo", "#!/bin/sh\necho " + "b" * 40 + "\n")
    _executable(
        fake_bin / "mv",
        "#!/bin/sh\n"
        "if [ \"$1\" = -Tf ]; then /bin/rm -f \"$3\"; exec /bin/mv \"$2\" \"$3\"; fi\n"
        "exec /bin/mv \"$@\"\n",
    )
    _executable(
        fake_bin / "readlink",
        "#!/bin/sh\n"
        "if [ \"$1\" = -f ]; then exec python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' \"$2\"; fi\n"
        "exec /usr/bin/readlink \"$@\"\n",
    )

    command = """
source "$DEPLOY_HELPER"
RELEASES="$TEST_RELEASES"
ACTIVE_LINK="$TEST_ACTIVE"
STATE_DIR="$TEST_STATE"
REPORT_DIR="$TEST_REPORTS"
HERMES_HOME="$TEST_HOME"
PLANNED_RESTART_HELPER="$TEST_PLANNED"
DEPLOY_HEALTH_WAIT_SECONDS=0
rollback_release \
  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  101 \
  "$TEST_PREVIOUS" \
  target_health_failed \
  bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
"""
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "DEPLOY_HELPER": str(DEPLOY_HELPER),
        "TEST_RELEASES": str(releases),
        "TEST_ACTIVE": str(active),
        "TEST_STATE": str(state),
        "TEST_REPORTS": str(reports),
        "TEST_HOME": str(hermes_home),
        "TEST_PLANNED": str(planned),
        "TEST_PREVIOUS": str(previous),
    }
    completed = subprocess.run(
        ["bash", "-c", command],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=20,
    )

    assert completed.returncode == expected_returncode, completed.stderr
    receipt = json.loads((state / "auto-sync-deploy-latest.json").read_text())
    assert receipt["status"] == expected_status
    assert receipt["failure_stage"] == "target_health_failed"
    if expected_returncode == 0:
        assert active.resolve() == previous.resolve()
        assert receipt["restored_release"] == str(previous)
        assert receipt["restored_head"] == "b" * 40
    else:
        # A rollback candidate whose source marker disagrees with Git is not
        # activated, even when the currently active target is unhealthy.
        assert active.resolve() == target.resolve()
        assert receipt["rollback_error"] == "previous_release_identity_invalid"
        assert "restored_source" not in receipt
