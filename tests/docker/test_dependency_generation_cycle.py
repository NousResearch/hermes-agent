"""Real supervisor restarts, live-reader safety, and the Docker dashboard launch contract."""

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def container_args(image):
    return [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,exec,mode=1777",
        "--mount",
        f"type=bind,source={ROOT},target=/work,readonly",
        "-e",
        "PYTHONPATH=/work",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "--entrypoint",
        "/bin/sh",
        image,
    ]


def test_both_readers_finish_work_restart_and_automatically_reclaim_the_old_generation(
    built_image,
):
    result = subprocess.run(
        container_args(built_image)
        + [
            "-c",
            "exec /opt/hermes/.venv/bin/python /work/tests/docker/generation_lifecycle_probe.py",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=150,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["result"] == "PASS"
    assert receipt["real_s6_restarts"] == 2
    assert receipt["busy_work_preserved"] and receipt["old_collected_automatically"]


@pytest.mark.parametrize("root", [True, False])
def test_dashboard_entrypoint_declares_supervision_after_both_privilege_paths(
    built_image, root
):
    args = container_args(built_image)
    if not root:
        args[2:2] = ["--user", "10000:10000"]
    script = """
mkdir /tmp/probe-bin
chmod 755 /tmp/probe-bin
printf '#!/bin/sh\nprintf "%%s\\n" "$@"\n' > /tmp/probe-bin/hermes
chmod 755 /tmp/probe-bin/hermes
export PATH="/tmp/probe-bin:/command:$PATH" HERMES_DASHBOARD=true
exec /bin/sh /work/docker/s6-rc.d/dashboard/run
"""
    result = subprocess.run(
        args + ["-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    argv = result.stdout.splitlines()
    assert argv[0] == "dashboard"
    assert argv.count("--external-supervisor") == 1
    assert "--no-open" in argv
