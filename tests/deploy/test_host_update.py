"""litco-host-update: flip `current`, then restart slots one at a time behind their drain (FIRM_AGENT_HOST 3.6).

The real sequence runs in a scratch tree. systemctl, curl, flock and uv are recording fakes;
the file work (release copy, symlink flip, unit install) is real.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from tests.host._load import HOST

SCRIPT = HOST / "litco-host-update"

FAKE_SYSTEMCTL = """#!/bin/sh
echo "systemctl $*" >> "$LOG"
case "$1" in
  list-units) for s in $SLOTS; do echo "litco-agent@$s.service loaded active running litco-agent matter slot $s"; done ;;
esac
exit 0
"""
# Answers /health for every port except $DOWN_PORT.
FAKE_CURL = """#!/bin/sh
for a in "$@"; do url="$a"; done
echo "curl $url" >> "$LOG"
case "$url" in *":$DOWN_PORT/"*) exit 7 ;; esac
exit 0
"""
# `uv sync` builds a venv whose python accepts the import check.
FAKE_UV = """#!/bin/sh
echo "uv $* (in $(pwd))" >> "$LOG"
mkdir -p "$UV_PROJECT_ENVIRONMENT/bin"
printf '#!/bin/sh\\nexit 0\\n' > "$UV_PROJECT_ENVIRONMENT/bin/python"
chmod +x "$UV_PROJECT_ENVIRONMENT/bin/python"
"""


@pytest.fixture
def host(tmp_path) -> dict:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in {"systemctl": FAKE_SYSTEMCTL, "curl": FAKE_CURL, "uv": FAKE_UV,
                       "flock": "#!/bin/sh\nexit 0\n"}.items():
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    prefix = tmp_path / "opt"
    (prefix / "releases").mkdir(parents=True)
    (prefix / "TOPOLOGY").write_text("machine\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    dirs = {name: tmp_path / name for name in ("systemd", "sbin_local")}
    for d in dirs.values():
        d.mkdir()
    log = tmp_path / "calls.log"
    log.touch()
    env = {k: v for k, v in os.environ.items() if not k.startswith("LITCO_")}
    env.update(PATH=f"{bin_dir}:/usr/bin:/bin", LOG=str(log), SLOTS="", DOWN_PORT="none",
               LITCO_PREFIX=str(prefix), LITCO_RUN_DIR=str(run_dir), LITCO_SYSTEMD_DIR=str(dirs["systemd"]),
               LITCO_SBIN_DIR=str(dirs["sbin_local"]),
               LITCO_UPDATE_LOCK=str(tmp_path / "update.lock"), LITCO_HOST_UPDATE_AS_ROOT="false")
    return {"env": env, "prefix": prefix, "run": run_dir, "log": log, "tmp": tmp_path, **dirs}


def make_source(tmp_path: Path, name: str) -> Path:
    """A release tree carrying the files the update installs, each tagged with the release name."""
    src = tmp_path / f"src-{name}"
    (src / "deploy" / "host").mkdir(parents=True)
    for f in ("litco-agent@.service", "litco-supervisor.service", "litco-supervisor", "litco-host-update",
              "litco-slot-probe", "litco-slot-import"):
        (src / "deploy" / "host" / f).write_text(f"# {f} from {name}\n")
    return src


def add_slot(host, slot: str, port: int) -> None:
    (host["run"] / f"{slot}.env").write_text(f"LITCO_MATTER_ID=m-{slot}\nLITCO_SLOT_PORT={port}\n"
                                             "LITCO_HOST_SECRET=hs-secret-never-read\n")
    host["env"]["SLOTS"] = (host["env"]["SLOTS"] + f" {slot}").strip()


def update(host, *args):
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env=host["env"], timeout=60)


def calls(host) -> list[str]:
    return host["log"].read_text().splitlines()


def test_update_builds_the_release_flips_current_and_restarts_slots_one_at_a_time(host):
    add_slot(host, "aa11", 8800)
    add_slot(host, "bb22", 8801)
    out = update(host, "--ref", "host-2", "--source", str(make_source(host["tmp"], "host-2")))
    assert out.returncode == 0, out.stderr + out.stdout
    rel = host["prefix"] / "releases" / "host-2"
    assert os.readlink(host["prefix"] / "current") == "releases/host-2"
    assert (rel / ".litco-release-ok").read_text().strip() == "host-2"
    assert (host["systemd"] / "litco-agent@.service").read_text() == "# litco-agent@.service from host-2\n"
    # Where litco-supervisor.service runs it from, not /usr/local/bin.
    assert (host["sbin_local"] / "litco-supervisor").read_text() == "# litco-supervisor from host-2\n"
    assert (host["sbin_local"] / "litco-host-update").exists()
    for tool in ("litco-slot-probe", "litco-slot-import"):
        assert (host["sbin_local"] / tool).read_text() == f"# {tool} from host-2\n"

    log = calls(host)
    assert any(c.startswith("uv sync --frozen --no-dev --extra messaging") and c.endswith(f"(in {rel})") for c in log)
    ordered = [c for c in log if re.match(r"systemctl (restart|daemon-reload|try-restart)|curl ", c)]
    assert ordered == [
        "systemctl daemon-reload",
        "systemctl try-restart litco-supervisor.service",
        "systemctl restart litco-agent@aa11.service",
        "curl http://127.0.0.1:8800/health",
        "systemctl restart litco-agent@bb22.service",
        "curl http://127.0.0.1:8801/health",
    ]


def test_a_slot_that_does_not_come_back_stops_the_rollout(host):
    add_slot(host, "aa11", 8800)
    add_slot(host, "bb22", 8801)
    host["env"]["DOWN_PORT"] = "8800"
    out = update(host, "--ref", "host-2", "--source", str(make_source(host["tmp"], "host-2")),
                 "--health-timeout", "0")
    assert out.returncode == 1
    assert "stopping the rollout at slot aa11" in out.stderr
    assert "systemctl restart litco-agent@bb22.service" not in calls(host)


def test_rolling_back_reuses_a_built_release_and_leaves_the_supervisor_alone(host):
    src = make_source(host["tmp"], "host-1")
    assert update(host, "--ref", "host-1", "--source", str(src)).returncode == 0
    assert update(host, "--ref", "host-2", "--source", str(make_source(host["tmp"], "host-2"))).returncode == 0
    add_slot(host, "aa11", 8800)
    host["log"].write_text("")
    # Back to host-1: already built, so no uv; the supervisor's code changes back, so it restarts.
    out = update(host, "--ref", "host-1")
    assert out.returncode == 0, out.stderr
    assert "already built; reusing it" in out.stdout
    assert os.readlink(host["prefix"] / "current") == "releases/host-1"
    assert not any(c.startswith("uv ") for c in calls(host))
    assert "systemctl try-restart litco-supervisor.service" in calls(host)
    # The same release again: nothing the supervisor runs changed, so it is not restarted.
    host["log"].write_text("")
    assert update(host, "--ref", "host-1").returncode == 0
    assert "systemctl try-restart litco-supervisor.service" not in calls(host)
    assert "systemctl restart litco-agent@aa11.service" in calls(host)


def test_no_restart_flips_and_installs_but_restarts_no_slot(host):
    add_slot(host, "aa11", 8800)
    out = update(host, "--ref", "host-2", "--source", str(make_source(host["tmp"], "host-2")), "--no-restart")
    assert out.returncode == 0, out.stderr
    assert os.readlink(host["prefix"] / "current") == "releases/host-2"
    assert not any("restart litco-agent@" in c for c in calls(host))


def test_update_refuses_a_per_matter_droplet(host):
    (host["prefix"] / "TOPOLOGY").write_text("per_matter\n")
    out = update(host, "--ref", "host-2", "--source", str(make_source(host["tmp"], "host-2")))
    assert out.returncode == 1 and "is not 'machine'" in out.stderr
    assert not (host["prefix"] / "current").exists()
    assert calls(host) == []


def test_dry_run_prints_the_plan_and_changes_nothing(host):
    out = update(host, "--ref", "host-3", "--dry-run")
    assert out.returncode == 0, out.stderr
    steps = re.findall(r"^PLAN \d\d  (.+)$", out.stdout, flags=re.M)
    assert steps == [
        f"prepare release host-3 in {host['prefix']}/releases/host-3",
        f"flip {host['prefix']}/current to releases/host-3",
        "install the release's units and tools; daemon-reload",
        "restart running slots one at a time, each behind its drain",
    ]
    assert "systemctl restart litco-agent@<slot>.service" in out.stdout
    assert calls(host) == []
    assert not (host["prefix"] / "releases" / "host-3").exists()


@pytest.mark.parametrize("args, message", [
    ([], "--ref <tag|commit> is required"),
    (["--ref", ".."], "may not start with a dot"),
    (["--ref", "v1", "--health-timeout", "soon"], "--health-timeout must be"),
    (["--ref", "v1", "--bogus"], "unknown argument"),
])
def test_bad_arguments(host, args, message):
    out = update(host, *args)
    assert out.returncode == 2 and message in out.stderr
    assert calls(host) == []


def test_a_release_older_than_the_slot_tools_still_updates(host):
    src = make_source(host["tmp"], "old")
    for tool in ("litco-slot-probe", "litco-slot-import"):
        (src / "deploy" / "host" / tool).unlink()
    out = update(host, "--ref", "old", "--source", str(src))
    assert out.returncode == 0, out.stderr
    assert not (host["sbin_local"] / "litco-slot-probe").exists()
    assert (host["sbin_local"] / "litco-supervisor").read_text() == "# litco-supervisor from old\n"
