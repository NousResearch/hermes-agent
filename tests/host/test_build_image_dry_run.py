"""build-image.sh: argument handling and the --dry-run plan. Never touches a cloud account."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from tests.host._load import HOST

SCRIPT = HOST / "build-image.sh"


@pytest.fixture
def fake_path(tmp_path) -> dict:
    """PATH whose doctl/ssh/scp only record that they were called."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "called"
    for tool in ("doctl", "ssh", "scp"):
        exe = bin_dir / tool
        exe.write_text(f'#!/bin/sh\necho "{tool} $*" >> "{marker}"\nexit 1\n')
        exe.chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}:/usr/bin:/bin")
    return {"env": env, "marker": marker}


def run(args, env):
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env=env)


def plan_steps(stdout: str) -> list[str]:
    return re.findall(r"^PLAN \d\d  (.+)$", stdout, flags=re.M)


def plan_commands(stdout: str) -> list[str]:
    return re.findall(r"^\s+\$ (.+)$", stdout, flags=re.M)


def test_dry_run_prints_the_full_plan_in_order_and_runs_nothing(fake_path):
    out = run(["--version", "v1.2.3", "--dry-run"], fake_path["env"])
    assert out.returncode == 0, out.stderr
    assert not fake_path["marker"].exists(), "dry run invoked doctl/ssh/scp"
    assert plan_steps(out.stdout) == [
        "refuse to overwrite an existing snapshot",
        "create builder droplet",
        "wait for SSH",
        "wait for the builder's own first boot (apt locks)",
        "copy install script",
        "install host (packages, Tailscale, Chromium, Python 3.14, hermes user, litco-agent@v1.2.3, venvs, Node)",
        "verify secret-free image",
        "remove builder key and clean cloud-init state (load-bearing)",
        "power off builder",
        "snapshot as litco-agent-host-v1.2.3",
        "delete builder",
    ]
    cmds = plan_commands(out.stdout)
    assert "doctl compute droplet create litco-agent-builder-v1-2-3 --region sfo3 --size s-4vcpu-8gb " \
           "--image ubuntu-24-04-x64 --ssh-keys <ssh-key>" in cmds[1]
    assert "--enable-monitoring" in cmds[1]
    assert cmds[5].endswith("bash /root/install-host.sh --ref v1.2.3 --repo https://github.com/yavarb/litco-agent.git")
    assert "cloud-init clean --logs --machine-id" in cmds[7]
    assert "rm -f /root/.ssh/authorized_keys" in cmds[7]
    assert cmds[9] == "doctl compute droplet-action snapshot <droplet-id> --snapshot-name litco-agent-host-v1.2.3 --wait"
    assert cmds[10] == "doctl compute droplet delete <droplet-id> --force"


def test_ref_region_size_and_key_overrides(fake_path):
    out = run(["--version", "v2", "--ref", "abc1234", "--region", "nyc3", "--size", "s-2vcpu-4gb",
               "--ssh-key", "aa:bb", "--dry-run"], fake_path["env"])
    assert out.returncode == 0
    cmds = "\n".join(plan_commands(out.stdout))
    assert "--region nyc3 --size s-2vcpu-4gb" in cmds and "--ssh-keys aa:bb" in cmds
    assert "--ref abc1234" in cmds and "litco-agent-host-v2 " in cmds + " "
    assert not fake_path["marker"].exists()


@pytest.mark.parametrize("args, message", [
    ([], "--version"),
    (["--version", "bad/name", "--dry-run"], "version must match"),
    (["--version", "v1"], "--ssh-key is required"),
    (["--version", "v1", "--bogus"], "unknown argument"),
])
def test_bad_arguments_fail_before_any_cloud_call(fake_path, args, message):
    out = run(args, fake_path["env"])
    assert out.returncode == 2
    assert message in out.stderr
    assert not fake_path["marker"].exists()


def test_script_is_strict_and_traps_builder_cleanup():
    text = SCRIPT.read_text()
    assert "set -euo pipefail" in text
    assert "trap cleanup_builder EXIT" in text  # behaviour is covered by the fake-cloud tests below
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_install_script_installs_what_the_image_needs():
    text = (HOST / "install-host.sh").read_text()
    assert "set -euo pipefail" in text
    for needle in ("ripgrep", "ffmpeg", "7zip", "unrar", "poppler-utils", "libreoffice-core",
                   "libreoffice-writer", "libreoffice-calc", "ufw", "fail2ban", "unattended-upgrades",
                   "tailscale.com/install.sh", "uv python install", "PYTHON_MINOR=3.14",
                   "loginctl enable-linger hermes", "uv sync --frozen", "duckdb pandas matplotlib python-docx "
                   "pymupdf requests openpyxl", '"agent-browser", "chromium"', "deb.nodesource.com"):
        assert needle in text, needle
    assert "/etc/litco-agent/env exists; image is not secret-free" in text
    assert ".env" not in text.replace("/etc/litco-agent/env", ""), "install script must not write env files"



def test_builder_is_tagged(fake_path):
    out = run(["--version", "v3", "--dry-run"], fake_path["env"])
    assert "--tag-names litco-host-builder" in plan_commands(out.stdout)[1]
    out = run(["--version", "v3", "--tag", "other-tag", "--dry-run"], fake_path["env"])
    assert "--tag-names other-tag" in plan_commands(out.stdout)[1]


FAKE_DOCTL = """#!/bin/sh
echo "doctl $*" >> "$LOG"
case "$*" in
  "compute snapshot list"*) exit 0 ;;
  "compute droplet create"*) {create} ;;
  "compute droplet list"*) printf '%s\\n' "123 litco-agent-builder-v3" "124 litco-agent-builder-v3" \\
                             "999 litkit-prod-1" "555 litco-agent-builder-v3-other" ;;
  *) exit 0 ;;
esac
"""


def _fake_cloud(tmp_path, create: str, ssh_ok: bool) -> dict:
    """doctl/ssh/scp fakes that log every call; the create branch is scripted."""
    bin_dir = tmp_path / "cloud"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    (bin_dir / "doctl").write_text(FAKE_DOCTL.replace("{create}", create))
    for tool in ("ssh", "scp"):
        (bin_dir / tool).write_text(f'#!/bin/sh\necho "{tool} $*" >> "$LOG"\nexit {0 if ssh_ok else 1}\n')
    for exe in bin_dir.iterdir():
        exe.chmod(0o755)
    env = dict(os.environ, PATH=f"{bin_dir}:/usr/bin:/bin", LOG=str(log))
    return {"env": env, "log": log}


def _deletes(log: Path) -> list[str]:
    return re.findall(r"^doctl compute droplet delete (\S+) --force$", log.read_text(), flags=re.M)


def test_failed_create_deletes_every_orphan_by_tag_and_exact_name(tmp_path):
    cloud = _fake_cloud(tmp_path, "exit 1", ssh_ok=True)
    out = run(["--version", "v3", "--ssh-key", "1"], cloud["env"])
    assert out.returncode == 1
    calls = cloud["log"].read_text()
    assert "doctl compute droplet list --tag-name litco-host-builder" in calls
    assert _deletes(cloud["log"]) == ["123", "124"]


def test_failure_after_create_deletes_the_known_builder_only(tmp_path):
    cloud = _fake_cloud(tmp_path, 'echo "777 203.0.113.9"', ssh_ok=False)
    env = dict(cloud["env"])
    out = subprocess.run(["bash", "-c", 'sleep() { :; }; export -f sleep; exec bash "$0" "$@"', str(SCRIPT),
                          "--version", "v3", "--ssh-key", "1"], capture_output=True, text=True, env=env)
    assert out.returncode == 1
    assert "droplet list" not in cloud["log"].read_text()
    assert _deletes(cloud["log"]) == ["777"]


def test_clean_run_disarms_the_trap_without_relisting(tmp_path):
    cloud = _fake_cloud(tmp_path, 'echo "777 203.0.113.9"', ssh_ok=True)
    out = run(["--version", "v3", "--ssh-key", "1"], cloud["env"])
    assert out.returncode == 0, out.stderr
    assert "DONE: litco-agent-host-v3" in out.stdout
    assert "droplet list" not in cloud["log"].read_text()
    assert _deletes(cloud["log"]) == ["777"]
