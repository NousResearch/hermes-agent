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
    assert "trap cleanup_builder EXIT" in text
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


def test_builder_is_tagged_and_cleanup_survives_signals_and_lost_ids(fake_path):
    out = run(["--version", "v3", "--dry-run"], fake_path["env"])
    cmds = plan_commands(out.stdout)
    assert "--tag-names litco-host-builder" in cmds[1]
    out = run(["--version", "v3", "--tag", "other-tag", "--dry-run"], fake_path["env"])
    assert "--tag-names other-tag" in plan_commands(out.stdout)[1]
    text = SCRIPT.read_text()
    assert "trap 'exit 130' INT" in text and "trap 'exit 143' TERM" in text
    assert 'droplet list --tag-name "$TAG"' in text
