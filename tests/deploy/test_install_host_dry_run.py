"""install-host.sh --dry-run: the per-matter image stays as it was; the machine image gets the slot units."""

from __future__ import annotations

import os
import re
import subprocess

import pytest

from tests.host._load import HOST

SCRIPT = HOST / "install-host.sh"
TOOLS = ("apt-get", "curl", "systemctl", "useradd", "loginctl", "git", "uv", "install", "chown", "nft", "sshd",
         "add-apt-repository", "dpkg-reconfigure", "sudo", "cp", "rm", "ln", "mv", "python3", "id")


@pytest.fixture
def fake_path(tmp_path) -> dict:
    """PATH whose system tools only record that they were called."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "called"
    for tool in TOOLS:
        exe = bin_dir / tool
        exe.write_text(f'#!/bin/sh\necho "{tool} $*" >> "{marker}"\nexit 1\n')
        exe.chmod(0o755)
    # sed, grep, cat and printf are what the dry run itself uses.
    env = {k: v for k, v in os.environ.items() if k != "LITCO_TOPOLOGY"}
    env["PATH"] = f"{bin_dir}:/usr/bin:/bin"
    return {"env": env, "marker": marker}


def dry_run(fake_path, *args, **env):
    out = subprocess.run(["bash", str(SCRIPT), *args, "--dry-run"], capture_output=True, text=True,
                         env=dict(fake_path["env"], **env))
    assert out.returncode == 0, out.stderr
    # Reading the source tree's commit (to name the release) is the one call a dry run makes.
    calls = fake_path["marker"].read_text().splitlines() if fake_path["marker"].exists() else []
    assert all(re.fullmatch(r"git -C \S+ rev-parse --short=12 HEAD", c) for c in calls), calls
    return out.stdout


def commands(stdout: str) -> list[str]:
    return re.findall(r"^ {9}\$ (.+)$", stdout, flags=re.M)


def written(stdout: str) -> dict:
    """{path: content} for every file the plan writes."""
    files, current = {}, None
    for line in stdout.splitlines():
        if m := re.match(r"^ {9}> (\S+) \(\d+\)$", line):
            current = files.setdefault(m.group(1), [])
        elif line.startswith("         | ") and current is not None:
            current.append(line[11:])
        else:
            current = None
    return {path: "\n".join(lines) for path, lines in files.items()}


@pytest.mark.parametrize("script", ["install-host.sh", "litco-host-update", "litco-agent-drain"])
def test_shell_scripts_parse(script):
    assert subprocess.run(["bash", "-n", str(HOST / script)]).returncode == 0


def test_per_matter_is_the_default_and_bakes_the_single_matter_unit(fake_path):
    out = dry_run(fake_path, "--ref", "v1")
    cmds = commands(out)
    assert "install -m 0644 /opt/litco-agent/app/deploy/host/litco-agent.service " \
           "/etc/systemd/system/litco-agent.service" in cmds
    assert "systemctl enable litco-agent.service" in cmds
    assert "loginctl enable-linger hermes" in cmds
    assert "git -C /opt/litco-agent/app checkout --detach v1" in cmds
    assert written(out)["/opt/litco-agent/TOPOLOGY"] == "per_matter"
    joined = "\n".join(cmds)
    for absent in ("litco-agent@.service", "litco-supervisor", "litco-slot-", "nftables", "/opt/litco-agent/current",
                   "releases/"):
        assert absent not in joined, absent
    assert "/etc/nftables.conf" not in written(out)


def machine_plan(fake_path, *args):
    out = dry_run(fake_path, "--ref", "host-2026.10.01", "--topology", "machine", *args)
    return out, commands(out), written(out)


def test_machine_installs_the_template_unit_supervisor_and_update_from_the_release(fake_path):
    out, cmds, files = machine_plan(fake_path)
    rel = "/opt/litco-agent/releases/host-2026.10.01"
    assert f"git -C {rel} checkout --detach host-2026.10.01" in cmds
    assert f"env -C {rel} UV_PROJECT_ENVIRONMENT={rel}/.venv uv sync --frozen --no-dev --extra messaging " \
           "--python <uv python find 3.14>" in cmds
    assert f"install -m 0644 {rel}/deploy/host/litco-agent@.service /etc/systemd/system/litco-agent@.service" in cmds
    assert f"install -m 0755 {rel}/deploy/host/litco-supervisor /usr/local/sbin/litco-supervisor" in cmds
    assert f"install -m 0644 {rel}/deploy/host/litco-supervisor.service " \
           "/etc/systemd/system/litco-supervisor.service" in cmds
    assert f"install -m 0755 {rel}/deploy/host/litco-host-update /usr/local/sbin/litco-host-update" in cmds
    for tool in ("litco-slot-probe", "litco-slot-import"):
        assert f"install -m 0755 {rel}/deploy/host/{tool} /usr/local/sbin/{tool}" in cmds
    assert "systemctl enable litco-supervisor.service" in cmds
    assert files["/opt/litco-agent/TOPOLOGY"] == "machine"
    assert files[f"{rel}/.litco-release-ok"] == "host-2026.10.01"
    assert "d /run/litco-agent 0700 root root -" in files["/etc/tmpfiles.d/litco-agent.conf"]
    assert "install -d -m 0711 -o root -g root /srv/litco /srv/litco/m" in cmds


def test_machine_does_not_bake_the_single_matter_unit_or_hermes_linger(fake_path):
    _, cmds, _ = machine_plan(fake_path)
    joined = "\n".join(cmds)
    assert "systemctl enable litco-agent.service" not in joined
    assert "/etc/systemd/system/litco-agent.service" not in joined.replace(
        "rm -f /etc/systemd/system/litco-agent.service", "")
    assert "loginctl enable-linger" not in joined


def test_machine_flips_current_only_after_the_release_is_built_and_marked(fake_path):
    _, cmds, _ = machine_plan(fake_path)
    rel = "/opt/litco-agent/releases/host-2026.10.01"
    sync = next(i for i, c in enumerate(cmds) if "uv sync" in c)
    chown = cmds.index(f"chown -R root:root {rel}")
    link = cmds.index("ln -sfn releases/host-2026.10.01 /opt/litco-agent/current.new")
    flip = next(i for i, c in enumerate(cmds) if "os.replace" in c)
    assert sync < chown < link < flip
    assert cmds[flip].endswith("/opt/litco-agent/current.new /opt/litco-agent/current")


def test_machine_firewall_admits_inbound_only_on_tailscale0_and_loads_at_next_boot(fake_path):
    _, cmds, files = machine_plan(fake_path)
    rules = files["/etc/nftables.conf"]
    assert "policy drop;" in rules
    assert re.findall(r"iifname \"(\S+)\" accept", rules) == ["tailscale0"]
    assert not re.search(r"tcp dport", rules), "no TCP port is open on the public interface"
    assert "nft -c -f /etc/nftables.conf" in cmds
    assert "systemctl enable nftables.service" in cmds
    # Loading now would cut the builder's public SSH session.
    assert not any(c in cmds for c in ("systemctl start nftables.service", "nft -f /etc/nftables.conf",
                                       "systemctl restart nftables.service"))


def test_machine_image_checks_every_secret_location(fake_path):
    _, cmds, _ = machine_plan(fake_path)
    checks = [c for c in cmds if "not secret-free" in c]
    assert len(checks) == 3
    for path in ("/etc/litco-agent/env", "/etc/litco-supervisor.env", "/run/litco-agent/*.env"):
        assert any(path in c for c in checks), path


def test_topology_comes_from_the_environment_when_no_flag_is_given(fake_path):
    out = dry_run(fake_path, "--ref", "v1", LITCO_TOPOLOGY="machine")
    assert written(out)["/opt/litco-agent/TOPOLOGY"] == "machine"
    out = dry_run(fake_path, "--ref", "v1", "--topology", "per_matter", LITCO_TOPOLOGY="machine")
    assert written(out)["/opt/litco-agent/TOPOLOGY"] == "per_matter"


def test_container_machine_skips_firewall_tailscale_and_enabling(fake_path):
    out = dry_run(fake_path, "--source", "/src", "--container", "--topology", "machine")
    joined = "\n".join(commands(out))
    assert "/opt/litco-agent/releases/local" in joined
    for absent in ("nft ", "tailscale", "systemctl enable", "loginctl"):
        assert absent not in joined, absent
    assert "/etc/nftables.conf" not in written(out)


def test_release_names_are_path_safe(fake_path):
    out = dry_run(fake_path, "--ref", "origin/feature x", "--topology", "machine")
    assert "/opt/litco-agent/releases/origin-feature-x" in out
    refused = subprocess.run(["bash", str(SCRIPT), "--ref", "..", "--topology", "machine", "--dry-run"],
                             capture_output=True, text=True, env=fake_path["env"])
    assert refused.returncode == 2 and "may not start with a dot" in refused.stderr


@pytest.mark.parametrize("args, message, code", [
    (["--ref", "v1", "--topology", "shared", "--dry-run"], "--topology must be", 2),
    (["--dry-run"], "--ref <git-ref> is required", 2),
    (["--ref", "v1", "--bogus"], "unknown argument", 2),
])
def test_bad_arguments(fake_path, args, message, code):
    out = subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env=fake_path["env"])
    assert out.returncode == code and message in out.stderr
    assert not fake_path["marker"].exists()


@pytest.mark.parametrize("topology", ["per_matter", "machine"])
def test_no_plan_writes_an_env_file(fake_path, topology):
    out = dry_run(fake_path, "--ref", "v1", "--topology", topology)
    env_file = r"(\S*/)?(env|\S*\.env)"
    assert not [p for p in written(out) if re.fullmatch(env_file, p)]
    assert not [c for c in commands(out) if re.search(rf">\s*{env_file}(\s|$)", c)]
