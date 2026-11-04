"""Root acting on the hermes-owned gateway tree must not follow hermes-planted symlinks.

``/run/hermes-service`` is owned by hermes, so a hermes process can rename anything root
creates there and put a symlink in its place. These tests play that hermes process for real,
inside the image, against root registration (``HERMES_DOCKER_EXEC_AS_ROOT=1`` or a direct
Python call) and root lifecycle control.
"""
from __future__ import annotations

from tests.docker.conftest import docker_exec, docker_exec_sh, poll_container, start_container

SCANDIR = "/run/hermes-service"
STAGING = "/run/.hermes-service.staging"

# Runs as root. After the slot's supervise skeleton is seeded, the hermes account tries to
# swap every half-built slot it can reach (in the scandir and in root's staging dir).
_SWAP_DURING_REGISTRATION = f"""
import os, subprocess
from hermes_cli import service_manager as sm
assert os.geteuid() == 0
os.umask(0o077)
os.mkdir('/opt/victim', 0o700)
seed = sm._seed_supervise_skeleton
attack = (
    'for d in {SCANDIR}/.gateway-* {STAGING}/gateway-*; do '
    '[ -e "$d" ] || continue; '
    'mv "$d" "$d.moved" && ln -s /opt/victim "$d" && echo "swapped $d"; '
    'done; true'
)
def seed_then_swap(svc_dir):
    seed(svc_dir)
    out = subprocess.run(['/command/s6-setuidgid', 'hermes', 'sh', '-c', attack],
                         capture_output=True, text=True)
    print('attacker:', repr(out.stdout), repr(out.stderr))
sm._seed_supervise_skeleton = seed_then_swap
sm.S6ServiceManager().register_profile_gateway('swap', start_now=False)
"""


def test_root_registration_survives_staging_swap(built_image: str, container_name: str) -> None:
    start_container(built_image, container_name)
    registered = docker_exec(container_name, "python3", "-c", _SWAP_DURING_REGISTRATION,
                             user="root", extra_docker_args=("-e", "HERMES_DOCKER_EXEC_AS_ROOT=1"))
    assert registered.returncode == 0, registered.stderr
    # hermes could not reach root's half-built slot anywhere.
    assert "swapped" not in registered.stdout, registered.stdout
    victim = docker_exec_sh(container_name, "stat -c '%u:%g %a' /opt/victim", user="root")
    assert victim.stdout.strip() == "0:0 700", victim.stdout
    staging = docker_exec_sh(container_name, f"stat -c '%U:%G %a' {STAGING}; ls -A {STAGING}",
                             user="root")
    assert staging.stdout.splitlines() == ["root:root 700"], staging.stdout
    slot = f"{SCANDIR}/gateway-swap"
    published = docker_exec_sh(container_name, f"test -d {slot} && test ! -L {slot} && "
                               f"stat -c '%U:%G' {slot} {slot}/run {slot}/supervise")
    assert published.returncode == 0, published.stderr
    assert published.stdout.splitlines() == ["hermes:hermes"] * 3
    ok, state = poll_container(container_name, f"/command/s6-svstat {slot}", deadline_s=30)
    assert ok, state


def test_root_lifecycle_control_does_not_follow_control_symlink(
    built_image: str, container_name: str,
) -> None:
    """s6-svc as root would write its command byte through a hermes-planted control symlink."""
    start_container(built_image, container_name)
    registered = docker_exec(container_name, "python3", "-c", (
        "from hermes_cli.service_manager import S6ServiceManager; "
        "S6ServiceManager().register_profile_gateway('ctl', start_now=False)"
    ), user="root", extra_docker_args=("-e", "HERMES_DOCKER_EXEC_AS_ROOT=1"))
    assert registered.returncode == 0, registered.stderr
    slot = f"{SCANDIR}/gateway-ctl"
    ok, state = poll_container(container_name, f"/command/s6-svstat {slot}", deadline_s=30)
    assert ok, state
    seeded = docker_exec_sh(container_name, (
        "printf ORIGINAL > /opt/victim-file && chmod 0600 /opt/victim-file"
    ), user="root")
    assert seeded.returncode == 0, seeded.stderr
    planted = docker_exec_sh(container_name, (
        f"rm {slot}/supervise/control && ln -s /opt/victim-file {slot}/supervise/control"
    ))
    assert planted.returncode == 0, planted.stderr
    control = docker_exec(container_name, "python3", "-c", (
        "from hermes_cli.service_manager import S6ServiceManager, S6CommandError\n"
        "try:\n"
        "    S6ServiceManager().restart('gateway-ctl')\n"
        "except S6CommandError as exc:\n"
        "    print('refused', exc)\n"
    ), user="root", extra_docker_args=("-e", "HERMES_DOCKER_EXEC_AS_ROOT=1"))
    assert control.returncode == 0, control.stderr
    victim = docker_exec_sh(container_name, "cat /opt/victim-file; echo; stat -c '%u %a' "
                            "/opt/victim-file", user="root")
    assert victim.stdout.splitlines() == ["ORIGINAL", "0 600"], victim.stdout
