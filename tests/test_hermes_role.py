"""Regression guard for docker/hermes-role.sh — the container role gate.

Background: the gateway and dashboard containers run the SAME image, the
SAME /init + cont-init.d hooks, and SHARE the ~/.hermes volume at /opt/data.
Only the gateway must seed/run the per-profile gateways and the :11435 Claude
proxy; the proxy's s6-log takes an exclusive lock on the shared
/opt/data/logs/claude-proxy/. When the dashboard container also started the
proxy, the two s6-log processes deadlocked on that lock and the gateway's
logger flapped forever on "Resource busy" (exit 111), taking the bots down.

hermes-role.sh resolves the role (consulted by 02-reconcile-profiles and
03-seed-data-services) so the dashboard container brings up neither. This
test locks that resolution: explicit $HERMES_ROLE wins, otherwise the role is
inferred from the container CMD carried in the process argv, defaulting to
`gateway` so a detection miss never silently runs nothing.

Pure POSIX-sh + /proc; no docker required, so it runs in the normal suite.
"""
from __future__ import annotations

import shlex
import shutil
import subprocess
import time
from pathlib import Path

import pytest

ROLE_SCRIPT = Path(__file__).resolve().parent.parent / "docker" / "hermes-role.sh"

# Minimal clean env so a stray HERMES_ROLE in the test runner's environment
# can't leak into the cases that exercise CMD inference.
_BASE_ENV = {"PATH": "/usr/bin:/bin"}


def _resolve_role(extra_env: dict[str, str] | None = None) -> str:
    """Source hermes-role.sh in a fresh POSIX shell and print hermes_role."""
    env = dict(_BASE_ENV)
    if extra_env:
        env.update(extra_env)
    r = subprocess.run(
        ["sh", "-c", f". {shlex.quote(str(ROLE_SCRIPT))}; hermes_role"],
        capture_output=True, text=True, timeout=15, env=env,
    )
    assert r.returncode == 0, f"hermes_role failed: {r.stderr!r}"
    return r.stdout.strip()


def test_script_present_and_sourceable() -> None:
    """The helper must exist and source cleanly (defines the function only)."""
    assert ROLE_SCRIPT.is_file(), f"missing {ROLE_SCRIPT}"
    r = subprocess.run(
        ["sh", "-c", f". {shlex.quote(str(ROLE_SCRIPT))}; type hermes_role"],
        capture_output=True, text=True, timeout=15, env=dict(_BASE_ENV),
    )
    assert r.returncode == 0, f"sourcing defined no hermes_role: {r.stderr!r}"


def test_explicit_gateway_env_wins() -> None:
    assert _resolve_role({"HERMES_ROLE": "gateway"}) == "gateway"


def test_explicit_dashboard_env_wins() -> None:
    assert _resolve_role({"HERMES_ROLE": "dashboard"}) == "dashboard"


def test_unknown_env_falls_through_to_default() -> None:
    """A junk HERMES_ROLE is ignored; with no CMD match we default gateway."""
    assert _resolve_role({"HERMES_ROLE": "bogus"}) == "gateway"


def test_default_is_gateway_without_any_signal() -> None:
    """No env, no `dashboard` CMD in any process -> fail safe to gateway."""
    assert _resolve_role() == "gateway"


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash exec -a")
def test_cmd_inference_detects_dashboard_from_argv() -> None:
    """With HERMES_ROLE unset, a process whose argv is
    `.../main-wrapper.sh dashboard ...` (how s6's rc.init carries the CMD)
    makes hermes_role infer the dashboard role.
    """
    argv0 = "/opt/hermes/docker/main-wrapper.sh dashboard --host 127.0.0.1 --no-open"
    # exec -a sets argv[0] to the full string; /proc/<pid>/cmdline then reads
    # back exactly that, mirroring rc.init's `... main-wrapper.sh dashboard ...`.
    proc = subprocess.Popen(
        ["bash", "-c", f"exec -a {shlex.quote(argv0)} sleep 30"],
    )
    try:
        # Give the exec a moment so the process is visible in /proc.
        deadline = time.monotonic() + 5.0
        role = ""
        while time.monotonic() < deadline:
            role = _resolve_role()  # HERMES_ROLE intentionally unset
            if role == "dashboard":
                break
            time.sleep(0.1)
        assert role == "dashboard", (
            f"CMD inference should detect dashboard from argv; got {role!r}"
        )
    finally:
        proc.terminate()
        proc.wait(timeout=10)
