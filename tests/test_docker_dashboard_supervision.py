"""Dashboard supervision contracts for the s6 finish script and compose overrides.

Regression for #49567 / #56401: a providerless non-loopback dashboard bind is
a permanent configuration refusal — the supervisor must park the slot (not
restart-loop), the default bind must be safe loopback, and the macOS Docker
Desktop compose override must publish the dashboard on host loopback without
an unauthenticated non-loopback bind.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

import hermes_yaml as yaml

REPO = Path(__file__).resolve().parents[1]
FINISH = REPO / "docker" / "s6-rc.d" / "dashboard" / "finish"
RUN = REPO / "docker" / "s6-rc.d" / "dashboard" / "run"


def _run_sh(script: Path, *args: str, env: dict) -> int:
    sh = shutil.which("sh")
    if sh is None:
        pytest.skip("requires a POSIX sh to exercise the shipped s6 script")
    assert sh is not None  # for type checkers after the skip above
    result = subprocess.run(
        [sh, str(script), *args],
        env={**os.environ, **env},
        capture_output=True, text=True, timeout=10,
    )
    return result.returncode


def test_finish_parks_the_auth_gate_refusal_instead_of_restarting():
    """EX_CONFIG (78) from the auth gate must map to s6's permanent-failure
    marker (125) so s6-supervise stops the restart loop (#49567)."""
    assert _run_sh(FINISH, "78", "0", env={"HERMES_DASHBOARD": "1"}) == 125


def test_finish_still_restarts_real_crashes_when_enabled():
    """Non-78 exits of an enabled dashboard stay restartable (exit 0) and the
    disabled-slot path keeps its permanent-failure marker."""
    assert _run_sh(FINISH, "1", "0", env={"HERMES_DASHBOARD": "1"}) == 0
    assert _run_sh(FINISH, "256", "15", env={"HERMES_DASHBOARD": "yes"}) == 0
    assert _run_sh(FINISH, "0", "0", env={"HERMES_DASHBOARD": ""}) == 125


def test_dashboard_service_defaults_to_loopback_bind():
    """The s6 dashboard run script's HERMES_DASHBOARD_HOST default must be
    127.0.0.1 — matching the `hermes dashboard` CLI — so a providerless fresh
    container never hits the auth-gate refusal restart loop (#49567)."""
    # Run the script's expansion the way s6 would with the env unset:
    # dash_host=${HERMES_DASHBOARD_HOST:-<default>}.
    text = RUN.read_text(encoding="utf-8")
    marker = 'dash_host="${HERMES_DASHBOARD_HOST:-'
    start = text.index(marker) + len(marker)
    default = text[start:text.index("}", start)]
    assert default == "127.0.0.1", (
        "s6 dashboard default bind must stay loopback; got "
        f"{default!r} (a providerless non-loopback default re-arms the "
        "#49567 restart loop)"
    )


def _load_compose(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def test_base_compose_keeps_loopback_dashboard_command():
    """The base compose dashboard service binds 127.0.0.1 (localhost-only
    posture documented in the file itself)."""
    services = _load_compose(REPO / "docker-compose.yml")["services"]
    command = services["dashboard"]["command"]
    assert "127.0.0.1" in command, command
    assert "--insecure" not in command


def test_macos_override_publishes_loopback_port_without_public_bind():
    """The macOS Docker Desktop override (#56401) must:

    1. drop `network_mode: host` (which Docker Desktop maps into its Linux
       VM, ignoring `ports:` entirely), and
    2. publish the dashboard on host LOOPBACK only, without introducing the
       `0.0.0.0` + `--insecure` unauthenticated pattern (the gap flagged on
       the superseded contributor PRs).
    """
    override = _load_compose(REPO / "docker-compose.macos.yml")["services"]

    # Compose merge semantics: `network_mode` and `command` replace, `ports`
    # and `environment` append. Assert the override's own declarations.
    assert override["gateway"]["network_mode"] == "bridge", override["gateway"]
    assert override["dashboard"]["network_mode"] == "service:gateway", (
        "the dashboard must share the gateway's netns so gateway-liveness "
        "probing keeps working under bridge networking"
    )

    published = override["gateway"]["ports"]
    assert "127.0.0.1:9119:9119" in published, published
    for entry in published:
        host_ip = str(entry).split(":", 1)[0]
        assert host_ip == "127.0.0.1", f"non-loopback publish {entry!r}"

    # No unauthenticated non-loopback override of the dashboard command.
    dash_env = {str(e).split("=", 1)[0]: str(e).split("=", 1)[-1]
                for e in override["dashboard"].get("environment", [])}
    assert dash_env.get("HERMES_DASHBOARD_HOST", "127.0.0.1") == "127.0.0.1", dash_env
    command = override["dashboard"].get("command")
    if command is not None:
        assert "--insecure" not in command and "0.0.0.0" not in command, command


def test_macos_override_merges_cleanly_with_base():
    """`docker compose -f docker-compose.yml -f docker-compose.macos.yml`
    must be valid: the merged services keep one publishable 9119 and the
    dashboard ends up sharing the gateway's network namespace."""
    base = _load_compose(REPO / "docker-compose.yml")["services"]
    override = _load_compose(REPO / "docker-compose.macos.yml")["services"]

    merged_dash = {**base["dashboard"], **override["dashboard"]}
    assert merged_dash["network_mode"] == "service:gateway"
    # The base command survives (loopback bind); no `--insecure` appears.
    assert "127.0.0.1" in merged_dash["command"]
    assert "--insecure" not in merged_dash["command"]

    merged_gw = {**base["gateway"], **override["gateway"]}
    assert merged_gw["network_mode"] == "bridge"
