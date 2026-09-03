from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
STACK_SCRIPT = REPO_ROOT / "scripts" / "wisdom-demo-stack.sh"


def demo_env(tmp_path: Path) -> dict[str, str]:
    value = os.environ.copy()
    value.update({
        "HOME": str(tmp_path),
        "HERMES_HOME": str(tmp_path / "profile"),
        "HERMES_WISDOM_PYTHON": sys.executable,
        "HERMES_WISDOM_QUIET": "1",
        "WISDOM_AGENT_HOME": str(tmp_path / "profile"),
        "WISDOM_DEMO_HOME": str(tmp_path / "demo"),
        "WISDOM_DEMO_STATE_DIR": str(tmp_path / "state"),
        "WISDOM_TEST_CALLS": str(tmp_path / "gateway-calls"),
        "WISDOM_TEST_COUNT": str(tmp_path / "gateway-count"),
    })
    return value


def test_messaging_gateway_is_relaunched_after_supervised_restart(tmp_path: Path):
    command = r"""
source "$1"
mkdir -p "$STATE_DIR"

hermes() {
  local count=0
  if [[ -f "$WISDOM_TEST_COUNT" ]]; then
    count="$(cat "$WISDOM_TEST_COUNT")"
  fi
  count=$((count + 1))
  printf '%s\n' "$count" >"$WISDOM_TEST_COUNT"
  printf '%s\n' "$*" >>"$WISDOM_TEST_CALLS"
  if [[ "$count" -eq 1 ]]; then
    return 75
  fi
  return 78
}

start_messaging_gateway_process
gateway_supervisor_pid="${CHILD_PIDS[0]}"
set +e
wait "$gateway_supervisor_pid"
gateway_supervisor_exit=$?
set -e
printf 'supervisor_exit=%s\n' "$gateway_supervisor_exit"
"""
    result = subprocess.run(
        ["bash", "-c", command, "wisdom-demo-stack-test", str(STACK_SCRIPT)],
        cwd=REPO_ROOT,
        env=demo_env(tmp_path),
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )

    calls = (tmp_path / "gateway-calls").read_text().splitlines()
    log = (tmp_path / "state" / "messaging-gateway.log").read_text()

    assert calls == [
        "gateway run --replace --external-supervisor -v",
        "gateway run --replace --external-supervisor -v",
    ]
    assert "supervisor_exit=78" in result.stdout
    assert "requested restart; relaunching" in log
    assert "fatal configuration error; not restarting" in log


def test_demo_stack_status_includes_messaging_gateway(tmp_path: Path):
    result = subprocess.run(
        [str(STACK_SCRIPT), "status"],
        cwd=REPO_ROOT,
        env=demo_env(tmp_path),
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert "Messaging gateway" in result.stdout
    assert "down" in result.stdout


@pytest.mark.parametrize(
    ("has_shared_credentials", "expects_bootstrap"),
    [(False, True), (True, False)],
)
def test_authentication_bootstraps_hermes_only_without_shared_credentials(
    tmp_path: Path,
    has_shared_credentials: bool,
    expects_bootstrap: bool,
):
    portal_app = tmp_path / "portal"
    env_source = portal_app / "e2e" / "browser" / "env.source"
    env_source.parent.mkdir(parents=True)
    env_source.write_text("")

    demo_home = tmp_path / "demo"
    shared_auth = demo_home / "shared" / "nous_auth.json"
    shared_auth.parent.mkdir(parents=True)
    if has_shared_credentials:
        shared_auth.write_text('{"access_token":"existing"}')

    value = demo_env(tmp_path)
    value.update({
        "WISDOM_PORTAL_APP": str(portal_app),
        "WISDOM_TEST_PNPM_CALL": str(tmp_path / "pnpm-call"),
        "WISDOM_DEMO_HOME": str(demo_home),
    })
    command = r"""
source "$1"
mkdir -p "$STATE_DIR"

pnpm() {
  printf '%s\n' "$*" >"$WISDOM_TEST_PNPM_CALL"
  printf '{"cookies":[]}\n'
}
hermes() {
  if [[ "$1 $2" == "wisdom setup" ]]; then
    printf '{}\n'
  fi
  return 0
}
open_portal_login() { :; }

authenticate_demo
"""
    subprocess.run(
        ["bash", "-c", command, "wisdom-demo-auth-test", str(STACK_SCRIPT)],
        cwd=REPO_ROOT,
        env=value,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )

    pnpm_call = (tmp_path / "pnpm-call").read_text()
    assert ("--hermes-home" in pnpm_call) is expects_bootstrap
