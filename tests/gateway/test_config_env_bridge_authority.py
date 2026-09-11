"""Regression tests for the config.yaml → env var bridge in gateway/run.py.

Guards against the 60-vs-500 bug where a stale `.env HERMES_MAX_ITERATIONS=60`
entry silently shadowed `agent.max_turns: 500` in config.yaml because the
bridge used `if X not in os.environ` guards. After PR#18413 the bridge
treats config.yaml as authoritative and unconditionally overwrites .env
values for `agent.*`, `display.*`, `timezone`, and `security.*` keys.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run_gateway_import(hermes_home: Path, initial_env: dict[str, str]) -> dict[str, str]:
    """Import gateway.run in a clean subprocess and return the post-import env.

    The bridge runs at module-import time, so simply importing is enough
    to exercise it. Running in a subprocess isolates the test from other
    import side effects and makes the "what ends up in os.environ" check
    deterministic.
    """
    script = textwrap.dedent(
        f"""
        import os, sys
        sys.path.insert(0, {str(PROJECT_ROOT)!r})

        try:
            from gateway import run  # noqa: F401  — module import triggers bridge
        except Exception as exc:
            print(f"IMPORT_ERROR:{{type(exc).__name__}}:{{exc}}", file=sys.stderr)
            sys.exit(2)

        for k in (
            "HERMES_MAX_ITERATIONS",
            "HERMES_AGENT_TIMEOUT",
            "HERMES_AGENT_TIMEOUT_WARNING",
            "HERMES_TURN_LEASE_TIMEOUT",
            "HERMES_SESSION_STALL_TIMEOUT",
            "HERMES_GATEWAY_BUSY_INPUT_MODE",
            "HERMES_GATEWAY_BUSY_TEXT_MODE",
            "HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT",
            "HERMES_TIMEZONE",
        ):
            v = os.environ.get(k)
            if v is not None:
                print(f"{{k}}={{v}}")
        """
    )
    env = dict(initial_env)
    env["HERMES_HOME"] = str(hermes_home)
    # Keep interpreter paths plus the Windows bootstrap variables required by
    # stdlib platform detection and native dependency loading.  The child is
    # otherwise intentionally clean so stale Hermes settings cannot leak in.
    for k in (
        "PATH",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "HOME",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "LOCALAPPDATA",
        "APPDATA",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
    ):
        if k in os.environ and k not in env:
            env[k] = os.environ[k]

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        pytest.fail(
            f"gateway.run import failed (rc={result.returncode})\n"
            f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
        )
    out: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


def _write_config(home: Path, agent_cfg: dict | None = None, display_cfg: dict | None = None,
                  timezone: str | None = None, gateway_cfg: dict | None = None) -> None:
    import yaml
    cfg: dict = {}
    if agent_cfg:
        cfg["agent"] = agent_cfg
    if display_cfg:
        cfg["display"] = display_cfg
    if gateway_cfg:
        cfg["gateway"] = gateway_cfg
    if timezone:
        cfg["timezone"] = timezone
    (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")


def _write_env(home: Path, entries: dict[str, str]) -> None:
    lines = [f"{k}={v}\n" for k, v in entries.items()]
    (home / ".env").write_text("".join(lines), encoding="utf-8")


@pytest.fixture
def hermes_home(tmp_path: Path) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    return home


def test_config_gateway_timeout_wins_over_stale_env(hermes_home: Path) -> None:
    """Every agent.* bridge key must be config-authoritative, not .env-authoritative."""
    _write_config(hermes_home, agent_cfg={
        "gateway_timeout": 1800,
        "gateway_timeout_warning": 900,
        "session_stall_timeout": 300,
    })
    _write_env(hermes_home, {
        "HERMES_AGENT_TIMEOUT": "60",
        "HERMES_AGENT_TIMEOUT_WARNING": "30",
        "HERMES_SESSION_STALL_TIMEOUT": "15",
    })

    env = _run_gateway_import(hermes_home, initial_env={})

    assert env.get("HERMES_AGENT_TIMEOUT") == "1800"
    assert env.get("HERMES_AGENT_TIMEOUT_WARNING") == "900"
    assert env.get("HERMES_SESSION_STALL_TIMEOUT") == "300"


def test_config_turn_lease_timeout_wins_over_stale_env(hermes_home: Path) -> None:
    """The user-facing lease wait budget belongs to config.yaml."""
    _write_config(
        hermes_home,
        agent_cfg={"gateway_turn_lease_timeout": 600},
    )
    _write_env(
        hermes_home,
        {"HERMES_TURN_LEASE_TIMEOUT": "60"},
    )

    env = _run_gateway_import(hermes_home, initial_env={})

    assert env.get("HERMES_TURN_LEASE_TIMEOUT") == "600"


def test_default_turn_lease_timeout_overrides_stale_env_when_key_is_omitted(
    hermes_home: Path,
) -> None:
    """The internal env mirror must never become a second config source."""
    _write_env(
        hermes_home,
        {"HERMES_TURN_LEASE_TIMEOUT": "60"},
    )

    env = _run_gateway_import(hermes_home, initial_env={})

    assert env.get("HERMES_TURN_LEASE_TIMEOUT") == "5"


def test_default_turn_lease_timeout_matches_the_runtime_fallback() -> None:
    """The advertised config default must match the fail-closed runtime fallback.

    Invariant, not a snapshot: whatever the default becomes, config and the
    lease registry's DEFAULT_LEASE_WAIT must move together.
    """
    from gateway.turn_lease import DEFAULT_LEASE_WAIT
    from hermes_cli.config import DEFAULT_CONFIG

    assert (
        float(DEFAULT_CONFIG["agent"]["gateway_turn_lease_timeout"])
        == DEFAULT_LEASE_WAIT
    )


def test_config_platform_connect_timeout_supplies_env_when_unset(hermes_home: Path) -> None:
    """config.yaml:gateway.platform_connect_timeout supplies the env var when
    it isn't already set (#19776 — config surface for the Discord connect
    timeout, replacing the undocumented env-var-only workaround)."""
    _write_config(hermes_home, gateway_cfg={"platform_connect_timeout": 90})

    env = _run_gateway_import(hermes_home, initial_env={})

    assert env.get("HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT") == "90"


def test_env_platform_connect_timeout_wins_over_config(hermes_home: Path) -> None:
    """Unlike the agent.*/display.*/timezone bridges (config-authoritative),
    HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT is the manual-override escape hatch:
    an explicitly-set env var WINS over config.yaml. This divergence is
    intentional (#19776) — the env var is the operator's emergency knob."""
    _write_config(hermes_home, gateway_cfg={"platform_connect_timeout": 90})

    env = _run_gateway_import(
        hermes_home,
        initial_env={"HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT": "120"},
    )

    assert env.get("HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT") == "120"
