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
            "HERMES_GATEWAY_BUSY_INPUT_MODE",
            "HERMES_TIMEZONE",
        ):
            v = os.environ.get(k)
            if v is not None:
                print(f"{{k}}={{v}}")
        """
    )
    env = dict(initial_env)
    env["HERMES_HOME"] = str(hermes_home)
    # Keep PATH / PYTHONPATH so venv imports resolve.
    for k in ("PATH", "PYTHONPATH", "VIRTUAL_ENV", "HOME"):
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
                  timezone: str | None = None) -> None:
    import yaml
    cfg: dict = {}
    if agent_cfg:
        cfg["agent"] = agent_cfg
    if display_cfg:
        cfg["display"] = display_cfg
    if timezone:
        cfg["timezone"] = timezone
    (home / "config.yaml").write_text(yaml.safe_dump(cfg))


def _write_env(home: Path, entries: dict[str, str]) -> None:
    lines = [f"{k}={v}\n" for k, v in entries.items()]
    (home / ".env").write_text("".join(lines))


@pytest.fixture
def hermes_home(tmp_path: Path) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    return home


def test_config_max_turns_wins_over_stale_env(hermes_home: Path) -> None:
    """Regression: config.yaml:agent.max_turns=500 must beat .env=60."""
    _write_config(hermes_home, agent_cfg={"max_turns": 500})
    _write_env(hermes_home, {"HERMES_MAX_ITERATIONS": "60"})

    env = _run_gateway_import(hermes_home, initial_env={})

    assert env.get("HERMES_MAX_ITERATIONS") == "500", (
        f"expected config.yaml max_turns=500 to win; got {env.get('HERMES_MAX_ITERATIONS')!r}. "
        "Stale .env value is shadowing config — the bridge lost its override."
    )


def test_config_gateway_timeout_wins_over_stale_env(hermes_home: Path) -> None:
    """Every agent.* bridge key must be config-authoritative, not .env-authoritative."""
    _write_config(hermes_home, agent_cfg={
        "gateway_timeout": 1800,
        "gateway_timeout_warning": 900,
    })
    _write_env(hermes_home, {
        "HERMES_AGENT_TIMEOUT": "60",
        "HERMES_AGENT_TIMEOUT_WARNING": "30",
    })

    env = _run_gateway_import(hermes_home, initial_env={})

    assert env.get("HERMES_AGENT_TIMEOUT") == "1800"
    assert env.get("HERMES_AGENT_TIMEOUT_WARNING") == "900"


def test_config_display_busy_input_mode_wins_over_stale_env(hermes_home: Path) -> None:
    _write_config(hermes_home, display_cfg={"busy_input_mode": "interrupt"})
    _write_env(hermes_home, {"HERMES_GATEWAY_BUSY_INPUT_MODE": "queue"})

    env = _run_gateway_import(hermes_home, initial_env={})

    assert env.get("HERMES_GATEWAY_BUSY_INPUT_MODE") == "interrupt"


def test_config_timezone_wins_over_stale_env(hermes_home: Path) -> None:
    _write_config(hermes_home, timezone="America/Los_Angeles")
    _write_env(hermes_home, {"HERMES_TIMEZONE": "UTC"})

    env = _run_gateway_import(hermes_home, initial_env={})

    assert env.get("HERMES_TIMEZONE") == "America/Los_Angeles"


def test_env_value_survives_when_config_omits_key(hermes_home: Path) -> None:
    """If config.yaml doesn't set max_turns, .env value must still pass through.

    The bridge only overwrites when the config key is present — an absent
    config key should NOT clobber the .env value.
    """
    _write_config(hermes_home, agent_cfg={})  # no max_turns
    _write_env(hermes_home, {"HERMES_MAX_ITERATIONS": "123"})

    env = _run_gateway_import(hermes_home, initial_env={})

    assert env.get("HERMES_MAX_ITERATIONS") == "123"
