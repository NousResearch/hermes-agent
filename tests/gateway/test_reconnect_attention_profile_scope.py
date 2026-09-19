"""Profile-scope leak regression for reconnect attention threshold.

Issue: _RECONNECT_ATTENTION_AFTER_SECONDS was a module-level constant that
captured os.environ at import time. Under multiplex, switching profiles and
re-bridging config had no effect — the threshold stayed locked to the launch
profile's value.

Test: two HERMES_HOME dirs (A -> B) prove the threshold moves with the
active profile after the fix.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_config(home: Path, agent_cfg: dict | None = None) -> None:
    cfg: dict = {}
    if agent_cfg:
        cfg["agent"] = agent_cfg
    (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")


def test_reconnect_attention_respects_profile_switch(tmp_path: Path) -> None:
    """Bridging a different profile's config must update the reconnect threshold.

    Before the fix, _RECONNECT_ATTENTION_AFTER_SECONDS was cached at import time,
    so profile B inherited profile A's threshold silently.
    """
    home_a = tmp_path / "profile_a"
    home_b = tmp_path / "profile_b"
    home_a.mkdir()
    home_b.mkdir()

    _write_config(home_a, agent_cfg={"reconnect_attention_after": 10})
    _write_config(home_b, agent_cfg={"reconnect_attention_after": 20})

    script = textwrap.dedent(
        f"""
        import os, sys, time
        sys.path.insert(0, {str(PROJECT_ROOT)!r})

        from pathlib import Path
        from gateway import run
        from gateway.run import _bridge_config_to_env, _load_bridge_config

        # Simulate serving profile A first
        cfg_a = _load_bridge_config(Path({str(home_a / 'config.yaml')!r}))
        _bridge_config_to_env(cfg_a)

        # 15 s > A's threshold of 10 -> should flag
        info_a = {{"queued_at": time.monotonic() - 15}}
        assert run._reconnect_needs_attention(info_a, time.monotonic()) is True, \
            "A threshold 10: 15s should flag"

        # 5 s < A's threshold of 10 -> should not flag
        info_a_ok = {{"queued_at": time.monotonic() - 5}}
        assert run._reconnect_needs_attention(info_a_ok, time.monotonic()) is False, \
            "A threshold 10: 5s should not flag"

        # Switch to profile B (same process, multiplex scenario)
        cfg_b = _load_bridge_config(Path({str(home_b / 'config.yaml')!r}))
        _bridge_config_to_env(cfg_b)

        # 15 s < B's threshold of 20 -> must NOT flag after fix
        info_b = {{"queued_at": time.monotonic() - 15}}
        result = run._reconnect_needs_attention(info_b, time.monotonic())
        print(f"RESULT={{result}}")
        """
    )

    env = dict(os.environ)
    env["HERMES_HOME"] = str(home_a)
    # Keep interpreter paths required by stdlib / platform detection
    for k in (
        "PATH", "PYTHONPATH", "VIRTUAL_ENV", "HOME", "USERPROFILE",
        "HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA", "APPDATA",
        "SYSTEMROOT", "TEMP", "TMP",
    ):
        if k in os.environ and k not in env:
            env[k] = os.environ[k]

    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"Subscript failed (rc={proc.returncode})\n"
            f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
        )
    assert "RESULT=False" in proc.stdout, (
        f"Expected False after B bridge (threshold 20, elapsed 15), got:\n"
        f"{proc.stdout}"
    )
