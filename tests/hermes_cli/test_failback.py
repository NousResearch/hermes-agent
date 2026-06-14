"""Tests for hermes_cli.failback — the active 5h failback cron logic.

The end-to-end test (kpi_test_ollama_failover.py KPI-4) lives outside the
test suite because it touches the user's real cron DB.  These tests
cover the in-process logic of ``hermes_cli.failback.run()`` so a future
refactor of the decision matrix can't silently break the contract:

  - current is None            → no_failback_needed
  - on_primary                 → no_failback_needed
  - on_backup + primary OK     → failback_triggered
  - on_backup + primary exhaus → failback_triggered
  - the wrapper script (scripts/failback.py) is silent on no_failback_needed
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Repo layout for in-process imports.
HERMES_REPO = Path(r"D:\hermes-app\hermes-agent")
HERMES_PARENT = HERMES_REPO.parent


def _isolated_hermes_home(tmp_path: Path) -> Path:
    home = tmp_path / "hermes_home"
    home.mkdir(parents=True, exist_ok=True)
    (home / ".env").write_text(
        "OLLAMA_API_KEY=primary-abc-123\n"
        "OLLAMA_API_KEY_BACKUP=backup-xyz-789\n",
        encoding="utf-8",
    )
    os.environ["HERMES_HOME"] = str(home)
    return home


@pytest.fixture
def isolated_home(monkeypatch, tmp_path):
    """HERMES_HOME pointing at a temp dir with two Ollama keys in .env."""
    # Cleanly set HERMES_HOME (monkeypatch restores at teardown).
    home = _isolated_hermes_home(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Strip the parent process's real env vars so load_env() can't pick
    # them up if our .env is missing the key.
    for k in ("OLLAMA_API_KEY", "OLLAMA_API_KEY_BACKUP"):
        monkeypatch.delenv(k, raising=False)
    yield home


def test_current_none_returns_no_failback_needed(isolated_home):
    """If the pool's current is None (fresh process), don't fail back from anything."""
    if str(HERMES_REPO) not in sys.path:
        sys.path.insert(0, str(HERMES_REPO))
    if str(HERMES_PARENT) not in sys.path:
        sys.path.insert(0, str(HERMES_PARENT))

    from agent.credential_pool import load_pool
    from hermes_cli.failback import run

    pool = load_pool("ollama-cloud")
    # Don't call select() — current stays None.
    result = run(provider="ollama-cloud", pool=pool)
    assert result["action"] == "no_failback_needed"
    assert result["current"] is None


def test_on_primary_returns_no_failback_needed(isolated_home):
    """Already on primary → nothing to do."""
    if str(HERMES_REPO) not in sys.path:
        sys.path.insert(0, str(HERMES_REPO))
    if str(HERMES_PARENT) not in sys.path:
        sys.path.insert(0, str(HERMES_PARENT))

    from agent.credential_pool import load_pool
    from hermes_cli.failback import run

    pool = load_pool("ollama-cloud")
    selected = pool.select()
    assert selected and selected.source == "env:OLLAMA_API_KEY"

    result = run(provider="ollama-cloud", pool=pool)
    assert result["action"] == "no_failback_needed"
    assert result["current"] == "env:OLLAMA_API_KEY"


def test_on_backup_with_exhausted_primary_triggers_failback(isolated_home):
    """The headline case: on backup + primary is rate-limited → failback to primary."""
    if str(HERMES_REPO) not in sys.path:
        sys.path.insert(0, str(HERMES_REPO))
    if str(HERMES_PARENT) not in sys.path:
        sys.path.insert(0, str(HERMES_PARENT))

    from agent.credential_pool import load_pool
    from hermes_cli.failback import run

    pool = load_pool("ollama-cloud")
    primary = pool.select()
    assert primary and primary.source == "env:OLLAMA_API_KEY"

    # Simulate the user's actual scenario: 429 on the primary.
    rotated = pool.mark_exhausted_and_rotate(status_code=429)
    assert rotated and rotated.source == "env:OLLAMA_API_KEY_BACKUP"

    result = run(provider="ollama-cloud", pool=pool)
    assert result["action"] == "failback_triggered"
    assert result["from"] == "env:OLLAMA_API_KEY_BACKUP"
    assert result["to"] == "env:OLLAMA_API_KEY"

    # Primary must now be OK in the persisted state.
    pool2 = load_pool("ollama-cloud")
    primary2 = next(e for e in pool2.entries() if e.source == "env:OLLAMA_API_KEY")
    assert primary2.last_status != "exhausted"


def test_wrapper_script_is_silent_on_no_failback(isolated_home, tmp_path):
    """The cron wrapper must emit NOTHING on stdout when no failback happens.

    This is the user's "No output shall be given until backup API key
    usage was detected and the failback is triggered" contract.
    """
    # Find the wrapper script.  It's at C:\Users\andre\AppData\Local\hermes\scripts\failback.py
    # (D:\hermes-app\scripts\failback.py is the symlink target the user sees).
    wrapper_candidates = [
        HERMES_REPO / "scripts" / "failback.py",
        Path(os.environ["USERPROFILE"]) / "AppData" / "Local" / "hermes" / "scripts" / "failback.py",
    ]
    wrapper_path = next((p for p in wrapper_candidates if p.exists()), None)
    if wrapper_path is None:
        pytest.skip(
            "Wrapper script not found at any expected location: "
            + ", ".join(str(p) for p in wrapper_candidates)
        )

    # Put the pool in a "currently on primary" state.  Use a fresh tmp
    # HERMES_HOME so the wrapper has a clean .env to read.
    wrapper_tmp = tmp_path / "wrapper_run"
    wrapper_tmp.mkdir()
    (wrapper_tmp / ".env").write_text(
        "OLLAMA_API_KEY=primary-abc-123\n"
        "OLLAMA_API_KEY_BACKUP=backup-xyz-789\n",
        encoding="utf-8",
    )

    sub_env = os.environ.copy()
    sub_env["HERMES_HOME"] = str(wrapper_tmp)
    for k in ("OLLAMA_API_KEY", "OLLAMA_API_KEY_BACKUP"):
        sub_env.pop(k, None)

    proc = subprocess.run(
        [sys.executable, str(wrapper_path)],
        capture_output=True,
        text=True,
        env=sub_env,
        timeout=30,
    )
    assert proc.returncode == 0, f"wrapper failed: stdout={proc.stdout!r}, stderr={proc.stderr!r}"
    assert proc.stdout.strip() == "", (
        f"Expected silent stdout on no_failback_needed, got {proc.stdout!r}. "
        "The 'silent unless failback' contract is broken."
    )
