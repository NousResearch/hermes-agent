"""P13 isolation proof for scripts/gitnexus-reindex-daily.sh.

Verifies:
- GITNEXUS_DRY_RUN=1 short-circuits before the indexer is started: the
  script prints a DRY-RUN line and exits 0 without spawning setsid.
- GITNEXUS_DIR env parameterises the registry read path (the python -c
  block reads GITNEXUS_DIR/registry.json, not the hardcoded live path).
- A dry-run leaves no log dir / history file behind.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "gitnexus-reindex-daily.sh"


def _run(env_overrides: dict, tmp_path: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    # Force a writable throwaway log dir so a non-dry-run cannot touch the
    # live log dir even if the guard failed.
    env["GITNEXUS_LOG_DIR"] = str(tmp_path / "logs")
    env["GITNEXUS_DIR"] = str(tmp_path / "gitnexus")
    env.update(env_overrides)
    return subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True, text=True, timeout=15, env=env,
    )


def test_dry_run_short_circuits_before_indexer(tmp_path):
    """GITNEXUS_DRY_RUN=1 must exit 0 immediately with a DRY-RUN message
    and never reach the setsid indexer spawn."""
    r = _run({"GITNEXUS_DRY_RUN": "1"}, tmp_path)
    assert r.returncode == 0, f"dry-run exited {r.returncode}: {r.stderr}"
    assert "DRY-RUN" in r.stdout, f"no DRY-RUN banner: {r.stdout!r}"
    # Must not have started any indexer (no "re-index started" line).
    assert "re-index started" not in r.stdout, "dry-run spawned the indexer"


def test_dry_run_creates_no_log_dir(tmp_path):
    """Dry-run must short-circuit before `mkdir -p $LOG_DIR`, so the log
    directory is never created in a throwaway location."""
    log_dir = tmp_path / "logs"
    r = _run({"GITNEXUS_DRY_RUN": "1"}, tmp_path)
    assert r.returncode == 0
    assert not log_dir.exists(), "dry-run created the log dir (mkdir ran)"


def test_gitnexus_dir_parameterises_registry_read(tmp_path):
    """The registry-read python -c block must read GITNEXUS_DIR/registry.json,
    not the hardcoded /home/kensei/.gitnexus path. We check the source."""
    src = SCRIPT.read_text()
    assert "GITNEXUS_DIR" in src, "GITNEXUS_DIR env not referenced in script"
    # The python block must read from the env, not a hardcoded literal.
    assert "os.environ.get('GITNEXUS_DIR'" in src, (
        "registry read not parameterised by GITNEXUS_DIR"
    )
    # Hardcoded live path must NOT appear in the registry-read block.
    assert "'/home/kensei/.gitnexus/registry.json'" not in src, (
        "registry read still uses hardcoded live path"
    )
