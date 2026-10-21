"""Migration must not create world-readable private files.

The OpenClaw→Hermes migration writes two categories of private data with a
bare ``Path.write_text`` — which creates new files at ``0o666 & ~umask``
(0o644 at the common umask 0o022):

  - ``~/.hermes/.env`` holds plaintext provider API keys (``--migrate-secrets``)
  - merged memory stores (``memories/MEMORY.md``, ``USER.md``) hold private
    user data

and snapshots backups of those files with ``shutil.copy2``, which preserves
the source's (possibly loose) mode.  All three write paths must be
owner-only (0o600).  Mirrors the hardening in PR #130620 for
``hermes_cli/agent_import.py`` and ``tools/memory_tool_store.py``.
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("posix")

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "migration"
    / "openclaw-migration"
    / "scripts"
    / "openclaw_to_hermes.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("openclaw_to_hermes_private", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def umask(mask: int):
    """Run the block under an explicit umask (restored afterwards).

    umask 0x000 is the maximally permissive red case: any write path that
    relies on ``open(path, "w")`` defaults creates 0o666 files here, so a
    regression cannot hide behind a restrictive developer umask.
    """
    old = os.umask(mask)
    try:
        yield
    finally:
        os.umask(old)


def file_mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_migration_env_file_is_owner_only(tmp_path: Path):
    """~/.hermes/.env holds plaintext provider keys — must land at 0o600."""
    mod = load_module()
    source = tmp_path / ".openclaw"
    target = tmp_path / ".hermes"
    source.mkdir(parents=True)
    target.mkdir()
    (source / "credentials").mkdir()
    (source / "openclaw.json").write_text(
        json.dumps({"channels": {"telegram": {"botToken": "123:abc"}}}),
        encoding="utf-8",
    )
    (source / "credentials" / "telegram-default-allowFrom.json").write_text(
        json.dumps({"allowFrom": ["111", "222"]}),
        encoding="utf-8",
    )

    env_path = target / ".env"
    with umask(0o000):
        mod.Migrator(
            source_root=source,
            target_root=target,
            execute=True,
            workspace_target=None,
            overwrite=False,
            migrate_secrets=True,
            output_dir=target / "migration-report",
        ).migrate()

    assert "TELEGRAM_BOT_TOKEN=123:abc" in env_path.read_text(encoding="utf-8")
    assert file_mode(env_path) == 0o600, (
        f".env with plaintext provider keys created at "
        f"{oct(file_mode(env_path))}, expected 0o600"
    )


def test_migration_memory_store_is_owner_only(tmp_path: Path):
    """Merged memories/MEMORY.md holds private user data — must land at 0o600."""
    mod = load_module()
    source = tmp_path / ".openclaw"
    target = tmp_path / ".hermes"
    source.mkdir(parents=True)
    target.mkdir()
    (source / "workspace").mkdir()
    (source / "openclaw.json").write_text("{}", encoding="utf-8")
    (source / "workspace" / "MEMORY.md").write_text(
        "# MEMORY.md\n\n- private note: birthday is April 3rd\n",
        encoding="utf-8",
    )

    memory_path = target / "memories" / "MEMORY.md"
    with umask(0o000):
        mod.Migrator(
            source_root=source,
            target_root=target,
            execute=True,
            workspace_target=None,
            overwrite=False,
            migrate_secrets=False,
            output_dir=target / "migration-report",
            selected_options={"memory"},
        ).migrate()

    assert memory_path.exists()
    assert "private note" in memory_path.read_text(encoding="utf-8")
    assert file_mode(memory_path) == 0o600, (
        f"merged memory store created at {oct(file_mode(memory_path))}, "
        f"expected 0o600"
    )


def test_migration_backup_of_loose_file_is_owner_only(tmp_path: Path):
    """copy2 preserves source modes; the .env backup of a loose live file
    would otherwise stay world-readable."""
    mod = load_module()
    source = tmp_path / ".openclaw"
    target = tmp_path / ".hermes"
    source.mkdir(parents=True)
    target.mkdir()
    (source / "credentials").mkdir()
    (source / "openclaw.json").write_text("{}", encoding="utf-8")
    (source / "credentials" / "telegram-default-allowFrom.json").write_text(
        json.dumps({"allowFrom": ["111"]}),
        encoding="utf-8",
    )
    # Pre-existing loose .env in the target (created by an older tool at 0644).
    env_path = target / ".env"
    env_path.write_text("PREEXISTING=1\n", encoding="utf-8")
    os.chmod(env_path, 0o644)

    with umask(0o000):
        mod.Migrator(
            source_root=source,
            target_root=target,
            execute=True,
            overwrite=True,
            workspace_target=None,
            migrate_secrets=True,
            output_dir=target / "migration-report",
            selected_options={"messaging-settings"},
        ).migrate()

    backups = list((target / "migration-report" / "backups").rglob(".env"))
    assert backups, "expected a backup of the pre-existing .env"
    assert file_mode(backups[0]) == 0o600, (
        f"backup of loose .env snapshotted at {oct(file_mode(backups[0]))}, "
        f"expected 0o600"
    )


def test_backup_existing_of_loose_file_tightens_to_owner_only(tmp_path: Path):
    """Directly verifies backup_existing tightens a 0o644 file to 0o600."""
    mod = load_module()
    loose = tmp_path / "loose.txt"
    loose.write_text("important data\n", encoding="utf-8")
    os.chmod(loose, 0o644)
    backup_dir = tmp_path / "backups"

    with umask(0o000):
        backup_path = mod.backup_existing(loose, backup_dir)

    assert backup_path is not None
    assert backup_path.exists()
    assert file_mode(backup_path) == 0o600


def test_migration_daily_memory_is_owner_only(tmp_path: Path):
    """workspace/memory/*.md daily notes merged into MEMORY.md must land at 0o600."""
    mod = load_module()
    source = tmp_path / ".openclaw"
    target = tmp_path / ".hermes"
    source.mkdir(parents=True)
    target.mkdir()
    (source / "workspace" / "memory").mkdir(parents=True)
    (source / "openclaw.json").write_text("{}", encoding="utf-8")
    (source / "workspace" / "memory" / "2026-03-01.md").write_text(
        "# March 1\n- Daily standup notes\n",
        encoding="utf-8",
    )

    memory_path = target / "memories" / "MEMORY.md"
    with umask(0o000):
        mod.Migrator(
            source_root=source,
            target_root=target,
            execute=True,
            workspace_target=None,
            overwrite=False,
            migrate_secrets=False,
            output_dir=target / "migration-report",
            selected_options={"daily-memory"},
        ).migrate()

    assert memory_path.exists()
    assert "Daily standup notes" in memory_path.read_text(encoding="utf-8")
    assert file_mode(memory_path) == 0o600, (
        f"daily memory store created at {oct(file_mode(memory_path))}, expected 0o600"
    )


def test_atomic_write_private_tightens_loose_existing_target(tmp_path: Path):
    """Every write through the helper ends at 0o600 — so a re-run also
    tightens a previously loose file (e.g. a 0o644 .env created by an older
    tool) instead of preserving its old bits."""
    mod = load_module()
    path = tmp_path / "store.md"
    path.write_text("existing\n", encoding="utf-8")
    os.chmod(path, 0o644)

    with umask(0o000):
        mod.atomic_write_private(path, "replaced\n")

    assert path.read_text(encoding="utf-8") == "replaced\n"
    assert file_mode(path) == 0o600
    # Atomic write — no temp file survives.
    assert [p.name for p in tmp_path.glob(".tmp*")] == []


def test_atomic_write_private_creates_new_file_owner_only_under_umask_000(tmp_path: Path):
    mod = load_module()
    path = tmp_path / "nested" / "fresh.env"

    with umask(0o000):
        mod.atomic_write_private(path, "KEY=value\n")

    assert path.read_text(encoding="utf-8") == "KEY=value\n"
    assert file_mode(path) == 0o600
    assert [p.name for p in path.parent.glob(".tmp*")] == []
