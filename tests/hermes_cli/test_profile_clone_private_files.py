"""Owner-only tightening of cloned and migrated private files (PR #131874 follow-up).

``copy2``/``copytree`` preserve source mode bits and a bare ``write_text`` derives them from the
umask, so a loose source (umask 0o644) leaked credentials and agent memory into every new profile
or migration target. The invariants: ``.env``, ``memories/MEMORY.md`` and ``memories/USER.md`` are
0o600 after every copy path, whatever the source mode was; other cloned files keep their modes.
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

import hermes_constants
from hermes_cli.profiles import create_profile

pytestmark = pytest.mark.platforms("posix")

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "migration"
    / "openclaw-migration"
    / "scripts"
    / "openclaw_to_hermes.py"
)

_ENV_TEXT = "OPENAI_API_KEY=sk-model-key\n"
_MEMORY_TEXT = "# MEMORY.md\n\n- remembered fact\n"
_USER_TEXT = "# USER.md\n\n- prefers concise answers\n"


@contextlib.contextmanager
def umask(mask: int):
    """Run the block under an explicit umask (restored afterwards).

    umask 0o000 is the maximally permissive red case: any write path that relies on default
    open() modes creates 0o666 files here, so a regression cannot hide behind a restrictive
    developer umask.
    """
    old = os.umask(mask)
    try:
        yield
    finally:
        os.umask(old)


def file_mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


# ---------------------------------------------------------------------------
# hermes profile create --clone / --clone-all (hermes_cli/profiles.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A source home with loose (0o664) private files — the leak the fix exists for."""
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    (root / "memories").mkdir()
    with umask(0o002):
        (root / ".env").write_text(_ENV_TEXT, encoding="utf-8")
        (root / "memories" / "MEMORY.md").write_text(_MEMORY_TEXT, encoding="utf-8")
        (root / "memories" / "USER.md").write_text(_USER_TEXT, encoding="utf-8")
    return root


def test_clone_tightens_loose_memory_stores(home):
    """--clone must not propagate a loose source store: every owner-only file lands 0o600."""
    with umask(0o000):
        profile_dir = create_profile("bot2", clone_config=True, no_alias=True)

    assert (profile_dir / "memories" / "MEMORY.md").read_text(encoding="utf-8") == _MEMORY_TEXT
    assert file_mode(profile_dir / ".env") == 0o600
    assert file_mode(profile_dir / "memories" / "MEMORY.md") == 0o600
    assert file_mode(profile_dir / "memories" / "USER.md") == 0o600


def test_clone_all_tightens_loose_memory_stores(home):
    """The --clone-all copytree path gets the same guarantee."""
    with umask(0o000):
        profile_dir = create_profile("full", clone_all=True, no_alias=True)

    assert (profile_dir / "memories" / "MEMORY.md").read_text(encoding="utf-8") == _MEMORY_TEXT
    assert file_mode(profile_dir / ".env") == 0o600
    assert file_mode(profile_dir / "memories" / "MEMORY.md") == 0o600
    assert file_mode(profile_dir / "memories" / "USER.md") == 0o600


def test_clone_leaves_non_private_files_alone(home):
    """Tightening is scoped to the owner-only set: other cloned files keep their modes."""
    (home / "SOUL.md").write_text("Be helpful.", encoding="utf-8")
    os.chmod(str(home / "SOUL.md"), 0o644)

    with umask(0o000):
        profile_dir = create_profile("bot2", clone_config=True, no_alias=True)

    assert file_mode(profile_dir / "SOUL.md") == 0o644


def test_clone_never_tightens_through_a_symlinked_store(home, tmp_path):
    """A store kept as a symlink into a dotfiles repo must survive tightening: the chmod stays on
    the profile's own regular file, never on the link target outside the profile."""
    shared = tmp_path / "shared-memory.md"
    shared.write_text(_MEMORY_TEXT, encoding="utf-8")
    os.chmod(str(shared), 0o644)
    (home / "memories" / "MEMORY.md").unlink()
    (home / "memories" / "MEMORY.md").symlink_to(shared)

    with umask(0o000):
        profile_dir = create_profile("full", clone_all=True, no_alias=True)

    clone_store = profile_dir / "memories" / "MEMORY.md"
    assert clone_store.is_symlink(), "clone_all must preserve symlink to external store"
    assert file_mode(shared) == 0o644, "symlink target outside profile must not be chmodded"
    assert shared.read_text(encoding="utf-8") == _MEMORY_TEXT


# ---------------------------------------------------------------------------
# OpenClaw→Hermes migration copy path (openclaw_to_hermes.py copy_file)
# ---------------------------------------------------------------------------


def load_module():
    spec = importlib.util.spec_from_file_location("openclaw_to_hermes_copyfile", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_migrator(mod, source: Path, target: Path):
    return mod.Migrator(
        source_root=source,
        target_root=target,
        execute=True,
        workspace_target=None,
        overwrite=True,
        migrate_secrets=False,
        output_dir=target / "migration-report",
    )


def test_migration_copy_file_tightens_private_destinations(tmp_path: Path):
    """copy_file's write branch lands private destinations 0o600 even under umask 0o000 and a
    0o666 pre-existing target; the non-private copy keeps its derived mode."""
    mod = load_module()
    source = tmp_path / "src"
    target = tmp_path / "dst"
    (source / "memories").mkdir(parents=True)
    (target / "memories").mkdir(parents=True)
    (source / ".env").write_text(_ENV_TEXT, encoding="utf-8")
    (source / "memories" / "MEMORY.md").write_text(_MEMORY_TEXT, encoding="utf-8")
    (source / "SOUL.md").write_text("Be helpful.", encoding="utf-8")
    (target / "memories" / "MEMORY.md").write_text("stale content", encoding="utf-8")
    os.chmod(str(target / "memories" / "MEMORY.md"), 0o666)

    with umask(0o000):
        migrator = make_migrator(mod, source, target)
        migrator.copy_file(source / ".env", target / ".env", "user-profile")
        migrator.copy_file(source / "memories" / "MEMORY.md", target / "memories" / "MEMORY.md", "memory")
        migrator.copy_file(source / "SOUL.md", target / "SOUL.md", "soul")

    assert file_mode(target / ".env") == 0o600
    assert file_mode(target / "memories" / "MEMORY.md") == 0o600
    # copy2 preserves the source's 0o644: tightening is scoped to the private set only.
    assert file_mode(target / "SOUL.md") == 0o644
