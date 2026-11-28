"""A named profile home a late writer is still bound to after ``profile delete`` or ``profile rename``."""

from __future__ import annotations

import shutil
from pathlib import Path

from hermes_constants import mark_named_profile_deleted, profile_tombstone_path

MOVES = ("delete", "rename")


def dead_named_home(root: Path, move: str) -> Path:
    """Build ``<root>/.hermes/profiles/b``, then delete it (tombstone + rmtree) or rename it to ``c``;
    return the old ``b`` path, which must stay absent after any write."""
    home = root / ".hermes" / "profiles" / "b"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    if move == "delete":
        mark_named_profile_deleted(home)
        shutil.rmtree(home)
    else:
        home.rename(home.with_name("c"))
    return home


def stale_writer_shell(home: Path) -> Path:
    """The directory a writer still bound to a deleted *home* re-created before the guard in
    ``hermes_constants`` refused it. Hosts upgraded from such builds still carry these shells,
    so ``profile create``, list and serve must keep handling them."""
    record = profile_tombstone_path(home)
    saved = record.read_bytes()
    record.unlink()
    try:
        home.mkdir(parents=True)
    finally:
        record.write_bytes(saved)
    return home
