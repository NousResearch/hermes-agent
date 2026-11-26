"""A named profile home a late writer is still bound to after ``profile delete`` or ``profile rename``."""

from __future__ import annotations

import shutil
from pathlib import Path

from hermes_constants import mark_named_profile_deleted

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
