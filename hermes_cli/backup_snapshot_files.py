"""File discovery for quick state snapshots."""
import os
from collections.abc import Iterator
from pathlib import Path


def iter_quick_snapshot_files(root: Path) -> Iterator[Path]:
    """Skip regenerable board trees before scanning their potentially large contents."""
    excluded = {"workspaces", "attachments"}
    for directory, dirs, files in os.walk(root, topdown=True, followlinks=False):
        dirs[:] = [name for name in dirs if name not in excluded]
        for name in files:
            # Match the existing component exclusion for files bearing these names too.
            if name not in excluded:
                yield Path(directory) / name
