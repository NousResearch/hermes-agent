"""Keep the Windows payload free of file names MakeAppx refuses to pack.

Windows ships the bundled payload only inside an MSIX. Microsoft reserves the
OPC content-types stream name `[Content_Types].xml` for the package itself
("don't use them for app payload files", App packages and deployment:
https://learn.microsoft.com/previous-versions/windows/apps/hh464929(v=win.10)),
and MakeAppx fails the whole pack with 0x8007007b "The filename, directory
name, or volume label syntax is incorrect" when a payload file of that name
sits anywhere in the layout, not only at the root.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from pm.package import InstallError

RESERVED_NAMES = frozenset({"[content_types].xml"})

# Trees no runtime code reads that carry a reserved name. python-docx ships the
# unzipped source of templates/default.docx beside the .docx its Document()
# loads; nothing opens the unzipped copy.
DEAD_TREES = (("docx", "templates", "default-docx-template"),)


def drop_msix_reserved_names(root: Path) -> list[Path]:
    """Remove the dead trees holding reserved names and return them (relative
    to ``root``); raise naming every other reserved-name file."""
    root = Path(root)
    removed: list[Path] = []
    offending: list[str] = []
    for directory, dirnames, filenames in os.walk(root):
        reserved = [name for name in filenames if name.lower() in RESERVED_NAMES]
        if not reserved:
            continue
        relative = Path(directory).relative_to(root)
        tree = _dead_tree(relative.parts)
        if tree is None:
            offending += [(relative / name).as_posix() for name in reserved]
            continue
        shutil.rmtree(root / tree)
        removed.append(tree)
        if tree == relative:
            dirnames[:] = []
    if offending:
        raise InstallError("msix", "MSIX reserves [Content_Types].xml; MakeAppx cannot pack "
                           + ", ".join(sorted(offending)), "drop the tree in DEAD_TREES if no runtime code reads it")
    return removed


def _dead_tree(parts: tuple[str, ...]) -> Path | None:
    """The leading path through the DEAD_TREES entry that contains ``parts``."""
    folded = tuple(part.lower() for part in parts)
    for tree in DEAD_TREES:
        for start in range(len(folded) - len(tree) + 1):
            if folded[start:start + len(tree)] == tree:
                return Path(*parts[:start + len(tree)])
    return None
