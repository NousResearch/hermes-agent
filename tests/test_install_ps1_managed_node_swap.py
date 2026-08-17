"""Regression: the Test-Node managed-Node stage-and-swap must stay same-directory.

Review concern on #81500: ``Rename-Item`` rejects a path in ``-NewName`` --
with one carve-out. PowerShell's FileSystemProvider strips the directory and
keeps the leaf when the ``-NewName`` path shares the directory of ``-Path``
(``FileSystemProvider.RenameItem``: ``Path.GetDirectoryName(path) ==
Path.GetDirectoryName(newName)`` -> ``newName = Path.GetFileName(newName)``);
only a path that *differs in directory* throws "represents a path or device
name". The official docs state the same exception ("you can't supply a path
for the value of the NewName parameter, unless the path is identical to the
path specified in the Path parameter").

The swap in ``Test-Node`` relies on exactly that carve-out: it renames
between ``$HermesHome\node`` and sibling ``node.new-*`` / ``node.old-*``
paths (same directory, same volume -- atomic rename). This test pins the
invariant so a future refactor cannot silently introduce a cross-directory
rename, which would throw on every Windows install and read as a false
"in use" deferral.

Source-level because Linux CI cannot execute the Windows installer (same
rationale as the other ``tests/test_install_ps1_*.py`` probes).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"


def _swap_block() -> str:
    text = INSTALL_PS1.read_text(encoding="utf-8")
    start = text.index("# Rename-swap instead of delete-then-move:")
    end = text.index("# Session PATH so the rest of this run sees node/npm.")
    return text[start:end]


def test_managed_node_swap_defines_staged_and_backup_as_siblings() -> None:
    swap = _swap_block()
    # $staged / $backup must be siblings of the live tree -- that is what
    # makes every Rename-Item -NewName below a same-directory path.
    assert re.search(r'\$staged\s*=\s*"\$HermesHome\\node\.new-\$stamp"', swap), (
        "$staged must be defined as a sibling of the live tree "
        '("$HermesHome\\node.new-$stamp")'
    )
    assert re.search(r'\$backup\s*=\s*"\$HermesHome\\node\.old-\$stamp"', swap), (
        "$backup must be defined as a sibling of the live tree "
        '("$HermesHome\\node.old-$stamp")'
    )


def test_managed_node_swap_renames_stay_within_hermes_home() -> None:
    swap = _swap_block()

    renames = re.findall(
        r"Rename-Item\s+(\S+)\s+(\S+)\s+-ErrorAction", swap
    )
    assert len(renames) == 4, (
        f"expected 4 Rename-Item calls in the swap block, found {len(renames)}"
    )
    # -NewName accepts a path only when it shares the directory of -Path.
    # $staged / $backup are pinned to same-directory siblings by the test
    # above, so every call is a same-directory rename. Pin the actual swap
    # state machine too: live tree aside to $backup, staged tree into
    # place, and $backup restored on the rollback path. A cross-directory
    # path (e.g. "$HermesHome\sub\node") or a swapped direction must fail.
    allowed_pairs = {
        ('"$HermesHome\\node"', "$backup"),  # live tree aside
        ("$staged", '"$HermesHome\\node"'),  # staged into place (existing + fresh)
        ("$backup", '"$HermesHome\\node"'),  # rollback restore
    }
    for src, dst in renames:
        assert (src, dst) in allowed_pairs, (
            f"Rename-Item {src} {dst}: not one of the expected swap "
            "transitions (live->backup, staged->live, backup->live rollback) "
            "-- Rename-Item rejects paths that differ in directory from -Path"
        )
