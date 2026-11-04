"""The git Hermes stages on Windows asks the bundled Git Credential Manager, never the helper picker.

Regression for #148552: PortableGit's ``etc/gitconfig`` sets ``credential.helper = helper-selector``,
a picker window git opens on every credential lookup. Unattended git (cron scripts, the gateway, the
agent's terminal) waits on it with nobody to answer. The Git for Windows installer sets ``manager``
instead, and PortableGit ships the same Git Credential Manager.
"""

from __future__ import annotations

import pytest

from pm.packages import Git
from pm.store import Store

# PortableGit 2.53.0.windows.3's etc/gitconfig, byte for byte.
STOCK = (
    b"[core]\n\tsymlinks = false\n\tautocrlf = true\n\tfscache = true\n[color]\n\tinteractive = true\n"
    b"\tui = auto\n[help]\n\tformat = html\n[diff \"astextplain\"]\n\ttextconv = astextplain\n[rebase]\n"
    b"\tautosquash = true\n[filter \"lfs\"]\n\tclean = git-lfs clean -- %f\n\tsmudge = git-lfs smudge -- %f\n"
    b"\tprocess = git-lfs filter-process\n\trequired = true\n[credential]\n\thelper = helper-selector\n"
)


MANAGER = STOCK.replace(b"helper = helper-selector", b"helper = manager")


@pytest.mark.parametrize(("shipped", "staged"), [
    (STOCK, MANAGER),
    (STOCK.replace(b"\n", b"\r\n"), MANAGER.replace(b"\n", b"\r\n")),
    (MANAGER, MANAGER),
], ids=["stock", "crlf", "already-staged"])
def test_staged_portable_git_names_the_credential_manager(tmp_path, shipped, staged):
    tree = tmp_path / "tree"
    (tree / "etc").mkdir(parents=True)
    (tree / "etc" / "gitconfig").write_bytes(shipped)

    Git().stage(Store(tmp_path / "store"), tree, "2.53.0+3", "win32-x64")

    assert (tree / "etc" / "gitconfig").read_bytes() == staged
