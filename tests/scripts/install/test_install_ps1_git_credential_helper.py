"""The installer stages the same git bytes pm does (#148552).

scripts/install.ps1 extracts the pinned PortableGit before pm records it, and pm's install keeps an
entry it cannot replace (a held git.exe) only when its digest equals the freshly staged tree
(#139088). So the installer's ``Set-PinnedGitCredentialHelper`` must make exactly the edit pm's
``Git.restage_edits`` makes: PortableGit's ``helper-selector`` picker becomes ``manager``.
"""
import subprocess
import textwrap
from pathlib import Path

import pytest

from pm.packages import Git
from tests.pm.test_git_credential_helper_default import STOCK

ROOT = Path(__file__).resolve().parents[3]
INSTALLER = ROOT / "scripts" / "install.ps1"


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("shipped", [STOCK, STOCK.replace(b"\n", b"\r\n")], ids=["lf", "crlf"])
def test_installer_names_the_credential_manager_exactly_as_pm_stages_it(tmp_path, shipped):
    tree = tmp_path / "tree"
    (tree / "etc").mkdir(parents=True)
    (tree / "etc" / "gitconfig").write_bytes(shipped)
    driver = tmp_path / "driver.ps1"
    driver.write_text(textwrap.dedent(f"""
        . '{INSTALLER}' -HermesHome '{tmp_path / "home"}'
        Set-PinnedGitCredentialHelper '{tree}'
    """), encoding="ascii")

    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(driver)],
        capture_output=True, timeout=300, check=False)

    assert result.returncode == 0, (result.stdout + result.stderr).decode("utf-8", errors="replace")
    expected = Git().restage_edits(_tree_with(tmp_path / "pm", shipped), "win32-x64")["etc/gitconfig"]
    assert (tree / "etc" / "gitconfig").read_bytes() == expected


def _tree_with(root: Path, gitconfig: bytes) -> Path:
    (root / "etc").mkdir(parents=True)
    (root / "etc" / "gitconfig").write_bytes(gitconfig)
    return root
