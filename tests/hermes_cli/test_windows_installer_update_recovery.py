import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"


def _install_ps1() -> str:
    return INSTALL_PS1.read_text(encoding="utf-8")


def test_windows_repository_stage_repairs_managed_checkout_before_checkout():
    text = _install_ps1()

    helper_match = re.search(
        r"function Repair-ManagedCheckoutBeforeUpdate \{(?P<body>.*?)\nfunction Install-Repository",
        text,
        re.DOTALL,
    )
    assert helper_match, "Install-Repository should use a named checkout repair helper"
    helper = helper_match.group("body")

    assert "core.autocrlf false" in helper
    assert "reset --hard HEAD" in helper
    assert "clean -fd" in helper

    update_block_match = re.search(
        r"Existing installation found, updating\.\.\.(?P<body>.*?)\$didUpdate = \$true",
        text,
        re.DOTALL,
    )
    assert update_block_match, "could not find Install-Repository's existing-install update block"
    update_block = update_block_match.group("body")

    repair_idx = update_block.find("Repair-ManagedCheckoutBeforeUpdate")
    checkout_idx = update_block.find("checkout --detach $Commit")
    assert repair_idx != -1, "existing install updates should repair the managed checkout"
    assert checkout_idx != -1, "test expected the pinned commit checkout path"
    assert repair_idx < checkout_idx
