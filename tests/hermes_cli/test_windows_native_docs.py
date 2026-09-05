from pathlib import Path


def test_windows_native_install_path_docs_match_installer() -> None:
    doc = Path("website/docs/user-guide/windows-native.md").read_text()
    install = Path("scripts/install.ps1").read_text()

    # The launchers live in the managed binary dir OUTSIDE the git checkout
    # (HERMES_HOME\bin, next to the managed uv) — NOT the whole venv\Scripts
    # (which would shadow the user's python, #83797) and NOT a dir inside
    # the checkout (which `hermes update`'s autostash swept off disk).
    assert "%LOCALAPPDATA%\\hermes\\bin" in doc
    assert (
        "Get-Command hermes        # should print "
        "C:\\Users\\<you>\\AppData\\Local\\hermes\\bin\\hermes.exe"
    ) in doc
    # Installer exposes $HermesHome\bin, and must copy the launchers into it.
    assert '$hermesBin = "$HermesHome\\bin"' in install
    assert "hermes.exe" in install and "hermes-acp.exe" in install
    # Guard against regressions to either legacy layout.
    assert '$hermesBin = "$InstallDir\\venv\\Scripts"' not in install
    assert '$hermesBin = "$InstallDir\\bin"' not in install
