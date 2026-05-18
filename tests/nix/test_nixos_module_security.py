from pathlib import Path

MODULE = Path(__file__).resolve().parents[2] / "nix" / "nixosModules.nix"


def _module_text() -> str:
    return MODULE.read_text()


def test_native_readwritepaths_does_not_duplicate_state_workspace() -> None:
    text = _module_text()

    assert "workingDirectoryInStateDir" in text
    assert "nativeReadWritePaths = [ cfg.stateDir ]" in text
    assert "++ lib.optional (! workingDirectoryInStateDir) cfg.workingDirectory" in text
    assert "ReadWritePaths = nativeReadWritePaths;" in text
    assert "ReadWritePaths = [\n              cfg.stateDir\n              cfg.workingDirectory\n            ];" not in text


def test_activation_refuses_symlinked_managed_directories() -> None:
    text = _module_text()

    assert "ensure_hermes_dir()" in text
    assert "if [ -L \"$_path\" ]; then" in text
    assert "chown -h ${cfg.user}:${cfg.group} \"$_path\"" in text
    assert "assert_parent_not_service_writable()" in text
    assert "ensure_hermes_dir ${lib.escapeShellArg cfg.workingDirectory} 2770" in text
    assert '"d ${cfg.workingDirectory}         2770 ${cfg.user} ${cfg.group} - -"' not in text
