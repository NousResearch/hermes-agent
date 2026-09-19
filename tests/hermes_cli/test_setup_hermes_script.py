from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_SCRIPT = REPO_ROOT / "setup-hermes.sh"


def test_setup_hermes_script_is_valid_shell():
    result = subprocess.run(["bash", "-n", str(SETUP_SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_setup_hermes_script_has_termux_path():
    content = SETUP_SCRIPT.read_text(encoding="utf-8")

    assert "is_termux()" in content
    assert ".[termux]" in content
    assert "constraints-termux.txt" in content
    assert "$PREFIX/bin" in content


def _extract_shell_region(start_pattern: str, end_pattern: str) -> str:
    """Pull a region out of setup-hermes.sh by awk-matching comment/code
    anchors rather than hardcoded line numbers, so the extraction survives
    unrelated edits to the script."""
    result = subprocess.run(
        [
            "awk",
            f"/{start_pattern}/{{p=1}} p{{if(/{end_pattern}/) exit; print}}",
            str(SETUP_SCRIPT),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _run_symlink_block(tmp_path: Path, create_acp_bin: bool) -> Path:
    """Run just setup-hermes.sh's command-link functions plus the PATH-setup
    symlink block (not the whole script — that does a full venv + all-extras
    install and takes minutes) against a fake SCRIPT_DIR/HOME, and return the
    fake HOME's ~/.local/bin dir for the caller to inspect."""
    repo_dir = tmp_path / "repo"
    home_dir = tmp_path / "home"
    venv_bin = repo_dir / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "hermes").write_text("#!/bin/sh\n")
    if create_acp_bin:
        (venv_bin / "hermes-acp").write_text("#!/bin/sh\n")

    functions = _extract_shell_region(r"^is_termux\(\) \{", r"^get_command_link_display_dir\(\) \{")
    # The end-pattern above stops one function short (it's also a start
    # anchor), so pull get_command_link_display_dir separately and append it.
    display_dir_fn = _extract_shell_region(r"^get_command_link_display_dir\(\) \{", r"^\}$")
    functions += display_dir_fn + "}\n"

    symlink_block = _extract_shell_region(
        r"# PATH setup — symlink hermes into a user-facing bin dir", r"^if is_termux; then$"
    )

    fragment = "#!/bin/bash\nset -e\n" + functions + f'\nSCRIPT_DIR="{repo_dir}"\n' + symlink_block
    script_path = tmp_path / "symlink_block.sh"
    script_path.write_text(fragment)

    env = {"HOME": str(home_dir), "PATH": "/usr/bin:/bin"}
    result = subprocess.run(
        ["bash", str(script_path)], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr

    return home_dir / ".local" / "bin"


def test_setup_hermes_script_symlinks_hermes_acp_when_present(tmp_path):
    command_link_dir = _run_symlink_block(tmp_path, create_acp_bin=True)

    hermes_link = command_link_dir / "hermes"
    acp_link = command_link_dir / "hermes-acp"

    assert hermes_link.is_symlink()
    assert acp_link.is_symlink()
    assert acp_link.resolve() == (tmp_path / "repo" / "venv" / "bin" / "hermes-acp").resolve()


def test_setup_hermes_script_skips_hermes_acp_when_absent(tmp_path):
    command_link_dir = _run_symlink_block(tmp_path, create_acp_bin=False)

    hermes_link = command_link_dir / "hermes"
    acp_link = command_link_dir / "hermes-acp"

    assert hermes_link.is_symlink()
    assert not acp_link.exists()
    assert not acp_link.is_symlink()
