"""Regression tests for install.sh Python environment sanitization.

When install.sh is launched from another Python-driven tool session, inherited
PYTHONPATH/PYTHONHOME can shadow the freshly installed checkout. The installer
must sanitize those vars both during installation and at runtime launch.
"""

import os
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def test_install_script_unsets_pythonpath_and_pythonhome_early() -> None:
    text = INSTALL_SH.read_text()

    # During install, inherited Python env must be sanitized before pip/venv use.
    assert 'unset PYTHONPATH' in text
    assert 'unset PYTHONHOME' in text


def test_hermes_launcher_wrapper_clears_python_env_before_exec() -> None:
    text = INSTALL_SH.read_text()

    # Wrapper should clear env and forward args untouched to the venv entrypoint.
    assert 'launcher_tmp="$(mktemp "$command_link_dir/.hermes-launcher.XXXXXX")"' in text
    assert 'cat > "$launcher_tmp" <<EOF' in text
    assert 'unset PYTHONPATH' in text
    assert 'unset PYTHONHOME' in text
    assert 'exec "$HERMES_BIN" "\\$@"' in text
    assert 'mv -f "$launcher_tmp" "$command_link_dir/hermes"' in text


def test_setup_path_does_not_clobber_symlinked_venv_entrypoint(tmp_path) -> None:
    """setup_path() must replace the launcher symlink, not overwrite its target."""
    install_source = INSTALL_SH.read_text(encoding="utf-8")
    install_prelude, _, _ = install_source.rpartition("\nmain\n")
    assert install_prelude, "expected trailing `main` call in scripts/install.sh"

    install_dir = tmp_path / "install"
    home = tmp_path / "home"
    venv_bin = install_dir / "venv" / "bin"
    launcher_dir = home / ".local" / "bin"
    hermes_bin = venv_bin / "hermes"
    launcher = launcher_dir / "hermes"

    venv_bin.mkdir(parents=True)
    launcher_dir.mkdir(parents=True)
    hermes_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    hermes_bin.chmod(hermes_bin.stat().st_mode | stat.S_IXUSR)
    launcher.symlink_to(hermes_bin)

    subprocess.run(
        [
            "bash",
            "-lc",
            f"""
            set -euo pipefail
            {install_prelude}
            log_info() {{ :; }}
            log_success() {{ :; }}
            log_warn() {{ :; }}
            USE_VENV=true
            INSTALL_DIR="$TEST_INSTALL_DIR"
            HOME="$TEST_HOME"
            DISTRO=linux
            ROOT_FHS_LAYOUT=false
            setup_path
            """,
        ],
        check=True,
        env={
            **os.environ,
            "TEST_INSTALL_DIR": str(install_dir),
            "TEST_HOME": str(home),
        },
        text=True,
    )

    assert hermes_bin.read_text(encoding="utf-8") == "#!/bin/sh\nexit 0\n"
    assert not launcher.is_symlink()
    launcher_text = launcher.read_text(encoding="utf-8")
    assert 'unset PYTHONPATH' in launcher_text
    assert 'unset PYTHONHOME' in launcher_text
    assert f'exec "{hermes_bin}" "$@"' in launcher_text
