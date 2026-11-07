import os
from pathlib import Path
import shlex
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_SCRIPT = REPO_ROOT / "setup-hermes.sh"


def test_setup_hermes_script_is_valid_shell():
    result = subprocess.run(["bash", "-n", str(SETUP_SCRIPT)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.platforms("posix")
@pytest.mark.skipif(getattr(os, "geteuid", lambda: 1)() == 0, reason="root ignores file permissions")
def test_read_only_managed_zshrc_is_left_alone_and_setup_continues(tmp_path):
    """Regression for #15016: home-manager/nix-darwin link ~/.zshrc to a read-only store file."""
    store = tmp_path / "store"
    store.mkdir()
    managed = store / "zshrc"
    managed.write_text("# managed by home-manager\n", encoding="utf-8")
    managed.chmod(0o444)
    store.chmod(0o555)
    home = tmp_path / "home"
    home.mkdir()
    (home / ".zshrc").symlink_to(managed)
    env = dict(os.environ, HOME=str(home), SHELL="/bin/zsh")
    # setup-hermes.sh runs under `set -e`; reaching the echo proves the step did not abort it.
    script = f"source {shlex.quote(SETUP_SCRIPT.as_posix())}\nwire_shell_path\necho continued\n"
    try:
        result = subprocess.run(["bash", "-c", script], env=env, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=30, check=False)
    finally:
        store.chmod(0o755)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.rstrip().endswith("continued"), result.stdout + result.stderr
    assert managed.read_text(encoding="utf-8") == "# managed by home-manager\n"
    _, _, instruction = result.stdout.partition(f"Cannot write {home / '.zshrc'}")
    assert 'export PATH="$HOME/.local/bin:$PATH"' in instruction, result.stdout
