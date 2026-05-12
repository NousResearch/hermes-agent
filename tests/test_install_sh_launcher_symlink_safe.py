"""Regression tests for install.sh launcher writing over a pre-existing symlink.

setup-hermes.sh creates ``~/.local/bin/hermes`` as a symlink → ``venv/bin/hermes``
via ``ln -sf``. If install.sh is then run on the same machine, the launcher
write at ``setup_path`` previously used ``cat > "$command_link_dir/hermes"``
which follows the symlink and truncates the venv entry point, leaving a shim
whose ``exec "$HERMES_BIN" "$@"`` calls itself. The ``hermes`` command then
hangs in an infinite recursion.

These tests guard the fix: ``rm -f`` must precede the ``cat >`` so a
pre-existing symlink is removed before the shim is written.
"""

import os
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def test_rm_minus_f_precedes_cat_in_setup_path() -> None:
    text = INSTALL_SH.read_text()

    rm_marker = 'rm -f "$command_link_dir/hermes"'
    cat_marker = 'cat > "$command_link_dir/hermes" <<EOF'

    assert rm_marker in text, (
        "setup_path must `rm -f` the launcher path before writing, so a "
        "pre-existing symlink (from setup-hermes.sh) isn't followed and "
        "clobbered."
    )
    assert cat_marker in text, "launcher write line missing"
    assert text.index(rm_marker) < text.index(cat_marker), (
        "`rm -f` must come BEFORE the `cat >` redirect — otherwise the "
        "redirect follows any pre-existing symlink and overwrites its target."
    )


def test_launcher_write_pattern_does_not_clobber_symlink_target(tmp_path: Path) -> None:
    """End-to-end: simulate the exact filesystem layout that triggered the bug
    and verify the patched write pattern leaves the venv entry point intact."""
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    venv_hermes = venv_bin / "hermes"
    sentinel = "#!/usr/bin/env python3\n# real entry point - must not be truncated\n"
    venv_hermes.write_text(sentinel)
    venv_hermes.chmod(0o755)

    link_dir = tmp_path / "local" / "bin"
    link_dir.mkdir(parents=True)
    link_path = link_dir / "hermes"
    link_path.symlink_to(venv_hermes)

    assert link_path.is_symlink()
    assert link_path.resolve() == venv_hermes.resolve()

    # Mirror the post-fix setup_path snippet from scripts/install.sh.
    script = f'''
set -eu
command_link_dir={link_dir!s}
HERMES_BIN={venv_hermes!s}
mkdir -p "$command_link_dir"
rm -f "$command_link_dir/hermes"
cat > "$command_link_dir/hermes" <<EOF
#!/usr/bin/env bash
unset PYTHONPATH
unset PYTHONHOME
exec "$HERMES_BIN" "\\$@"
EOF
chmod +x "$command_link_dir/hermes"
'''
    subprocess.run(["bash", "-c", script], check=True)

    assert venv_hermes.read_text() == sentinel, (
        "venv entry point was clobbered — `rm -f` did not protect the symlink "
        "target from being truncated by the launcher write."
    )

    assert not link_path.is_symlink(), "shim should be a regular file, not a symlink"
    shim_text = link_path.read_text()
    assert "unset PYTHONPATH" in shim_text
    assert "unset PYTHONHOME" in shim_text
    assert f'exec "{venv_hermes}" "$@"' in shim_text
    assert os.stat(link_path).st_mode & stat.S_IXUSR, "shim must be executable"
