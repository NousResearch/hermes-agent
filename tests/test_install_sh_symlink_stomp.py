"""Regression tests for install.sh shim creation overwriting a prior symlink.

When users re-run scripts/install.sh on a system installed with an older layout,
$command_link_dir/hermes used to be a symlink pointing at venv/bin/hermes (the
pip-generated entry point). The shim writer used `cat > "$command_link_dir/hermes"`
which follows the symlink and rewrites the venv entry point with the shim body,
producing an `exec "$HERMES_BIN"` that calls itself. Result: `hermes` hangs on
every invocation. (#21454, #21513)

The fix is to `rm -f` the destination before the heredoc so the shim is created
as a regular file in $command_link_dir, leaving venv/bin/hermes intact.
"""

from __future__ import annotations

import os
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def test_shim_writer_removes_existing_path_before_heredoc() -> None:
    """Static guard: `rm -f` must precede the `cat >` heredoc for the shim."""
    text = INSTALL_SH.read_text()
    heredoc_marker = 'cat > "$command_link_dir/hermes" <<EOF'
    idx = text.index(heredoc_marker)
    preamble = text[:idx]
    last_rm = preamble.rfind('rm -f "$command_link_dir/hermes"')
    assert last_rm != -1, "missing `rm -f` before shim heredoc"
    last_mkdir = preamble.rfind('mkdir -p "$command_link_dir"')
    assert last_mkdir < last_rm < idx, "rm must sit between mkdir and heredoc"


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink semantics")
def test_shim_write_does_not_stomp_symlinked_venv_entrypoint(tmp_path: Path) -> None:
    """Behavioural reproducer: drive the real shim-write block from install.sh.

    Pre-create the symlink-to-venv layout that legacy installs left behind, run
    the shim creation block, and assert the venv entry point survives untouched.
    """
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    venv_entry = venv_bin / "hermes"
    pip_body = "#!/usr/bin/env python3\n# pip-generated entry point\nprint('venv-hermes')\n"
    venv_entry.write_text(pip_body)
    venv_entry.chmod(0o755)

    command_link_dir = tmp_path / "bin"
    command_link_dir.mkdir()
    legacy_link = command_link_dir / "hermes"
    legacy_link.symlink_to(venv_entry)
    assert legacy_link.is_symlink()

    text = INSTALL_SH.read_text()
    heredoc_start = text.index('cat > "$command_link_dir/hermes" <<EOF')
    heredoc_end = text.index('EOF\n    chmod +x "$command_link_dir/hermes"', heredoc_start)
    block_start = text.rfind('mkdir -p "$command_link_dir"', 0, heredoc_start)
    block_end = heredoc_end + len('EOF\n    chmod +x "$command_link_dir/hermes"')
    block = text[block_start:block_end]

    script = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail
        command_link_dir={command_link_dir!s}
        HERMES_BIN={venv_entry!s}
        {block}
        """
    )
    runner = tmp_path / "run.sh"
    runner.write_text(script)
    runner.chmod(0o755)

    subprocess.run(["bash", str(runner)], check=True)

    assert venv_entry.read_text() == pip_body, "venv entry point was overwritten through symlink"
    shim = command_link_dir / "hermes"
    assert shim.exists()
    assert not shim.is_symlink()
    shim_body = shim.read_text()
    assert "unset PYTHONPATH" in shim_body
    assert "unset PYTHONHOME" in shim_body
    assert f'exec "{venv_entry}"' in shim_body
    assert os.stat(shim).st_mode & stat.S_IXUSR
