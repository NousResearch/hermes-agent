"""Directory literals in script source are not executable script references."""

import os

import pytest

from cron.lifecycle_guard import (
    contains_gateway_lifecycle_command_or_referenced_script,
)


def _classify_script(tmp_path, source):
    script = tmp_path / "inspect_paths.py"
    script.write_text("#!/usr/bin/env python3\n" + source)
    script.chmod(0o755)
    # Scan an actual referenced script without executing it.
    return contains_gateway_lifecycle_command_or_referenced_script(
        "./inspect_paths.py", cwd=str(tmp_path)
    )


def test_existing_directory_literal_is_safe(tmp_path):
    directory = tmp_path / "existing output"
    directory.mkdir()
    assert not _classify_script(
        tmp_path, f"from pathlib import Path\nROOT = Path({str(directory)!r})\n"
    )


def test_nonexistent_path_literal_is_safe(tmp_path):
    missing = tmp_path / "missing output"
    assert not missing.exists()
    assert not _classify_script(
        tmp_path, f"from pathlib import Path\nROOT = Path({str(missing)!r})\n"
    )


def test_literal_gateway_restart_in_script_is_unsafe(tmp_path):
    assert _classify_script(
        tmp_path, 'import os\nos.system("hermes gateway restart")\n'
    )


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="Requires POSIX FIFOs")
def test_fifo_reference_remains_unsafe(tmp_path):
    # A FIFO can supply script bytes; unlike a directory it must remain unsafe.
    fifo = tmp_path / "script-input"
    os.mkfifo(fifo)
    assert _classify_script(
        tmp_path, f"from pathlib import Path\nROOT = Path({str(fifo)!r})\n"
    )
