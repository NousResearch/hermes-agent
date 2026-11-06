"""Directory references must not open a descriptor under the SQLite lifecycle lock."""

from pathlib import Path

import pytest

from cron import lifecycle_guard


@pytest.mark.parametrize("symlink", [False, pytest.param(True, marks=pytest.mark.platforms("posix"))])
def test_directory_reference_is_skipped_before_open(tmp_path, monkeypatch, symlink):
    directory = tmp_path / "Application Data"
    directory.mkdir()
    candidate = directory
    if symlink:
        candidate = tmp_path / "linked-data"
        candidate.symlink_to(directory, target_is_directory=True)

    opened = []
    scanned = []
    real_open = lifecycle_guard.os.open
    real_reader = lifecycle_guard._read_referenced_script

    def recording_open(path, *args, **kwargs):
        opened.append(Path(path))
        return real_open(path, *args, **kwargs)

    def recording_reader(path, **kwargs):
        scanned.append(path)
        return real_reader(path, **kwargs)

    monkeypatch.setattr(lifecycle_guard.os, "open", recording_open)
    monkeypatch.setattr(lifecycle_guard, "_read_referenced_script", recording_reader)
    assert real_reader(candidate) == (None, False)
    assert candidate not in opened

    command = f"python3 - <<'PY'\nfrom pathlib import Path\nprint(Path('{candidate}').is_dir())\nPY"
    assert lifecycle_guard.scan_gateway_lifecycle(command, cwd=str(tmp_path)) == (False, None)
    assert candidate in scanned
    assert candidate not in opened

    # Directory references must not prevent inspection of subsequent regular scripts.
    script = tmp_path / "restart.sh"
    script.write_text("hermes gateway restart\n", encoding="utf-8")
    assert lifecycle_guard.scan_gateway_lifecycle(f"bash {script}", cwd=str(tmp_path)) == (True, None)
    assert script in opened


@pytest.mark.platforms("posix")
def test_non_directory_special_file_still_fails_closed(tmp_path):
    fifo = tmp_path / "not-a-script"
    lifecycle_guard.os.mkfifo(fifo)
    assert lifecycle_guard._read_referenced_script(fifo) == (None, True)
