"""_first_hint_file must not block session startup on a stalled hint read.

Regression for #10047: the startup hint seeding path
(``SubdirectoryHintTracker.__init__`` -> ``_first_hint_file``) did a bare
synchronous ``read_text`` — an evicted iCloud Drive file downloads on read and
blocks at the OS level for minutes. teknium1's commit 972b94bd2935 ("time out
slow context file reads") covered every context-file read but missed this call
site. The read now goes through ``_read_text_with_timeout`` like the sibling
``_load_hints_for_directory`` path.
"""

import time
from pathlib import Path

from agent import subdirectory_hints as sh_mod


def _patch_timeout(monkeypatch, seconds):
    import sys

    pb_mod = sys.modules[sh_mod._read_text_with_timeout.__module__]
    monkeypatch.setattr(pb_mod, "_get_context_file_read_timeout", lambda: seconds)


def test_first_hint_file_times_out_on_slow_read(tmp_path, monkeypatch):
    """A stalled hint file read is skipped instead of blocking startup seeding."""
    (tmp_path / "AGENTS.md").write_text("Root project instructions", encoding="utf-8")
    _patch_timeout(monkeypatch, 0.05)

    original_read_text = Path.read_text

    def slow_read_text(self, *args, **kwargs):
        time.sleep(1.0)
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", slow_read_text)

    start = time.monotonic()
    result = sh_mod._first_hint_file(tmp_path)
    elapsed = time.monotonic() - start

    assert result is None, "slow hint file should be skipped, not block seeding"
    assert elapsed < 0.5, f"hint seeding blocked for {elapsed:.2f}s"


def test_first_hint_file_normal_path(tmp_path):
    """Healthy hint files still load with content stripped."""
    (tmp_path / "AGENTS.md").write_text("  Root instructions\n", encoding="utf-8")
    path, content = sh_mod._first_hint_file(tmp_path)
    assert path == tmp_path / "AGENTS.md"
    assert content == "Root instructions"


def test_first_hint_file_no_hints(tmp_path):
    """A directory with no hint files yields None."""
    (tmp_path / "main.py").write_text("print('hello')", encoding="utf-8")
    assert sh_mod._first_hint_file(tmp_path) is None
