"""Regression coverage for bounded startup subdirectory-hint reads (#10047)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from agent import subdirectory_hints as sh_mod


def test_tracker_init_bounds_slow_root_hint_read(tmp_path, monkeypatch):
    """The real tracker startup path must not wait for a stalled root hint read."""
    hint = tmp_path / "AGENTS.md"
    hint.write_text("Root project instructions", encoding="utf-8")

    pb_mod = sys.modules[sh_mod._read_text_with_timeout.__module__]
    monkeypatch.setattr(pb_mod, "_get_context_file_read_timeout", lambda: 0.05)

    original_read_text = Path.read_text

    def slow_read_text(self, *args, **kwargs):
        time.sleep(1.0)
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", slow_read_text)

    started = time.monotonic()
    tracker = sh_mod.SubdirectoryHintTracker(str(tmp_path))
    elapsed = time.monotonic() - started

    assert elapsed < 0.5, f"tracker startup blocked for {elapsed:.2f}s"
    assert tracker._loaded_digests == set()


def test_first_hint_file_healthy_path_is_unchanged(tmp_path):
    hint = tmp_path / "AGENTS.md"
    hint.write_text("  Root instructions\n", encoding="utf-8")

    path, body = sh_mod._first_hint_file(tmp_path)

    assert path == hint
    assert body == "Root instructions"


def test_first_hint_file_reads_vetted_symlink_target_through_bounded_helper(tmp_path, monkeypatch):
    target = tmp_path / "real-agents.md"
    target.write_text("Resolved instructions", encoding="utf-8")
    link = tmp_path / "AGENTS.md"
    try:
        link.symlink_to(target.name)
    except OSError:
        pytest.skip("symlinks unavailable on this platform")

    seen = []
    original = sh_mod._read_text_with_timeout

    def capture(path, timeout=None):
        seen.append(path)
        return original(path, timeout=timeout)

    monkeypatch.setattr(sh_mod, "_read_text_with_timeout", capture)

    path, body = sh_mod._first_hint_file(tmp_path)

    assert path == link
    assert body == "Resolved instructions"
    assert seen == [target.resolve()]
