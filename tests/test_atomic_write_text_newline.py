"""``atomic_write_text(newline=...)`` — exact bytes on every platform.

The default text-handle translation (``newline=None``) turns every ``"\\n"`` into
``os.linesep`` — CRLF on Windows — so a caller that rewrites a file one line at a time
silently rewrites the whole file's line endings (this is what made ``skill_manage``'s
patch turn an LF ``SKILL.md`` into CRLF: dirty worktree, whole-file diff). ``newline=""``
opts out of translation so the caller's bytes land as-is, on Windows and on Linux alike.
"""

import os
from pathlib import Path

import pytest

from utils import atomic_write_text


class TestAtomicWriteTextNewline:
    def test_no_translation_writes_exact_bytes(self, tmp_path):
        path = tmp_path / "exact.md"
        atomic_write_text(path, "a\nb\n", newline="")
        assert path.read_bytes() == b"a\nb\n"

    def test_no_translation_keeps_crlf_content(self, tmp_path):
        path = tmp_path / "crlf.md"
        atomic_write_text(path, "a\r\nb\r\n", newline="")
        assert path.read_bytes() == b"a\r\nb\r\n"

    def test_no_translation_on_overwrite_with_mode_preservation(self, tmp_path):
        """The opt-in must survive the rewrite path callers actually use."""
        path = tmp_path / "existing.md"
        path.write_bytes(b"old\n")
        atomic_write_text(path, "new\n", newline="", preserve_mode=True, create_mode=0o644)
        assert path.read_bytes() == b"new\n"

    @pytest.mark.skipif(os.linesep != "\r\n", reason="only a CRLF platform translates")
    def test_default_still_translates_on_a_crlf_platform(self, tmp_path):
        """Pins why callers must opt in: the default is platform-dependent."""
        path = tmp_path / "translated.md"
        atomic_write_text(path, "a\nb\n")
        assert path.read_bytes() == b"a\r\nb\r\n"
