"""#121572: all renderers must use the non-imitable compression marker."""

from __future__ import annotations

import pathlib

from agent.compression_marker import (
    _COMPRESSION_MARKER_PREFIX,
    _COMPRESSION_MARKER_RE,
    elide_text,
)


def test_elide_text_embeds_counts_and_prefix():
    out = elide_text("x" * 1000, 100)
    assert out.startswith("x" * 100)
    assert _COMPRESSION_MARKER_PREFIX in out
    assert out.endswith("\u27eb")
    assert _COMPRESSION_MARKER_RE.search(out) is not None


def test_elide_text_with_tail():
    out = elide_text("x" * 1000, 100, 50)
    assert out.startswith("x" * 100)
    assert out.endswith("x" * 50)
    assert _COMPRESSION_MARKER_PREFIX in out


def test_elide_text_short_passthrough():
    assert elide_text("short", 100) == "short"


def test_no_bare_truncation_marker_in_code():
    root = pathlib.Path(__file__).resolve().parents[2]
    bad: list[str] = []
    for p in sorted(root.rglob("*.py")):
        s = str(p)
        if "/tests/" in s or "\\tests\\" in s or s.endswith("test_compression_marker_elide.py"):
            continue
        try:
            lines = p.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "...[truncated]" in line:
                bad.append(f"{p.relative_to(root)}:{i}")
    assert not bad, f"bare truncation markers remain: {bad[:20]}"
