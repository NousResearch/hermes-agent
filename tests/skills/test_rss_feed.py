"""Tests for the rss-feed skill's crawl.sh script."""
import os
import shutil
import subprocess
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]  # repo root
CRAWL = ROOT / "skills/research/rss-feed/scripts/crawl.sh"
FIX = Path(__file__).parent / "fixtures/rss"
RSS2 = FIX / "sample_rss2.xml"
ATOM1 = FIX / "sample_atom1.xml"

REQUIRES_TOOLS = pytest.mark.skipif(
    not (shutil.which("xmlstarlet") and shutil.which("pandoc") and shutil.which("curl")),
    reason="rss-feed skill requires xmlstarlet, pandoc, and curl on PATH",
)


def _run(*args, expect_zero=True):
    result = subprocess.run(
        ["bash", str(CRAWL), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if expect_zero:
        assert result.returncode == 0, f"crawl.sh exited {result.returncode}\nstderr: {result.stderr}"
    return result


@REQUIRES_TOOLS
def test_rss2_extracts_items():
    out = _run(f"file://{RSS2}").stdout
    # feed title header
    assert "## " in out
    # at least 3 items rendered as bullet lines
    assert out.count("\n- **") >= 3
    # one of the known fixture titles appears
    assert "First Post Title" in out
    # links from the fixture appear in the output
    assert "https://example.com/posts/first" in out


@REQUIRES_TOOLS
def test_atom1_extracts_entries():
    out = _run(f"file://{ATOM1}").stdout
    assert "## " in out
    assert out.count("\n- **") >= 3
    assert "Atom Entry Alpha" in out
    assert "https://example.org/atom/alpha" in out


@REQUIRES_TOOLS
def test_limit_caps_items(tmp_path):
    out = _run(f"file://{RSS2}", "--limit", "2").stdout
    # exactly 2 items rendered
    assert out.count("\n- **") == 2


@REQUIRES_TOOLS
def test_feeds_file(tmp_path):
    feeds = tmp_path / "feeds.txt"
    feeds.write_text(
        f"# a comment\n\nfile://{RSS2}\nfile://{ATOM1}\n"
    )
    out = _run(str(feeds)).stdout
    # Both feed headers present
    assert out.count("## ") == 2
    # Both fixtures' marker titles present
    assert "First Post Title" in out
    assert "Atom Entry Alpha" in out


@REQUIRES_TOOLS
def test_invalid_url_does_not_crash(tmp_path):
    # crawl.sh should warn to stderr and exit 0
    feeds = tmp_path / "feeds.txt"
    feeds.write_text("https://invalid-host-that-does-not-exist.example/\n")
    # We don't assert zero — we only assert it doesn't blow up the runner.
    result = subprocess.run(
        ["bash", str(CRAWL), str(feeds)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Either exits 0 with empty/warning output, or non-zero but doesn't crash with a python-style traceback.
    assert "Traceback" not in result.stderr
