"""Regression test for scripts/kensei_review_daily.py.

Verifies the no_agent aggregator with a complete temp fixture:
- finds source files under temp HERMES_HOME / repo / wiki roots
- writes a valid dark-mode HTML review with real content
- prints a MEDIA: tag pointing at the written file
- source artifacts are byte-identical before/after (read-only)
- all-empty fixture prints [SILENT] and writes no output file
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
from html.parser import HTMLParser
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "kensei_review_daily.py"


class _TagChecker(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack: list[str] = []
        self.ok = True
        self.li_count = 0

    def handle_starttag(self, tag, attrs):
        if tag not in ("meta", "br", "img", "hr", "link"):
            self.stack.append(tag)
        if tag == "li":
            self.li_count += 1

    def handle_endtag(self, tag):
        if tag in ("meta", "br", "img", "hr", "link"):
            return
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        else:
            self.ok = False


def _sha_files(*paths: Path) -> dict[str, str]:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}


def _build_fixture(home: Path, repo: Path, wiki: Path) -> dict[str, Path]:
    """Seed the three upstream artifacts + mashups log under temp roots."""
    digest_dir = home / "runbooks" / "research-digest"
    synth_dir = home / "runbooks" / "research-paper-synthesis"
    radar_dir = repo / "runbooks" / "github-radar"
    mashups_file = wiki / "_meta" / "paper-mashups.md"
    for d in (digest_dir, synth_dir, radar_dir):
        d.mkdir(parents=True, exist_ok=True)
    mashups_file.parent.mkdir(parents=True, exist_ok=True)

    digest_path = digest_dir / "research-brief-2026-07-27.html"
    digest_path.write_text(
        "<h2>Topic One</h2><p>First digest body line long enough</p>\n"
        "<h2>Topic Two</h2><p>Second digest body line long enough</p>\n",
        encoding="utf-8",
    )
    synth_path = synth_dir / "research-paper-synthesis-2026-07-27.html"
    synth_path.write_text(
        "<h2>Full Treatment Papers</h2><p>Paper A — notable finding here</p>\n"
        "<h2>Mashup Ideas</h2><p>Combine X and Y for Z</p>\n",
        encoding="utf-8",
    )
    # radar: dated subdir with the txt file (>200 bytes)
    radar_sub = radar_dir / "2026-07-27"
    radar_sub.mkdir()
    radar_path = radar_sub / "github-radar-repos.txt"
    radar_path.write_text(
        "[REPO] alpha-repo | stars:100 | lang:python | classification:ADOPT "
        "| score:8.5 | url:https://example.com/alpha\n"
        "Description: A notable repo with a long enough description text\n"
        "Why it matters: It demonstrates the pattern well for our stack\n"
        "---\n",
        encoding="utf-8",
    )
    mashups_file.write_text(
        "# mashups log\n2026-07-27 first mashup entry\n2026-07-27 second entry\n",
        encoding="utf-8",
    )
    return {
        "digest": digest_path, "synth": synth_path,
        "radar": radar_path, "mashups": mashups_file,
    }


def _run_script(home: Path, repo: Path, wiki: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["HERMES_HOME"] = str(home)
    env["KENSEI_REPO_ROOT"] = str(repo)
    env["KENSEI_WIKI_ROOT"] = str(wiki)
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
    )


def test_review_daily_aggregates_temp_fixture_and_preserves_sources(tmp_path: Path) -> None:
    home = tmp_path / "hermes"
    repo = tmp_path / "repo"
    wiki = tmp_path / "wiki"
    home.mkdir(); repo.mkdir(); wiki.mkdir()
    srcs = _build_fixture(home, repo, wiki)
    before = _sha_files(*srcs.values())

    r = _run_script(home, repo, wiki)
    assert r.returncode == 0, r.stderr

    # MEDIA: tag pointing at an html file under temp OUT_DIR
    media_lines = [l for l in r.stdout.splitlines() if l.startswith("MEDIA:")]
    assert len(media_lines) == 1, f"expected one MEDIA line: {r.stdout!r}"
    media_path = Path(media_lines[0].split("MEDIA:", 1)[1].strip())
    assert media_path.exists(), f"media path missing: {media_path}"
    assert media_path.is_relative_to(home), f"media outside temp home: {media_path}"
    assert media_path.name == "review.html"

    # stdout summary mentions aggregated signals
    assert "signals aggregated" in r.stdout, r.stdout

    # HTML is well-formed and contains real content
    html_text = media_path.read_text()
    checker = _TagChecker()
    checker.feed(html_text)
    assert checker.ok, f"unbalanced HTML tags: {checker.stack}"
    assert "#fbbf24" in html_text, "accent colour missing"
    assert checker.li_count > 0, "no content items in HTML"
    assert "alpha-repo" in html_text, "radar repo missing from HTML"

    # Source artifacts unchanged
    after = _sha_files(*srcs.values())
    assert after == before, "source artifacts mutated by aggregator"


def test_review_daily_silent_when_all_sources_empty(tmp_path: Path) -> None:
    home = tmp_path / "hermes"
    repo = tmp_path / "repo"
    wiki = tmp_path / "wiki"
    home.mkdir(); repo.mkdir(); wiki.mkdir()
    # Create empty source dirs so the script finds nothing
    (home / "runbooks" / "research-digest").mkdir(parents=True)
    (home / "runbooks" / "research-paper-synthesis").mkdir(parents=True)
    (repo / "runbooks" / "github-radar").mkdir(parents=True)
    (wiki / "_meta").mkdir(parents=True)

    r = _run_script(home, repo, wiki)
    assert r.returncode == 0, r.stderr
    assert "[SILENT]" in r.stdout, f"expected [SILENT], got: {r.stdout!r}"
    # No output file written
    out_dir = home / "runbooks" / "kensei-review-daily"
    assert not out_dir.exists() or not any(out_dir.rglob("review.html")), \
        "output file written despite silent"
