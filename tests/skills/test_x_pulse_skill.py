"""Tests for optional-skills/social-media/x-pulse/scripts/build_digest.py."""

import re
import sys
from pathlib import Path

import pytest

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "social-media"
    / "x-pulse"
)
sys.path.insert(0, str(SKILL_DIR / "scripts"))

import build_digest as bd  # noqa: E402


# ── dedupe_citations ────────────────────────────────────────────────────────

class TestDedupeCitations:
    def test_dedupes_by_normalized_url(self):
        cites = [
            {"url": "https://x.com/a/status/1", "title": "one"},
            {"url": "https://x.com/a/status/1/", "title": "dup trailing slash"},
            {"url": "HTTPS://X.com/a/status/1#frag", "title": "dup case+frag"},
            {"url": "https://x.com/b/status/2", "title": "two"},
        ]
        out = bd.dedupe_citations(cites)
        assert [c["url"] for c in out] == [
            "https://x.com/a/status/1",
            "https://x.com/b/status/2",
        ]
        # First-seen title/order preserved.
        assert out[0]["title"] == "one"

    def test_skips_empty_and_non_dict(self):
        cites = [{"url": ""}, "not-a-dict", None, {"title": "no url"}]
        assert bd.dedupe_citations(cites) == []

    def test_none_input(self):
        assert bd.dedupe_citations(None) == []


# ── build_digest ────────────────────────────────────────────────────────────

class TestBuildDigest:
    def _ok(self, answer, citations, degraded=False):
        return {"success": True, "answer": answer, "citations": citations,
                "degraded": degraded}

    def test_multi_topic_digest_and_global_sources_dedup(self):
        entries = [
            {"topic": "AI", "result": self._ok(
                "Open models are hot.",
                [{"url": "https://x.com/a/status/1", "title": "post A"}])},
            {"topic": "Space", "result": self._ok(
                "Launch reactions.",
                [{"url": "https://x.com/a/status/1", "title": "same post"},
                 {"url": "https://x.com/b/status/2", "title": "post B"}])},
        ]
        out = bd.build_digest(entries, title="X Pulse", date_str="2026-07-10")
        assert out.startswith("# X Pulse — 2026-07-10")
        assert "### AI" in out and "### Space" in out
        assert "Open models are hot." in out
        # Global Sources dedups the shared post → 2 unique sources, not 3.
        assert "**Sources (2)**" in out

    def test_degraded_result_is_flagged(self):
        entries = [{"topic": "Rumor", "result": self._ok(
            "Might be true.", [], degraded=True)}]
        out = bd.build_digest(entries)
        assert "⚠️ Unsourced" in out

    def test_failed_result_renders_no_result(self):
        entries = [{"topic": "Broken", "result": {"success": False, "error": "boom"}}]
        out = bd.build_digest(entries)
        assert "### Broken" in out
        assert "_(no result: boom)_" in out
        # No Sources section when there are no citations anywhere.
        assert "**Sources" not in out

    def test_empty_entries(self):
        out = bd.build_digest([])
        assert "_No topics provided._" in out

    def test_untitled_topic_fallback(self):
        entries = [{"result": self._ok("x", [])}]
        assert "### (untitled)" in bd.build_digest(entries)


# ── main() CLI ──────────────────────────────────────────────────────────────

class TestMain:
    def test_stdin_json_to_stdout(self, monkeypatch, capsys):
        import io
        payload = '[{"topic":"AI","result":{"success":true,"answer":"hi","citations":[],"degraded":false}}]'
        monkeypatch.setattr("sys.stdin", io.StringIO(payload))
        rc = bd.main(["--title", "X Pulse"])
        assert rc == 0
        assert "### AI" in capsys.readouterr().out

    def test_invalid_json_returns_2(self, monkeypatch):
        import io
        monkeypatch.setattr("sys.stdin", io.StringIO("{not json"))
        assert bd.main([]) == 2

    def test_non_list_returns_2(self, monkeypatch):
        import io
        monkeypatch.setattr("sys.stdin", io.StringIO('{"topic":"x"}'))
        assert bd.main([]) == 2


# ── SKILL.md frontmatter standards ──────────────────────────────────────────

def test_description_meets_hardline_standard():
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    m = re.search(r'^description:\s*"?(.*?)"?\s*$', text, re.MULTILINE)
    assert m, "SKILL.md must declare a description"
    desc = m.group(1)
    assert len(desc) <= 60, f"description is {len(desc)} chars (max 60): {desc!r}"
    assert desc.endswith("."), "description must end with a period"


def test_referenced_sections_present():
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    for section in ("## When to Use", "## Prerequisites", "## How to Run",
                    "## Procedure", "## Pitfalls", "## Verification"):
        assert section in text, f"missing section: {section}"
