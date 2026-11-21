"""Tests for skills/productivity/md-to-html."""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("markdown")
pytest.importorskip("pygments")

REPO = Path(__file__).resolve().parents[2]
SKILL = REPO / "skills" / "productivity" / "md-to-html"
SKILL_MD = SKILL / "SKILL.md"
SCRIPT = SKILL / "scripts" / "md2html.py"
TEMPLATE_CSS = SKILL / "templates" / "style.css"

SAMPLE_MD = """\
# Smoke Test Doc

A **no LLM involved** conversion check.

## Status table

| Status | Meaning |
|--------|---------|
| **green** | good |
| **red** | bad |

## Code

```python
def hello(name: str) -> str:
    return f"hello {name}"
```

> Deterministic output.

Done.
"""


def _frontmatter():
    text = SKILL_MD.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


def _load_script():
    spec = importlib.util.spec_from_file_location("md2html", SCRIPT)
    assert spec and spec.loader, f"cannot import {SCRIPT}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def md2html():
    return _load_script()


@pytest.fixture()
def sample(tmp_path):
    src = tmp_path / "sample.md"
    src.write_text(SAMPLE_MD, encoding="utf-8")
    return src


class TestFrontmatter:
    def test_name_matches_directory(self):
        assert _frontmatter()["name"] == "md-to-html"

    def test_description_length_and_period(self):
        desc = _frontmatter()["description"]
        assert len(desc) <= 60
        assert desc.rstrip().endswith(".")

    def test_author_credits_the_human(self):
        author = _frontmatter()["author"]
        assert "Hermes Agent" not in author or author.index("Anderson") < author.index(
            "Hermes Agent"
        )

    def test_license_and_platforms(self):
        fm = _frontmatter()
        assert fm["license"] == "MIT"
        assert set(fm["platforms"]) == {"linux", "macos", "windows"}

    def test_required_sections_present(self):
        body = SKILL_MD.read_text(encoding="utf-8")
        for section in (
            "## When to Use",
            "## Prerequisites",
            "## How to Run",
            "## Procedure",
            "## Pitfalls",
            "## Verification",
        ):
            assert section in body, f"SKILL.md missing {section}"

    def test_no_machine_local_paths(self):
        body = SKILL_MD.read_text(encoding="utf-8")
        assert not re.search(r"/home/(?!runner\b)[a-z0-9_-]+/", body)


class TestScriptShipped:
    def test_script_and_template_exist(self):
        assert SCRIPT.is_file()
        assert TEMPLATE_CSS.is_file()

    def test_template_is_resolved_relative_to_script(self):
        source = SCRIPT.read_text(encoding="utf-8")
        assert "templates" in source and "__file__" in source


class TestRender:
    def test_title_from_first_h1(self, md2html, sample):
        text = sample.read_text(encoding="utf-8")
        assert md2html.extract_title(text, "fallback") == "Smoke Test Doc"

    def test_title_falls_back_when_no_h1(self, md2html):
        assert md2html.extract_title("just text\n", "My Notes") == "My Notes"

    def test_render_produces_selfcontained_html(self, md2html, sample):
        text = sample.read_text(encoding="utf-8")
        css = TEMPLATE_CSS.read_text(encoding="utf-8")
        html = md2html.render(text, "Smoke Test Doc", css, "", "2026-09-30")
        assert html.startswith("<!DOCTYPE html>")
        assert "<table>" in html and "<th>" in html
        assert "codehilite" in html  # pygments-highlighted fenced code
        assert "<blockquote>" in html
        assert "st-ok" in html and "st-bad" in html  # status-word spans
        assert "--mark-ok" in html  # template CSS inlined
        assert "**green**" not in html  # no unrendered markdown

    def test_status_words_are_case_insensitive(self, md2html):
        out = md2html.status_postpass("<strong>RED</strong> and <strong>Gray</strong>")
        assert 'class="st-bad"' in out and 'class="st-neutral"' in out

    @pytest.mark.parametrize(
        ("title", "escaped"),
        [
            ("<script>alert(1)</script>", "&lt;script&gt;alert(1)&lt;/script&gt;"),
            (
                "</title></head><script>alert(1)</script>",
                "&lt;/title&gt;&lt;/head&gt;&lt;script&gt;alert(1)&lt;/script&gt;",
            ),
            ("<img src=x onerror=alert(1)>", "&lt;img src=x onerror=alert(1)&gt;"),
            ("<b>Bold</b> & Title", "&lt;b&gt;Bold&lt;/b&gt; &amp; Title"),
        ],
    )
    @pytest.mark.parametrize("from_heading", [True, False])
    def test_title_is_escaped(self, md2html, title, escaped, from_heading):
        text = f"# {title}\n" if from_heading else "just text\n"
        extracted = md2html.extract_title(text, title)
        page = md2html.render(text, extracted, "", "", "2026-09-30")
        assert f"<title>{escaped}</title>" in page
        header = page.split("<header", 1)[1].split("</header>", 1)[0]
        assert f"<h1>{escaped}</h1>" in header
        assert "<script>" not in page.split("</head>", 1)[0]
        assert "<script>" not in header

    def test_raw_html_body_remains_supported(self, md2html):
        page = md2html.render("<b>Body</b>", "Title", "", "", "2026-09-30")
        assert "<b>Body</b>" in page


class TestCli:
    def test_cli_writes_sibling_html(self, tmp_path, sample):
        out = tmp_path / "sample.html"
        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(sample)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        assert out.is_file()
        assert "Smoke Test Doc" in out.read_text(encoding="utf-8")

    def test_cli_out_and_stdout(self, tmp_path, sample):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(sample), "--stdout"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.startswith("<!DOCTYPE html>")
        # no sibling file written in stdout mode
        assert not (tmp_path / "sample.html").exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))