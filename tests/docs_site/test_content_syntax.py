"""Fumadocs syntax contracts for the docs content tree.

Docusaurus and Fumadocs disagree on two authored forms:

- titled admonitions: `:::tip Title` (Docusaurus) vs `:::tip[Title]` (Fumadocs)
- heading anchors: `## Heading {#id}` vs `## Heading [#id]`

The old forms render as literal text or the wrong fragment ID under the
Fumadocs MDX pipeline, so the migration converted the whole tree with
`apps/docs/scripts/normalize_content_syntax.py`. These tests hold both
directions: the converter changes exactly what it must (fence-aware, IDs
verbatim, untitled admonitions untouched, malformed tokens left alone) and
the tree carries no stale form.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NORMALIZER = REPO_ROOT / "apps" / "docs" / "scripts" / "normalize_content_syntax.py"
CONTENT = REPO_ROOT / "apps" / "docs" / "content"

# Hand-authored pages that use evaluated JSX (style={{...}}, className,
# useBaseUrl, next/link components) and were therefore renamed .md → .mdx.
MDX_RENAMES = [
    "en/getting-started/quickstart.mdx",
    "zh-Hans/getting-started/quickstart.mdx",
    "en/user-guide/configuration.mdx",
    "zh-Hans/user-guide/configuration.mdx",
    "en/user-guide/features/tool-gateway.mdx",
    "en/user-guide/messaging/webhooks.mdx",
    "zh-Hans/user-guide/messaging/webhooks.mdx",
    "en/user-guide/tui.mdx",
    "en/user-guide/sessions.mdx",
    "zh-Hans/user-guide/sessions.mdx",
    "en/user-guide/cli.mdx",
    "zh-Hans/user-guide/cli.mdx",
]


@pytest.fixture(scope="module")
def norm():
    spec = importlib.util.spec_from_file_location("normalize_content_syntax", NORMALIZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Titled admonitions: :::kind Title -> :::kind[Title]
# ---------------------------------------------------------------------------

def test_titled_admonition_becomes_bracket_form(norm):
    text = ":::tip Platform Support\nRuns on Linux.\n:::\n"
    result, flagged = norm.normalize_fumadocs_syntax(text)
    assert result == ":::tip[Platform Support]\nRuns on Linux.\n:::\n"
    assert flagged == []

def test_titled_admonition_title_with_markup_is_preserved_verbatim(norm):
    text = ":::warning `--dev` and `HERMES_TUI_DIR` are mutually exclusive\nBody.\n:::\n"
    result, _ = norm.normalize_fumadocs_syntax(text)
    assert ":::warning[`--dev` and `HERMES_TUI_DIR` are mutually exclusive]\n" in result

def test_titled_admonition_inside_code_fence_is_untouched(norm):
    text = "```\n:::tip Not an admonition\n```\n:::tip Real\n"
    result, _ = norm.normalize_fumadocs_syntax(text)
    assert ":::tip Not an admonition" in result  # fenced example stays
    assert ":::tip[Real]" in result

def test_untitled_admonition_and_closer_are_untouched(norm):
    text = ":::info\nBody.\n:::\n"
    result, _ = norm.normalize_fumadocs_syntax(text)
    assert result == text

def test_already_bracketed_title_is_untouched(norm):
    text = ":::warning[Token shown only once]\nBody.\n:::\n"
    result, _ = norm.normalize_fumadocs_syntax(text)
    assert result == text

def test_indented_titled_admonition_is_converted(norm):
    text = "  :::note Nested\nBody.\n  :::\n"
    result, _ = norm.normalize_fumadocs_syntax(text)
    assert "  :::note[Nested]" in result


# ---------------------------------------------------------------------------
# Heading anchors: {#id} -> [#id], ID verbatim
# ---------------------------------------------------------------------------

def test_heading_anchor_becomes_bracket_form(norm):
    text = "## Configuring Routes {#configuring-routes}\n"
    result, flagged = norm.normalize_fumadocs_syntax(text)
    assert result == "## Configuring Routes [#configuring-routes]\n"
    assert flagged == []

def test_cjk_heading_without_space_keeps_exact_id(norm):
    text = "## 插件路径（推荐）{#plugin-path-recommended}\n"
    result, _ = norm.normalize_fumadocs_syntax(text)
    assert result == "## 插件路径（推荐） [#plugin-path-recommended]\n"

def test_heading_anchor_inside_code_fence_is_untouched(norm):
    text = "```\n## Fenced {#fenced-id}\n```\n## Real {#real-id}\n"
    result, _ = norm.normalize_fumadocs_syntax(text)
    assert "## Fenced {#fenced-id}" in result
    assert "## Real [#real-id]" in result

def test_anchor_inside_inline_code_is_untouched(norm):
    text = "## Escaping `{#foo}` in prose\n"
    result, flagged = norm.normalize_fumadocs_syntax(text)
    assert result == text
    assert flagged == []

def test_malformed_anchor_is_left_alone_and_flagged(norm):
    text = "## Broken {#not a valid id}\n"
    result, flagged = norm.normalize_fumadocs_syntax(text)
    assert result == text
    assert flagged and "not a valid id" in flagged[0]

def test_empty_anchor_is_left_alone_and_flagged(norm):
    text = "## Broken {#}\n"
    result, flagged = norm.normalize_fumadocs_syntax(text)
    assert result == text
    assert flagged


# ---------------------------------------------------------------------------
# Generated-page comments: {/* ... */} (MDX) -> <!-- ... --> (plain .md)
# ---------------------------------------------------------------------------

def test_generated_comment_becomes_html_comment(norm):
    text = (
        "{/* This page is auto-generated from the skill's SKILL.md by "
        "website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}\n"
    )
    result, _ = norm.normalize_fumadocs_syntax(text)
    assert result == (
        "<!-- This page is auto-generated from the skill's SKILL.md by "
        "apps/docs/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. -->\n"
    )

def test_fenced_generated_comment_example_is_untouched(norm):
    text = "```\n{/* not a comment here */}\n```\n"
    result, _ = norm.normalize_fumadocs_syntax(text)
    assert result == text

def test_html_comments_become_mdx_comments_for_converted_pages(norm):
    """A page converted to .mdx keeps the root `no-tmp` lint's opt-out marker
    (the marker string survives), but in the MDX comment form — HTML comments
    are not valid in MDX. Prose sharing the line is preserved verbatim."""
    text = "instead of `/tmp` <!-- no-tmp: ok — explains why /tmp is avoided -->\n"
    result, _ = norm.normalize_fumadocs_syntax(text, target="mdx")
    assert result == "instead of `/tmp` {/* no-tmp: ok — explains why /tmp is avoided */}\n"


# ---------------------------------------------------------------------------
# End-state guards over the real content tree
# ---------------------------------------------------------------------------

def _content_files():
    assert CONTENT.is_dir(), "apps/docs/content is missing — relocation incomplete?"
    return sorted(p for p in CONTENT.rglob("*") if p.suffix in (".md", ".mdx") and p.is_file())


def test_no_docusaurus_syntax_remains_outside_code_fences(norm):
    """Every hand-converted page must carry the Fumadocs forms; a stale
    `:::kind Title` or heading `{#id}` renders wrong, not just unstyled."""
    offenders = []
    for path in _content_files():
        text = path.read_text(encoding="utf-8")
        target = "mdx" if path.suffix == ".mdx" else "md"
        result, flagged = norm.normalize_fumadocs_syntax(text, target=target)
        if result != text or flagged:
            offenders.append(path.relative_to(CONTENT).as_posix())
    assert not offenders, (
        f"{len(offenders)} content files still carry Docusaurus-authored syntax "
        f"(titled admonition, heading anchor, or malformed token): {offenders[:10]}"
    )

def test_no_mdx_comment_remains_in_plain_markdown(norm):
    """`{/* ... */}` renders as visible text in plain Markdown pages; only
    actual MDX (.mdx) pages may keep MDX comment syntax."""
    offenders = []
    for path in _content_files():
        if path.suffix != ".md":
            continue
        text = path.read_text(encoding="utf-8")
        for in_fence, line in norm.iter_fence_aware_lines(text):
            if not in_fence and "{/*" in line:
                offenders.append(f"{path.relative_to(CONTENT)}: {line.strip()[:60]}")
                break
    assert not offenders, f"plain .md pages with visible MDX comments: {offenders[:10]}"

def test_no_docusaurus_imports_remain_in_content():
    """@docusaurus/* and @site/* aliases do not resolve in the Fumadocs app."""
    offenders = []
    for path in _content_files():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("import ") and ("@docusaurus/" in line or "@site/" in line):
                offenders.append(f"{path.relative_to(CONTENT)}: {line.strip()[:80]}")
    assert not offenders, offenders

def test_jsx_pages_are_mdx_and_no_stale_md_twin_remains():
    for rel in MDX_RENAMES:
        assert (CONTENT / rel).is_file(), f"{rel} missing — JSX page must be .mdx"
        stale = CONTENT / (rel[: -len(".mdx")] + ".md")
        assert not stale.exists(), f"{stale.relative_to(CONTENT)} still exists alongside the .mdx"

def test_links_to_renamed_pages_use_the_mdx_extension(norm):
    """A relative link to a renamed page must follow the file on disk;
    a stale `.md` target 404s on GitHub and breaks route resolution."""
    offenders = []
    for path in _content_files():
        text = path.read_text(encoding="utf-8")
        for in_fence, line in norm.iter_fence_aware_lines(text):
            if in_fence:
                continue
            for target in norm.iter_markdown_link_targets(line):
                if "://" in target or target.startswith(("#", "mailto:", "/")):
                    continue  # absolute URL, fragment, or site route
                resolved = (path.parent / target.split("#")[0].split("?")[0]).resolve()
                if resolved.suffix == ".md" and with_suffix(resolved, ".mdx") in RENAMED_PATHS:
                    offenders.append(f"{path.relative_to(CONTENT)}: {target}")
    assert not offenders, offenders[:10]


RENAMED_PATHS = {(CONTENT / rel).resolve() for rel in MDX_RENAMES}


def with_suffix(path: Path, suffix: str) -> Path:
    return path.with_suffix(suffix)
