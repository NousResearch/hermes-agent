"""Tests for apps/docs/scripts/generate-skill-docs.py.

The generator turns every `skills/**/SKILL.md` into a docs page before
the `docs-site-checks` CI workflow runs `ascii-guard lint` on the result. If
a SKILL.md contains ASCII diagrams (box-drawing chars in a fenced code block)
without its own `<!-- ascii-guard-ignore -->` markers, the generator must
add them defensively — otherwise every PR touching `apps/docs/**` fails lint
on unrelated skill content.

Regression for issue #15305.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "apps" / "docs" / "scripts" / "generate-skill-docs.py"

@pytest.fixture(scope="module")
def gen_module():
    """Load generate-skill-docs.py as a module (hyphenated filename, not importable via normal import)."""
    spec = importlib.util.spec_from_file_location("generate_skill_docs", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_code_block_without_box_chars_is_not_wrapped(gen_module):
    """Plain bash/python code blocks should stay uncluttered."""
    body = "Intro.\n\n```bash\npip install foo\nfoo --run\n```\n\nOutro."
    result = gen_module.mdx_escape_body(body)
    assert "ascii-guard-ignore" not in result
    assert "pip install foo" in result

def test_code_block_with_box_chars_gets_wrapped(gen_module):
    """A code fence containing Unicode box-drawing chars must be wrapped in
    ascii-guard-ignore comments so the docs-site-checks lint can't fail on
    a skill's own diagram (issue #15305)."""
    body = (
        "Some text.\n\n"
        "```\n"
        "┌─────────┐\n"
        "│ diagram │\n"
        "└─────────┘\n"
        "```\n\n"
        "More text."
    )
    result = gen_module.mdx_escape_body(body)
    assert "<!-- ascii-guard-ignore -->" in result
    assert "<!-- ascii-guard-ignore-end -->" in result
    # The wrapper must sit OUTSIDE the fence, not inside.
    wrap_open = result.index("<!-- ascii-guard-ignore -->")
    fence_open = result.index("```\n┌")
    assert wrap_open < fence_open

def test_multiple_code_blocks_only_box_ones_wrapped(gen_module):
    """Mixed body: plain code stays plain, box code gets wrapped."""
    body = (
        "```bash\necho hi\n```\n\n"
        "```\n┌──┐\n│  │\n└──┘\n```\n\n"
        "```python\nprint('ok')\n```"
    )
    result = gen_module.mdx_escape_body(body)
    # exactly one wrap pair
    assert result.count("<!-- ascii-guard-ignore -->") == 1
    assert result.count("<!-- ascii-guard-ignore-end -->") == 1
    # plain blocks untouched
    assert "echo hi" in result
    assert "print('ok')" in result

def test_tilde_fenced_box_is_wrapped(gen_module):
    """The generator supports both ``` and ~~~ fences — both must be covered."""
    body = "~~~\n│ box │\n~~~"
    result = gen_module.mdx_escape_body(body)
    assert "<!-- ascii-guard-ignore -->" in result

def test_already_wrapped_source_double_wraps_harmlessly(gen_module):
    """If the SKILL.md already has ascii-guard-ignore markers, the generator's
    extra wrap is harmless (ascii-guard tolerates adjacent duplicate markers).
    The test just verifies we don't crash and the content survives."""
    body = (
        "<!-- ascii-guard-ignore -->\n"
        "```\n┌─┐\n└─┘\n```\n"
        "<!-- ascii-guard-ignore-end -->"
    )
    result = gen_module.mdx_escape_body(body)
    assert "┌─┐" in result
    # At least one marker pair survives
    assert "<!-- ascii-guard-ignore -->" in result
    assert "<!-- ascii-guard-ignore-end -->" in result


# ---------------------------------------------------------------------------
# Fumadocs layout contracts
# ---------------------------------------------------------------------------

def test_generator_roots_follow_the_relocated_layout(gen_module):
    """The app moved website/ → apps/docs/; docs content lives at
    apps/docs/content/{en,zh-Hans} and navigation at src/lib/navigation-data.ts.
    A stale root arithmetic silently writes pages into a dead tree."""
    assert gen_module.DOCS == gen_module.REPO / "apps" / "docs" / "content" / "en"
    assert gen_module.ZH_HANS_DOCS == gen_module.REPO / "apps" / "docs" / "content" / "zh-Hans"
    assert gen_module.NAVIGATION_DATA == gen_module.REPO / "apps" / "docs" / "src" / "lib" / "navigation-data.ts"

def test_generated_page_carries_an_invisible_markdown_comment(gen_module):
    """Per-skill pages are plain .md; `{/* ... */}` is MDX-only and would
    render as visible text. The generator must emit `<!-- ... -->`."""
    meta = {"source_kind": "bundled", "category": "apple", "sub": None, "slug": "apple-notes", "rel_path": "apple/apple-notes"}
    fm = {"name": "Apple Notes", "description": "Manage notes."}
    page = gen_module.render_skill_page(meta, fm, "Body text.")
    assert "<!-- This page is auto-generated from the skill's SKILL.md by apps/docs/scripts/generate-skill-docs.py." in page
    assert "{/*" not in page

def test_generated_page_body_is_fumadocs_normalized(gen_module):
    """A SKILL.md authored with Docusaurus-era syntax must not carry that
    syntax into the generated page, where it renders wrong."""
    meta = {"source_kind": "bundled", "category": "apple", "sub": None, "slug": "apple-notes", "rel_path": "apple/apple-notes"}
    fm = {"name": "Apple Notes", "description": "Manage notes."}
    body = ":::tip Pro tip\nUse it.\n:::\n\n## Setup {#setup}\nSteps.\n"
    page = gen_module.render_skill_page(meta, fm, body)
    assert ":::tip[Pro tip]" in page
    assert ":::tip Pro tip" not in page
    assert "## Setup [#setup]" in page
    assert "## Setup {#setup}" not in page

NAVIGATION_TEMPLATE = """import type {{ NavigationConfig }} from './navigation-types';

const navigation: NavigationConfig = {{
  docs: [
    'user-stories',
    // <skills-tree-generated>
    // stale generated content
    // </skills-tree-generated>
    {{
      type: 'category',
      label: 'Getting Started',
      items: ['getting-started/quickstart'],
    }},
  ],
}};

export default navigation;
"""


def test_write_sidebar_replaces_only_the_generated_block(gen_module, tmp_path):
    """The generated Skills subtree is fenced by explicit markers in
    navigation-data.ts; the hand-curated tree around it must survive."""
    nav = tmp_path / "navigation-data.ts"
    nav.write_text(NAVIGATION_TEMPLATE, encoding="utf-8")
    entries = [{
        "source_kind": "bundled",
        "category": "apple",
        "sub": None,
        "slug": "apple-notes",
    }]

    gen_module.write_sidebar([(meta, None) for meta in entries], navigation_path=nav)

    text = nav.read_text(encoding="utf-8")
    assert "// <skills-tree-generated>" in text and "// </skills-tree-generated>" in text
    assert "stale generated content" not in text
    assert "label: 'Skills'" in text
    assert "label: 'apple'" in text
    assert "'user-guide/skills/bundled/apple/apple-apple-notes'" in text
    # The hand-curated tree is untouched.
    assert "label: 'Getting Started'" in text
    assert "'getting-started/quickstart'" in text
    assert "export default navigation;" in text

def test_write_sidebar_requires_the_markers(gen_module, tmp_path):
    """A navigation-data.ts without markers must fail loudly, not silently
    skip the update (the sidebar would drift from the shipped skills)."""
    nav = tmp_path / "navigation-data.ts"
    nav.write_text("const navigation: NavigationConfig = { docs: [] };\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="skills-tree-generated"):
        gen_module.write_sidebar([], navigation_path=nav)