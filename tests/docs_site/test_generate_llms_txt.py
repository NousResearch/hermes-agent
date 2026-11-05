"""`llms.txt` is how an LLM learns what Hermes can do.

It is the index every model reads when pointed at our docs — including Hermes
itself, whose `hermes-agent` skill routes unknown-feature questions there.
`apps/docs/` is never packaged, so there is no shipped copy to fall back on.

The index used to be a hand-written list of page paths, and it rotted to 53%
coverage: Bot Mode, the desktop app, computer use, web search, and 22 messaging
platforms were all absent, which is why an agent asked how to make bots talk to
each other answered that it couldn't. These tests hold the two directions of
that contract — every page reachable, every link real — so the index tracks the
docs tree instead of someone's memory of it.

`llms-full.txt` concatenates corpus bodies that are
authored in Fumadocs MDX: `:::tip[Title]` callouts and relative `.md`/`.mdx`
file links are renderer instructions, not prose. The file is published for
agents that have no MDX runtime and no repository checkout, so the generator
normalizes both away — while leaving code fences, inline-code examples, the
`<!-- source: … -->` provenance comments, and the llms.txt index contract
untouched. The tests below generate the real artifact from a real (temporary)
docs tree to hold that.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "apps" / "docs" / "scripts" / "generate-llms-txt.py"

@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("generate_llms_txt", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@pytest.fixture(scope="module")
def index(gen) -> str:
    return gen.emit_llms_index()

def _linked(gen, index: str) -> set[str]:
    return set(re.findall(rf"\]\({re.escape(gen.SITE_BASE)}/([^)]+)\)", index))

def _pages_on_disk(gen) -> set[str]:
    """Walk the docs tree directly, duplicating only the two documented
    exclusions.

    Deliberately does not call `iter_docs()`: the index is built from that
    enumeration, so checking one against the other would pass even if the
    enumerator went blind to a whole tree — which is the failure being guarded.
    """
    pages = set()
    for path in (*gen.DOCS.rglob("*.md"), *gen.DOCS.rglob("*.mdx")):
        rel = path.relative_to(gen.DOCS).with_suffix("")
        slug = rel.parent.as_posix() if rel.name == "index" else rel.as_posix()
        # The docs landing page is the index's subject; per-skill pages are
        # summarized by the two catalog reference pages.
        if slug == "." or slug.startswith(("user-guide/skills/bundled", "user-guide/skills/optional")):
            continue
        pages.add(slug)
    return pages

def test_every_docs_page_is_indexed(gen, index):
    """The regression: a page the index omits is a feature the agent denies."""
    pages = _pages_on_disk(gen)
    assert len(pages) > 100, "docs root resolved wrong — the rest of this file proves nothing"

    missing = pages - _linked(gen, index)
    assert not missing, (
        f"{len(missing)} docs pages missing from llms.txt: {sorted(missing)} — "
        "they should have been absorbed into a section automatically"
    )

def test_the_enumerator_sees_the_whole_docs_tree(gen):
    """Everything downstream trusts `iter_docs()`, so pin it to the filesystem."""
    assert set(gen.iter_docs()) == _pages_on_disk(gen)

def test_every_indexed_page_exists(gen, index):
    """The other direction: a renamed page leaves the index pointing at a 404."""
    for slug in sorted(_linked(gen, index)):
        assert gen.doc_path(slug) is not None, (
            f"llms.txt links {slug}, which is not in the docs tree — "
            "drop the SECTIONS row and let the page be absorbed under its new path"
        )

def test_pages_are_listed_once(gen, index):
    """Curating a page must promote it, not duplicate it."""
    entries = re.findall(rf"^- \[.*?\]\({re.escape(gen.SITE_BASE)}/([^)]+)\)", index, re.MULTILINE)
    duplicated = {slug for slug in entries if entries.count(slug) > 1}
    assert not duplicated, f"listed more than once in llms.txt: {sorted(duplicated)}"

def test_curation_orders_pages_without_gatekeeping_them(gen):
    """SECTIONS decides what leads a section, never what the index contains."""
    curated = {slug for _section, items in gen.SECTIONS for slug, _t, _d in items}
    pages = set(gen.iter_docs())

    assert curated < pages, "every page is curated — absorption is no longer exercised"
    assert gen.section_for("user-guide/features/some-feature-shipped-tomorrow") in dict(gen.ABSORB)
    assert gen.section_for("a-tree-nobody-anticipated/page") == gen.MISC_SECTION

def test_section_landing_pages_resolve_to_their_directory(gen):
    """`messaging/index.md` is served at `/messaging`; `/messaging/index` 404s."""
    assert gen.slug_for(gen.DOCS / "user-guide" / "messaging" / "index.md") == "user-guide/messaging"
    assert "user-guide/messaging/index" not in _linked(gen, gen.emit_llms_index())

def test_mdx_pages_are_indexed_without_their_imports(gen):
    """MDX docs are real pages; their component imports are not prose."""
    mdx = [p for p in gen.DOCS.rglob("*.mdx") if gen.slug_for(p)]
    assert mdx, "no .mdx docs — this test no longer guards anything"
    assert {gen.slug_for(p) for p in mdx} <= set(gen.iter_docs())

    _meta, body = gen.read_frontmatter(mdx[0])
    assert not re.search(r"^import\s", body, re.MULTILINE)

def test_per_skill_catalog_pages_stay_out(gen):
    """~195 generated skill pages would bury the product docs in the index."""
    assert not [slug for slug in gen.iter_docs() if slug.startswith(gen.SKILL_CATALOG)]
    assert "reference/skills-catalog" in gen.iter_docs(), "the summary page must remain"


def test_h1_titles_lose_their_anchor_suffixes(gen):
    """Fumadocs moved heading anchors into the heading text (`[#id]`,
    replacing Docusaurus's `{#id}`); a title parsed from an H1 must not
    carry either form into llms.txt."""
    assert gen.strip_anchor_suffix("Configuring Hermes [#configuration]") == "Configuring Hermes"
    assert gen.strip_anchor_suffix("Configuring Hermes {#configuration}") == "Configuring Hermes"
    assert gen.strip_anchor_suffix("插件路径（推荐）[#plugin-path-recommended]") == "插件路径（推荐）"
    # Not an anchor: braces/brackets that are prose stay verbatim.
    assert gen.strip_anchor_suffix("Using `{#foo}` in prose") == "Using `{#foo}` in prose"
    assert gen.strip_anchor_suffix("Hermes [1]") == "Hermes [1]"

def test_fumadocs_anchors_do_not_leak_into_index_titles(gen, index):
    """Every indexed title must be clean prose — check the emitted entries
    rather than trusting per-page fixtures."""
    titles = re.findall(r"^- \[(.*?)\]\(", index, re.MULTILINE)
    assert titles
    dirty = [t for t in titles if "[#" in t or "{#" in t]
    assert not dirty, f"anchor residue in llms.txt titles: {dirty[:5]}"


# --- llms-full.txt must read standalone --------------------------------------
#
# The full-text emitter copies corpus bodies verbatim, so everything that only
# makes sense inside the Fumadocs MDX pipeline — `:::kind[Title]` container
# directives, relative `.md`/`.mdx` file links — ships to readers that have no
# MDX runtime and no repository checkout. The tests below point the generator's
# own machinery at a real temporary docs tree and assert on the emitted file,
# so every claim is proven by actual generation, not by unit-testing helpers.


@pytest.fixture()
def mini_docs(tmp_path, gen, monkeypatch):
    """A real on-disk mini corpus under tmp_path; the generator's DOCS and
    SECTIONS are repointed at it, so emit_llms_full() does real file I/O."""
    content = tmp_path / "content" / "en"
    (content / "user-guide").mkdir(parents=True)
    (content / "user-guide" / "cli.md").write_text(
        "---\n"
        "title: CLI\n"
        "description: The command line.\n"
        "---\n"
        "\n"
        "Intro paragraph.\n"
        "\n"
        ":::tip[Fastest path: Nous Portal]\n"
        "Use Nous Portal to get running.\n"
        ":::\n"
        "\n"
        ":::warning\n"
        "Secrets stay in `.env`.\n"
        ":::\n"
        "\n"
        "End paragraph.\n",
        encoding="utf-8",
    )
    (content / "user-guide" / "other.md").write_text(
        "---\n"
        "title: Other\n"
        "---\n"
        "\n"
        "Other page.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gen, "DOCS", content)
    monkeypatch.setattr(gen, "SECTIONS", [])
    return gen


def test_llms_full_callouts_become_readable_labels(mini_docs):
    """`:::tip[Title]` / `:::kind` are Fumadocs container directives — renderer
    syntax an agent reading llms-full.txt cannot interpret. They must become
    plain Markdown labels, with the callout's own content preserved verbatim
    and the directive markers gone."""
    full = mini_docs.emit_llms_full()

    assert "**Tip — Fastest path: Nous Portal**" in full
    assert "**Warning**" in full
    assert "Use Nous Portal to get running." in full
    assert "Secrets stay in `.env`." in full
    assert ":::" not in full, "renderer directive syntax leaked into llms-full.txt"


def test_llms_full_callouts_keep_paragraph_boundaries(mini_docs, tmp_path):
    """A callout that starts or ends directly against surrounding prose must
    not weld those paragraphs together once the markers are gone."""
    content = tmp_path / "content" / "en"
    (content / "user-guide" / "cli.md").write_text(
        "---\n"
        "title: CLI\n"
        "---\n"
        "\n"
        "Attachment size limit: **100 MB**.\n"
        ":::warning\n"
        "Rate limits apply.\n"
        ":::\n"
        "Next paragraph text.\n",
        encoding="utf-8",
    )
    full = mini_docs.emit_llms_full()

    assert "**Warning**" in full
    # label separated from the preceding paragraph, closing marker replaced by
    # a paragraph break instead of gluing "Rate limits apply." to the next text
    assert "Attachment size limit: **100 MB**.\n\n**Warning**" in full
    assert "Rate limits apply.\n\nNext paragraph text." in full


def test_llms_full_leaves_fenced_examples_alone(mini_docs, tmp_path):
    """A `:::tip` inside a fenced code block is a code example documenting the
    syntax, not a callout — the fence content must survive byte-identical."""
    content = tmp_path / "content" / "en"
    (content / "user-guide" / "cli.md").write_text(
        "---\n"
        "title: CLI\n"
        "---\n"
        "\n"
        "Syntax example:\n"
        "\n"
        "```md\n"
        ":::tip[Fenced example]\n"
        "stays literal\n"
        ":::\n"
        "```\n"
        "\n"
        "Real callout:\n"
        "\n"
        ":::tip[Real one]\n"
        "converted\n"
        ":::\n",
        encoding="utf-8",
    )
    full = mini_docs.emit_llms_full()

    assert "```md\n:::tip[Fenced example]\nstays literal\n:::\n```" in full
    assert ":::tip[Fenced example]" in full, "fence content was rewritten"
    assert "**Tip — Real one**" in full
    assert full.count(":::tip[") == 1


def test_llms_full_document_links_become_published_urls(mini_docs, tmp_path):
    """A relative `./page.md` link resolves against the site's content tree at
    build time; in the published file it points at a file the reader does not
    have. The generator must emit the published page URL instead, with the
    `#anchor` kept verbatim and `/index` landing pages mapped to their routes.
    Non-document targets (external URLs, fragments) stay untouched, and a link
    that resolves to no file is left as-is rather than turned into a guess."""
    content = tmp_path / "content" / "en"
    (content / "reference").mkdir()
    (content / "reference" / "env.md").write_text(
        "---\ntitle: Environment\n---\n\nEnv page.\n", encoding="utf-8")
    (content / "user-guide" / "sub").mkdir()
    (content / "user-guide" / "sub" / "page.mdx").write_text(
        "---\ntitle: Page\n---\n\nSub page.\n", encoding="utf-8")
    (content / "user-guide" / "cli.md").write_text(
        "---\n"
        "title: CLI\n"
        "---\n"
        "\n"
        "See [Other](./other.md) and [Env](../reference/env.md#usage) plus "
        "[Page](sub/page.mdx).\n"
        "\n"
        "External [upstream](https://example.com/foo.md), local [jump](#section), "
        "missing [gone](./missing.md), image ![badge](./other.md).\n",
        encoding="utf-8",
    )
    full = mini_docs.emit_llms_full()
    base = mini_docs.SITE_BASE

    assert f"[Other]({base}/user-guide/other)" in full
    assert f"[Env]({base}/reference/env#usage)" in full
    assert f"[Page]({base}/user-guide/sub/page)" in full
    # non-document targets are not the generator's business
    assert "[upstream](https://example.com/foo.md)" in full
    assert "[jump](#section)" in full
    assert "![badge](./other.md)" in full
    # an unresolvable link is left raw rather than turned into a fabricated URL
    assert "[gone](./missing.md)" in full


def test_llms_full_drops_heading_ids_and_mdx_comments_from_prose(mini_docs, tmp_path):
    """Heading `[#id]` suffixes and `{/* ... */}` comments are authoring syntax;
    the same text inside code stays verbatim."""
    content = tmp_path / "content" / "en"
    (content / "user-guide" / "cli.md").write_text(
        "---\n"
        "title: CLI\n"
        "---\n"
        "\n"
        "## Running the Webapp [#running-the-webapp]\n"
        "\n"
        "Uses `/tmp` {/* no-tmp: ok */} on Arch.\n"
        "\n"
        "```md\n"
        "## Example [#kept]\n"
        "{/* kept */}\n"
        "```\n"
        "\n"
        "Write `{/* inline */}` in MDX.\n",
        encoding="utf-8",
    )
    full = mini_docs.emit_llms_full()

    assert "## Running the Webapp\n" in full
    assert "[#running-the-webapp]" not in full
    assert "Uses `/tmp` on Arch." in full
    assert "no-tmp" not in full
    assert "```md\n## Example [#kept]\n{/* kept */}\n```" in full
    assert "Write `{/* inline */}` in MDX." in full


def test_llms_full_leaves_indented_code_blocks_alone(mini_docs, tmp_path):
    """Four-space indented code is code too: its callout and link examples stay
    verbatim, while the same lines as prose and as list continuations convert."""
    content = tmp_path / "content" / "en"
    (content / "user-guide" / "cli.md").write_text(
        "---\n"
        "title: CLI\n"
        "---\n"
        "\n"
        "Syntax example:\n"
        "\n"
        "    :::tip[Example]\n"
        "    [x](./installation.md)\n"
        "    :::\n"
        "\n"
        "- A list item that wraps onto\n"
        "    an indented [continuation](./other.md).\n"
        "\n"
        ":::tip[Real one]\n"
        "converted\n"
        ":::\n",
        encoding="utf-8",
    )
    full = mini_docs.emit_llms_full()

    assert "    :::tip[Example]\n    [x](./installation.md)\n    :::" in full
    assert "**Tip — Example**" not in full
    assert "**Tip — Real one**" in full
    assert "an indented [continuation](https://hermes-agent.nousresearch.com/docs/user-guide/other)" in full


def test_llms_full_leaves_inline_code_link_examples_alone(mini_docs, tmp_path):
    """Link syntax inside inline code — single-line spans, spans that cross a
    line break, and fenced examples — is documentation of the syntax, not a
    link. A blockquoted fence must also not flip the span tracking and mute
    normalization for the rest of the page."""
    content = tmp_path / "content" / "en"
    (content / "user-guide" / "cli.md").write_text(
        "---\n"
        "title: CLI\n"
        "---\n"
        "\n"
        "Write `[x](./other.md)` for a file link, or see [the real one](./other.md).\n"
        "\n"
        "```md\n"
        "[x](./other.md)\n"
        ":::tip[fenced]\n"
        "```\n"
        "\n"
        "> ```bash\n"
        "> hermes config set x '[y](./other.md)'\n"
        "> ```\n"
        "\n"
        "A span `that crosses\n"
        "[x](./other.md) stays literal` and ends.\n"
        "\n"
        ":::note\n"
        "After the blockquote fence, [links](./other.md) still normalize.\n"
        ":::\n",
        encoding="utf-8",
    )
    full = mini_docs.emit_llms_full()
    base = mini_docs.SITE_BASE

    assert "`[x](./other.md)`" in full, "single-line code span was rewritten"
    assert "[the real one](" + base + "/user-guide/other)" in full
    assert "```md\n[x](./other.md)\n:::tip[fenced]\n```" in full, "fenced example was rewritten"
    assert "> hermes config set x '[y](./other.md)'" in full
    assert "A span `that crosses\n[x](./other.md) stays literal`" in full, (
        "cross-line code span was rewritten"
    )
    # the blockquoted fence must not leave a phantom open span behind
    assert "**Note**" in full
    assert "[links](" + base + "/user-guide/other)" in full


def test_llms_full_inline_code_closers_need_exact_length(mini_docs, tmp_path):
    """A run of N backticks opens a span and only a run of exactly N closes
    it — a longer run is content, not a closer. The span must swallow the
    lines up to its real closer; prose after that closer is live again."""
    content = tmp_path / "content" / "en"
    (content / "user-guide" / "cli.md").write_text(
        "---\n"
        "title: CLI\n"
        "---\n"
        "\n"
        "A span `opens with one, and [first](./other.md) is inside it\n"
        "while a `` run of two is content, not the closer\n"
        "until [second](./other.md) ` closes the span.\n"
        "\n"
        "Then [third](./other.md) is live prose again.\n",
        encoding="utf-8",
    )
    full = mini_docs.emit_llms_full()
    base = mini_docs.SITE_BASE

    # the whole span — opener line, run-of-two line, closing line — is code
    assert "[first](./other.md)" in full, "opener-line span content was rewritten"
    assert "[second](./other.md)" in full, "span closed on a mismatched longer run"
    # prose after the real closer is live
    assert "[third](" + base + "/user-guide/other)" in full


def test_llms_full_escaped_backticks_do_not_open_spans(mini_docs, tmp_path):
    """An escaped backtick is a literal character: it opens no code span, so
    it must not mute the prose that follows — on its own line or the next.
    Inside an open span the backslash is content, so the backtick still
    closes the span (CommonMark)."""
    content = tmp_path / "content" / "en"
    (content / "user-guide" / "cli.md").write_text(
        "---\n"
        "title: CLI\n"
        "---\n"
        "\n"
        "An escaped \\` backtick is literal, so [here](./other.md) rewrites\n"
        "and [next](./other.md) on the following line still rewrites.\n"
        "\n"
        "Inside `a span \\` the backslash is content and [tail](./other.md) is live.\n",
        encoding="utf-8",
    )
    full = mini_docs.emit_llms_full()
    base = mini_docs.SITE_BASE

    assert "[here](" + base + "/user-guide/other)" in full
    assert "[next](" + base + "/user-guide/other)" in full, (
        "escaped backtick muted the next line's prose"
    )
    assert "[tail](" + base + "/user-guide/other)" in full


def test_llms_full_root_landing_page_maps_to_site_root(mini_docs, tmp_path):
    """The docs landing page — `index.mdx` at the content root, with no
    frontmatter slug — is the site root, `/docs/` rather than `/docs/index`
    (the docs app's own resolver collapses it the same way). A nested
    `index` still collapses to its directory route."""
    content = tmp_path / "content" / "en"
    (content / "index.mdx").write_text(
        "---\ntitle: Home\n---\n\nHome page.\n", encoding="utf-8")
    (content / "user-guide" / "landing").mkdir()
    (content / "user-guide" / "landing" / "index.md").write_text(
        "---\ntitle: Landing\n---\n\nLanding page.\n", encoding="utf-8")
    (content / "user-guide" / "cli.md").write_text(
        "---\n"
        "title: CLI\n"
        "---\n"
        "\n"
        "See [home](../index.mdx#top), [home-rooted](/docs/index.mdx) and "
        "[landing](./landing/index.md).\n",
        encoding="utf-8",
    )
    full = mini_docs.emit_llms_full()
    base = mini_docs.SITE_BASE

    assert f"[home]({base}/#top)" in full, "root index did not collapse to the site root"
    assert f"[home-rooted]({base}/)" in full, "rooted root index did not collapse"
    assert f"[landing]({base}/user-guide/landing)" in full


def test_llms_full_targets_escaping_the_docs_tree_stay_authored(mini_docs, tmp_path):
    """A rooted `/docs/../../x.md` target — or any target that resolves
    (via `..` or a symlink) outside the content tree — is not a page this
    tree publishes, even when the file exists; the link stays as authored
    rather than becoming a fabricated URL."""
    content = tmp_path / "content" / "en"
    outside = tmp_path / "outside.md"
    outside.write_text("---\ntitle: Outside\n---\n\nOutside page.\n", encoding="utf-8")
    try:
        (content / "trap.md").symlink_to(outside)
        (content / "alias.md").symlink_to(content / "user-guide" / "other.md")
    except (OSError, NotImplementedError):
        pass  # no symlink support — the `..` asserts below still hold
    (content / "user-guide" / "cli.md").write_text(
        "---\n"
        "title: CLI\n"
        "---\n"
        "\n"
        "Escapes: [rooted](/docs/../../outside.md), [relative](../../../outside.md), "
        "[trap](../trap.md), [rooted-trap](/docs/trap.md), [alias](../alias.md).\n",
        encoding="utf-8",
    )
    full = mini_docs.emit_llms_full()
    base = mini_docs.SITE_BASE

    assert "[rooted](/docs/../../outside.md)" in full, "rooted .. escape was rewritten"
    assert "[relative](../../../outside.md)" in full, "relative .. escape was rewritten"
    assert "[trap](../trap.md)" in full, "symlink escaping the docs tree was rewritten"
    assert "[rooted-trap](/docs/trap.md)" in full, "rooted symlink escape was rewritten"
    # a symlink that stays inside the tree is still a real page — routed by
    # its own linked path, as the docs app routes it
    if (content / "alias.md").exists():
        assert "[alias](" + base + "/alias)" in full


def test_llms_full_leaves_double_and_triple_code_span_links_alone(mini_docs, tmp_path):
    """Delimiter length is part of the span: a link inside a two- or
    three-backtick span is code bytes, not a link, and a prose link on the
    same line still rewrites."""
    content = tmp_path / "content" / "en"
    (content / "user-guide" / "cli.md").write_text(
        "---\n"
        "title: CLI\n"
        "---\n"
        "\n"
        "See ``[kept](./other.md)`` and ```[kept3](./other.md)``` but "
        "[real](./other.md) rewrites.\n",
        encoding="utf-8",
    )
    full = mini_docs.emit_llms_full()
    base = mini_docs.SITE_BASE

    assert "``[kept](./other.md)``" in full, "double-backtick span was rewritten"
    assert "```[kept3](./other.md)```" in full, "triple-backtick span was rewritten"
    assert "[real](" + base + "/user-guide/other)" in full


def test_llms_full_honors_slug_overrides_and_landing_routes(mini_docs, tmp_path):
    """Frontmatter `slug:` overrides the route (the docs app honors it when it
    renders these same links), the tree's landing page routes to the site
    root, and rooted `/docs/...` authored targets resolve like relative ones."""
    content = tmp_path / "content" / "en"
    (content / "user-guide" / "landing").mkdir(parents=True)
    (content / "user-guide" / "landing" / "index.md").write_text(
        "---\n"
        "title: Landing\n"
        "slug: /custom/route\n"
        "---\n"
        "\n"
        "Landing page.\n",
        encoding="utf-8",
    )
    (content / "index.mdx").write_text(
        "---\n"
        "title: Home\n"
        "slug: /\n"
        "---\n"
        "\n"
        "Home page.\n",
        encoding="utf-8",
    )
    (content / "user-guide" / "cli.md").write_text(
        "---\n"
        "title: CLI\n"
        "---\n"
        "\n"
        "See [landing](./landing/index.md), [home](../index.mdx#top), "
        "[rooted](/reference/env.md) and [prefixed](/docs/reference/env.md).\n",
        encoding="utf-8",
    )
    (content / "reference").mkdir(exist_ok=True)
    (content / "reference" / "env.md").write_text(
        "---\ntitle: Environment\n---\n\nEnv page.\n", encoding="utf-8")
    full = mini_docs.emit_llms_full()
    base = mini_docs.SITE_BASE

    assert f"[landing]({base}/custom/route)" in full
    assert f"[home]({base}/#top)" in full
    assert f"[rooted]({base}/reference/env)" in full
    assert f"[prefixed]({base}/reference/env)" in full


# --- the emitted artifact, against the real corpus ---------------------------
#
# The guards below scan the *emitted* file with their own fence/span tracking
# (not the generator's `_line_kinds`): checking the generator with its own
# classifier would pass even if the classifier went blind.


@pytest.fixture(scope="module")
def full(gen) -> str:
    return gen.emit_llms_full()


def _span_carry(line: str, carry: int) -> int:
    """Delimiter length of the code span still open at the end of `line` —
    the guard's own mirror of the rules the generator must implement: a
    span opens with a run of N backticks, closes on the next run of exactly
    N, and an escaped backtick delimits only outside spans."""
    open_len = carry
    i = 0
    n = len(line)
    while i < n:
        if line[i] != "`":
            i += 1
            continue
        if open_len == 0:
            backslashes = 0
            j = i - 1
            while j >= 0 and line[j] == "\\":
                backslashes += 1
                j -= 1
            if backslashes % 2:
                i += 1
                continue
        run = i
        while i < n and line[i] == "`":
            i += 1
        if open_len == 0:
            open_len = i - run
        elif i - run == open_len:
            open_len = 0
    return open_len


def _live_lines(text: str) -> list[str]:
    """Lines of `text` outside fenced code and inline-code span content.

    This is the guard's own tracker, not the generator's
    `_line_kinds`/`_code_spans`: checking the generator with its own
    classifier would pass even if the classifier went blind. Same rules,
    mirrored independently — fences close at >= the opening length, spans
    at exactly it, escapes delimit only outside spans. A span opened on an
    earlier line mutes the lines that continue and close it, whole.
    """
    live: list[str] = []
    in_fence = False
    fence_char = ""
    fence_len = 0
    carry = 0
    for line in text.split("\n"):
        m = re.match(r"^(?:\s|>\s?)*(`{3,}|~{3,})", line)
        if in_fence:
            if m and m.group(1)[0] == fence_char and len(m.group(1)) >= fence_len:
                in_fence = False
            continue
        if m:
            fence_char, fence_len = m.group(1)[0], len(m.group(1))
            in_fence = True
            continue
        if carry:
            # muted whole; the first run of exactly `carry` backticks closes
            if any(len(run) == carry for run in re.findall(r"`+", line)):
                carry = 0
            continue
        carry = _span_carry(line, 0)
        live.append(line)
    return live


def test_llms_full_real_corpus_has_no_renderer_callout_syntax(gen, full):
    """No Fumadocs container directive may survive into the published
    llms-full.txt — not a single `:::` line."""
    leaked = [ln for ln in _live_lines(full) if ln.lstrip().startswith(":::")]
    assert not leaked, f"{len(leaked)} directive lines leaked: {leaked[:3]}"


def test_llms_full_real_corpus_document_links_are_published_urls(gen, full):
    """Every document link must be a published site URL (or an external link —
    those are not ours to rewrite), and every published URL must route back
    to a file that exists in the docs tree."""
    raw: list[str] = []
    dead: list[str] = []
    for line in _live_lines(full):
        for m in re.finditer(r"(?<!\!)\[[^\]]*\]\(([^)]+)\)", line):
            href = m.group(1)
            if re.match(r"^[a-z][a-z\d+.-]*:", href, re.IGNORECASE):
                continue  # external — not a document link
            if re.search(r"\.mdx?(?:[?#]|$)", href, re.IGNORECASE):
                raw.append(href)  # a .md/.mdx target that was not rewritten
            if href.startswith(f"{gen.SITE_BASE}/"):
                route = href[len(gen.SITE_BASE) + 1:].split("#")[0].split("?")[0].rstrip("/")
                if gen.doc_path(route) is None:
                    dead.append(href)
    assert not raw, f"unrewritten document links: {raw[:5]}"
    assert not dead, f"rewritten links pointing at no docs page: {dead[:5]}"


def test_llms_full_provenance_comments_stay_accurate(gen, full):
    """The `<!-- source: … -->` comments are the provenance trail for editors;
    each must name a file that actually exists, under the current (apps/docs)
    path — not the retired `website/` one."""
    comments = re.findall(r"<!-- source: (apps/docs/content/en/[^>]+) -->", full)
    assert comments, "source comments vanished from llms-full.txt"
    assert len(comments) == len(set(comments)), "a file is emitted twice"
    stale = [c for c in comments if c.startswith("website/")]
    assert not stale, f"provenance names the retired path: {stale[:3]}"
    missing = [c for c in comments if not (gen.DOCS / c[len("apps/docs/content/en/"):]).is_file()]
    assert not missing, f"provenance points at files that do not exist: {missing[:3]}"
