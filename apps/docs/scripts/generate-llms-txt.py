#!/usr/bin/env python3
"""Generate llms.txt and llms-full.txt for the Hermes docs site.

Outputs:
  apps/docs/public/llms.txt        — index of the docs, one link per page, grouped by
                                     section. Conforms to https://llmstxt.org.
  apps/docs/public/llms-full.txt   — every doc under `apps/docs/content/en/` concatenated,
                                     with `# <title>` headings and `<!-- source: … -->`
                                     comments separating files. Bodies are normalized
                                     for standalone readers (see `_convert_callouts`
                                     and `_rewrite_document_links`): the corpus is
                                     authored in Fumadocs MDX, but this file is read
                                     by agents with no MDX runtime and no repository
                                     checkout.

Both are driven by `iter_docs()`, which walks the docs tree. `SECTIONS` below
curates *order and grouping*, never membership: a page nobody curated still
gets indexed, under the section its path belongs to. That distinction is the
reason this file was rewritten — when the section list also decided membership,
it silently drifted to 53% coverage, and Bot Mode, the desktop app, computer
use, web search, and 22 messaging platforms were absent from the index every
LLM reads to learn what Hermes does.

Both publish at:
  https://hermes-agent.nousresearch.com/docs/llms.txt
  https://hermes-agent.nousresearch.com/docs/llms-full.txt

The `/docs/` prefix is not a mistake — the Next app is exported with
`basePath: '/docs'`, so everything in `apps/docs/public/` is served under the
/docs origin path. Clients and IDE plugins that probe the classic
`/llms.txt` root will miss these. Document the canonical URLs in the docs
index and in the repo README.

Called from `apps/docs/scripts/prebuild.mjs` on every `npm run start` /
`npm run build` so the output stays in sync with the docs tree.
"""

from __future__ import annotations

import posixpath
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DOCS_APP = SCRIPT_DIR.parent
DOCS = DOCS_APP / "content" / "en"
STATIC = DOCS_APP / "public"

SITE_BASE = "https://hermes-agent.nousresearch.com/docs"

# The product story: which pages lead, and in what order. Everything not named
# here is still indexed — ABSORB decides where it lands — so this list is safe
# to leave alone as the docs grow, and worth editing only to promote a page.
# Each entry: (docs-relative path without extension, display title, optional
# short desc). `None` desc → pulled from frontmatter `description:` field.
SECTIONS: list[tuple[str, list[tuple[str, str, str | None]]]] = [
    ("Getting Started", [
        ("getting-started/installation", "Installation", None),
        ("getting-started/quickstart", "Quickstart", None),
        ("getting-started/learning-path", "Learning Path", None),
        ("getting-started/updating", "Updating", None),
        ("getting-started/nix-setup", "Nix Setup", None),
    ]),
    ("Using Hermes", [
        ("user-guide/cli", "CLI", None),
        ("user-guide/tui", "TUI (Ink terminal UI)", None),
        ("user-guide/configuration", "Configuration", None),
        ("user-guide/configuring-models", "Configuring Models", None),
        ("user-guide/sessions", "Sessions", None),
        ("user-guide/profiles", "Profiles", None),
        ("user-guide/git-worktrees", "Git Worktrees", None),
        ("user-guide/docker", "Docker Backend", None),
        ("user-guide/security", "Security", None),
        ("user-guide/checkpoints-and-rollback", "Checkpoints & Rollback", None),
    ]),
    ("Core Features", [
        ("user-guide/features/overview", "Features Overview", None),
        ("user-guide/features/tools", "Tools", None),
        ("user-guide/features/skills", "Skills System", None),
        ("user-guide/features/curator", "Curator", None),
        ("user-guide/features/memory", "Memory", None),
        ("user-guide/features/memory-providers", "Memory Providers", None),
        ("user-guide/features/context-files", "Context Files", None),
        ("user-guide/features/context-references", "Context References", None),
        ("user-guide/features/personality", "Personality & SOUL.md", None),
        ("user-guide/features/plugins", "Plugins", None),
        ("user-guide/features/built-in-plugins", "Built-in Plugins", None),
    ]),
    ("Automation", [
        ("user-guide/features/cron", "Cron Jobs", None),
        ("user-guide/features/delegation", "Delegation", None),
        ("user-guide/features/kanban", "Kanban Multi-Agent", None),
        ("user-guide/features/kanban-tutorial", "Kanban Tutorial", None),
        ("user-guide/features/goals", "Persistent Goals", None),
        ("user-guide/features/code-execution", "Code Execution", None),
        ("user-guide/features/hooks", "Hooks", None),
        ("user-guide/features/batch-processing", "Batch Processing", None),
    ]),
    ("Media & Web", [
        ("user-guide/features/voice-mode", "Voice Mode", None),
        ("user-guide/features/browser", "Browser", None),
        ("user-guide/features/vision", "Vision", None),
        ("user-guide/features/image-generation", "Image Generation", None),
        ("user-guide/features/tts", "Text-to-Speech", None),
    ]),
    ("Messaging Platforms", [
        ("user-guide/messaging", "Overview", None),
        ("user-guide/messaging/telegram", "Telegram", None),
        ("user-guide/messaging/discord", "Discord", None),
        ("user-guide/messaging/slack", "Slack", None),
        ("user-guide/messaging/whatsapp", "WhatsApp", None),
        ("user-guide/messaging/signal", "Signal", None),
        ("user-guide/messaging/email", "Email", None),
        ("user-guide/messaging/sms", "SMS", None),
        ("user-guide/messaging/matrix", "Matrix", None),
        ("user-guide/messaging/mattermost", "Mattermost", None),
        ("user-guide/messaging/homeassistant", "Home Assistant", None),
        ("user-guide/messaging/webhooks", "Webhooks", None),
    ]),
    ("Integrations", [
        ("integrations", "Integrations Overview", None),
        ("integrations/providers", "Providers", None),
        ("user-guide/features/mcp", "MCP (Model Context Protocol)", None),
        ("user-guide/features/acp", "ACP (Agent Context Protocol)", None),
        ("user-guide/features/api-server", "API Server", None),
        ("user-guide/features/honcho", "Honcho Memory", None),
        ("user-guide/features/provider-routing", "Provider Routing", None),
        ("user-guide/features/fallback-providers", "Fallback Providers", None),
        ("user-guide/features/credential-pools", "Credential Pools", None),
    ]),
    ("Guides & Tutorials", [
        ("guides/tips", "Tips & Best Practices", None),
        ("guides/local-llm-on-mac", "Local LLMs on Mac", None),
        ("guides/daily-briefing-bot", "Daily Briefing Bot", None),
        ("guides/team-telegram-assistant", "Team Telegram Assistant", None),
        ("guides/python-library", "Use Hermes as a Python Library", None),
        ("guides/use-mcp-with-hermes", "Use MCP with Hermes", None),
        ("guides/use-voice-mode-with-hermes", "Use Voice Mode with Hermes", None),
        ("guides/use-soul-with-hermes", "Use SOUL.md with Hermes", None),
        ("guides/automate-with-cron", "Automate with Cron", None),
        ("guides/work-with-skills", "Work with Skills", None),
        ("guides/delegation-patterns", "Delegation Patterns", None),
        ("guides/github-pr-review-agent", "GitHub PR Review Agent", None),
    ]),
    ("Developer Guide", [
        ("developer-guide/contributing", "Contributing", None),
        ("developer-guide/architecture", "Architecture", None),
        ("developer-guide/agent-loop", "Agent Loop", None),
        ("developer-guide/prompt-assembly", "Prompt Assembly", None),
        ("developer-guide/context-compression-and-caching", "Context Compression & Caching", None),
        ("developer-guide/gateway-internals", "Gateway Internals", None),
        ("developer-guide/session-storage", "Session Storage", None),
        ("developer-guide/provider-runtime", "Provider Runtime", None),
        ("developer-guide/adding-tools", "Adding Tools", None),
        ("developer-guide/adding-providers", "Adding Providers", None),
        ("developer-guide/adding-platform-adapters", "Adding Platform Adapters", None),
        ("developer-guide/creating-skills", "Creating Skills", None),
        ("developer-guide/extending-the-cli", "Extending the CLI", None),
    ]),
    ("Reference", [
        ("reference/cli-commands", "CLI Commands", None),
        ("reference/slash-commands", "Slash Commands", None),
        ("reference/profile-commands", "Profile Commands", None),
        ("reference/environment-variables", "Environment Variables", None),
        ("reference/tools-reference", "Tools Reference", None),
        ("reference/toolsets-reference", "Toolsets Reference", None),
        ("reference/mcp-config-reference", "MCP Config Reference", None),
        ("reference/model-catalog", "Model Catalog", None),
        ("reference/skills-catalog", "Bundled Skills Catalog", "Table of all ~90 skills bundled with Hermes"),
        ("reference/optional-skills-catalog", "Optional Skills Catalog", "Table of ~60 additional installable skills"),
        ("reference/faq", "FAQ & Troubleshooting", None),
    ]),
]


DOC_EXTS = (".md", ".mdx")

# Per-skill pages are generated from the skill tree and summarized by the two
# catalog reference pages. Listing ~195 of them would bury the product docs in
# the index and add ~1.4 MB of duplicative material to llms-full.txt.
SKILL_CATALOG = ("user-guide/skills/bundled", "user-guide/skills/optional")

# Where a page nobody curated goes. First match wins, so a narrower prefix must
# precede the tree containing it. Anything matching nothing lands in
# MISC_SECTION — no path can drop a page out of the index.
ABSORB: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Getting Started", ("getting-started",)),
    ("Messaging Platforms", ("user-guide/messaging",)),
    ("Core Features", ("user-guide/features",)),
    ("Using Hermes", ("user-guide",)),
    ("Integrations", ("integrations",)),
    ("Guides & Tutorials", ("guides",)),
    ("Developer Guide", ("developer-guide",)),
    ("Reference", ("reference",)),
)
MISC_SECTION = "More"

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
DESC_RE = re.compile(r"^description:\s*(.+?)\s*$", re.MULTILINE)
TITLE_RE = re.compile(r"^title:\s*(.+?)\s*$", re.MULTILINE)
SLUG_RE = re.compile(r"^slug:\s*(.+?)\s*$", re.MULTILINE)
H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
# MDX pages open with component imports — markup plumbing, not prose.
MDX_IMPORT_RE = re.compile(r"^(?:import|export)\s.*$\n?", re.MULTILINE)
# Heading anchors render as part of the heading text (`[#id]` under Fumadocs,
# `{#id}` under Docusaurus); a *title* parsed out of an H1 must drop them.
_ANCHOR_SUFFIX_RE = re.compile(r"\s*\[#[A-Za-z0-9_-]+\]$")
_LEGACY_ANCHOR_SUFFIX_RE = re.compile(r"\s*\{#[A-Za-z0-9_-]+\}$")


def strip_anchor_suffix(title: str) -> str:
    """Drop a trailing heading-anchor marker from a title, whatever form it is in.

    Brackets/braces that are prose (`Hermes [1]`, `` `{#foo}` `` in a sentence)
    are untouched: the anchor regexes demand the `#` sigil and an identifier.
    """
    title = _ANCHOR_SUFFIX_RE.sub("", title)
    title = _LEGACY_ANCHOR_SUFFIX_RE.sub("", title)
    return title


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def read_frontmatter(path: Path) -> tuple[dict[str, str], str]:
    """Return ({title, description, slug}, body-markdown) for a doc file."""
    text = path.read_text(encoding="utf-8")
    m = FRONTMATTER_RE.match(text)
    meta: dict[str, str] = {}
    body = text
    if m:
        fm = m.group(1)
        body = text[m.end():]
        for key, pattern in (("description", DESC_RE), ("title", TITLE_RE), ("slug", SLUG_RE)):
            found = pattern.search(fm)
            if found:
                meta[key] = _unquote(found.group(1))
    if path.suffix == ".mdx":
        body = MDX_IMPORT_RE.sub("", body)
    return meta, body


def slug_for(path: Path) -> str:
    """URL slug for a page: `user-guide/messaging/index.md` → `user-guide/messaging`."""
    rel = path.relative_to(DOCS).with_suffix("")
    if rel.name == "index":
        rel = rel.parent
    return "" if str(rel) == "." else rel.as_posix()

def doc_path(slug: str) -> Path | None:
    """The file backing a slug, whether it's a page or a section landing page."""
    for ext in DOC_EXTS:
        for candidate in (DOCS / f"{slug}{ext}", DOCS / slug / f"index{ext}"):
            if candidate.exists():
                return candidate
    return None


def iter_docs() -> list[str]:
    """Every indexable page, as a slug — the one enumeration of the docs tree.

    llms.txt lists exactly these and llms-full.txt emits exactly these, so a
    page on disk cannot be missing from either output.
    """
    slugs = set()
    for ext in DOC_EXTS:
        for path in DOCS.rglob(f"*{ext}"):
            slug = slug_for(path)
            # The docs landing page is this index's subject, not an entry in it.
            if slug and not slug.startswith(SKILL_CATALOG):
                slugs.add(slug)
    return sorted(slugs)


def section_for(slug: str) -> str:
    for section, prefixes in ABSORB:
        if any(slug == prefix or slug.startswith(f"{prefix}/") for prefix in prefixes):
            return section
    return MISC_SECTION


def resolve_meta(slug: str) -> tuple[str, str]:
    """(title, description) for a page, falling back to its H1 then its slug."""
    path = doc_path(slug)
    if path is None:
        return slug, ""
    meta, body = read_frontmatter(path)
    title = meta.get("title")
    if not title:
        h1 = H1_RE.search(body)
        title = strip_anchor_suffix(h1.group(1)) if h1 else slug.rsplit("/", 1)[-1].replace("-", " ").title()
    return title, meta.get("description", "")


def resolve_desc(slug: str, provided: str | None) -> str:
    """Resolve short description for llms.txt entry."""
    return provided or resolve_meta(slug)[1]


def _entry(slug: str, title: str, desc: str) -> str:
    url = f"{SITE_BASE}/{slug}"
    return f"- [{title}]({url}): {desc}" if desc else f"- [{title}]({url})"


def emit_llms_index() -> str:
    """Build the llms.txt index: curated pages lead a section, the rest follow."""
    curated = {slug for _section, items in SECTIONS for slug, _title, _desc in items}
    absorbed: dict[str, list[str]] = {}
    for slug in iter_docs():
        if slug not in curated:
            absorbed.setdefault(section_for(slug), []).append(slug)

    lines: list[str] = []
    lines.append("# Hermes Agent")
    lines.append("")
    lines.append(
        "> The self-improving AI agent built by Nous Research. A terminal-native "
        "autonomous coding and task agent with persistent memory, agent-created skills, "
        "and a messaging gateway that lives on 21+ messaging platforms — 19 native to "
        "the gateway plus IRC and Microsoft Teams via plugins (Telegram, Discord, Slack, "
        "SMS, Matrix, ...). Runs on local, Docker, SSH, Daytona, Modal, Vercel Sandbox or plugin "
        "backends. Works with Nous Portal, OpenRouter, OpenAI, Anthropic, Google, or any "
        "OpenAI-compatible endpoint."
    )
    lines.append("")
    lines.append(
        "Install: `curl -fsSL https://raw.githubusercontent.com/NousResearch/"
        "hermes-agent/main/scripts/install.sh | bash`  "
        "(Linux, macOS, WSL2)"
    )
    lines.append("")
    lines.append("Repo: https://github.com/NousResearch/hermes-agent")
    lines.append("")

    for section, items in SECTIONS:
        lines.append(f"## {section}")
        lines.append("")
        for slug, title, desc_override in items:
            lines.append(_entry(slug, title, resolve_desc(slug, desc_override)))
        for slug in absorbed.pop(section, []):
            lines.append(_entry(slug, *resolve_meta(slug)))
        lines.append("")

    # Only MISC_SECTION can survive the pops — a page whose path matched no
    # section still has to appear somewhere.
    for section, slugs in absorbed.items():
        lines.append(f"## {section}")
        lines.append("")
        for slug in slugs:
            lines.append(_entry(slug, *resolve_meta(slug)))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# --- llms-full.txt: normalize authored syntax for standalone readers --------
#
# Corpus bodies carry two forms that only mean something inside the
# Fumadocs pipeline —
#
#   - `:::kind[Title]` … `:::` container directives (remark-directive
#     callouts) render as boxes on the site and as bare colons in a plain
#     text reader;
#   - relative `./page.md` / `../section/index.mdx` links are resolved
#     against the site's content directory at build time, so in the
#     published file they point at files the reader does not have.
#
# Both become what a plain Markdown reader understands: callouts become
# `**Kind — Title**` label lines (content untouched), document links become
# the published page URLs. Code fences and inline-code examples are never
# touched — a `:::tip` inside a fence is an example, not a callout, and the
# same holds for link syntax inside code.

# Whole-line container directive: `:::kind`, `:::kind[Title]`. The title runs
# to the last `]` on the line so bracketed titles (`The [SILENT] Trick`)
# survive whole; indent is kept so callouts inside list items stay aligned.
_CALLOUT_OPEN_RE = re.compile(
    r"^(?P<indent>\s*):::(?P<kind>[A-Za-z][A-Za-z0-9_-]*)(?:\[(?P<title>.*)\])?\s*$"
)
# Fence marker, possibly inside a blockquote (`> ```bash` fences are still
# fences — treating them as prose would flip the backtick parity and mute
# normalization for the rest of the document).
_FENCE_RE = re.compile(r"^(?:\s|>\s?)*(`{3,}|~{3,})")


def _code_spans(line: str, open_len: int) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
    """Inline-code spans in `line`: (closed spans, the span still open).

    `open_len` is the delimiter length of a span opened on an earlier line
    (0 = none). Closed spans are (start, end) byte ranges, delimiters
    included; a span still open at the end of the line is (start, delimiter
    length) — its extent within this line is (start, len(line)).

    CommonMark delimiter rules: a span opens with a run of N backticks and
    closes on the next run of exactly N — longer and shorter runs are
    content, never closers. Backslash escapes work in live text only, so
    an escaped backtick in prose opens nothing (and cannot mute later
    lines), while inside a span the backslash is content and the backtick
    closes it.
    """
    closed: list[tuple[int, int]] = []
    start = 0
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
                i += 1  # escaped in prose — a literal backtick, not a delimiter
                continue
        run = i
        while i < n and line[i] == "`":
            i += 1
        if open_len == 0:
            start, open_len = run, i - run
        elif i - run == open_len:
            closed.append((start, i))
            open_len = 0
    return closed, (start, open_len) if open_len else None


def _line_kinds(body: str) -> list[tuple[bool, str]]:
    """Classify every line as live prose or code, for normalization.

    Returns ``(live, line)`` per line, where live is False for fenced code
    (opening and closing marker included), for indented code blocks, and
    for lines that continue an
    inline code span opened on an earlier line. Both callout and link
    normalization touch live lines only — a ``:::tip`` or a ``](page.md)``
    inside code is an example, never authored syntax.

    Fences close on a marker of the same character and at least the opening
    length; inline spans follow `_code_spans` (exact-length closers,
    escape-aware). A live line that opens an unclosed span runs it into the
    following lines; a continuing or closing line is skipped whole, and span
    tracking stops at its closer — later backticks on an already-muted line
    must not mute unrelated prose after it (skipping can only miss a
    rewrite, never corrupt one).
    """
    kinds: list[tuple[bool, str]] = []
    in_fence = False
    fence_char = ""
    fence_len = 0
    carry = 0
    # An indented code block starts after a blank line (or at the top) and
    # runs while lines stay blank or indented by four or more columns. A
    # paragraph's indented continuation never starts one: CommonMark needs
    # the interruption, so list-item continuations stay live prose.
    prev_blank = True
    in_indented = False
    for line in body.split("\n"):
        blank = not line.strip()
        indented = line.startswith(("    ", "\t"))
        if not in_fence and not carry:
            if in_indented and not (blank or indented):
                in_indented = False
            if not in_indented and indented and not blank and prev_blank:
                in_indented = True
            if in_indented:
                kinds.append((False, line))
                prev_blank = blank
                continue
        prev_blank = blank
        m = _FENCE_RE.match(line)
        if in_fence:
            kinds.append((False, line))
            if m and m.group(1)[0] == fence_char and len(m.group(1)) >= fence_len:
                in_fence = False
            continue
        if m:
            fence_char, fence_len = m.group(1)[0], len(m.group(1))
            in_fence = True
            kinds.append((False, line))
            continue
        if carry:
            kinds.append((False, line))
            closed, _still_open = _code_spans(line, carry)
            if closed:
                carry = 0
            continue
        _closed, open_span = _code_spans(line, 0)
        carry = open_span[1] if open_span else 0
        kinds.append((True, line))
    return kinds


def _callout_label(kind: str, title: str | None) -> str:
    """`tip` + `Fastest path: Nous Portal` → `**Tip — Fastest path: Nous Portal**`."""
    label = kind[0].upper() + kind[1:]
    return f"**{label} — {title}**" if title else f"**{label}**"


def _convert_callouts(body: str) -> str:
    """Replace Fumadocs container directives with plain Markdown labels.

    The opening `:::kind[Title]` line becomes a `**Kind — Title**` label; the
    closing `:::` line is dropped. The callout's own content is never touched.
    Where a marker sat directly against surrounding prose, a blank line keeps
    the old paragraph boundaries — dropping the marker must not weld two
    paragraphs into one.
    """
    src = body.split("\n")
    live = [is_live for is_live, _line in _line_kinds(body)]
    out: list[str] = []
    for i, line in enumerate(src):
        if not live[i]:
            out.append(line)
            continue
        m = _CALLOUT_OPEN_RE.match(line)
        if m is not None:
            if out and out[-1].strip():
                out.append("")
            out.append(f"{m.group('indent')}{_callout_label(m.group('kind'), m.group('title'))}")
            continue
        if line.strip() == ":::":
            # Keep a paragraph break when live text follows the callout.
            if i + 1 < len(src) and src[i + 1].strip() and out and out[-1].strip():
                out.append("")
            continue
        out.append(line)
    return "\n".join(out)


# An authored Markdown file link: `[text](./page.md#anchor)`. Only these get
# URL treatment — images (`![…](…)` via the lookbehind), external URLs,
# pure fragments and non-document paths are somebody else's to resolve.
_DOC_LINK_RE = re.compile(r"(?<!\!)\[(?P<text>[^\]]*)\]\((?P<href>[^)\s]+)\)")
_EXTERNAL_RE = re.compile(r"^(?:[a-z][a-z\d+.-]*:|//|#)", re.IGNORECASE)
_DOC_TARGET_RE = re.compile(r"^([^?#]+\.mdx?)([?#].*)?$", re.IGNORECASE)


def _published_url(href: str, source_rel: str) -> str | None:
    """Published site URL for an authored `.md`/`.mdx` link target.

    Returns None when the target is not a document this tree can stand behind
    (external URL, fragment, non-Markdown file, a path that resolves outside
    the docs root — by `..` traversal or symlink — or a file that does not
    exist); those links are left exactly as authored rather than turned into
    a guessed URL.

    The mapping mirrors the docs app's own link resolver
    (`apps/docs/src/lib/content.mjs`, `resolveDocumentLink`), which owns what
    these links become on the rendered site: rooted targets drop a `/docs`
    prefix, relative targets resolve against the linking page's directory,
    `index` landing pages collapse to their directory route — the root
    `index` to the site root itself — and a frontmatter `slug:` override
    wins. llms-full.txt is built from the English tree only,
    so targets resolve in that same tree — the site's locale fallback never
    has anything to fall back to here.
    """
    if _EXTERNAL_RE.match(href):
        return None
    m = _DOC_TARGET_RE.match(href)
    if m is None:
        return None
    file_part, suffix = m.group(1), m.group(2) or ""
    if file_part.startswith("/"):
        rel = re.sub(r"^/(?:docs/)?", "", file_part)
    else:
        rel = posixpath.normpath(posixpath.join(posixpath.dirname(source_rel), file_part))
    path = DOCS / rel
    # Both `..` traversal and symlinks can point outside the content tree,
    # for rooted and relative targets alike; one containment check on the
    # resolved path covers every form.
    if not path.resolve().is_relative_to(DOCS.resolve()):
        return None  # escapes the docs tree — leave as authored
    if not path.is_file():
        return None
    meta, _body = read_frontmatter(path)
    # `(?:^|/)index$` — the same collapse the docs app applies: a nested
    # index routes to its directory, the root index to the site root.
    route = meta.get("slug") or re.sub(r"(?:^|/)index$", "", re.sub(r"\.mdx?$", "", rel))
    parts = [p for p in route.split("/") if p]
    return f"{SITE_BASE}/{'/'.join(parts)}{suffix}"


_WRAPPED_TAIL_RE = re.compile(r"^(?P<text>[^\[\]]*)\]\((?P<href>[^)\s]+)\)")


def _opens_link(line: str) -> bool:
    """True when the line's last `[` (outside inline code) is never closed."""
    closed, open_span = _code_spans(line, 0)
    if open_span is not None:
        return False
    code = closed
    start = line.rfind("[")
    while start >= 0 and any(cs <= start < ce for cs, ce in code):
        start = line.rfind("[", 0, start)
    return start >= 0 and "]" not in line[start:] and not line[start - 1 : start] == "!"


def _rewrite_document_links(body: str, source_rel: str) -> str:
    """Turn authored document links into published page URLs, line by line.

    `source_rel` is the linking page's path relative to DOCS (posix style) —
    the base the authored links resolve against. Code fences and inline-code
    spans are skipped: `_line_kinds` flags the former, and a link that begins
    inside an inline-code span of any delimiter length stays byte-identical,
    while prose links — including links whose *text* contains code spans —
    still rewrite.
    """
    out: list[str] = []
    # Hard-wrapped prose can split a link's text across lines: `[multi-gateway`
    # then `guide](./page.md)`. Remember a live line that leaves a `[` open so
    # the next one's leading `text](href)` is rewritten as the link's tail.
    open_link = False
    for live, line in _line_kinds(body):
        if not live:
            open_link = False
            out.append(line)
            continue

        if open_link:
            tail = _WRAPPED_TAIL_RE.match(line)
            if tail:
                url = _published_url(tail.group("href"), source_rel)
                if url is not None:
                    line = f"{tail.group('text')}]({url})" + line[tail.end():]
        open_link = _opens_link(line)

        # A link that begins inside a code span — any delimiter length — is
        # an example, not authored syntax; code bytes stay verbatim.
        closed, open_span = _code_spans(line, 0)
        code = closed if open_span is None else [*closed, (open_span[0], len(line))]

        def rewrite(m: re.Match[str]) -> str:
            if any(cs <= m.start() < ce for cs, ce in code):
                return m.group(0)  # link example inside inline code
            url = _published_url(m.group("href"), source_rel)
            return m.group(0) if url is None else f"[{m.group('text')}]({url})"

        out.append(_DOC_LINK_RE.sub(rewrite, line))
    return "\n".join(out)


_HEADING_ANCHOR_RE = re.compile(r"^(#{1,6}\s.*?)\s*\[#[A-Za-z0-9_-]+\]\s*$")
_MDX_COMMENT_RE = re.compile(r"\s*\{/\*.*?\*/\}")


def _strip_authoring_syntax(body: str) -> str:
    """Drop heading `[#id]` suffixes and `{/* ... */}` comments from prose.

    Both are instructions to the renderer, not text. Code fences, indented
    code and inline-code spans keep them verbatim, as `_line_kinds` and
    `_code_spans` define those.
    """
    out: list[str] = []
    for live, line in _line_kinds(body):
        if live:
            line = _HEADING_ANCHOR_RE.sub(r"\1", line)
            if "{/*" in line:
                closed, open_span = _code_spans(line, 0)
                code = closed if open_span is None else [*closed, (open_span[0], len(line))]

                def drop(m: re.Match[str]) -> str:
                    brace = m.start() + m.group(0).index("{")
                    return m.group(0) if any(cs <= brace < ce for cs, ce in code) else ""

                line = _MDX_COMMENT_RE.sub(drop, line)
        out.append(line)
    return "\n".join(out)


def emit_llms_full() -> str:
    """Concatenate every doc under apps/docs/content/en/ into a single markdown file.

    Bodies are normalized for standalone readers before emission: callout
    directives become Markdown labels and repository-relative document links
    become published page URLs. See `_convert_callouts` for why the authored
    forms cannot ship as-is.
    """
    seen: set[Path] = set()
    chunks: list[str] = [
        "# Hermes Agent — Full Documentation\n",
        (
            "This file is the entire Hermes Agent documentation concatenated for LLM "
            "context ingestion. Section order reflects docs-site navigation: Getting "
            "Started, Using Hermes, Features, Messaging, Integrations, Guides, "
            "Developer Guide, Reference, then everything else.\n"
        ),
        "Canonical site: https://hermes-agent.nousresearch.com/docs\n",
        "Short index: https://hermes-agent.nousresearch.com/docs/llms.txt\n",
        "\n---\n\n",
    ]

    def emit_file(slug: str) -> None:
        path = doc_path(slug)
        if path is None or path in seen:
            return
        seen.add(path)
        title, _desc = resolve_meta(slug)
        _meta, body = read_frontmatter(path)
        source_rel = path.relative_to(DOCS).as_posix()
        body = _convert_callouts(_strip_authoring_syntax(_rewrite_document_links(body, source_rel)))
        chunks.append(f"<!-- source: apps/docs/content/en/{path.relative_to(DOCS)} -->\n")
        chunks.append(f"# {title}\n\n")
        chunks.append(body.rstrip() + "\n\n---\n\n")

    # Curated order first, so a reader truncating on token budget keeps the
    # pages that matter most; then everything else the docs tree holds.
    for _section, items in SECTIONS:
        for slug, _title, _desc in items:
            emit_file(slug)
    for slug in iter_docs():
        emit_file(slug)

    return "".join(chunks).rstrip() + "\n"


def main() -> None:
    STATIC.mkdir(exist_ok=True)
    index = emit_llms_index()
    full = emit_llms_full()
    (STATIC / "llms.txt").write_text(index, encoding="utf-8")
    (STATIC / "llms-full.txt").write_text(full, encoding="utf-8")
    print(f"Wrote {STATIC / 'llms.txt'} ({len(index):,} bytes)")
    print(f"Wrote {STATIC / 'llms-full.txt'} ({len(full):,} bytes)")


if __name__ == "__main__":
    main()
