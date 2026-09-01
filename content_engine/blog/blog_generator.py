"""Blog generator - stream-aware long-form draft generation.

Reuses the article_generator LLM chain (llm_generate._call_llm / _llm_configs /
_load_voice_skill / gate_post) and article_gates.check for quality + secret
scan, but injects the stream voice, word_target, and section_target into the
system prompt. Output is a draft dict with the blog frontmatter fields
(tier, tags, source, format) set from the stream config.
"""
from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Optional

import context_enrich
import kb_retrieve
from llm_generate import _call_llm, _llm_configs, _load_voice_skill, gate_post


class ReviewUnavailable(RuntimeError):
    """Raised when strict_review is True and the editorial reviewer degrades.

    In strict mode (bulk backfill), a degraded verdict means the draft was
    never actually reviewed. Halt instead of staging an unreviewed post.
    The daily pipeline uses strict_review=False (default) and never raises.
    """

from blog.blog_streams import STREAMS, tags_for
from blog.blog_slug import slugify
from blog.source_grounding import ground_post
from blog.blog_gate import case_study_check as _case_study_check


HOUSE_STYLE_PATH = Path(__file__).resolve().parents[1] / "docs" / "sahilblog-house-style-v1.md"


def _load_house_style() -> str:
    """Load the shared SahilBlog style contract without making it mandatory.

    The contract is repository-owned and deliberately separate from the
    per-brand voice skill. A short fallback keeps the generator usable in
    packaged or test environments where the docs directory is absent.
    """
    try:
        return HOUSE_STYLE_PATH.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return (
            "Write as a technically fluent product manager. Explain the "
            "mechanism only as far as it supports a product decision, user "
            "impact, delivery trade-off, cost, risk or measurable outcome. "
            "Use concrete evidence, varied natural prose and no invented "
            "experience or specifics. Let the article's form follow its material."
        )


def enrich_signal(sig: dict) -> str:
    """Per-signal rich context blob via context_enrich."""
    return context_enrich.enrich(sig) or ""


def retrieve_kb(topic: str, limit: int = 3) -> list[str]:
    """Author's prior takes from kb_retrieve."""
    return kb_retrieve.retrieve(topic, limit=limit) or []


def _call_llm_first(system: str, user: str) -> Optional[str]:
    """Try the LLM chain once; return first non-empty body or None.

    Uses the longform chain (CommandCode deepseek-v4-flash, opencode minimax-m3)
    since blog posts are long-form content where prose quality matters.
    """
    for cfg in _llm_configs(longform=True):
        body = _call_llm(system, user, cfg, timeout=180, max_tokens=8000)
        if body:
            return body
    return None


def _extract_title(body: str) -> Optional[str]:
    """First `# Title` line if present, else None."""
    for line in body.splitlines():
        m = re.match(r"^#\s+(.+?)\s*$", line)
        if m:
            return m.group(1).strip()
    return None


def _lede_to_description(body_md: str, max_len: int = 180) -> str:
    """Extract a one-line description (deck) from the first non-heading paragraph."""
    for line in body_md.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # Clean markdown emphasis for a plain-text deck.
        clean = re.sub(r"[*_`#]", "", s).strip()
        if clean:
            return clean[:max_len]
    return ""


_DEPTH_CONTRACT = (
    "## What makes this worth reading (depth and value contract)\n"
    "Write something a busy builder finishes and immediately uses. Every "
    "section must repay the reader's time with at least one of: a mechanism "
    "explained from first principles, a concrete worked example with real "
    "specifics, an honest trade-off and the reasoning behind it, a decision "
    "rule they can reuse, a pitfall and how to avoid it, or an alternative you "
    "considered and rejected and why. No padding: every paragraph must advance "
    "understanding."
)


def _editorial_brief(plan: dict) -> str:
    """Render an approved idea's purpose fields for every writing format."""
    brief = plan.get("editorial_brief") or {}
    labels = (
        ("Post thesis", "post_thesis"),
        ("Concrete takeaway", "concrete_takeaway"),
        ("Evidence anchor", "evidence_anchor"),
        ("Gap claim", "gap_claim"),
        ("Stream and format rationale", "stream_format_rationale"),
    )
    lines = [f"- {label}: {brief[key]}" for label, key in labels if brief.get(key)]
    return "\n".join(lines) or "(no approved editorial brief)"


def build_blog_prompt(stream: str, plan: dict, context_blob: str,
                      kb_snippets: list[str],
                      wiki_entries: Optional[list[dict]] = None,
                      retry_feedback: Optional[str] = None,
                      verification: Optional[dict] = None) -> dict:
    """System + user prompt for the blog LLM call, stream-aware.

    When ``verification`` is set (from news_verify), inject verified snippets
    or an unverified warning into the system prompt so the AI never fabricates
    an unverified named event.
    """
    s = STREAMS[stream]
    voice = s["voice"]
    word_target = s["word_target"]
    section_target = s["section_target"]
    title_hint = (plan.get("title_hint") or "").strip()
    signal_lines = "\n".join(
        f"- {sig.get('summary', '')}"
        for sig in plan.get("signals", [])
    ) or "(no signals)"
    takes = "\n".join(f"- {t}" for t in (kb_snippets or [])) or "(none on file)"

    rules = [
        f"- Length: ~{word_target} words. Full article, not a long post. "
        "Do not stop short.",
        "- British English. No em-dashes.",
        "- No AI-isms. No 'Let's dive in' / 'In today's world' / 'Great question'.",
        "- No invented statistics. Cite only numbers and terms that appear in the context.",
        "- Prefer concrete specifics over abstraction every time.",
        f"- Structure: use {section_target} as a planning target, not a rigid "
        "template. Use H2 headings only for real changes of subject; let the "
        "article form follow the material.",
        "- One `# Title` (specific, not clickbait).",
    ]
    if retry_feedback:
        rules.append(f"- Previous attempt rejected: {retry_feedback}")

    # Stream-specific mandatory sections.
    stream_mandatory = ""
    if stream == "pm":
        stream_mandatory = (
            "- If the supplied material contains a personal observation or decision, "
            "use it to ground the article's conclusion. Do not manufacture a "
            "`## Reflection` section when the material does not support one."
        )
    elif stream == "builder":
        stream_mandatory = (
            "- When the article describes a real build or infrastructure change, "
            "state the supported trade-offs and what remains unproven. Do not add "
            "a ceremonial reality-check section if the material has no such detail."
        )
    elif stream == "research":
        stream_mandatory = (
            "- MANDATORY roundup structure: open with a 2-3 sentence thesis that "
            "states the week's through-line; then FIVE numbered, scannable entries "
            "(`## 01`, `## 02`, ... — each titled with the specific finding, not a "
            "generic category); every entry MUST follow the recurring container: "
            "(a) the finding in plain English, (b) the evidence with named source "
            "and link, (c) the mechanism or context (one paragraph, jargon defined "
            "the first time it appears), (d) why a builder or PM should care, "
            "(e) honest limitations — what this does not show. Close with "
            "`## Takeaways` (three to five concrete moves, including one explicit "
            "'no action needed' verdict) and a final `## What I'd try next` "
            "(one short section naming the open question the roundup surfaced). "
            "Numbered entries (`## 01` etc.) count toward the section_target."
        )

    # Inject verification context into the system rules.
    if verification:
        claim = verification.get("query", "")
        if verification.get("verified"):
            snippets = verification.get("snippets", [])
            lines = [f"- Verified background for '{claim}':"]
            for s_ in snippets[:2]:
                lines.append(
                    f"  - {s_.get('title', '')}: {s_.get('snippet', '')}"
                )
            rules.extend(lines)
        else:
            rules.append(
                f"- WARNING: The event '{claim}' is UNVERIFIED. Do NOT state it "
                "as fact. Write the durable pattern or economics instead."
            )

    # Research stream is roundup-shaped (numbered entries, recurring container,
    # explicit takeaways + a try-next section) rather than the default essay
    # shape (lede → H2 sections → takeaways). Build a distinct prompt skeleton so
    # the writer emits the right structure instead of forcing the roundup into
    # essay framing. Voice, depth contract, and verification context are reused.
    if stream == "research":
        return _build_research_roundup_prompt(
            s, plan, context_blob, kb_snippets, wiki_entries=wiki_entries,
            retry_feedback=retry_feedback, verification=verification,
            title_hint=title_hint, signal_lines=signal_lines, takes=takes,
            rules=rules, stream_mandatory=stream_mandatory,
        )

    system = "\n".join([
        f"You are writing a long-form blog essay for SahilBlog, stream '{stream}'.",
        "",
        "## Brand voice (use exactly)", voice,
        "",
        "## SahilBlog house style (primary editorial contract)", _load_house_style(),
        "",
        "## Per-stream structure rule", s.get("structure", ""),
        "",
        _DEPTH_CONTRACT,
        "",
        "## Structure (follow the argument)",
        "- One `# Title` (specific, not clickbait).",
        "- Open with the concrete problem, observation, decision or result as "
        "early as the supplied material allows.",
        f"- Use roughly {section_target} or fewer `## H2` sections only when "
        "they mark a real change of subject. Move from situation to mechanism "
        "to product consequence, evidence, trade-off or boundary as the topic "
        "requires; do not force every stage.",
        "- Include a practical next move, decision rule or implication only when "
        "the evidence supports one. A short, complete ending is valid.",
        stream_mandatory,
        "",
        "## Rules", *rules,
    ])

    user = "\n".join([
        f"Title hint: {title_hint}" if title_hint else "",
        "## Chosen signals", signal_lines,
        "",
        "## Approved editorial brief (honour this contract)", _editorial_brief(plan),
        "",
        "## Real context (ground the article in this; quote numbers and "
        "tool names verbatim where they help)", context_blob or "(none)",
        "",
        "## Author's prior takes (reflect this thinking, do not repeat)", takes,
    ])

    # Inject LLM-WIKI context as a supporting section when entries found.
    if wiki_entries:
        wiki_lines = ["## LLM-WIKI knowledge base context"]
        wiki_lines.append(
            "Relevant entries from your internal knowledge base. Adapt and "
            "tailor this material to the stream voice — do not paste raw."
        )
        for w in wiki_entries:
            wiki_lines.append(f"\n### {w['title']}")
            wiki_lines.append(f"Source: wiki/{w['page']}")
            wiki_lines.append(w["excerpt"])
        user += "\n\n" + "\n".join(wiki_lines)

    user += "\n\nWrite the article now."
    user = "\n".join(line for line in user.splitlines() if line is not None)

    return {"system": system, "user": user}


def _build_research_roundup_prompt(
    s: dict,
    plan: dict,
    context_blob: str,
    kb_snippets: list[str],
    *,
    wiki_entries: Optional[list[dict]],
    retry_feedback: Optional[str],
    verification: Optional[dict],
    title_hint: str,
    signal_lines: str,
    takes: str,
    rules: list,
    stream_mandatory: str,
) -> dict:
    """Roundup-shaped prompt for the Approach A research stream.

    Distinct from the default essay skeleton: numbered scannable entries with a
    recurring tightly scoped container, an explicit Takeaways section, and an
    article-plus-social packaging rule that demands the deck and the first
    numbered entry be publishable as a self-contained X/LinkedIn post.

    Voice, the shared depth contract, the verification context, retry feedback,
    wiki context and the editor's brief are reused from the essay path — the
    only structural divergence is the skeleton (roundup vs essay).
    """
    voice = s["voice"]
    word_target = s["word_target"]
    section_target = s["section_target"]
    entries_target = s.get("entries_target", 5)

    system = "\n".join([
        f"You are writing a curated-research roundup for SahilBlog, stream "
        f"'research' (Approach A lane).",
        "",
        "## Brand voice (use exactly)", voice,
        "",
        "## SahilBlog house style (primary editorial contract)", _load_house_style(),
        "",
        "## Per-stream structure rule", s.get("structure", ""),
        "",
        _DEPTH_CONTRACT,
        "",
        "## Structure (mandatory — roundup shape, NOT essay shape)",
        f"- One `# Title` that names the week's through-line (specific, not "
        "clickbait).",
        "- A 2-3 sentence thesis paragraph immediately after the title that "
        "states the through-line in plain English.",
        f"- Exactly {entries_target} numbered, scannable entries: `## 01`, "
        "`## 02`, ... Each entry heading must be the specific finding, NOT a "
        "generic category like 'Models' or 'Tools'.",
        "- Each entry MUST follow the recurring tightly scoped container:",
        "  (a) the finding in plain English — one sentence, no jargon without "
        "definition,",
        "  (b) the evidence — named source, link, date,",
        "  (c) the mechanism or context — one paragraph, jargon defined the "
        "first time it appears,",
        "  (d) why a builder or PM should care — concrete,",
        "  (e) honest limitations — what this does not show, what could be "
        "wrong, where the evidence is thin.",
        "- One `## Takeaways` section: three to five concrete moves a builder "
        "or PM can make this week, and one explicit 'no action needed' item.",
        "- One final `## What I'd try next` section naming the open question "
        "the roundup surfaced.",
        "- Article-plus-social packaging: the thesis + the first numbered "
        "entry MUST be publishable as a self-contained X/LinkedIn post "
        "without the rest of the body. Keep the entry self-contained.",
        f"- Total length ~{word_target} words; section_target={section_target} "
        f"counts thesis + {entries_target} entries + Takeaways + 'try next'.",
        stream_mandatory,
        "",
        "## Rules", *rules,
    ])

    user_parts = [
        f"Title hint: {title_hint}" if title_hint else "",
        "## Chosen signals (your raw inputs — pull only entries that survive a "
        "cross-source check; drop anything that does not earn its slot)",
        signal_lines,
        "",
        "## Approved editorial brief (honour this contract)", _editorial_brief(plan),
        "",
        "## Real context (ground the roundup in this; quote numbers and tool "
        "names verbatim where they help; cite each source at least once)",
        context_blob or "(none)",
        "",
        "## Author's prior takes (reflect this thinking, do not repeat)", takes,
    ]
    user = "\n".join(user_parts)

    if wiki_entries:
        wiki_lines = ["## LLM-WIKI knowledge base context"]
        wiki_lines.append(
            "Relevant entries from your internal knowledge base. Adapt and "
            "tailor this material to the research voice — do not paste raw."
        )
        for w in wiki_entries:
            wiki_lines.append(f"\n### {w['title']}")
            wiki_lines.append(f"Source: wiki/{w['page']}")
            wiki_lines.append(w["excerpt"])
        user += "\n\n" + "\n".join(wiki_lines)

    if verification:
        claim = verification.get("query", "")
        if not verification.get("verified"):
            user += (
                "\n\n## Verification warning\n"
                f"The event '{claim}' is UNVERIFIED. Do NOT state it as fact "
                "in any roundup entry; reframe to the durable pattern or "
                "economics instead."
            )

    if retry_feedback:
        user += f"\n\n## Retry feedback from previous attempt\n{retry_feedback}"

    user += "\n\nWrite the curated-research roundup now."
    user = "\n".join(line for line in user.splitlines() if line is not None)

    return {"system": system, "user": user}


def build_blueprint_prompt(stream: str, plan: dict, context_blob: str,
                               kb_snippets: list[str],
                               wiki_entries: Optional[list[dict]] = None,
                               retry_feedback: Optional[str] = None,
                               verification: Optional[dict] = None) -> dict:
    """System + user prompt for blueprint-format blog posts.

    Blueprint posts differ from essays:
      - Include a primitive mapping table (concept -> implementation mapping)
      - Include a Mermaid architecture diagram (code block)
      - Step-by-step sequence instead of narrative prose
      - More structured, less editorial
    """
    s = STREAMS[stream]
    voice = s["voice"]
    word_target = s["word_target"]
    title_hint = (plan.get("title_hint") or "").strip()
    signal_lines = "\n".join(
        f"- {sig.get('summary', '')}"
        for sig in plan.get("signals", [])
    ) or "(no signals)"
    takes = "\n".join(f"- {t}" for t in (kb_snippets or [])) or "(none on file)"

    rules = [
        f"- Length: ~{word_target} words. Full article, not a long post.",
        "- British English. No em-dashes.",
        "- No AI-isms. No 'Let's dive in' / 'In today's world'.",
        "- No invented statistics.",
        "- Format: BLUEPRINT (architectural analysis, not essay).",
        "- MUST include a primitive mapping table in markdown:",
        "  | Concept | Implementation | Trade-off |",
        "  |--------|---------------|-----------|",
        "- MUST include a Mermaid architecture diagram in a ```mermaid code block.",
        "- Step-by-step sequence: numbered steps for implementation.",
        "- Structure: 5+ `## H2` sections. The illustrator keys off headings.",
        "- One `# Title` (specific, not clickbait).",
        "- One `## What I'd try next` section at the end.",
    ]
    if retry_feedback:
        rules.append(f"- Previous attempt rejected: {retry_feedback}")

    system = "\n".join([
        f"You are writing a blueprint-format blog essay for SahilBlog, stream '{stream}'.",
        "",
        "## Brand voice (use exactly)", voice,
        "",
        "## Blueprint format rules",
        "Blueprint posts are architectural analysis pieces. They include:",
        "1. A primitive mapping table mapping concepts to implementations and trade-offs.",
        "2. A Mermaid diagram showing the architecture (flowchart or sequence diagram).",
        "3. Numbered step-by-step implementation sequence.",
        "4. Less narrative, more structured than essays.",
        "",
        _DEPTH_CONTRACT,
        "",
        "## Rules", *rules,
    ])

    user = "\n".join([
        f"Title hint: {title_hint}" if title_hint else "",
        "## Chosen signals", signal_lines,
        "",
        "## Approved editorial brief (honour this contract)", _editorial_brief(plan),
        "",
        "## Real context", context_blob or "(none)",
        "",
        "## Author's prior takes", takes,
        "",
        "Write the blueprint article now. Include the mapping table and Mermaid diagram.",
    ])

    return {"system": system, "user": user}


def _framework_prompt_builder(stream: str, plan: dict, context_blob: str,
                                    kb_snippets: list[str],
                                    wiki_entries: Optional[list[dict]] = None,
                                    retry_feedback: Optional[str] = None) -> dict:
    """System + user prompt for original framework generation.

    Framework posts are a specialized form of blueprint posts that introduce
    a named, reusable analytical construct:
      - Name (2-4 words)
      - 3-5 levels/stages
      - Identification criteria for each level
      - Actionable guidance per level
      - Mermaid diagram showing the levels
    """
    s = STREAMS[stream]
    voice = s["voice"]
    word_target = s["word_target"]
    title_hint = (plan.get("title_hint") or "").strip()
    signal_lines = "\n".join(
        f"- {sig.get('summary', '')}"
        for sig in plan.get("signals", [])
    ) or "(no signals)"

    rules = [
        f"- Length: ~{word_target} words.",
        "- British English. No em-dashes.",
        "- No AI-isms.",
        "- Format: FRAMEWORK (original analytical construct).",
        "- MUST give the framework a memorable 2-4 word NAME.",
        "- MUST define 3-5 levels/stages of the framework.",
        "- For each level: identification criteria + actionable guidance.",
        "- MUST include a Mermaid diagram showing the levels (flowchart TD).",
        "- MUST include a primitive mapping table: | Level | Criteria | Action |.",
        "- One `# Title` that includes the framework name.",
        "- One `## What I'd try next` section at the end.",
    ]
    if retry_feedback:
        rules.append(f"- Previous attempt rejected: {retry_feedback}")

    system = "\n".join([
        f"You are writing an original framework-generation post for SahilBlog, stream '{stream}'.",
        "",
        "## Brand voice", voice,
        "",
        "## Framework format rules",
        "An original framework post introduces a reusable analytical construct.",
        "It must be:",
        "1. Named (2-4 word name, memorable).",
        "2. Structured into 3-5 levels/stages.",
        "3. Each level has identification criteria and actionable guidance.",
        "4. Include a Mermaid diagram.",
        "5. Include a mapping table.",
        "",
        _DEPTH_CONTRACT,
        "",
        "## Rules", *rules,
    ])

    user = "\n".join([
        f"Framework seed: {title_hint}" if title_hint else "",
        "## Approved editorial brief (honour this contract)", _editorial_brief(plan),
        "",
        "## Context", context_blob or "(none)",
        "",
        "Write the framework post. Name it, define the levels, include the diagram.",
    ])

    return {"system": system, "user": user}


def _wiki_home() -> Path:
    """Resolve the canonical wiki root while preserving explicit overrides."""
    return Path(
        os.environ.get("WIKI_PATH", str(Path.home() / "docs" / "wiki"))
    ).expanduser()


WIKI_HOME = _wiki_home()


def _wiki_context_for(topic: str, max_results: int = 2) -> list[dict]:
    """Search the LLM-WIKI for relevant entries and return excerpts.

    Searches concept, comparison, and repo pages by keyword matching on
    the topic string. Returns up to max_results entries with title, page
    path, and excerpt (first 2 substantive paragraphs).

    Falls back to empty list silently on any IO error. Does NOT pretend
    wiki backing exists when no match is found.
    """
    if not WIKI_HOME.exists():
        return []
    result = []
    # Search in the most structured wiki subdirs
    search_dirs = [
        WIKI_HOME / "concepts",
        WIKI_HOME / "comparisons",
        WIKI_HOME / "repos",
        WIKI_HOME / "raw" / "articles",
        WIKI_HOME / "raw" / "papers",
    ]
    keywords = topic.lower().split()
    # Filter to substantive words only (5+ chars, not stop words)
    stop_words = {"their", "there", "about", "which", "that", "this",
                   "with", "what", "when", "where", "how", "they",
                   "been", "have", "from", "into", "over", "such",
                   "than", "then", "them", "these", "those", "would"}
    keywords = [k for k in keywords if len(k) >= 5 and k not in stop_words][:8]

    if not keywords:
        return []

    try:
        for sd in search_dirs:
            if not sd.exists():
                continue
            for md_file in sorted(sd.glob("*.md")):
                if len(result) >= max_results:
                    break
                text = md_file.read_text(encoding="utf-8", errors="replace")
                text_lower = text.lower()

                # Extract title from first # heading (after YAML frontmatter)
                yaml_end = text.find("---", 3) if text.startswith("---") else -1
                body_start = yaml_end + 3 if yaml_end > 0 else 0
                body = text[body_start:]
                title = ""
                for line in body.splitlines():
                    if line.startswith("# "):
                        title = line.lstrip("# ").strip().lower()
                        break

                # Strong concept match: 2+ keyword matches in body AND
                # at least 1 keyword in the page title (ensures the wiki
                # page is about the same concept as the topic).
                kw_in_body = [k for k in keywords if k in text_lower]
                kw_in_title = [k for k in keywords if k in title]

                if len(kw_in_body) < 2:
                    continue
                if not kw_in_title:
                    continue  # page title must share at least 1 keyword

                # Extract excerpt (skip YAML frontmatter)
                paragraphs = []
                for line in body.splitlines():
                    stripped = line.strip()
                    if not stripped:
                        continue
                    if not line.startswith("#"):
                        paragraphs.append(stripped)
                if not title:
                    title = md_file.stem.replace("-", " ").title()

                excerpt = ""
                count = 0
                for p in paragraphs:
                    if count >= 2:
                        break
                    excerpt += p + "\n\n"
                    count += 1

                result.append({
                    "title": title[:80],
                    "page": str(md_file.relative_to(WIKI_HOME)),
                    "excerpt": excerpt.strip()[:500],
                })
        return result
    except (OSError, IOError):
        return []


def write(plan: dict, stream: str = "ai",
         max_retries: int = 1,
         retry_feedback: Optional[str] = None,
         verification: Optional[dict] = None) -> Optional[dict]:
    """Generate one blog draft. Returns None when the LLM chain is dead.

    When retry_feedback is set, it is threaded into build_blog_prompt so the
    LLM sees the gate's issues and can fix them on the retry.
    When verification is set (from news_verify), it is injected into the prompt.
    """
    if not plan or not plan.get("signals"):
        return None
    s = STREAMS[stream]

    context_blob = "\n\n---\n\n".join(
        enrich_signal(sig) for sig in plan["signals"][:3]
    ) or ""
    kb = retrieve_kb(plan.get("title_hint", "") or plan["signals"][0].get("summary", ""))
    kb = (kb or [])[:3]

    # LLM-WIKI context: search for relevant wiki entries matching the topic
    wiki_entries = _wiki_context_for(
        plan.get("title_hint", "") or plan["signals"][0].get("summary", "")
    )

    last_body: Optional[str] = None
    for attempt in range(max_retries + 1):
        prompts = build_blog_prompt(stream, plan, context_blob, kb,
                                    wiki_entries=wiki_entries,
                                    retry_feedback=retry_feedback,
                                    verification=verification)
        body = _call_llm_first(prompts["system"], prompts["user"])
        if body:
            last_body = body
            break
    if not last_body:
        return None

    title = _extract_title(last_body) or plan.get("title_hint", "Post")
    description = _lede_to_description(last_body)
    if not description:
        # Fall back to the title hint if the body has no lede paragraph.
        description = (plan.get("title_hint") or title)[:180]

    # Tags: merge the stream base_tags with the plan's topic tags (the router
    # already merges them, but re-merge here so write() is safe standalone).
    topic_tags = [t for t in (plan.get("tags") or []) if t not in s["base_tags"]]
    final_tags = tags_for(stream, topic_tags)

    return {
        "title": title,
        "description": description,
        "body_md": last_body.strip(),
        "slug": slugify(title),
        "tier": s["tier"],
        "tags": final_tags,
        "format": s["format"],
        "source": plan.get("source") or s["source"],
        "stream": stream,
        "signals": plan["signals"],
        "context": context_blob,
        "kb_snippets": kb,
    }


def gate_check(draft: dict) -> tuple[str, list[str]]:
    """Run article_gates.check on a blog draft; return (status, issues).

    status in 'ok' | 'fail'. Reuses the article gate so the same slop /
    em-dash / length / data-integrity / secret-scan rules apply.
    """
    import article_gates as ag
    res = ag.check(draft)
    return ("ok" if res.passed else "fail"), res.issues


def _redact_draft(draft: dict) -> None:
    """Apply secret redaction to the draft body in-place."""
    import article_gates as ag
    gate_res = ag.check(draft)
    draft["body_md"] = gate_res.redacted_body


def _verify_claims(claims: list[str]) -> list[str]:
    """Run news_verify on a list of claim strings.

    Returns a list of warning strings to inject into the retry feedback for
    claims that could not be verified. Verified claims are noted as confirmed
    context. Unverified claims get a reframe instruction.
    """
    if not claims:
        return []
    from blog.news_verify import verify_event
    warnings = []
    for claim in claims:
        result = verify_event(claim)
        if not result["verified"]:
            warnings.append(
                f"The claim '{claim}' is UNVERIFIED. Do NOT state it as fact. "
                "Reframe to the durable pattern or economics."
            )
    return warnings


def write_with_gate(plan: dict, stream: str = "ai",
                    max_retries: int = 1,
                    verification: Optional[dict] = None,
                    strict_review: bool = False,
                    case_study_exempt: bool = False) -> Optional[dict]:
    """Generate a draft, run deterministic gate + editorial reviewer, retry once.

    Pipeline:
    1. Generate draft via the LLM chain.
    2. Deterministic gate (article_gates.check): slop, em-dash, length, secrets.
    3. Editorial reviewer (blog_reviewer.review): voice, accuracy, secret-sauce,
       hype, structure via an independent LLM call.
    4. If either gate fails: collect all issues + verify claims_to_verify via
       news_verify, feed everything into a single retry LLM call.
    5. Stage only if both gates clear on the first or retry attempt.

    Returns the draft (with redacted body) on pass, or None.

    When ``strict_review`` is False (default, daily pipeline), a degraded
    reviewer verdict (LLM unavailable, malformed JSON) is treated as a neutral
    pass and the pipeline continues on the deterministic gate alone.

    When ``strict_review`` is True (bulk backfill), a degraded verdict raises
    ``ReviewUnavailable`` so the caller can halt the run instead of staging an
    unreviewed draft. The exception message names the post title.
    """
    from blog.blog_reviewer import review as _review

    def _check_strict(verdict: dict, title: str) -> None:
        if strict_review and verdict.get("degraded"):
            raise ReviewUnavailable(
                f"Editorial reviewer degraded for '{title}'; "
                "strict mode refuses to stage an unreviewed draft"
            )

    draft = write(plan, stream=stream, max_retries=max_retries,
                  verification=verification)
    if not draft:
        return None

    post_title = draft.get("title", "(untitled)")

    # --- Source grounding: inject primary-source links before gating ---
    grounding = ground_post(draft, stream=stream)
    draft["body_md"] = grounding["body_md"]
    if grounding["dead_links"]:
        _dead_links = grounding["dead_links"]
    else:
        _dead_links = []

    # --- First attempt: deterministic gate + editorial reviewer ---
    status, gate_issues = gate_check(draft)
    # Case study gate: AI stream must have named company + number.
    # Backlog pregen (research/PR-derived topics) is exempt — forcing a
    # "named company + number" onto a research analysis produces unnatural
    # content and blocks legitimate posts (see blog-backlog-pregen failures).
    if case_study_exempt:
        cs_status, cs_issues = "ok", []
    else:
        cs_status, cs_issues = _case_study_check(
            draft.get("body_md", ""), stream=stream
        )
    if cs_issues:
        gate_issues.extend(cs_issues)
        if cs_status == "fail":
            status = "fail"
    review_result = _review(draft, stream)
    _check_strict(review_result, post_title)
    review_issues = review_result["issues"] if not review_result["passed"] else []
    claims = review_result.get("claims_to_verify", [])

    if status == "ok" and review_result["passed"] and not claims:
        _redact_draft(draft)
        return draft

    # --- Collect all issues for the retry ---
    all_issues = list(gate_issues) + list(review_issues)
    # Verify any claims the reviewer flagged.
    claim_warnings = _verify_claims(claims)
    all_issues.extend(claim_warnings)
    # Dead links from source grounding go into retry feedback.
    if _dead_links:
        all_issues.append(
            "Dead links to fix: " + ", ".join(_dead_links[:3])
        )
    feedback = "; ".join(all_issues) or "rejected by quality gate"

    # --- Single retry with combined feedback ---
    draft2 = write(plan, stream=stream, max_retries=max_retries,
                   retry_feedback=feedback, verification=verification)
    if not draft2:
        return None

    status2, gate_issues2 = gate_check(draft2)
    review_result2 = _review(draft2, stream)
    _check_strict(review_result2, post_title)
    review_passed2 = review_result2["passed"]
    # On retry, we do NOT re-verify claims (avoid infinite loops). If the
    # reviewer still flags claims, they become issues but we accept the draft
    # if both gates pass and the reviewer no longer blocks.
    if status2 == "ok" and review_passed2:
        _redact_draft(draft2)
        return draft2

    # Last resort: if the deterministic gate passes and the reviewer only
    # flagged claims (not quality issues), accept with the gate's redaction.
    # The reviewer degrading to neutral pass on infra failure is handled above;
    # here we handle the case where the retry genuinely passed the deterministic
    # gate but the reviewer LLM returned a second negative verdict.
    if status2 == "ok" and not review_result2["issues"]:
        # Reviewer returned a score below threshold but no concrete issues.
        # Check score: if < 6, log warning (strict mode rejects).
        retry_score = review_result2.get("score", 5)
        if retry_score < 6:
            import logging
            logging.getLogger("blog_generator").warning(
                "Retry accepted with low reviewer score %s for '%s' "
                "(no concrete issues, strict=%s)",
                retry_score, post_title, strict_review,
            )
            if strict_review:
                return None
        _redact_draft(draft2)
        return draft2

    return None