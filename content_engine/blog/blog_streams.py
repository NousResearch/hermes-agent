"""Blog stream configuration - the single source of per-stream truth.

Four streams map to the verified SahilBlog ingestion contract:
  - ai:        tier=ai (surfaced on /ai). source=research-paper.
  - pm:        tier=pm with non-AI tags. source=research-paper.
  - builder:   tier=builder. source=manual.
  - research:  tier=research. source=curated-roundup. Approach A lane —
               DAIR.AI-inspired curated-analysis / research-roundup. Numbered
               scannable entries, recurring tightly scoped containers,
               plain-English technical translation, cross-source synthesis,
               article-plus-social packaging. NOTE: tier="research" and
               source="curated-roundup" are new enum values; until the
               SahilBlog Astro schema (src/content.config.ts) is updated,
               the assembler clamps them to "pm" and "manual" respectively
               so production ingestion is unaffected.

Voices match the public pillar pages and the approved editorial contract:
  - AI: long-view analysis of AI concepts, practices, strategies and news.
  - PM: enterprise SaaS AI adoption, product strategy and human-led transformation.
  - Builder's Log: evidence-backed, plain-English accounts of real work.
  - Research: weekly curated roundup — what changed, what it means, what
              to do, with explicit judgement and honest limitations.

All streams use the SahilBlog house visual contract.
"""
from __future__ import annotations

STREAMS: dict[str, dict] = {
    "ai": {
        "goal": (
            "Long-view analysis of AI concepts, movements, practices and news, "
            "including prompt, harness and loop engineering plus AI theory and strategy "
            "such as speculative decoding, memory management, MCPs and skills."
        ),
        "source_categories": [
            "ai_concepts", "ai_movements_practices_news",
            "prompt_harness_loop_engineering", "ai_theory_strategy",
        ],
        # Surfaced on /ai (tier=ai, its own page).
        "tier": "ai",
        "base_tags": ["ai"],
        "source": "research-paper",
        "format": "essay",
        "voice": (
            "Professional, analytical and thesis-driven. Lead with a concrete, "
            "well-supported claim; explain the mechanism plainly; avoid hype. Cover "
            "durable technical concepts or rigorously sourced industry/economics "
            "analysis with what, why, how and implications."
        ),
        "structure": (
            "Open with the thesis. Explain the mechanism with real figures. For any "
            "NAMED current event (acquisition, regulation, product launch) you MUST "
            "only state facts that were verified via news_verify; if unverified, "
            "reframe to the durable pattern/economics and do not assert the event as "
            "fact. Close with implications."
        ),
        "word_target": 1700,
        "section_target": 6,
        "sources": ["paper_synthesis", "ai_news", "ai_labs", "harness_cli", "compute_economics"],
        "image_palette_brand": "sahil_twitter",
        # Blueprint format: architectural analysis with primitive mapping tables
        # and Mermaid diagrams. Rotate based on topic type.
        "formats": ["essay", "blueprint"],
    },
    "pm": {
        "goal": (
            "Practical PM insight on enterprise SaaS AI adoption, product strategy, "
            "usable frameworks and human-led AI transformation."
        ),
        "source_categories": [
            "enterprise_saas_ai_adoption", "product_strategy",
            "practical_frameworks", "human_led_transformation",
        ],
        "tier": "pm",
        "base_tags": ["product-management"],
        "source": "research-paper",
        "format": "essay",
        "voice": (
            "Technically fluent product-management analysis. Translate the technical "
            "mechanism into product choices, user impact, delivery constraints, "
            "adoption, risk, cost and ownership. Use Sahil's real experience and "
            "judgement when the supplied material supports it. Write plainly, take a "
            "bounded position, and keep uncertainty where it matters. Do not write "
            "like a software manual or a generic PM thought-leadership post."
        ),
        "structure": (
            "Choose the form that fits the evidence. Start with the real problem, "
            "observation or decision; explain only the technical mechanism needed to "
            "understand it; then connect it to the product consequence, trade-off, "
            "experiment or decision rule. Use a reflective personal take when the source "
            "supports it. End when the argument is complete rather than adding a "
            "ritual takeaway heading."
        ),
        "word_target": 1500,
        "section_target": 5,
        "sources": ["paper_synthesis", "pm_frameworks", "ai_adoption", "pm_tools"],
        "image_palette_brand": "sahil_twitter",
    },
    "builder": {
        "goal": (
            "Plain-English shipped-feature, post-mortem and infrastructure notes from "
            "KENSEI work and OSS contributions."
        ),
        "source_categories": [
            "kensei_shipped_features", "oss_contributions", "infrastructure_notes",
            "mnemosyne", "hermes_agent", "turbofit", "turbohaul_manager",
        ],
        "tier": "builder",
        "base_tags": ["kensei", "build"],
        "source": "manual",
        "format": "essay",
        "voice": (
            "Professional practitioner log in plain English. Explain a shipped feature, "
            "post-mortem or infrastructure change with evidence appropriate to the "
            "public record. Be candid about trade-offs and uncertainty; never imply a "
            "result that has not been supported."
        ),
        "structure": (
            "1) problem; 2) impact; 3) change or build; 4) intended outcome; 5) a "
            "candid reality-check on trade-offs and what remains unproven. Code snippets "
            "may illustrate public concepts but must not expose proprietary internals."
        ),
        "word_target": 1400,
        "section_target": 5,
        "sources": ["paper_synthesis", "github_repos", "kensei_app", "tool_exploration", "sahil_repos"],
        "image_palette_brand": "sahil_twitter",
    },
    # Approach A: distinct research-roundup / curated-analysis lane.
    # DAIR.AI-inspired clarity and information architecture. Numbered scannable
    # entries, recurring tightly scoped containers, plain-English technical
    # translation, cross-source synthesis, article-plus-social packaging.
    # Retains Sahil's direct voice, evidence/source integrity, clear judgement,
    # builder relevance, honest limitations, and practical translation.
    #
    # The tier ("research") and source label ("curated-roundup") are NEW values.
    # The SahilBlog Astro site (src/content.config.ts) does not yet enumerate
    # them. Until that downstream schema is updated, the assembler
    # (blog_assembler._normalise_frontmatter via schema_contract) will silently
    # clamp tier=research → tier=pm and source=curated-roundup → source=manual.
    # The aggregator pipeline stays at this version so the new lane is
    # available in the engine contract but does not break production until
    # the consumer is updated. See docs/RESEARCH_ROUNDUP_LANE.md.
    "research": {
        "goal": (
            "Curated, evidence-first roundup of recent research, tools and "
            "ecosystem developments that matter to builders and PMs operating in "
            "the AI agent / personal-agent space. Cross-source synthesis with "
            "explicit judgement, not a news feed."
        ),
        "source_categories": [
            "research_papers", "model_provider_releases", "tooling_releases",
            "ecosystem_signals", "framework_repos",
        ],
        # NEW tier. Will be clamped to "pm" by the Astro schema until updated.
        "tier": "research",
        "base_tags": ["research", "roundup"],
        # NEW source label. Will be clamped to "manual" by the Astro schema
        # until updated. Tracks the lane's provenance in the engine.
        "source": "curated-roundup",
        # NEW format. "roundup" is a distinct shape from essay/blueprint —
        # see structure below. Also clamped by the Astro schema until updated.
        "format": "roundup",
        "voice": (
            "Sahil's direct voice, evidence-first. Lead with the takeaway, not the "
            "hype. Plain-English technical translation: define jargon in one phrase "
            "the first time it appears. Clear judgement — name what is overhyped, "
            "what is genuinely new, and what to ignore. Cross-source synthesis: when "
            "two or more independent sources point the same way, say so; when they "
            "disagree, name the disagreement. Honest limitations: every claim carries "
            "a source and, where appropriate, a 'what this does not show' note. "
            "Builder relevance: every entry must answer 'what would I, a builder or "
            "PM, actually do with this on Monday morning?'. Practical translation: "
            "end each entry with one concrete next step, or a clear 'no action "
            "needed' verdict."
        ),
        "structure": (
            "Open with a 2-3 sentence thesis that states the week's through-line. "
            "Then numbered, scannable entries (## 01, ## 02, ...) each titled with "
            "the specific finding, not a generic category. Every entry follows a "
            "recurring tightly scoped container: 1) the finding in plain English; "
            "2) the evidence (named source, link, date); 3) the mechanism or "
            "context (one paragraph, no jargon without definition); 4) why a "
            "builder or PM should care; 5) honest limitations — what this does "
            "not show, what could be wrong, where the evidence is thin. End with "
            "## Takeaways — three to five concrete moves a builder or PM can make "
            "this week, and one explicit 'no action needed' item. Then ## What "
            "I'd try next — one short section naming the open question this "
            "roundup surfaced. Article-plus-social packaging: the deck and the "
            "first numbered entry must be publishable as a self-contained X/LinkedIn "
            "post without the rest of the body."
        ),
        "word_target": 1500,
        "section_target": 7,  # thesis + 5 entries + takeaways + 'try next'
        # Sources are the cross-source synthesis markers. The generator must
        # surface at least two distinct sources per post; a single-source
        # roundup is invalid for this lane.
        "sources": [
            "paper_synthesis", "arxiv", "ai_news", "harness_cli",
            "ai_labs", "github_repos", "tool_exploration",
        ],
        "image_palette_brand": "sahil_twitter",
        # Lane-specific knobs the generator threads into the prompt.
        # article_plus_social: produce a self-contained social hook for the
        # first entry (so the same draft packages as blog + X/LinkedIn).
        # require_two_distinct_sources: minimum cross-source count per post.
        # entries_target: the number of numbered entries (independent of
        # section_target, which counts the structural sections).
        "require_two_distinct_sources": True,
        "article_plus_social": True,
        "entries_target": 5,
    },
}


def tags_for(stream: str, topic_tags: list[str]) -> list[str]:
    """Merge a stream's base_tags with topic-specific tags, de-duplicated,
    order-preserved (base_tags first, then new topic_tags)."""
    base = STREAMS[stream]["base_tags"]
    merged: list[str] = list(base)
    for t in topic_tags or []:
        if t not in merged:
            merged.append(t)
    return merged
