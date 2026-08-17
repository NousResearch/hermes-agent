"""Blog stream configuration - the single source of per-stream truth.

Three streams map to the verified SahilBlog ingestion contract:
  - ai:      tier=ai (surfaced on /ai). source=research-paper.
  - pm:      tier=pm with non-AI tags. source=research-paper.
  - builder: tier=builder. source=manual.

Voices match the public pillar pages and the approved editorial contract:
  - AI: long-view analysis of AI concepts, practices, strategies and news.
  - PM: enterprise SaaS AI adoption, product strategy and human-led transformation.
  - Builder's Log: evidence-backed, plain-English accounts of real work.

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
            "Educational for the PM market: translate AI research findings into "
            "product-management practice. Explain the concept plainly, then apply it "
            "to PM work -- workflows, adoption, skills, decisions. Authoritative but "
            "accessible; minimal jargon. Personal experience is occasional seasoning, "
            "not the substance."
        ),
        "structure": (
            "1) the research concept explained simply; 2) why it matters for PMs; "
            "3) concrete application (workflow / adoption / skill); 4) a short "
            "'## Reflection' section with a considered personal take. Every post "
            "carries the reflective section."
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
