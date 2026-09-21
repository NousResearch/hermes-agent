"""Long-form ArticleDraft generator for X Articles.

Mirrors the educational-post path in llm_generate.build_educational_prompt
but emits a structured markdown article: title, lede, 3-5 H2 sections, a
"what I'd try next" takeaway, and a soft build-in-public sign-off. Length
target ~700-1200 words for deep_dive, ~500-800 for digest.

Reuses llm_generate._llm_configs() / _call_llm() / _load_voice_skill() /
gate_post() so the LLM chain and quality rules stay in one place. The
article-specific structure rules (>=3 H2, takeaway section) are
enforced by article_gates.check, not here.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Optional

import context_enrich
import kb_retrieve
from llm_generate import (
    _call_llm,
    _llm_configs,
    _load_voice_skill,
    gate_post,
)


# Per-mode target words. Long-form articles must earn the reader's time, so
# these are full-article lengths, not extended posts. The gate floor
# (ARTICLE_MIN_WORDS) sits well below the lower mode so a slightly short but
# rich article still passes.
WORDS_TARGET = {"deep_dive": 2200, "digest": 1300}

# brand -> publishing platform for the article.
_PLATFORM_BY_BRAND = {"sahil_twitter": "twitter", "sahil_linkedin": "linkedin"}

# Article craft rules, distilled from the on-box skills so the writing reflects
# real best practice rather than the model's defaults:
#   - ~/.hermes/skills/social-content/references/x-algorithm-for-you-feed-2026.md
#     (X: binary quality gate, slop detection, dwell-time ranking, original insight)
#   - ~/.hermes/skills/social-content/SKILL.md LinkedIn section ("what works" /
#     "what doesn't" / format tips: stories with a lesson, contrarian takes,
#     behind-the-scenes building, original data, hook before the fold)
#   - ~/.hermes/skills/avoid-ai-writing (AI-ism removal)
_DEPTH_CONTRACT = (
    "## What makes this worth reading (depth and value contract)\n"
    "Write something a busy builder finishes and immediately uses. Every "
    "section must repay the reader's time with at least one of: a mechanism "
    "explained from first principles, a concrete worked example with real "
    "specifics, an honest trade-off and the reasoning behind it, a decision "
    "rule they can reuse, a pitfall and how to avoid it, or an alternative you "
    "considered and rejected and why. Show the actual code or config and walk "
    "through why each part matters, not merely what it does. Include a section "
    "that tells the reader how to apply this in their own system, with concrete "
    "steps. Anticipate the obvious 'but what about...' objection and answer it. "
    "No padding: every paragraph must advance understanding."
)

_REACH_X = (
    "## What earns reach on X (2026 For You algorithm)\n"
    "The quality screen is binary: low-quality or slop content gets zero "
    "out-of-network distribution. Grok explicitly tags AI slop, so avoid "
    "repetitive templates, boilerplate mantras, and over-polished sameness; "
    "vary sentence length and section rhythm so it reads like a person. Dwell "
    "time is a ranking signal, so the opening must hook and the body must hold "
    "a technical reader to the end. Original insight and real data win; generic "
    "recaps lose."
)

_REACH_LINKEDIN = (
    "## What earns reach on LinkedIn\n"
    "Professionals reward personal stories with a clear lesson, contrarian but "
    "defensible takes, genuine behind-the-scenes of building, and original data "
    "or insight. The first line is everything: it must hook before the 'see "
    "more' fold. Write in short paragraphs with line breaks for skimmability. "
    "Avoid corporate speak, generic motivational filler, and overt promotion. "
    "Lead with the human and the lesson, not the announcement. Slightly more "
    "context-setting than X, but the same rule holds: every section must teach "
    "something concrete."
)


def _article_craft(platform: str) -> str:
    """Depth contract plus the platform-specific reach guidance."""
    reach = _REACH_LINKEDIN if (platform or "").lower() == "linkedin" else _REACH_X
    return f"{_DEPTH_CONTRACT}\n\n{reach}"


def platform_for(brand: str) -> str:
    """Publishing platform for a brand. Defaults to twitter."""
    return _PLATFORM_BY_BRAND.get((brand or "").lower(), "twitter")


def _signal_topic_text(sig: dict) -> str:
    return (sig.get("summary")
            or sig.get("variables", {}).get("summary", "")
            or sig.get("variables", {}).get("title", "")
            or sig.get("signal_id", "")).strip()


def enrich_signal(sig: dict) -> str:
    """Per-signal rich context blob via context_enrich."""
    return context_enrich.enrich(sig) or ""


def retrieve_kb(topic: str, limit: int = 3) -> list[str]:
    """Author's prior takes from kb_retrieve."""
    return kb_retrieve.retrieve(topic, limit=limit) or []


def build_article_prompt(brand: str, plan: dict, context_blob: str,
                         kb_snippets: list[str],
                         retry_feedback: Optional[str] = None) -> dict:
    """System + user prompt for the long-form article LLM call."""
    voice = _load_voice_skill(brand)
    if not voice:
        voice = (f"Write a long-form technical article for {brand}, direct and "
                 f"specific. Builders-leaning, British English, no em-dashes.")
    mode = plan.get("mode", "deep_dive")
    platform = platform_for(brand)
    label = "LinkedIn article" if platform == "linkedin" else "X Article"
    target = WORDS_TARGET.get(mode, 900)
    title_hint = (plan.get("title_hint") or "").strip()
    pillars_hint = ", ".join({s.get("pillar", "") for s in plan.get("signals", [])
                              if s.get("pillar")})
    signal_lines = "\n".join(
        f"- ({s.get('signal_type','')}) {s.get('summary','')}"
        for s in plan.get("signals", [])
    ) or "(no signals)"
    takes = "\n".join(f"- {t}" for t in (kb_snippets or [])) or "(none on file)"

    rules = [
        f"- Length: ~{target} words. This is a full article, not a long post; "
        "do not stop short. Aim for the target and only stop when the topic is "
        "genuinely covered in depth.",
        "- British English. No em-dashes.",
        "- No AI-isms. No 'Let's dive in' / 'In today's world' / 'Great question'.",
        "- Do NOT invent statistics, user counts, growth numbers, dates, or metrics. "
        "Cite only numbers and terms that appear in the context.",
        "- Use real data, tool names, and quotes from the context where they help.",
        "- Prefer concrete specifics over abstraction every time.",
    ]
    if retry_feedback:
        rules.append(f"- Previous attempt rejected: {retry_feedback}")

    system = "\n".join([
        f"You are writing a long-form {label} for '{brand}', mode '{mode}'.",
        "",
        "## Brand voice (use exactly)", voice,
        "",
        "## Educational frame",
        ("You are writing EDUCATIONAL long-form content for the AI builder "
         "community. Teach a concrete idea grounded in the author's real work: "
         "explain the mechanism in depth, cite specific details (real numbers, "
         "tools, techniques), and leave the reader able to do something new. "
         "Builders-leaning but accessible to technical peers. Never generic "
         "hype. Never invent facts beyond the context."),
        "",
        _article_craft(platform),
        "",
        "## Structure (mandatory)",
        "- One `# Title` (specific, not clickbait).",
        "- 2-3 sentence lede / hook paragraph immediately after the title.",
        "- 4-6 `## H2` sections in prose that move from the problem, to the "
        "mechanism, to a worked example with real code/config, to the "
        "trade-offs and pitfalls. Use code and quote blocks where they help.",
        "- One `## How to apply this` section with concrete steps the reader "
        "can follow in their own system.",
        "- One final `## What I'd try next` section (or `## Takeaway` / "
        "`## Try this next` — same idea, your wording).",
        "- Optional soft build-in-public sign-off at the end.",
        "",
        "## Rules", *rules,
    ])

    user = "\n".join([
        f"Title hint: {title_hint}" if title_hint else "",
        f"Pillars: {pillars_hint}" if pillars_hint else "",
        "## Chosen signals", signal_lines,
        "",
        "## Real context (ground the article in this; quote numbers and "
        "tool names verbatim where they help)", context_blob or "(none)",
        "",
        "## Author's prior takes (reflect this thinking, do not repeat)", takes,
        "",
        "Write the article now.",
    ])
    user = "\n".join(line for line in user.splitlines() if line is not None)

    return {"system": system, "user": user}


def _call_llm_first(system: str, user: str) -> Optional[str]:
    """Try the LLM chain once; return first non-empty body or None.

    Kept as its own helper so tests can monkeypatch the LLM call without
    coupling to the rest of the chain.
    """
    for cfg in _llm_configs(longform=True):
        # Long-form: lift the token cap and timeout so a full ~2200-word
        # article completes instead of truncating before its closing section.
        body = _call_llm(system, user, cfg, timeout=180, max_tokens=8000)
        if body:
            return body
    return None


def _slug_from_title(title: str) -> str:
    """2-4 words, kebab-case, lower-case, ASCII-only, fallback 'article'."""
    words = re.findall(r"[a-zA-Z0-9]+", title.lower())
    if not words:
        return "article"
    return "-".join(words[:4]) or "article"


def _has_minimum_structure(body_md: str) -> bool:
    """Quick shape check used inside the retry loop before gate_post."""
    h2_count = sum(1 for line in body_md.splitlines() if line.startswith("## "))
    return h2_count >= 3


def write(plan: dict, brand: str = "sahil_twitter",
          dry_run: bool = False, dryrun_dir: Optional[Path] = None,
          max_retries: int = 1) -> Optional[dict]:
    """Generate one ArticleDraft. Returns None when the chain is dead.

    write() is the single LLM call: the structure check, gate, and retry
    loop live in `article_gates.check()` and the pipeline orchestrator.
    `max_retries` here is a safety net for transient chain failures
    (endpoint dead, quota hit) and re-invokes the LLM with the same
    prompt — no `retry_feedback` is threaded by design; that lives in the
    higher-level gate path.

    In dry_run mode, the LLM is never called; the function writes a marker
    file to `dryrun_dir` summarising what would have been generated and
    returns a draft dict with `dryrun=True` and `dryrun_path` set.
    """
    if not plan or not plan.get("signals"):
        return None

    # Aggregate context from all chosen signals (cap at 3 to stay readable).
    context_blob = "\n\n---\n\n".join(
        enrich_signal(s) for s in plan["signals"][:3]
    ) or ""
    kb = retrieve_kb(plan.get("title_hint", "") or plan["signals"][0].get("summary", ""))
    kb = (kb or [])[:3]

    if dry_run:
        slug = _slug_from_title(plan.get("title_hint", "") or "article")
        if dryrun_dir is not None:
            dryrun_dir = Path(dryrun_dir)
            dryrun_dir.mkdir(parents=True, exist_ok=True)
            marker = dryrun_dir / f"dryrun-{slug}.md"
            marker.write_text(
                f"<!-- dryrun -->\n# {plan.get('title_hint','Article')}\n\n"
                f"Mode: {plan.get('mode')}\nPillar: {plan.get('pillar')}\n\n"
                f"Signals: {len(plan.get('signals',[]))}\nContext chars: {len(context_blob)}\n",
                encoding="utf-8",
            )
            return {
                "title": plan.get("title_hint", "Article"),
                "body_md": "",
                "mode": plan.get("mode"),
                "pillar": plan.get("pillar"),
                "slug": slug,
                "signals": plan["signals"],
                "context": context_blob,
                "kb_snippets": kb,
                "dryrun": True,
                "dryrun_path": str(marker),
            }
        return {
            "title": plan.get("title_hint", "Article"),
            "body_md": "",
            "mode": plan.get("mode"),
            "pillar": plan.get("pillar"),
            "slug": slug,
            "signals": plan["signals"],
            "context": context_blob,
            "kb_snippets": kb,
            "dryrun": True,
        }

    last_body: Optional[str] = None
    for attempt in range(max_retries + 1):
        prompts = build_article_prompt(brand, plan, context_blob, kb)
        body = _call_llm_first(prompts["system"], prompts["user"])
        if body:
            last_body = body
            break
    if not last_body:
        return None
    title = _extract_title(last_body) or plan.get("title_hint", "Article")
    return {
        "title": title,
        "body_md": last_body.strip(),
        "mode": plan.get("mode"),
        "pillar": plan.get("pillar"),
        "slug": _slug_from_title(title),
        "signals": plan["signals"],
        "context": context_blob,
        "kb_snippets": kb,
    }


def _extract_title(body: str) -> Optional[str]:
    """First `# Title` line if present, else None."""
    for line in body.splitlines():
        m = re.match(r"^#\s+(.+?)\s*$", line)
        if m:
            return m.group(1).strip()
    return None
