"""Source-aware editorial router for the content pipeline.

Pure decision function: no generation, no image routing, no DB writes.
Deterministic routing based on signal source, content analysis, and platform rules.

VCP 2.0 (image_router.py) remains the visual-only layer — this module ONLY
decides what to write and where to publish it.

Usage:
    route_signal(signal) -> {
        "decision": "x_post | linkedin_post | blog | article | skip",
        "platform": "twitter | linkedin | blog | article",
        "content_type": "text | text+image | article | blog",
        "scores": {"x_fit": ..., "linkedin_fit": ..., ...},
        "rationale": "...",
        "source_provenance": {...},
    }
"""

from __future__ import annotations
import re
from typing import Optional


# ── Signal type classification ──

# Signal types that are native / high-fit for X / Twitter
X_NATIVE_SIGNALS: dict[str, str] = {
    "github_push": "build_in_public",
    "hermes_pr": "build_in_public",
    "hermes_skill": "ai_tools",
    "research_tool": "ai_tools",
    "gitradar_repo": "ai_tools",
    "architecture": "tutorial",
    "harness_change": "harness_tuning",
    "screenshot": "build_in_public",
}

# Signal types that fit LinkedIn naturally (PM / leadership / business angle)
LINKEDIN_NATIVE_SIGNALS: dict[str, str] = {
    "research_signal": "pm_thought",
    "architecture": "leadership",
    "harness_change": "ai_engineering",
}

# Signal types eligible for long-form (blog / X Articles)
LONG_FORM_ELIGIBLE: set[str] = {
    "research_tool", "research_signal", "gitradar_repo",
    "architecture", "harness_change", "github_push",
}

# Signal types that are raw technical with no inherent business lesson
# (LinkedIn penalty unless they carry a reflective angle)
LINKEDIN_RAW_TECH: set[str] = {
    "github_push", "hermes_pr", "hermes_skill",
}

# Signal types with verified source evidence (live system proof)
VERIFIED_SOURCES: set[str] = {
    "github_push", "hermes_pr", "hermes_skill",
    "harness_change", "research_tool", "research_signal",
    "gitradar_repo",
}

CLAIMED_SOURCES: set[str] = {"architecture", "screenshot"}

# ── Thresholds ──

X_MAX_FRESHNESS_HOURS = 144        # 6 days — older gets penalised on X
ARTICLE_SHALLOW_CHARS = 60         # below this → too shallow for 1300 words
BLOG_MIN_CONFIDENCE = 30           # below this → weak evidence for blog
LINKEDIN_MAX_AGE_HOURS = 720       # 30 days

# Thresholds per platform — score must meet or exceed these
SCORE_THRESHOLDS: dict[str, float] = {
    "x_post": 40,
    "linkedin_post": 35,
    "blog": 40,
    "article": 45,
}

# ── Text analysis patterns ──

_LINKEDIN_REFLECTIVE = [
    "lesson", "learned", "discovered", "realised", "realized",
    "why i", "how we", "approach", "framework", "decision",
    "tradeoff", "trade-off", "stakeholder", "team", "organisation",
    "organization", "adoption", "culture", "strategy", "leadership",
    "pm ", "product ", "customer", "managed",
]

_LINKEDIN_TECH_ONLY = [
    "push", "commit", "merged", "deployed", "installed",
    "upgraded", "configured", "set up", "ran ", "triggered",
    "refactored", "bumped", "patched",
]


# ── Scoring primitives ──


def _freshness_hours(signal: dict) -> int:
    """Return the freshness window in hours."""
    hours = signal.get("freshness_hours")
    if hours is None:
        return 168
    return int(hours)


def _freshness_score(signal: dict) -> float:
    """Normalised freshness 0–100. Higher = fresher, more suited to X."""
    hours = _freshness_hours(signal)
    if hours <= 0:
        return 50
    if hours >= 720:
        return 0
    if hours <= 24:
        return 100
    return max(0, 100 - ((hours - 24) / 696) * 100)


def _proof_strength(signal: dict) -> float:
    """How strong the evidence is, 0–100."""
    t = signal.get("signal_type", "")
    if t in VERIFIED_SOURCES:
        return 90
    if t in CLAIMED_SOURCES:
        return 60
    if signal.get("source_override"):
        return 60
    return 20


def _source_confidence(signal: dict) -> float:
    """Confidence in the source itself, 0–100."""
    t = signal.get("signal_type", "")
    if t in VERIFIED_SOURCES:
        return 95
    if t == "screenshot":
        return 90
    if t == "architecture":
        return 80
    if signal.get("source_override"):
        return 50
    return 20


def _summary_text(signal: dict) -> str:
    """Aggregate all text fields from a signal for analysis."""
    v = signal.get("variables") or {}
    parts = [
        signal.get("summary", ""),
        signal.get("title_hint", ""),
        signal.get("topic", ""),
        v.get("summary", ""),
        v.get("title", ""),
        v.get("pr_title", ""),
        v.get("repo_name", ""),
        v.get("description", ""),
    ]
    return " ".join(p for p in parts if p).strip()


def _has_reader_lesson(text: str) -> bool:
    """Check if text contains a business / leadership / PM lesson for LinkedIn."""
    t = text.lower()
    reflective = sum(1 for c in _LINKEDIN_REFLECTIVE if c in t)
    tech_only = sum(1 for c in _LINKEDIN_TECH_ONLY if c in t)
    return reflective >= 1 and reflective >= tech_only


def _is_shallow(signal: dict) -> bool:
    """Signal too shallow to support a 1,300-word article."""
    summary = _summary_text(signal)
    return len(summary.strip()) < ARTICLE_SHALLOW_CHARS


def _has_source_evidence(signal: dict) -> bool:
    """Signal carries source evidence somewhere (provenance, URL, override)."""
    if signal.get("source_provenance"):
        prov = signal["source_provenance"]
        if prov.get("confidence") in ("verified", "claimed"):
            return True
        if prov.get("evidence") or prov.get("source_url"):
            return True
    if signal.get("source_url"):
        return True
    if signal.get("source_override"):
        return True
    if signal.get("signal_type") in VERIFIED_SOURCES:
        return True
    # Manual blog queue entries are accepted only when explicitly curated or
    # supported by evidence. A bare title_hint is not enough for long-form.
    if signal.get("topic_id") and signal.get("title_hint"):
        if signal.get("curated") is True:
            return True
        if signal.get("evidence") or signal.get("source_notes") or signal.get("context"):
            return True
    return False


def _is_generic_evergreen(signal: dict) -> bool:
    """True if the signal is a generic topic-bank entry, not a real signal."""
    if signal.get("signal_type"):
        return False  # Real signal — never generic
    topic = (signal.get("topic") or signal.get("title_hint") or "").lower()
    for pat in (r"^how to", r"^tips for", r"^guide to", r"^what is",
                r"^the (ultimate|complete|best)", r"^top \d+",
                r"^\d+ (ways|tips|reasons)"):
        if re.match(pat, topic):
            return True
    return False


def _has_product_metric(text: str) -> bool:
    """Check if text references a product metric that needs provenance."""
    patterns = [
        r"\d+ household", r"\d+ test", r"savings", r"£\d+",
        r"\d+% reduction", r"\d+% improvement",
    ]
    return any(re.search(p, text.lower()) for p in patterns)


# ── Source provenance builder ──


def _build_provenance(signal: dict) -> dict:
    """Build structured source_provenance from a signal dict."""
    st = signal.get("signal_type", "manual")
    sid = signal.get("signal_id") or signal.get("topic_id", "")
    variables = signal.get("variables") or {}

    # URL
    url = variables.get("url", "")
    if not url and st == "github_push" and variables.get("repo_name"):
        url = f"https://github.com/Sahil-SS9/{variables['repo_name']}"
    if not url and st == "harness_change":
        repo = variables.get("repo", "")
        sha = variables.get("sha", "")
        if repo and sha:
            url = f"https://github.com/Sahil-SS9/{repo}/commit/{sha}"

    # Summary
    summary = (signal.get("summary") or signal.get("title_hint", "")
               or variables.get("summary", "") or variables.get("title", "")
               or variables.get("repo_name", "") or signal.get("topic", ""))

    # Evidence
    evidence = []
    if url:
        evidence.append(f"URL: {url}")
    if summary:
        evidence.append(f"desc: {summary[:120]}")
    if variables.get("sha"):
        evidence.append(f"commit: {variables['sha']}")
    if variables.get("date"):
        evidence.append(f"date: {variables['date']}")
    if signal.get("evidence"):
        evidence.extend(str(e) for e in signal.get("evidence") or [])
    if signal.get("source_notes"):
        evidence.append(f"notes: {str(signal.get('source_notes'))[:160]}")
    if signal.get("context"):
        evidence.append(f"context: {str(signal.get('context'))[:160]}")
    if not evidence:
        evidence.append("manual entry")

    # Confidence label
    if st in VERIFIED_SOURCES:
        conf = "verified"
    elif st == "screenshot":
        conf = "verified"
    elif st == "architecture":
        conf = "stated"
    elif signal.get("source_override") or signal.get("curated") is True or signal.get("evidence") or signal.get("source_notes") or signal.get("context"):
        conf = "claimed"
    else:
        conf = "manual"

    return {
        "source_type": st,
        "source_id": sid,
        "source_url": url,
        "source_summary": summary[:200],
        "evidence": evidence,
        "confidence": conf,
    }


# ── Platform fit scorers ──


def _score_x_fit(signal: dict) -> float:
    """Score 0–100 for X / Twitter."""
    t = signal.get("signal_type", "")

    if t in X_NATIVE_SIGNALS:
        base = 85.0
    elif t == "research_signal":
        base = 55.0
    elif t == "screenshot":
        base = 95.0
    elif t == "harness_change":
        base = 90.0
    else:
        base = 35.0

    if t == "harness_change" and signal.get("priority", 0) >= 8:
        base += 5  # deep-dive material works on X with threads

    if _is_generic_evergreen(signal):
        base -= 25

    base += _freshness_score(signal) * 0.15

    hours = _freshness_hours(signal)
    if hours > X_MAX_FRESHNESS_HOURS:
        base -= 20

    return max(0, min(100, base))


def _score_linkedin_fit(signal: dict) -> float:
    """Score 0–100 for LinkedIn."""
    t = signal.get("signal_type", "")
    text = _summary_text(signal)

    if t == "research_signal":
        base = 80.0
    elif t == "architecture":
        base = 75.0
    elif t == "harness_change":
        base = 65.0
    elif t == "research_tool":
        base = 55.0
    elif t in LINKEDIN_RAW_TECH:
        base = 35.0
    else:
        base = 25.0

    if _has_reader_lesson(text):
        base += 20
    elif t in LINKEDIN_RAW_TECH:
        base -= 20

    if _has_product_metric(text) and not _has_source_evidence(signal):
        base -= 30

    hours = _freshness_hours(signal)
    if hours > LINKEDIN_MAX_AGE_HOURS:
        base -= 15

    return max(0, min(100, base))


def _score_blog_fit(signal: dict) -> float:
    """Score 0–100 for SahilBlog (AI / PM / Builder streams)."""
    t = signal.get("signal_type", "")

    # Manual queue entries for blog. Human-curated entries are valid, but
    # bare prompts without evidence are deliberately weaker.
    if signal.get("topic_id") and signal.get("title_hint"):
        if signal.get("curated") is True or signal.get("evidence") or signal.get("source_notes") or signal.get("context"):
            base = 80.0
        else:
            base = 35.0
    elif t in LONG_FORM_ELIGIBLE:
        base = 75.0
    else:
        base = 15.0

    if not _has_source_evidence(signal):
        base -= 25

    if signal.get("source_override"):
        base += 10

    return max(0, min(100, base))


def _score_article_fit(signal: dict) -> float:
    """Score 0–100 for X Articles (long-form, deep-dive)."""
    t = signal.get("signal_type", "")
    priority = signal.get("priority", 0)

    if t == "harness_change" and priority >= 8:
        base = 90.0
    elif t == "architecture":
        base = 80.0
    elif t == "harness_change":
        base = 60.0
    elif t == "github_push":
        base = 40.0
    else:
        base = 10.0

    if _is_shallow(signal):
        base -= 30
    if not _has_source_evidence(signal):
        base -= 20

    return max(0, min(100, base))


def _score_visual_fit(signal: dict) -> float:
    """Score 0–100 for visual content potential."""
    t = signal.get("signal_type", "")
    if t == "screenshot":
        return 95
    if signal.get("screenshot_path"):
        return 90
    if signal.get("visual_description"):
        return 80

    pillar = (signal.get("pillar") or "").lower()
    if pillar in ("data", "data_driven", "comparison", "infographic"):
        return 70
    if t in X_NATIVE_SIGNALS:
        return 60

    return 25


# ── Skip logic ──


def _check_skip_for(signal: dict, platform: str) -> tuple[bool, str]:
    """Check specific skip conditions for a (signal, platform) pair."""
    t = signal.get("signal_type", "")
    text = _summary_text(signal)

    # Blog / Article need source evidence
    if platform in ("blog", "article"):
        if not _has_source_evidence(signal):
            return True, "No source evidence for long-form content"

    # LinkedIn: raw technical without lesson
    if platform == "linkedin" and t in LINKEDIN_RAW_TECH:
        if not _has_reader_lesson(text):
            return True, "Raw technical update without business/reader lesson"

    # Product metric without provenance
    if _has_product_metric(text) and not _has_source_evidence(signal):
        return True, "Product metric without source provenance"

    # Commit subject without lesson on LinkedIn
    if platform == "linkedin":
        v = signal.get("variables") or {}
        maybe_commit = v.get("sha") or ("pr_title" in v)
        if maybe_commit and not _has_reader_lesson(text):
            return True, "Commit/PR subject without reader lesson"

    return False, ""


# ── Content type determination ──


def _recommend_content_type(signal: dict, platform: str) -> str:
    """Choose content type based on signal and platform."""
    # Screenshots and visuals → text+image
    if signal.get("signal_type") == "screenshot":
        return "text+image"
    if signal.get("screenshot_path"):
        return "text+image"
    if signal.get("visual_description"):
        return "text+image"

    if platform == "article":
        return "article"
    if platform == "blog":
        return "blog"

    if _score_visual_fit(signal) >= 70:
        return "text+image"

    return "text"


def choose_content_type(signal: dict, brand: str, pillar: str,
                        platform: str) -> Optional[str]:
    """Source-aware content type decision.

    Args:
        signal: The content signal (may be empty dict for static topics).
        brand, pillar, platform: Existing routing parameters.

    Returns:
        Content type string or None to use the static fallback.

    This is the bridge between editorial_router and the existing
    _choose_content_type in llm_drafts. When a real signal is available,
    the router determines the content type. When there is no signal
    (app brand static topics), returns None so the caller uses the
    hardcoded static map.
    """
    if not signal:
        return None  # No signal context — use static fallback
    has_signal_context = bool(
        signal.get("signal_type")
        or (signal.get("topic_id") and signal.get("title_hint") and (
            signal.get("curated") is True
            or signal.get("evidence")
            or signal.get("source_notes")
            or signal.get("context")
            or signal.get("source_override")
        ))
    )
    if not has_signal_context:
        return None  # Static/generic topic — use static fallback

    # Map platform names to route_signal surface names
    PLATFORM_TO_SURFACE = {
        "twitter": "x_post",
        "linkedin": "linkedin_post",
        "blog": "blog",
        "article": "article",
    }
    surface = PLATFORM_TO_SURFACE.get(platform, f"{platform}_post")
    result = route_signal(signal, candidate_surfaces=[surface])
    if result["decision"] == "skip":
        return None
    return result["content_type"]


# ── Public API ──


def route_signal(signal: dict,
                 candidate_surfaces: Optional[list[str]] = None) -> dict:
    """Route a content signal to the best platform.

    Pure decision function: no side effects, no DB writes, no generation.

    Args:
        signal: Content signal dict (activity_collector, topic bank,
                blog queue, screenshot).
        candidate_surfaces: Allowed platforms. Defaults to all.

    Returns:
        Decision dict with keys: decision, platform, content_type,
        scores, rationale, source_provenance.
    """
    if candidate_surfaces is None:
        candidate_surfaces = ["x_post", "linkedin_post", "blog", "article"]

    # Build provenance
    provenance = _build_provenance(signal)

    # Score all surfaces
    scores = {
        "x_fit": round(_score_x_fit(signal), 1),
        "linkedin_fit": round(_score_linkedin_fit(signal), 1),
        "blog_fit": round(_score_blog_fit(signal), 1),
        "article_fit": round(_score_article_fit(signal), 1),
        "visual_fit": round(_score_visual_fit(signal), 1),
        "source_confidence": round(_source_confidence(signal), 1),
        "proof_strength": round(_proof_strength(signal), 1),
        "freshness": round(_freshness_score(signal), 1),
    }

    # ── Decision logic ──

    # Surface-to-platform mapping
    SURFACE_PLATFORM = {
        "x_post": "twitter",
        "linkedin_post": "linkedin",
        "blog": "blog",
        "article": "article",
    }

    # Priority order: deeper surfaces first (article > blog > linkedin > x)
    priority_order = ["article", "blog", "linkedin_post", "x_post"]

    # Check skip for each surface
    all_skipped = True
    skip_reasons = []
    candidates = []

    for surf in priority_order:
        if surf not in candidate_surfaces:
            continue
        plat = SURFACE_PLATFORM[surf]
        skip, reason = _check_skip_for(signal, plat)
        if skip:
            skip_reasons.append(f"{plat}: {reason}")
            continue
        all_skipped = False

        score_map = {
            "x_post": scores["x_fit"],
            "linkedin_post": scores["linkedin_fit"],
            "blog": scores["blog_fit"],
            "article": scores["article_fit"],
        }
        threshold = SCORE_THRESHOLDS[surf]

        if score_map[surf] >= threshold:
            candidates.append((score_map[surf], surf, plat, reason))

    # Log skip results for all surfaces if everything skipped
    if all_skipped or not candidates:
        if skip_reasons and all_skipped:
            rationale = "All platforms skipped: " + " | ".join(skip_reasons)
        elif not candidates:
            # No surface meets its threshold
            best = max((s for s in priority_order if s in candidate_surfaces),
                       key=lambda s: {"x_post": scores["x_fit"],
                                      "linkedin_post": scores["linkedin_fit"],
                                      "blog": scores["blog_fit"],
                                      "article": scores["article_fit"]}[s],
                       default="x_post")
            score_key = {
                "x_post": "x_fit",
                "linkedin_post": "linkedin_fit",
                "blog": "blog_fit",
                "article": "article_fit",
            }.get(best, "x_fit")
            rationale = (f"Best surface '{best}' below threshold "
                         f"({scores.get(score_key, 0):.0f})")
        else:
            rationale = " | ".join(skip_reasons)

        return {
            "decision": "skip",
            "platform": "",
            "content_type": "text",
            "scores": scores,
            "rationale": rationale,
            "source_provenance": provenance,
        }

    # Pick highest-scoring candidate
    candidates.sort(key=lambda x: (-x[0], x[1]))
    winner_score, winner_surface, winner_plat, _ = candidates[0]

    content_type = _recommend_content_type(signal, winner_plat)

    # Build rationale
    reasons = []
    t = signal.get("signal_type", "manual")
    if t in X_NATIVE_SIGNALS:
        reasons.append(f"Signal '{t}' is native X content")
    if t in LINKEDIN_NATIVE_SIGNALS:
        reasons.append(f"Signal '{t}' fits LinkedIn angle")
    if scores["freshness"] >= 80:
        reasons.append("Fresh signal")
    if scores["proof_strength"] >= 80:
        reasons.append("Verified source")
    if scores["visual_fit"] >= 70:
        reasons.append("Has visual content")
    if _has_reader_lesson(_summary_text(signal)):
        reasons.append("Has business/reader lesson")

    label_map = {
        "x_post": "X post",
        "linkedin_post": "LinkedIn post",
        "blog": "SahilBlog",
        "article": "X Article",
    }
    label = label_map.get(winner_surface, winner_surface)
    rationale = f"{label} (score={winner_score:.0f}): " + "; ".join(reasons) if reasons else label

    return {
        "decision": winner_surface,
        "platform": winner_plat,
        "content_type": content_type,
        "scores": scores,
        "rationale": rationale,
        "source_provenance": provenance,
    }
