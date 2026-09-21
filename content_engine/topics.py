"""Topic collector: football fixtures + brand topic banks + activity signals + temporal gates.

Key changes in v2.1:
- 30+ topics per brand (expanded 5x)
- last_used timestamp per topic entry
- get_topics skips topics used within 30 days
- fetch_fixtures injects real fixture data
- _load_activity has 7-day recency gate
- STALE_TECH_REFS auto-rejection gate
"""

import json
import os
import random
import re
import sys
import uuid
from datetime import UTC, datetime, timedelta
from typing import List, Dict, Any

# Strip git/commit artefacts so a raw commit subject never becomes a post topic
# or title (e.g. "feat(content): non-infographic scene transplant" leaking in).
_COMMIT_PREFIX = re.compile(
    r"^(?:feat|fix|chore|docs|refactor|style|test|perf|build|ci|revert|merge|wip|hotfix)"
    r"(?:\([^)]*\))?!?:\s*", re.I)
_TRAILER = re.compile(r"\s*(?:Co-Authored-By:|Signed-off-by:).*$", re.I | re.S)


def _clean_topic_summary(text: str) -> str:
    """Turn a raw signal summary (often a git commit subject) into clean topic
    text: drop conventional-commit prefixes, sign-off trailers, merge noise.
    Returns "" for pure git noise so the caller can fall back to the pillar."""
    s = (text or "").strip()
    s = _TRAILER.sub("", s).strip()
    for _ in range(2):  # handle a doubled prefix, e.g. "feat: fix: ..."
        stripped = _COMMIT_PREFIX.sub("", s).strip()
        if stripped == s:
            break
        s = stripped
    if re.match(r"^merge\b", s, re.I):  # bare merge commits carry no topic
        return ""
    return s

try:
    import requests
except ImportError:
    requests = None  # type: ignore

from config import FOOTBALL_API_BASE
from database import get_recently_used_topics, log_topic_usage

# ── Lazy-loaded educational components ──
_EDU_ENRICH = None
_EDU_KB = None

def _load_edu():
    global _EDU_ENRICH, _EDU_KB
    if _EDU_ENRICH is not None:
        return True
    try:
        import context_enrich as ce
        import kb_retrieve as kb
        _EDU_ENRICH = ce.enrich
        _EDU_KB = kb.retrieve
        return True
    except ImportError:
        return False

# ── Stale tech reference gate ──
STALE_TECH_REFS = [
    "GPT-4o", "gpt-4o", "ChatGPT 4o", "chatgpt 4o",
    "GPT-4", "gpt-4", "ChatGPT 4", "chatgpt 4",
    "GPT-3", "gpt-3", "ChatGPT 3", "chatgpt 3",
    "Claude 3", "claude 3", "Claude 3.5", "claude 3.5",
    "DALL-E 2", "dall-e 2", "Midjourney v5", "midjourney v5",
]

def has_stale_tech_reference(text: str) -> tuple[bool, str]:
    """Check if text contains outdated AI tech references. Returns (has_stale, first_match)."""
    for ref in STALE_TECH_REFS:
        if ref in text:
            return True, ref
    return False, ""

# Map signal types to educational pillars per brand
TWITTER_EDU_SIGNAL_MAP = {
    "github_push": "Agent Build Notes",
    "hermes_pr": "Agent Build Notes",
    "hermes_skill": "Agent Build Notes",
    "harness_change": "Harness Tuning",
    "research_tool": "Radar Finds",
    "research_signal": "Paper Takes",
    "gitradar_repo": "Radar Finds",
    "architecture": "AI Patterns",
}
LINKEDIN_EDU_SIGNAL_MAP = {
    "github_push": "AI Engineering Notes",
    "hermes_pr": "Agentic Systems in Practice",
    "hermes_skill": "Agentic Systems in Practice",
    "harness_change": "AI Engineering Notes",
    "research_tool": "Tooling Signals",
    "research_signal": "Research to Practice",
    "gitradar_repo": "Tooling Signals",
    "architecture": "AI Engineering Notes",
}

# ── Activity collector (lazy loaded) ──
_ACTIVITY_COLLECTOR = None
_ACTIVITY_MARKER = None

def _load_activity():
    global _ACTIVITY_COLLECTOR, _ACTIVITY_MARKER
    if _ACTIVITY_COLLECTOR is not None:
        return True
    try:
        from activity_collector import collect_all as _ca, mark_signals_used as _msu
        _ACTIVITY_COLLECTOR = _ca
        _ACTIVITY_MARKER = _msu
        return True
    except ImportError:
        return False

# ── Expanded topic banks (30+ per brand) ──

TOPIC_BANKS: Dict[str, List[Dict]] = {
    "matchdaymaestro": [
        {"pillar": "live_predictions", "topic": "Next weekend's big fixtures", "last_used": None},
        {"pillar": "game_modes", "topic": "Strike501 daily challenge", "last_used": None},
        {"pillar": "friend_battles", "topic": "1v1 battle invitation", "last_used": None},
        {"pillar": "progression", "topic": "RPG level-up milestone", "last_used": None},
        {"pillar": "football_beat", "topic": "Weekly stat mic-drop", "last_used": None},
        {"pillar": "live_predictions", "topic": "Derby day prediction prompt", "last_used": None},
        {"pillar": "game_modes", "topic": "PlayerCombination T9 puzzle reveal", "last_used": None},
        {"pillar": "friend_battles", "topic": "Mini-league standings update", "last_used": None},
        {"pillar": "live_predictions", "topic": "Champions League midweek fixtures", "last_used": None},
        {"pillar": "live_predictions", "topic": "Late-goal prediction window", "last_used": None},
        {"pillar": "game_modes", "topic": "Weekly quiz perfect score chase", "last_used": None},
        {"pillar": "game_modes", "topic": "Hard mode unlock milestone", "last_used": None},
        {"pillar": "friend_battles", "topic": "Group chat challenge thread", "last_used": None},
        {"pillar": "friend_battles", "topic": "Season reset new league code", "last_used": None},
        {"pillar": "progression", "topic": "Ace Striker tier unlock", "last_used": None},
        {"pillar": "progression", "topic": "Achievement hunting checklist", "last_used": None},
        {"pillar": "football_beat", "topic": "Top 6 home form analysis", "last_used": None},
        {"pillar": "football_beat", "topic": "Underdog upset pattern alert", "last_used": None},
        {"pillar": "football_beat", "topic": "Penalty stat of the week", "last_used": None},
        {"pillar": "football_beat", "topic": "Corner conversion rate watch", "last_used": None},
        {"pillar": "live_predictions", "topic": "Halftime reset strategy", "last_used": None},
        {"pillar": "game_modes", "topic": "Streak preservation tactics", "last_used": None},
        {"pillar": "friend_battles", "topic": "Head-to-head rivalry week", "last_used": None},
        {"pillar": "progression", "topic": "Season-end leaderboard push", "last_used": None},
        {"pillar": "football_beat", "topic": "Autopilot Markov chain insight", "last_used": None},
        {"pillar": "live_predictions", "topic": "Momentum shift detection", "last_used": None},
        {"pillar": "game_modes", "topic": "New game mode teaser", "last_used": None},
        {"pillar": "friend_battles", "topic": "Coin economy deep dive", "last_used": None},
        {"pillar": "progression", "topic": "Legend tier grind reality", "last_used": None},
        {"pillar": "football_beat", "topic": "Referee decision impact stat", "last_used": None},
    ],
    "plenishd": [
        {"pillar": "product", "topic": "UK supermarket price comparison", "last_used": None},
        {"pillar": "voice", "topic": "Voice-first inventory update", "last_used": None},
        {"pillar": "kitchen", "topic": "Leftover-to-recipe hack", "last_used": None},
        {"pillar": "shopping", "topic": "Smart shopping list build", "last_used": None},
        {"pillar": "launch", "topic": "Pre-launch sign-up", "last_used": None},
        {"pillar": "product", "topic": "Fridge-scan to meal plan", "last_used": None},
        {"pillar": "kitchen", "topic": "Waste-less Wednesday tip", "last_used": None},
        {"pillar": "product", "topic": "Multi-store split optimisation", "last_used": None},
        {"pillar": "voice", "topic": "Hands-free cooking command", "last_used": None},
        {"pillar": "kitchen", "topic": "Freezer archaeology expedition", "last_used": None},
        {"pillar": "shopping", "topic": "Clubcard vs Nectar vs Aldi", "last_used": None},
        {"pillar": "launch", "topic": "Beta tester onboarding story", "last_used": None},
        {"pillar": "product", "topic": "Price drop alert in action", "last_used": None},
        {"pillar": "voice", "topic": "Family shopping list chaos", "last_used": None},
        {"pillar": "kitchen", "topic": "Herb rescue mission", "last_used": None},
        {"pillar": "shopping", "topic": "Same basket cheaper split", "last_used": None},
        {"pillar": "launch", "topic": "Problem origin story", "last_used": None},
        {"pillar": "product", "topic": "Real-world product testing", "last_used": None},
        {"pillar": "voice", "topic": "Gloved-hands voice query", "last_used": None},
        {"pillar": "kitchen", "topic": "Sunday-night fridge panic", "last_used": None},
        {"pillar": "shopping", "topic": "Shared list activity feed", "last_used": None},
        {"pillar": "launch", "topic": "Shared shopping problem solved", "last_used": None},
        {"pillar": "product", "topic": "Product testing insights", "last_used": None},
        {"pillar": "voice", "topic": "Kids adding items by voice", "last_used": None},
        {"pillar": "kitchen", "topic": "Tupperware mystery safe zone", "last_used": None},
        {"pillar": "shopping", "topic": "Morrison's 7pm dash list", "last_used": None},
        {"pillar": "launch", "topic": "Real kitchen pain point story", "last_used": None},
        {"pillar": "product", "topic": "Stock-check feature walkthrough", "last_used": None},
        {"pillar": "voice", "topic": "Recipe suggestion from stock", "last_used": None},
        {"pillar": "kitchen", "topic": "Meal-plan vs reality gap", "last_used": None},
    ],
    "sahil_twitter": [
        {"pillar": "build_in_public", "topic": "Weekly indie dev update", "last_used": None},
        {"pillar": "ai_tools", "topic": "Claude Code workflow tip", "last_used": None},
        {"pillar": "sly_product", "topic": "Built-this-because problem", "last_used": None},
        {"pillar": "football", "topic": "Man United match reaction", "last_used": None},
        {"pillar": "wry", "topic": "AI industry observation", "last_used": None},
        {"pillar": "build_in_public", "topic": "App milestone screenshot", "last_used": None},
        {"pillar": "ai_tools", "topic": "Agentic loop pattern", "last_used": None},
        {"pillar": "tutorial", "topic": "How-to: content pipeline setup", "last_used": None},
        {"pillar": "tutorial", "topic": "Framework: feature decision guide", "last_used": None},
        {"pillar": "data", "topic": "Solo dev time breakdown", "last_used": None},
        {"pillar": "data", "topic": "Content output before vs after automation", "last_used": None},
        {"pillar": "promotion", "topic": "Plenishd problem-solution story", "last_used": None},
        {"pillar": "promotion", "topic": "MatchdayMaestro feature origin story", "last_used": None},
        {"pillar": "build_in_public", "topic": "Convex schema migration saga", "last_used": None},
        {"pillar": "build_in_public", "topic": "Feature kill decision diary", "last_used": None},
        {"pillar": "ai_tools", "topic": "Kimi K2.6 vs Qwen3 workflow", "last_used": None},
        {"pillar": "ai_tools", "topic": "MCP server integration pattern", "last_used": None},
        {"pillar": "tutorial", "topic": "Step-by-step: debugging in 5 min", "last_used": None},
        {"pillar": "tutorial", "topic": "RAG chatbot architecture guide", "last_used": None},
        {"pillar": "data", "topic": "4-app portfolio cost breakdown", "last_used": None},
        {"pillar": "data", "topic": "Reply vs like engagement ratio", "last_used": None},
        {"pillar": "promotion", "topic": "CoachOS parent visibility feature", "last_used": None},
        {"pillar": "promotion", "topic": "Kick-tionary shipped story", "last_used": None},
        {"pillar": "football", "topic": "United tactical shift watch", "last_used": None},
        {"pillar": "football", "topic": "Grassroots coaching Sunday diary", "last_used": None},
        {"pillar": "wry", "topic": "Vibe coding security reality", "last_used": None},
        {"pillar": "wry", "topic": "AI demo vs production gap", "last_used": None},
        {"pillar": "build_in_public", "topic": "Server midnight panic log", "last_used": None},
        {"pillar": "data", "topic": "Indie dev stack cost £0/month", "last_used": None},
        {"pillar": "tutorial", "topic": "Postiz direct-DB insert guide", "last_used": None},
    ],
    "sahil_linkedin": [
        {"pillar": "pm_thought", "topic": "Context over models thesis", "last_used": None},
        {"pillar": "indie", "topic": "PM to indie builder journey", "last_used": None},
        {"pillar": "ai", "topic": "Enterprise AI adoption pattern", "last_used": None},
        {"pillar": "leadership", "topic": "Team velocity insight", "last_used": None},
        {"pillar": "pm_thought", "topic": "RAG chatbot case study", "last_used": None},
        {"pillar": "indie", "topic": "Four apps lesson learn", "last_used": None},
        {"pillar": "ai", "topic": "Vibe coding production reality", "last_used": None},
        {"pillar": "pm_thought", "topic": "Requirements engineering before AI", "last_used": None},
        {"pillar": "pm_thought", "topic": "Killing a 90% done feature", "last_used": None},
        {"pillar": "indie", "topic": "Context switching as indie PM", "last_used": None},
        {"pillar": "indie", "topic": "Skills compounding across apps", "last_used": None},
        {"pillar": "ai", "topic": "Context quality vs volume debate", "last_used": None},
        {"pillar": "ai", "topic": "Security-first vibe coding", "last_used": None},
        {"pillar": "leadership", "topic": "Information management over backlog", "last_used": None},
        {"pillar": "leadership", "topic": "Senior PM constraint thinking", "last_used": None},
        {"pillar": "pm_thought", "topic": "Datadog State of AI Engineering", "last_used": None},
        {"pillar": "pm_thought", "topic": "Eval framework for AI features", "last_used": None},
        {"pillar": "indie", "topic": "Bedtime story generator product lesson", "last_used": None},
        {"pillar": "indie", "topic": "Portfolio strategy: ship one learn next", "last_used": None},
        {"pillar": "ai", "topic": "Model selection is not the bottleneck", "last_used": None},
        {"pillar": "ai", "topic": "Veracode AI-generated code audit", "last_used": None},
        {"pillar": "leadership", "topic": "Hiring manager 2026 PM criteria", "last_used": None},
        {"pillar": "leadership", "topic": "Best PMs manage information not product", "last_used": None},
        {"pillar": "pm_thought", "topic": "Demo to production six-week reality", "last_used": None},
        {"pillar": "pm_thought", "topic": "GitHub Copilot XSS vulnerability data", "last_used": None},
        {"pillar": "indie", "topic": "Solo dev CI/CD pipeline", "last_used": None},
        {"pillar": "ai", "topic": "OpenRouter vulnerability scan results", "last_used": None},
        {"pillar": "leadership", "topic": "AI adoption cultural stall points", "last_used": None},
        {"pillar": "leadership", "topic": "Metrics that actually matter", "last_used": None},
        {"pillar": "pm_thought", "topic": "Retrieval layer before prompt tuning", "last_used": None},
    ],
    "coachos": [
        {"pillar": "session_plan", "topic": "Grassroots drill of the week", "last_used": None},
        {"pillar": "coach_life", "topic": "Managing volunteers and parents", "last_used": None},
        {"pillar": "player_dev", "topic": "Age-appropriate tactics", "last_used": None},
        {"pillar": "community", "topic": "CoachOS feature walkthrough", "last_used": None},
        {"pillar": "session_plan", "topic": "Warm-up routine template", "last_used": None},
        {"pillar": "player_dev", "topic": "Tactics teaching for under-10s", "last_used": None},
        {"pillar": "session_plan", "topic": "60-minute session structure", "last_used": None},
        {"pillar": "session_plan", "topic": "Block periodisation primer", "last_used": None},
        {"pillar": "session_plan", "topic": "Progression-based planning", "last_used": None},
        {"pillar": "session_plan", "topic": "Post-session voice debrief", "last_used": None},
        {"pillar": "coach_life", "topic": "Clipboard upgrade story", "last_used": None},
        {"pillar": "coach_life", "topic": "Parent communication stream", "last_used": None},
        {"pillar": "coach_life", "topic": "End-of-season review evidence", "last_used": None},
        {"pillar": "coach_life", "topic": "Managing expectations on pitch", "last_used": None},
        {"pillar": "coach_life", "topic": "Kit bag admin reality", "last_used": None},
        {"pillar": "player_dev", "topic": "Individual baseline tracking", "last_used": None},
        {"pillar": "player_dev", "topic": "Progression over drills", "last_used": None},
        {"pillar": "player_dev", "topic": "Confidence-building conversation", "last_used": None},
        {"pillar": "player_dev", "topic": "Development timeline evidence", "last_used": None},
        {"pillar": "player_dev", "topic": "U7 vs U12 vs U16 approach", "last_used": None},
        {"pillar": "community", "topic": "Beta tester coach spotlight", "last_used": None},
        {"pillar": "community", "topic": "Beta tester coach story", "last_used": None},
        {"pillar": "community", "topic": "Coach stories and insights", "last_used": None},
        {"pillar": "community", "topic": "Pre-season planning window", "last_used": None},
        {"pillar": "community", "topic": "Built from real conversations", "last_used": None},
        {"pillar": "community", "topic": "Coach onboarding walkthrough", "last_used": None},
        {"pillar": "community", "topic": "August 2026 launch countdown", "last_used": None},
        {"pillar": "community", "topic": "Grassroots coaching challenges", "last_used": None},
        {"pillar": "community", "topic": "Community stories walkthrough", "last_used": None},
        {"pillar": "community", "topic": "Feedback loop iteration story", "last_used": None},
        {"pillar": "community", "topic": "Sunday morning toolkit", "last_used": None},
    ],
}

# Map signal types to content pillars for LinkedIn
LINKEDIN_SIGNAL_MAP = {
    "github_push": "indie",
    "hermes_pr": "indie",
    "hermes_skill": "ai",
    "research_tool": "ai",
    "research_signal": "pm_thought",
    "gitradar_repo": "ai",
    "architecture": "ai",
}

# ── Signal recency: reject signals older than 7 days ──
SIGNAL_MAX_AGE_DAYS = 7

def _signal_to_topic(signal: dict, platform: str) -> Dict[str, Any]:
    """Convert an activity signal into a topic dict with activity_data metadata.
    Rejects stale signals (older than 7 days).
    """
    variables = signal["variables"]
    signal_type = signal["signal_type"]

    # Recency gate
    signal_ts = signal.get("timestamp", "")
    if signal_ts:
        try:
            ts = datetime.fromisoformat(signal_ts.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            age = (datetime.now(UTC) - ts).days
            if age > SIGNAL_MAX_AGE_DAYS:
                return {}  # Empty = will be filtered out
        except ValueError:
            pass

    if signal_type == "github_push":
        topic = f"Just pushed {variables['repo_name']}"
    elif signal_type == "hermes_pr":
        topic = f"Submitted: {_clean_topic_summary(variables['pr_title']) or variables['pr_title']}"
    elif signal_type == "hermes_skill":
        topic = f"New skill: {variables['skill_name']}"
    elif signal_type == "research_tool":
        topic = f"Tool: {variables['title']}"
    elif signal_type == "research_signal":
        topic = f"Signal: {variables['title']}"
    elif signal_type == "gitradar_repo":
        topic = f"Radar: {variables['repo_name']}"
    elif signal_type == "architecture":
        topic = f"Arch: {variables['topic_label']}"
    else:
        topic = f"Update: {signal_type}"

    if platform == "linkedin":
        pillar = LINKEDIN_SIGNAL_MAP.get(signal_type, "indie")
    else:
        pillar = signal.get("pillar", "build_in_public")

    return {
        "id": str(uuid.uuid4())[:8],
        "pillar": pillar,
        "topic": topic,
        "activity_data": {
            "signal_type": signal_type,
            "variables": variables,
            "signal_id": signal["signal_id"],
        },
        "source_provenance": {
            "source_type": signal_type,
            "source_id": signal.get("signal_id", ""),
            "source_url": variables.get("url", ""),
            "source_summary": topic,
            "evidence": [topic],
            "confidence": "verified",
        },
        "_routing": signal.get("_routing"),
    }


def _screenshot_topics(brand: str, platform: str, max_n: int) -> List[Dict]:
    """Ingest dropped screenshots into captioned, high-priority topics (feature 1).

    Reads CONTENT_SCREENSHOT_INBOX, captions each image via free Gemini vision,
    and moves the file to a processed/ subdir so it is ingested exactly once.
    Returns [] when vision is unavailable, the inbox is empty, or anything fails.
    """
    if max_n <= 0:
        return []
    try:
        import config as cfg
        if not cfg.GEMINI_VISION_ENABLED:
            return []
        import gemini_vision
        if not gemini_vision.available():
            return []
        inbox = cfg.CONTENT_SCREENSHOT_INBOX
        if not os.path.isdir(inbox):
            return []
        exts = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".heic")
        files = sorted(
            os.path.join(inbox, f) for f in os.listdir(inbox)
            if f.lower().endswith(exts) and os.path.isfile(os.path.join(inbox, f))
        )[:max_n]
        if not files:
            return []
        processed = os.path.join(inbox, "processed")
        os.makedirs(processed, exist_ok=True)
        out: List[Dict] = []
        for path in files:
            topic = gemini_vision.describe_screenshot(path, brand)
            # Move out of the inbox regardless of usability so it is not retried.
            dest = os.path.join(processed, os.path.basename(path))
            try:
                os.replace(path, dest)
            except OSError:
                pass
            if topic:
                topic["id"] = str(uuid.uuid4())[:8]
                topic["screenshot_path"] = dest
                out.append(topic)
        return out
    except Exception as e:  # noqa: BLE001 (ingestion must degrade, not crash)
        print(f"[topics] screenshot ingestion failed: {e}", file=sys.stderr)
        return []


def get_topics(brand: str, count: int = 6, skip_used: bool = True) -> List[Dict]:
    """Return N topic objects, mixing real activity for personal brands with static banks.

    If skip_used=True, topics used within the last 30 days are excluded.
    """

    # ── Personal brands: screenshot ingestion + real activity + static ──
    if brand in ("sahil_twitter", "sahil_linkedin"):
        platform = "twitter" if brand == "sahil_twitter" else "linkedin"
        from config import SCREENSHOT_MAX_PER_RUN
        shot_topics = _screenshot_topics(brand, platform, min(SCREENSHOT_MAX_PER_RUN, count))
    else:
        shot_topics = []

    if brand in ("sahil_twitter", "sahil_linkedin") and (_load_activity() or shot_topics):
        platform = "twitter" if brand == "sahil_twitter" else "linkedin"
        try:
            if _ACTIVITY_COLLECTOR is None:
                raise RuntimeError("activity collector unavailable")
            result = _ACTIVITY_COLLECTOR()
            signals = result["signals"]
            state = result["state"]
        except Exception as e:
            print(f"[topics] activity_collector failed: {e}", file=sys.stderr)
            signals = []
            state = {"used_signals": []}

        # ── Editorial router: filter signals by platform fit ──
        if signals:
            try:
                from editorial_router import route_signal
                filtered = []
                for sig in signals:
                    routing = route_signal(sig)
                    if routing["decision"] == "skip":
                        continue
                    routed_platform = routing.get("platform", "")
                    # Accept if platform matches OR routing says a different
                    # high-fit platform (the signal is useful elsewhere).
                    # For personal brands, keep signals that route to the
                    # requested platform.
                    if routed_platform == platform:
                        sig["_routing"] = routing
                        filtered.append(sig)
                    elif routing["decision"] in ("blog", "article"):
                        # Long-form eligible — keep with lower priority
                        sig["priority"] = max(1, sig.get("priority", 5) - 2)
                        sig["_routing"] = routing
                        filtered.append(sig)
                    elif routing["scores"]["x_fit"] < 40 and routing["scores"]["linkedin_fit"] < 40:
                        # Very low fit overall — skip
                        continue
                    else:
                        # Moderate fit but different platform — keep with lower priority
                        sig["priority"] = max(1, sig.get("priority", 5) - 1)
                        sig["_routing"] = routing
                        filtered.append(sig)
                signals = filtered
            except Exception as e:
                print(f"[topics] editorial_router failed: {e}", file=sys.stderr)

        topics: List[Dict] = list(shot_topics)  # screenshots take priority
        slots = max(0, count - len(topics))     # capacity left for activity/static
        used_ids: List[str] = []
        brand_config = None
        try:
            from config import BRANDS
            brand_config = BRANDS.get(brand, {})
        except Exception:
            pass
        edu_mix = brand_config.get("educational_mix", 0.65) if brand_config else 0.65
        edu_count = int(slots * edu_mix) if signals else 0
        edu_signal_map = TWITTER_EDU_SIGNAL_MAP if brand == "sahil_twitter" else LINKEDIN_EDU_SIGNAL_MAP

        # Convert educational signals first (enriched with context + KB)
        edu_signals = signals[:edu_count]
        if edu_signals and _load_edu():
            import context_enrich
            import kb_retrieve
            for signal in edu_signals:
                stype = signal.get("signal_type", "")
                pillar = edu_signal_map.get(stype, "AI Patterns")
                summary = (signal.get("summary")
                           or signal.get("variables", {}).get("summary", "")
                           or signal.get("variables", {}).get("repo_name", "")
                           or signal.get("variables", {}).get("title", ""))
                summary = _clean_topic_summary(summary)
                topic_text = f"{pillar}: {summary}" if summary else pillar
                try:
                    context = context_enrich.enrich(signal)
                except Exception:
                    context = ""
                try:
                    kb = kb_retrieve.retrieve(summary, limit=2)
                except Exception:
                    kb = []
                topics.append({
                    "id": str(uuid.uuid4())[:8],
                    "pillar": pillar,
                    "topic": topic_text,
                    "educational": True,
                    "context": context,
                    "kb_snippets": kb,
                    "_routing": signal.get("_routing"),
                })
                used_ids.append(signal["signal_id"])

        # Remaining signals use existing non-educational path
        remaining_signals = signals[edu_count:slots]
        for signal in remaining_signals:
            topic = _signal_to_topic(signal, platform)
            if topic:
                topics.append(topic)
                used_ids.append(signal["signal_id"])

        if used_ids and state.get("used_signals") is not None and _ACTIVITY_MARKER is not None:
            try:
                _ACTIVITY_MARKER(state, used_ids)
            except Exception as e:
                print(f"[topics] mark_signals_used failed: {e}", file=sys.stderr)

        # ── sahil_twitter anti-scripted guardrail ──
        # No static topic bank fallback. Sahil personal X drafts MUST start
        # from a current internal activity/live signal or manually supplied
        # source. No source → no personal X draft.
        if brand == "sahil_twitter":
            return topics

        # Fill remaining slots with static topics (respecting 30-day skip)
        if len(topics) < count:
            bank = TOPIC_BANKS.get(brand, [])
            if skip_used:
                recent_ids = get_recently_used_topics(brand, days=30)
            else:
                recent_ids = []
            available = [t for t in bank if t.get("id") not in recent_ids]
            if not available:
                available = bank  # Fallback if all used
            needed = count - len(topics)
            for t in random.sample(available, min(needed, len(available))):
                tid = str(uuid.uuid4())[:8]
                topics.append({"id": tid, **t})
                # Mark as used in DB
                log_topic_usage(tid, brand, t["topic"], platform)

        return topics

    # ── MatchdayMaestro: football fixtures + static topics ──
    if brand == "matchdaymaestro":
        topics = []
        fixtures = fetch_fixtures()
        if fixtures:
            for f in fixtures[:count]:
                topic_id = f"fx_{f['home']}_{f['away']}_{f['date']}"
                # Only include future fixtures (temporal gate)
                try:
                    match_date = datetime.strptime(f["date"], "%Y-%m-%d").date()
                    if match_date < datetime.now(UTC).date():
                        continue  # Skip past fixtures
                except ValueError:
                    pass
                topics.append({
                    "id": topic_id,
                    "pillar": "live_predictions",
                    "topic": f"{f['home']} vs {f['away']} -- {f['date']}",
                    "fixture": f,
                })
        # Fill remaining slots with static topics (skip used)
        if len(topics) < count:
            needed = count - len(topics)
            bank = TOPIC_BANKS.get(brand, [])
            if skip_used:
                recent_ids = get_recently_used_topics(brand, days=30)
                available = [t for t in bank if t.get("id") not in recent_ids]
                if not available:
                    available = bank
            else:
                available = bank
            for t in random.sample(available, min(needed, len(available))):
                tid = t.get("id") or str(uuid.uuid4())[:8]
                topics.append({"id": tid, **t})
                log_topic_usage(tid, brand, t["topic"], "twitter")
        return topics

    # ── Other brands: pure static topics with recency skip ──
    # sahil_twitter anti-scripted guardrail: no static fallback here either.
    if brand == "sahil_twitter":
        return []

    topics = []
    bank = TOPIC_BANKS.get(brand, [])
    if bank:
        if skip_used:
            recent_ids = get_recently_used_topics(brand, days=30)
            available = [t for t in bank if t.get("id") not in recent_ids]
            if not available:
                available = bank
        else:
            available = bank
        for t in random.sample(available, min(count, len(available))):
            tid = t.get("id") or str(uuid.uuid4())[:8]
            topics.append({"id": tid, **t})
            log_topic_usage(tid, brand, t["topic"], "")
    return topics


def fetch_fixtures() -> List[Dict]:
    """Try free football-data.org for Premier League next matches.

    Returns [] on any failure; logs the failure mode to stderr.
    """
    if requests is None:
        print("fetch_fixtures: requests not installed", file=sys.stderr)
        return []
    try:
        # All upcoming scheduled matches, soonest first. The old query pinned
        # matchday=38, which is empty or wrong-season for most of the year.
        url = f"{FOOTBALL_API_BASE}/competitions/PL/matches?status=SCHEDULED"
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            print(f"fetch_fixtures: HTTP {resp.status_code}", file=sys.stderr)
            return []
        data = resp.json()
        matches = []
        upcoming = sorted(data.get("matches", []), key=lambda m: m.get("utcDate", ""))
        for m in upcoming[:6]:
            matches.append({
                "home": m["homeTeam"]["shortName"],
                "away": m["awayTeam"]["shortName"],
                "date": m["utcDate"][:10],
                "time": m["utcDate"][11:16],
            })
        return matches
    except (requests.RequestException, ValueError, KeyError) as exc:
        print(f"fetch_fixtures: {type(exc).__name__}: {exc}", file=sys.stderr)
        return []
