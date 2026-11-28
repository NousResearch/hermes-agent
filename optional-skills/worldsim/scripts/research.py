"""
Hermes Simulator — Intelligence Gathering Pipeline v2

Full-spectrum OSINT research engine for personality modeling.
Searches text, extracts content, browses live pages, analyzes
images with vision, and cross-references across platforms.

Run via execute_code. The agent adapts searches based on findings.
"""

from hermes_tools import web_search, web_extract, terminal
import json
import time
import urllib.parse

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

AGGREGATOR_SITES = [
    "buttondown.com/ainews",
    "news.smol.ai",
    "techmeme.com",
    "latent.space",
]

# Verified working fallback data sources (tested April 2026)
# Priority order: X API > nitter.cz > ThreadReaderApp > GitHub > Reddit > HN
FALLBACK_SOURCES = {
    "nitter": "https://nitter.cz/{handle}",           # web_extract — full timeline
    "threadreader": "https://threadreaderapp.com/user/{handle}",  # web_extract — historical threads
    "github_profile": "https://api.github.com/users/{handle}",   # curl — profile + README
    "github_events": "https://api.github.com/users/{handle}/events",  # curl — recent activity
    "reddit_user": "https://www.reddit.com/user/{handle}.json",  # curl w/ User-Agent
    "reddit_comments": "https://www.reddit.com/user/{handle}/comments.json",
    "hn_search": "https://hn.algolia.com/api/v1/search?query={handle}&tags=comment",
}

# CONFIRMED BLOCKED (don't waste calls on these):
# - LinkedIn (web_extract blocked, browser auth wall)
# - Instagram viewers (imginn, picuki, dumpoir, gramhir — all 403)
# - Most nitter instances (dead or 403, ONLY nitter.cz works via web_extract)
# - Wayback Machine for tweets (sparse, no JS content)
# - Google Cache of Twitter (empty)
# - Archive.today (429 + CAPTCHA)
# - Twitter Syndication API (rate limited)

AI_SUBREDDITS = [
    "LocalLLaMA", "MachineLearning", "singularity",
    "ChatGPT", "ClaudeAI", "OpenAI", "StableDiffusion",
]

PLATFORMS = ["twitter", "instagram", "linkedin", "github", "reddit", "youtube"]

# ═══════════════════════════════════════════════════════════════
# HELPER: safe web_search with validation
# ═══════════════════════════════════════════════════════════════

def _safe_web_search(query: str, limit: int = 5) -> list:
    """Run web_search and return results list, with validation."""
    r = web_search(query, limit=limit)
    if not isinstance(r, dict) or "data" not in r:
        print(f"  [WARNING] web_search returned no 'data' key for query: {query[:80]}")
        return []
    data = r.get("data", {})
    if not isinstance(data, dict):
        return []
    return data.get("web", []) or []


# ═══════════════════════════════════════════════════════════════
# CORE SEARCH FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def search_identity(handle: str) -> dict:
    """Establish who they are across the internet."""
    results = {}
    results["twitter_identity"] = _safe_web_search(f"@{handle} twitter bio role company", limit=5)
    results["general_identity"] = _safe_web_search(f"{handle} known for", limit=5)
    return results


def search_voice(handle: str) -> dict:
    """How do they actually talk/write."""
    results = {}
    results["takes"] = _safe_web_search(f"{handle} twitter hot takes opinions", limit=5)

    for agg in AGGREGATOR_SITES[:2]:
        hits = _safe_web_search(f"site:{agg} {handle}", limit=3)
        if hits:
            # Use full domain as key, not split('.')[0]
            results[f"agg_{agg}"] = hits
    return results


def search_positions(handle: str, topics: list = None, domain: str = None) -> dict:
    """What are their known positions."""
    results = {}
    if topics:
        for topic in topics[:3]:
            results[f"topic_{topic}"] = _safe_web_search(f"{handle} {topic} opinion take", limit=5)

    # Build controversy query — only add domain keywords if specified
    controversy_query = f"{handle} debate disagree controversial"
    if domain:
        controversy_query += f" {domain}"
    results["controversies"] = _safe_web_search(controversy_query, limit=5)
    return results


def search_longform(handle: str, real_name: str = None, domain: str = None) -> dict:
    """Blogs, interviews, essays."""
    results = {}
    name = real_name or handle

    blog_query = f"{name} blog substack essay"
    interview_query = f"{name} interview podcast"
    if domain:
        blog_query += f" {domain}"
        interview_query += f" {domain}"

    results["blogs"] = _safe_web_search(blog_query, limit=5)
    results["interviews"] = _safe_web_search(interview_query, limit=5)
    return results


# ═══════════════════════════════════════════════════════════════
# CROSS-PLATFORM DISCOVERY
# ═══════════════════════════════════════════════════════════════

def discover_platforms(handle: str, real_name: str = None) -> dict:
    """Find someone across all platforms."""
    name = real_name or handle
    results = {}

    # Instagram
    results["instagram"] = _safe_web_search(f"{name} instagram OR site:instagram.com/{handle}", limit=5)

    # LinkedIn
    results["linkedin"] = _safe_web_search(f"{name} linkedin OR site:linkedin.com/in", limit=5)

    # Reddit
    results["reddit"] = _safe_web_search(f"{name} reddit account OR site:reddit.com/user", limit=5)

    # GitHub
    results["github"] = _safe_web_search(f"{handle} github OR site:github.com/{handle}", limit=5)

    # YouTube
    results["youtube"] = _safe_web_search(f"{name} youtube channel OR talk OR interview", limit=5)

    # Personal site
    results["personal_site"] = _safe_web_search(f"{name} personal website blog about", limit=5)

    # Hacker News
    results["hackernews"] = _safe_web_search(f"site:news.ycombinator.com {handle} OR {name}", limit=3)

    return results


def discover_instagram(handle: str = None, real_name: str = None) -> dict:
    """Focused Instagram discovery."""
    results = {}
    name = real_name or handle

    # Try to find their IG handle
    results["ig_search"] = _safe_web_search(f"{name} instagram profile", limit=5)

    # If we have a candidate IG URL, try to extract
    ig_urls = []
    for item in results.get("ig_search", []):
        if not isinstance(item, dict):
            continue
        url = item.get("url", "")
        if "instagram.com/" in url and "/p/" not in url:
            ig_urls.append(url)

    if ig_urls:
        # Try to extract IG profile page
        r = web_extract(urls=ig_urls[:1])
        results["ig_profile"] = r.get("results", [])

    return results


# ═══════════════════════════════════════════════════════════════
# VISUAL INTELLIGENCE
# ═══════════════════════════════════════════════════════════════

# NOTE: These functions use browser_* and vision_analyze which are
# NOT available in execute_code. They are called DIRECTLY by the
# agent after the execute_code research phase.
#
# The agent should:
# 1. Run this script via execute_code for text-based research
# 2. Then use browser/vision tools directly for visual research
#
# Visual research tasks for the agent:
#
# INSTAGRAM VISUAL:
#   browser_navigate("https://www.instagram.com/{ig_handle}/")
#   browser_vision(question="Describe this Instagram profile: bio, pic, grid, aesthetic, follower count")
#   browser_get_images()  # collect image URLs
#   vision_analyze(image_url="{url}", question="Describe: setting, people, mood, style")
#
# PROFILE PIC ANALYSIS:
#   vision_analyze(image_url="{pic_url}", question="Describe: appearance, clothing, setting, expression, professional vs casual")
#
# REVERSE IMAGE SEARCH (Yandex):
#   # Upload to catbox if behind auth:
#   terminal("curl -F 'reqtype=fileupload' -F 'fileToUpload=@{path}' https://catbox.moe/user/api.php")
#   browser_navigate(f"https://yandex.com/images/search?rpt=imageview&url={encoded_url}")
#
# PAGE SCREENSHOT ANALYSIS:
#   browser_vision(question="Read all text, usernames, post content, dates, engagement numbers")


# ═══════════════════════════════════════════════════════════════
# INTERACTION MAPPING
# ═══════════════════════════════════════════════════════════════

def search_interactions(handle: str, other_handles: list = None) -> dict:
    """How they interact with other simulation targets."""
    results = {}
    if other_handles:
        for other in other_handles[:4]:
            hits = _safe_web_search(f"{handle} {other} twitter interaction debate reply", limit=3)
            if hits:
                results[f"with_{other}"] = hits
    return results


def search_social_graph(handle: str) -> dict:
    """Who do they interact with most? Allies and rivals."""
    results = {}

    results["frequent_interactions"] = _safe_web_search(f"@{handle} twitter reply thread conversation with", limit=5)
    results["conflicts"] = _safe_web_search(f"@{handle} disagree argue beef ratio", limit=5)
    results["allies"] = _safe_web_search(f"@{handle} agree support endorse recommend", limit=5)

    return results


# ═══════════════════════════════════════════════════════════════
# DEEP EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_content(urls: list) -> list:
    """Pull full content from high-value URLs."""
    if not urls:
        return []
    r = web_extract(urls=urls[:3])
    return r.get("results", [])


def extract_best_urls(findings: dict, max_urls: int = 5) -> list:
    """Find the most promising URLs in research findings for deep extraction."""
    seen_urls = set()  # URL deduplication
    priority_domains = [
        "substack.com", "medium.com", "blog", "essay",
        "interview", "podcast", "youtube.com", "arxiv.org",
    ]

    def score_url(url, desc):
        score = 0
        for domain in priority_domains:
            if domain in url.lower() or domain in desc.lower():
                score += 2
        if any(w in desc.lower() for w in ["interview", "spoke", "told", "said", "wrote"]):
            score += 1
        return score

    candidates = []

    def collect(obj):
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    url = item.get("url") or ""
                    desc = item.get("description") or item.get("text") or ""
                    if url and url not in seen_urls and not any(x in url for x in ["x.com", "twitter.com", "instagram.com"]):
                        seen_urls.add(url)
                        candidates.append((score_url(url, desc), url))
        elif isinstance(obj, dict):
            for v in obj.values():
                collect(v)

    collect(findings)
    candidates.sort(key=lambda x: -x[0])
    return [url for _, url in candidates[:max_urls]]


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def research_person(handle: str, fidelity: int = 70,
                    topics: list = None,
                    other_handles: list = None,
                    real_name: str = None,
                    domain: str = None) -> dict:
    """
    Full research pipeline for one person.
    Returns dict with all findings organized by category.

    Args:
        handle: Twitter/X handle (without @)
        fidelity: Research depth 0-100
        topics: Specific topics to research
        other_handles: Other people to check interactions with
        real_name: Real name if different from handle
        domain: Domain context (e.g., 'AI', 'politics', 'gaming').
                When None, no domain keywords are added to searches.
                When set, adds relevant domain keywords.
    """
    print(f"\n{'='*60}")
    print(f"  RESEARCHING: @{handle} | Fidelity: {fidelity}%")
    if domain:
        print(f"  Domain: {domain}")
    print(f"{'='*60}")

    findings = {"handle": handle, "fidelity": fidelity, "visual_tasks": []}

    # ─── Phase 1: Identity (always) ───
    print(f"\n  [IDENTITY] Who are they...")
    findings["identity"] = search_identity(handle)

    if fidelity <= 30:
        if topics:
            findings["quick_topic"] = _safe_web_search(f"{handle} {topics[0]}", limit=3)
        return findings

    # ─── Phase 2: Voice (fidelity 31+) ───
    print(f"\n  [VOICE] How do they talk...")
    findings["voice"] = search_voice(handle)

    # ─── Phase 3: Positions (fidelity 31+) ───
    print(f"\n  [POSITIONS] What do they believe...")
    findings["positions"] = search_positions(handle, topics, domain=domain)

    if fidelity <= 50:
        return findings

    # ─── Phase 4: Cross-platform (fidelity 51+) ───
    print(f"\n  [PLATFORMS] Finding them everywhere...")
    findings["platforms"] = discover_platforms(handle, real_name)

    if fidelity <= 70:
        return findings

    # ─── Phase 5: Longform (fidelity 71+) ───
    print(f"\n  [LONGFORM] Blogs, interviews, essays...")
    findings["longform"] = search_longform(handle, real_name, domain=domain)

    # ─── Phase 6: Social graph (fidelity 71+) ───
    print(f"\n  [SOCIAL GRAPH] Who do they interact with...")
    findings["social_graph"] = search_social_graph(handle)

    # ─── Phase 7: Interaction mapping (fidelity 71+) ───
    if other_handles:
        print(f"\n  [INTERACTIONS] With other targets: {other_handles}...")
        findings["interactions"] = search_interactions(handle, other_handles)

    # ─── Phase 8: Instagram deep dive (fidelity 80+) ───
    if fidelity >= 80:
        print(f"\n  [INSTAGRAM] Visual identity...")
        findings["instagram"] = discover_instagram(handle, real_name)

        # Queue visual tasks for the agent to do after execute_code
        findings["visual_tasks"].append({
            "type": "instagram_profile",
            "instruction": f"browser_navigate to Instagram profile, use browser_vision to analyze",
            "handle": handle,
        })

    # ─── Phase 9: Deep extraction (fidelity 85+) ───
    if fidelity >= 85:
        print(f"\n  [DEEP EXTRACT] Pulling longform content...")
        best_urls = extract_best_urls(findings, max_urls=4)
        if best_urls:
            print(f"    Extracting {len(best_urls)} URLs: {best_urls}")
            findings["deep_extracts"] = extract_content(best_urls)

    # ─── Phase 10: Profile pic analysis (fidelity 90+) ───
    if fidelity >= 90:
        findings["visual_tasks"].append({
            "type": "profile_pic_analysis",
            "instruction": "Find and analyze profile pictures across platforms with vision_analyze",
            "handle": handle,
        })
        findings["visual_tasks"].append({
            "type": "reverse_image_search",
            "instruction": "Reverse image search profile pic via Yandex to find alt accounts",
            "handle": handle,
        })

    return findings


def research_all(handles: list, fidelity: int = 70,
                 topics: list = None, domain: str = None) -> dict:
    """Research all simulation targets."""
    all_findings = {}

    for handle in handles:
        clean = handle.lstrip("@")
        others = [h.lstrip("@") for h in handles if h.lstrip("@") != clean]

        findings = research_person(
            handle=clean,
            fidelity=fidelity,
            topics=topics,
            other_handles=others,
            domain=domain,
        )
        all_findings[clean] = findings

    return all_findings


# ═══════════════════════════════════════════════════════════════
# REPORTING
# ═══════════════════════════════════════════════════════════════

def count_data_points(obj) -> int:
    """Count total search result items in findings (only meaningful items with >50 char text)."""
    total = 0
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                text = item.get("description") or item.get("text") or ""
                if len(text) > 50:
                    total += 1
                else:
                    # Still count non-dict items or items without text fields
                    total += 1
            else:
                total += 1
    elif isinstance(obj, dict):
        for k, v in obj.items():
            # Skip metadata keys
            if k in ("handle", "fidelity", "visual_tasks"):
                continue
            total += count_data_points(v)
    return total


def count_quality_data_points(obj) -> int:
    """Count search result items with substantial text (description/text > 50 chars)."""
    total = 0
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                text = item.get("description") or item.get("text") or ""
                if len(text) > 50:
                    total += 1
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("handle", "fidelity", "visual_tasks"):
                continue
            total += count_quality_data_points(v)
    return total


def summarize_findings(findings: dict) -> str:
    """Compact summary of what we found."""
    handle = findings.get("handle", "unknown")
    fidelity = findings.get("fidelity", 0)
    total = count_data_points(findings)
    quality = count_quality_data_points(findings)
    visual_tasks = findings.get("visual_tasks", [])

    lines = [
        f"\n{'━'*60}",
        f"  @{handle} | Fidelity: {fidelity}% | Data points: {total} ({quality} quality)",
        f"{'━'*60}",
    ]

    # Identity snippets
    identity = findings.get("identity", {})
    for key in ["twitter_identity", "general_identity"]:
        for item in identity.get(key, [])[:2]:
            if not isinstance(item, dict):
                continue
            desc = (item.get("description") or "")[:180]
            if desc:
                lines.append(f"  [{key.upper()}] {desc}")

    # Platform discovery results
    platforms = findings.get("platforms", {})
    found_platforms = []
    for platform, items in platforms.items():
        if isinstance(items, list) and len(items) > 0:
            found_platforms.append(platform)
    if found_platforms:
        lines.append(f"  [PLATFORMS FOUND] {', '.join(found_platforms)}")

    # Voice samples from aggregators
    voice = findings.get("voice", {})
    for key, items in voice.items():
        if isinstance(items, list):
            for item in items[:1]:
                if not isinstance(item, dict):
                    continue
                desc = (item.get("description") or "")[:180]
                if desc and handle.lower() in desc.lower():
                    lines.append(f"  [VOICE] {desc}")

    # Deep extracts
    for extract in findings.get("deep_extracts", [])[:2]:
        if not isinstance(extract, dict):
            continue
        title = extract.get("title", "untitled")
        content = (extract.get("content") or "")[:200]
        if content:
            lines.append(f"  [LONGFORM: {title}] {content}...")

    # Pending visual tasks
    if visual_tasks:
        lines.append(f"  [VISUAL TASKS QUEUED] {len(visual_tasks)} tasks for agent to execute:")
        for task in visual_tasks:
            lines.append(f"    → {task.get('type', '?')}: {task.get('instruction', '?')[:80]}")

    # Confidence estimate — based on quality data points
    if quality >= 30:
        conf = "HIGH"
    elif quality >= 15:
        conf = "MEDIUM"
    elif quality >= 5:
        conf = "LOW"
    else:
        conf = "INSUFFICIENT"
    lines.append(f"\n  CONFIDENCE: {conf} ({quality} quality data points, {total} total)")

    return "\n".join(lines)


def report_visual_tasks(all_findings: dict) -> str:
    """Collect all visual tasks across all targets for agent to execute."""
    lines = ["\n" + "═"*60, "  VISUAL INTELLIGENCE TASKS (agent must execute directly)", "═"*60]

    any_tasks = False
    for handle, findings in all_findings.items():
        for task in findings.get("visual_tasks", []):
            any_tasks = True
            lines.append(f"\n  @{handle} — {task.get('type', '?')}:")
            lines.append(f"    {task.get('instruction', '?')}")

    if not any_tasks:
        lines.append("  No visual tasks queued (fidelity < 80)")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# CHECK AVAILABLE TOOLS
# ═══════════════════════════════════════════════════════════════

def check_x_cli() -> bool:
    """Check if x-cli is available."""
    try:
        r = terminal("which x-cli 2>/dev/null && echo 'FOUND' || echo 'NOT_FOUND'")
        return "FOUND" in r.get("output", "")
    except:
        return False


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── CONFIGURE THESE ──
    HANDLES = ["teknium1", "basedjensen"]
    FIDELITY = 80
    TOPICS = ["open source AI", "compute scaling"]
    DOMAIN = None  # Set to 'AI', 'politics', etc. to add domain keywords
    # ─────────────────────

    has_xcli = check_x_cli()
    print(f"x-cli available: {has_xcli}")
    print(f"Targets: {HANDLES}")
    print(f"Fidelity: {FIDELITY}%")
    print(f"Topics: {TOPICS}")
    print(f"Domain: {DOMAIN}")

    results = research_all(HANDLES, fidelity=FIDELITY, topics=TOPICS, domain=DOMAIN)

    for handle, findings in results.items():
        print(summarize_findings(findings))

    print(report_visual_tasks(results))
    print("\n\nResearch phase complete. Agent should now:")
    print("1. Execute any queued visual tasks (browser/vision)")
    print("2. Compile dossiers from all findings")
    print("3. Run simulation")
