#!/usr/bin/env python3
"""

# P13: disabled-staging guard — exit early when cron is disabled
import os as _os, sys as _sys
if _os.environ.get("DRY_RUN") == "1":
    print(f"[DRY_RUN] {_os.path.basename(__file__)}")
    _sys.exit(0)
GitHub Radar — Automated Repo Discovery Pipeline (v4.1.0)
Dual-mode: daily (lightweight new finds) + weekly (full re-evaluation).

Daily mode (default, `--mode daily`):
- Queries GitHub Search API for repos pushed in last 7 days
- Deduplicates against cache (14-day TTL)
- Scrapes GitHub Trending (daily + weekly) as a primary source
- Outputs: "X new finds (Y cached, Z in scope)"

Weekly mode (`--mode weekly`):
- Full re-scan without cache dedup
- Compares each found repo's pushed_at against cached last_pushed
- Flags repos with new activity since last weekly check
- Outputs: "X repos re-evaluated, Y with new activity, Z new finds"
- Refreshes cache with updated pushed_at timestamps

v4.1.0 — Dual-mode, pushed_at tracking, trending as primary source.
"""

import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request
import urllib.parse
from collections import defaultdict
from datetime import datetime, timezone, timedelta

# ── Mode ─────────────────────────────────────────────────────────────

MODE = "daily"  # default, overridden by --mode weekly
if "--mode" in sys.argv:
    idx = sys.argv.index("--mode")
    if idx + 1 < len(sys.argv):
        MODE = sys.argv[idx + 1]
if "--help" in sys.argv or "-h" in sys.argv:
    print(f"Usage: {sys.argv[0]} [--mode daily|weekly]")
    print("  --mode daily   (default) Lightweight new-find scan with cache dedup")
    print("  --mode weekly  Full re-evaluation: re-scans cached repos and flags new activity")
    sys.exit(0)

# ── Data Paths ──────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "github-radar")
CACHE_FILE = os.path.join(DATA_DIR, "cache.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "discoveries.json")
METRICS_FILE = os.path.join(DATA_DIR, "metrics.json")
THRESHOLDS_FILE = os.path.join(DATA_DIR, "thresholds.json")
CACHE_STATS_FILE = os.path.join(DATA_DIR, "cache-stats.json")

DATE_STR = datetime.now(timezone.utc).strftime("%Y-%m-%d")
TIME_STR = datetime.now(timezone.utc).strftime("%H%M")
DAY_LABEL = datetime.now(timezone.utc).strftime("%d/%m/%Y")
TODAY_ISO = datetime.now(timezone.utc).strftime("%Y-%m-%d")

RUNBOOKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runbooks", "github-radar")
RUNBOOKS_TODAY = os.path.join(RUNBOOKS_DIR, DATE_STR)
HTML_FILE = os.path.join(RUNBOOKS_TODAY, f"github-radar-{DATE_STR}-{TIME_STR}.html")
REPOS_TXT_FILE = os.path.join(RUNBOOKS_TODAY, "github-radar-repos.txt")

# ── Static Config ───────────────────────────────────────────────────

RECENCY_DAYS = 7
CREATED_RECENCY_DAYS = 90
MAX_RESULTS_PER_QUERY = 100
MAX_PAGES = 10

# Trending: now scrapes BOTH daily and weekly as primary sources
TRENDING_URLS = [
    ("daily", "https://github.com/trending?since=daily"),
    ("weekly", "https://github.com/trending?since=weekly"),
]

QUERY_TEMPLATES = [
    {"q": "stars:>{stars}", "sort": "stars", "order": "desc", "star_base": 100},
    {"q": "language:python stars:>{stars}", "sort": "stars", "order": "desc", "star_base": 50},
    {"q": "language:typescript stars:>{stars}", "sort": "stars", "order": "desc", "star_base": 50},
    {"q": "language:go stars:>{stars}", "sort": "stars", "order": "desc", "star_base": 50},
    {"q": "language:rust stars:>{stars}", "sort": "stars", "order": "desc", "star_base": 50},
    {"q": "topic:mcp stars:>{stars}", "sort": "stars", "order": "desc", "star_base": 10},
    {"q": "topic:agent-framework stars:>{stars}", "sort": "stars", "order": "desc", "star_base": 10},
    {"q": "topic:developer-tools stars:>{stars}", "sort": "stars", "order": "desc", "star_base": 20},
    {"q": "topic:hermes-plugin stars:>{stars}", "sort": "stars", "order": "desc", "star_base": 5},
    # NEW: stack-specific queries for Sahil's portfolio
    {"q": "topic:react-native stars:>{stars}", "sort": "stars", "order": "desc", "star_base": 20},
    {"q": "topic:flutter stars:>{stars}", "sort": "stars", "order": "desc", "star_base": 20},
    {"q": "topic:voice-assistant stars:>{stars}", "sort": "stars", "order": "desc", "star_base": 10},
    {"q": "topic:sports-prediction stars:>{stars}", "sort": "stars", "order": "desc", "star_base": 5},
]

RELEVANCE_HIGH = [
    "mcp server", "mcp", "agent", "agentic", "claude", "anthropic", "llm",
    "react native", "expo", "convex", "supabase", "hermes", "deepgram",
    "orchestration", "rag", "memory", "skill", "plugin",
]
RELEVANCE_GENERIC = [
    "python", "typescript", "rust", "react", "flutter", "cli", "tui",
    "terminal", "voice", "tts", "stt", "proxy", "filter",
    "kitchen", "grocery", "pantry", "inventory", "e-ink",
    "football", "soccer", "sports", "grassroots", "matchday",
    "prediction", "cv builder", "portfolio",
]
RELEVANCE_HIGH_WEIGHT = 20
RELEVANCE_GENERIC_WEIGHT = 6
RELEVANCE_SIGNAL_FLOOR = 20

CLASSIFY_HIGH = 68
CLASSIFY_EXTRACT = 60
RELEVANCE_FLOOR = 10

# ── Default thresholds ──────────────────────────────────────────────

DEFAULT_THRESHOLDS = {
    "star_threshold": 75,
    "min_star_threshold": 25,
    "max_star_threshold": 500,
    "noise_keywords": ["awesome", "curated list", "awesome list", "learn", "tutorial", "list", "resource", "cheatsheet"],
    "language_filters": ["HTML", "CSS", "Markdown"],
    "spam_name_patterns": [
        r"-(19|20)\d\d(-|$)",
        r"(?i)(crack|keygen|nulled|allprompts|free-?download|activation-?key|license-?key|-latest-|version-\d)",
    ],
    "dead_repo_forks_ratio": 3.0,
    "dead_repo_min_stars": 10,
    "consecutive_noise_high_days": 3,
    "consecutive_signal_good_days": 3,
    "consecutive_signal_low_days": 5,
    "noise_high_threshold_pct": 40.0,
    "noise_low_threshold_pct": 20.0,
    "signal_high_threshold_pct": 60.0,
    "signal_low_threshold_pct": 10.0,
    "star_adjust_step": 25,
    "history": [],
    "last_tuned": None,
}

# ── Dark Theme CSS ──────────────────────────────────────────────────

DARK_CSS = """
:root { color-scheme: dark; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #11100f; color: #e0dcd6; margin: 0; padding: 20px; line-height: 1.5; }
.container { max-width: 900px; margin: 0 auto; }
h1 { color: #fbbf24; border-bottom: 2px solid #fbbf24; padding-bottom: 8px; margin-bottom: 16px; }
h2 { color: #fbbf24; margin-top: 28px; font-size: 1.1em; text-transform: uppercase; letter-spacing: 0.03em; }
.stats { display: grid; grid-template-columns: repeat(auto-fit,minmax(140px,1fr)); gap: 8px; margin: 12px 0 20px; }
.stat { background: #1c1a18; border: 1px solid #34302c; border-radius: 6px; padding: 10px 12px; }
.stat-label { font-size: 0.75em; color: #a8a29e; text-transform: uppercase; }
.stat-value { font-size: 1.25em; font-weight: 600;
color: #fbbf24; }
.repo { background: #1c1a18; border: 1px solid #34302c; border-radius: 8px; padding: 14px; margin-bottom: 10px; }
.repo-header { display: flex; align-items: baseline; gap: 8px; margin-bottom: 6px; flex-wrap: wrap; }
.repo-name { font-weight: 600; color: #fbbf24; font-size: 1.05em; }
.repo-stars { color: #a8a29e; font-size: 0.85em; }
.repo-lang { color: #a8a29e; font-size: 0.85em; }
.repo-meta { margin-bottom: 6px; }
.repo-score { font-size: 0.85em; color: #a8a29e; }
.repo-desc { color: #e0dcd6; margin-bottom: 6px; }
.repo-why { color: #d6d3d1; font-style: italic; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75em; font-weight: 600; margin-right: 6px; }
.badge-green { background: #3f6212; color: #bef264; }
.badge-yellow { background: #713f12; color: #fde047; }
.badge-red { background: #7f1d1d; color: #fca5a5; }
.badge-blue { background: #1e3a5f; color: #93c5fd; }
.badge-grey { background: #3d3d3d; color: #d4d4d4; }
.footer { margin-top: 20px; color: #78716c; font-size: 0.8em; border-top: 1px solid #34302c; padding-top: 10px; }
a { color: #60a5fa; text-decoration: none; }
a:hover { text-decoration: underline; }
"""

# ── Thresholds ──────────────────────────────────────────────────────

def load_thresholds():
    if not os.path.exists(THRESHOLDS_FILE):
        save_thresholds(DEFAULT_THRESHOLDS)
        return dict(DEFAULT_THRESHOLDS)
    try:
        with open(THRESHOLDS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        merged = dict(DEFAULT_THRESHOLDS)
        merged.update(data)
        return merged
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULT_THRESHOLDS)

def save_thresholds(thresholds):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(THRESHOLDS_FILE, "w", encoding="utf-8") as f:
            json.dump(thresholds, f, indent=2, default=str)
    except OSError as e:
        print(f"WARN: Failed to write thresholds: {e}", file=sys.stderr)

def build_queries(thresholds):
    base_threshold = thresholds["star_threshold"]
    queries = []
    for tpl in QUERY_TEMPLATES:
        star_eff = max(tpl["star_base"], int(base_threshold * 0.75))
        q = tpl["q"].replace("{stars}", str(star_eff))
        queries.append({"q": q, "sort": tpl["sort"], "order": tpl["order"]})
    return queries

# ── Self-Tuning ─────────────────────────────────────────────────────

METRICS_LOOKBACK_DAYS = 7

def load_metrics():
    if not os.path.exists(METRICS_FILE):
        return []
    try:
        with open(METRICS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

def append_metrics_entry(entry):
    metrics = load_metrics()
    metrics.append(entry)
    if len(metrics) > 365:
        metrics = metrics[-365:]
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    except OSError as e:
        print(f"WARN: Failed to write metrics: {e}", file=sys.stderr)

def self_tune(thresholds, noise_rate_pct, signal_rate_pct):
    actions = []
    metrics = load_metrics()
    today = {"noise_rate_pct": noise_rate_pct, "signal_rate_pct": signal_rate_pct}
    recent = metrics[-METRICS_LOOKBACK_DAYS:] if len(metrics) >= METRICS_LOOKBACK_DAYS else metrics
    window = list(recent) + [today] if recent else [today]
    if len(window) < 2:
        return thresholds, ["Not enough data to tune (need 2+ runs)"]
    high_noise_consecutive = 0
    good_signal_consecutive = 0
    low_signal_consecutive = 0
    low_noise_consecutive = 0
    for entry in window:
        nr = entry.get("noise_rate_pct", 0)
        sr = entry.get("signal_rate_pct", 0)
        if nr >= thresholds["noise_high_threshold_pct"]:
            high_noise_consecutive += 1
        else:
            high_noise_consecutive = 0
        if sr >= thresholds["signal_high_threshold_pct"]:
            good_signal_consecutive += 1
        else:
            good_signal_consecutive = 0
        if sr < thresholds["signal_low_threshold_pct"]:
            low_signal_consecutive += 1
        else:
            low_signal_consecutive = 0
        if nr < thresholds["noise_low_threshold_pct"]:
            low_noise_consecutive += 1
        else:
            low_noise_consecutive = 0

    adj = thresholds["star_adjust_step"]
    limit_high = thresholds["max_star_threshold"]
    limit_low = thresholds["min_star_threshold"]
    st = thresholds["star_threshold"]

    if high_noise_consecutive >= thresholds["consecutive_noise_high_days"]:
        st = min(st + adj, limit_high)
        actions.append(f"RAISE: noise {noise_rate_pct}% for {high_noise_consecutive}d → star_threshold={st}")
    elif low_signal_consecutive >= thresholds["consecutive_signal_low_days"]:
        st = min(st + adj, limit_high)
        actions.append(f"RAISE: signal {signal_rate_pct}% for {low_signal_consecutive}d → star_threshold={st}")
    elif good_signal_consecutive >= thresholds["consecutive_signal_good_days"]:
        st = max(st - adj, limit_low)
        actions.append(f"LOWER: signal {signal_rate_pct}% for {good_signal_consecutive}d → star_threshold={st}")
    elif low_noise_consecutive >= thresholds["consecutive_noise_high_days"]:
        st = max(st - adj, limit_low)
        actions.append(f"LOWER: noise {noise_rate_pct}% for {low_noise_consecutive}d → star_threshold={st}")
    else:
        actions.append(f"HOLD: noise {noise_rate_pct}%, signal {signal_rate_pct}% — star_threshold stays at {st}")

    thresholds["star_threshold"] = st
    hist = thresholds.get("history", [])
    hist.append({
        "tuned_at": datetime.now(timezone.utc).isoformat(),
        "noise_rate_pct": noise_rate_pct,
        "signal_rate_pct": signal_rate_pct,
        "star_threshold": st,
        "actions": list(actions),
    })
    thresholds["history"] = hist[-30:]
    thresholds["last_tuned"] = datetime.now(timezone.utc).isoformat()
    return thresholds, actions

# ── GitHub API ──────────────────────────────────────────────────────

SEARCH_RATE_LIMIT = 30  # GitHub Search API: 30 requests per minute
_search_call_timestamps = []  # track call times for rate limiting

def wait_for_rate_limit():
    """Ensure we don't exceed GitHub Search API rate limit (30 req/min)."""
    global _search_call_timestamps
    now = time.time()
    # Prune timestamps older than 60s
    _search_call_timestamps = [t for t in _search_call_timestamps if now - t < 60]
    if len(_search_call_timestamps) >= SEARCH_RATE_LIMIT:
        # Need to wait until oldest timestamp expires (60s after it was recorded)
        wait = _search_call_timestamps[0] + 60 - now
        if wait > 0:
            time.sleep(wait + 0.5)  # add buffer
        # Prune again after waiting
        _search_call_timestamps = [t for t in _search_call_timestamps if time.time() - t < 60]
    _search_call_timestamps.append(time.time())

def gh_auth_token():
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if token:
        return token
    try:
        r = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            return r.stdout.strip()
    except FileNotFoundError:
        pass
    return None

def get_date_filters():
    pushed_cutoff = datetime.now(timezone.utc) - timedelta(days=RECENCY_DAYS)
    created_cutoff = datetime.now(timezone.utc) - timedelta(days=CREATED_RECENCY_DAYS)
    return pushed_cutoff.strftime("%Y-%m-%d"), created_cutoff.strftime("%Y-%m-%d")

def github_search(query, sort="stars", order="desc", per_page=100, page=1):
    token = gh_auth_token()
    if not token:
        return [], 0
    pushed_q, created_q = get_date_filters()
    full_q = f"{query} pushed:>{pushed_q} created:>{created_q}"
    params = urllib.parse.urlencode({
        "q": full_q, "sort": sort, "order": order,
        "per_page": per_page, "page": page
    })
    url = f"https://api.github.com/search/repositories?{params}"
    wait_for_rate_limit()  # throttle to 30 req/min
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "KENSEI-GitHub-Radar/1.0")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            return data.get("items", []), data.get("total_count", 0)
    except urllib.error.HTTPError as e:
        print(f"WARN: GitHub API error {e.code} for query '{query[:60]}': {e.reason}", file=sys.stderr)
        return [], 0
    except Exception as e:
        print(f"WARN: GitHub API exception: {e}", file=sys.stderr)
        return [], 0

# ── GitHub Trending (PRIMARY SOURCE) ────────────────────────────────

# Enhanced trending scrape: extracts stars and description from the trending page
def scrape_trending(url, label):
    """Scrape one trending page. Returns list of dicts with full_name, stars, description."""
    results = []
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "KENSEI-GitHub-Radar/1.0", "Accept": "text/html",
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        # Extract owner/name from all repo cards
        for match in re.finditer(
            r'<h2[^>]*class="[^"]*h3[^"]*"[^>]*>.*?<a[^>]*href="/([^/"]+)/([^/"]+)"',
            html, re.DOTALL
        ):
            owner, repo = match.group(1), match.group(2)
            if owner and repo:
                results.append({
                    "full_name": f"{owner}/{repo}",
                    "source": f"trending-{label}",
                    "stars": 0,  # trending scrape can't reliably get stars; will be corrected by API if found
                    "description": "",
                })
    except Exception as e:
        print(f"WARN: Trending scrape failed ({label}): {e}", file=sys.stderr)
    return results

def scrape_all_trending():
    """Scrape all trending pages (daily + weekly). Returns deduped list."""
    seen = set()
    all_results = []
    for label, url in TRENDING_URLS:
        repos = scrape_trending(url, label)
        for r in repos:
            if r["full_name"] not in seen:
                seen.add(r["full_name"])
                all_results.append(r)
    print(f"TRENDING: {len(all_results)} unique repos from trending pages", file=sys.stderr)
    return all_results

# ── Repo Data ───────────────────────────────────────────────────────

def parse_star_count(item):
    val = item.get("stargazers_count") or item.get("stars") or 0
    return int(val)

def extract_repo(item):
    return {
        "full_name": item.get("full_name", ""),
        "description": (item.get("description") or "").strip(),
        "stars": parse_star_count(item),
        "forks": item.get("forks_count", 0) or 0,
        "language": item.get("language") or "",
        "topics": item.get("topics", []) or [],
        "created_at": item.get("created_at", ""),
        "pushed_at": item.get("pushed_at", ""),
        "open_issues": item.get("open_issues_count", 0) or 0,
        "license": item.get("license", {}).get("spdx_id", "") if item.get("license") else "",
        "html_url": item.get("html_url", ""),
        "source": "api",
    }

# ── Pre-Filtering ──────────────────────────────────────────────────

def build_noise_patterns(thresholds):
    keywords = thresholds.get("noise_keywords", DEFAULT_THRESHOLDS["noise_keywords"])
    lang_filters = thresholds.get("language_filters", DEFAULT_THRESHOLDS["language_filters"])
    fork_ratio = thresholds.get("dead_repo_forks_ratio", 3.0)
    dead_min = thresholds.get("dead_repo_min_stars", 10)
    spam_patterns = thresholds.get("spam_name_patterns", DEFAULT_THRESHOLDS["spam_name_patterns"])
    return {
        "awesome_list": lambda r: any(
            kw in (r.get("description", "") + " " + " ".join(r.get("topics", []))).lower()
            for kw in ["awesome", "curated list", "awesome list"]
        ),
        "tutorial_content": lambda r: any(
            r["full_name"].lower().startswith(prefix)
            for prefix in ["learn-", "awesome-", "tutorial-"]
        ),
        "dead_repo": lambda r: (
            parse_star_count(r) < dead_min
            or (r.get("forks", 0) or 0) > parse_star_count(r) * fork_ratio
        ),
        "non_code": lambda r: r.get("language", "") in lang_filters,
        "name_noise": lambda r: any(
            kw in r["full_name"].lower().split("/")[1]
            for kw in keywords
        ),
        "spam_name": lambda r: any(
            re.search(p, r["full_name"].split("/")[-1])
            for p in spam_patterns
        ),
    }

NOISE_ORDER = ["awesome_list", "non_code", "name_noise", "spam_name", "tutorial_content", "dead_repo"]

def classify_noise(repo, thresholds):
    patterns = build_noise_patterns(thresholds)
    for rule in NOISE_ORDER:
        if patterns[rule](repo):
            return True, rule
    return False, ""

def deduplicate(repos):
    seen = {}
    for r in repos:
        name = r["full_name"]
        if name not in seen or parse_star_count(r) > parse_star_count(seen[name]):
            seen[name] = r
    return list(seen.values())

# ── Scoring ──────────────────────────────────────────────────────────

def score_repo(repo):
    stars = parse_star_count(repo)
    star_score = min(100.0, 20 + 80 * math.log10(stars + 1) / math.log10(2000))
    text_blob = f"{repo.get('description', '')} {' '.join(repo.get('topics', []))} {repo.get('language', '')}".lower()
    high_hits = sum(1 for kw in RELEVANCE_HIGH if kw in text_blob)
    generic_hits = sum(1 for kw in RELEVANCE_GENERIC if kw in text_blob)
    relevance_score = min(100, high_hits * RELEVANCE_HIGH_WEIGHT + generic_hits * RELEVANCE_GENERIC_WEIGHT)
    try:
        pushed = datetime.fromisoformat(repo.get("pushed_at", "").replace("Z", "+00:00"))
        days_since_push = (datetime.now(timezone.utc) - pushed).days
        freshness_score = max(0, 100 - days_since_push * 10)
    except (ValueError, TypeError):
        freshness_score = 50
    try:
        created = datetime.fromisoformat(repo.get("created_at", "").replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - created).days
        active_score = 100 if age_days < 30 else (80 if age_days < 90 else 50)
    except (ValueError, TypeError):
        active_score = 50
    score = round(star_score * 0.4 + relevance_score * 0.3 + freshness_score * 0.2 + active_score * 0.1, 1)
    on_mission = bool(repo.get("description", "").strip()) and relevance_score >= RELEVANCE_SIGNAL_FLOOR
    classification, why = "INSPIRATION", "Moderate signal: worth noting for future reference."
    if score >= CLASSIFY_HIGH:
        if any(kw in text_blob for kw in ["mcp server", "mcp", "plugin", "skill"]):
            classification, why = "PLUGIN/SKILL", "Agent tooling: aligns with Hermes/KENSEI ecosystem."
        elif any(kw in text_blob for kw in ["product", "saas", "platform", "app", "react native", "expo"]):
            classification, why = "FORK/PRODUCT", "Could become a standalone product: evaluate for Plenishd/CoachOS or new spinoff."
        elif any(kw in text_blob for kw in ["proxy", "filter", "terminal", "cli", "tui", "voice"]):
            classification, why = "ADOPT", "Direct internal tool: install or integrate into KENSEI workflows."
        elif on_mission:
            classification, why = "EXTRACT", "On-mission high-signal repo: extract concept or pattern."
        else:
            classification, why = "INSPIRATION", "High score but off-mission: worth noting for future reference."
    elif score >= CLASSIFY_EXTRACT:
        if any(kw in text_blob for kw in ["mcp server", "mcp", "plugin", "skill"]):
            classification, why = "PLUGIN/SKILL", "Agent tooling: aligns with Hermes/KENSEI ecosystem."
        elif on_mission:
            classification, why = "EXTRACT", "On-mission repo: worth extracting concepts or patterns."
        else:
            classification, why = "INSPIRATION", "Mid-score but off-mission: file for reference."
    elif relevance_score >= RELEVANCE_FLOOR:
        classification, why = "INSPIRATION", "Low score but relevant: file for reference."
    if relevance_score < RELEVANCE_FLOOR and classification != "NOISE":
        classification, why = "INSPIRATION", "High-star but off-mission: demoted from higher classification."
    return score, classification, why

def classify_all(repos):
    for r in repos:
        score, classification, why = score_repo(r)
        r["score"] = score
        r["classification"] = classification
        r["why"] = why
    return sorted(repos, key=lambda x: x.get("score", 0), reverse=True)

# ── Enrichment (API-based, best-effort) ─────────────────────────────

ENRICH_BASE_URL = "https://openrouter.ai/api/v1"
ENRICH_MODEL = "nvidia/nemotron-3-super-120b-a12b"

def enrich_why(picks):
    if not picks:
        return
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return
    listing = "\n".join(
        f"- {r['full_name']} ★{r['stars']} {r.get('language','')}: {r.get('description','')[:120]}"
        for r in picks[:10]
    )
    prompt = (
        "You triage new GitHub repos for an AI-agent builder. For each repo below, "
        "write ONE concrete sentence (max 22 words) on why it matters or what is worth "
        "taking from it. Be specific to the repo, not generic. No marketing fluff.\n\n"
        f"{listing}\n\n"
        'Reply with ONLY a JSON object mapping each repo full_name to its sentence, '
        'e.g. {"owner/repo": "..."}.'
    )
    payload = json.dumps({
        "model": ENRICH_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "stream": False,
    }).encode()
    req = urllib.request.Request(f"{ENRICH_BASE_URL}/chat/completions", data=payload)
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode())
        content = data["choices"][0]["message"]["content"]
        match = re.search(r"\{.*\}", content, re.DOTALL)
        mapping = json.loads(match.group(0) if match else content)
    except Exception as e:
        print(f"ENRICH: failed, keeping deterministic text ({e})", file=sys.stderr)
        return
    enriched = 0
    for r in picks:
        line = mapping.get(r["full_name"])
        if isinstance(line, str) and line.strip():
            r["why"] = line.strip()
            r["why_enriched"] = True
            enriched += 1
    print(f"ENRICH: {enriched}/{len(picks)} top picks enriched", file=sys.stderr)

# ── Cache v2 (pushed_at tracking) ────────────────────────────────────

CACHE_TTL_DAYS = 14

def load_cache_v2():
    """
    Load cache with pushed_at tracking.
    Returns (active_set: set of repo names, pushed_map: {name: pushed_at}).
    """
    if not os.path.exists(CACHE_FILE):
        return set(), {}
    try:
        with open(CACHE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        seen_data = data.get("seen", {})
        if isinstance(seen_data, list):
            # Old format: list of names — clear it (migrate to v2)
            print(f"CACHE: old list format ({len(seen_data)} entries), migrating to v2", file=sys.stderr)
            return set(), {}
        if isinstance(seen_data, dict):
            now = datetime.now(timezone.utc)
            active = set()
            pushed_map = {}
            expired = 0
            for repo, entry in seen_data.items():
                # Support both new {name: {first_seen, last_pushed}} and old {name: timestamp_str}
                if isinstance(entry, str):
                    date_str = entry
                    pushed = ""
                elif isinstance(entry, dict):
                    date_str = entry.get("first_seen", "")
                    pushed = entry.get("last_pushed", "")
                else:
                    date_str = ""
                    pushed = ""
                try:
                    seen_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    if (now - seen_date).days < CACHE_TTL_DAYS:
                        active.add(repo)
                        if pushed:
                            pushed_map[repo] = pushed
                    else:
                        expired += 1
                except (ValueError, TypeError):
                    expired += 1
            if expired > 0:
                print(f"CACHE: pruned {expired} expired entries (TTL={CACHE_TTL_DAYS}d), {len(active)} active", file=sys.stderr)
            return active, pushed_map
    except (json.JSONDecodeError, KeyError):
        return set(), {}

def save_cache_v2(seen_set, pushed_map=None):
    """
    Save cache preserving first-seen timestamps + updating last_pushed.
    seen_set: set of repo full_names to keep in cache
    pushed_map: optional {name: pushed_at_str} to update timestamps
    """
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        now_str = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        # Load existing data to preserve first_seen
        old_entries = {}
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, encoding="utf-8") as f:
                    old_data = json.load(f)
                old_seen = old_data.get("seen", {})
                if isinstance(old_seen, dict):
                    for repo, entry in old_seen.items():
                        if isinstance(entry, str):
                            old_entries[repo] = {"first_seen": entry, "last_pushed": ""}
                        elif isinstance(entry, dict):
                            old_entries[repo] = entry
            except (json.JSONDecodeError, KeyError):
                pass
        # Build merged dict
        new_entries = {}
        preserved = 0
        new_count = 0
        for repo in seen_set:
            if repo in old_entries:
                entry = dict(old_entries[repo])
                # Update last_pushed if we have new data
                if pushed_map and repo in pushed_map and pushed_map[repo]:
                    entry["last_pushed"] = pushed_map[repo]
                entry["last_checked"] = now_str
                new_entries[repo] = entry
                preserved += 1
            else:
                entry = {
                    "first_seen": now_str,
                    "last_pushed": (pushed_map or {}).get(repo, ""),
                    "last_checked": now_str,
                }
                new_entries[repo] = entry
                new_count += 1
        if new_count > 0 or preserved > 0:
            print(f"CACHE: {preserved} preserved, {new_count} new entries", file=sys.stderr)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({"seen": new_entries}, f)
        # Write stats companion file
        stats = {
            "total": len(new_entries),
            "preserved": preserved,
            "new": new_count,
            "pushed_tracked": sum(1 for e in new_entries.values() if e.get("last_pushed")),
            "updated_at": now_str,
        }
        with open(CACHE_STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        print(f"WARN: Failed to write cache: {e}", file=sys.stderr)

# ── Collection ──────────────────────────────────────────────────────

def collect_daily(queries):
    """
    Daily mode: dedup against cache. Trending is a primary source.
    Returns (repos, cache_stats) where cache_stats = {total, new, refreshed}.
    """
    active_cache, pushed_map = load_cache_v2()
    seen = set(active_cache)
    all_repos = []
    rate_limited = False

    # Phase 1: GitHub Search API (dedup'd against cache)
    for query_def in queries:
        if rate_limited:
            break
        q = query_def["q"]
        sort = query_def.get("sort", "stars")
        order = query_def.get("order", "desc")
        page = 1
        while page <= MAX_PAGES and not rate_limited:
            items, total = github_search(q, sort, order, 100, page)
            if not items:
                break
            for item in items:
                fn = item.get("full_name", "")
                if fn and fn not in seen:
                    seen.add(fn)
                    all_repos.append(extract_repo(item))
            page += 1
            if page <= MAX_PAGES:
                pass  # rate limiting handled by wait_for_rate_limit() in github_search

    # Phase 2: Trending (PRIMARY source — scrape daily AND weekly)
    trending = scrape_all_trending()
    trending_new = 0
    for t in trending:
        name = t["full_name"]
        if name not in seen:
            seen.add(name)
            all_repos.append({
                "full_name": name, "description": "", "stars": 0, "forks": 0,
                "language": "", "topics": [], "created_at": "", "pushed_at": "",
                "open_issues": 0, "license": "", "html_url": f"https://github.com/{name}",
                "source": "trending",
            })
            trending_new += 1

    # Build pushed_map from collected repos (update last_pushed in cache)
    collected_pushed = {}
    for r in all_repos:
        if r.get("pushed_at"):
            collected_pushed[r["full_name"]] = r["pushed_at"]

    save_cache_v2(seen, collected_pushed)

    cache_stats = {
        "total": len(seen),
        "new_api": len(all_repos) - trending_new,
        "new_trending": trending_new,
        "refreshed": trending_new,
    }
    print(f"DAILY: {len(all_repos)} total ({len(all_repos)-trending_new} API, {trending_new} trending), {len(active_cache)} cached", file=sys.stderr)
    return all_repos, cache_stats


WEEKLY_PAGES = 5  # weekly uses fewer pages than daily to stay within rate limits

def collect_weekly(queries):
    """
    Weekly mode: full re-scan without cache dedup.
    Uses fewer pages (WEEKLY_PAGES=5) to stay within GitHub's 30 req/min search API limit.
    Compares pushed_at against cached last_pushed to find repos with new activity.
    Returns (repos, re_eval_stats) where re_eval_stats = {total, with_new_activity, new_finds}.
    """
    active_cache, pushed_map = load_cache_v2()
    all_repos = []
    seen = set()  # session dedup (across queries), not cache dedup
    rate_limited = False

    # Phase 1: Full API scan — find ALL matching repos, including cached ones
    for query_def in queries:
        if rate_limited:
            break
        q = query_def["q"]
        sort = query_def.get("sort", "stars")
        order = query_def.get("order", "desc")
        page = 1
        while page <= WEEKLY_PAGES and not rate_limited:
            items, total = github_search(q, sort, order, 100, page)
            if not items:
                break
            for item in items:
                fn = item.get("full_name", "")
                if fn and fn not in seen:
                    seen.add(fn)
                    all_repos.append(extract_repo(item))
            page += 1
            if page <= WEEKLY_PAGES:
                pass  # rate limiting handled by wait_for_rate_limit() in github_search

    # Phase 2: Trending
    trending = scrape_all_trending()
    for t in trending:
        name = t["full_name"]
        if name not in seen:
            seen.add(name)
            all_repos.append({
                "full_name": name, "description": "", "stars": 0, "forks": 0,
                "language": "", "topics": [], "created_at": "", "pushed_at": "",
                "open_issues": 0, "license": "", "html_url": f"https://github.com/{name}",
                "source": "trending",
            })

    # Phase 3: Compare against cached pushed_at
    re_evaluated = len(active_cache)
    with_new_activity = 0
    new_finds = 0
    for repo in all_repos:
        name = repo["full_name"]
        cached_pushed = pushed_map.get(name, "")
        current_pushed = repo.get("pushed_at", "")
        if name in active_cache:
            if current_pushed and cached_pushed and current_pushed > cached_pushed:
                with_new_activity += 1
                repo["new_activity"] = True
                repo["old_pushed"] = cached_pushed
        else:
            new_finds += 1
            repo["new_find"] = True

    # Update cache with all findings (including pushed_at)
    updated_pushed = {}
    for r in all_repos:
        if r.get("pushed_at"):
            updated_pushed[r["full_name"]] = r["pushed_at"]

    save_cache_v2(seen, updated_pushed)

    stats = {
        "re_evaluated": re_evaluated,
        "with_new_activity": with_new_activity,
        "new_finds": new_finds,
        "total_in_scope": len(seen),
    }
    print(f"WEEKLY: re-evaluated {re_evaluated} cached repos, {with_new_activity} with new activity, {new_finds} new finds", file=sys.stderr)
    return all_repos, stats


# ── Filtering ──────────────────────────────────────────────────────

def filter_repos(repos, thresholds):
    keep = []
    filtered = defaultdict(list)
    for repo in repos:
        is_noise, reason = classify_noise(repo, thresholds)
        if is_noise:
            filtered[reason].append(repo["full_name"])
        else:
            keep.append(repo)
    print(f"FILTER: {len(keep)} kept, {len(repos)-len(keep)} filtered", file=sys.stderr)
    for reason, names in sorted(filtered.items()):
        print(f"  {reason}: {len(names)}", file=sys.stderr)
    return keep, dict(filtered)

def deduplicate_repos(repos):
    result = deduplicate(repos)
    print(f"DEDUP: {len(result)} unique ({len(repos)-len(result)} removed)", file=sys.stderr)
    return result

# ── Text / HTML Builders ────────────────────────────────────────────

def badge_class(label):
    return {
        "ADOPT": "badge-green",
        "EXTRACT": "badge-blue",
        "PLUGIN/SKILL": "badge-yellow",
        "FORK/PRODUCT": "badge-red",
        "INSPIRATION": "badge-grey",
    }.get(label, "badge-grey")

def build_html(repos, stats, extra_stats=None):
    rows = []
    rows.append("<!DOCTYPE html><html><head><meta charset='utf-8'><title>GitHub Radar</title>")
    rows.append(f"<style>{DARK_CSS}</style></head><body><div class='container'>")
    rows.append(f"<h1>GitHub Radar · {DAY_LABEL} ({MODE.upper()})</h1>")

    # Mode-aware stats header
    if MODE == "weekly" and extra_stats:
        rows.append("<div class='stats'>")
        for label, value in [
            ("Re-evaluated", extra_stats.get("re_evaluated", 0)),
            ("New activity", extra_stats.get("with_new_activity", 0)),
            ("New finds", extra_stats.get("new_finds", 0)),
            ("In scope", extra_stats.get("total_in_scope", 0)),
        ]:
            rows.append(f'<div class="stat"><div class="stat-label">{label}</div><div class="stat-value">{value}</div></div>')
        rows.append("</div>")

    rows.append("<div class='stats'>")
    for label, value in [
        ("Repos", stats["after_dedup"]),
        ("Scored", stats["scored"]),
        ("ADOPT", len([r for r in repos if r["classification"] == "ADOPT"])),
        ("EXTRACT", len([r for r in repos if r["classification"] == "EXTRACT"])),
        ("PLUGIN/SKILL", len([r for r in repos if r["classification"] == "PLUGIN/SKILL"])),
        ("FORK/PRODUCT", len([r for r in repos if r["classification"] == "FORK/PRODUCT"])),
        ("INSPIRATION", len([r for r in repos if r["classification"] == "INSPIRATION"])),
        ("NOISE", stats["noise"]),
    ]:
        rows.append(f'<div class="stat"><div class="stat-label">{label}</div><div class="stat-value">{value}</div></div>')
    rows.append("</div>")

    # Weekly re-evaluation section
    if MODE == "weekly":
        new_activity = [r for r in repos if r.get("new_activity")]
        if new_activity:
            rows.append("<h2>Repos With New Activity</h2>")
            for r in new_activity[:20]:
                bc = badge_class(r.get("classification", ""))
                rows.append(f'<div class="repo">')
                rows.append(f'<div class="repo-header">')
                rows.append(f'<span class="repo-name"><a href="{r["html_url"]}" target="_blank">{r["full_name"]}</a></span>')
                rows.append(f'<span class="repo-stars">★ {r["stars"]}</span>')
                rows.append(f'<span class="repo-lang">{r.get("language") or "Unknown"}</span>')
                rows.append(f'<span class="badge {bc}">{r.get("classification", "UNSORTED")}</span>')
                rows.append(f'<span class="repo-score">Score: {r.get("score", 0)}</span>')
                rows.append(f'</div>')
                if r.get("description"):
                    rows.append(f'<div class="repo-desc">{r["description"]}</div>')
                if r.get("old_pushed"):
                    rows.append(f'<div class="repo-why">→ Last checked pushed: {r["old_pushed"][:10]}, now: {r.get("pushed_at","?")[:10]}</div>')
                rows.append(f'</div>')

    # Top repos
    top_repos = [r for r in repos if r.get("classification") not in ("NOISE", None)][:15]
    if top_repos:
        rows.append(f"<h2>Top {len(top_repos)} Scored Repos</h2>")
    for r in top_repos:
        bc = badge_class(r["classification"])
        prefix = "🔄 " if r.get("new_activity") else "🆕 " if r.get("new_find") else ""
        rows.append(f'<div class="repo">')
        rows.append(f'<div class="repo-header">')
        rows.append(f'<span class="repo-name"><a href="{r["html_url"]}" target="_blank">{prefix}{r["full_name"]}</a></span>')
        rows.append(f'<span class="repo-stars">★ {r["stars"]}</span>')
        rows.append(f'<span class="repo-lang">{r["language"] or "Unknown"}</span>')
        rows.append(f'<span class="badge {bc}">{r["classification"]}</span>')
        rows.append(f'<span class="repo-score">Score: {r["score"]}</span>')
        rows.append(f'</div>')
        if r.get("description"):
            rows.append(f'<div class="repo-desc">{r["description"]}</div>')
        rows.append(f'<div class="repo-why">→ {r["why"]}</div>')
        rows.append(f'</div>')

    if len(repos) > len(top_repos):
        noise_count = len([r for r in repos if r.get("classification") == "NOISE"])
        rows.append(f"<div class='footer'>+ {noise_count} noise-repos filtered. Full data in {OUTPUT_FILE}.</div>")

    rows.append("</div></body></html>")
    return "\n".join(rows)

def build_repos_text(repos, stats, extra_stats=None):
    lines = [
        f"**GitHub Radar** · {DAY_LABEL} {datetime.now(timezone.utc).strftime('%H:%M')} ({MODE.upper()})",
        (
            f"{stats['after_dedup']} repos · {stats['scored']} scored · "
            f"{len([r for r in repos if r['classification']=='ADOPT'])} ADOPT · "
            f"{len([r for r in repos if r['classification']=='EXTRACT'])} EXTRACT · "
            f"{len([r for r in repos if r['classification']=='PLUGIN/SKILL'])} PLUGIN/SKILL · "
            f"{len([r for r in repos if r['classification']=='FORK/PRODUCT'])} FORK/PRODUCT · "
            f"{len([r for r in repos if r['classification']=='INSPIRATION'])} INSPIRATION"
        ),
        "",
    ]
    if extra_stats:
        if MODE == "daily":
            lines.append(f"**Cache:** {extra_stats.get('total',0)} repos cached · "
                        f"{extra_stats.get('new_api',0)} new API finds · "
                        f"{extra_stats.get('new_trending',0)} from trending")
            lines.append("")
        elif MODE == "weekly":
            lines.append(f"**Weekly re-evaluation:** {extra_stats.get('re_evaluated',0)} repos checked · "
                        f"{extra_stats.get('with_new_activity',0)} with new activity · "
                        f"{extra_stats.get('new_finds',0)} new finds")
            lines.append("")

    lines.append("---REPOS---")
    for r in repos:
        if r.get("classification") == "NOISE":
            continue
        prefix = "[NEW ACTIVITY] " if r.get("new_activity") else "[NEW FIND] " if r.get("new_find") else ""
        repo_line = (
            f"{prefix}[REPO] {r['full_name']}"
            f" | stars:{r['stars']} | lang:{r.get('language','')}"
            f" | classification:{r['classification']}"
            f" | score:{r['score']} | url:{r['html_url']}"
        )
        lines.append(repo_line)
        lines.append(f"  Description: {r.get('description', 'No description')}")
        lines.append(f"  Why it matters: {r['why']}")
    lines.append("---END REPOS---")
    lines.append("")

    lines.append("Top 3 this batch:")
    shown = 0
    for r in repos:
        if r.get("classification") == "NOISE":
            continue
        shown += 1
        if shown > 3:
            break
        prefix = "🔄 " if r.get("new_activity") else "🆕 " if r.get("new_find") else ""
        lines.append(f"{shown}. {prefix}{r['full_name']} — {r['why']}")

    lines.append("")
    lines.append("Tuning:")
    for action in stats.get("tuning_actions", []):
        lines.append(f"  • {action}")

    return "\n".join(lines)

def build_discord_summary(repos, stats, extra_stats=None):
    """Concise Discord summary — 3-6 lines, mode-aware."""
    top = [r for r in repos if r.get("classification") not in ("NOISE", None)][:3]
    lines = [
        f"**GitHub Radar** · {DAY_LABEL} {datetime.now(timezone.utc).strftime('%H:%M')} ({MODE.upper()})",
        "",
    ]
    if MODE == "daily" and extra_stats and stats.get("after_dedup", 0) == 0:
        lines.append(f"`{stats['after_dedup']} repos · {stats['scored']} scored` — "
                    f"`{extra_stats.get('total',0)} cached`")
        if extra_stats.get("new_trending", 0) > 0:
            lines.append(f"`{extra_stats['new_trending']} new from trending`")
    elif MODE == "weekly" and extra_stats:
        lines.append(f"`{extra_stats.get('re_evaluated',0)} re-evaluated · "
                    f"{extra_stats.get('with_new_activity',0)} with new activity · "
                    f"{extra_stats.get('new_finds',0)} new finds`")
    else:
        lines.append(f"`{stats['after_dedup']} repos · {stats['scored']} scored`")

    if top:
        lines.append("")
        lines.append("**Top picks**")
        for r in top:
            prefix = "🔄 " if r.get("new_activity") else "🆕 " if r.get("new_find") else ""
            lines.append(f"• {prefix}`{r['classification']}` [{r['full_name']}]({r['html_url']}) — ★ {r['stars']}, score {r['score']}")

    lines.append("")
    adopt = len([r for r in repos if r['classification']=='ADOPT'])
    extract = len([r for r in repos if r['classification']=='EXTRACT'])
    plugin = len([r for r in repos if r['classification']=='PLUGIN/SKILL'])
    forkp = len([r for r in repos if r['classification']=='FORK/PRODUCT'])
    insp = len([r for r in repos if r['classification']=='INSPIRATION'])
    lines.append(f"**Summary:** `ADOPT:{adopt} EXTRACT:{extract} PLUGIN/SKILL:{plugin} FORK/PRODUCT:{forkp} INSPIRATION:{insp}`")

    if extra_stats and MODE == "daily":
        lines.append(f"`{extra_stats.get('total',0)} cached · {extra_stats.get('new_trending',0)} trending`")

    lines.append("")
    lines.append(f"**Full detail:** MEDIA:{HTML_FILE}")
    return "\n".join(lines)

# ── Main ───────────────────────────────────────────────────────────

def main():
    thresholds = load_thresholds()
    queries = build_queries(thresholds)
    print(f"CONFIG: mode={MODE}, star_threshold={thresholds['star_threshold']}, queries={len(queries)}", file=sys.stderr)

    # Stage 1: Collection (mode-dependent)
    if MODE == "weekly":
        all_repos, extra_stats = collect_weekly(queries)
    else:
        all_repos, extra_stats = collect_daily(queries)

    # Stage 2: Filter + dedup
    filtered, filter_reasons = filter_repos(all_repos, thresholds)
    final = deduplicate_repos(filtered)

    # Score + classify
    scored = classify_all(final)

    # Enrich top picks
    enrich_why(scored)

    # Stats
    total = len(all_repos)
    noise_count = total - len(filtered)
    signal_count = len(final)
    noise_rate_pct = round(noise_count / total * 100, 1) if total > 0 else 0.0
    signal_rate_pct = round(signal_count / total * 100, 1) if total > 0 else 0.0

    # Self-tune
    thresholds, tuning_actions = self_tune(thresholds, noise_rate_pct, signal_rate_pct)
    save_thresholds(thresholds)

    print("TUNING:", file=sys.stderr)
    for a in tuning_actions:
        print(f"  {a}", file=sys.stderr)

    stats = {
        "total_collected": total,
        "after_filter": len(filtered),
        "after_dedup": len(final),
        "scored": len(scored),
        "noise": noise_count,
        "signal_rate_pct": signal_rate_pct,
        "noise_rate_pct": noise_rate_pct,
        "active_threshold": thresholds["star_threshold"],
        "tuning_actions": tuning_actions,
        "collection_queries": len(queries),
    }

    # Write JSON output
    output = {
        "collected_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mode": MODE,
        "stats": stats,
        "filter_reasons": {k: len(v) for k, v in filter_reasons.items()},
        "tuning": {"actions": tuning_actions, "thresholds": {k: thresholds[k] for k in ["star_threshold", "noise_keywords", "language_filters"]}},
        "repos": scored,
        "extra_stats": extra_stats,
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # Write HTML report + structured repos text
    os.makedirs(RUNBOOKS_TODAY, exist_ok=True)

    html = build_html(scored, stats, extra_stats)
    with open(HTML_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    repos_txt = build_repos_text(scored, stats, extra_stats)
    with open(REPOS_TXT_FILE, "w", encoding="utf-8") as f:
        f.write(repos_txt)

    # Print Discord summary
    print(build_discord_summary(scored, stats, extra_stats))

    # Metrics
    metrics_entry = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "mode": MODE,
        "total_repos": total,
        "actionable": signal_count,
        "noise": noise_count,
        "signal_rate_pct": signal_rate_pct,
        "noise_rate_pct": noise_rate_pct,
        "star_threshold": thresholds["star_threshold"],
    }
    if MODE == "weekly" and extra_stats:
        metrics_entry["re_evaluated"] = extra_stats.get("re_evaluated", 0)
        metrics_entry["with_new_activity"] = extra_stats.get("with_new_activity", 0)
        metrics_entry["new_finds"] = extra_stats.get("new_finds", 0)
    elif extra_stats:
        metrics_entry["cached_total"] = extra_stats.get("total", 0)
    append_metrics_entry(metrics_entry)


if __name__ == "__main__":
    main()
