#!/usr/bin/env python3
"""Importance-aware multi-source search — ranks by signal, not search order.

News by cross-outlet coverage, X by influential accounts, YouTube by view
velocity (views per day since upload). Discovery + synthesis via Codex OAuth
(free with a ChatGPT subscription); YouTube metadata via yt-dlp flat search.
Optional deep-fetch engine via IMPORTANCE_FETCH_DIR; falls back to plain HTTP.
Config: search_domains.json at the skill root (one level above scripts/)."""
import os
import sys
import json
import math
import time
import logging
import subprocess
import urllib.error
import urllib.request

logger = logging.getLogger("importance_search")

try:
    from agent.auxiliary_client import call_llm, extract_content_or_reasoning
    from tools.url_safety import is_safe_url
    from tools.website_policy import check_website_access
except ImportError as exc:
    raise SystemExit(
        "importance-search needs the hermes-agent package importable "
        "(run from the hermes-agent root or set PYTHONPATH): %s" % exc)

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(os.path.dirname(HERE), "search_domains.json")


def codex_call(prompt, instructions, with_search=False, timeout=200):
    try:
        resp = call_llm(
            provider="codex",
            model="gpt-5.5",
            api_mode="codex_responses",
            messages=[{"role": "system", "content": instructions},
                      {"role": "user", "content": prompt}],
            tools=[{"type": "web_search"}] if with_search else None,
            timeout=timeout,
        )
    except Exception as exc:
        logger.warning("Codex request failed: %s", exc)
        return ""
    return extract_content_or_reasoning(resp).strip()


def yt_flat_search(query, n=8):
    """Flat search returns view_count without the bot-gate that blocks full
    extraction on datacenter IPs; youtubetab:approximate_date adds a
    best-effort upload timestamp to flat entries."""
    try:
        r = subprocess.run([sys.executable, "-m", "yt_dlp", "ytsearch%d:%s" % (n, query),
                            "--flat-playlist", "--dump-json", "--no-warnings",
                            "--extractor-args", "youtubetab:approximate_date"],
                           capture_output=True, text=True, timeout=90)
    except Exception as exc:
        logger.warning("yt-dlp search failed for %r: %s", query, exc)
        return []
    out = []
    for line in r.stdout.splitlines():
        try:
            v = json.loads(line)
        except json.JSONDecodeError:
            continue
        if v.get("id"):
            out.append({"title": v.get("title", ""), "url": "https://youtu.be/" + v["id"],
                        "views": v.get("view_count") or 0, "channel": v.get("channel", "") or "",
                        "duration": v.get("duration") or 0, "timestamp": v.get("timestamp") or 0})
    return out


def view_velocity(item, now=None):
    """Views per day since upload. approximate_date is best-effort — undated
    items are treated as ~30 days old rather than ranked by raw view total."""
    views = item.get("views") or 0
    ts = item.get("timestamp") or 0
    if ts <= 0:
        return views / 30.0
    age_days = max(((now if now is not None else time.time()) - ts) / 86400.0, 1.0)
    return views / age_days


def youtube_top(domain, k=3):
    items, seen = [], set()
    for q in domain.get("youtube_queries", []):
        for pos, it in enumerate(yt_flat_search(q, 8)):
            if it["url"] in seen:
                continue
            seen.add(it["url"])
            it["pos"] = pos
            items.append(it)
    items = [it for it in items if it["duration"] == 0 or it["duration"] >= 90]
    now = time.time()
    for it in items:
        it["velocity"] = view_velocity(it, now)
        it["score"] = math.log10(it["velocity"] + 1) - it["pos"] * 0.04
    items.sort(key=lambda x: x["score"], reverse=True)
    return items[:k]


def _parse_lines(ans):
    out = []
    for line in ans.splitlines():
        if "|" in line and ("http" in line or "@" in line):
            parts = [p.strip() for p in line.split("|")]
            out.append({"title": parts[0].lstrip("-* "), "src": parts[1] if len(parts) > 1 else "",
                        "url": next((p for p in parts if p.startswith("http")), "")})
    return out


def x_top(domain, k=3):
    accs = domain.get("x_influencers", [])
    if not accs:
        return []
    al = ", ".join("@" + a for a in accs)
    ans = codex_call("On X, find the most important recent posts (last 24-72h) from these influential accounts: " + al +
                     ". Field: " + domain.get("label", "") + ". Output up to " + str(k) +
                     " lines formatted: title | account | url. Most impactful first.",
                     "Find recent high-impact posts from the given influential X accounts.", with_search=True)
    return _parse_lines(ans)[:k]


def news_top(domain, k=3):
    kw = ", ".join(domain.get("keywords", []))
    ans = codex_call("Field '" + domain.get("label", "") + "' (" + kw + "): find today's most important news. "
                     "Importance = cross-outlet coverage + recency. Output up to " + str(k) +
                     " lines formatted: title | outlet | url, most important first.",
                     "Find today's most important news by cross-outlet coverage and recency.", with_search=True)
    return _parse_lines(ans)[:k]


def _access_error(url):
    """Reason string if the URL must not be fetched, else None."""
    if not is_safe_url(url):
        return "blocked by URL safety check (scheme or private/internal address)"
    blocked = check_website_access(url)
    if blocked:
        return blocked.get("message", "blocked by website policy")
    return None


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-validate every redirect target — a public URL must not 302 into a
    private/internal address or a policy-blocked host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        reason = _access_error(newurl)
        if reason:
            raise urllib.error.HTTPError(newurl, code, "redirect blocked: " + reason, headers, fp)
        return urllib.request.HTTPRedirectHandler.redirect_request(
            self, req, fp, code, msg, headers, newurl)


def deep_fetch(url):
    reason = _access_error(url)
    if reason:
        logger.warning("deep-fetch refused for %s: %s", url, reason)
        return ""
    fdir = os.environ.get("IMPORTANCE_FETCH_DIR")
    if fdir and os.path.isdir(fdir):
        try:
            r = subprocess.run([sys.executable, "-m", "engine", url, "--max-attempts", "5", "--timeout", "18"],
                               cwd=fdir, capture_output=True, text=True, timeout=80)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()[:1500]
        except Exception as exc:
            logger.debug("deep-fetch engine failed for %s: %s", url, exc)
    try:
        opener = urllib.request.build_opener(_SafeRedirectHandler())
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        return opener.open(req, timeout=15).read().decode("utf-8", "ignore")[:1500]
    except Exception as exc:
        logger.debug("plain fetch failed for %s: %s", url, exc)
        return ""


def load_config():
    try:
        with open(CONFIG) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("importance-search: cannot load config %s: %s" % (CONFIG, exc))


def resolve_domain(cfg, domain_key):
    domains = cfg.get("domains", {})
    if not domains:
        raise SystemExit("importance-search: no domains configured in %s" % CONFIG)
    if domain_key in domains:
        return domains[domain_key]
    fallback = next(iter(domains))
    logger.warning("Unknown domain %r; falling back to %r. Known: %s",
                   domain_key, fallback, ", ".join(domains))
    return domains[fallback]


def run(domain_key):
    cfg = load_config()
    domain = resolve_domain(cfg, domain_key)
    news, x, yt = news_top(domain), x_top(domain), youtube_top(domain)
    detail = deep_fetch(news[0]["url"]) if news else ""
    out = ["# Importance briefing — " + domain.get("label", domain_key), ""]
    out.append("## News (coverage / recency)")
    for n in news:
        out.append("- **%s** (%s)\n  %s" % (n["title"], n.get("src", ""), n["url"]))
    out.append("\n## X influencers")
    for t in x:
        out.append("- **%s** (%s)\n  %s" % (t["title"], t.get("src", ""), t.get("url", "")))
    out.append("\n## YouTube (view velocity)")
    for v in yt:
        out.append("- **%s** (%s)\n  %s views, ~%s/day — %s" % (
            v["title"], v.get("channel", ""), "{:,}".format(v["views"]),
            "{:,.0f}".format(v.get("velocity", 0)), v["url"]))
    raw = "\n".join(out)
    insight = codex_call("Items today:\n" + raw[:2200] + (("\n\nTop news body:\n" + detail) if detail else "") +
                         "\n\nWrite ONLY a 2-line insight on why today's flow matters. No clichés.",
                         "Output ONLY a 2-line insight, no item list.")
    insight = "\n".join([line for line in insight.splitlines() if line.strip()][:2])
    return ("💡 " + insight + "\n\n" + raw) if insight else raw


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(run(sys.argv[1] if len(sys.argv) > 1 else "ai-tech"))
