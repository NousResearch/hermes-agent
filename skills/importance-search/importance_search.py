#!/usr/bin/env python3
"""Importance-aware multi-source search — ranks by signal, not search order.

News by cross-outlet coverage, X by influential accounts, YouTube by view velocity.
Discovery + synthesis via Codex OAuth (free, ChatGPT subscription); YouTube metadata via yt-dlp.
Optional deep-fetch engine via IMPORTANCE_FETCH_DIR (insane-search style); falls back to plain HTTP.
Config: search_domains.json (next to this file)."""
import os
import sys
import json
import math
import logging
import subprocess
import urllib.request

logger = logging.getLogger("importance_search")

sys.path.insert(0, os.path.expanduser("~/hermes-agent"))
# Codex OAuth helpers. These are underscore-prefixed (internal) in auxiliary_client —
# isolated in this single block so that if a public API lands, only this import changes.
# Raise a clear error instead of failing obscurely if they are renamed/removed.
try:
    from agent.auxiliary_client import (  # noqa: E402
        _read_codex_access_token as _codex_token,
        _codex_cloudflare_headers as _codex_headers,
        _CODEX_AUX_BASE_URL as _CODEX_BASE,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "importance-search needs Codex OAuth helpers from agent.auxiliary_client "
        "(_read_codex_access_token, _codex_cloudflare_headers, _CODEX_AUX_BASE_URL). "
        "If these were renamed or made public, update this import block."
    ) from exc

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "search_domains.json")


def codex_call(prompt, instructions, with_search=False, timeout=200):
    tok = _codex_token()
    if not tok:
        logger.warning("No Codex access token available; returning empty result.")
        return ""
    h = _codex_headers(tok)
    h["Authorization"] = "Bearer " + tok
    h["Content-Type"] = "application/json"
    p = {"model": "gpt-5.5", "instructions": instructions,
         "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]}],
         "stream": True, "store": False}
    if with_search:
        p["tools"] = [{"type": "web_search"}]
    try:
        raw = urllib.request.urlopen(urllib.request.Request(_CODEX_BASE + "/responses",
              data=json.dumps(p).encode(), headers=h, method="POST"), timeout=timeout).read().decode()
    except Exception as exc:
        logger.warning("Codex request failed: %s", exc)
        return ""
    out = ""
    for line in raw.splitlines():
        if line.startswith("data:"):
            try:
                ev = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "response.output_text.delta":
                out += ev.get("delta", "")
    return out.strip()


def yt_flat_search(query, n=8):
    """Flat search returns view_count without the bot-gate that blocks full extraction on datacenter IPs."""
    try:
        r = subprocess.run([sys.executable, "-m", "yt_dlp", "ytsearch%d:%s" % (n, query),
                            "--flat-playlist", "--dump-json", "--no-warnings"],
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
                        "duration": v.get("duration") or 0})
    return out


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
    for it in items:
        it["score"] = math.log10(it["views"] + 1) - it["pos"] * 0.04
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


def deep_fetch(url):
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
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        return urllib.request.urlopen(req, timeout=15).read().decode("utf-8", "ignore")[:1500]
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
    out.append("\n## YouTube (by views)")
    for v in yt:
        out.append("- **%s** (%s)\n  %s views — %s" % (v["title"], v.get("channel", ""),
                                                       "{:,}".format(v["views"]), v["url"]))
    raw = "\n".join(out)
    insight = codex_call("Items today:\n" + raw[:2200] + (("\n\nTop news body:\n" + detail) if detail else "") +
                         "\n\nWrite ONLY a 2-line insight on why today's flow matters. No clichés.",
                         "Output ONLY a 2-line insight, no item list.")
    insight = "\n".join([line for line in insight.splitlines() if line.strip()][:2])
    return ("💡 " + insight + "\n\n" + raw) if insight else raw


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print(run(sys.argv[1] if len(sys.argv) > 1 else "ai-tech"))
