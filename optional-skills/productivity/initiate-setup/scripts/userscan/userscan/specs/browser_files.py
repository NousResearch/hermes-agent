"""Browser, comms/work and files probes (families: browser, comms_work, files).

Ids follow reports/14-triage.json. Sources are the user's own browser profiles, Explorer registry keys
and known folders. Output is counts, histograms and category totals; content-derived values (search
terms, named domains, repo names, work hostnames, profile-root dir names) are tier T2. Cookies,
Login Data, Web Data and token stores are only stat'ed (h.meta), never opened.

Several probes share one expensive read (History copies, the known-folder walk). The first probe that
needs it computes it under a lock and caches it on the HostAccess object; the others read the cache.
"""
from __future__ import annotations

import collections
import configparser
import datetime as dt
import glob
import json
import os
import re
import shutil
import sqlite3
import statistics
import threading
import time
import urllib.parse

from userscan.registry import REGISTRY, probe

WEBKIT_OFFSET = 11644473600
FT_OFFSET = 116444736000000000

# Operator (lab) noise. ROOT applies to names directly under the profile root (side homes, scratch dirs);
# DEEP applies at any depth and only lists names that are never the user's own.
OPERATOR_RX = re.compile(r"^(hn-e2e|ns960|ns923.*|lhm|shots|user-insights-lab|userscan.*|\.?hermes-.*)$", re.I)
OPERATOR_DEEP_RX = re.compile(r"(hn-e2e|ns960|ns923|user-insights-lab|userscan-)", re.I)

HKCU_EXPLORER = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Explorer"


def _probe(id, **kw):
    """Register unless another spec module already owns the id (first registration wins)."""
    if id in REGISTRY:
        return lambda fn: fn
    return probe(id, **kw)


# ---------------------------------------------------------------- shared helpers

_GUARD = threading.Lock()


def _cached(h, key, fn):
    with _GUARD:
        cache = h.__dict__.setdefault("_bf_cache", {})
        locks = h.__dict__.setdefault("_bf_locks", {})
        lock = locks.setdefault(key, threading.Lock())
    with lock:
        if key not in cache:
            t = time.perf_counter()
            try:
                cache[key] = fn(h)
            except Exception as e:  # cache the failure so every dependent probe reports it once
                cache[key] = {"__error__": f"{type(e).__name__}: {e}"}
            cache.setdefault("_ms", {})[key] = round((time.perf_counter() - t) * 1000, 2)
        v = cache[key]
    if isinstance(v, dict) and "__error__" in v:
        raise RuntimeError(v["__error__"])
    return v


def _cache_ms(h, key):
    return h.__dict__.get("_bf_cache", {}).get("_ms", {}).get(key)


def _env(name):
    return os.environ.get(name, "")


def _home():
    return os.path.expanduser("~")


def _redact(p):
    if not p:
        return p
    home = _home()
    if p.lower().startswith(home.lower()):
        return "%USERPROFILE%" + p[len(home):]
    return p


def _isdir(p):
    try:
        return os.path.isdir(p)
    except OSError:
        return False


def _load_json(p, max_bytes=16_000_000):
    try:
        if os.path.getsize(p) > max_bytes:
            return None
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            return json.load(f)
    except Exception:
        return None


def _iso(ts):
    try:
        return dt.datetime.fromtimestamp(ts).isoformat(timespec="seconds") if ts else None
    except (OSError, ValueError, OverflowError):
        return None


def _webkit_iso(v):
    try:
        v = int(v)
        return _iso(v / 1e6 - WEBKIT_OFFSET) if v > 0 else None
    except (TypeError, ValueError):
        return None


def _ft_iso(ft):
    try:
        ft = int(ft)
        if FT_OFFSET < ft < 150000000000000000:
            return _iso((ft - FT_OFFSET) / 1e7)
    except (TypeError, ValueError):
        pass
    return None


def _winreg():
    import winreg
    return winreg


_ROOTS = {"HKLM": "HKEY_LOCAL_MACHINE", "HKCU": "HKEY_CURRENT_USER", "HKU": "HKEY_USERS"}


def _reg_key(path):
    wr = _winreg()
    hive, _, sub = path.partition("\\")
    return wr.OpenKey(getattr(wr, _ROOTS[hive]), sub, 0, wr.KEY_READ)


def _reg_values(path, limit=2000):
    """{name: value} for a key, or None if the key is missing."""
    try:
        k = _reg_key(path)
    except OSError:
        return None
    out = {}
    wr = _winreg()
    with k:
        for i in range(limit):
            try:
                n, v, _t = wr.EnumValue(k, i)
            except OSError:
                break
            out[n] = v
    return out


def _reg_lastwrite(path):
    try:
        with _reg_key(path) as k:
            return _ft_iso(_winreg().QueryInfoKey(k)[2])
    except OSError:
        return None


def _age_bucket(age_s):
    d = age_s / 86400
    for lim, name in ((1, "lt_1d"), (7, "lt_7d"), (30, "lt_30d"), (90, "lt_90d"), (365, "lt_1y"), (730, "lt_2y")):
        if d < lim:
            return name
    return "ge_2y"


AGE_KEYS = ["lt_1d", "lt_7d", "lt_30d", "lt_90d", "lt_1y", "lt_2y", "ge_2y"]

FILE_ATTRIBUTE_REPARSE_POINT = 0x400
CLOUD_ATTRS = 0x400000 | 0x40000 | 0x1000   # RECALL_ON_DATA_ACCESS | RECALL_ON_OPEN | OFFLINE
PLACEHOLDER_REPARSE_OK = 0x400000 | 0x40000 | 0x100000 | 0x80000


def _walk(root, max_depth=6, max_entries=100000, budget_s=5.0, on_file=None, skip_dir=None):
    """Bounded iterative walk: depth, entry and time caps. Skips non-cloud reparse points (junction loops)."""
    t0 = time.perf_counter()
    deadline = t0 + budget_s
    st = {"files": 0, "dirs": 0, "bytes": 0, "entries": 0, "truncated": None, "depth": max_depth,
          "cloud_placeholders": 0, "reparse_skipped": 0, "errors": 0}
    if not _isdir(root):
        st["root_exists"] = False
        return st
    stack = [(root, 0)]
    while stack:
        path, depth = stack.pop()
        try:
            it = os.scandir(path)
        except OSError:
            st["errors"] += 1
            continue
        with it:
            for e in it:
                st["entries"] += 1
                if st["entries"] > max_entries:
                    st["truncated"] = "entries"
                    stack.clear()
                    break
                if (st["entries"] & 511) == 0 and time.perf_counter() > deadline:
                    st["truncated"] = "time"
                    stack.clear()
                    break
                try:
                    s = e.stat(follow_symlinks=False)
                except OSError:
                    st["errors"] += 1
                    continue
                attrs = getattr(s, "st_file_attributes", 0)
                if attrs & FILE_ATTRIBUTE_REPARSE_POINT and not attrs & PLACEHOLDER_REPARSE_OK:
                    st["reparse_skipped"] += 1
                    continue
                if e.is_dir(follow_symlinks=False):
                    if skip_dir and skip_dir(e.name, e.path):
                        continue
                    st["dirs"] += 1
                    if depth + 1 < max_depth:
                        stack.append((e.path, depth + 1))
                else:
                    st["files"] += 1
                    st["bytes"] += s.st_size
                    if attrs & CLOUD_ATTRS:
                        st["cloud_placeholders"] += 1
                    if on_file:
                        on_file(e, s, depth)
    st["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return st


# ---------------------------------------------------------------- browser catalog

def _catalog_paths():
    L, R, H = _env("LOCALAPPDATA"), _env("APPDATA"), _home()
    chromium = [
        ("brave", rf"{L}\BraveSoftware\Brave-Browser\User Data"),
        ("edge", rf"{L}\Microsoft\Edge\User Data"),
        ("chrome", rf"{L}\Google\Chrome\User Data"),
        ("chrome_beta", rf"{L}\Google\Chrome Beta\User Data"),
        ("chrome_dev", rf"{L}\Google\Chrome Dev\User Data"),
        ("chrome_canary", rf"{L}\Google\Chrome SxS\User Data"),
        ("chromium", rf"{L}\Chromium\User Data"),
        ("edge_beta", rf"{L}\Microsoft\Edge Beta\User Data"),
        ("edge_dev", rf"{L}\Microsoft\Edge Dev\User Data"),
        ("edge_canary", rf"{L}\Microsoft\Edge SxS\User Data"),
        ("brave_beta", rf"{L}\BraveSoftware\Brave-Browser-Beta\User Data"),
        ("brave_nightly", rf"{L}\BraveSoftware\Brave-Browser-Nightly\User Data"),
        ("vivaldi", rf"{L}\Vivaldi\User Data"),
        ("opera", rf"{R}\Opera Software\Opera Stable"),
        ("opera_gx", rf"{R}\Opera Software\Opera GX Stable"),
        ("opera_air", rf"{R}\Opera Software\Opera Air Stable"),
        ("arc", rf"{L}\Packages\TheBrowserCompany.Arc_*\LocalCache\Local\Arc\User Data"),
        ("comet", rf"{L}\Perplexity\Comet\User Data"),
        ("dia", rf"{L}\Dia\User Data"),
        ("atlas", rf"{L}\OpenAI\Atlas\User Data"),
        ("thorium", rf"{L}\Thorium\User Data"),
        ("supermium", rf"{L}\Supermium\User Data"),
        ("yandex", rf"{L}\Yandex\YandexBrowser\User Data"),
        ("cent", rf"{L}\CentBrowser\User Data"),
        ("epic", rf"{L}\Epic Privacy Browser\User Data"),
        ("sidekick", rf"{L}\Sidekick\User Data"),
        ("wavebox", rf"{L}\WaveboxApp\User Data"),
    ]
    gecko = [
        ("firefox", rf"{R}\Mozilla\Firefox"),
        ("zen", rf"{R}\zen"),
        ("floorp", rf"{R}\Floorp"),
        ("librewolf", rf"{R}\librewolf"),
        ("waterfox", rf"{R}\Waterfox"),
        ("tor", rf"{H}\Desktop\Tor Browser"),
    ]
    return chromium, gecko


def _catalog(h):
    chromium, gecko = _catalog_paths()
    found_c, found_g = {}, {}
    for bid, ud in chromium:
        hits = glob.glob(ud) if "*" in ud else ([ud] if _isdir(ud) else [])
        if hits:
            found_c[bid] = hits[0]
    for bid, root in gecko:
        ok = _isdir(root) if bid == "tor" else os.path.isfile(os.path.join(root, "profiles.ini"))
        if ok:
            found_g[bid] = root
    return {"chromium": found_c, "gecko": found_g, "checked": len(chromium) + len(gecko)}


def _profiles(h):
    """[(browser_id, user_data, profile_dir, info_cache_entry)] for every chromium profile on disk."""
    cat = _cached(h, "catalog", _catalog)
    out = []
    for bid, ud in cat["chromium"].items():
        ls = _cached(h, "ls:" + bid, lambda _h, ud=ud: _load_json(os.path.join(ud, "Local State")) or {})
        cache = ((ls.get("profile") or {}).get("info_cache") or {}) if isinstance(ls, dict) else {}
        dirs = set(cache)
        try:
            for d in os.listdir(ud):
                if (d == "Default" or d.startswith("Profile ")) and os.path.isfile(os.path.join(ud, d, "Preferences")):
                    dirs.add(d)
        except OSError:
            pass
        if bid.startswith("opera"):
            dirs = {""}
        for d in sorted(dirs):
            if d and not _isdir(os.path.join(ud, d)):
                continue
            out.append((bid, ud, d, cache.get(d, {}) if d else {}))
    return out


def _pkey(bid, pdir):
    return f"{bid}:{pdir or 'root'}"


# ---------------------------------------------------------------- domain categories

DOMAIN_CATS = {
    "ai": ["chatgpt.com", "openai.com", "claude.ai", "anthropic.com", "gemini.google.com", "perplexity.ai",
           "huggingface.co", "nousresearch.com", "openrouter.ai", "grok.com", "x.ai", "mistral.ai", "deepseek.com",
           "aistudio.google.com", "ollama.com", "replicate.com", "civitai.com", "lmarena.ai", "together.ai",
           "groq.com", "fal.ai", "cursor.com", "copilot.microsoft.com", "notebooklm.google.com", "chat.qwen.ai",
           "kimi.com", "z.ai", "moonshot.ai", "cerebras.ai", "entelligence.ai", "deepmind.google", "wisprflow.ai"],
    "dev": ["github.com", "gitlab.com", "stackoverflow.com", "stackexchange.com", "npmjs.com", "pypi.org",
            "docs.python.org", "developer.mozilla.org", "learn.microsoft.com", "vercel.com", "netlify.com",
            "readthedocs.io", "docs.rs", "crates.io", "githubusercontent.com", "github.io", "go.dev",
            "rust-lang.org", "nodejs.org", "docker.com", "cloudflare.com", "supabase.com", "railway.app", "fly.io",
            "render.com", "linear.app", "sentry.io", "betterstack.com", "astral.sh", "nvidia.com", "tailscale.com",
            "digitalocean.com", "aws.amazon.com", "console.cloud.google.com", "portal.azure.com", "localhost",
            "127.0.0.1", "codeberg.org", "sourceforge.net", "jsdelivr.com", "unpkg.com", "regex101.com"],
    "search": ["google.com", "bing.com", "duckduckgo.com", "search.brave.com", "kagi.com", "yandex.com",
               "ecosia.org", "startpage.com"],
    "social": ["x.com", "twitter.com", "reddit.com", "linkedin.com", "facebook.com", "instagram.com", "threads.net",
               "bsky.app", "mastodon.social", "discord.com", "tiktok.com", "news.ycombinator.com", "t.co",
               "whatsapp.com", "telegram.org", "quora.com"],
    "video": ["youtube.com", "youtu.be", "twitch.tv", "netflix.com", "primevideo.com", "hotstar.com", "spotify.com",
              "vimeo.com", "crunchyroll.com", "soundcloud.com", "music.youtube.com", "jiocinema.com",
              "disneyplus.com", "kick.com"],
    "gaming": ["steampowered.com", "steamcommunity.com", "epicgames.com", "nexusmods.com", "ign.com", "gamespot.com",
               "pcgamingwiki.com", "protondb.com", "gog.com", "riotgames.com", "playvalorant.com",
               "leagueoflegends.com", "fandom.com", "ea.com", "xbox.com", "playstation.com", "battle.net",
               "howlongtobeat.com", "speedrun.com", "rockstargames.com", "curseforge.com", "modrinth.com",
               "gta5-mods.com"],
    "hardware": ["techpowerup.com", "tomshardware.com", "anandtech.com", "rtings.com", "pcpartpicker.com",
                 "asus.com", "msi.com", "intel.com", "amd.com", "geforce.com", "nvidia.in", "guru3d.com",
                 "overclock.net", "hwinfo.com", "testufo.com", "blurbusters.com", "displayspecifications.com"],
    "shopping": ["amazon.in", "amazon.com", "flipkart.com", "ebay.com", "aliexpress.com", "myntra.com",
                 "bigbasket.com", "swiggy.com", "zomato.com", "blinkit.com", "zeptonow.com", "etsy.com",
                 "bestbuy.com", "newegg.com", "croma.com", "reliancedigital.in", "ajio.com", "nykaa.com"],
    "productivity": ["mail.google.com", "gmail.com", "docs.google.com", "drive.google.com", "calendar.google.com",
                     "outlook.live.com", "outlook.office.com", "notion.so", "notion.site", "office.com",
                     "microsoft365.com", "onedrive.live.com", "dropbox.com", "figma.com", "slack.com", "zoom.us",
                     "meet.google.com", "teams.microsoft.com", "trello.com", "airtable.com", "miro.com", "canva.com",
                     "sheets.google.com", "box.com", "accounts.google.com", "myaccount.google.com"],
    "news_reading": ["medium.com", "substack.com", "nytimes.com", "theverge.com", "arstechnica.com", "bbc.com",
                     "bbc.co.uk", "theguardian.com", "wikipedia.org", "arxiv.org", "techcrunch.com", "bloomberg.com",
                     "reuters.com", "thehindu.com", "indiatimes.com", "ndtv.com", "wired.com", "moneycontrol.com",
                     "livemint.com"],
    "finance": ["zerodha.com", "groww.in", "paypal.com", "stripe.com", "hdfcbank.com", "icicibank.com", "sbi.co.in",
                "coinbase.com", "binance.com", "tradingview.com", "wise.com", "paytm.com", "phonepe.com",
                "razorpay.com", "incometax.gov.in"],
    "travel": ["booking.com", "airbnb.com", "makemytrip.com", "goibibo.com", "expedia.com", "skyscanner.com",
               "cleartrip.com", "irctc.co.in", "uber.com", "olacabs.com", "tripadvisor.com", "maps.google.com",
               "agoda.com", "ixigo.com"],
    "password_manager": ["bitwarden.com", "1password.com", "lastpass.com", "keepersecurity.com", "proton.me"],
    "media_unofficial": ["1337x.to", "rarbgdump.com", "fmhy.net", "animepahe.pw", "animepahe.ru",
                         "fitgirl-repacks.site", "subtitlecat.com", "my-subs.co", "opensubtitles.org",
                         "thepiratebay.org", "nyaa.si", "yts.mx", "yts-official.biz", "pixeldrain.com"],
}
_SUFFIX = {d: c for c, ds in DOMAIN_CATS.items() for d in ds}
# Hosts that are never named in any output (torrent, unofficial streaming, adult).
SENSITIVE_RX = re.compile(r"(torrent|1337x|piratebay|yts|nyaa|rarbg|fitgirl|animepahe|anime|hentai|porn|xxx|xvideos|"
                          r"xnxx|onlyfans|nsfw|subtitle|my-subs|pixeldrain|\bsex)", re.I)
SENSITIVE_CATS = {"media_unofficial", "adult"}


def _categorize(host):
    if not host:
        return "other"
    if "://" in host or host.endswith(":"):
        return "local_file" if host.startswith("file") else "browser_internal"
    if SENSITIVE_RX.search(host) and _SUFFIX.get(host) is None:
        return "adult" if re.search(r"porn|xxx|xvideos|xnxx|onlyfans|hentai|nsfw|\bsex", host) else "media_unofficial"
    parts = host.split(".")
    for i in range(len(parts)):
        c = _SUFFIX.get(".".join(parts[i:]))
        if c:
            return c
    if host.endswith((".local", ".ts.net")) or re.match(r"^(10|192\.168|172\.(1[6-9]|2\d|3[01]))\.", host):
        return "dev"
    return "other"


def _host_of(url):
    try:
        p = urllib.parse.urlsplit(url)
    except ValueError:
        return "?", None
    if p.scheme in ("http", "https"):
        hn = (p.hostname or "").lower()
        return (hn[4:] if hn.startswith("www.") else hn), p
    if p.scheme == "file":
        return "file://", p
    return (p.scheme + "://" + (p.hostname or "")) if p.scheme else "?", p


SERVICES = {
    "discord": r"(^|\.)discord\.com$", "slack": r"(^|\.)slack\.com$", "teams_work": r"^teams\.(microsoft|cloud\.microsoft)(\.com)?$",
    "teams_personal": r"^teams\.live\.com$", "zoom": r"(^|\.)zoom\.us$", "google_meet": r"^meet\.google\.com$",
    "whatsapp_web": r"^web\.whatsapp\.com$", "telegram_web": r"^web\.telegram\.org$", "gmail": r"^mail\.google\.com$",
    "google_calendar": r"^calendar\.google\.com$", "outlook_personal": r"^outlook\.live\.com$",
    "outlook_work": r"^outlook\.(office|office365|cloud\.microsoft)(\.com)?$", "proton_mail": r"^mail\.proton\.me$",
    "notion": r"(^|\.)notion\.(so|site|com)$", "google_docs": r"^(docs|drive|sheets)\.google\.com$",
    "office_web": r"(^|\.)(office\.com|microsoft365\.com|sharepoint\.com)$", "linear": r"^linear\.app$",
    "github": r"^github\.com$", "figma": r"(^|\.)figma\.com$", "nousresearch": r"(^|\.)nousresearch\.com$",
}
_SVC_RX = [(k, re.compile(v)) for k, v in SERVICES.items()]
WORK_HOST_SVCS = ("slack", "nousresearch", "discord", "notion", "linear")

TERM_THEMES = {
    "dev": ["error", "exception", "failed", "traceback", "npm", "pip", "python", "rust", "typescript", "javascript",
            "node", "docker", "git", "github", "powershell", "wsl", "api", "sdk", "cli", "linux", "ssh", "regex",
            "json", "sql", "cuda", "react", "electron", "vite", "compile", "build", "repo:"],
    "ai": ["gpt", "claude", "llm", "openai", "anthropic", "gemini", "ollama", "hermes", "model", "agent", "llama",
           "qwen", "deepseek", "mcp", "prompt", "diffusion", "comfyui", "lora", "nous", " ai "],
    "gaming": ["game", "steam", "mod", "walkthrough", "boss", "elden", "cyberpunk", "witcher", "valorant", "forza",
               "death stranding", "helldivers", "gta", "fps", "patch notes", "trophy", "achievement", "control",
               "expedition 33", "arc raiders"],
    "hardware": ["rtx", "gpu", "nvidia", "cpu", "ram", "ssd", "motherboard", "bios", "driver", "monitor",
                 "overclock", "undervolt", "psu", "5090", "14700k", "z790", "laptop", "arm64", "snapdragon", "switch"],
    "shopping": ["buy", "price", "review", " vs ", "best ", "deal", "cheap", "discount", "amazon", "flipkart"],
    "howto": ["how to", "how do", "what is", "why ", "tutorial", "guide", "example"],
    "media": ["movie", "series", "episode", "season", "song", "lyrics", "trailer", "anime", "netflix", "youtube",
              "watch", "cast", "ranked"],
}
SAFE_TERM = re.compile(r"^[\w\s\-\.\+#:'/?&,()]{2,60}$", re.UNICODE)


def _themes(term):
    t = " " + term.lower() + " "
    return [k for k, ws in TERM_THEMES.items() if any(w in t for w in ws)] or ["other"]


def _safe_term(term):
    if not SAFE_TERM.match(term) or "@" in term or re.search(r"\d{6,}|[A-Za-z0-9_\-]{24,}", term):
        return False
    if re.search(r"(password|passwd|token|secret|api[_ ]?key|otp|login|aadhaar|pan card)", term, re.I):
        return False
    if SENSITIVE_RX.search(term) or re.search(r"\b(fuck\w*|shit\w*|nude\w*)\b", term, re.I):
        return False
    return True


# Skill SQL (browser-insights/references/sql-queries.md): pages per day and site buckets, on the urls table.
SQL_PAGES_PER_DAY = ("SELECT ROUND(CAST(COUNT(*) AS FLOAT) / COUNT(DISTINCT date(datetime((last_visit_time/1000000)"
                     "-11644473600, 'unixepoch', 'localtime'))), 1) FROM urls")
SQL_SITE_BUCKETS = """SELECT CASE
    WHEN url LIKE '%github.com%' THEN 'GitHub'
    WHEN url LIKE '%google.com%' THEN 'Google'
    WHEN url LIKE '%youtube.com%' THEN 'YouTube'
    WHEN url LIKE '%twitter.com%' OR url LIKE '%x.com%' THEN 'Twitter/X'
    WHEN url LIKE '%reddit.com%' THEN 'Reddit'
    WHEN url LIKE '%linkedin.com%' THEN 'LinkedIn'
    WHEN url LIKE '%stackoverflow.com%' THEN 'StackOverflow'
    WHEN url LIKE '%chatgpt.com%' OR url LIKE '%openai.com%' THEN 'OpenAI/ChatGPT'
    WHEN url LIKE '%claude.ai%' OR url LIKE '%anthropic.com%' THEN 'Claude/Anthropic'
    WHEN url LIKE '%localhost%' OR url LIKE '%127.0.0.1%' THEN 'Localhost Dev'
    WHEN url LIKE '%netflix.com%' THEN 'Netflix'
    WHEN url LIKE '%spotify.com%' THEN 'Spotify'
    WHEN url LIKE '%amazon.%' THEN 'Amazon'
    ELSE 'Other' END AS site, COUNT(*) FROM urls GROUP BY site ORDER BY 2 DESC"""
LT = f"datetime((visit_time/1000000)-{WEBKIT_OFFSET}, 'unixepoch', 'localtime')"
MAX_VISIT_ROWS = 300000


def _copy_db(h, src, tag):
    """Copy a live SQLite DB plus -wal/-journal into h.scratch() under a unique name. (path, via, ms)."""
    t = time.perf_counter()
    dst = os.path.join(h.scratch(), re.sub(r"[^A-Za-z0-9_.-]", "_", tag))
    via = "copy"
    try:
        shutil.copyfile(src, dst)
    except OSError:
        got, rung = h.copy_locked(src, os.path.basename(dst))
        if not got:
            return None, "failed", round((time.perf_counter() - t) * 1000, 2)
        via = rung
    for suf in ("-wal", "-journal"):
        if os.path.exists(src + suf):
            try:
                shutil.copyfile(src + suf, dst + suf)
            except OSError:
                via += "+wal_locked"
    return dst, via, round((time.perf_counter() - t) * 1000, 2)


def _drop_db(dst):
    for suf in ("", "-wal", "-journal", "-shm"):
        try:
            os.remove(dst + suf)
        except OSError:
            pass


def _query_history(db):
    """All aggregates for one Chromium History copy. Raw domain/term counters stay in memory only."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        cur = con.cursor()
        tables = {r[0] for r in cur.execute("select name from sqlite_master where type='table'")}
        r = {"urls": cur.execute("select count(*) from urls").fetchone()[0],
             "visits": cur.execute("select count(*) from visits").fetchone()[0]}
        mn, mx = cur.execute("select min(visit_time), max(visit_time) from visits").fetchone()
        r["first_visit"], r["last_visit"] = _webkit_iso(mn), _webkit_iso(mx)
        r["active_days"] = cur.execute(f"select count(distinct date({LT})) from visits").fetchone()[0]
        r["visits_per_active_day"] = round(r["visits"] / r["active_days"], 1) if r["active_days"] else 0
        try:
            r["pages_per_day_urls"] = cur.execute(SQL_PAGES_PER_DAY).fetchone()[0]
        except sqlite3.Error:
            r["pages_per_day_urls"] = None
        r["typed_urls"] = cur.execute("select count(*) from urls where typed_count>0").fetchone()[0]
        hours = [0] * 24
        for hr, c in cur.execute(f"select cast(strftime('%H',{LT}) as int), count(*) from visits group by 1"):
            if hr is not None:
                hours[int(hr)] = c
        wd = [0] * 7
        for d, c in cur.execute(f"select cast(strftime('%w',{LT}) as int), count(*) from visits group by 1"):
            if d is not None:
                wd[int(d)] = c
        r["hours"], r["weekdays_sun0"] = hours, wd
        r["months"] = {m: c for m, c in cur.execute(f"select strftime('%Y-%m',{LT}), count(*) from visits group by 1")}
        b = cur.execute(f"select date({LT}) d, count(*) c from visits group by d order by c desc limit 1").fetchone()
        r["busiest_day"] = {"date": b[0], "visits": b[1]} if b else None
        try:
            r["site_buckets"] = {s: c for s, c in cur.execute(SQL_SITE_BUCKETS)}
        except sqlite3.Error:
            r["site_buckets"] = {}
        dom, cats = collections.Counter(), collections.Counter()
        gh, ports = collections.Counter(), collections.Counter()
        svc = {}
        ids = {"discord_guilds": set(), "slack_workspaces": set(), "gmail_slots": set(), "linear_workspaces": set()}
        work_hosts = {}
        now = time.time()
        for url, vt in cur.execute(f"select u.url, v.visit_time from visits v join urls u on u.id=v.url "
                                   f"limit {MAX_VISIT_ROWS}"):
            host, sp = _host_of(url or "")
            dom[host] += 1
            cats[_categorize(host)] += 1
            if sp is None:
                continue
            if host == "github.com":
                parts = sp.path.strip("/").split("/")
                if len(parts) >= 2 and parts[0] not in ("settings", "notifications", "orgs", "search", "login",
                                                        "sessions", "marketplace", "features", "topics", "apps"):
                    gh[parts[0] + "/" + parts[1]] += 1
            elif host in ("localhost", "127.0.0.1"):
                try:
                    ports[sp.port or 80] += 1
                except ValueError:
                    pass
            for k, rx in _SVC_RX:
                if rx.search(host):
                    t = (vt or 0) / 1e6 - WEBKIT_OFFSET
                    a = svc.setdefault(k, {"visits": 0, "days": set(), "last": 0, "visits_30d": 0})
                    a["visits"] += 1
                    a["days"].add(time.strftime("%Y-%m-%d", time.localtime(max(t, 0))))
                    a["last"] = max(a["last"], t)
                    if now - t < 30 * 86400:
                        a["visits_30d"] += 1
                    if k in WORK_HOST_SVCS:
                        work_hosts.setdefault(k, collections.Counter())[host] += 1
                    path = sp.path
                    m = k == "discord" and re.match(r"/channels/(\d{15,21})/", path)
                    if m:
                        ids["discord_guilds"].add(m.group(1))
                    m = k == "slack" and re.match(r"/client/(T[A-Z0-9]+)", path)
                    if m:
                        ids["slack_workspaces"].add(m.group(1))
                    m = k == "gmail" and re.match(r"/mail/u/(\d+)", path)
                    if m:
                        ids["gmail_slots"].add(m.group(1))
                    m = k == "linear" and re.match(r"/([a-z0-9-]+)/", path)
                    if m and m.group(1) not in ("login", "join", "settings", "invite"):
                        ids["linear_workspaces"].add(m.group(1))
                    break
        r["_domains"], r["_categories"] = dom, cats
        r["_github"], r["_ports"] = gh, ports
        r["_services"] = {k: {"visits": v["visits"], "active_days": len(v["days"]), "visits_30d": v["visits_30d"],
                              "last": _iso(v["last"])} for k, v in svc.items()}
        r["_ids"] = {k: len(v) for k, v in ids.items()}
        r["_work_hosts"] = work_hosts
        if "downloads" in tables:
            exts = collections.Counter()
            n = 0
            for (tp,) in cur.execute("select target_path from downloads limit 50000"):
                n += 1
                e = os.path.splitext(tp or "")[1].lower()
                exts[e if re.fullmatch(r"\.[a-z0-9]{1,6}", e or "") else "(other)"] += 1
            r["downloads"] = {"count": n, "by_ext": dict(exts.most_common(10))}
        if "keyword_search_terms" in tables:
            terms, engines = collections.Counter(), collections.Counter()
            for term, url in cur.execute("select k.term, u.url from keyword_search_terms k "
                                         "left join urls u on u.id=k.url_id limit 50000"):
                if term:
                    terms[term] += 1
                    engines[_host_of(url or "")[0]] += 1
            r["_terms"], r["_engines"] = terms, engines
        return r
    finally:
        con.close()


def _history(h):
    """{profile_key: aggregates} over every chromium profile History, plus per-profile copy timings."""
    out = {}
    for bid, ud, pdir, _info in _profiles(h):
        src = os.path.join(ud, pdir, "History") if pdir else os.path.join(ud, "History")
        if not os.path.isfile(src):
            continue
        key = _pkey(bid, pdir)
        db, via, cms = _copy_db(h, src, f"hist_{key}")
        row = {"browser": bid, "profile_dir": pdir or ".", "is_default_profile": pdir in ("Default", ""),
               "bytes": os.path.getsize(src), "via": via, "copy_ms": cms}
        if db:
            t = time.perf_counter()
            try:
                row.update(_query_history(db))
            except sqlite3.Error as e:
                row["error"] = f"sqlite: {e}"
            finally:
                _drop_db(db)
            row["query_ms"] = round((time.perf_counter() - t) * 1000, 2)
        out[key] = row
    return out


def _gecko_history(h):
    out = {}
    cat = _cached(h, "catalog", _catalog)
    for bid, root in cat["gecko"].items():
        ini = os.path.join(root, "profiles.ini")
        cp = configparser.ConfigParser()
        try:
            cp.read(ini, encoding="utf-8")
        except (configparser.Error, OSError):
            continue
        for sec in cp.sections():
            if not sec.startswith("Profile") or "Path" not in cp[sec]:
                continue
            p = cp[sec]["Path"]
            pdir = os.path.join(root, p) if cp[sec].get("IsRelative", "1") == "1" else p
            src = os.path.join(pdir, "places.sqlite")
            if not os.path.isfile(src):
                continue
            key = f"{bid}:{sec}"
            db, via, cms = _copy_db(h, src, f"places_{key}")
            row = {"browser": bid, "via": via, "copy_ms": cms, "bytes": os.path.getsize(src)}
            if db:
                try:
                    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
                    lt = "datetime(visit_date/1000000, 'unixepoch', 'localtime')"
                    row["visits"] = con.execute("select count(*) from moz_historyvisits").fetchone()[0]
                    row["urls"] = con.execute("select count(*) from moz_places").fetchone()[0]
                    mn, mx = con.execute("select min(visit_date), max(visit_date) from moz_historyvisits").fetchone()
                    row["first_visit"] = _iso((mn or 0) / 1e6)
                    row["last_visit"] = _iso((mx or 0) / 1e6)
                    row["active_days"] = con.execute(f"select count(distinct date({lt})) from moz_historyvisits").fetchone()[0]
                    hours = [0] * 24
                    for hr, c in con.execute(f"select cast(strftime('%H',{lt}) as int), count(*) from moz_historyvisits group by 1"):
                        if hr is not None:
                            hours[int(hr)] = c
                    row["hours"] = hours
                    cats = collections.Counter()
                    for (url,) in con.execute("select p.url from moz_historyvisits v join moz_places p on p.id=v.place_id "
                                              f"limit {MAX_VISIT_ROWS}"):
                        cats[_categorize(_host_of(url or "")[0])] += 1
                    row["category_visits"] = dict(cats.most_common())
                    con.close()
                except sqlite3.Error as e:
                    row["error"] = f"sqlite: {e}"
                finally:
                    _drop_db(db)
            out[key] = row
    return out


def _hist_rows(h):
    return [r for r in _cached(h, "history", _history).values() if "visits" in r]


def _public(r, keys):
    return {k: r.get(k) for k in keys if k in r}


# ================================================================ family: browser — L1

@_probe(id="browser.brave.present", level="L1", family="browser", tier="T0", collect="core")
def brave_present(h, facts):
    """Brave user-data dir exists."""
    m = h.meta(r"%LOCALAPPDATA%\BraveSoftware\Brave-Browser\User Data\Local State")
    return {"present": _isdir(h.expand(r"%LOCALAPPDATA%\BraveSoftware\Brave-Browser\User Data")),
            "local_state_mtime": _iso(m.get("mtime"))}


@_probe(id="browser.edge.present", level="L1", family="browser", tier="T0", collect="core")
def edge_present(h, facts):
    """Edge user-data dir exists (Edge is preinstalled; presence is not use)."""
    m = h.meta(r"%LOCALAPPDATA%\Microsoft\Edge\User Data\Local State")
    return {"present": _isdir(h.expand(r"%LOCALAPPDATA%\Microsoft\Edge\User Data")),
            "local_state_mtime": _iso(m.get("mtime"))}


@_probe(id="browser.catalog", level="L1", family="browser", tier="T0", collect="core")
def browser_catalog(h, facts):
    """Which catalogued Chromium/Gecko browsers have a user-data root (33 path stats)."""
    c = _cached(h, "catalog", _catalog)
    return {"present": bool(c["chromium"] or c["gecko"]), "chromium": sorted(c["chromium"]),
            "gecko": sorted(c["gecko"]), "checked": c["checked"]}


PROGID_MAP = {"bravehtml": "brave", "bravepdf": "brave", "msedgehtm": "edge", "msedgepdf": "edge",
              "chromehtml": "chrome", "firefoxurl": "firefox", "firefoxhtml": "firefox", "vivaldihtm": "vivaldi",
              "operastable": "opera", "operagxstable": "opera_gx", "thoriumhtm": "thorium", "comethtml": "comet",
              "zenhtml": "zen", "ie.http": "iexplore", "appxq0fevzme2pys62n3e0fbqa7peapykr8v": "edge_legacy"}


def _progid(kind, name):
    base = (rf"{HKCU_EXPLORER}\FileExts\{name}" if kind == "ext" else
            rf"HKCU\Software\Microsoft\Windows\Shell\Associations\UrlAssociations\{name}")
    for sub in ("UserChoiceLatest", "UserChoice"):
        v = _reg_values(base + "\\" + sub)
        if v and v.get("ProgId"):
            return v["ProgId"]
    return None


def _resolve_progid(pid):
    p = (pid or "").lower()
    hit = next((v for k, v in PROGID_MAP.items() if p.startswith(k)), None)
    if hit:
        return hit
    if p.startswith("appx"):
        v = _reg_values(rf"HKCU\Software\Classes\{pid}\Application") or {}
        aumid = v.get("AppUserModelId") or v.get("AppUserModelID")
        if not aumid:
            try:
                wr = _winreg()
                with wr.OpenKey(wr.HKEY_CLASSES_ROOT, pid + r"\Application") as k:
                    aumid = wr.QueryValueEx(k, "AppUserModelID")[0]
            except OSError:
                aumid = None
        if aumid:
            return aumid.split("_")[0].split("!")[0]
    return pid


@_probe(id="browser.default", level="L1", family="browser", tier="T0", collect="core")
def browser_default(h, facts):
    """Default browser from UserChoice ProgId (http/https/.html/.pdf) and when it was set."""
    got = {k: _progid(kind, n) for k, kind, n in (("http", "url", "http"), ("https", "url", "https"),
                                                  (".html", "ext", ".html"), (".pdf", "ext", ".pdf"))}
    pid = got.get("https") or got.get("http")
    if not pid:
        return {"present": False}
    set_at = _reg_lastwrite(r"HKCU\Software\Microsoft\Windows\Shell\Associations\UrlAssociations\https\UserChoice")
    return {"present": True, "browser": _resolve_progid(pid), "prog_id": pid,
            "handlers": {k: _resolve_progid(v) for k, v in got.items() if v}, "userchoice_set": set_at}


@_probe(id="browser.registered", level="L1", family="browser", tier="T0", collect="core")
def browser_registered(h, facts):
    """Browsers registered with Windows (StartMenuInternet + App Paths)."""
    names = set()
    for p in (r"HKLM\SOFTWARE\Clients\StartMenuInternet", r"HKCU\SOFTWARE\Clients\StartMenuInternet",
              r"HKLM\SOFTWARE\WOW6432Node\Clients\StartMenuInternet"):
        names.update(h.reg(p) or [])
    rx = re.compile(r"^(chrome|msedge|brave|firefox|vivaldi|opera|launcher|comet|dia|thorium|zen|floorp|librewolf|"
                    r"waterfox|arc)\.exe$", re.I)
    app_paths = sorted(k.lower() for k in (h.reg(r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths") or [])
                       if rx.match(k))
    return {"present": bool(names or app_paths), "start_menu_internet": sorted(names), "app_paths": app_paths}


@_probe(id="defaults.mailto_pdf_media", level="L1", family="browser", tier="T0", collect="core")
def defaults_mailto_pdf_media(h, facts):
    """Default handlers for mailto, PDF, video, audio and images."""
    got = {"mailto": _progid("url", "mailto")}
    for ext in (".pdf", ".mp4", ".mkv", ".mp3", ".jpg", ".png", ".txt", ".md", ".py"):
        got[ext] = _progid("ext", ext)
    got = {k: _resolve_progid(v) for k, v in got.items() if v}
    return {"present": bool(got), "handlers": got}


T3_FILES = ("Network\\Cookies", "Cookies", "Login Data", "Login Data For Account", "Web Data", "Account Web Data")


@_probe(id="browser.t3_presence", level="L1", family="browser", tier="T3", collect="core")
def browser_t3_presence(h, facts):
    """Credential/cookie stores per profile: stat only (size), never opened."""
    rows = {}
    for bid, ud, pdir, _i in _profiles(h):
        base = os.path.join(ud, pdir) if pdir else ud
        sizes = {}
        for f in T3_FILES:
            m = h.meta(os.path.join(base, f))
            if m.get("present"):
                sizes[f.replace("Network\\", "")] = m["bytes"]
        if sizes:
            rows[_pkey(bid, pdir)] = sizes
    return {"present": bool(rows), "profiles": rows}


@_probe(id="browser.generic_sweep", level="L1", family="browser", tier="T0", collect="extended")
def browser_generic_sweep(h, facts):
    """Glob for Chromium forks not in the catalog (User Data dirs with Local State + Default)."""
    known = {os.path.normcase(p) for p in _cached(h, "catalog", _catalog)["chromium"].values()}
    L, R = _env("LOCALAPPDATA"), _env("APPDATA")
    unknown = []
    for pat in (rf"{L}\*\User Data\Local State", rf"{L}\*\*\User Data\Local State", rf"{R}\*\User Data\Local State"):
        for p in glob.glob(pat)[:200]:
            d = os.path.dirname(p)
            if os.path.normcase(d) not in known and _isdir(os.path.join(d, "Default")) and not OPERATOR_DEEP_RX.search(d):
                unknown.append(_redact(d))
    return {"present": True, "unknown_forks": len(unknown), "roots": unknown[:10]}


# ================================================================ family: browser — L2

def _vkey(s):
    return tuple(int(x) for x in s.split(".") if x.isdigit())


def _installed_version(ud):
    """Highest version dir under the browser's Application folder(s); 'Last Version' lags until next launch."""
    rel = os.path.relpath(os.path.dirname(ud), _env("LOCALAPPDATA")) if ud.lower().startswith(
        _env("LOCALAPPDATA").lower()) else None
    if not rel or rel.startswith(".."):
        return None
    vers = []
    for base in (_env("LOCALAPPDATA"), _env("ProgramFiles"), _env("ProgramFiles(x86)")):
        app = os.path.join(base, rel, "Application")
        try:
            vers += [d for d in os.listdir(app) if re.fullmatch(r"\d+(\.\d+){2,3}", d)]
        except OSError:
            continue
    return max(vers, key=_vkey) if vers else None


@_probe(id="browser.local_state", level="L2", family="browser", tier="T0", collect="core", gate="browser.catalog")
def browser_local_state(h, facts):
    """Per-browser Local State: version, profile count, install date, last live, launch count."""
    out = {}
    for bid, ud in _cached(h, "catalog", _catalog)["chromium"].items():
        ls = _cached(h, "ls:" + bid, lambda _h, ud=ud: _load_json(os.path.join(ud, "Local State")) or {})
        if not isinstance(ls, dict):
            continue
        cache = (ls.get("profile") or {}).get("info_cache") or {}
        un = ls.get("uninstall_metrics") or {}
        stab = (ls.get("user_experience_metrics") or {}).get("stability") or {}
        try:
            with open(os.path.join(ud, "Last Version"), encoding="utf-8") as f:
                lv = f.read().strip()[:40]
        except OSError:
            lv = None
        inst = un.get("installation_date2")
        out[bid] = {"version": lv, "installed_version": _installed_version(ud), "profiles": len(cache),
                    "custom_named_profiles": sum(1 for v in cache.values() if not v.get("is_using_default_name", True)),
                    "installation_date": _iso(int(inst)) if str(inst or "").isdigit() else None,
                    "launch_count": un.get("launch_count"),
                    "browser_last_live": _webkit_iso(stab.get("browser_last_live_timestamp"))}
    return {"present": bool(out), "browsers": out}


@_probe(id="browser.tab_stats", level="L2", family="browser", tier="T1", collect="core", gate="browser.catalog")
def browser_tab_stats(h, facts):
    """Max tabs per window, max windows (Local State tab_stats)."""
    out = {}
    for bid, ud in _cached(h, "catalog", _catalog)["chromium"].items():
        ls = _cached(h, "ls:" + bid, lambda _h, ud=ud: _load_json(os.path.join(ud, "Local State")) or {})
        ts = (ls.get("tab_stats") or {}) if isinstance(ls, dict) else {}
        vals = {k: v for k, v in ts.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
                and k in ("max_tabs_per_window", "total_tab_count_max", "window_count_max")}
        if vals:
            out[bid] = vals
    return {"present": bool(out), "browsers": out}


@_probe(id="browser.profile_prefs", level="L2", family="browser", tier="T0", collect="core", gate="browser.catalog")
def browser_profile_prefs(h, facts):
    """Per profile: creation time, created-by version, exit type, signed-in, sync requested, search provider."""
    rows, signed = {}, 0
    for bid, ud, pdir, info in _profiles(h):
        p = os.path.join(ud, pdir, "Preferences") if pdir else os.path.join(ud, "Preferences")
        pr = _load_json(p)
        if not isinstance(pr, dict):
            continue
        prof = pr.get("profile") or {}
        acct = len(pr.get("account_info") or [])
        is_signed = bool(acct or info.get("gaia_id") or info.get("user_name"))
        signed += is_signed
        dsp = ((pr.get("default_search_provider_data") or {}).get("template_url_data") or {}).get("short_name")
        rows[_pkey(bid, pdir)] = {
            "default_profile": pdir in ("Default", ""),
            "custom_name": info.get("is_using_default_name") is False,
            "creation_time": _webkit_iso(prof.get("creation_time")),
            "created_by_version": prof.get("created_by_version"),
            "exit_type": prof.get("exit_type"),
            "signed_in": is_signed,
            "sync_requested": (pr.get("sync") or {}).get("requested"),
            "default_search_provider": dsp,
        }
    return {"present": bool(rows), "profiles": len(rows), "signed_in_profiles": signed, "by_profile": rows}


@_probe(id="browser.extensions", level="L2", family="browser", tier="T0", collect="core", gate="browser.catalog")
def browser_extensions(h, facts):
    """Extension counts per profile; names only for user-installed ones (location 1 webstore / 4 unpacked)."""
    rows, user_names = {}, []
    for bid, ud, pdir, _i in _profiles(h):
        base = os.path.join(ud, pdir) if pdir else ud
        settings = {}
        for src in ("Secure Preferences", "Preferences"):
            j = _load_json(os.path.join(base, src))
            if isinstance(j, dict):
                settings.update(((j.get("extensions") or {}).get("settings")) or {})
        on_disk = [e for e in h.list_dir(os.path.join(base, "Extensions"), 300) if e != "Temp"]
        user = [k for k, v in settings.items() if isinstance(v, dict) and v.get("location") in (1, 4)]
        for eid in user[:30]:
            name = None
            vers = sorted(h.list_dir(os.path.join(base, "Extensions", eid), 20))
            if vers:
                man = _load_json(os.path.join(base, "Extensions", eid, vers[-1], "manifest.json")) or {}
                name = man.get("name") if isinstance(man, dict) else None
                if isinstance(name, str) and name.startswith("__MSG_"):
                    key = name[6:-2].lower()
                    for loc in ((man.get("default_locale") or "en"), "en", "en_US"):
                        msgs = _load_json(os.path.join(base, "Extensions", eid, vers[-1], "_locales", loc,
                                                       "messages.json")) or {}
                        hit = next((v.get("message") for k, v in msgs.items()
                                    if k.lower() == key and isinstance(v, dict)), None) if isinstance(msgs, dict) else None
                        if hit:
                            name = hit
                            break
            user_names.append(name or (settings[eid].get("manifest") or {}).get("name") or eid)
        rows[_pkey(bid, pdir)] = {"on_disk": len(on_disk), "settings_entries": len(settings), "user_installed": len(user)}
    return {"present": bool(rows), "user_installed": len(user_names), "user_installed_names": sorted(set(user_names)),
            "by_profile": rows}


@_probe(id="browser.bookmarks", level="L2", family="browser", tier="T1", collect="core", gate="browser.catalog")
def browser_bookmarks(h, facts):
    """Bookmark counts per profile and by URL category (no titles or URLs)."""
    rows, total, cats = {}, 0, collections.Counter()
    for bid, ud, pdir, _i in _profiles(h):
        bm = _load_json(os.path.join(ud, pdir, "Bookmarks") if pdir else os.path.join(ud, "Bookmarks"))
        if not isinstance(bm, dict) or "roots" not in bm:
            continue
        cnt = {"urls": 0, "folders": 0}
        stack = [(v, 0) for v in bm["roots"].values() if isinstance(v, dict)]
        seen = 0
        while stack and seen < 50000:
            n, depth = stack.pop()
            seen += 1
            if n.get("type") == "url":
                cnt["urls"] += 1
                cats[_categorize(_host_of(n.get("url", ""))[0])] += 1
            elif n.get("type") == "folder" and depth < 30:
                if depth > 0:
                    cnt["folders"] += 1
                stack.extend((c, depth + 1) for c in n.get("children", []) if isinstance(c, dict))
        rows[_pkey(bid, pdir)] = cnt
        total += cnt["urls"]
    return {"present": True, "total": total, "by_profile": rows,
            "by_category": {k: v for k, v in cats.items() if k not in SENSITIVE_CATS} |
            ({"sensitive": sum(cats[c] for c in SENSITIVE_CATS)} if any(cats[c] for c in SENSITIVE_CATS) else {})}


@_probe(id="browser.history.copy", level="L2", family="browser", tier="T1", collect="extended", gate="browser.catalog")
def history_copy(h, facts):
    """History DB copies (with -wal/-journal) per profile: size, copy rung, copy/query ms."""
    hist = _cached(h, "history", _history)
    rows = {k: _public(v, ("bytes", "via", "copy_ms", "query_ms", "error")) for k, v in hist.items()}
    return {"present": bool(rows), "profiles": rows, "shared_block_ms": _cache_ms(h, "history")}


@_probe(id="browser.history.stats", level="L2", family="browser", tier="T1", collect="extended", gate="browser.catalog")
def history_stats(h, facts):
    """Visits, URLs, first/last visit, active days, visits and pages per active day, per profile and total."""
    rows = _hist_rows(h)
    if not rows:
        return {"present": False}
    per = {}
    by_browser = collections.Counter()
    for r in rows:
        k = _pkey(r["browser"], r["profile_dir"] if r["profile_dir"] != "." else "")
        per[k] = _public(r, ("visits", "urls", "first_visit", "last_visit", "active_days", "visits_per_active_day",
                             "pages_per_day_urls", "typed_urls", "busiest_day"))
        per[k]["default_profile"] = r["is_default_profile"]
        eng = r.get("_engines")
        if eng is not None:
            terms = r.get("_terms") or {}
            per[k]["searches"] = sum(terms.values())
            per[k]["search_engines"] = dict(eng.most_common(3))
        by_browser[r["browser"]] += r["visits"]
    return {"present": True, "visits_total": sum(by_browser.values()), "by_browser": dict(by_browser.most_common()),
            "by_profile": per}


@_probe(id="browser.history.hour_weekday", level="L2", family="browser", tier="T1", collect="extended",
        gate="browser.catalog")
def history_hour_weekday(h, facts):
    """Local-time hour (24) and weekday (Sun=0) visit histograms over all profiles, plus months."""
    rows = _hist_rows(h) + list(_cached(h, "gecko_history", _gecko_history).values())
    rows = [r for r in rows if r.get("hours")]
    if not rows:
        return {"present": False}
    hours = [sum(r["hours"][i] for r in rows) for i in range(24)]
    wd = [sum((r.get("weekdays_sun0") or [0] * 7)[i] for r in rows) for i in range(7)]
    months = collections.Counter()
    for r in rows:
        months.update(r.get("months") or {})
    tot = sum(hours) or 1
    return {"present": True, "hours": hours, "weekdays_sun0": wd, "months": dict(sorted(months.items())),
            "night_share_00_05": round(sum(hours[0:5]) / tot, 3),
            "weekend_share": round((wd[0] + wd[6]) / (sum(wd) or 1), 3)}


def _category_totals(h):
    cats = collections.Counter()
    for r in _hist_rows(h):
        cats.update(r.get("_categories") or {})
    for r in _cached(h, "gecko_history", _gecko_history).values():
        cats.update(r.get("category_visits") or {})
    return cats


@_probe(id="browser.history.top_domains", level="L2", family="browser", tier="T1", collect="extended",
        gate="browser.catalog")
def history_top_domains(h, facts):
    """Visit totals by site category (named domains are in browser.history.top_domains_named, T2)."""
    cats = _category_totals(h)
    if not cats:
        return {"present": False}
    sensitive = sum(cats.pop(c, 0) for c in list(SENSITIVE_CATS))
    distinct = set()
    for r in _hist_rows(h):
        distinct.update((r.get("_domains") or {}).keys())
    out = dict(cats.most_common())
    if sensitive:
        out["sensitive"] = sensitive
    return {"present": True, "category_visits": out, "distinct_domains": len(distinct)}


@_probe(id="browser.history.top_domains_named", level="L2", family="browser", tier="T2", collect="extended",
        gate="browser.catalog")
def history_top_domains_named(h, facts):
    """Top domains (default profiles only, sensitive hosts never named), skill site buckets, comms services."""
    rows = _hist_rows(h)
    if not rows:
        return {"present": False}
    dom, buckets, svc = collections.Counter(), collections.Counter(), {}
    for r in rows:
        buckets.update(r.get("site_buckets") or {})
        if not r["is_default_profile"]:
            continue
        for d, c in (r.get("_domains") or {}).items():
            dom["(sensitive)" if _categorize(d) in SENSITIVE_CATS else d] += c
        for k, v in (r.get("_services") or {}).items():
            a = svc.setdefault(k, {"visits": 0, "active_days": 0, "visits_30d": 0, "last": None})
            a["visits"] += v["visits"]
            a["active_days"] += v["active_days"]
            a["visits_30d"] += v["visits_30d"]
            a["last"] = max(filter(None, (a["last"], v["last"])), default=None)
    return {"present": True,
            "top_domains": [{"domain": d, "visits": c, "category": _categorize(d)} for d, c in dom.most_common(15)],
            "skill_site_buckets_urls": dict(buckets.most_common()),
            "comms_services": dict(sorted(svc.items(), key=lambda x: -x[1]["visits"]))}


@_probe(id="browser.history.localhost", level="L2", family="browser", tier="T1", collect="extended",
        gate="browser.catalog")
def history_localhost(h, facts):
    """Visits to localhost/127.0.0.1 by port (dev servers, OAuth callbacks)."""
    ports = collections.Counter()
    for r in _hist_rows(h):
        ports.update(r.get("_ports") or {})
    if not ports:
        return {"present": False}
    return {"present": True, "visits": sum(ports.values()), "ports": {str(k): v for k, v in ports.most_common(10)}}


@_probe(id="browser.history.downloads", level="L2", family="browser", tier="T1", collect="extended",
        gate="browser.catalog")
def history_downloads(h, facts):
    """Browser download count by file extension."""
    n, exts = 0, collections.Counter()
    for r in _hist_rows(h):
        d = r.get("downloads")
        if d:
            n += d["count"]
            exts.update(d["by_ext"])
    if not n:
        return {"present": False}
    return {"present": True, "count": n, "by_ext": dict(exts.most_common(10))}


@_probe(id="browser.history.search_terms", level="L2", family="browser", tier="T2", collect="extended",
        gate="browser.catalog")
def history_search_terms(h, facts):
    """Search count, engines, mean words, themes by distinct term, <= 5 filtered examples (default profiles)."""
    terms, engines = collections.Counter(), collections.Counter()
    rows_all = 0
    for r in _hist_rows(h):
        if r.get("_terms") is None:
            continue
        rows_all += sum(r["_terms"].values())
        if r["is_default_profile"]:
            terms.update(r["_terms"])
            for e, c in (r.get("_engines") or {}).items():
                engines["(sensitive)" if _categorize(e) in SENSITIVE_CATS else e] += c
    if not rows_all:
        return {"present": False}
    themes = collections.Counter(t for term in terms for t in _themes(term))
    return {"present": True, "rows": sum(terms.values()), "rows_all_profiles": rows_all, "distinct_terms": len(terms),
            "engines": dict(engines.most_common(5)),
            "avg_words": round(sum(len(t.split()) for t in terms) / len(terms), 2) if terms else 0,
            "themes_distinct_terms": dict(themes.most_common()),
            "examples": [t for t, _ in terms.most_common(80) if _safe_term(t)][:5]}


@_probe(id="browser.history.github_repos", level="L2", family="browser", tier="T2", collect="extended",
        gate="browser.catalog")
def history_github_repos(h, facts):
    """Distinct github.com owner/repo paths visited; top 5 names."""
    gh = collections.Counter()
    for r in _hist_rows(h):
        gh.update(r.get("_github") or {})
    if not gh:
        return {"present": False}
    return {"present": True, "distinct": len(gh), "top": [k for k, _ in gh.most_common(5)]}


@_probe(id="browser.firefox.history", level="L2", family="browser", tier="T1", collect="extended",
        gate="browser.catalog")
def firefox_history(h, facts):
    """Gecko (Firefox/Zen/Floorp/LibreWolf/Waterfox) places.sqlite visit stats and category totals."""
    rows = _cached(h, "gecko_history", _gecko_history)
    if not rows:
        return {"present": False}
    return {"present": True, "profiles": {k: {kk: vv for kk, vv in v.items() if kk != "hours"}
                                          for k, v in rows.items()}}


# L3-level browser rollups. derive.py reads these ids from facts, so they are probes over the shared cache.

@_probe(id="l3.browser_category_share", level="L2", family="browser", tier="T1", collect="extended",
        gate="browser.catalog")
def l3_browser_category_share(h, facts):
    """Share of visits per category (0..1), sensitive categories folded into one bucket."""
    cats = _category_totals(h)
    tot = sum(cats.values())
    if not tot:
        return {"present": False}
    sens = sum(cats.pop(c, 0) for c in list(SENSITIVE_CATS))
    if sens:
        cats["sensitive"] = sens
    return {"present": True, "visits": tot, "shares": {k: round(v / tot, 3) for k, v in cats.most_common()},
            "password_manager_visits": cats.get("password_manager", 0)}


@_probe(id="l3.primary_browser", level="L2", family="browser", tier="T1", collect="extended", gate="browser.catalog")
def l3_primary_browser(h, facts):
    """Browser with the most history visits; falls back to the default browser."""
    by = collections.Counter()
    for r in _hist_rows(h):
        by[r["browser"]] += r["visits"]
    for r in _cached(h, "gecko_history", _gecko_history).values():
        by[r["browser"]] += r.get("visits", 0)
    d = facts.get("browser.default") or {}
    d = d.get("value") or {} if "value" in d else d
    dflt = d.get("browser") if isinstance(d, dict) else None
    if not by and not dflt:
        return {"present": False}
    primary = by.most_common(1)[0][0] if by and by.most_common(1)[0][1] > 0 else dflt
    return {"present": True, "primary": primary, "visits_by_browser": dict(by.most_common()),
            "matches_default": primary == dflt if dflt else None}


@_probe(id="l3.browser_rhythm", level="L2", family="browser", tier="T1", collect="extended", gate="browser.catalog")
def l3_browser_rhythm(h, facts):
    """Night share (00-05), weekend share and top-3 peak hours from history visits."""
    rows = [r for r in _hist_rows(h) if r.get("hours")]
    if not rows:
        return {"present": False}
    hours = [sum(r["hours"][i] for r in rows) for i in range(24)]
    wd = [sum(r["weekdays_sun0"][i] for r in rows) for i in range(7)]
    tot = sum(hours) or 1
    return {"present": True, "night_share_00_05": round(sum(hours[0:5]) / tot, 3),
            "weekend_share": round((wd[0] + wd[6]) / (sum(wd) or 1), 3),
            "peak_hours": sorted(range(24), key=lambda i: -hours[i])[:3]}


@_probe(id="l3.browser_persona_hints", level="L2", family="browser", tier="T1", collect="extended",
        gate="browser.catalog")
def l3_browser_persona_hints(h, facts):
    """Developer (dev+ai >= 20% of visits) and gamer (gaming >= 5%) hints from category shares."""
    cats = _category_totals(h)
    tot = sum(cats.values())
    if not tot:
        return {"present": False}
    dev = (cats["dev"] + cats["ai"]) / tot
    gam = cats["gaming"] / tot
    return {"present": True, "developer": dev >= 0.20, "dev_ai_share": round(dev, 3),
            "gamer": gam >= 0.05, "gaming_share": round(gam, 3),
            "note": "gaming share is weak evidence; gaming shows up in search themes more than domains"}


# ================================================================ family: comms_work

def _appx(h):
    out = {}
    for k in h.reg(r"HKCU\Software\Classes\Local Settings\Software\Microsoft\Windows\CurrentVersion\AppModel"
                   r"\Repository\Packages") or []:
        parts = k.split("_")
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    return out


def _prefetch(h):
    out = {}
    try:
        with os.scandir(r"C:\Windows\Prefetch") as it:
            for i, e in enumerate(it):
                if i > 5000:
                    break
                if e.name.endswith(".pf"):
                    exe = e.name.rsplit("-", 1)[0].upper()
                    try:
                        out[exe] = max(out.get(exe, 0), e.stat().st_mtime)
                    except OSError:
                        pass
    except OSError:
        return {}
    return out


def _run_keys(h):
    names = []
    for p in (r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run", r"HKLM\Software\Microsoft\Windows\CurrentVersion\Run"):
        names.extend((_reg_values(p) or {}).keys())
    return names


def _app_rows(h, table):
    L, R = _env("LOCALAPPDATA"), _env("APPDATA")
    appx = _cached(h, "appx", _appx)
    pf = _cached(h, "prefetch", _prefetch)
    run = _cached(h, "runkeys", _run_keys)
    out = {}
    for aid, (paths, pfs, pkgs, launched_paths) in table.items():
        hit = [p for p in paths if os.path.exists(p.format(L=L, R=R))]
        ax = {a: appx[a] for a in pkgs if a in appx}
        if not hit and not ax:
            continue
        last = max((pf[e] for e in pfs if e in pf), default=None)
        launched = None
        if launched_paths:
            launched = any(os.path.exists(p.format(L=L, R=R)) and bool(os.listdir(p.format(L=L, R=R)))
                           for p in launched_paths if _isdir(p.format(L=L, R=R)))
        row = {"installed": True, "appx_version": next(iter(ax.values()), None), "prefetch_last": _iso(last),
               "autostart": any(re.search(aid.split("_")[0], r, re.I) for r in run)}
        if launched is not None:
            row["launched"] = launched
        row["recent_use"] = bool(last and time.time() - last < 30 * 86400) or bool(launched)
        out[aid] = row
    return out


COMMS_APPS = {
    # id: (install paths, prefetch exes, appx package families, dirs that are non-empty only after first launch)
    "slack": ([r"{L}\slack", r"{R}\Slack"], ["SLACK.EXE"], ["91750D7E.Slack"], []),
    "teams_new": ([r"{L}\Packages\MSTeams_8wekyb3d8bbwe"], ["MS-TEAMS.EXE", "MSTEAMS.EXE"], ["MSTeams"],
                  [r"{L}\Packages\MSTeams_8wekyb3d8bbwe\LocalCache\Microsoft\MSTeams"]),
    "teams_classic": ([r"{L}\Microsoft\Teams", r"{R}\Microsoft\Teams"], ["TEAMS.EXE"], [], []),
    "zoom": ([r"{R}\Zoom", r"{L}\Zoom"], ["ZOOM.EXE"], ["ZoomVideoCommunications"], []),
    "telegram": ([r"{R}\Telegram Desktop"], ["TELEGRAM.EXE"], ["TelegramMessengerLLP.TelegramDesktop"], []),
    "whatsapp": ([r"{L}\Packages\5319275A.WhatsAppDesktop_cv1g1gvanyjgm", r"{L}\WhatsApp"], ["WHATSAPP.EXE"],
                 ["5319275A.WhatsAppDesktop"], []),
    "signal": ([r"{R}\Signal", r"{L}\Programs\signal-desktop"], ["SIGNAL.EXE"], [], []),
    "skype": ([r"{R}\Microsoft\Skype for Desktop"], ["SKYPE.EXE"], ["Microsoft.SkypeApp"], []),
    "webex": ([r"{L}\CiscoSpark", r"{L}\WebEx"], ["CISCOCOLLABHOST.EXE", "WEBEX.EXE"], [], []),
    "element": ([r"{R}\Element"], ["ELEMENT.EXE"], [], []),
    "thunderbird": ([r"{R}\Thunderbird"], ["THUNDERBIRD.EXE"], [], []),
    "outlook_classic": ([r"{L}\Microsoft\Outlook"], ["OUTLOOK.EXE"], [], []),
    "outlook_new": ([r"{L}\Packages\Microsoft.OutlookForWindows_8wekyb3d8bbwe"], ["OLK.EXE"],
                    ["Microsoft.OutlookForWindows"], [r"{L}\Microsoft\Olk"]),
    "mail_calendar_legacy": ([r"{L}\Packages\microsoft.windowscommunicationsapps_8wekyb3d8bbwe"],
                             ["HXOUTLOOK.EXE"], ["microsoft.windowscommunicationsapps"], []),
}

NOTES_APPS = {
    "notion": ([r"{R}\Notion", r"{L}\Programs\Notion"], ["NOTION.EXE"], [], []),
    "notion_calendar": ([r"{R}\Notion Calendar", r"{L}\Programs\cron-web"], ["NOTION CALENDAR.EXE"], [], []),
    "obsidian": ([r"{R}\obsidian\obsidian.json", r"{L}\Programs\obsidian"], ["OBSIDIAN.EXE"], [], []),
    "onenote": ([r"{L}\Microsoft\OneNote"], ["ONENOTE.EXE"], ["Microsoft.Office.OneNote"], []),
}


@_probe(id="comms.native_apps", level="L1", family="comms_work", tier="T0", collect="core")
def comms_native_apps(h, facts):
    """Native comms/mail apps installed (paths, appx) with prefetch last run and first-launch evidence."""
    rows = _app_rows(h, COMMS_APPS)
    if not rows:
        return {"present": False}
    return rows


@_probe(id="discord.present", level="L1", family="comms_work", tier="T0", collect="core")
def discord_present(h, facts):
    """Discord desktop installed; version from app-* dir; autostart."""
    roam, local = h.expand(r"%APPDATA%\discord"), h.expand(r"%LOCALAPPDATA%\Discord")
    if not (_isdir(roam) or _isdir(local)):
        return {"present": False}
    vers = sorted(d[4:] for d in h.list_dir(local, 100) if d.startswith("app-"))
    run = _cached(h, "runkeys", _run_keys)
    return {"present": True, "version": vers[-1] if vers else None, "autostart": any("discord" in r.lower() for r in run)}


@_probe(id="office.c2r", level="L1", family="comms_work", tier="T0", collect="core")
def office_c2r(h, facts):
    """Click-to-Run Office: product ids, version, platform, consumer vs business licence."""
    c = _reg_values(r"HKLM\SOFTWARE\Microsoft\Office\ClickToRun\Configuration") or {}
    pids = c.get("ProductReleaseIds")
    if not pids:
        return {"present": False}
    kind = ("consumer" if re.search(r"HomePrem|Personal|HomeStudent|Professional2", pids) else
            "business" if re.search(r"O365Business|ProPlus|Enterprise", pids) else "other")
    return {"present": True, "products": pids, "version": c.get("VersionToReport"), "platform": c.get("Platform"),
            "license_kind": kind}


@_probe(id="onedrive.accounts", level="L1", family="comms_work", tier="T0", collect="core")
def onedrive_accounts(h, facts):
    """OneDrive client and account slots; signed in only when UserEmail/cid/ConfiguredTenantId is set."""
    base = r"HKCU\SOFTWARE\Microsoft\OneDrive"
    exe = (h.exists(r"%LOCALAPPDATA%\Microsoft\OneDrive\OneDrive.exe") or
           h.exists(r"C:\Program Files\Microsoft OneDrive\OneDrive.exe"))
    keys = [k for k in (h.reg(base + r"\Accounts") or []) if k != "FileCoAuth"]
    if not exe and not keys:
        return {"present": False}
    signed = business = 0
    results = []
    for k in keys:
        v = _reg_values(base + r"\Accounts" + "\\" + k) or {}
        is_signed = bool(v.get("UserEmail") or v.get("cid") or v.get("ConfiguredTenantId"))
        signed += is_signed
        business += bool(is_signed and v.get("ConfiguredTenantId"))
        if isinstance(v.get("LastSignInResult"), int):
            results.append(hex(v["LastSignInResult"] & 0xFFFFFFFF))
    return {"present": True, "client_installed": bool(exe), "version": h.reg(base, "Version"),
            "account_slots": len(keys), "signed_in": signed, "business": business,
            "last_sign_in_results": sorted(set(results))}


@_probe(id="productivity.notes_apps", level="L1", family="comms_work", tier="T0", collect="core")
def notes_apps(h, facts):
    """Notes apps installed: Notion, Notion Calendar, Obsidian, OneNote."""
    rows = _app_rows(h, NOTES_APPS)
    return {"present": bool(rows), "apps": sorted(rows), "detail": rows}


@_probe(id="work.accounts", level="L1", family="comms_work", tier="T3", collect="core")
def work_accounts(h, facts):
    """Counts only: WAM token cache files, AAD broker accounts, MSA-linked logon marker. Files never opened."""
    tb = h.count_dir(r"%LOCALAPPDATA%\Microsoft\TokenBroker\Cache", 5000)
    aad = h.count_dir(r"%LOCALAPPDATA%\Packages\Microsoft.AAD.BrokerPlugin_cw5n1h2txyewy\AC\TokenBroker\Accounts", 5000)
    msa = h.reg(r"HKCU\SOFTWARE\Microsoft\IdentityCRL\UserExtendedProperties")
    aad_storage = h.reg(r"HKCU\SOFTWARE\Microsoft\Windows\CurrentVersion\AAD\Storage") or []
    return {"present": True, "tokenbroker_cache_files": max(tb, 0), "aad_broker_accounts": max(aad, 0),
            "aad_storage_keys": len(aad_storage), "windows_logon_msa_linked": bool(msa)}


@_probe(id="work.edition_org", level="L1", family="comms_work", tier="T0", collect="core")
def work_edition_org(h, facts):
    """Windows edition and whether RegisteredOrganization is set (the org name itself is not emitted)."""
    cv = r"HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion"
    ed = h.reg(cv, "EditionID")
    org = (h.reg(cv, "RegisteredOrganization") or "").strip()
    return {"present": bool(ed), "edition": ed, "registered_org_set": bool(org)}


@_probe(id="work.mdm", level="L1", family="comms_work", tier="T0", collect="core")
def work_mdm(h, facts):
    """Real MDM enrollments only (UPN, ProviderID MS DM Server, or DiscoveryServiceFullURL)."""
    base = r"HKLM\SOFTWARE\Microsoft\Enrollments"
    keys = h.reg(base) or []
    real = []
    for g in keys[:500]:
        v = _reg_values(base + "\\" + g) or {}
        if v.get("UPN") or v.get("ProviderID") in ("MS DM Server", "WMI_Bridge_Server") or v.get("DiscoveryServiceFullURL"):
            real.append({"type": v.get("EnrollmentType"), "provider": v.get("ProviderID")})
    omadm = h.reg(r"HKLM\SOFTWARE\Microsoft\Provisioning\OMADM\Accounts") or []
    return {"present": True, "real_enrollments": len(real), "enrollments": real[:5], "enrollment_keys_total": len(keys),
            "omadm_accounts": len(omadm),
            "intune_ime": h.exists(r"C:\Program Files (x86)\Microsoft Intune Management Extension")}


@_probe(id="work.mdm_policy_areas", level="L1", family="comms_work", tier="T0", collect="core")
def work_mdm_policy_areas(h, facts):
    """PolicyManager device areas and providers (corroborates MDM; built-in areas exist on every box)."""
    areas = h.reg(r"HKLM\SOFTWARE\Microsoft\PolicyManager\current\device") or []
    prov = h.reg(r"HKLM\SOFTWARE\Microsoft\PolicyManager\providers") or []
    return {"present": True, "device_areas": len(areas), "providers": len(prov), "areas": sorted(areas)[:40]}


@_probe(id="work.policies", level="L1", family="comms_work", tier="T0", collect="core")
def work_policies(h, facts):
    """Group-policy keys for browsers/Office/OneDrive/Teams/Slack/Zoom (HKLM+HKCU), value counts only."""
    out, apps = {}, {}
    for hive in ("HKLM", "HKCU"):
        out[hive + "_vendors"] = sorted(h.reg(hive + r"\SOFTWARE\Policies") or [])
        for b in (r"Google\Chrome", r"BraveSoftware\Brave", r"Microsoft\Edge", r"Mozilla\Firefox", r"Microsoft\Office",
                  r"Microsoft\OneDrive", r"Microsoft\Teams", "Slack", "Zoom"):
            p = hive + r"\SOFTWARE\Policies" + "\\" + b
            v, sk = _reg_values(p), h.reg(p)
            if v is not None or sk is not None:
                apps[f"{hive}:{b}"] = {"values": len(v or {}), "subkeys": len(sk or [])}
        for area in ("TenantRestrictions", "WorkplaceJoin"):
            v = _reg_values(hive + r"\SOFTWARE\Policies\Microsoft\Windows" + "\\" + area)
            if v:
                out.setdefault("windows_area_value_names", {})[f"{hive}:{area}"] = sorted(v)[:10]
    configured = {k: v for k, v in apps.items() if v["values"] or v["subkeys"]}
    return {"present": True, "app_policies": configured, "app_policy_count": len(configured), **out}


AGENT_RX = re.compile(r"^(IntuneManagementExtension|CSFalcon\w*|ZSATunnel|ZSAService|PanGPS|vpnagent|csc_\w*|Netskope\w*|"
                      r"stAgent\w*|SentinelAgent|Tanium\w*|CylanceSvc|CcmExec|jamf\w*|Kandji\w*|FortiClient\w*|"
                      r"NinjaRMM\w*|ManageEngine\w*|LTService|DattoRMM|CiscoAMP|OktaVerify\w*|DuoAuth\w*|Sense|"
                      r"GlobalProtect\w*|CarbonBlack|CbDefense|elastic-agent|WinCollect|Qualys\w*|Rapid7\w*|"
                      r"TaniumClient|TrendMicro\w*|SophosMCS\w*|Sophos\w*)$", re.I)


@_probe(id="work.security_agents", level="L1", family="comms_work", tier="T0", collect="core")
def work_security_agents(h, facts):
    """EDR/ZTNA/RMM agents from the Services registry (not sc query). Sense (Defender) is reported separately."""
    svcs = h.reg(r"HKLM\SYSTEM\CurrentControlSet\Services") or []
    hits = sorted(s for s in svcs if AGENT_RX.match(s))
    onboard = h.reg(r"HKLM\SOFTWARE\Microsoft\Windows Advanced Threat Protection\Status", "OnboardingState")
    return {"present": True, "agents": [s for s in hits if s.lower() != "sense"], "sense_present": "Sense" in hits,
            "defender_atp_onboarded": bool(onboard), "services_scanned": len(svcs)}


@_probe(id="work.join_state", level="L2", family="comms_work", tier="T0", collect="extended", gate="work.edition_org",
        timeout_ms=5000)
def work_join_state(h, facts):
    """Entra/AD/workplace join from dsregcmd /status, with the CloudDomainJoin registry as a cheap cross-check."""
    join_info = h.reg(r"HKLM\SYSTEM\CurrentControlSet\Control\CloudDomainJoin\JoinInfo") or []
    domain = h.reg(r"HKLM\SYSTEM\CurrentControlSet\Services\Tcpip\Parameters", "Domain") or ""
    txt = h.run(["dsregcmd", "/status"], timeout_ms=5000, text=False) or ""
    got = {}
    for k in ("AzureAdJoined", "EnterpriseJoined", "DomainJoined", "WorkplaceJoined", "AzureAdPrt"):
        m = re.search(rf"^\s*{k}\s*:\s*(\w+)", txt, re.M)
        if m:
            got[k] = m.group(1).upper() == "YES"
    joined = got.get("AzureAdJoined") or got.get("DomainJoined") or got.get("EnterpriseJoined")
    return {"present": True, "source": "dsregcmd" if got else "registry",
            "azure_ad_joined": got.get("AzureAdJoined", bool(join_info)), "domain_joined": got.get("DomainJoined"),
            "enterprise_joined": got.get("EnterpriseJoined"), "workplace_joined": got.get("WorkplaceJoined"),
            "azure_ad_prt": got.get("AzureAdPrt"), "cloud_join_info_keys": len(join_info),
            "tcpip_domain_set": bool(domain), "state": "joined" if joined else "workgroup"}


@_probe(id="comms.teams_launched", level="L2", family="comms_work", tier="T0", collect="core",
        gate="comms.native_apps")
def comms_teams_launched(h, facts):
    """New Teams actually started: LocalCache\\Microsoft\\MSTeams exists; tfw = work, tfl = personal profiles."""
    b = h.expand(r"%LOCALAPPDATA%\Packages\MSTeams_8wekyb3d8bbwe\LocalCache\Microsoft\MSTeams")
    if not _isdir(b):
        return {"present": False}
    prof = h.list_dir(os.path.join(b, "EBWebView"), 100)
    work = sum(1 for p in prof if "tfw" in p.lower())
    return {"present": True, "work_profile": work > 0, "work_profiles": work,
            "personal_profiles": sum(1 for p in prof if "tfl" in p.lower()),
            "log_files": max(h.count_dir(os.path.join(b, "Logs"), 5000), 0)}


def _userassist(h, pattern):
    rx = re.compile(pattern, re.I)
    base = HKCU_EXPLORER + r"\UserAssist"
    rot = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
                        "NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm")
    focus, last = 0, None
    for g in h.reg(base) or []:
        for n, v in (_reg_values(base + "\\" + g + r"\Count") or {}).items():
            if isinstance(v, bytes) and len(v) >= 68 and rx.search(n.translate(rot)):
                focus += int.from_bytes(v[12:16], "little") // 1000
                t = _ft_iso(int.from_bytes(v[60:68], "little"))
                last = max(filter(None, (last, t)), default=None)
    return focus, last


@_probe(id="discord.usage", level="L2", family="comms_work", tier="T1", collect="extended", gate="discord.present",
        timeout_ms=3000)
def discord_usage(h, facts):
    """Signed-in and voice days from renderer_js.log line tags; absent when the app never got past login."""
    log = h.expand(r"%APPDATA%\discord\logs\renderer_js.log")
    pf = _cached(h, "prefetch", _prefetch).get("DISCORD.EXE")
    focus_s, ua_last = _userassist(h, r"discord")
    days, gw, rtc = set(), set(), set()
    last_gw = last_auth = None
    size = 0
    if os.path.isfile(log):
        size = os.path.getsize(log)
        rx = re.compile(r"^\[(\d{4}-\d\d-\d\d) ([\d:.]+)\] \[\w+\]\s+\[?([A-Za-z_]+)")
        with open(log, encoding="utf-8", errors="replace") as fh:
            if size > 16_000_000:
                fh.seek(size - 16_000_000)
            for line in fh:
                m = rx.match(line)
                if not m:
                    continue
                d, tm, tag = m.groups()
                days.add(d)
                if tag == "GatewaySocket":
                    gw.add(d)
                    last_gw = d + "T" + tm[:8]
                elif tag.startswith("RTC"):
                    rtc.add(d)
                elif tag == "useAuthWebsocket":
                    last_auth = d + "T" + tm[:8]
    used = bool(gw or rtc)
    return {"present": used, "log_days": len(days), "first_day": min(days) if days else None,
            "last_day": max(days) if days else None, "gateway_days": len(gw), "rtc_days": len(rtc),
            "logged_in_last_session": bool(last_gw and (not last_auth or last_gw > last_auth)),
            "on_login_screen": bool(last_auth and (not last_gw or last_auth > last_gw)),
            "prefetch_last": _iso(pf), "userassist_focus_s": focus_s, "userassist_last": ua_last,
            "log_bytes": size}


@_probe(id="office.usage", level="L2", family="comms_work", tier="T1", collect="core", gate="office.c2r")
def office_usage(h, facts):
    """Office identity count, File/Place MRU item counts per app, Office exe prefetch dates (no names or paths)."""
    base = r"HKCU\SOFTWARE\Microsoft\Office"
    ids = h.reg(base + r"\16.0\Common\Identity\Identities") or []
    mru = {}
    for app in ("Word", "Excel", "PowerPoint", "OneNote", "Access", "Visio", "Publisher"):
        n = 0
        for sub in ("File MRU", "Place MRU"):
            n += sum(1 for k in (_reg_values(rf"{base}\16.0\{app}\{sub}") or {}) if k.lower().startswith("item"))
            for u in h.reg(rf"{base}\16.0\{app}\User MRU") or []:
                n += sum(1 for k in (_reg_values(rf"{base}\16.0\{app}\User MRU\{u}\{sub}") or {})
                         if k.lower().startswith("item"))
        if n:
            mru[app] = n
    pf = _cached(h, "prefetch", _prefetch)
    runs = {e: _iso(pf[e]) for e in ("WINWORD.EXE", "EXCEL.EXE", "POWERPNT.EXE", "OUTLOOK.EXE", "ONENOTE.EXE") if e in pf}
    return {"present": True, "identities": len(ids), "mru_items": sum(mru.values()), "mru_by_app": mru,
            "prefetch_last": runs, "used": bool(mru or runs)}


@_probe(id="obsidian.vaults", level="L2", family="comms_work", tier="T1", collect="extended",
        gate="productivity.notes_apps")
def obsidian_vaults(h, facts):
    """Obsidian vault count and note counts (vault names and paths are not emitted)."""
    j = _load_json(h.expand(r"%APPDATA%\obsidian\obsidian.json"))
    vs = (j or {}).get("vaults") or {} if isinstance(j, dict) else {}
    if not vs:
        return {"present": False}
    notes, files, newest = 0, 0, 0

    def on_file(e, s, depth):
        nonlocal notes, newest
        if e.name.lower().endswith(".md"):
            notes += 1
        newest = max(newest, s.st_mtime)
    for v in list(vs.values())[:10]:
        p = v.get("path") if isinstance(v, dict) else None
        if p and _isdir(p):
            st = _walk(p, max_depth=6, max_entries=30000, budget_s=1.5, on_file=on_file,
                       skip_dir=lambda n, _p: n in (".obsidian", ".git", ".trash", "node_modules"))
            files += st["files"]
    return {"present": True, "vaults": len(vs), "open": sum(1 for v in vs.values() if isinstance(v, dict) and v.get("open")),
            "notes_md": notes, "files": files, "newest_note": _iso(newest)}


def _history_ids(h):
    tot = collections.Counter()
    for r in _hist_rows(h):
        tot.update(r.get("_ids") or {})
    return tot


@_probe(id="browser.history.workspaces", level="L2", family="comms_work", tier="T2", collect="extended",
        gate="browser.catalog")
def history_workspaces(h, facts):
    """Distinct Discord guild / Slack workspace / Gmail slot / Linear workspace ids seen in history (counts only)."""
    rows = _hist_rows(h)
    if not rows:
        return {"present": False}
    ids = _history_ids(h)
    return {"discord": ids["discord_guilds"], "slack": ids["slack_workspaces"], "gmail": ids["gmail_slots"],
            "linear": ids["linear_workspaces"]}


@_probe(id="browser.history.work_hosts", level="L2", family="comms_work", tier="T2", collect="extended",
        gate="browser.catalog")
def history_work_hosts(h, facts):
    """Top hostnames for Slack, nousresearch.com, Discord, Notion and Linear (default profiles)."""
    hosts = {}
    for r in _hist_rows(h):
        if not r["is_default_profile"]:
            continue
        for k, c in (r.get("_work_hosts") or {}).items():
            hosts.setdefault(k, collections.Counter()).update(c)
    if not hosts:
        return {"present": False}
    return {"present": True, "hosts": {k: [x for x, _ in v.most_common(5)] for k, v in hosts.items()}}


# ================================================================ family: files

KF_GUIDS = {
    "Desktop": "{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}", "Documents": "{FDD39AD0-238F-46AF-ADB4-6C85480369C7}",
    "Downloads": "{374DE290-123F-4565-9164-39C4925E467B}", "Pictures": "{33E28130-4E1E-4676-835A-98395C3BC3BB}",
    "Videos": "{18989B1D-99B5-455B-841C-AB7C74E4DDFC}", "Music": "{4BD8D571-6D19-48D3-BE97-422220080E43}",
    "Screenshots": "{B7BEDE81-DF94-4682-A7D8-57A52620B86F}", "Captures": "{EDC0FE71-98D8-4F4A-B920-C8DC133CB165}",
    "CameraRoll": "{AB5FB87B-7CE2-4F83-915D-550846C9537B}",
}


def _known_folders(h):
    import ctypes

    class GUID(ctypes.Structure):
        _fields_ = [("Data1", ctypes.c_ulong), ("Data2", ctypes.c_ushort), ("Data3", ctypes.c_ushort),
                    ("Data4", ctypes.c_ubyte * 8)]
    out = {}
    for name, g in KF_GUIDS.items():
        guid = GUID()
        ctypes.oledll.ole32.CLSIDFromString(ctypes.c_wchar_p(g), ctypes.byref(guid))
        p = ctypes.c_wchar_p()
        try:
            ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0x4000, None, ctypes.byref(p))  # DONT_VERIFY
            out[name] = p.value
        except OSError:
            out[name] = None
        finally:
            if p:
                ctypes.windll.ole32.CoTaskMemFree(p)
    return out


def _kf(h, name):
    try:
        p = _cached(h, "kf", _known_folders).get(name)
    except RuntimeError:
        p = None
    return p or os.path.join(_home(), name)


@_probe(id="files.known_folders", level="L1", family="files", tier="T0", collect="core")
def files_known_folders(h, facts):
    """Known-folder paths (SHGetKnownFolderPath, DONT_VERIFY): exists, under profile, under OneDrive."""
    kf = _cached(h, "kf", _known_folders)
    home = _home().lower()
    rows = {n: {"path": _redact(p), "exists": bool(p and _isdir(p)),
                "under_profile": bool(p and p.lower().startswith(home)),
                "under_onedrive": bool(p and "onedrive" in p.lower())} for n, p in kf.items()}
    return {"present": any(r["exists"] for r in rows.values()), "folders": rows}


@_probe(id="sync.other", level="L1", family="files", tier="T0", collect="core")
def sync_other(h, facts):
    """Other sync clients: Dropbox, Google Drive, iCloud, Box, MEGA, Proton Drive, pCloud, Syncthing."""
    cands = {"dropbox": r"%LOCALAPPDATA%\Dropbox\info.json", "google_drive": r"%LOCALAPPDATA%\Google\DriveFS",
             "icloud": r"%LOCALAPPDATA%\Apple Inc\iCloud", "box": r"%LOCALAPPDATA%\Box",
             "mega": r"%LOCALAPPDATA%\MEGAsync", "proton_drive": r"%LOCALAPPDATA%\Proton\Proton Drive",
             "pcloud": r"%APPDATA%\pCloud", "syncthing": r"%LOCALAPPDATA%\Syncthing"}
    found = [k for k, p in cands.items() if h.exists(p)]
    roots = h.reg(r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\SyncRootManager") or []
    providers = sorted({r.split("!")[0] for r in roots if not r.lower().startswith("onedrive")})
    return {"present": bool(found), "clients": found, "sync_root_providers_non_onedrive": providers[:10]}


@_probe(id="jumplists.counts", level="L1", family="files", tier="T0", collect="core")
def jumplists_counts(h, facts):
    """Jump-list file counts and sizes (Automatic/CustomDestinations); entries never parsed."""
    base = h.expand(r"%APPDATA%\Microsoft\Windows\Recent")
    out = {}
    for sub in ("AutomaticDestinations", "CustomDestinations"):
        n = b = 0
        try:
            with os.scandir(os.path.join(base, sub)) as it:
                for i, e in enumerate(it):
                    if i >= 5000:
                        break
                    if e.is_file():
                        n += 1
                        b += e.stat().st_size
        except OSError:
            continue
        out[sub] = {"files": n, "bytes": b}
    return {"present": bool(out), **out}


@_probe(id="onedrive.kfm", level="L1", family="files", tier="T0", collect="core", gate="onedrive.accounts")
def onedrive_kfm(h, facts):
    """Known-folder move: Desktop/Documents/Pictures redirected into OneDrive (User Shell Folders)."""
    usf = _reg_values(HKCU_EXPLORER + r"\User Shell Folders") or {}
    red = [k for k in ("Desktop", "Personal", "My Pictures", "My Video", "My Music") if "onedrive" in str(usf.get(k, "")).lower()]
    return {"present": True, "redirected": bool(red), "folders": red}


@_probe(id="onedrive.root_size", level="L2", family="files", tier="T0", collect="core", gate="onedrive.accounts")
def onedrive_root_size(h, facts):
    """Bounded size of %USERPROFILE%\\OneDrive* without hydrating placeholders."""
    home = _home()
    roots = {}
    for name in h.list_dir(home, 500):
        if name.lower().startswith("onedrive") and _isdir(os.path.join(home, name)):
            st = _walk(os.path.join(home, name), max_depth=6, max_entries=50000, budget_s=2.0)
            roots[name] = {k: st[k] for k in ("files", "bytes", "cloud_placeholders", "truncated")}
    if not roots:
        return {"present": False}
    return {"present": True, "roots": roots, "files": sum(r["files"] for r in roots.values()),
            "bytes": sum(r["bytes"] for r in roots.values())}


def _recent(h):
    rec = h.expand(r"%APPDATA%\Microsoft\Windows\Recent")
    now = time.time()
    ext, ages, mts = collections.Counter(), collections.Counter(), []
    operator = 0
    try:
        it = os.scandir(rec)
    except OSError:
        return None
    with it:
        for i, e in enumerate(it):
            if i >= 20000:
                break
            if not e.name.lower().endswith(".lnk") or not e.is_file():
                continue
            if OPERATOR_DEEP_RX.search(e.name) or OPERATOR_RX.match(e.name[:-4]):
                operator += 1
                continue
            try:
                s = e.stat()
            except OSError:
                continue
            mts.append(s.st_mtime)
            ages[_age_bucket(now - s.st_mtime)] += 1
            x = os.path.splitext(e.name[:-4])[1].lower()
            ext[x if re.fullmatch(r"\.[a-z0-9]{1,5}", x or "") else "(folder_or_none)"] += 1
    return {"count": len(mts), "operator_filtered": operator, "ext": ext, "ages": ages, "mts": mts}


@_probe(id="files.recent_lnk", level="L2", family="files", tier="T1", collect="core", gate="files.known_folders")
def files_recent_lnk(h, facts):
    """Recent .lnk count, date range, folder vs file split, target extension counts (no names)."""
    r = _recent(h)
    if not r or not r["count"]:
        return {"present": False}
    now = time.time()
    mts = r["mts"]
    folders = r["ext"].get("(folder_or_none)", 0)
    return {"present": True, "count": r["count"], "operator_filtered": r["operator_filtered"],
            "oldest": _iso(min(mts)), "newest": _iso(max(mts)), "folder_or_none": folders,
            "files": r["count"] - folders, "target_ext_top": dict(r["ext"].most_common(10)),
            "age_mtime": {k: r["ages"].get(k, 0) for k in AGE_KEYS},
            "active_days_30d": len({time.strftime("%Y-%m-%d", time.localtime(t)) for t in mts if now - t < 30 * 86400})}


SHOT_NAME = re.compile(r"^(screenshot|screen shot|capture|snip|screen ?recording|scr_|shot_)", re.I)
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".jxr", ".heic", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv", ".m4v"}
AUDIO_EXT = {".mp3", ".flac", ".wav", ".m4a", ".aac", ".ogg", ".opus"}
DOC_EXT = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".txt", ".md", ".csv", ".odt", ".rtf"}
CODE_EXT = {".py", ".js", ".ts", ".json", ".toml", ".yaml", ".yml", ".sh", ".ps1", ".bat", ".rs", ".go", ".c", ".cpp",
            ".h", ".java", ".ipynb", ".log", ".rst", ".html", ".css"}
INSTALLER_EXT = {".exe", ".msi", ".msix", ".appx", ".msixbundle", ".appxbundle"}
ARCHIVE_EXT = {".zip", ".7z", ".rar", ".tar", ".gz", ".iso", ".xz", ".bz2", ".zst"}


def _kind(ext):
    for k, s in (("image", IMG_EXT), ("video", VIDEO_EXT), ("audio", AUDIO_EXT), ("document", DOC_EXT),
                 ("code", CODE_EXT), ("installer", INSTALLER_EXT), ("archive", ARCHIVE_EXT)):
        if ext in s:
            return k
    return "other"


def _composition(h):
    now = time.time()
    deadline = time.perf_counter() + 8.0
    shots = []
    seen = set()
    res = {}
    for name in ("Desktop", "Documents", "Downloads", "Pictures", "Videos", "Music"):
        p = _kf(h, name)
        ext_n, ext_b, kind_b = collections.Counter(), collections.Counter(), collections.Counter()
        ages, years = collections.Counter(), collections.Counter()
        dup = [0]

        def on_file(e, s, depth, label=name):
            ext = os.path.splitext(e.name)[1].lower() or "(none)"
            if len(ext) > 12:
                ext = "(long)"
            ext_n[ext] += 1
            ext_b[ext] += s.st_size
            kind_b[_kind(ext)] += s.st_size
            ages[_age_bucket(now - s.st_mtime)] += 1
            years[time.localtime(s.st_mtime).tm_year] += 1
            if re.search(r"\(\d+\)(\.[^.]+)?$", e.name):
                dup[0] += 1
            if ext in IMG_EXT or ext in (".mp4", ".mkv"):
                low = e.path.lower()
                if (SHOT_NAME.match(e.name) or any(s_ in low for s_ in ("\\screenshots\\", "\\captures\\", "\\sharex\\",
                                                                        "\\ansel\\"))) and low not in seen:
                    seen.add(low)
                    shots.append((s.st_mtime, label, ext in IMG_EXT))
        budget = max(0.2, min(3.0, deadline - time.perf_counter()))
        st = _walk(p, max_depth=6, max_entries=100000, budget_s=budget, on_file=on_file,
                   skip_dir=lambda n, path: bool(OPERATOR_DEEP_RX.search(n)))
        top = {"files": 0, "dirs": 0, "shortcuts": 0}
        for e in h.list_dir(p, 5000):
            full = os.path.join(p, e)
            if e.lower() == "desktop.ini":
                continue
            if _isdir(full):
                top["dirs"] += 1
            else:
                top["files"] += 1
                top["shortcuts"] += e.lower().endswith((".lnk", ".url"))
        res[name] = {"files": st["files"], "dirs": st["dirs"], "bytes": st["bytes"], "truncated": st["truncated"],
                     "depth": 6, "top_level": top, "ext_top_by_count": dict(ext_n.most_common(10)),
                     "ext_top_by_bytes": dict(ext_b.most_common(6)), "kind_bytes": dict(kind_b),
                     "age_mtime": {k: ages.get(k, 0) for k in AGE_KEYS},
                     "oldest_year": min(years) if years else None, "newest_year": max(years) if years else None,
                     "dup_suffix_files": dup[0], "ms": st.get("ms")}
    return {"folders": res, "shots": shots}


@_probe(id="files.composition", level="L2", family="files", tier="T1", collect="extended", gate="files.known_folders",
        timeout_ms=9000)
def files_composition(h, facts):
    """Per known folder: files, bytes, extension/kind/age histograms, depth <= 6 (no file names)."""
    c = _cached(h, "composition", _composition)
    folders = c["folders"]
    kind = collections.Counter()
    for f in folders.values():
        kind.update(f["kind_bytes"])
    tot = sum(kind.values()) or 1
    dl = folders.get("Downloads", {})
    old = sum(dl.get("age_mtime", {}).get(k, 0) for k in ("lt_1y", "lt_2y", "ge_2y"))
    return {"present": True, "folders": folders, "total_bytes": sum(f["bytes"] for f in folders.values()),
            "byte_share": {k: round(v / tot, 3) for k, v in kind.most_common()},
            "downloads_older_90d_share": round(old / dl["files"], 3) if dl.get("files") else None,
            "shared_block_ms": _cache_ms(h, "composition")}


@_probe(id="files.screenshots", level="L2", family="files", tier="T1", collect="extended", gate="files.known_folders",
        timeout_ms=9000)
def files_screenshots(h, facts):
    """Screenshot/capture count, active days, cadence and local hour/weekday histograms (from the composition walk)."""
    c = _cached(h, "composition", _composition)
    imgs = sorted(t for t, _l, is_img in c["shots"] if is_img)
    vids = sum(1 for _t, _l, is_img in c["shots"] if not is_img)
    shots_kf = _kf(h, "Screenshots")
    if not imgs:
        return {"present": False}
    now = time.time()
    days = {time.strftime("%Y-%m-%d", time.localtime(t)) for t in imgs}
    gaps = [(b - a) / 3600 for a, b in zip(imgs, imgs[1:])]
    return {"present": True, "count": len(imgs), "video_captures": vids,
            "by_folder": dict(collections.Counter(l for _t, l, i in c["shots"] if i)),
            "screenshots_kf_exists": _isdir(shots_kf), "first": _iso(imgs[0]), "last": _iso(imgs[-1]),
            "last_30d": sum(1 for t in imgs if now - t < 30 * 86400), "active_days": len(days),
            "per_active_day": round(len(imgs) / len(days), 2),
            "median_gap_h": round(statistics.median(gaps), 2) if gaps else None,
            "hours": [sum(1 for t in imgs if time.localtime(t).tm_hour == hr) for hr in range(24)],
            "weekdays_mon0": [sum(1 for t in imgs if time.localtime(t).tm_wday == d) for d in range(7)]}


def _user_sid(h):
    home = _home().lower()
    base = r"HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\ProfileList"
    for sid in h.reg(base) or []:
        if sid.startswith("S-1-5-21-") and (h.reg(base + "\\" + sid, "ProfileImagePath") or "").lower() == home:
            return sid
    return None


@_probe(id="files.clutter", level="L2", family="files", tier="T1", collect="extended", gate="files.known_folders",
        timeout_ms=6000)
def files_clutter(h, facts):
    """Recycle Bin, TEMP and thumbnail-cache sizes; Desktop item count; Windows Search index size."""
    out = {"present": True}
    sid = _user_sid(h)
    rb = rf"C:\$Recycle.Bin\{sid}" if sid else None
    if rb and _isdir(rb):
        items = [0]

        def cb(e, s, depth):
            if depth == 0 and e.name.startswith("$R"):
                items[0] += 1
        st = _walk(rb, max_depth=6, max_entries=100000, budget_s=2.0, on_file=cb)
        out["recycle_bin"] = {"items": items[0], "files": st["files"], "bytes": st["bytes"], "truncated": st["truncated"]}
    tmp = _env("TEMP")
    if tmp and _isdir(tmp):
        st = _walk(tmp, max_depth=5, max_entries=100000, budget_s=2.0,
                   skip_dir=lambda n, p: n == "userscan" or bool(OPERATOR_DEEP_RX.search(n)))
        out["temp_files"], out["temp_bytes"], out["temp_truncated"] = st["files"], st["bytes"], st["truncated"]
    ex = h.expand(r"%LOCALAPPDATA%\Microsoft\Windows\Explorer")
    tb = sum((h.meta(os.path.join(ex, n)).get("bytes") or 0) for n in h.list_dir(ex, 500)
             if n.lower().startswith(("thumbcache_", "iconcache_")))
    out["thumbcache_bytes"] = tb
    idx = h.meta(r"C:\ProgramData\Microsoft\Search\Data\Applications\Windows\Windows.db")
    out["search_index_bytes"] = idx.get("bytes")
    desk = _kf(h, "Desktop")
    names = [n for n in h.list_dir(desk, 5000) if n.lower() != "desktop.ini"]
    out["desktop_items"] = len(names)
    out["desktop_shortcuts"] = sum(1 for n in names if n.lower().endswith((".lnk", ".url")))
    return out


def _mru_counts(path):
    try:
        subs = []
        with _reg_key(path) as k:
            wr = _winreg()
            i = 0
            while i < 2000:
                try:
                    subs.append(wr.EnumKey(k, i))
                except OSError:
                    break
                i += 1
    except OSError:
        return None
    per = {}
    for sk in subs:
        v = _reg_values(path + "\\" + sk) or {}
        per[sk.lower()] = sum(1 for n in v if n.isdigit())
    return per


EXT_OK = re.compile(r"^\.?[a-z0-9]{1,6}$|^folder$|^\*$")


def _bucket(per):
    clean, odd = collections.Counter(), 0
    for e, c in per.items():
        if EXT_OK.match(e):
            clean[e] += c
        else:
            odd += 1
            clean["(url_or_odd)"] += c
    return clean, odd


def _utf16_first(b):
    try:
        return b.decode("utf-16le", errors="ignore").split("\x00", 1)[0]
    except Exception:
        return None


@_probe(id="files.opensave_mru", level="L2", family="files", tier="T1", collect="extended", gate="files.known_folders")
def files_opensave_mru(h, facts):
    """File-dialog history: per-extension item counts and the app exe names that used the dialog (no paths)."""
    cd = HKCU_EXPLORER + r"\ComDlg32"
    per = _mru_counts(cd + r"\OpenSavePidlMRU")
    apps = collections.Counter()
    for sub in ("LastVisitedPidlMRU", "CIDSizeMRU"):
        for n, v in (_reg_values(cd + "\\" + sub) or {}).items():
            if n.isdigit() and isinstance(v, bytes):
                a = _utf16_first(v)
                if a and re.fullmatch(r"[\w .\-]{1,64}\.exe", a, re.I):
                    apps[a.lower()] += 1
    if per is None and not apps:
        return {"present": False}
    clean, odd = _bucket({k: v for k, v in (per or {}).items() if k != "*"})
    return {"present": True, "all_items": (per or {}).get("*"), "ext_top": dict(clean.most_common(12)),
            "odd_keys_redacted": odd, "apps": dict(apps.most_common(12)),
            "last_write": _reg_lastwrite(cd + r"\OpenSavePidlMRU")}


@_probe(id="files.recentdocs", level="L2", family="files", tier="T1", collect="extended", gate="files.known_folders")
def files_recentdocs(h, facts):
    """Explorer RecentDocs: total items and per-extension counts; URL/search-built subkeys bucketed."""
    path = HKCU_EXPLORER + r"\RecentDocs"
    top = _reg_values(path)
    if top is None:
        return {"present": False}
    per = _mru_counts(path) or {}
    clean, odd = _bucket(per)
    return {"present": True, "total_items": sum(1 for n in top if n.isdigit()), "ext_subkeys": len(per),
            "odd_keys_redacted": odd, "ext_top": dict(clean.most_common(12)), "last_write": _reg_lastwrite(path)}


def _reg_domain(host):
    if not host or re.fullmatch(r"[\d.]+", host):
        return host
    parts = host.split(".")
    if len(parts) >= 3 and parts[-2] in ("co", "com", "org", "net", "ac", "gov") and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


@_probe(id="files.download_sources", level="L2", family="files", tier="T1", collect="deep", gate="files.known_folders",
        timeout_ms=6000)
def files_download_sources(h, facts):
    """Zone.Identifier ADS in Downloads (depth <= 2): zone counts and source-site category counts (no hosts)."""
    root = _kf(h, "Downloads")
    if not _isdir(root):
        return {"present": False}
    zone, cats, ext_cat = collections.Counter(), collections.Counter(), collections.Counter()
    n = with_ads = no_host = 0
    deadline = time.perf_counter() + 5.0
    stack, trunc = [(root, 0)], None
    while stack:
        d, depth = stack.pop()
        try:
            entries = list(os.scandir(d))
        except OSError:
            continue
        for e in entries:
            if e.is_dir(follow_symlinks=False):
                if depth + 1 < 2 and not OPERATOR_DEEP_RX.search(e.name):
                    stack.append((e.path, depth + 1))
                continue
            n += 1
            if n > 5000 or time.perf_counter() > deadline:
                trunc = "cap"
                stack.clear()
                break
            try:
                with open(e.path + ":Zone.Identifier", "r", encoding="utf-8", errors="replace") as f:
                    txt = f.read(4096)
            except OSError:
                continue
            with_ads += 1
            kv = dict(line.split("=", 1) for line in txt.splitlines() if "=" in line)
            zone[kv.get("ZoneId", "?").strip()] += 1
            url = (kv.get("HostUrl") or "").strip()
            if not url or url.startswith("about:"):
                no_host += 1
                continue
            host, _ = _host_of(url)
            c = _categorize(_reg_domain(host)) if host else "other"
            c = "sensitive" if c in SENSITIVE_CATS else c
            cats[c] += 1
            ext_cat[f"{c}:{os.path.splitext(e.name)[1].lower() or '(none)'}"] += 1
    return {"present": True, "files_scanned": n, "depth": 2, "truncated": trunc, "with_zone_identifier": with_ads,
            "no_host_url": no_host, "zones": dict(zone), "source_categories": dict(cats.most_common()),
            "category_ext_top": dict(ext_cat.most_common(10))}


STANDARD_DIRS = {"appdata", "desktop", "documents", "downloads", "pictures", "videos", "music", "onedrive", "contacts",
                 "favorites", "links", "saved games", "searches", "3d objects"}


@_probe(id="files.profile_size_walk", level="L2", family="files", tier="T2", collect="deep", gate="files.known_folders",
        timeout_ms=30000)
def files_profile_size_walk(h, facts):
    """Size of each top-level profile dir (depth <= 10, 300k entries/dir, 25 s total); operator dirs excluded."""
    home = _home()
    deadline = time.perf_counter() + 25.0
    rows = []
    loose = 0
    try:
        entries = list(os.scandir(home))
    except OSError:
        return {"present": False}
    for e in entries:
        try:
            s = e.stat(follow_symlinks=False)
        except OSError:
            continue
        if not e.is_dir(follow_symlinks=False):
            loose += 1
            continue
        if getattr(s, "st_file_attributes", 0) & FILE_ATTRIBUTE_REPARSE_POINT:
            continue
        low = e.name.lower()
        cls = ("operator" if OPERATOR_RX.match(e.name) else "dot" if low.startswith(".") else
               "standard" if low in STANDARD_DIRS else "custom")
        if cls == "operator":
            rows.append({"name": e.name, "class": cls, "bytes": 0, "skipped": "operator"})
            continue
        remain = deadline - time.perf_counter()
        if remain <= 0:
            rows.append({"name": e.name, "class": cls, "bytes": 0, "skipped": "budget"})
            continue
        st = _walk(e.path, max_depth=10, max_entries=300000,
                   budget_s=min(12.0 if low == "appdata" else 4.0, remain))
        rows.append({"name": e.name, "class": cls, "bytes": st["bytes"], "files": st["files"],
                     "truncated": st["truncated"], "ms": st.get("ms")})
    rows.sort(key=lambda r: -r["bytes"])
    shown = 0
    for r in rows:
        if r["class"] == "custom":
            shown += 1
            if shown > 5:
                r["name"] = f"custom#{shown}"
    cls = collections.Counter(r["class"] for r in rows)
    return {"present": True, "top_dirs": [r for r in rows if r["class"] != "operator"][:12],
            "class_counts": dict(cls), "custom_root_dirs": cls.get("custom", 0),
            "operator_dirs_excluded": cls.get("operator", 0),
            "custom_bytes": sum(r["bytes"] for r in rows if r["class"] == "custom"),
            "total_bytes": sum(r["bytes"] for r in rows), "loose_root_files": loose,
            "truncated_dirs": sum(1 for r in rows if r.get("truncated") or r.get("skipped") == "budget")}
