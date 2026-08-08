"""Civitai skill — shared helpers (HTTP, Meilisearch, formatting)."""
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API_BASE = "https://civitai.com/api/v1"
MEILI_URL = "https://search-new.civitai.com/multi-search"

# NOTE FOR SECURITY SCANNERS:
# The 64-char hex string below is NOT a secret. It is Civitai's public
# Meilisearch search-only key, embedded in the JavaScript bundle that every
# browser visiting civitai.com downloads. It is not tied to any account and
# is the documented way to query Civitai's search index. If it rotates, the
# user can override it via the MEILISEARCH_KEY env var — see SKILL.md
# Pitfalls #5 for the recovery procedure.
_MEILI_PUBLIC_KEY = "8c46eb2508e21db1e9828a97968d91ab1ca1caa5f70a00e88a2ba1e286603b61"
MEILI_KEY = os.environ.get("MEILISEARCH_KEY") or _MEILI_PUBLIC_KEY

UA = "civitai-skill/1.0"

SORT_MAP = {
    "Most Downloaded": "metrics.downloadCount:desc",
    "Highest Rated":   "metrics.thumbsUpCount:desc",
    "Most Collected":  "metrics.collectedCount:desc",
    "Most Comments":   "metrics.commentCount:desc",
    "Most Tipped":     "metrics.tippedAmountCount:desc",
    "Newest":          "createdAt:desc",
    "Oldest":          "createdAt:asc",
}

ATTRS = [
    "id", "name", "type", "nsfw", "nsfwLevel", "metrics", "user",
    "triggerWords", "tags", "version", "createdAt",
]

BROWSING_LEVEL_MAP = {
    "PG": 1, "PG-13": 2, "PG13": 2, "R": 4, "X": 8, "XXX": 16,
}


def die(msg, code=2):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def _esc(s):
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _coerce(v):
    """Coerce a query-param value to the form urlencode expects."""
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, list):
        return [str(x) for x in v]
    return str(v)


def api_get(path, params=None):
    url = f"{API_BASE}/{path.lstrip('/')}"
    if params:
        clean = {k: _coerce(v) for k, v in params.items() if v is not None}
        if clean:
            url += "?" + urllib.parse.urlencode(clean, doseq=True)

    headers = {"User-Agent": UA}
    key = os.environ.get("CIVITAI_API_KEY", "")
    if key:
        headers["Authorization"] = f"Bearer {key}"

    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503, 504) and attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
            if e.code == 404:
                die(f"not found: {path}")
            if e.code in (401, 403):
                die("auth failed — check CIVITAI_API_KEY")
            if e.code == 429:
                die("rate limited, try again later")
            die(f"http {e.code}: {e.read()[:200].decode(errors='replace')}")
        except urllib.error.URLError:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            die("network error after retries")
    die("retries exhausted")


def meili_search(q, types=None, base_model=None, tag=None, username=None,
                 sort="Most Downloaded", nsfw=None, limit=20, offset=0):
    f = []
    if types:
        c = [f'type = "{_esc(t)}"' for t in types]
        f.append(c[0] if len(c) == 1 else "(" + " OR ".join(c) + ")")
    if base_model:
        f.append(f'version.baseModel = "{_esc(base_model)}"')
    if tag:
        f.append(f'tags.name = "{_esc(tag.lower())}"')
    if username:
        f.append(f'user.username = "{_esc(username)}"')
    if nsfw is False:
        f.append("nsfwLevel = 1")

    inner = {
        "q": q or "",
        "indexUid": "models_v9",
        "limit": min(limit, 100),
        "offset": offset,
        "sort": [SORT_MAP.get(sort, "metrics.downloadCount:desc")],
        "attributesToRetrieve": ATTRS,
    }
    if f:
        inner["filter"] = " AND ".join(f)

    body = json.dumps({"queries": [inner]}).encode()
    req = urllib.request.Request(
        MEILI_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {MEILI_KEY}",
            "Content-Type": "application/json",
            "User-Agent": UA,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())["results"][0]
    except urllib.error.HTTPError as e:
        die(f"meilisearch http {e.code}: {e.read()[:200].decode(errors='replace')}")
    except urllib.error.URLError:
        die("network error reaching Meilisearch")


def parse_browsing_level(s):
    if not s or s.lower() == "all":
        return 31
    m = 0
    for part in s.upper().replace(" ", "").split(","):
        m |= BROWSING_LEVEL_MAP.get(part, 0)
    return m or 31


def fmt_size(kb):
    try:
        kb = float(kb)
    except (TypeError, ValueError):
        return "?"
    if kb >= 1024 * 1024:
        return f"{kb / 1024 / 1024:.2f} GB"
    if kb >= 1024:
        return f"{kb / 1024:.1f} MB"
    return f"{kb:.0f} KB"


def fmt_int(n):
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return str(n)


def truncate(s, n=500):
    if not s:
        return ""
    s = str(s).replace("\r\n", " ").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n].rstrip() + "…"


def emit_json(data):
    print(json.dumps(data, indent=2, default=str))


# --------------------------------------------------------------------------- #
# JSON slimming helpers — shared by show.py and search.py
#
# Hermes terminal transport caps stdout around 20-50 KB; raw Civitai /models
# payloads can exceed 100 KB per model (HTML descriptions, per-version image
# arrays, six redundant hash formats). When --ids fetches 16+ models in one
# call, the response easily reaches 600 KB+. Oversized JSON gets truncated
# mid-stream and breaks json.loads downstream. These helpers strip everything
# an agent doesn't need from --json output so it stays well under the cap.
# --------------------------------------------------------------------------- #

MAX_VERSIONS_IN_JSON = 20


def slim_files(files):
    """Trim each file dict to fields agents consume.

    Keeps only SHA256 + AutoV2 hashes (drops AutoV1, AutoV3, CRC32, BLAKE3).
    """
    out = []
    for f in (files or []):
        raw = f.get("hashes") or {}
        hashes = {k: raw[k] for k in ("SHA256", "AutoV2") if raw.get(k)}
        out.append({
            "name":             f.get("name"),
            "sizeKB":           f.get("sizeKB"),
            "type":             f.get("type"),
            "primary":          f.get("primary"),
            "pickleScanResult": f.get("pickleScanResult"),
            "virusScanResult":  f.get("virusScanResult"),
            "hashes":           hashes,
            "downloadUrl":      f.get("downloadUrl"),
        })
    return out


def slim_model(m):
    """Compact /models/{id} payload safe for terminal stdout.

    Drops HTML description blob and per-version image arrays; caps versions
    at MAX_VERSIONS_IN_JSON. Surfaces a _versions_truncated flag so callers
    can detect clipped models. Used by show.py model and by search.py models
    when --json is set on a REST path (--ids or non-query search).
    """
    raw_versions = m.get("modelVersions") or []
    truncated = len(raw_versions) > MAX_VERSIONS_IN_JSON
    versions_slim = []
    for v in raw_versions[:MAX_VERSIONS_IN_JSON]:
        versions_slim.append({
            "id":           v.get("id"),
            "name":         v.get("name"),
            "baseModel":    v.get("baseModel"),
            "createdAt":    v.get("createdAt"),
            "publishedAt":  v.get("publishedAt"),
            "status":       v.get("status"),
            "trainedWords": v.get("trainedWords"),
            "files":        slim_files(v.get("files")),
        })
    return {
        "id":                  m.get("id"),
        "name":                m.get("name"),
        "type":                m.get("type"),
        "nsfw":                m.get("nsfw"),
        "stats":               m.get("stats"),
        "creator":             m.get("creator"),
        "tags":                m.get("tags"),
        "modelVersions":       versions_slim,
        "_versions_truncated": truncated,
    }


def slim_version(v):
    """Compact single-version payload for --json (show.py version)."""
    pm = v.get("model") or {}
    return {
        "id":           v.get("id"),
        "modelId":      v.get("modelId"),
        "name":         v.get("name"),
        "baseModel":    v.get("baseModel"),
        "trainedWords": v.get("trainedWords"),
        "model":        {"name": pm.get("name"), "type": pm.get("type")},
        "files":        slim_files(v.get("files")),
    }


def slim_prompt_image(im):
    """Compact image entry for prompt-mining --json (show.py prompts)."""
    meta = im.get("meta") or {}
    stats = im.get("stats") or {}
    user = im.get("username") or (im.get("user") or {}).get("username")
    reactions = sum(stats.get(k, 0) for k in
                    ("heartCount", "likeCount", "laughCount", "cryCount"))
    loras = []
    for r in (meta.get("resources") or []):
        if (r.get("type") or "").lower() in ("lora", "locon", "dora"):
            loras.append({
                "name":   r.get("name"),
                "type":   r.get("type"),
                "weight": r.get("weight"),
            })
    return {
        "id":        im.get("id"),
        "username":  user,
        "reactions": reactions,
        "baseModel": im.get("baseModel"),
        "url":       f"https://civitai.com/images/{im.get('id', '')}",
        "meta": {
            "prompt":         meta.get("prompt"),
            "negativePrompt": meta.get("negativePrompt"),
            "steps":          meta.get("steps"),
            "sampler":        meta.get("sampler"),
            "cfgScale":       meta.get("cfgScale"),
            "seed":           meta.get("seed"),
            "resources":      loras,
        },
    }