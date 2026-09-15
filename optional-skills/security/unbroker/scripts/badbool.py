"""Pull and parse the Big-Ass Data Broker Opt-Out List (BADBOOL) into broker records.

BADBOOL (https://github.com/yaelwrites/Big-Ass-Data-Broker-Opt-Out-List) is a
maintained, frequently-updated markdown list. `refresh` fetches it and parses the
"People Search Sites" section into records that merge UNDER the curated DB (curated
records always win). Auto-parsed records carry source="BADBOOL-auto" and
confidence="auto" so the agent treats their URLs as best guesses to verify first.

`parse()` is pure (markdown in, records out) so it is tested offline; `fetch()` is
the only network call and can be bypassed by passing markdown directly to refresh().
"""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import storage

DEFAULT_URL = (
    "https://raw.githubusercontent.com/yaelwrites/"
    "Big-Ass-Data-Broker-Opt-Out-List/master/README.md"
)
USER_AGENT = "Mozilla/5.0 (compatible; unbroker/1.0; data opt-out)"

# BADBOOL legend symbols.
SYMBOLS = {
    "crucial": "\U0001F490",  # 💐
    "high": "\u2620",          # ☠
    "gov_id": "\U0001F3AB",    # 🎫
    "phone": "\U0001F4DE",     # 📞
    "payment": "\U0001F4B0",   # 💰
}

_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_OPTOUT_HINT = re.compile(
    r"opt[\- ]?out|optout|removal|remove|suppress|control-privacy|delete", re.I
)
_FIND_HINT = re.compile(r"find|your information|search|look ?up|look for", re.I)


def slug(name: str) -> str:
    # Drop a trailing .com/.org/.info on the displayed name so "FastPeopleSearch.com"
    # matches the curated id "fastpeoplesearch"; keep .net/.id so distinct sites differ.
    n = re.sub(r"\.(com|org|info)\b", "", name.strip(), flags=re.I)
    return re.sub(r"[^a-z0-9]+", "", n.lower())


def _heading_flags(heading: str) -> tuple[str, dict]:
    flags = {key: (sym in heading) for key, sym in SYMBOLS.items()}
    name = heading
    for sym in SYMBOLS.values():
        name = name.replace(sym, "")
    name = name.replace("\ufe0f", "").strip()
    return name, flags


def _priority(flags: dict) -> str:
    if flags["crucial"]:
        return "crucial"
    if flags["high"]:
        return "high"
    return "standard"


def _pick(links: list[tuple[str, str]], hint: re.Pattern) -> str | None:
    for _text, url in links:
        if hint.search(url):
            return url
    for text, url in links:
        if hint.search(text):
            return url
    return None


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()[:600]


def _build(name: str, flags: dict, body: str) -> dict:
    links = _LINK_RE.findall(body)
    web = [(t, u) for t, u in links if u.lower().startswith("http")]
    mailtos = [u[7:] for _t, u in links if u.lower().startswith("mailto:")]
    optout_url = _pick(web, _OPTOUT_HINT)
    search_url = _pick(web, _FIND_HINT) or (web[0][1] if web else None)

    if flags["phone"]:
        method = "phone"
    elif optout_url:
        method = "web_form"
    elif mailtos:
        method = "email"
    else:
        method = "manual"

    return {
        "id": slug(name),
        "name": name,
        "category": "people_search",
        "priority": _priority(flags),
        "jurisdictions": ["US"],
        "search": {"method": "url_pattern", "url": search_url, "fetch": "browser",
                   "match_signal": "result", "by": ["name", "phone", "address"]},
        "optout": {
            "method": method,
            "url": optout_url,
            "email": mailtos[0] if mailtos else None,
            "requires": {
                "gov_id": flags["gov_id"],
                "phone_voice": flags["phone"],
                "payment": flags["payment"],
                "email_verification": False,
                "captcha": False,
                "account": False,
                "phone_callback": False,
            },
            "inputs": ["full_name", "contact_email"],
            "notes": _clean(body),
            "links": [{"text": t, "url": u} for t, u in links],
            "est_processing_days": 14,  # unknown for auto records; drives next_recheck_at
        },
        "source": "BADBOOL-auto",
        "confidence": "auto",
        "last_verified": None,
    }


def parse(markdown: str) -> list[dict]:
    """Parse the 'People Search Sites' section of BADBOOL into broker records."""
    records: list[dict] = []
    in_people = False
    heading: str | None = None
    body: list[str] = []

    def flush() -> None:
        nonlocal heading, body
        if heading is not None:
            name, flags = _heading_flags(heading)
            if name:
                records.append(_build(name, flags, "\n".join(body).strip()))
        heading, body = None, []

    for line in markdown.splitlines():
        if line.startswith("## "):
            flush()
            in_people = line[3:].strip().lower().startswith("people search")
            continue
        if not in_people:
            continue
        if line.startswith("### "):
            flush()
            heading = line[4:].strip()
        elif heading is not None:
            body.append(line)
    flush()
    return records


def fetch(url: str = DEFAULT_URL, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


MIN_EXPECTED = 20  # BADBOOL's People Search section lists ~47; far fewer => upstream reorg, warn


def refresh(cache_path: Path, url: str = DEFAULT_URL, markdown: str | None = None) -> dict:
    """Fetch (or accept) BADBOOL markdown, parse it, and write the snapshot cache."""
    md = markdown if markdown is not None else fetch(url)
    records = parse(md)
    storage.write_json(cache_path, records)
    out = {"parsed": len(records), "cache_path": str(cache_path), "source_url": url}
    if len(records) < MIN_EXPECTED:
        out["warning"] = (f"only {len(records)} parsed (expected >{MIN_EXPECTED}); BADBOOL's "
                          "'People Search Sites' section may have moved/reorganized - check the parser")
    return out
