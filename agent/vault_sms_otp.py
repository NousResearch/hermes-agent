"""SMS one-time codes for browser_vault_enter_code, read from the local macOS Messages store.

Opt-in (``vault.sms_otp.enabled``) and allowlisted per site. A code is taken only from a text that
(1) was sent by one of the senders configured for the page's registrable domain (exact match on the
short code, phone number or sender ID; a site with no senders is never eligible), (2) arrived after the
code request window opened (default: 120 s before the call), (3) contains every configured keyword as a
whole word, if any are configured, (4) contains a single plausible code, and (5) has not been used
before. Each text is consumed on use: its message id is recorded (hashed) under the Hermes home, so the
same code is never filled twice. The code is returned to the caller for in-page injection only; the
audit log records site, sender tail and outcome, never the code.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

from hermes_platform.host.facts import os_family

logger = logging.getLogger(__name__)

_APPLE_EPOCH = 978307200  # 2001-01-01 UTC
_MULTI_TLDS = {"co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "co.jp", "co.nz", "com.br",
               "com.mx", "co.in", "com.cy", "com.tr", "co.za", "com.sg", "com.hk"}
# Code phrasing first ("code is 123456", "123456 is your code"), then a lone 4-8 digit token.
_CODE_PATTERNS = (
    re.compile(r"(?:code|passcode|otp|pin)\D{0,20}?(\d{4,8})\b", re.IGNORECASE),
    re.compile(r"\b(\d{4,8})\b[^.\n]{0,40}(?:code|passcode|otp)", re.IGNORECASE),
)
_LONE_DIGITS = re.compile(r"(?<![\d$.,])\b(\d{4,8})\b(?![.,]\d)")


def registrable_domain(origin: str) -> str:
    host = (urlparse(origin).hostname or origin).lower().strip(".")
    parts = host.split(".")
    n = 3 if ".".join(parts[-2:]) in _MULTI_TLDS else 2
    return ".".join(parts[-n:])


def _config() -> dict:
    try:
        from hermes_cli.config import cfg_get, read_raw_config
        cfg = cfg_get(read_raw_config(), "vault", "sms_otp", default={})
        return cfg if isinstance(cfg, dict) else {}
    except Exception as exc:
        logger.debug("sms_otp config read failed: %s", exc)
        return {}


def normalize_sender(sender: str) -> str:
    """Comparable form of a Messages handle: phone numbers and short codes reduce to their digits
    (``+1 (555) 010-0100`` -> ``15550100100``); emails and alphanumeric sender IDs are lowercased.
    Matching is exact on that form, so configure full international numbers."""
    s = str(sender or "").strip().lower()
    if s and re.fullmatch(r"[+\d\s().-]+", s):
        return re.sub(r"\D", "", s)
    return s


def site_rules(origin: str, cfg: Optional[dict] = None) -> Optional[dict[str, list[str]]]:
    """``{"senders": [...], "keywords": [...]}`` for the page's site when SMS codes are enabled for it,
    else None. A site entry without at least one sender is refused: body text is not an identity."""
    cfg = _config() if cfg is None else cfg
    if not cfg.get("enabled") or os_family() != "darwin":
        return None
    entry = (cfg.get("sites") or {}).get(registrable_domain(origin))
    if not isinstance(entry, dict):
        return None

    def _list(value) -> list[str]:
        items = value if isinstance(value, list) else [value] if value else []
        return [str(v).strip() for v in items if str(v).strip()]

    senders = [normalize_sender(v) for v in _list(entry.get("senders"))]
    senders = [s for s in senders if s]
    if not senders:
        return None
    return {"senders": senders, "keywords": [k.lower() for k in _list(entry.get("keywords"))]}


def _has_word(body: str, keyword: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", body) is not None


def extract_code(text: str) -> Optional[str]:
    """The code only when the text holds exactly one 4-8 digit token and reads like a code message;
    anything with several candidates (order numbers, amounts) is refused rather than guessed."""
    found = {m.group(1) for m in _LONE_DIGITS.finditer(text)}
    if len(found) != 1:
        return None
    code = found.pop()
    return code if any(p.search(text) for p in _CODE_PATTERNS) else None


def _body(text: Optional[str], attributed: Optional[bytes]) -> str:
    if text:
        return text
    if not attributed:
        return ""
    # Newer macOS stores some bodies only in the typedstream attributedBody; the string follows NSString.
    raw = attributed.split(b"NSString", 1)[-1]
    return raw.decode("utf-8", "ignore")


def _recent_texts(db: Path, since_epoch: float) -> list[dict[str, str]]:
    since_ns = int((since_epoch - _APPLE_EPOCH) * 1e9)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2)
    try:
        rows = con.execute(
            "SELECT m.ROWID, m.date, h.id, m.text, m.attributedBody FROM message m "
            "LEFT JOIN handle h ON m.handle_id = h.ROWID "
            "WHERE m.is_from_me = 0 AND m.date >= ? ORDER BY m.date DESC LIMIT 25", (since_ns,)).fetchall()
    finally:
        con.close()
    return [{"id": f"{rid}:{d}", "sender": str(h or ""), "body": _body(t, a)} for rid, d, h, t, a in rows]


_USED_TTL_SECONDS = 24 * 3600


def _used_path() -> Path:
    from hermes_constants import get_hermes_home
    return Path(get_hermes_home()) / "state" / "vault_sms_otp_used.json"


def _message_key(message_id: str) -> str:
    return hashlib.sha256(message_id.encode()).hexdigest()


def _used_keys() -> dict[str, float]:
    from utils import read_json_or_empty
    data = read_json_or_empty(_used_path())
    return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}


def _mark_used(message_id: str) -> None:
    """Record a consumed text (hashed message id only) and drop entries older than a day. A failed
    write raises, so the caller never fills a code it could not mark as used."""
    from utils import atomic_json_write
    now = time.time()
    keys = {k: t for k, t in _used_keys().items() if now - t < _USED_TTL_SECONDS}
    keys[_message_key(message_id)] = now
    path = _used_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, keys, mode=0o600)


def _audit(site: str, sender: str, outcome: str) -> None:
    try:
        from hermes_constants import get_hermes_home
        path = Path(get_hermes_home()) / "logs" / "vault_sms_otp.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "site": site,
                                 "sender": sender[-4:] if sender else "", "outcome": outcome}) + "\n")
    except Exception as exc:
        logger.debug("sms_otp audit write failed: %s", exc)


def wait_for_code(origin: str, *, cfg: Optional[dict] = None, db: Optional[Path] = None,
                  now: Optional[float] = None, sleep=time.sleep) -> Optional[str]:
    """Poll Messages for a code for ``origin``. None when disabled, not allowlisted, or nothing arrives."""
    cfg = _config() if cfg is None else cfg
    rules = site_rules(origin, cfg)
    site = registrable_domain(origin)
    if not rules:
        return None
    senders, keywords = set(rules["senders"]), rules["keywords"]
    db = db or Path.home() / "Library" / "Messages" / "chat.db"
    start = time.time() if now is None else now
    window_open = start - float(cfg.get("lookback_seconds", 120))
    deadline = start + float(cfg.get("max_wait_seconds", 90))
    while True:
        try:
            texts = _recent_texts(db, window_open)
        except sqlite3.Error as exc:
            _audit(site, "", f"messages_unreadable:{type(exc).__name__}")
            return None
        used = _used_keys()
        for msg in texts:  # newest first; only allowlisted senders are ever considered
            if normalize_sender(msg["sender"]) not in senders:
                continue
            if _message_key(msg["id"]) in used:
                continue
            body = msg["body"].lower()
            if not all(_has_word(body, k) for k in keywords):
                continue
            code = extract_code(msg["body"])
            if not code:
                continue
            try:
                _mark_used(msg["id"])
            except Exception as exc:  # fail closed: an unrecorded code could be replayed
                _audit(site, msg["sender"], f"ledger_unwritable:{type(exc).__name__}")
                return None
            _audit(site, msg["sender"], "filled")
            return code
        if now is not None or time.time() >= deadline:  # `now` given = single deterministic pass (tests)
            _audit(site, "", "no_matching_text")
            return None
        sleep(3)
