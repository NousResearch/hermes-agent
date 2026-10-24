"""SMS one-time codes for browser_vault_enter_code, read from the local macOS Messages store.

Opt-in (``vault.sms_otp.enabled``) and allowlisted per site: a code is taken only from a text that
(1) arrived after the code request window opened (default: 120 s before the call), (2) mentions one of
the keywords configured for the page's registrable domain, and (3) contains a single plausible code.
The code is returned to the caller for in-page injection only; the audit log records site, sender and
outcome, never the code.
"""

from __future__ import annotations

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
    re.compile(r"(?:code|passcode|otp|pin)\D{0,20}?(\d{4,8})\b", re.I),
    re.compile(r"\b(\d{4,8})\b[^.\n]{0,40}(?:code|passcode|otp)", re.I),
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


def site_keywords(origin: str, cfg: Optional[dict] = None) -> Optional[List[str]]:
    """Keywords for the page's site when SMS codes are enabled for it, else None."""
    cfg = _config() if cfg is None else cfg
    if not cfg.get("enabled") or os_family() != "darwin":
        return None
    sites = cfg.get("sites") or {}
    kws = sites.get(registrable_domain(origin))
    if not kws:
        return None
    return [str(k).lower() for k in (kws if isinstance(kws, list) else [kws]) if str(k).strip()]


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


def _recent_texts(db: Path, since_epoch: float) -> List[Dict[str, str]]:
    since_ns = int((since_epoch - _APPLE_EPOCH) * 1e9)
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2)
    try:
        rows = con.execute(
            "SELECT m.date, h.id, m.text, m.attributedBody FROM message m "
            "LEFT JOIN handle h ON m.handle_id = h.ROWID "
            "WHERE m.is_from_me = 0 AND m.date >= ? ORDER BY m.date DESC LIMIT 25", (since_ns,)).fetchall()
    finally:
        con.close()
    return [{"sender": str(h or ""), "body": _body(t, a)} for _, h, t, a in rows]


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
    keywords = site_keywords(origin, cfg)
    site = registrable_domain(origin)
    if not keywords:
        return None
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
        for msg in texts:  # newest first
            body = msg["body"].lower()
            if not any(k in body for k in keywords):
                continue
            code = extract_code(msg["body"])
            if code:
                _audit(site, msg["sender"], "filled")
                return code
        if now is not None or time.time() >= deadline:  # `now` given = single deterministic pass (tests)
            _audit(site, "", "no_matching_text")
            return None
        sleep(3)
