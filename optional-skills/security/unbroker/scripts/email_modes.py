"""Email modes A/B/C helpers + anti-phishing verification-link extraction.

Mode A (default): render a ready-to-send draft to disk; the operator sends it.
Mode B/C: the agent SENDS via a Hermes email mechanism (IMAP/SMTP gateway,
`himalaya`, AgentMail, or Gmail via `google-workspace`) and READS the reply to
resolve the verification link with `extract_verification_link`. Those transports
are driven by the agent through native tools; this module stays network-free so
the hermetic tests pass.
"""
from __future__ import annotations

import re
from pathlib import Path

import legal
import paths

_LINK_RE = re.compile(r"https?://[^\s\"'<>)\]]+", re.IGNORECASE)
_VERIFY_HINTS = ("opt", "remov", "verif", "confirm", "unsubscrib", "suppress", "delete", "privacy")


def render_draft(broker: dict, fields: dict, out_dir: Path | None = None) -> Path:
    """Mode A: write a ready-to-send opt-out email for the operator to send."""
    body = legal.render_optout_email(broker, fields)
    out_dir = out_dir or (paths.data_dir() / "drafts")
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{broker.get('id', 'broker')}.txt"
    fp.write_text(body, encoding="utf-8")
    return fp


def render_request_draft(broker: dict, fields: dict, kind: str = "generic",
                         out_dir: Path | None = None) -> Path:
    """Mode A: write a ready-to-send request of a specific KIND.

    kind: generic | ccpa | ccpa_agent | ccpa_indirect | gdpr. Used for indirect-exposure
    (ccpa_indirect) and explicit legal requests, where the generic opt-out wording is wrong.
    The filename is suffixed with the kind so an indirect request does not overwrite an opt-out draft.
    """
    body = legal.render_request(kind, broker, fields)
    out_dir = out_dir or (paths.data_dir() / "drafts")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if kind == "generic" else f"-{kind}"
    fp = out_dir / f"{broker.get('id', 'broker')}{suffix}.txt"
    fp.write_text(body, encoding="utf-8")
    return fp


def extract_verification_link(email_body: str, broker: dict | None = None) -> str | None:
    """Return the most likely opt-out/verification link from an email body.

    Anti-phishing: a link is only returned if its URL matches an opt-out hint
    and/or the broker's own domain; arbitrary links score 0 and are ignored.
    """
    candidates = _LINK_RE.findall(email_body or "")
    if not candidates:
        return None

    domain = ""
    if broker:
        url = (broker.get("optout") or {}).get("url") or (broker.get("search") or {}).get("url") or ""
        m = re.search(r"https?://([^/]+)", url)
        if m:
            domain = m.group(1).replace("www.", "")

    best_score, best_link = 0, None
    for link in candidates:
        low = link.lower()
        score = 0
        if any(h in low for h in _VERIFY_HINTS):
            score += 2
        if domain and domain in low:
            score += 3
        if score > best_score:
            best_score, best_link = score, link
    return best_link
