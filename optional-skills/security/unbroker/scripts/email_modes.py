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
from urllib.parse import urlsplit

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


def broker_domains(broker: dict) -> set[str]:
    """Use only explicitly declared domains, never a guessed public-suffix tail."""
    domains = set()
    for section in ("optout", "search"):
        try:
            host = urlsplit((broker.get(section) or {}).get("url") or "").hostname
        except ValueError:
            continue
        if host:
            domains.add(host.lower().removeprefix("www."))
    address = (broker.get("optout") or {}).get("email") or ""
    if "@" in address:
        domains.add(address.rsplit("@", 1)[1].lower())
    return domains


def domain_matches(host: str, domains: set[str]) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def extract_verification_link(email_body: str, broker: dict | None = None) -> str | None:
    """Return an HTTPS broker-origin candidate, not proof of sender/request identity.

    The operator must bind the email to the pending subject/request and inspect the
    final official destination before opening it. Third-party verification providers
    require separate review; generic verification words never authorize an origin.
    """
    domains = broker_domains(broker or {})
    best_score, best_link = 0, None
    for link in _LINK_RE.findall(email_body or ""):
        try:
            parsed = urlsplit(link)
            if (parsed.scheme.lower() != "https" or parsed.username is not None
                    or parsed.password is not None or parsed.port not in (None, 443)
                    or not domain_matches(parsed.hostname or "", domains)):
                continue
        except ValueError:
            continue
        score = 1 + 2 * any(h in (parsed.path + parsed.query).lower() for h in _VERIFY_HINTS)
        if score > best_score:
            best_score, best_link = score, link
    return best_link
