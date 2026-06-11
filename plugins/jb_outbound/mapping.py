"""Mapping d'un appel d'outil sortant → DraftRequest (pour l'affichage de la proposition).

Le DraftRequest est relayé au control-plane et affiché au client. On y met le STRICT nécessaire à
l'affichage (kind / destinataire / titre / aperçu) ; les arguments complets de l'outil (corps,
pièces) restent LOCAUX (cf. store.py) et ne transitent pas — minimisation RGPD. Le replay se fait
à l'identique depuis le store, donc l'extraction ci-dessous n'a qu'un rôle d'aperçu (best-effort).

`kind` doit appartenir à PROPOSAL_KINDS côté @jb/validation : email | post | sms.
"""

from __future__ import annotations

from typing import Any, Dict

_RECIPIENT_KEYS = ("recipient_email", "recipient", "to", "to_email", "email", "chat_id", "channel")
_SUBJECT_KEYS = ("subject", "title", "headline")
_BODY_KEYS = ("body", "text", "content", "message", "caption")


def _first(args: Dict[str, Any], keys) -> str:
    for k in keys:
        v = args.get(k)
        if isinstance(v, (str, int)) and str(v).strip():
            return str(v).strip()
    # Composio imbrique parfois les paramètres sous "arguments"/"params"/"input".
    for nest in ("arguments", "params", "input", "data"):
        sub = args.get(nest)
        if isinstance(sub, dict):
            r = _first(sub, keys)
            if r:
                return r
    return ""


def kind_for(tool_name: str) -> str:
    upper = (tool_name or "").upper()
    if "EMAIL" in upper or "GMAIL" in upper or "OUTLOOK" in upper or "MAIL" in upper:
        return "email"
    if any(s in upper for s in ("LINKEDIN", "TWITTER", "FACEBOOK", "INSTAGRAM", "POST", "PUBLISH")):
        return "post"
    return "sms"  # send_message Telegram + défaut (message direct)


def _truncate(s: str, n: int = 140) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def to_draft(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    args = args or {}

    # Action d'un MCP additionnel managé : on affiche le LIBELLÉ fourni par l'opérateur (white-label),
    # jamais le nom technique de l'outil. kind="action" → l'instance rejoue l'outil à l'approbation.
    from . import managed
    action = managed.action_for(tool_name)
    if action is not None:
        body = _first(args, _BODY_KEYS) or _first(args, _SUBJECT_KEYS)
        return {
            "kind": str(action.get("kind") or "action"),
            "title": str(action.get("label") or "Action à valider"),
            "preview": _truncate(body),
            "to": _first(args, _RECIPIENT_KEYS),
            # channel ≠ telegram/email → côté portail, l'instance exécute (replay), pas le control-plane.
            "payload": {"channel": "mcp"},
        }

    kind = kind_for(tool_name)
    to = _first(args, _RECIPIENT_KEYS)
    subject = _first(args, _SUBJECT_KEYS)
    body = _first(args, _BODY_KEYS)

    if kind == "email":
        title = subject or (f"Email à {to}" if to else "Email à préparer")
    elif kind == "post":
        title = f"Publication ({to})" if to else "Publication à préparer"
    else:
        title = f"Message à {to}" if to else "Message à préparer"

    preview = _truncate(body or subject or "")
    return {
        "kind": kind,
        "title": title,
        "preview": preview,
        "to": to,
        # payload minimal : on n'y met JAMAIS le corps/les args (PII reste locale au store).
        "payload": {"channel": "telegram" if kind == "sms" else "composio"},
    }
