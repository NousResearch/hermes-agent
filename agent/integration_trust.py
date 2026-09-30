"""How far Czesiek may go on its own inside a connected service, and a record of what it did there.

Trust is one of four levels per integration (``approvals.integrations.<id>`` in ``config.yaml``):

``read``     only reads; any write is blocked
``propose``  reads and shows exactly what it would do; any write is blocked until the user raises the level
``ask``      acts only after the user says yes in the approval prompt (the default)
``auto``     acts without asking

Only actions that *change* something count (send a mail, create or delete an event, share or delete a file,
write a sheet). Reading is never gated here. Google is the one integration enforced today, through its
``google_api.py`` command line; other services have no write path to classify yet.

This module is pure policy plus the decision log; :mod:`tools.approval_integrations` plugs it into the
approval gate so the level cannot be bypassed by the model.
"""

from __future__ import annotations

import json
import re
import shlex
import time
from dataclasses import dataclass
from typing import Any, Optional

from hermes_constants import get_hermes_home

LEVELS = ("read", "propose", "ask", "auto")
DEFAULT_LEVEL = "ask"
INTEGRATIONS = ("google",)

LOG_FILE = "integration_actions.jsonl"
_LOG_MAX_BYTES = 1_000_000
_LOG_KEEP_LINES = 500

_GOOGLE_CALL = re.compile(r"google_api\.py\s+(gmail|calendar|drive|sheets|docs)\s+([a-z][a-z-]*)")

# (group, subcommand) -> (english label, polish label). Everything not listed reads.
_GOOGLE_WRITES = {
    ("gmail", "send"): ("Send an email", "Wysłanie maila"),
    ("gmail", "reply"): ("Reply to an email", "Odpowiedź na maila"),
    ("gmail", "modify"): ("Change mail labels", "Zmiana etykiet maila"),
    ("calendar", "create"): ("Create a calendar event", "Dodanie wydarzenia do kalendarza"),
    ("calendar", "delete"): ("Delete a calendar event", "Usunięcie wydarzenia z kalendarza"),
    ("drive", "upload"): ("Upload a file to Drive", "Wgranie pliku na Dysk"),
    ("drive", "create-folder"): ("Create a Drive folder", "Utworzenie folderu na Dysku"),
    ("drive", "share"): ("Share a Drive file", "Udostępnienie pliku z Dysku"),
    ("drive", "delete"): ("Delete a Drive file", "Usunięcie pliku z Dysku"),
    ("sheets", "update"): ("Write to a spreadsheet", "Zapis do arkusza"),
    ("sheets", "append"): ("Add rows to a spreadsheet", "Dopisanie wierszy do arkusza"),
    ("sheets", "create"): ("Create a spreadsheet", "Utworzenie arkusza"),
    ("docs", "create"): ("Create a document", "Utworzenie dokumentu"),
    ("docs", "append"): ("Add text to a document", "Dopisanie tekstu do dokumentu"),
}

_PREVIEW_FLAGS = (("--to", "to"), ("--subject", "subject"), ("--summary", "title"), ("--start", "start"),
                  ("--email", "with"), ("--role", "role"))


@dataclass(frozen=True)
class IntegrationAction:
    integration: str
    action: str       # "gmail.send"
    label_en: str
    label_pl: str
    preview: str      # recipient / subject / time — never the body

    @property
    def key(self) -> str:
        return f"integration:{self.integration}:{self.action}"

    def describe(self, lang: str = "en") -> str:
        label = self.label_pl if lang.startswith("pl") else self.label_en
        return f"{label}: {self.preview}" if self.preview else label


def _flag_values(segment: str) -> str:
    try:
        tokens = shlex.split(segment, comments=False, posix=True)
    except ValueError:
        return ""
    found = []
    for flag, name in _PREVIEW_FLAGS:
        if flag in tokens:
            index = tokens.index(flag)
            if index + 1 < len(tokens):
                value = tokens[index + 1]
                found.append(f"{name} {value[:80]}" if name != "subject" else f"„{value[:80]}”")
    return " · ".join(found)


def classify_command(command: str) -> Optional[IntegrationAction]:
    """The first *writing* integration call in a shell command, or None when it only reads."""
    for match in _GOOGLE_CALL.finditer(command or ""):
        group, sub = match.group(1), match.group(2)
        labels = _GOOGLE_WRITES.get((group, sub))
        if labels:
            segment = re.split(r"\s*(?:&&|\|\||;|\|)\s*", command[match.start():], maxsplit=1)[0]
            return IntegrationAction("google", f"{group}.{sub}", labels[0], labels[1], _flag_values(segment))
    return None


def normalize_level(value: Any) -> str:
    return value if isinstance(value, str) and value in LEVELS else DEFAULT_LEVEL


def trust_level(integration: str, approvals_cfg: Optional[dict] = None) -> str:
    """The configured level for ``integration``; anything missing or unrecognised is the safe default."""
    if approvals_cfg is None:
        from tools import approval_context
        approvals_cfg = approval_context._get_approval_config()
    integrations = approvals_cfg.get("integrations") if isinstance(approvals_cfg, dict) else None
    return normalize_level(integrations.get(integration) if isinstance(integrations, dict) else None)


def all_levels() -> dict[str, str]:
    from tools import approval_context
    cfg = approval_context._get_approval_config()
    return {name: trust_level(name, cfg) for name in INTEGRATIONS}


def set_level(integration: str, level: str) -> dict[str, str]:
    if integration not in INTEGRATIONS:
        raise ValueError(f"unknown integration: {integration}")
    if level not in LEVELS:
        raise ValueError(f"level must be one of {', '.join(LEVELS)}")
    from hermes_cli.config import read_raw_config, save_config
    raw = read_raw_config()
    approvals = raw.get("approvals") if isinstance(raw.get("approvals"), dict) else {}
    integrations = approvals.get("integrations") if isinstance(approvals.get("integrations"), dict) else {}
    integrations[integration] = level
    approvals["integrations"] = integrations
    raw["approvals"] = approvals
    save_config(raw)
    return all_levels()


def block_message(action: IntegrationAction, level: str) -> str:
    if level == "read":
        return (f"BLOCKED: Google is set to 'read only' (approvals.integrations.google), so this write is not "
                f"allowed: {action.describe()}. Do NOT retry or work around it. Tell the user you could not do it "
                "and that they can raise Google's trust level under Integrations → Trust levels.")
    return (f"BLOCKED: Google is set to 'read and propose' (approvals.integrations.google). Do NOT run this: "
            f"{action.describe()}. Instead show the user exactly what you would do — recipient, subject and the "
            "full text, or the exact change — and tell them they can allow it by raising Google's trust level "
            "under Integrations → Trust levels.")


# ── decision log ───────────────────────────────────────────────────────────

def log_decision(action: IntegrationAction, decision: str, level: str) -> None:
    """Append one line to the integration log. ``decision``: approved | denied | blocked | auto. Never the body."""
    try:
        path = get_hermes_home() / LOG_FILE
        entry = {"ts": int(time.time()), "integration": action.integration, "action": action.action,
                 "label": action.label_en, "label_pl": action.label_pl, "preview": action.preview,
                 "decision": decision, "level": level}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if path.stat().st_size > _LOG_MAX_BYTES:
            lines = path.read_text(encoding="utf-8").splitlines()[-_LOG_KEEP_LINES:]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError:
        pass  # a log that cannot be written must never block or fail the action's gate


def read_log(limit: int = 50) -> list[dict[str, Any]]:
    """Most recent decisions first."""
    try:
        lines = (get_hermes_home() / LOG_FILE).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in reversed(lines):
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
        if len(out) >= limit:
            break
    return out
