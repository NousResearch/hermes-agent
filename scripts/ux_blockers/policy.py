"""Pure UX classification and candidate-scoped maintainer commands."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

CONFIRMED = "ux:confirmed"
BLOCKER = "impact:ux-release-blocker"
FRICTION = "impact:ux-friction"
REVIEW = "ux:needs-review"
MARKER = "<!-- hermes-ux-triage:v1 -->"
RECEIPT = "<!-- hermes-ux-confirmation:v1 -->"
BOT = "github-actions[bot]"
# The existing policy issue is the append-only bot ledger. Closing it does not
# remove records. Fork adopters select their own issue in GitHub's constructor.
LEDGER_ISSUE = 157333
SURFACES = ("Desktop", "Terminal", "Telegram", "Discord")
JOURNEYS = ("Install / setup", "Sign-in / provider setup", "First reply",
            "Return to a conversation", "Update / recovery")
RECOVERIES = ("Clear action in the same surface", "Terminal or another surface required",
              "Manual files or configuration required", "Reinstall or outside help required",
              "No recovery found", "Unknown")
LABELS = {
    BLOCKER: ("b60205", "Proposed UX P0: supported journey blocked without normal recovery"),
    FRICTION: ("fbca04", "UX P1: disrupted supported journey with in-product recovery"),
    REVIEW: ("d4c5f9", "UX recoverability assessment needs maintainer review"),
    CONFIRMED: ("b60205", "Maintainer-confirmed UX blocker retained until candidate disposition"),
}


@dataclass(frozen=True)
class Assessment:
    priority: str
    reason: str
    evidence: str


def sections(body: str) -> dict[str, str]:
    parts = re.split(r"(?m)^### (.+)\s*$", body)
    return {parts[i].strip(): parts[i + 1].strip() for i in range(1, len(parts), 2)}


def classify(body: str) -> Assessment:
    fields = sections(body)
    evidence = fields.get("User Journey and Recovery (optional)", "")
    if (fields.get("UX surface") not in SURFACES or fields.get("UX journey") not in JOURNEYS
            or not evidence or evidence == "_No response_"):
        return Assessment("unknown", "Identify a supported surface, journey and recovery evidence.", evidence)
    progress = fields.get("UX progress")
    recovery = fields.get("UX recovery")
    if progress == "Blocked or falsely reports success" and recovery in RECOVERIES[1:5]:
        return Assessment("P0", "The reported core journey is blocked without normal recovery.", evidence)
    if recovery == RECOVERIES[0] and progress in ("Blocked or falsely reports success", "Disrupted but usable"):
        return Assessment("P1", "The reported journey is disrupted with an in-product recovery.", evidence)
    return Assessment("unknown", "The report does not establish the UX P0/P1 recovery boundary.", evidence)


@dataclass(frozen=True)
class Command:
    action: str
    candidate: str = ""
    disposition: str = ""
    evidence: str = ""
    reason: str = ""
    canonical: int = 0


def parse_command(body: str) -> Command | None:
    """Require a whole comment: prose, quote blocks and code blocks cannot execute."""
    text = body.strip()
    duplicate = re.fullmatch(r"/ux duplicate #?([1-9][0-9]*)", text)
    if duplicate:
        return Command("duplicate", canonical=int(duplicate[1]))
    match = re.fullmatch(
        r"/ux (confirm|resolve) (?:(fixed|recovered|unaffected|exception) )?"
        r"([a-f0-9]{40}) (https://\S+) ([^\r\n]+)", text)
    if match is None or (match[1] == "resolve") != bool(match[2]):
        return None
    try:
        url = urlsplit(match[4])
    except ValueError:
        return None
    if not url.hostname or url.username or url.password or not match[5].strip():
        return None
    return Command(match[1], match[3], match[2] or "", match[4], match[5].strip())


def latest_commands(comments: list[dict], trusted) -> list[tuple[dict, Command]]:
    """Only unedited, current write-authority human commands can disposition a gate."""
    commands = []
    for comment in sorted(comments, key=lambda row: row["id"]):
        user = comment["user"]
        if user.get("type") != "User":
            continue
        body = comment.get("body") or ""
        command = parse_command(body)
        if command is None:
            continue
        if trusted(user["login"]):
            # Edited authority is ambiguous. Keep a blocking event so an older
            # resolution cannot become effective after a confirmation is edited.
            commands.append((comment, command if comment.get("updated_at") == comment["created_at"]
                             else Command("invalid")))
    # A reporter can delete their own comment. The bot's acknowledgement keeps
    # a confirmed blocker from disappearing when its source is removed/edited.
    by_id = {row["id"]: row for row in comments}
    for row in comments:
        if row["user"].get("login") != BOT or row["user"].get("type") != "Bot":
            continue
        body = row.get("body") or ""
        if not body.startswith(RECEIPT):
            continue
        receipt = json.loads(body[len(RECEIPT):].strip())
        source = by_id.get(receipt["id"])
        if (source is None or source["user"].get("type") != "User"
                or source.get("updated_at") != source["created_at"]
                or not trusted(source["user"]["login"])
                or hashlib.sha256((source.get("body") or "").encode()).hexdigest() != receipt["sha256"]):
            commands.append(({**row, "id": receipt["id"]}, Command("invalid")))
    return sorted(commands, key=lambda pair: pair[0]["id"])


def decision(comments: list[dict], candidate: str, trusted) -> dict:
    commands = latest_commands(comments, trusted)
    confirmations = [(row, cmd) for row, cmd in commands if cmd.action in ("confirm", "invalid")]
    if not confirmations:
        return {"status": "blocked", "reason": "Confirmed index has no valid maintainer confirmation."}
    confirmation = confirmations[-1][0]
    later = [(row, cmd) for row, cmd in commands if row["id"] > confirmation["id"]]
    dispositions = [(row, cmd) for row, cmd in later if cmd.action == "duplicate"
                    or (cmd.action == "resolve" and cmd.candidate == candidate)]
    if dispositions and dispositions[-1][1].action == "resolve":
        row, cmd = dispositions[-1]
        return {"status": "cleared", "disposition": cmd.disposition,
                "evidence": cmd.evidence, "reason": cmd.reason, "comment": row["html_url"]}
    if dispositions:
        row, cmd = dispositions[-1]
        return {"status": "duplicate", "canonical": cmd.canonical, "comment": row["html_url"]}
    return {"status": "blocked", "reason": "Awaiting disposition for this exact candidate.",
            "comment": confirmation["html_url"]}
