"""Opt-in rollover policy and immutable handoff admission; no model or route mutation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import unquote, urlsplit
import uuid


class RolloverBlocked(RuntimeError):
    """Continuity could not be established; the current owner must not be replaced."""


class RolloverPreflightBlocked(RolloverBlocked):
    """The bound agent was refused before its conversation began."""


@dataclass(frozen=True)
class RolloverPolicy:
    handoff_skill: str
    packet_dir: Path
    bootstrap_prompt: str
    runway_tokens: int
    safety_margin_tokens: int

    @classmethod
    def for_lane(cls, config: dict, session_key: str) -> RolloverPolicy | None:
        raw = config.get("session_rollover") or {}
        if not isinstance(raw, dict):
            raise RolloverBlocked("session_rollover must be a mapping")
        if raw.get("enabled") is not True:
            return None
        lanes = raw.get("session_keys")
        if not isinstance(lanes, list) or not lanes or not all(isinstance(k, str) and k for k in lanes):
            raise RolloverBlocked("session_rollover requires exact session_keys")
        if session_key not in lanes:
            return None
        skill = raw.get("handoff_skill")
        root = raw.get("packet_dir")
        prompt = raw.get("bootstrap_prompt")
        if not isinstance(skill, str) or not skill.strip():
            raise RolloverBlocked("session_rollover requires a discoverable handoff_skill")
        if not isinstance(root, str) or not Path(root).is_absolute():
            raise RolloverBlocked("session_rollover.packet_dir must be an absolute durable path")
        if not isinstance(prompt, str) or prompt.count("{packet}") != 1:
            raise RolloverBlocked("session_rollover.bootstrap_prompt must bind exactly one {packet}")
        amounts = [raw.get("runway_tokens"), raw.get("safety_margin_tokens")]
        if any(type(n) is not int or n <= 0 for n in amounts):
            raise RolloverBlocked("session_rollover requires measured positive runway_tokens and safety_margin_tokens")
        return cls(skill.strip(), Path(root).resolve(), prompt, *amounts)

    def new_packet_path(self, session_id: str) -> Path:
        now = datetime.now().astimezone()
        directory = self.packet_dir / now.strftime("%Y/%m/%d")
        directory.mkdir(parents=True, exist_ok=True)
        # Check the actual destination, not os.access (ACLs and read-only mounts differ).
        with tempfile.TemporaryFile(dir=directory) as probe:
            probe.write(b"rollover preflight")
            probe.flush()
            os.fsync(probe.fileno())
        identity = hashlib.sha256(session_id.encode()).hexdigest()[:12]
        return directory / f"{now:%Y-%m-%d-%H%M}-handoff-{identity}-{uuid.uuid4().hex[:8]}.md"


_DIRECTIVE_FIELDS = (
    "Standing mandate", "Current objective", "Current mode", "Next action and authority",
    "Pending obligations", "Completed work", "Superseded/stopped/cancelled instructions",
    "Do not repeat until reconciled",
)
_MODES = {"continue-authorized", "await-user", "monitor", "reconcile", "idle"}
_LINK = re.compile(r"(?<!!)\[[^\]\n]*\]\((<[^>]+>|[^)\s]+)(?:\s+\"[^\"]*\")?\)")


def validate_packet(packet: Path, restart_message: str) -> dict:
    """Read exactly the requested artifact and preserve the author's final message byte-for-byte."""
    if packet.is_symlink() or not packet.is_file():
        raise RolloverBlocked("The bound handoff packet is missing or is a symlink")
    data = packet.read_bytes()
    text = data.decode("utf-8")
    match = re.search(r"(?m)^## Continuation Directive\s*$", text)
    if match is None:
        raise RolloverBlocked("The handoff has no Continuation Directive")
    directive = re.split(r"(?m)^## ", text[match.end():], maxsplit=1)[0]
    fields = {}
    for name in _DIRECTIVE_FIELDS:
        field = re.search(rf"(?m)^- \*\*{re.escape(name)}:\*\*\s*(.+)$", directive)
        if field is None or not field.group(1).strip() or re.fullmatch(r"\[[^\]\n]+\]", field.group(1).strip()):
            raise RolloverBlocked(f"The handoff directive has no completed {name}")
        fields[name] = field.group(1).strip()
    mode = fields["Current mode"].strip("` ").split()[0].rstrip("`;:,.—")
    if mode not in _MODES:
        raise RolloverBlocked("The handoff directive has no unambiguous continuation mode")
    if not isinstance(restart_message, str) or str(packet) not in restart_message:
        raise RolloverBlocked("The actual restart message does not reference the bound packet")
    if not re.search(r"(?m)^# Handoff:", restart_message):
        raise RolloverBlocked("The old session did not finish its paste-ready restart message")
    # Local references must exist before publication. Remote references remain authorities
    # the successor must re-read; this is not a second web crawler or authority oracle.
    for reference in _LINK.findall(text):
        target = unquote(reference.strip("<>"))
        parsed = urlsplit(target)
        if parsed.scheme or not parsed.path:
            continue
        resolved = (packet.parent / parsed.path).resolve()
        if not resolved.exists():
            raise RolloverBlocked(f"Unresolved handoff reference: {target}")
    return {"packet_path": str(packet), "packet_sha256": hashlib.sha256(data).hexdigest(),
            "restart_message": restart_message, "mode": mode}


def verify_packet_binding(receipt: dict) -> None:
    packet = Path(receipt["packet_path"])
    if packet.is_symlink() or hashlib.sha256(packet.read_bytes()).hexdigest() != receipt["packet_sha256"]:
        raise RolloverBlocked("The committed handoff packet changed; reconcile before continuing")


def handoff_instruction(policy: RolloverPolicy, packet: Path, session_id: str) -> str:
    from agent.skill_commands import _build_skill_message, _load_skill_payload

    loaded = _load_skill_payload(policy.handoff_skill, task_id=session_id)
    if loaded is None:
        raise RolloverBlocked(f"Handoff skill is unavailable: {policy.handoff_skill}")
    payload, directory, name = loaded
    skill = _build_skill_message(
        payload, directory,
        f'[Internal continuity context: the configured "{name}" procedure is loaded. '
        'This is not a user command or new approval.]', session_id=session_id,
    )
    return (
        skill + "\n\n[Internal continuity instruction — no new authority]\n"
        "Stop starting ordinary work. Execute the complete handoff procedure with your existing tools, "
        "including its supporting templates and every required packet section. Reconcile the latest "
        "user direction, workers, processes, approvals and uncertain outcomes. Do not launch workers "
        "or take operational actions while closing out. Missing authority must be recorded as a block, "
        "not invented. Keep independent infrastructure running.\n"
        f"Write the complete packet to exactly {packet}. Use absolute paths for load-bearing local "
        "references and the packet reference in your final paste-ready message. Fill every Continuation "
        "Directive field, including explicit none where appropriate. Return the actual short paste-ready "
        "message unchanged in your final response; do not create a competing recap. Do not reset or "
        "create a session yourself. The gateway will validate this artifact before any identity switch."
    )
