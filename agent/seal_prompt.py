"""Prompt builders for the /seal and /unseal session-handoff commands.

The gateway stays thin: these builders inject a self-contained protocol prompt and
fall through to the agent (same pattern as /learn and /plan). All reasoning —
what was settled, which skill owns it — is the agent's job. The handoff file is a
small pointer record (skill name + SSOT path + one open question), never a recap:
pasting a recap back into the chat would re-inflate the session that /seal exists
to empty.

The protocol is deliberately skill-agnostic. If the deployment has a session-seal
skill, the agent loads it; the prompt alone is enough for the bridge to work.
"""

from __future__ import annotations

import os
from pathlib import Path

_HANDOFF_FIELDS = (
    "skill: <lowercase skill name that owns this subject, or none>",
    "ssot: <path to the topic's single-source-of-truth file, or none>",
    "open: <one sentence describing the single open question, or none>",
    "sealed_session: <session id>",
    "consumed: no",
)


def handoff_dir() -> Path:
    """Where handoff files live. Config-overridable (gateway.seal.handoff_dir),
    default ``$HERMES_HOME/handoffs``."""
    try:
        from hermes_cli.config import load_config  # lazy: import-light for gateway boot

        cfg = load_config() or {}
        override = (
            cfg.get("gateway", {}).get("seal", {}).get("handoff_dir")
            if isinstance(cfg.get("gateway"), dict)
            else None
        )
        if isinstance(override, str) and override.strip():
            return Path(override).expanduser()
    except Exception:
        pass
    home = os.environ.get("HERMES_HOME", "").strip()
    base = Path(home).expanduser() if home else Path.home() / ".hermes"
    return base / "handoffs"


def handoff_path(chat_key: str) -> Path:
    """Handoff file for a chat. Keyed by the *chat*, not the session id, so the
    bridge survives the ``/new`` that follows a seal."""
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in (chat_key or "cli"))
    return handoff_dir() / f"{safe or 'cli'}.md"


def build_seal_prompt(args: str, *, session_id: str = "", chat_key: str = "") -> str:
    """Seal this conversation into a handoff pointer, then stop.

    ``chat_key`` must be stable across ``/new`` (platform + chat id, not the
    session id) — that is what makes the bridge survive the session reset.
    """
    path = handoff_path(chat_key)
    fields = "\n".join(_HANDOFF_FIELDS)
    extra = f"\n\nOperator note (may steer the open question): {args}" if args.strip() else ""
    return (
        "Run the session-seal protocol for this conversation.\n\n"
        "1. If a session-seal skill exists, load it and follow it; otherwise follow the "
        "steps below verbatim.\n"
        "2. Name the ONE existing skill that owns this subject's settled rules. Do not "
        "create a skill per chat and do not invent a home — if none owns it, say so. "
        "Settle any rules this conversation actually reached into that skill "
        "(imperative rule + why; no dates, no incident narration). If nothing was "
        "settled, write nothing there. Subject *data* (prices, specs, dates, verdicts) "
        "belongs in the topic's SSOT file, patched and dated — not in the skill, not in "
        "memory. Skill and SSOT must state the same truth; reconcile any drift now.\n"
        f"3. Write the handoff pointer file at `{path}` (create the directory if "
        "needed) with exactly these lines:\n\n"
        f"{fields}\n\n"
        f"Use `sealed_session: {session_id or '<session id>'}`.\n"
        "4. Reply with ONLY:\n\n"
        "   SEALED <skill>\n"
        "   /new\n"
        "   then say: unseal\n\n"
        "5. Stop. Do not keep working the problem, do not search old sessions, and do "
        "not summarize the thread in your reply — the handoff carries pointers only."
        + extra
    )


def build_unseal_prompt(args: str, *, session_id: str = "", chat_key: str = "") -> str:
    """Re-open the handoff written by /seal before the last /new."""
    path = handoff_path(chat_key)
    extra = f"\n\nOperator note: {args}" if args.strip() else ""
    return (
        "Resume from the session handoff.\n\n"
        f"1. Read `{path}`. If it is missing, say so and stop — do not guess which "
        "skill from memory of another window.\n"
        "2. Load the skill named in `skill:` and, if `ssot:` names a path, read that "
        "file (or the section the open question points at). Load both — the SSOT is "
        "authoritative for facts, the skill is method. Never load only one.\n"
        "3. State the open question in one line, then work that question only. If "
        "`open: none`, report the loaded state briefly and wait for direction.\n"
        "4. Do not search or reconstruct the sealed session unless skill AND SSOT both "
        "lack a fact the open question requires. A search hit is a slice, never 'load "
        "the old session'.\n"
        f"5. Set `consumed: yes` in `{path}` once the skill has loaded. A second unseal "
        "in the same chat still loads the skill but reports the handoff was already "
        "used — it does not re-open the old thread."
        + extra
    )
