"""``hermes approvals suggest`` — mine approval history into allowlist proposals.

Hermes has no dedicated approval-decision ledger: ``always`` answers land in ``command_allowlist``
(config.yaml) via :func:`tools.approval.save_permanent_allowlist`, while ``once``/``session``
approvals are in-memory only. So this module mines *implied approvals*: a command that matches a
dangerous-command class (the same :func:`tools.approval.detect_dangerous_command` classifier that
triggers the prompt) AND whose tool result is not a block/denial marker must have been approved by
the user (once, session, always, smart-approve, or yolo) before it ran.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional


# ---------------------------------------------------------------------------
# Safety exclusions
# ---------------------------------------------------------------------------

# Dangerous-class descriptions matching ANY of these are never proposed, regardless of approval
# frequency (matched case-insensitively against tools.approval's DANGEROUS_PATTERNS / execution-flag
# descriptions). Deliberately conservative: a benign class accidentally excluded costs the user
# one manual config edit; a destructive class accidentally proposed costs them data.
_UNSAFE_CLASS_PATTERNS = [
    # deletion / destruction: recursive delete, find -delete, xargs rm, git reset --hard, mkfs, dd
    r"delete", r"\brm\b", r"destro", r"wipe", r"format", r"\bdisk\b", r"block device", r"\bdd\b",
    # process / host: fork bomb, force/regex/all kills, self-termination, shutdown
    r"fork bomb", r"kill (?:all )?process", r"kill all", r"self-termination", r"shutdown", r"reboot",
    # privilege / credentials / system files
    r"\bsudo\b", r"privilege", r"credential", r"\bssh\b", r"shell.rc", r"system config", r"system file",
    # data & permissions: SQL DROP/TRUNCATE/DELETE-without-WHERE, chown/chmod, world-writable, overwrite
    r"\bsql\b", r"\bchown\b", r"\bchmod\b", r"writable", r"overwrite", r"in-place edit",
    # remote/obfuscated execution: pipe-to-shell, encoded PowerShell, heredoc, substitutions
    r"pipe", r"obfuscation", r"remote content", r"remote script", r"heredoc", r"encoded",
    r"command substitution", r"process substitution",
    r"hardline",
]
_UNSAFE_CLASS_RE = re.compile("|".join(_UNSAFE_CLASS_PATTERNS), re.IGNORECASE)

# Root binaries that must never anchor a proposed command glob, even if the class survived the
# description filter. Prefix match for the mkfs family.
_UNSAFE_ROOT_BINARIES = {
    "rm", "rmdir", "unlink", "shred", "dd", "fdisk", "parted", "wipefs",
    "sudo", "doas", "su", "chmod", "chown", "chgrp",
    "kill", "killall", "pkill",
    "halt", "shutdown", "reboot", "poweroff", "init",
    "del", "format", "truncate", "mkswap",
}
_UNSAFE_ROOT_PREFIXES = ("mkfs",)

# Substrings in a role='tool' result that mean the command did NOT execute with user consent
# (blocked, denied, timed out, or still pending). Kept in sync with tools/approval.py templates.
_BLOCK_MARKERS = (
    "BLOCKED (hardline)", "BLOCKED: User denied", "BLOCKED: Action ",
    "BLOCKED: Command flagged as dangerous", "BLOCKED: approval required",
    "BLOCKED: Failed to send approval request", "The user has NOT consented",
    "Asking the user for approval", "approval_required", "BLOCKED by user deny rule",
)


@dataclass
class Proposal:
    """One ranked allowlist proposal."""

    pattern: str            # command glob ("git push *") or class key
    kind: str               # "glob" | "class"
    count: int = 0
    classes: set = field(default_factory=set)
    examples: list = field(default_factory=list)

    def add_example(self, command: str) -> None:
        short = command.strip()
        if len(short) > 100:
            short = short[:97] + "..."
        if short not in self.examples and len(self.examples) < 3:
            self.examples.append(short)


# ---------------------------------------------------------------------------
# Scan: session DB -> (command, class description) records
# ---------------------------------------------------------------------------

def default_db_path() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "state.db"


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def _fetch_rows(cur) -> Iterator[tuple]:
    """Stream cursor rows in 2000-row batches."""
    while rows := cur.fetchmany(2000):
        yield from rows


def _json_or_none(raw):
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def _iter_terminal_calls(con: sqlite3.Connection, since_ts: float) -> Iterator[tuple[str, str]]:
    """Yield ``(tool_call_id, command)`` for every terminal tool call."""
    cur = con.execute(
        "SELECT tool_calls FROM messages WHERE role='assistant' AND tool_calls IS NOT NULL "
        "AND tool_calls LIKE '%terminal%' AND timestamp >= ?",
        (since_ts,),
    )
    for (raw,) in _fetch_rows(cur):
        calls = _json_or_none(raw)
        for call in calls if isinstance(calls, list) else ():
            fn = call.get("function") or {} if isinstance(call, dict) else {}
            if fn.get("name") != "terminal":
                continue
            args = _json_or_none(fn.get("arguments") or "{}")
            if args is None:
                continue
            command = args.get("command")
            if isinstance(command, str) and command.strip():
                yield (call.get("id") or "", command)


def _blocked_tool_call_ids(con: sqlite3.Connection, since_ts: float) -> set:
    """Collect tool_call_ids whose result shows the command never ran freely."""
    cur = con.execute(
        "SELECT tool_call_id, content FROM messages "
        "WHERE role='tool' AND tool_call_id IS NOT NULL AND timestamp >= ? "
        "AND (content LIKE '%BLOCKED%' OR content LIKE '%approval%')",
        (since_ts,),
    )
    return {
        tool_call_id
        for tool_call_id, content in _fetch_rows(cur)
        if content and any(marker in content for marker in _BLOCK_MARKERS)
    }


def scan_approval_history(db_path: Optional[Path] = None, days: int = 90) -> list[tuple[str, str]]:
    """``(command, dangerous_class_description)`` records for dangerous-classified terminal commands
    that actually executed (i.e. carried an implied user approval).
    """
    from tools.approval_detection import detect_dangerous_command, detect_hardline_command
    path = Path(db_path) if db_path else default_db_path()
    if not path.exists():
        return []

    since_ts = 0.0 if days <= 0 else time.time() - days * 86400

    records: list[tuple[str, str]] = []
    con = _connect_readonly(path)
    try:
        blocked = _blocked_tool_call_ids(con, since_ts)
        for tool_call_id, command in _iter_terminal_calls(con, since_ts):
            if tool_call_id in blocked:
                continue
            # Hardline commands are unconditionally blocked at runtime; never mine them (defense in
            # depth against stale DB rows).
            if detect_hardline_command(command)[0]:
                continue
            is_dangerous, _key, description = detect_dangerous_command(command)
            if is_dangerous:
                records.append((command, description))
    finally:
        con.close()
    return records


# ---------------------------------------------------------------------------
# Normalize -> aggregate -> rank -> exclude
# ---------------------------------------------------------------------------

def normalize_command(command: str) -> str:
    """Fold user/hermes home prefixes and collapse whitespace."""
    from tools.approval_detection import _rewrite_resolved_hermes_home, _rewrite_resolved_user_home
    return " ".join(_rewrite_resolved_user_home(_rewrite_resolved_hermes_home(command)).split())


def is_unsafe_class(description: str) -> bool:
    """True when a dangerous-class description must never be proposed."""
    return bool(_UNSAFE_CLASS_RE.search(description or ""))


def _unsafe_root_binary(token: str) -> bool:
    tok = token.lower().rsplit("/", 1)[-1]
    return tok in _UNSAFE_ROOT_BINARIES or tok.startswith(_UNSAFE_ROOT_PREFIXES)


def derive_glob(normalized: str) -> Optional[str]:
    """Derive a narrow command glob (``git push *``) from a simple command.

    Returns None for compound commands (shell operators — the runtime allowlist matcher refuses
    those anyway) and for commands anchored on an unsafe root binary.
    """
    from tools.approval_floors import _has_allowlist_shell_operator
    tokens = normalized.split()
    if _has_allowlist_shell_operator(normalized) or not tokens or _unsafe_root_binary(tokens[0]):
        return None
    if len(tokens) == 1:
        return tokens[0]
    second = tokens[1]
    if second.startswith("-") or any(ch in second for ch in "*?[$"):
        return f"{tokens[0]} *"
    return f"{tokens[0]} {second} *"


def build_proposals(
    records: Iterable[tuple[str, str]], existing: Optional[set] = None, min_count: int = 2,
    limit: int = 20,
) -> list[Proposal]:
    """Aggregate scan records into a ranked, safety-filtered proposal list.

    Grain: a command glob (``git push *``) for simple commands; the dangerous-class description
    itself (the same key an interactive ``[a]lways`` answer persists) for compound commands where no
    safe glob can be derived.
    """
    existing = existing or set()
    by_pattern: dict[tuple[str, str], Proposal] = {}

    for command, description in records:
        if is_unsafe_class(description):
            continue
        normalized = normalize_command(command)
        glob = derive_glob(normalized)
        pattern, kind = (glob, "glob") if glob is not None else (description, "class")
        if pattern in existing:
            continue
        proposal = by_pattern.setdefault((pattern, kind), Proposal(pattern=pattern, kind=kind))
        proposal.count += 1
        proposal.classes.add(description)
        proposal.add_example(normalized)

    ranked = sorted((p for p in by_pattern.values() if p.count >= max(min_count, 1)), key=lambda p: (-p.count, p.pattern))
    return ranked[: max(limit, 1)]


# ---------------------------------------------------------------------------
# Apply / render
# ---------------------------------------------------------------------------

def parse_apply_indices(spec: str, total: int) -> list[int]:
    """Parse ``"1,3"`` into validated zero-based indices."""
    indices: list[int] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            n = int(part)
        except ValueError:
            raise ValueError(f"invalid selection {part!r} — expected numbers like 1,3")
        if n < 1 or n > total:
            raise ValueError(f"selection {n} out of range (1..{total})")
        if (n - 1) not in indices:
            indices.append(n - 1)
    if not indices:
        raise ValueError("no valid selections in --apply")
    return indices


def apply_proposals(proposals: list[Proposal], indices: list[int]) -> set:
    """Merge chosen proposal patterns into command_allowlist and persist."""
    import tools.approval as approval_module
    merged = set(approval_module.load_permanent_allowlist()) | {proposals[idx].pattern for idx in indices}
    approval_module.save_permanent_allowlist(merged)
    # Keep the in-process allowlist consistent so a long-lived process sees the new entries
    # immediately (mirrors the interactive 'always' path).
    approval_module.load_permanent(merged)
    return merged


def _render_text(proposals: list[Proposal], days: int) -> None:
    window = "all history" if days <= 0 else f"last {days} days"
    if not proposals:
        print(
            f"No allowlist candidates found in approval history ({window}).\n"
            "Either nothing dangerous was approved often enough "
            "(see --min-count/--days), or the approved classes are excluded for safety."
        )
        return
    print(f"Proposed command_allowlist additions (from approval history, {window}):\n")
    for i, p in enumerate(proposals, 1):
        kind = " (class key)" if p.kind == "class" else ""
        print(f"  {i}. {p.pattern}    — approved {p.count}x{kind}")
        for cls in sorted(p.classes):
            print(f"       class: {cls}")
        for ex in p.examples:
            print(f"       e.g. {ex}")
    print(
        "\nNothing has been changed. Apply selected entries with:\n"
        "  hermes approvals suggest --apply 1,3\n"
        "Entries are merged into command_allowlist in ~/.hermes/config.yaml."
    )


def suggest_command(args) -> int:
    """Entry point for ``hermes approvals suggest``."""
    db_path = Path(args.db) if getattr(args, "db", None) else default_db_path()
    days = getattr(args, "days", 90)
    if not db_path.exists():
        print(f"Session database not found: {db_path}")
        return 1

    import tools.approval as approval_module
    existing = set(approval_module.load_permanent_allowlist())
    proposals = build_proposals(
        scan_approval_history(db_path, days=days), existing=existing,
        min_count=getattr(args, "min_count", 2), limit=getattr(args, "limit", 20),
    )
    as_json = getattr(args, "json", False)

    apply_spec = getattr(args, "apply_indices", None)
    if apply_spec:
        try:
            indices = parse_apply_indices(apply_spec, len(proposals))
        except ValueError as exc:
            print(f"--apply error: {exc}")
            return 1
        merged = apply_proposals(proposals, indices)
        applied = [proposals[i].pattern for i in indices]
        if as_json:
            print(json.dumps({"applied": applied, "allowlist_size": len(merged)}))
        else:
            print("Added to command_allowlist:")
            for pattern in applied:
                print(f"  + {pattern}")
            print(f"\ncommand_allowlist now has {len(merged)} entries (~/.hermes/config.yaml).")
        return 0

    if as_json:
        payload = {
            "db": str(db_path),
            "days": days,
            "proposals": [
                {
                    "n": i, "pattern": p.pattern, "kind": p.kind, "count": p.count,
                    "classes": sorted(p.classes), "examples": p.examples,
                }
                for i, p in enumerate(proposals, 1)
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0

    _render_text(proposals, days)
    return 0


def approvals_command(args) -> int:
    """Dispatch ``hermes approvals <subcommand>``."""
    sub = getattr(args, "approvals_command", None)
    if sub == "suggest":
        return suggest_command(args)
    if sub == "test":
        from hermes_cli.approvals_test import approvals_test_command
        return approvals_test_command(args)
    print(
        "usage: hermes approvals <subcommand>\n"
        "\n"
        "subcommands:\n"
        "  suggest    Mine past approval decisions into a proposed\n"
        "             command_allowlist (dry by default; --apply N,M to merge)\n"
        "  test       Dry-run the approval verdict for a command without\n"
        "             executing it (exit 0 allow / 2 ask / 3 deny)\n"
        "\n"
        "Run `hermes approvals <subcommand> -h` for details."
    )
    return 1
