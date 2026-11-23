"""``hermes approvals suggest`` — mine approval history into allowlist proposals.

Hermes has no dedicated approval-decision ledger: ``always`` answers land in ``command_allowlist``
(config.yaml) via :func:`tools.approval.save_permanent_allowlist`, while ``once``/``session``
approvals are in-memory only. So this module mines the approvals a person gave: a command that matches a
dangerous-command class (the same :func:`tools.approval.detect_dangerous_command` classifier that
triggers the prompt) whose tool result carries the runtime's "approved by the user" note. Runs that
smart approval, an allowlist entry, a remembered grant, yolo or ``approvals.mode: off`` let through
are not anyone's decision to repeat, so they are not counted.

Only the classes in :data:`_PROPOSABLE_CLASSES` are ever proposed: a class is out until someone
decides it is routine enough to approve forever (#158903).
"""

from __future__ import annotations

import fnmatch
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional


# ---------------------------------------------------------------------------
# Safety exclusions
# ---------------------------------------------------------------------------

# The only dangerous classes ever proposed, each with the exact shape its command glob must have.
# Nothing else is proposed: not a glob of any other shape (``timeout 30 *``, ``env docker *``,
# ``docker container *``, ``A=/docker kill *``), not a class key (the runtime matches a class key against
# the command's first finding only, so it also waves through whatever is chained after it), and no
# other class (force push, reverse shells, restarts that kill running agents, approval-setting changes,
# uninstalls, daemon redirects, inline interpreter or shell code, ``compose down -v``, a class nobody
# reviewed yet): one manual config edit is cheaper than a remembered mistake (#158903).
_PROPOSABLE_CLASSES: dict[str, re.Pattern] = {
    "docker restart/stop/kill (container lifecycle)": re.compile(
        r"(?:/(?:[\w.+-]+/)*)?docker (?:restart|stop|kill) \*"),
}

# Defence in depth over the list above: descriptions matching ANY of these are never proposed even if
# someone adds them to it (matched case-insensitively).
_UNSAFE_CLASS_PATTERNS = [
    # deletion / destruction: recursive delete, find -delete, xargs rm, git reset --hard, mkfs, dd
    r"delete", r"\brm\b", r"destro", r"wipe", r"format", r"\bdisk\b", r"block device", r"\bdd\b",
    # process / host: fork bomb, force/regex/all kills, self-termination, shutdown
    r"fork bomb", r"kill (?:all )?process", r"kill all", r"self-termination", r"shutdown", r"reboot",
    # privilege / credentials / system files
    r"\bsudo\b", r"privilege", r"credential", r"\bssh\b", r"shell.rc", r"system config", r"system file",
    # data & permissions: SQL DROP/TRUNCATE/DELETE-without-WHERE, chown/chmod, world-writable, overwrite
    r"\bsql\b", r"\bchown\b", r"\bchmod\b", r"writable", r"overwrite", r"in-place edit",
    # Windows registry writes outlive the session that approved them (reg add, setx, Set-ItemProperty)
    r"registry",
    # remote/obfuscated execution: pipe-to-shell, encoded PowerShell, heredoc, substitutions
    r"pipe", r"obfuscation", r"remote content", r"remote script", r"heredoc", r"encoded",
    r"command substitution", r"process substitution",
    # a command the scanners could not read to the end: approved one run at a time, never remembered
    r"hardline", r"too complex to scan",
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

# The note tools/terminal_tool.py records under the result's "approval" key on a command a person
# approved at the prompt.
_USER_APPROVED_NOTE = re.compile(
    r"Command required approval \(.*\) and was approved by the user(?:, then interrupted)?\.", re.DOTALL)


@dataclass
class Proposal:
    """One ranked allowlist proposal."""

    pattern: str            # command glob ("docker restart *")
    kind: str               # always "glob"; kept in the JSON output for its consumers
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

def _store_target(db_path: Optional[Path] = None) -> Optional[Path]:
    """What a scan reads: the ``db_path`` file when one is given, else the current profile's folder, which names
    its store; None when that store does not exist."""
    if db_path:
        return Path(db_path) if Path(db_path).exists() else None
    from hermes_constants import get_hermes_home
    from persistence_assembly import profile_scope, store_exists
    scope = profile_scope(home=get_hermes_home())
    return scope.profile_dir if store_exists(scope) else None


def _shown_store(db_path: Optional[Path] = None) -> Path:
    """The store file the output names: the ``db_path`` given, else the file that holds the current profile's store."""
    if db_path:
        return Path(db_path)
    from hermes_constants import get_hermes_home
    from persistence_assembly import profile_scope, store_location
    scope = profile_scope(home=get_hermes_home())
    return store_location(scope) or scope.profile_dir


def _json_or_none(raw):
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def _iter_terminal_calls(messages, since_ts: float) -> Iterator[tuple[str, str]]:
    """Yield ``(tool_call_id, command)`` for every terminal tool call."""
    for raw in messages.terminal_tool_call_payloads(since_ts=since_ts):
        calls = _json_or_none(raw)
        for call in calls if isinstance(calls, list) else ():
            fn = call.get("function") or {} if isinstance(call, dict) else {}
            if fn.get("name") != "terminal":
                continue
            args = _json_or_none(fn.get("arguments") or "{}")
            if not isinstance(args, dict):  # a rejected call can carry `[]` or a bare string
                continue
            command = args.get("command")
            if isinstance(command, str) and command.strip():
                yield (call.get("id") or "", command)


def _user_approved_tool_call_ids(messages, since_ts: float) -> set:
    """tool_call_ids whose result says a person approved the command at the prompt. Only the result's own
    ``approval`` note counts: the command's output, which the agent controls, may print the same words."""
    approved = set()
    for tool_call_id, content in messages.blocked_tool_result_payloads(since_ts=since_ts):
        try:  # the tool's own JSON: subdirectory hints, LSP verdicts and plugin notices are appended after it
            payload = json.JSONDecoder().raw_decode(content)[0]
        except (TypeError, ValueError):
            continue
        note = payload.get("approval") if isinstance(payload, dict) else None
        if isinstance(note, str) and _USER_APPROVED_NOTE.fullmatch(note):
            approved.add(tool_call_id)
    return approved


def scan_approval_history(db_path: Optional[Path] = None, days: int = 90) -> list[tuple[str, str]]:
    """``(command, dangerous_class_description)`` records for dangerous-classified terminal commands
    a person approved at the prompt and that then ran.
    """
    from persistence_assembly import open_persistence_reader
    from tools.approval_detection import detect_dangerous_command, detect_hardline_command, hardline_scan_incomplete
    path = _store_target(db_path)
    if path is None:
        return []

    since_ts = 0.0 if days <= 0 else time.time() - days * 86400

    records: list[tuple[str, str]] = []
    with open_persistence_reader(path) as store:
        approved = _user_approved_tool_call_ids(store.messages, since_ts)
        # A compaction keeps the archived row next to its carried copy: the same call, stored twice.
        calls = list(dict.fromkeys(_iter_terminal_calls(store.messages, since_ts)))
        # Call ids are unique within a session only (a provider that sends none gets ids derived from the
        # call, others number them from call_0), and nothing here ties a result to its session, so an id
        # two different commands share could credit one session's approval to another's command. Ids are
        # counted over all history: an approval wait can carry a result past the cutoff, not its call.
        history = calls if since_ts <= 0 else dict.fromkeys(_iter_terminal_calls(store.messages, 0.0))
        id_uses = Counter(tool_call_id for tool_call_id, _command in history)
        for tool_call_id, command in calls:
            if tool_call_id not in approved or id_uses[tool_call_id] > 1:
                continue
            # Hardline commands are unconditionally blocked at runtime; never mine them (defense in
            # depth against stale DB rows). Nor commands the hardline scan could not read to the end:
            # their approval is one run at a time, so nothing about them was ever meant to persist.
            if detect_hardline_command(command)[0] or hardline_scan_incomplete(command):
                continue
            is_dangerous, _key, description = detect_dangerous_command(command)
            if is_dangerous:
                records.append((command, description))
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
    return description not in _PROPOSABLE_CLASSES or bool(_UNSAFE_CLASS_RE.search(description or ""))


def _unsafe_root_binary(token: str) -> bool:
    tok = token.lower().rsplit("/", 1)[-1]
    return tok in _UNSAFE_ROOT_BINARIES or tok.startswith(_UNSAFE_ROOT_PREFIXES)


def derive_glob(normalized: str) -> Optional[str]:
    """Derive a narrow command glob (``docker restart *``) from a simple command.

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


def _glob_matches_source(glob: str, command: str) -> bool:
    """True when the runtime allowlist check would let *glob* approve *command* as written.
    The glob comes from the home-folded command (``~/bin/docker restart *``), which the source
    ``/home/me/bin/docker restart web`` never matches; nor does a multi-line source, which the
    runtime refuses before comparing (``derive_glob`` only saw it with newlines folded)."""
    from tools.approval_floors import _has_allowlist_shell_operator

    command = command.strip()
    if _has_allowlist_shell_operator(command):
        return False
    return command == glob or fnmatch.fnmatchcase(command, glob)


def build_proposals(
    records: Iterable[tuple[str, str]], existing: Optional[set] = None, min_count: int = 2,
    limit: int = 20,
) -> list[Proposal]:
    """Aggregate scan records into a ranked, safety-filtered proposal list.

    Grain: a command glob of the exact shape its class allows (``docker restart *``); a command that
    yields no such glob is not proposed.
    """
    from agent.redact import redact_sensitive_text

    existing = existing or set()
    by_pattern: dict[str, Proposal] = {}

    for command, description in records:
        if is_unsafe_class(description):
            continue
        # Commands mined from past tool calls can embed credentials (URL userinfo,
        # env assignments, bearer tokens). Examples are echoed to the operator, so
        # mask them like every other display boundary — classification above
        # still sees the raw command. The glob is derived from the raw command:
        # a redacted `***` inside a persisted pattern would be three fnmatch
        # wildcards, so when redaction touches the tokens the glob embeds, the
        # command is not proposed.
        raw = normalize_command(command)
        normalized = redact_sensitive_text(raw, force=True)
        glob = derive_glob(raw)
        if glob is None or not _PROPOSABLE_CLASSES[description].fullmatch(glob):
            continue  # `docker *`, `timeout 30 *`, a compound command: wider than the class
        if not _glob_matches_source(glob, command) or derive_glob(normalized) != glob:
            continue  # the runtime matches the command as written; a redacted token is three wildcards
        if glob in existing:
            continue
        proposal = by_pattern.setdefault(glob, Proposal(pattern=glob, kind="glob"))
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
        print(f"  {i}. {p.pattern}    — approved {p.count}x")
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
    db_path = _store_target(getattr(args, "db", None))
    shown = _shown_store(getattr(args, "db", None))
    days = getattr(args, "days", 90)
    if db_path is None:
        print(f"Session database not found: {shown}")
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
            "db": str(shown),
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
