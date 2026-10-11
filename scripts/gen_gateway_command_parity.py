#!/usr/bin/env -S bash -c 'exec "$BASH" "$(dirname "$0")/run-in-hermes-env" python3 "$0" "$@"'
"""Regenerate website/docs/developer-guide/gateway-command-parity.md from the code.

For every ``COMMAND_REGISTRY`` command it records what each local client does with it on the
shared gateway -- the classic CLI view (``hermes_cli/gateway_chat_commands.py``), Ink
(``ui-tui/src/app/slash/commands/*.ts``), Desktop (``apps/desktop/src/lib/desktop-slash-commands.ts``
+ the registry's ``desktop=`` dump), ACP (``acp_adapter`` -> ``slash_mutation``) -- and the
gateway's own ``slash.exec`` verdict for a local session (``gateway/session_commands.py``).
Every column is derived from source, so a verdict that changes without a regenerated table
fails ``tests/hermes_cli/test_gateway_command_parity.py``. ``--check`` writes nothing and exits 1
on drift.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "website" / "docs" / "developer-guide" / "gateway-command-parity.md"

# Per-session settings that write config.yaml / messaging-only state through their gateway handler:
# enabling them as-is would print success and change nothing for a local session.
TOGGLES = frozenset({"fast", "reasoning", "yolo", "footer", "personality", "busy", "voice", "approvals",
                     "verbose", "codex-runtime"})
# Why a command with a gateway handler is NOT a read for a local session (probe: the real handler
# run through slash.exec against a local session, tests/gateway/test_session_read_commands.py).
NOT_A_LOCAL_READ = {
    "agents": "gateway-wide: lists every chat's running agents and the process's async jobs",
    "platform": "messaging adapters; `pause`/`resume` mutate them",
    "sessions": "messaging origin-scoped listing (`/sessions all` needs a messaging admin)",
    "debug": "uploads logs to a public paste",
    "curator": "no gateway handler (`_handle_curator_command` is missing)",
    "save": "writes an export file on the gateway host",
    "rollback": "restores filesystem checkpoints",
    "resume": "switches the messaging chat's session; local clients resume by id",
    "skills": "write-approval queue; `/skills approve|approval` write",
}
# The read-only argument forms of a read whose other subcommands write.
READ_FORMS = {"usage": "bare only (`reset` refused)", "memory": "bare / `pending`",
              "suggestions": "bare only", "kanban": "list/show/stats/assignees/context/runs/log/diagnostics"}


def _served_methods():
    from unittest.mock import MagicMock
    from gateway.session_controls import AuthorityConnection
    from gateway.session_group_controls import GROUP_METHODS
    return set(AuthorityConnection.handlers(MagicMock())) | set(GROUP_METHODS) | {"profiles.list"}


def gateway_cell(name):
    from gateway.session_commands import command_verdict
    verdict = command_verdict(name)
    if verdict == "read":
        return "read" + (f" ({READ_FORMS[name]})" if name in READ_FORMS else "")
    if verdict == "refused" and name in READ_FORMS:
        return f"read ({READ_FORMS[name]})"
    if verdict == "control":
        return "control"
    return "refused"


def port_note(cmd):
    if cmd.name in TOGGLES:
        return "port: per-session setting"
    if cmd.name in NOT_A_LOCAL_READ:
        return NOT_A_LOCAL_READ[cmd.name]
    if cmd.gateway_only:
        return "messaging-only"
    if cmd.cli_only:
        return "client-local (cli_only)"
    return ""


def classic_cell(name, gateway):
    from hermes_cli.gateway_chat_commands import VIEW_ROUTES
    if name in VIEW_ROUTES:
        return VIEW_ROUTES[name]
    return "slash.exec" if gateway != "refused" else "refused: not available yet"


def acp_cell(name, gateway):
    from hermes_cli.gateway_client import GatewayClientError
    from hermes_cli.gateway_mutations import slash_mutation
    try:
        operation, _ = slash_mutation("/" + name, "x" if name == "model" else "")
    except GatewayClientError:
        return "refused"
    return "ACP fork_session" if operation == "branch" else f"session.mutate {operation}"


def _ink_commands():
    """``{name_or_alias: (canonical_ink_name, block_text)}`` from the static Ink tables."""
    table = {}
    for path in sorted((ROOT / "ui-tui" / "src" / "app" / "slash" / "commands").glob("*.ts")):
        text = path.read_text(encoding="utf-8-sig")
        for block in re.split(r"\n  \{\n", text)[1:]:
            block = block.split("\n  },", 1)[0]
            match = re.search(r"^    name: '([\w-]+)'", block, re.MULTILINE)
            if not match:
                continue
            aliases = re.search(r"^    aliases: \[([^\]]*)\]", block, re.MULTILINE)
            names = [match.group(1)] + (re.findall(r"'([\w-]+)'", aliases.group(1)) if aliases else [])
            for name in names:
                table.setdefault(name, (match.group(1), block))
    return table


def ink_cell(cmd, gateway, ink, served):
    entry = next((ink[n] for n in (cmd.name, *cmd.aliases) if n in ink), None)
    if entry is None:
        return "slash.exec" if gateway != "refused" else "slash.exec: refused"
    _, block = entry
    methods = set(re.findall(r"(?:rpc|request)(?:<[^>]*>)?\(\s*'([a-z_]+\.[a-z_.]+)'", block))
    if "canonical.controls.notAvailable" in block:
        # The shared gateway refuses it in the client (no owner verb yet), whatever it calls on a
        # standalone backend; never reported as a working route. A bare form that still reads the
        # current value through a served ``.get`` is named.
        reads = sorted(m for m in methods & served if m.endswith(".get"))
        return (", ".join(reads) + "; change refused: not available yet") if reads else "refused: not available yet"
    if "isCanonical" in block:
        return "canonical route"
    return _rpc_cell(methods, served) if methods else "local"


def _sidecar_live_session_methods():
    """Sidecar RPCs that resolve their ``session_id`` in the sidecar's own session map
    (``@_rpc(..., live_session=True)``), which never holds a shared-gateway session."""
    text = "".join(p.read_text(encoding="utf-8-sig") for p in sorted((ROOT / "tui_gateway").glob("*.py")))
    return set(re.findall(r'_rpc\(\s*"([\w.]+)"[^)]*\blive_session=True', text))


def _rpc_cell(methods, served):
    """An RPC the authority does not serve: session-namespace verbs answer -32601 (the client's
    "out of sync" error); any other method reaches the legacy sidecar (``tui_gateway/ws.py``),
    where a live-session handler answers 4001 for every shared-gateway session."""
    from tui_gateway.ws_legacy_fallback import _SESSION_FALLBACK_ALLOWED, _SESSION_NAMESPACES
    broken = sorted(m for m in methods - served
                    if m.startswith(_SESSION_NAMESPACES) and m not in _SESSION_FALLBACK_ALLOWED)
    if broken:
        return "**broken**: " + ", ".join(broken) + " (-32601)"
    unbound = sorted((methods - served) & _sidecar_live_session_methods())
    if unbound:
        return "**broken**: " + ", ".join(unbound) + " (4001 session not found)"
    legacy = sorted(methods - served)
    return ", ".join(sorted(methods & served)) + ("; " if methods & served and legacy else "") + (
        "sidecar: " + ", ".join(legacy) if legacy else "")


def _desktop_specs():
    text = (ROOT / "apps" / "desktop" / "src" / "lib" / "desktop-slash-commands.ts").read_text(encoding="utf-8-sig")
    body = text.split("const DESKTOP_COMMAND_SPECS", 1)[1].split("\n]\n", 1)[0]
    specs = {}
    for chunk in re.split(r"\n  \{", body)[1:]:
        name = re.search(r"name: '(/[\w-]+)'", chunk)
        surface = re.search(r"surface: (action|picker|rpc|exec|unavailable)\((?:\s*'([\w.\-]+)')?", chunk)
        if not (name and surface):
            continue
        aliases = re.search(r"aliases: \[([^\]]*)\]", chunk)
        kind, detail = surface.groups()
        for key in [name.group(1)] + (re.findall(r"'(/[\w-]+)'", aliases.group(1)) if aliases else []):
            specs.setdefault(key, (kind, detail or ""))
    return specs


def desktop_cell(cmd, gateway, specs, served):
    for key in (cmd.name, *cmd.aliases):
        spec = specs.get("/" + key)
        if spec is None:
            continue
        kind, detail = spec
        if kind == "rpc":
            # runRpc falls back to slash.exec on -32601 (use-prompt-actions/slash.ts).
            if detail in served:
                return f"rpc {detail}"
            return "slash.exec (rpc fallback)" if gateway != "refused" else "slash.exec: refused"
        if kind == "exec":
            return "slash.exec" if gateway != "refused" else "slash.exec: refused"
        return f"{kind}: {detail}" if detail else kind
    if cmd.desktop and cmd.desktop != "hidden":
        return f"unavailable: {cmd.desktop}"
    return "slash.exec" if gateway != "refused" else "slash.exec: refused"


HEADER = """---
title: Gateway command parity
description: What each local client does with every slash command on the shared gateway
---

<!-- GENERATED by scripts/gen_gateway_command_parity.py — DO NOT EDIT. -->
<!-- tests/hermes_cli/test_gateway_command_parity.py fails when this table is stale. -->

# Gateway command parity (local sessions)

Every local client (classic CLI, Ink TUI, Desktop, ACP) attaches to the one gateway that owns
the session. A slash command either runs in the client against a canonical verb
(`session.mutate`, `session.create`, `prompt.receipt`, ...), or goes to the owner as
`slash.exec`, where `gateway/session_commands.py` runs only the reviewed reads below and refuses
the rest (`unsupported_command`). The classic CLI prints a refusal as
"/x is not available on the shared gateway yet" with a link to this page.

**Gateway (`slash.exec`)**: `read` = the real handler runs for a local session (proved by
`tests/gateway/test_session_read_commands.py`); `control` = admission-fenced write; `refused`.
**Notes**: `port: per-session setting` toggles write config.yaml or messaging-only state
through their gateway handler, so they stay refused until ported to the session's frozen
policy; `client-local (cli_only)` commands are terminal/UI features of a client.

"""


def render() -> str:
    sys.path.insert(0, str(ROOT))
    from hermes_cli.commands import COMMAND_REGISTRY
    served = _served_methods()
    ink, specs = _ink_commands(), _desktop_specs()
    rows = ["| Command | Gateway (`slash.exec`) | Classic CLI | Ink | Desktop | ACP | Notes |",
            "|---|---|---|---|---|---|---|"]
    counts = {}
    for cmd in sorted(COMMAND_REGISTRY, key=lambda c: c.name):
        gateway = gateway_cell(cmd.name)
        counts[gateway.split(" ")[0]] = counts.get(gateway.split(" ")[0], 0) + 1
        aliases = f" ({', '.join('/' + a for a in cmd.aliases)})" if cmd.aliases else ""
        cells = [f"`/{cmd.name}`{aliases}", gateway, classic_cell(cmd.name, gateway),
                 ink_cell(cmd, gateway, ink, served), desktop_cell(cmd, gateway, specs, served),
                 acp_cell(cmd.name, gateway), port_note(cmd)]
        rows.append("| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |")
    summary = ", ".join(f"{count} {kind}" for kind, count in sorted(counts.items()))
    return HEADER + f"{len(COMMAND_REGISTRY)} registry commands: {summary}.\n\n" + "\n".join(rows) + "\n"


def check(out: Path = OUT) -> int:
    committed = out.read_text(encoding="utf-8-sig") if out.exists() else ""
    if committed == render():
        return 0
    print(f"{out.relative_to(ROOT)} is stale; run scripts/gen_gateway_command_parity.py", file=sys.stderr)
    return 1


def main(argv) -> int:
    if "--check" in argv:
        return check()
    OUT.write_text(render(), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
