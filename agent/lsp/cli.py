"""``hermes lsp`` CLI subcommand: status / list / install / install-all / restart / which.

Handlers live here (not in ``hermes_cli/main.py``) so the LSP module ships self-contained.
"""
from __future__ import annotations

import argparse
import sys

_STATUS_MARKERS = {"installed": "✓", "missing": "·", "manual-only": "?"}

# (name, help, optional (flag, add_argument kwargs), handler(args)) — order defines the help listing.
_SUBCOMMANDS = [
    ("status", "Show LSP service status",
     ("--json", {"action": "store_true", "help": "Emit machine-readable JSON"}),
     lambda a: _cmd_status(getattr(a, "json", False))),
    ("list", "List supported language servers",
     ("--installed-only", {"action": "store_true", "help": "Only show servers whose binary is currently available"}),
     lambda a: _cmd_list(getattr(a, "installed_only", False))),
    ("install", "Install a server binary", ("server", {"help": "Server id (e.g. pyright, gopls)"}),
     lambda a: _cmd_install(a.server)),
    ("install-all", "Install every server with a known auto-install recipe",
     ("--include-manual", {"action": "store_true", "help": "Even attempt servers marked manual-install (best effort)"}),
     lambda a: _cmd_install_all(getattr(a, "include_manual", False))),
    ("restart", "Tear down running LSP clients (next edit re-spawns)", None, lambda a: _cmd_restart()),
    ("which", "Print binary path for a server", ("server", {"help": "Server id"}), lambda a: _cmd_which(a.server)),
]
_COMMANDS = {name: handler for name, _, _, handler in _SUBCOMMANDS}


def register_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Wire the ``hermes lsp`` subcommand tree into the main argparse."""
    parser = subparsers.add_parser(
        "lsp",
        help="Language Server Protocol management",
        description="Manage the LSP layer that powers post-write semantic diagnostics in write_file/patch.",
    )
    sub = parser.add_subparsers(dest="lsp_command")
    for name, help_text, arg, _handler in _SUBCOMMANDS:
        p = sub.add_parser(name, help=help_text)
        if arg is not None:
            p.add_argument(arg[0], **arg[1])
    parser.set_defaults(func=run_lsp_command)


def run_lsp_command(args: argparse.Namespace) -> int:
    """Top-level dispatcher for ``hermes lsp <subcommand>``."""
    sub = getattr(args, "lsp_command", None) or "status"
    try:
        handler = _COMMANDS.get(sub)
        if handler is None:
            sys.stderr.write(f"unknown lsp subcommand: {sub}\n")
            return 2
        return handler(args)
    except KeyboardInterrupt:
        return 130


def _status_for(server_id: str) -> str:
    from agent.lsp.install import detect_status
    return detect_status(_recipe_pkg_for(server_id))


def _cmd_status(emit_json: bool) -> int:
    from agent.lsp import get_service
    from agent.lsp.servers import SERVERS
    svc = get_service()
    info = svc.get_status() if svc is not None else {"enabled": False}
    if emit_json:
        import json
        registry = [{"server_id": s.server_id, "extensions": list(s.extensions), "description": s.description,
                     "binary_status": _status_for(s.server_id)} for s in SERVERS]
        sys.stdout.write(json.dumps({"service": info, "registry": registry}, indent=2) + "\n")
        return 0

    out = ["LSP Service", "===========", f"  enabled:         {info.get('enabled', False)}"]
    if svc is not None:
        out += [f"  wait_mode:       {info.get('wait_mode')}",
                f"  wait_timeout:    {info.get('wait_timeout')}s",
                f"  install_strategy:{info.get('install_strategy')}"]
        clients = info.get("clients") or []
        out.append(f"  active clients:  {len(clients)}" if clients else "  active clients:  none")
        out += [f"    - {c['server_id']:20s} state={c['state']:10s} root={c['workspace_root']}" for c in clients]
        broken = info.get("broken") or []
        if broken:
            out.append(f"  broken pairs:    {len(broken)}")
            out += [f"    - {b}" for b in broken]
        if disabled := info.get("disabled_servers"):
            out.append(f"  disabled in cfg: {', '.join(disabled)}")
    # Sidecar gaps the registry table can't show (bash-language-server -> shellcheck).
    if backend_warnings := _backend_warnings():
        out += ["", "Backend warnings", "================"] + [f"  ! {line}" for line in backend_warnings]
    out += ["", "Registered Servers", "=================="]
    for s in SERVERS:
        status = _status_for(s.server_id)
        ext_summary = ", ".join(list(s.extensions)[:5])
        if len(s.extensions) > 5:
            ext_summary += f", … (+{len(s.extensions) - 5})"
        out.append(f"  {_STATUS_MARKERS.get(status, ' ')} {s.server_id:24s} [{status:11s}] {ext_summary}")
        if s.description:
            out.append(f"      {s.description}")
    sys.stdout.write("\n".join(out) + "\n")
    return 0


def _cmd_list(installed_only: bool) -> int:
    from agent.lsp.servers import SERVERS
    for s in SERVERS:
        status = _status_for(s.server_id)
        if not (installed_only and status != "installed"):
            sys.stdout.write(f"{s.server_id:24s} [{status:11s}] {','.join(s.extensions)}\n")
    return 0


def _cmd_install(server_id: str) -> int:
    from agent.lsp.install import try_install, INSTALL_RECIPES
    pkg = _recipe_pkg_for(server_id)
    if _status_for(server_id) == "installed":
        sys.stdout.write(f"{server_id} already installed\n")
        return 0
    sys.stdout.write(f"installing {server_id} (pkg={pkg}) ...\n")
    sys.stdout.flush()
    bin_path = try_install(pkg, "auto")
    if bin_path is not None:
        sys.stdout.write(f"installed: {bin_path}\n")
        return 0
    manual = (INSTALL_RECIPES.get(pkg) or {}).get("strategy") == "manual"
    sys.stderr.write(f"{server_id}: this server requires a manual install. See documentation.\n" if manual
                     else f"{server_id}: install failed (see logs).\n")
    return 1


def _cmd_install_all(include_manual: bool) -> int:
    from agent.lsp.servers import SERVERS
    from agent.lsp.install import try_install, INSTALL_RECIPES
    rc = 0
    for s in SERVERS:
        pkg = _recipe_pkg_for(s.server_id)
        recipe = INSTALL_RECIPES.get(pkg)
        if recipe is None or (recipe.get("strategy") == "manual" and not include_manual):
            continue
        if _status_for(s.server_id) == "installed":
            sys.stdout.write(f"  {s.server_id:24s} already installed\n")
            continue
        sys.stdout.write(f"  installing {s.server_id} (pkg={pkg}) ... ")
        sys.stdout.flush()
        path = try_install(pkg, "auto")
        sys.stdout.write(f"ok ({path})\n" if path else "FAILED\n")
        rc = rc if path else 1
    return rc


def _cmd_restart() -> int:
    from agent.lsp import shutdown_service
    shutdown_service()
    sys.stdout.write("LSP service shut down. Next edit will respawn clients.\n")
    return 0


def _cmd_which(server_id: str) -> int:
    from agent.lsp.install import INSTALL_RECIPES, _existing_binary
    resolved = _existing_binary((INSTALL_RECIPES.get(server_id) or {}).get("bin", server_id))
    if resolved:
        sys.stdout.write(resolved + "\n")
        return 0
    sys.stderr.write(f"{server_id}: not installed\n")
    return 1


# server_id → install-recipe key, where the two differ.
_RECIPE_ALIASES = {
    "vue-language-server": "@vue/language-server",
    "astro-language-server": "@astrojs/language-server",
    "dockerfile-ls": "dockerfile-language-server-nodejs",
    "typescript": "typescript-language-server",
}


def _recipe_pkg_for(server_id: str) -> str:
    """Map a registry ``server_id`` to its install-recipe package key."""
    return _RECIPE_ALIASES.get(server_id, server_id)


def _backend_warnings() -> list:
    """Notes about missing sidecar tools that make a server spawn fine but emit nothing (e.g. shellcheck)."""
    import shutil
    from agent.lsp.install import _existing_binary
    if _existing_binary("bash-language-server") is not None and shutil.which("shellcheck") is None:
        return ["bash-language-server is installed but shellcheck is missing — "
                "diagnostics will be empty (apt: shellcheck, brew: shellcheck, scoop: shellcheck)."]
    return []
