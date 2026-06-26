"""CLI for the Freebuff Hermes plugin."""

from __future__ import annotations

import argparse
import json

from . import core
from . import proxy as proxy_mgr


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="freebuff_command")

    subs.add_parser("status", help="Show npm CLI, native binary, proxy, and model wiring")
    subs.add_parser("doctor", help="Preflight checks for Node, binary, token, proxy, and config")

    subs.add_parser(
        "setup",
        help="Enable plugin in config.yaml and add freebuff toolset to CLI",
    )

    connect = subs.add_parser(
        "connect",
        help="Start local OpenAI proxy and set model.provider=freebuff for Hermes AI",
    )
    connect.add_argument(
        "--no-apply-model",
        action="store_true",
        help="Start proxy and sync token only; do not change model.provider",
    )
    connect.add_argument(
        "--no-start-proxy",
        action="store_true",
        help="Configure model only; do not spawn the local proxy process",
    )
    connect.add_argument(
        "--skip-proxy-install",
        action="store_true",
        help="Do not pip-install freebuff2api automatically",
    )

    install = subs.add_parser("install", help="Run npm install -g freebuff")
    install.add_argument(
        "--local",
        action="store_true",
        help="Install into current project instead of global (-g)",
    )

    run = subs.add_parser("run", help="Launch interactive Freebuff TUI in workdir")
    run.add_argument("workdir", nargs="?", default="", help="Project directory")
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Print launch command without spawning",
    )

    skill = subs.add_parser("skill", help="Copy bundled freebuff skill into ~/.hermes/skills/")
    skill.add_argument("--force", action="store_true", help="Replace existing skill directory")

    proxy = subs.add_parser("proxy", help="Manage the local freebuff2api OpenAI-compatible proxy")
    proxy_sub = proxy.add_subparsers(dest="freebuff_proxy_command")
    proxy_sub.add_parser("status", help="Proxy install/running state and health probe")
    proxy_sub.add_parser("install", help="pip install pinned freebuff2api from GitHub")
    start = proxy_sub.add_parser("start", help="Start proxy (uses FREEBUFF_TOKEN + FREEBUFF_PROXY_API_KEY)")
    start.add_argument(
        "--force-restart",
        action="store_true",
        help="Stop any existing proxy before starting",
    )
    proxy_sub.add_parser("stop", help="Stop background proxy process")

    subparser.set_defaults(func=freebuff_command)


def _print(payload: dict) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get("ok", True) else 1


def freebuff_command(args: argparse.Namespace) -> int:
    command = getattr(args, "freebuff_command", None)
    if not command:
        print(
            "usage: hermes freebuff {setup,install,status,doctor,run,skill,connect,proxy ...}"
        )
        return 2

    if command == "status":
        return _print(core.status())
    if command == "doctor":
        return _print(core.doctor())
    if command == "setup":
        return _print(core.setup())
    if command == "connect":
        return _print(
            core.connect(
                apply_model=not getattr(args, "no_apply_model", False),
                start_proxy=not getattr(args, "no_start_proxy", False),
                install_proxy=not getattr(args, "skip_proxy_install", False),
            )
        )
    if command == "install":
        return _print(core.install(global_install=not getattr(args, "local", False)))
    if command == "run":
        workdir = (getattr(args, "workdir", "") or "").strip() or None
        return _print(
            core.run(
                workdir=workdir,
                dry_run=getattr(args, "dry_run", False),
            )
        )
    if command == "skill":
        return _print(core.sync_skill(force=getattr(args, "force", False)))
    if command == "proxy":
        sub = getattr(args, "freebuff_proxy_command", None) or "status"
        if sub == "status":
            return _print(proxy_mgr.proxy_status())
        if sub == "install":
            return _print(proxy_mgr.install_proxy())
        if sub == "start":
            return _print(proxy_mgr.start_proxy(force_restart=getattr(args, "force_restart", False)))
        if sub == "stop":
            return _print(proxy_mgr.stop_proxy())
        print(f"unknown freebuff proxy subcommand: {sub}")
        return 2

    print(f"unknown freebuff subcommand: {command}")
    return 2
