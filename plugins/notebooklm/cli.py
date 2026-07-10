"""CLI commands for the NotebookLM Hermes plugin."""

from __future__ import annotations

import argparse
import json

from . import core
from . import mcp_stack
from . import bridge


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="notebooklm_command")

    subs.add_parser("status", help="Show NotebookLM plugin readiness")

    setup = subs.add_parser(
        "setup", help="Save NotebookLM Enterprise settings to the Hermes .env file"
    )
    setup.add_argument("--project-number", default="")
    setup.add_argument("--location", default="")
    setup.add_argument("--endpoint-location", default="")
    setup.add_argument("--notebook-id", default="")
    setup.add_argument("--notebook-title", default="")
    setup.add_argument("--access-token", default="")
    setup.add_argument("--mcp-notebook-id", default="")

    setup_mcp = subs.add_parser(
        "setup-mcp",
        help="Install notebooklm-mcp-cli, register MCP server, and install Hermes skill",
    )
    setup_mcp.add_argument("--skip-cli-install", action="store_true")
    setup_mcp.add_argument("--skip-mcp-register", action="store_true")
    setup_mcp.add_argument("--skip-skill", action="store_true")
    setup_mcp.add_argument("--dry-run", action="store_true")

    subs.add_parser("login", help="Check NotebookLM consumer auth via nlm login --check")
    subs.add_parser("doctor", help="Run nlm doctor and show plugin status")
    subs.add_parser("notebooks", help="List NotebookLM notebooks via nlm")

    collect = subs.add_parser(
        "collect", help="Collect implementation logs and X activity into Markdown"
    )
    collect.add_argument("--max-logs", type=int, default=None)
    collect.add_argument("--max-x-events", type=int, default=None)
    collect.add_argument("--output-path", default="")

    brainstorm = subs.add_parser(
        "brainstorm", help="Generate X post brainstorming from a source bundle"
    )
    brainstorm.add_argument("--source-path", default="")
    brainstorm.add_argument("--idea-count", type=int, default=None)
    brainstorm.add_argument("--output-path", default="")
    brainstorm.add_argument("--provider", default="")
    brainstorm.add_argument("--model", default="")

    sync = subs.add_parser(
        "sync", help="Upload a source bundle to NotebookLM Enterprise"
    )
    sync.add_argument("--source-path", default="")
    sync.add_argument("--notebook-id", default="")
    sync.add_argument("--create-notebook", action="store_true")
    sync.add_argument("--save-notebook-id", action="store_true")
    sync.add_argument(
        "--consumer",
        action="store_true",
        help="Force consumer NotebookLM sync via notebooklm-mcp-cli",
    )
    sync.add_argument("--wait", action="store_true")

    run = subs.add_parser("run", help="Collect, brainstorm, and optionally sync")
    run.add_argument("--sync", action="store_true")
    run.add_argument("--create-notebook", action="store_true")
    run.add_argument("--idea-count", type=int, default=None)
    run.add_argument("--provider", default="")
    run.add_argument("--model", default="")

    subparser.set_defaults(func=notebooklm_command)


def notebooklm_command(args: argparse.Namespace) -> int:
    command = getattr(args, "notebooklm_command", None)
    if not command:
        print(
            "usage: hermes notebooklm "
            "{status,setup,setup-mcp,login,doctor,notebooks,collect,brainstorm,sync,run}"
        )
        return 2
    if command == "status":
        return _print(core.status())
    if command == "setup":
        return _print(_setup_values(args))
    if command == "setup-mcp":
        return _print(
            mcp_stack.setup_mcp_stack(
                install_cli=not getattr(args, "skip_cli_install", False),
                register_mcp=not getattr(args, "skip_mcp_register", False),
                install_skill=not getattr(args, "skip_skill", False),
                dry_run=bool(getattr(args, "dry_run", False)),
            )
        )
    if command == "login":
        return _print(bridge.auth_status())
    if command == "doctor":
        return _print({"doctor": bridge.doctor(), "status": core.status()})
    if command == "notebooks":
        return _print(bridge.list_notebooks())
    if command == "collect":
        return _print(
            core.collect_source(
                max_logs=getattr(args, "max_logs", None),
                max_x_events=getattr(args, "max_x_events", None),
                output_path=getattr(args, "output_path", "") or None,
            )
        )
    if command == "brainstorm":
        return _print(
            core.brainstorm_posts(
                source_path=getattr(args, "source_path", "") or None,
                idea_count=getattr(args, "idea_count", None),
                output_path=getattr(args, "output_path", "") or None,
                provider=getattr(args, "provider", "") or None,
                model=getattr(args, "model", "") or None,
            )
        )
    if command == "sync":
        mode = "consumer" if getattr(args, "consumer", False) else "auto"
        return _print(
            core.sync_source(
                source_path=getattr(args, "source_path", "") or None,
                notebook_id=getattr(args, "notebook_id", "") or None,
                create_if_missing=bool(getattr(args, "create_notebook", False)),
                save_notebook_id=bool(getattr(args, "save_notebook_id", False)),
                mode=mode,
                wait=bool(getattr(args, "wait", False)),
            )
        )
    if command == "run":
        return _print(
            core.run_pipeline(
                do_sync=bool(getattr(args, "sync", False)),
                create_if_missing=bool(getattr(args, "create_notebook", False)),
                idea_count=getattr(args, "idea_count", None),
                provider=getattr(args, "provider", "") or None,
                model=getattr(args, "model", "") or None,
            )
        )
    print(f"unknown command: {command}")
    return 2


def _setup_values(args: argparse.Namespace) -> dict:
    values = {
        "NOTEBOOKLM_ENTERPRISE_PROJECT_NUMBER": getattr(args, "project_number", ""),
        "NOTEBOOKLM_ENTERPRISE_LOCATION": getattr(args, "location", ""),
        "NOTEBOOKLM_ENTERPRISE_ENDPOINT_LOCATION": getattr(
            args, "endpoint_location", ""
        ),
        "NOTEBOOKLM_ENTERPRISE_NOTEBOOK_ID": getattr(args, "notebook_id", ""),
        "NOTEBOOKLM_ENTERPRISE_NOTEBOOK_TITLE": getattr(args, "notebook_title", ""),
        "NOTEBOOKLM_ENTERPRISE_ACCESS_TOKEN": getattr(args, "access_token", ""),
        "NOTEBOOKLM_MCP_NOTEBOOK_ID": getattr(args, "mcp_notebook_id", ""),
    }
    result = core.save_setup_values(values)
    result["status"] = core.status()
    return result


def _print(data: dict) -> int:
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0 if data.get("ok") else 1
