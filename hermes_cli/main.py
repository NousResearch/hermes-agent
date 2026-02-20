#!/usr/bin/env python3
"""
Hermes CLI - Main entry point.

Usage:
    hermes                     # Interactive chat (default)
    hermes chat                # Interactive chat
    hermes gateway             # Run gateway in foreground
    hermes gateway start       # Start gateway as service
    hermes gateway stop        # Stop gateway service
    hermes gateway status      # Show gateway status
    hermes gateway install     # Install gateway service
    hermes gateway uninstall   # Uninstall gateway service
    hermes setup               # Interactive setup wizard
    hermes status              # Show status of all components
    hermes cron                # Manage cron jobs
    hermes cron list           # List cron jobs
    hermes cron daemon         # Run cron daemon
    hermes doctor              # Check configuration and dependencies
    hermes version             # Show version
    hermes update              # Update to latest version
    hermes uninstall           # Uninstall Hermes Agent
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file
from dotenv import load_dotenv
env_path = PROJECT_ROOT / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

from hermes_cli import __version__


def cmd_chat(args):
    """Run interactive chat CLI."""
    # Import and run the CLI
    from cli import main as cli_main
    
    # Build kwargs from args
    kwargs = {
        "model": args.model,
        "toolsets": args.toolsets,
        "verbose": args.verbose,
        "query": args.query,
    }
    # Filter out None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    
    cli_main(**kwargs)


def cmd_gateway(args):
    """Gateway management commands."""
    from hermes_cli.gateway import gateway_command
    gateway_command(args)


def cmd_setup(args):
    """Interactive setup wizard."""
    from hermes_cli.setup import run_setup_wizard
    run_setup_wizard(args)


def cmd_status(args):
    """Show status of all components."""
    from hermes_cli.status import show_status
    show_status(args)


def cmd_cron(args):
    """Cron job management."""
    from hermes_cli.cron import cron_command
    cron_command(args)


def cmd_doctor(args):
    """Check configuration and dependencies."""
    from hermes_cli.doctor import run_doctor
    run_doctor(args)


def cmd_config(args):
    """Configuration management."""
    from hermes_cli.config import config_command
    config_command(args)


def cmd_version(args):
    """Show version."""
    print(f"Hermes Agent v{__version__}")
    print(f"Project: {PROJECT_ROOT}")
    
    # Show Python version
    print(f"Python: {sys.version.split()[0]}")
    
    # Check for key dependencies
    try:
        import openai
        print(f"OpenAI SDK: {openai.__version__}")
    except ImportError:
        print("OpenAI SDK: Not installed")


def cmd_uninstall(args):
    """Uninstall Hermes Agent."""
    from hermes_cli.uninstall import run_uninstall
    run_uninstall(args)


def cmd_update(args):
    """Update Hermes Agent to the latest version."""
    import subprocess
    import shutil
    
    print("🦋 Updating Hermes Agent...")
    print()
    
    # Check if we're in a git repo
    git_dir = PROJECT_ROOT / '.git'
    if not git_dir.exists():
        print("✗ Not a git repository. Please reinstall:")
        print("  curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash")
        sys.exit(1)
    
    # Fetch and pull
    try:
        print("→ Fetching updates...")
        subprocess.run(["git", "fetch", "origin"], cwd=PROJECT_ROOT, check=True)
        
        # Get current branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True
        )
        branch = result.stdout.strip()
        
        # Check if there are updates
        result = subprocess.run(
            ["git", "rev-list", f"HEAD..origin/{branch}", "--count"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True
        )
        commit_count = int(result.stdout.strip())
        
        if commit_count == 0:
            print("✓ Already up to date!")
            return
        
        print(f"→ Found {commit_count} new commit(s)")
        print("→ Pulling updates...")
        subprocess.run(["git", "pull", "origin", branch], cwd=PROJECT_ROOT, check=True)
        
        # Reinstall Python dependencies (prefer uv for speed, fall back to pip)
        print("→ Updating Python dependencies...")
        uv_bin = shutil.which("uv")
        if uv_bin:
            subprocess.run(
                [uv_bin, "pip", "install", "-e", ".", "--quiet"],
                cwd=PROJECT_ROOT, check=True,
                env={**os.environ, "VIRTUAL_ENV": str(PROJECT_ROOT / "venv")}
            )
        else:
            venv_pip = PROJECT_ROOT / "venv" / "bin" / "pip"
            if venv_pip.exists():
                subprocess.run([str(venv_pip), "install", "-e", ".", "--quiet"], cwd=PROJECT_ROOT, check=True)
            else:
                subprocess.run(["pip", "install", "-e", ".", "--quiet"], cwd=PROJECT_ROOT, check=True)
        
        # Check for Node.js deps
        if (PROJECT_ROOT / "package.json").exists():
            import shutil
            if shutil.which("npm"):
                print("→ Updating Node.js dependencies...")
                subprocess.run(["npm", "install", "--silent"], cwd=PROJECT_ROOT, check=False)
        
        print()
        print("✓ Code updated!")
        
        # Sync any new bundled skills (manifest-based -- won't overwrite or re-add deleted skills)
        try:
            from tools.skills_sync import sync_skills
            print()
            print("→ Checking for new bundled skills...")
            result = sync_skills(quiet=True)
            if result["copied"]:
                print(f"  + {len(result['copied'])} new skill(s): {', '.join(result['copied'])}")
            else:
                print("  ✓ Skills are up to date")
        except Exception:
            pass
        
        # Check for config migrations
        print()
        print("→ Checking configuration for new options...")
        
        from hermes_cli.config import (
            get_missing_env_vars, get_missing_config_fields, 
            check_config_version, migrate_config
        )
        
        missing_env = get_missing_env_vars(required_only=True)
        missing_config = get_missing_config_fields()
        current_ver, latest_ver = check_config_version()
        
        needs_migration = missing_env or missing_config or current_ver < latest_ver
        
        if needs_migration:
            print()
            if missing_env:
                print(f"  ⚠️  {len(missing_env)} new required setting(s) need configuration")
            if missing_config:
                print(f"  ℹ️  {len(missing_config)} new config option(s) available")
            
            print()
            response = input("Would you like to configure them now? [Y/n]: ").strip().lower()
            
            if response in ('', 'y', 'yes'):
                print()
                results = migrate_config(interactive=True, quiet=False)
                
                if results["env_added"] or results["config_added"]:
                    print()
                    print("✓ Configuration updated!")
            else:
                print()
                print("Skipped. Run 'hermes config migrate' later to configure.")
        else:
            print("  ✓ Configuration is up to date")
        
        print()
        print("✓ Update complete!")
        print()
        print("Note: If you have the gateway service running, restart it:")
        print("  hermes gateway restart")
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Update failed: {e}")
        sys.exit(1)


def main():
    """Main entry point for hermes CLI."""
    parser = argparse.ArgumentParser(
        prog="hermes",
        description="Hermes Agent - AI assistant with tool-calling capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    hermes                        Start interactive chat
    hermes chat -q "Hello"        Single query mode
    hermes setup                  Run setup wizard
    hermes config                 View configuration
    hermes config edit            Edit config in $EDITOR
    hermes config set model gpt-4 Set a config value
    hermes gateway                Run messaging gateway
    hermes gateway install        Install as system service
    hermes update                 Update to latest version

For more help on a command:
    hermes <command> --help
"""
    )
    
    parser.add_argument(
        "--version", "-V",
        action="store_true",
        help="Show version and exit"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # =========================================================================
    # chat command
    # =========================================================================
    chat_parser = subparsers.add_parser(
        "chat",
        help="Interactive chat with the agent",
        description="Start an interactive chat session with Hermes Agent"
    )
    chat_parser.add_argument(
        "-q", "--query",
        help="Single query (non-interactive mode)"
    )
    chat_parser.add_argument(
        "-m", "--model",
        help="Model to use (e.g., anthropic/claude-sonnet-4)"
    )
    chat_parser.add_argument(
        "-t", "--toolsets",
        help="Comma-separated toolsets to enable"
    )
    chat_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    chat_parser.set_defaults(func=cmd_chat)
    
    # =========================================================================
    # gateway command
    # =========================================================================
    gateway_parser = subparsers.add_parser(
        "gateway",
        help="Messaging gateway management",
        description="Manage the messaging gateway (Telegram, Discord, WhatsApp)"
    )
    gateway_subparsers = gateway_parser.add_subparsers(dest="gateway_command")
    
    # gateway run (default)
    gateway_run = gateway_subparsers.add_parser("run", help="Run gateway in foreground")
    gateway_run.add_argument("-v", "--verbose", action="store_true")
    
    # gateway start
    gateway_start = gateway_subparsers.add_parser("start", help="Start gateway service")
    
    # gateway stop
    gateway_stop = gateway_subparsers.add_parser("stop", help="Stop gateway service")
    
    # gateway restart
    gateway_restart = gateway_subparsers.add_parser("restart", help="Restart gateway service")
    
    # gateway status
    gateway_status = gateway_subparsers.add_parser("status", help="Show gateway status")
    gateway_status.add_argument("--deep", action="store_true", help="Deep status check")
    
    # gateway install
    gateway_install = gateway_subparsers.add_parser("install", help="Install gateway as service")
    gateway_install.add_argument("--force", action="store_true", help="Force reinstall")
    
    # gateway uninstall
    gateway_uninstall = gateway_subparsers.add_parser("uninstall", help="Uninstall gateway service")
    
    gateway_parser.set_defaults(func=cmd_gateway)
    
    # =========================================================================
    # setup command
    # =========================================================================
    setup_parser = subparsers.add_parser(
        "setup",
        help="Interactive setup wizard",
        description="Configure Hermes Agent with an interactive wizard"
    )
    setup_parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Non-interactive mode (use defaults/env vars)"
    )
    setup_parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset configuration to defaults"
    )
    setup_parser.set_defaults(func=cmd_setup)
    
    # =========================================================================
    # status command
    # =========================================================================
    status_parser = subparsers.add_parser(
        "status",
        help="Show status of all components",
        description="Display status of Hermes Agent components"
    )
    status_parser.add_argument(
        "--all",
        action="store_true",
        help="Show all details (redacted for sharing)"
    )
    status_parser.add_argument(
        "--deep",
        action="store_true",
        help="Run deep checks (may take longer)"
    )
    status_parser.set_defaults(func=cmd_status)
    
    # =========================================================================
    # cron command
    # =========================================================================
    cron_parser = subparsers.add_parser(
        "cron",
        help="Cron job management",
        description="Manage scheduled tasks"
    )
    cron_subparsers = cron_parser.add_subparsers(dest="cron_command")
    
    # cron list
    cron_list = cron_subparsers.add_parser("list", help="List scheduled jobs")
    cron_list.add_argument("--all", action="store_true", help="Include disabled jobs")
    
    # cron daemon
    cron_daemon = cron_subparsers.add_parser("daemon", help="Run cron daemon")
    cron_daemon.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    
    # cron tick
    cron_tick = cron_subparsers.add_parser("tick", help="Run due jobs once (for system cron)")
    
    cron_parser.set_defaults(func=cmd_cron)
    
    # =========================================================================
    # doctor command
    # =========================================================================
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check configuration and dependencies",
        description="Diagnose issues with Hermes Agent setup"
    )
    doctor_parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to fix issues automatically"
    )
    doctor_parser.set_defaults(func=cmd_doctor)
    
    # =========================================================================
    # config command
    # =========================================================================
    config_parser = subparsers.add_parser(
        "config",
        help="View and edit configuration",
        description="Manage Hermes Agent configuration"
    )
    config_subparsers = config_parser.add_subparsers(dest="config_command")
    
    # config show (default)
    config_show = config_subparsers.add_parser("show", help="Show current configuration")
    
    # config edit
    config_edit = config_subparsers.add_parser("edit", help="Open config file in editor")
    
    # config set
    config_set = config_subparsers.add_parser("set", help="Set a configuration value")
    config_set.add_argument("key", nargs="?", help="Configuration key (e.g., model, terminal.backend)")
    config_set.add_argument("value", nargs="?", help="Value to set")
    
    # config path
    config_path = config_subparsers.add_parser("path", help="Print config file path")
    
    # config env-path
    config_env = config_subparsers.add_parser("env-path", help="Print .env file path")
    
    # config check
    config_check = config_subparsers.add_parser("check", help="Check for missing/outdated config")
    
    # config migrate
    config_migrate = config_subparsers.add_parser("migrate", help="Update config with new options")
    
    config_parser.set_defaults(func=cmd_config)
    
    # =========================================================================
    # pairing command
    # =========================================================================
    pairing_parser = subparsers.add_parser(
        "pairing",
        help="Manage DM pairing codes for user authorization",
        description="Approve or revoke user access via pairing codes"
    )
    pairing_sub = pairing_parser.add_subparsers(dest="pairing_action")

    pairing_list_parser = pairing_sub.add_parser("list", help="Show pending + approved users")

    pairing_approve_parser = pairing_sub.add_parser("approve", help="Approve a pairing code")
    pairing_approve_parser.add_argument("platform", help="Platform name (telegram, discord, slack, whatsapp)")
    pairing_approve_parser.add_argument("code", help="Pairing code to approve")

    pairing_revoke_parser = pairing_sub.add_parser("revoke", help="Revoke user access")
    pairing_revoke_parser.add_argument("platform", help="Platform name")
    pairing_revoke_parser.add_argument("user_id", help="User ID to revoke")

    pairing_clear_parser = pairing_sub.add_parser("clear-pending", help="Clear all pending codes")

    def cmd_pairing(args):
        from hermes_cli.pairing import pairing_command
        pairing_command(args)

    pairing_parser.set_defaults(func=cmd_pairing)

    # =========================================================================
    # skills command
    # =========================================================================
    skills_parser = subparsers.add_parser(
        "skills",
        help="Skills Hub — search, install, and manage skills from online registries",
        description="Search, install, inspect, audit, and manage skills from GitHub, ClawHub, and other registries."
    )
    skills_subparsers = skills_parser.add_subparsers(dest="skills_action")

    skills_search = skills_subparsers.add_parser("search", help="Search skill registries")
    skills_search.add_argument("query", help="Search query")
    skills_search.add_argument("--source", default="all", choices=["all", "github", "clawhub", "lobehub"])
    skills_search.add_argument("--limit", type=int, default=10, help="Max results")

    skills_install = skills_subparsers.add_parser("install", help="Install a skill")
    skills_install.add_argument("identifier", help="Skill identifier (e.g. openai/skills/skill-creator)")
    skills_install.add_argument("--category", default="", help="Category folder to install into")
    skills_install.add_argument("--force", action="store_true", help="Install despite caution verdict")

    skills_inspect = skills_subparsers.add_parser("inspect", help="Preview a skill without installing")
    skills_inspect.add_argument("identifier", help="Skill identifier")

    skills_list = skills_subparsers.add_parser("list", help="List installed skills")
    skills_list.add_argument("--source", default="all", choices=["all", "hub", "builtin"])

    skills_audit = skills_subparsers.add_parser("audit", help="Re-scan installed hub skills")
    skills_audit.add_argument("name", nargs="?", help="Specific skill to audit (default: all)")

    skills_uninstall = skills_subparsers.add_parser("uninstall", help="Remove a hub-installed skill")
    skills_uninstall.add_argument("name", help="Skill name to remove")

    skills_publish = skills_subparsers.add_parser("publish", help="Publish a skill to a registry")
    skills_publish.add_argument("skill_path", help="Path to skill directory")
    skills_publish.add_argument("--to", default="github", choices=["github", "clawhub"], help="Target registry")
    skills_publish.add_argument("--repo", default="", help="Target GitHub repo (e.g. openai/skills)")

    skills_snapshot = skills_subparsers.add_parser("snapshot", help="Export/import skill configurations")
    snapshot_subparsers = skills_snapshot.add_subparsers(dest="snapshot_action")
    snap_export = snapshot_subparsers.add_parser("export", help="Export installed skills to a file")
    snap_export.add_argument("output", help="Output JSON file path")
    snap_import = snapshot_subparsers.add_parser("import", help="Import and install skills from a file")
    snap_import.add_argument("input", help="Input JSON file path")
    snap_import.add_argument("--force", action="store_true", help="Force install despite caution verdict")

    skills_tap = skills_subparsers.add_parser("tap", help="Manage skill sources")
    tap_subparsers = skills_tap.add_subparsers(dest="tap_action")
    tap_subparsers.add_parser("list", help="List configured taps")
    tap_add = tap_subparsers.add_parser("add", help="Add a GitHub repo as skill source")
    tap_add.add_argument("repo", help="GitHub repo (e.g. owner/repo)")
    tap_rm = tap_subparsers.add_parser("remove", help="Remove a tap")
    tap_rm.add_argument("name", help="Tap name to remove")

    def cmd_skills(args):
        from hermes_cli.skills_hub import skills_command
        skills_command(args)

    skills_parser.set_defaults(func=cmd_skills)

    # =========================================================================
    # sessions command
    # =========================================================================
    sessions_parser = subparsers.add_parser(
        "sessions",
        help="Manage session history (list, export, prune, delete)",
        description="View and manage the SQLite session store"
    )
    sessions_subparsers = sessions_parser.add_subparsers(dest="sessions_action")

    sessions_list = sessions_subparsers.add_parser("list", help="List recent sessions")
    sessions_list.add_argument("--source", help="Filter by source (cli, telegram, discord, etc.)")
    sessions_list.add_argument("--limit", type=int, default=20, help="Max sessions to show")

    sessions_export = sessions_subparsers.add_parser("export", help="Export sessions to a JSONL file")
    sessions_export.add_argument("output", help="Output JSONL file path")
    sessions_export.add_argument("--source", help="Filter by source")
    sessions_export.add_argument("--session-id", help="Export a specific session")

    sessions_delete = sessions_subparsers.add_parser("delete", help="Delete a specific session")
    sessions_delete.add_argument("session_id", help="Session ID to delete")
    sessions_delete.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    sessions_prune = sessions_subparsers.add_parser("prune", help="Delete old sessions")
    sessions_prune.add_argument("--older-than", type=int, default=90, help="Delete sessions older than N days (default: 90)")
    sessions_prune.add_argument("--source", help="Only prune sessions from this source")
    sessions_prune.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    sessions_stats = sessions_subparsers.add_parser("stats", help="Show session store statistics")

    def cmd_sessions(args):
        import json as _json
        try:
            from hermes_state import SessionDB
            db = SessionDB()
        except Exception as e:
            print(f"Error: Could not open session database: {e}")
            return

        action = args.sessions_action

        if action == "list":
            sessions = db.search_sessions(source=args.source, limit=args.limit)
            if not sessions:
                print("No sessions found.")
                return
            print(f"{'ID':<30} {'Source':<12} {'Model':<30} {'Messages':>8} {'Started'}")
            print("─" * 100)
            from datetime import datetime
            for s in sessions:
                started = datetime.fromtimestamp(s["started_at"]).strftime("%Y-%m-%d %H:%M") if s["started_at"] else "?"
                model = (s.get("model") or "?")[:28]
                ended = " (ended)" if s.get("ended_at") else ""
                print(f"{s['id']:<30} {s['source']:<12} {model:<30} {s['message_count']:>8} {started}{ended}")

        elif action == "export":
            if args.session_id:
                data = db.export_session(args.session_id)
                if not data:
                    print(f"Session '{args.session_id}' not found.")
                    return
                with open(args.output, "w") as f:
                    f.write(_json.dumps(data, ensure_ascii=False) + "\n")
                print(f"Exported 1 session to {args.output}")
            else:
                sessions = db.export_all(source=args.source)
                with open(args.output, "w") as f:
                    for s in sessions:
                        f.write(_json.dumps(s, ensure_ascii=False) + "\n")
                print(f"Exported {len(sessions)} sessions to {args.output}")

        elif action == "delete":
            if not args.yes:
                confirm = input(f"Delete session '{args.session_id}' and all its messages? [y/N] ")
                if confirm.lower() not in ("y", "yes"):
                    print("Cancelled.")
                    return
            if db.delete_session(args.session_id):
                print(f"Deleted session '{args.session_id}'.")
            else:
                print(f"Session '{args.session_id}' not found.")

        elif action == "prune":
            days = args.older_than
            source_msg = f" from '{args.source}'" if args.source else ""
            if not args.yes:
                confirm = input(f"Delete all ended sessions older than {days} days{source_msg}? [y/N] ")
                if confirm.lower() not in ("y", "yes"):
                    print("Cancelled.")
                    return
            count = db.prune_sessions(older_than_days=days, source=args.source)
            print(f"Pruned {count} session(s).")

        elif action == "stats":
            total = db.session_count()
            msgs = db.message_count()
            print(f"Total sessions: {total}")
            print(f"Total messages: {msgs}")
            for src in ["cli", "telegram", "discord", "whatsapp", "slack"]:
                c = db.session_count(source=src)
                if c > 0:
                    print(f"  {src}: {c} sessions")
            import os
            db_path = db.db_path
            if db_path.exists():
                size_mb = os.path.getsize(db_path) / (1024 * 1024)
                print(f"Database size: {size_mb:.1f} MB")

        else:
            sessions_parser.print_help()

        db.close()

    sessions_parser.set_defaults(func=cmd_sessions)

    # =========================================================================
    # version command
    # =========================================================================
    version_parser = subparsers.add_parser(
        "version",
        help="Show version information"
    )
    version_parser.set_defaults(func=cmd_version)
    
    # =========================================================================
    # update command
    # =========================================================================
    update_parser = subparsers.add_parser(
        "update",
        help="Update Hermes Agent to the latest version",
        description="Pull the latest changes from git and reinstall dependencies"
    )
    update_parser.set_defaults(func=cmd_update)
    
    # =========================================================================
    # uninstall command
    # =========================================================================
    uninstall_parser = subparsers.add_parser(
        "uninstall",
        help="Uninstall Hermes Agent",
        description="Remove Hermes Agent from your system. Can keep configs/data for reinstall."
    )
    uninstall_parser.add_argument(
        "--full",
        action="store_true",
        help="Full uninstall - remove everything including configs and data"
    )
    uninstall_parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompts"
    )
    uninstall_parser.set_defaults(func=cmd_uninstall)
    
    # =========================================================================
    # Parse and execute
    # =========================================================================
    args = parser.parse_args()
    
    # Handle --version flag
    if args.version:
        cmd_version(args)
        return
    
    # Default to chat if no command specified
    if args.command is None:
        # No command = run chat
        args.query = None
        args.model = None
        args.toolsets = None
        args.verbose = False
        cmd_chat(args)
        return
    
    # Execute the command
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
