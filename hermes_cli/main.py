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
