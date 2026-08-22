#!/usr/bin/env python3
"""Whoop API sync tool — pull fitness data from Whoop via OAuth2.

Usage:
    python3 scripts/whoop_sync.py setup      # First-time setup: enter credentials, auth, auto-cron
    python3 scripts/whoop_sync.py pull        # Pull data from all endpoints
    python3 scripts/whoop_sync.py refresh     # Manually refresh tokens
    python3 scripts/whoop_sync.py daemon      # Continuous sync on interval
    python3 scripts/whoop_sync.py status      # Show auth status and next steps

Options:
    --interval N     Seconds between pulls (daemon mode, default: 1800)
    --data-dir PATH  Output directory (default: ./whoop_data)
    --date DATE      Pull specific date YYYY-MM-DD (default: yesterday)
    -h, --help       Show this help message
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add scripts directory to path for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from whoop_client import WhoopClient
from whoop_endpoints import all_endpoints, get_endpoint
from whoop_oauth import start_oauth_flow
from whoop_storage import (
    KEYCHAIN_SERVICE,
    load_tokens,
    load_client_credentials,
    save_client_credentials,
    save_tokens,
)

PRIVACY_POLICY_URL = "https://nousresearch.github.io/hermes-agent/whoop-privacy.html"


def cmd_setup(args: argparse.Namespace) -> None:
    """Interactive first-time setup: credentials, OAuth, cron.

    Steps:
    1. Check for existing credentials — skip prompt if found
    2. Prompt for client_id and client_secret
    3. Store credentials in Keychain (macOS) or config file
    4. Open browser for Whoop OAuth authorization
    5. Exchange auth code for tokens, store them
    6. Set up cron for automatic token refresh
    """
    print("=" * 60)
    print("  Whoop API Setup")
    print("=" * 60)
    print()

    # Step 1: Check existing credentials
    existing_creds = load_client_credentials()
    if existing_creds:
        print(f"  ✓ Found existing credentials (client_id: {existing_creds['client_id'][:8]}...)")
        print()
        client_id = existing_creds["client_id"]
        client_secret = existing_creds["client_secret"]
    else:
        print("  You need a Whoop Developer App to use this skill.")
        print()
        print("  If you haven't registered yet:")
        print("    1. Go to https://developer.whoop.com")
        print(f"    2. Create a new app")
        print(f"    3. Set the Redirect URI to: http://localhost:8647/callback")
        print(f"    4. Set the Privacy Policy URL to: {PRIVACY_POLICY_URL}")
        print("    5. Copy your Client ID and Client Secret")
        print()

        client_id = input("  Enter your Whoop Client ID: ").strip()
        if not client_id:
            sys.exit("ERROR: Client ID is required.")

        client_secret = input("  Enter your Whoop Client Secret: ").strip()
        if not client_secret:
            sys.exit("ERROR: Client Secret is required.")

        # Step 2: Store credentials
        save_client_credentials(client_id, client_secret)
        print()
        print("  ✓ Credentials stored securely.")

    # Step 3: Check for existing tokens
    existing_tokens = load_tokens()
    if existing_tokens and existing_tokens.get("expires_at", 0) > time.time():
        print("  ✓ Found valid auth tokens. You're already authenticated!")
        print()
        _setup_cron()
        print()
        print("  Setup complete! Run `python3 scripts/whoop_sync.py pull` to fetch data.")
        return

    # Step 4: OAuth flow
    print("  Opening browser for Whoop authorization...")
    print("  Authorize the app and come back here.")
    print()
    try:
        start_oauth_flow(client_id, client_secret)
    except Exception as e:
        print(f"\n  ✗ OAuth flow failed: {e}")
        print("  Try running `python3 scripts/whoop_sync.py setup` again.")
        sys.exit(1)

    # Step 5: Verify tokens
    tokens = load_tokens()
    if not tokens:
        sys.exit("ERROR: Token storage failed. Check permissions and try again.")

    print()
    print("  ✓ Authenticated successfully!")
    print()

    # Step 6: Set up cron for token refresh
    _setup_cron()

    print()
    print("=" * 60)
    print("  Setup complete!")
    print()
    print("  What's next:")
    print("    • Pull data:     python3 scripts/whoop_sync.py pull")
    print("    • Check status:  python3 scripts/whoop_sync.py status")
    print("    • Continuous:     python3 scripts/whoop_sync.py daemon")
    print("=" * 60)


def _setup_cron() -> None:
    """Set up Hermes cron for token refresh (55 min) and data pull (daily).

    Creates two cron jobs if they don't already exist:
    1. Token refresh every 55 minutes (keeps access token alive)
    2. Daily data pull at 7:00 AM (local time)

    For token refresh: installs a wrapper script in HERMES_HOME/scripts/
    (required by hermes cron --script) and creates a no-agent cron job.
    For daily pull: creates an agent-based cron job with a prompt.

    Skips if Hermes CLI is not installed.
    """
    import shutil

    script_dir = Path(__file__).parent.resolve()
    refresh_script = script_dir / "whoop_token_refresh.py"
    sync_script = script_dir / "whoop_sync.py"

    # Heremes cron --script requires scripts to live in HERMES_HOME/scripts/
    hermes_home = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
    scripts_dir = hermes_home / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    # Install wrapper for token refresh (no-agent script cron)
    wrapper_path = scripts_dir / "whoop_token_refresh.py"
    wrapper_content = f'''#!/usr/bin/env python3
"""Whoop token refresh — cron wrapper.

Delegates to the whoop_token_refresh module in the skill directory.
Regenerated by whoop_sync.py setup — do not edit manually.
"""
import sys
from pathlib import Path

SKILL_DIR = {str(script_dir)!r}
sys.path.insert(0, SKILL_DIR)

from whoop_token_refresh import main

sys.exit(main())
'''
    wrapper_path.write_text(wrapper_content)
    print(f"  ✓ Installed token refresh wrapper to {wrapper_path}")

    # Install wrapper for daily pull (agent-based cron runs from workdir)
    pull_wrapper_path = scripts_dir / "whoop_daily_pull.py"
    pull_wrapper_content = f'''#!/usr/bin/env python3
"""Whoop daily pull — cron wrapper.

Delegates to whoop_sync.py pull in the skill directory.
Regenerated by whoop_sync.py setup — do not edit manually.
"""
import sys
from pathlib import Path

SKILL_DIR = {str(script_dir)!r}
sys.path.insert(0, SKILL_DIR)
sys.argv = ["whoop_sync.py", "pull"]

from whoop_sync import main

sys.exit(main())
'''
    pull_wrapper_path.write_text(pull_wrapper_content)
    print(f"  ✓ Installed daily pull wrapper to {pull_wrapper_path}")

    # Check if hermes is available
    try:
        result = subprocess.run(
            ["hermes", "cron", "list"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            _print_cron_instructions(refresh_script, sync_script)
            return
    except (FileNotFoundError, subprocess.TimeoutExpired):
        _print_cron_instructions(refresh_script, sync_script)
        return

    # Check if refresh cron already exists
    result = subprocess.run(
        ["hermes", "cron", "list"],
        capture_output=True, text=True, timeout=10,
    )
    existing = result.stdout or ""

    if "whoop-token-refresh" in existing:
        print("  ✓ Token refresh cron already exists, skipping.")
    else:
        print("  Setting up token refresh cron (every 55 minutes)...")
        result = subprocess.run(
            [
                "hermes", "cron", "create",
                "*/55 * * * *",
                "--name", "whoop-token-refresh",
                "--no-agent",
                "--script", "whoop_token_refresh.py",
            ],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            print("  ✓ Token refresh cron created.")
        else:
            print(f"  ⚠ Failed to create token refresh cron: {result.stderr.strip()}")
            _print_cron_instructions(refresh_script, sync_script)

    if "whoop-daily-pull" in existing:
        print("  ✓ Daily data pull cron already exists, skipping.")
    else:
        print("  Setting up daily data pull cron (7:00 AM local)...")
        result = subprocess.run(
            [
                "hermes", "cron", "create",
                "0 7 * * *",
                "--name", "whoop-daily-pull",
                "--no-agent",
                "--script", "whoop_daily_pull.py",
            ],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            print("  ✓ Daily data pull cron created.")
        else:
            print(f"  ⚠ Failed to create daily pull cron: {result.stderr.strip()}")
            _print_cron_instructions(refresh_script, sync_script)


def _print_cron_instructions(refresh_script: Path, sync_script: Path) -> None:
    """Print manual cron setup instructions when Hermes cron is unavailable."""
    print("  To set up scheduled sync, add to your crontab (crontab -e):")
    print(f"    */55 * * * * cd {sync_script.parent} && python3 {refresh_script}")
    print(f"    0 7 * * * cd {sync_script.parent} && python3 {sync_script} pull")
    print()
    print("  Or with Hermes cron:")
    print(f"    hermes cron create \"*/55 * * * *\" --name whoop-token-refresh --no-agent --script {refresh_script}")
    print(f"    hermes cron create \"0 7 * * *\" --name whoop-daily-pull \"Pull Whoop data\" --workdir {sync_script.parent}")


def cmd_pull(args: argparse.Namespace) -> None:
    """Pull data from all Whoop endpoints and save to disk."""
    client = WhoopClient(data_dir=args.data_dir)
    results = client.pull_all(date=args.date)

    date_str = args.date or (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    output_dir = client.save_data(results, date=date_str)

    total = sum(len(v) for v in results.values())
    print(f"\nPulled {total} total records to {output_dir}")


def cmd_refresh(args: argparse.Namespace) -> None:
    """Manually refresh the Whoop access token."""
    client = WhoopClient(data_dir=args.data_dir)
    try:
        client._refresh_if_needed()
        print("Token refreshed successfully.")
    except Exception as e:
        print(f"Token refresh failed: {e}")
        print("Run `whoop_sync.py setup` to re-authenticate.")
        sys.exit(1)


def cmd_status(args: argparse.Namespace) -> None:
    """Show authentication status and configuration info."""
    print("Whoop API Status")
    print("=" * 40)

    # Check credentials
    creds = load_client_credentials()
    if creds:
        print(f"  Credentials: ✓ (client_id: {creds['client_id'][:8]}...)")
    else:
        print("  Credentials: ✗ Not found. Run `setup` first.")
        return

    # Check tokens
    tokens = load_tokens()
    if tokens:
        expires_at = tokens.get("expires_at", 0)
        remaining = expires_at - time.time()
        if remaining > 0:
            hours = int(remaining // 3600)
            mins = int((remaining % 3600) // 60)
            print(f"  Auth tokens: ✓ (expires in {hours}h {mins}m)")
        else:
            print("  Auth tokens: ✗ Expired. Run `setup` to re-authenticate.")
    else:
        print("  Auth tokens: ✗ Not found. Run `setup` to authenticate.")

    # Check cron
    try:
        result = subprocess.run(
            ["hermes", "cron", "list"],
            capture_output=True, text=True, timeout=10,
        )
        if "whoop-token-refresh" in (result.stdout or ""):
            print("  Token refresh cron: ✓")
        else:
            print("  Token refresh cron: ✗ Not set up. Run `setup` to create.")
        if "whoop-daily-pull" in (result.stdout or ""):
            print("  Daily pull cron: ✓")
        else:
            print("  Daily pull cron: ✗ Not set up. Run `setup` to create.")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  Cron: Unknown (hermes CLI not available)")

    print()
    print(f"  Privacy policy: {PRIVACY_POLICY_URL}")
    print(f"  Data directory: {args.data_dir or './whoop_data'}")


def cmd_daemon(args: argparse.Namespace) -> None:
    """Run continuous sync on an interval."""
    interval = args.interval
    print(f"Starting Whoop daemon (interval: {interval}s)")
    print(f"Data directory: {args.data_dir}")
    print("Press Ctrl+C to stop.\n")

    running = True

    def signal_handler(signum, frame):
        nonlocal running
        print("\nShutting down gracefully...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    while running:
        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            print(f"[{timestamp}] Starting pull...")
            client = WhoopClient(data_dir=args.data_dir)
            results = client.pull_all(date=args.date)

            date_str = args.date or (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
            output_dir = client.save_data(results, date=date_str)

            total = sum(len(v) for v in results.values())
            print(f"[{timestamp}] Pulled {total} records to {output_dir}")
        except Exception as e:
            print(f"[{timestamp}] Error: {e}")

        for _ in range(int(max(1, interval))):
            if not running:
                break
            time.sleep(1)

    print("Daemon stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Whoop API sync tool — pull fitness data from Whoop via OAuth2.",
    )
    parser.add_argument(
        "command",
        choices=["setup", "pull", "refresh", "daemon", "status"],
        help="Command to run",
    )
    parser.add_argument(
        "--interval", type=int, default=1800,
        help="Seconds between pulls in daemon mode (default: 1800)",
    )
    parser.add_argument(
        "--data-dir", type=str, default="whoop_data",
        help="Output directory for pulled data (default: whoop_data)",
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Pull specific date YYYY-MM-DD (default: yesterday)",
    )

    args = parser.parse_args()

    commands = {
        "setup": cmd_setup,
        "pull": cmd_pull,
        "refresh": cmd_refresh,
        "daemon": cmd_daemon,
        "status": cmd_status,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()