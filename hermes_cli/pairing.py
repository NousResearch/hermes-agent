"""
CLI commands for the DM pairing system.

Usage:
    hermes pairing list              # Show all pending + approved users
    hermes pairing approve <platform> <code>  # Approve a pairing code
    hermes pairing revoke <platform> <user_id> # Revoke user access
    hermes pairing clear-pending     # Clear all expired/pending codes
"""


def pairing_command(args):
    """Handle hermes pairing subcommands."""
    from gateway.pairing import PairingStore

    store = PairingStore()
    action = getattr(args, "pairing_action", None)

    if action == "list":
        _cmd_list(store)
    elif action == "approve":
        _cmd_approve(store, args.platform, args.code)
    elif action == "revoke":
        _cmd_revoke(store, args.platform, args.user_id)
    elif action == "clear-pending":
        _cmd_clear_pending(store)
    else:
        print("Usage: hermes pairing {list|approve|revoke|clear-pending}")
        print("Run 'hermes pairing --help' for details.")


def _cmd_list(store):
    """List all pending and approved users."""
    pending = store.list_pending()
    approved = store.list_approved()

    if not pending and not approved:
        print("No pairing data found. No one has tried to pair yet~")
        return

    if pending:
        print(f"\n  Pending Pairing Requests ({len(pending)}):")
        print(f"  {'Platform':<12} {'Code':<10} {'User ID':<20} {'Name':<20} {'Age'}")
        print(f"  {'--------':<12} {'----':<10} {'-------':<20} {'----':<20} {'---'}")
        for p in pending:
            print(
                f"  {p['platform']:<12} {p['code']:<10} {p['user_id']:<20} "
                f"{p.get('user_name', ''):<20} {p['age_minutes']}m ago"
            )
    else:
        print("\n  No pending pairing requests.")

    if approved:
        print(f"\n  Approved Users ({len(approved)}):")
        print(f"  {'Platform':<12} {'User ID':<20} {'Name':<20}")
        print(f"  {'--------':<12} {'-------':<20} {'----':<20}")
        for a in approved:
            print(f"  {a['platform']:<12} {a['user_id']:<20} {a.get('user_name', ''):<20}")
    else:
        print("\n  No approved users.")

    print()


def _cmd_approve(store, platform: str, code: str):
    """Approve a pairing code."""
    platform = platform.lower().strip()
    code = code.upper().strip()

    result = store.approve_code(platform, code)
    if result:
        uid = result["user_id"]
        name = result.get("user_name", "")
        display = f"{name} ({uid})" if name else uid
        print(f"\n  Approved! User {display} on {platform} can now use the bot~")
        print(f"  They'll be recognized automatically on their next message.\n")
    else:
        print(f"\n  Code '{code}' not found or expired for platform '{platform}'.")
        print(f"  Run 'hermes pairing list' to see pending codes.\n")


def _cmd_revoke(store, platform: str, user_id: str):
    """Revoke a user's access."""
    platform = platform.lower().strip()

    if store.revoke(platform, user_id):
        print(f"\n  Revoked access for user {user_id} on {platform}.\n")
    else:
        print(f"\n  User {user_id} not found in approved list for {platform}.\n")


def _cmd_clear_pending(store):
    """Clear all pending pairing codes."""
    count = store.clear_pending()
    if count:
        print(f"\n  Cleared {count} pending pairing request(s).\n")
    else:
        print("\n  No pending requests to clear.\n")
