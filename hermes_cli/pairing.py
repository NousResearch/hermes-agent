"""CLI commands for the DM pairing system."""

def pairing_command(args):
    """Handle hermes pairing subcommands."""
    from gateway.pairing import PairingStore

    store = PairingStore()
    handlers = {
        "list": lambda: _cmd_list(store),
        "approve": lambda: _cmd_approve(store, args.platform, args.code),
        "revoke": lambda: _cmd_revoke(store, args.platform, args.user_id),
        "clear-pending": lambda: _cmd_clear_pending(store),
    }
    handler = handlers.get(getattr(args, "pairing_action", None))
    if handler is None:
        print("Usage: hermes pairing {list|approve|revoke|clear-pending}")
        print("Run 'hermes pairing --help' for details.")
    else:
        handler()


def _cmd_list(store):
    """List all pending and approved users."""
    pending = store.list_pending()
    approved = store.list_approved()
    if not pending and not approved:
        print("No pairing data found. No one has tried to pair yet~")
        return

    if pending:
        print(f"\n  Pending Pairing Requests ({len(pending)}):")
        print(f"  {'Platform':<12} {'Request ID':<18} {'User ID':<20} {'Name':<20} {'Age'}")
        print(f"  {'--------':<12} {'----------':<18} {'-------':<20} {'----':<20} {'---'}")
        for p in pending:
            print(
                f"  {p['platform']:<12} {(p.get('request_id') or '-'):<18} {p['user_id']:<20} "
                f"{(p.get('user_name') or ''):<20} {p['age_minutes']}m ago"
            )
        print("\n  Approve with: hermes pairing approve <platform> <request-id>")
        print("  The code the bot DM'd the user also works if they relay it.")
    else:
        print("\n  No pending pairing requests.")

    if approved:
        print(f"\n  Approved Users ({len(approved)}):")
        print(f"  {'Platform':<12} {'User ID':<20} {'Name':<20}")
        print(f"  {'--------':<12} {'-------':<20} {'----':<20}")
        for a in approved:
            print(f"  {a['platform']:<12} {a['user_id']:<20} {(a.get('user_name') or ''):<20}")
    else:
        print("\n  No approved users.")

    print()


def _cmd_approve(store, platform: str, code: str):
    """Approve a pairing request id (from ``pairing list``) or a DM'd code."""
    platform = platform.lower().strip()
    code = code.strip()

    if store.looks_like_request_id(code):
        result = store.approve_request(platform, code)
    else:
        result = store.approve_code(platform, code.upper())
    if result:
        uid, name = result["user_id"], result.get("user_name") or ""
        display = f"{name} ({uid})" if name else uid
        print(f"\n  Approved! User {display} on {platform} can now use the bot~")
        print("  They'll be recognized automatically on their next message.\n")
    elif store._is_locked_out(platform):
        # approve_code returns None for both invalid codes and lockout — say which.
        # Tell the operator it's lockout so they don't chase a "wrong code" rabbit hole (#10195).
        import time as _time
        lockout_until = store._load_json(store._rate_limit_path()).get(f"_lockout:{platform}", 0)
        mins = max(0, int(lockout_until - _time.time())) // 60
        print(f"\n  Platform '{platform}' is locked out after too many failed approval attempts.")
        print(f"  Lockout clears in ~{mins} minute(s).")
        print(f"  To reset sooner, delete the '_lockout:{platform}' entry from ~/.hermes/platforms/pairing/_rate_limits.json\n")
    else:
        print(f"\n  Pairing request or code '{code}' not found or expired for platform '{platform}'.")
        print("  Run 'hermes pairing list' to see pending requests.\n")


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
