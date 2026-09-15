"""CLI commands for the DM pairing system."""

from argparse import Namespace


def pairing_command(args):
    """Handle hermes pairing subcommands."""
    from gateway.pairing import PairingStore

    store = PairingStore()
    handlers = {
        "list": lambda: _cmd_list(store),
        "approve": lambda: _cmd_approve(
            store,
            args.platform,
            args.code,
            notify=getattr(args, "notify", False),
            admin=getattr(args, "admin", False),
        ),
        "revoke": lambda: _cmd_revoke(store, args.platform, args.user_id),
        "clear-pending": lambda: _cmd_clear_pending(store),
    }
    handler = handlers.get(getattr(args, "pairing_action", None))
    if handler is None:
        print("Usage: hermes pairing {list|approve|revoke|clear-pending}")
        print("Run 'hermes pairing --help' for details.")
        return 0
    else:
        return handler() or 0


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


def _admin_ids(raw) -> list[str]:
    """Normalize the supported config representations of ``allow_admin_from``."""
    if isinstance(raw, str):
        values = raw.split(",")
    elif isinstance(raw, (list, tuple, set, frozenset)):
        values = raw
    elif raw is None:
        values = ()
    else:
        values = (raw,)
    return [str(value).strip() for value in values if str(value).strip()]


def _grant_dm_admin(platform: str, user_id: str) -> bool:
    """Add one explicitly selected paired user to the platform's DM admin list."""
    from hermes_cli.config import read_raw_config, write_platform_config_field

    config = read_raw_config()
    platforms = config.get("platforms") if isinstance(config, dict) else None
    platform_config = platforms.get(platform) if isinstance(platforms, dict) else None
    existing = _admin_ids(platform_config.get("allow_admin_from") if isinstance(platform_config, dict) else None)
    if user_id in existing:
        return False
    write_platform_config_field(platform, "allow_admin_from", [*existing, user_id], raw=True)
    return True


def _has_dm_admin(platform: str) -> bool:
    """Whether the platform has opted in to direct-message slash-command gating."""
    from hermes_cli.config import read_raw_config

    config = read_raw_config()
    platforms = config.get("platforms") if isinstance(config, dict) else None
    platform_config = platforms.get(platform) if isinstance(platforms, dict) else None
    return bool(_admin_ids(platform_config.get("allow_admin_from") if isinstance(platform_config, dict) else None))


def _send_approval_receipt(platform: str, user_id: str) -> int:
    """Use the ``hermes send`` path so configured and live adapters share delivery behavior."""
    from hermes_cli.send_cmd import cmd_send

    args = Namespace(
        to=f"{platform}:{user_id}",
        message="Your pairing request has been approved. You can now use this bot.",
        file=None,
        subject=None,
        list_targets=False,
        json=False,
        quiet=True,
    )
    try:
        cmd_send(args)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:  # Delivery is best-effort; the durable grant already succeeded.
        print(f"  Receipt delivery failed: {exc}")
        return 1
    return 0


def _cmd_approve(store, platform: str, code: str, *, notify: bool = False, admin: bool = False) -> int:
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
        if admin:
            if _grant_dm_admin(platform, uid):
                print("  Added as a direct-message slash-command admin.")
                print("  This enables DM slash gating: other users keep only /help and /whoami.\n")
            else:
                print("  This user is already a direct-message slash-command admin.\n")
        elif not _has_dm_admin(platform):
            print("  To also make a paired user a DM slash-command admin, approve with --admin.")
            print("  Admin access is opt-in; group chats use group_allow_admin_from separately.\n")
        if notify:
            receipt_status = _send_approval_receipt(platform, uid)
            if receipt_status == 0:
                print("  Receipt delivered to the approved user.\n")
            else:
                print("  Receipt delivery failed; the pairing approval remains active.\n")
            return receipt_status
        return 0
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
    return 0


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
