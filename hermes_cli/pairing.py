"""
CLI commands for the DM pairing system.

Usage:
    hermes pairing list              # Show all pending + approved users
    hermes pairing approve <platform> <code-or-target>  # Approve a pairing code or Kasia contact
    hermes pairing revoke <platform> <user_id-or-target> # Revoke user access
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

    generic_pending = [p for p in pending if p.get("platform") != "kasia"]
    kasia_pending = [p for p in pending if p.get("platform") == "kasia"]

    if generic_pending:
        print(f"\n  Pending Pairing Requests ({len(generic_pending)}):")
        print(f"  {'Platform':<12} {'Code':<10} {'User ID':<20} {'Name':<20} {'Age'}")
        print(f"  {'--------':<12} {'----':<10} {'-------':<20} {'----':<20} {'---'}")
        for p in generic_pending:
            print(
                f"  {p['platform']:<12} {p['code']:<10} {p['user_id']:<20} "
                f"{p.get('user_name', ''):<20} {p['age_minutes']}m ago"
            )
    elif pending:
        print("\n  No pending pairing-code requests.")

    if kasia_pending:
        print(f"\n  Pending Kasia Contacts ({len(kasia_pending)}):")
        print(f"  {'Target':<36} {'Address':<40} {'Age'}")
        print(f"  {'------':<36} {'-------':<40} {'---'}")
        for p in kasia_pending:
            print(
                f"  {_target_label(p):<36} "
                f"{(p.get('canonical_address') or p['user_id']):<40} "
                f"{p['age_minutes']}m ago"
            )
    else:
        print("\n  No pending pairing requests.")

    generic_approved = [a for a in approved if a.get("platform") != "kasia"]
    kasia_approved = [a for a in approved if a.get("platform") == "kasia"]

    if generic_approved:
        print(f"\n  Approved Users ({len(generic_approved)}):")
        print(f"  {'Platform':<12} {'User ID':<20} {'Name':<20}")
        print(f"  {'--------':<12} {'-------':<20} {'----':<20}")
        for a in generic_approved:
            print(f"  {a['platform']:<12} {a['user_id']:<20} {a.get('user_name', ''):<20}")
    elif approved:
        print("\n  No approved non-Kasia users.")

    if kasia_approved:
        print(f"\n  Approved Kasia Contacts ({len(kasia_approved)}):")
        print(f"  {'Target':<36} {'Address':<40}")
        print(f"  {'------':<36} {'-------':<40}")
        for a in kasia_approved:
            print(
                f"  {_target_label(a):<36} "
                f"{(a.get('canonical_address') or a['user_id']):<40}"
            )
    else:
        print("\n  No approved users.")

    print()


def _cmd_approve(store, platform: str, code: str):
    """Approve a pairing code or Kasia contact."""
    platform = platform.lower().strip()
    code = code.strip()

    if platform == "kasia":
        result = store.approve_identity(platform, code)
        if result:
            print(
                f"\n  Approved Kasia contact {_target_label(result)} "
                f"({result.get('canonical_address') or result['user_id']}).\n"
            )
            activation = _complete_live_kasia_approval(result)
            activation_status = str(activation.get("status") or "").strip().lower()
            if activation_status in {"responded", "initiated", "already_active", "pending"}:
                print(
                    "  Live Kasia transport is ready now.\n"
                    if activation_status != "pending"
                    else "  Live Kasia transport is in progress now.\n"
                )
            elif activation_status == "bridge_unavailable":
                print(
                    "  Approval is saved. The Kasia bridge is not reachable right now, "
                    "so transport will complete when Hermes reconnects or the peer retries.\n"
                )
            elif activation_status == "failed":
                detail = activation.get("initiate_error") or activation.get("respond_error") or "unknown error"
                print(
                    "  Approval is saved, but Hermes could not complete the live Kasia handshake: "
                    f"{detail}\n"
                )
        else:
            print(f"\n  Kasia target '{code}' was not found or could not be resolved.\n")
        return

    result = store.approve_code(platform, code.upper())
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

    if platform == "kasia":
        if store.revoke(platform, user_id):
            print(f"\n  Revoked Kasia contact {user_id}.\n")
        else:
            print(f"\n  Kasia contact '{user_id}' was not found in the approved list.\n")
        return

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


def _target_label(entry: dict) -> str:
    return entry.get("kns_name") or entry.get("display_name") or entry.get("user_name") or entry.get("user_id", "")


def _complete_live_kasia_approval(entry: dict) -> dict:
    from hermes_cli.kasia import complete_kasia_contact_approval

    return complete_kasia_contact_approval(
        entry.get("canonical_address") or entry.get("user_id") or entry.get("original_target") or "",
        display_name=entry.get("display_name") or entry.get("user_name"),
    )
