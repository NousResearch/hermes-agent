"""Explicit operator acknowledgment of verified gateway-restart obligations."""


def handle_gateway_ack_args(args, project_root) -> bool:
    """Handle a single requested acknowledgment under the update lock."""
    protected = getattr(args, "acknowledge_protected_gateway", None)
    receipt = getattr(args, "acknowledge_gateway_receipt", None)
    if not (protected or receipt):
        return False
    if protected and receipt:
        raise SystemExit("Acknowledge only one gateway debt at a time.")
    if getattr(args, "install_id", False) or getattr(args, "set_channel", None) is not None:
        raise SystemExit("Gateway acknowledgment cannot be combined with update metadata options.")
    from hermes_cli.update_host_obligation import acknowledge_protected_gateway, acknowledge_receipt_gateway
    from hermes_cli.update_lock import UpdateLock, describe_holder
    lock = UpdateLock(install_root=project_root)
    if not lock.acquire():
        raise SystemExit(describe_holder(lock.holder))
    try:
        acknowledge = acknowledge_protected_gateway if protected else acknowledge_receipt_gateway
        try:
            applied = acknowledge(protected or receipt)
        except (OSError, ValueError, TypeError) as exc:
            raise SystemExit(f"Could not record acknowledgment ({exc}); the restart obligation remains outstanding.") from exc
        if not applied:
            raise SystemExit("No matching gateway debt; the token may be stale or the record unavailable.")
        print("Acknowledged the selected entry; unrelated or unverified obligations remain recorded.")
    finally:
        lock.release()
    return True
