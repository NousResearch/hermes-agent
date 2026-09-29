"""``hermes share`` — connection codes and paired devices, straight from the share store.

Works whether or not the backend is running: codes and devices live in
``<home>/tailcat/``, which the running share listener reads on every request.
"""

from __future__ import annotations

import sys
import time

from hermes_cli import tailcat_share_store as store


def _ago(ts) -> str:
    if not ts:
        return "never"
    seconds = max(0, int(time.time() - float(ts)))
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            return f"{seconds // size}{unit} ago"
    return "just now"


def _print_devices() -> None:
    devices = store.load_devices()
    if not devices:
        print("No paired devices.")
        return
    width = max(len(device["name"]) for device in devices)
    for device in devices:
        print(f"  {device['id']}  {device['name']:<{width}}  paired {_ago(device.get('created_at'))}, "
              f"last seen {_ago(device.get('last_seen_at'))}")


def _status(_args) -> int:
    state = store.load_share_state()
    address = state.get("address", "")
    if address:
        print(f"Shared over tailcat: address {store.address_fingerprint(address)}, port {state.get('port')}")
    else:
        print("Not shared yet. Start with: hermes serve --share tailcat")
    _print_devices()
    return 0


def _code(_args) -> int:
    state = store.load_share_state()
    if not state.get("address") or not state.get("port"):
        print("Sharing has not started on this profile. Run: hermes serve --share tailcat", file=sys.stderr)
        return 1
    code = store.mint_code(state["address"], int(state["port"]))
    print(code.render())
    print(f"\nPaste it into Hermes Desktop:\n  Settings → Gateways → Add connection → Tailcat\n"
          f"It works once and expires in {store.CODE_TTL_S // 60} minutes.", file=sys.stderr)
    return 0


def _devices(_args) -> int:
    _print_devices()
    return 0


def _revoke(args) -> int:
    if store.revoke_device(args.device_id):
        print(f"Revoked {args.device_id}. Its connections close within seconds.")
        return 0
    print(f"No paired device {args.device_id}.", file=sys.stderr)
    return 1


def _reset(args) -> int:
    if not args.yes:
        reply = input("This gives the share a new address and unpairs every device. Continue? [y/N] ")
        if reply.strip().lower() not in ("y", "yes"):
            return 1
    store.forget_identity()
    print("Share identity reset. Restart `hermes serve --share tailcat` to publish the new address.")
    return 0


_ACTIONS = {"status": _status, "code": _code, "devices": _devices, "revoke": _revoke, "reset": _reset}


def share_command(args) -> int:
    return _ACTIONS[getattr(args, "share_action", None) or "status"](args)
