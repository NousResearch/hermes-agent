"""CLI command handler for marker (memory locations)."""

import json
import sys

from hermes_state import SessionDB, get_last_init_error
from agent.memory_locations import MemoryLocationStore


def cmd_marker(args):
    """Manage memory locations (markers)."""
    try:
        db = SessionDB()
    except Exception:
        err = get_last_init_error() or "unknown error"
        print(f"Session database not available: {err}")
        raise SystemExit(1)

    store = MemoryLocationStore(session_db=db)

    action = getattr(args, "marker_action", None)

    if action == "create":
        # For CLI "create", we don't have a current session ID from the TUI.
        # We can either require --session or create a persistent marker.
        session_id = getattr(args, "session", None)
        tags = args.tags.split(",") if args.tags else None
        
        loc_id = store.create(
            session_id=session_id,
            label=args.label,
            time_type="point",
            tags=tags,
            is_persistent=bool(getattr(args, "persistent", False)),
        )
        print(f"Created memory location {loc_id}: '{args.label}'")
        
    elif action == "list":
        locations = store.list(
            session_id=getattr(args, "session", None),
            persistent_only=bool(getattr(args, "persistent", False)),
        )
        if getattr(args, "json", False):
            print(json.dumps(locations, indent=2))
        else:
            if not locations:
                print("No memory locations found.")
                return
            print(f"{'ID':<5} | {'Persistent':<10} | {'Label':<30} | {'Session'}")
            print("-" * 70)
            for loc in locations:
                is_p = "Yes" if loc.get("is_persistent") else "No"
                label = loc.get("label", "")
                if len(label) > 28:
                    label = label[:28] + "..."
                sid = loc.get("session_id", "") or "(global)"
                print(f"{loc.get('id'):<5} | {is_p:<10} | {label:<30} | {sid}")
                
    elif action == "goto":
        loc = store.get(args.location_id)
        if not loc:
            print(f"Memory location {args.location_id} not found.")
            raise SystemExit(1)
        
        # Resolve anchor to get the message ID
        session_id = loc.get("session_id")
        anchor_guid = loc.get("anchor_guid")
        
        if anchor_guid and session_id:
            msg_id = store.resolve_anchor(anchor_guid, session_id)
            if msg_id:
                print(f"Jumping to location {args.location_id} (Message ID: {msg_id})")
                print(f"Session: {session_id}")
            else:
                print(f"Warning: Anchor for location {args.location_id} is orphaned (message no longer exists).")
        else:
            print(f"Location {args.location_id} does not have a message anchor.")
            
    elif action == "delete":
        if store.delete(args.location_id):
            print(f"Deleted memory location {args.location_id}.")
        else:
            print(f"Memory location {args.location_id} not found.")
            raise SystemExit(1)
    else:
        print("Use 'hermes marker --help' for usage information.")
