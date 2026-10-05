"""Explicit content-free business validation signals; never permission to mutate."""
import json
import sqlite3


def cmd_attention(args):
    from hermes_cli.profiles import get_active_profile_name
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB
    from hermes_state_observations import validate_observation_ids

    try:
        validate_observation_ids([args.session_id])
        validate_observation_ids([args.turn_id])
        for value in (args.request_id, args.request_turn_id):
            if value is not None:
                validate_observation_ids([value])
    except ValueError:
        print(json.dumps({"ok": False, "error": "Invalid exact identifier"}))
        return 1
    path = get_hermes_home() / "state.db"
    profile = get_active_profile_name()
    profile = "default" if profile == "custom" else profile
    # First verify identity with an immutable open: failed declarations never create a store.
    try:
        with SessionDB(path, read_only=True) as db:
            row = db.read_session_observations([args.session_id], profile=profile)[0]
        if row["turn_id"] != args.turn_id or row["execution"] == "unknown":
            print(json.dumps({"ok": False, "error": "Unknown identity or stale turn"}))
            return 1
        with SessionDB(path) as db:
            if args.attention_action == "open":
                request = db.open_session_attention(args.session_id, args.turn_id, "validation")
                result = {"ok": bool(request), "request_id": request, "turn_id": args.turn_id}
            else:
                ok = bool(args.request_id) and db.resolve_session_attention(
                    args.session_id, args.turn_id, args.request_id, request_turn_id=args.request_turn_id)
                result = {"ok": bool(ok), "request_id": args.request_id, "turn_id": args.turn_id}
    except (sqlite3.DatabaseError, OSError, ValueError):
        result = {"ok": False, "error": "Observation storage unavailable"}
    print(json.dumps(result))
    return 0 if result["ok"] else 1
