"""``hermes move``: carry one profile to another Hermes (a Hermes Cloud agent, a new machine).

``hermes move export``      write a move bundle of the active profile
``hermes move receive FILE`` preview, then merge (or ``--replace``) a bundle into the active profile
``hermes move undo ID``     restore what a received move changed
``hermes move list``        moves received by this profile

Desktop's "Move to Hermes Cloud" drives the same code over ``/api/move/*``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# The parts a move carries (``move_receive.STEPS``), spelled here so building the parser stays
# import-free.
STEP_NAMES = ("config", "soul", "memory", "skills", "cron", "sessions")


def build_move_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "move", help="Move this profile to another Hermes (Hermes Cloud, a new computer)",
        description="Export the active profile as a move bundle, or receive one into it. A move never "
                    "carries API keys or bot tokens. By default it merges, keeping what the receiving "
                    "agent already has; --replace lets the bundle win for the parts you name, and undo "
                    "restores either.")
    sub = parser.add_subparsers(dest="move_command")

    export = sub.add_parser("export", help="Write a move bundle of the active profile")
    export.add_argument("-o", "--output", help="Output path (default: ./hermes-move-<profile>.zip)")
    export.add_argument("--no-history", action="store_true", help="Leave chat history behind")
    export.add_argument("--history-limit", type=int, default=200, help="Most recent conversations to carry")
    export.add_argument("--dry-run", action="store_true", help="Only list what would be carried")
    export.add_argument("--json", action="store_true", help="Print the inventory as JSON")
    export.set_defaults(func=_cmd_export)

    receive = sub.add_parser("receive", help="Preview, then merge a move bundle into the active profile")
    receive.add_argument("bundle", help="Path to a move bundle")
    receive.add_argument("--replace", metavar="PARTS", default="",
                         help="Comma-separated parts where the bundle wins instead of merging: "
                              f"{', '.join(STEP_NAMES)}, or 'all'. Replaced conversations are archived, "
                              "not deleted, and `hermes move undo` restores everything")
    receive.add_argument("--only", metavar="PARTS", default="",
                         help="Comma-separated parts to apply (default: all)")
    receive.add_argument("-y", "--yes", action="store_true", help="Apply without asking")
    receive.add_argument("--json", action="store_true", help="Print the result as JSON")
    receive.set_defaults(func=_cmd_receive)

    undo = sub.add_parser("undo", help="Restore what a received move changed")
    undo.add_argument("move_id")
    undo.set_defaults(func=_cmd_undo)

    listing = sub.add_parser("list", help="Moves received by this profile")
    listing.set_defaults(func=_cmd_list)
    parser.set_defaults(func=lambda args: parser.print_help())


def _home_and_profile() -> tuple[Path, str]:
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    profile = home.name if home.parent.name == "profiles" else "default"
    return home, profile


def _print_inventory(inv: dict) -> None:
    print(f"  Skills: {len(inv['skills'])}"
          + (f" ({len(inv['skills_skipped'])} too large, left behind)" if inv["skills_skipped"] else ""))
    mem = inv.get("memory_entries") or {}
    print(f"  Memory: {mem.get('MEMORY.md', 0)} notes, {mem.get('USER.md', 0)} about you")
    print(f"  Persona (SOUL.md): {'yes' if inv['has_soul'] else 'no'}")
    print(f"  Scheduled jobs: {len(inv['cron_jobs'])} (arrive paused)")
    print(f"  Conversations: {inv['sessions']}"
          + (f" ({inv['sessions_skipped']} too large, left behind)" if inv["sessions_skipped"] else ""))
    local_mcp = [m["name"] for m in inv["mcp_servers"] if not m["portable"]]
    if local_mcp:
        print(f"  MCP servers that only run on this computer: {', '.join(local_mcp)}")
    if inv["keys_to_reenter"]:
        print("  Keys to enter again on the other side (never carried):")
        for key in inv["keys_to_reenter"]:
            print(f"    {key['key']}  ({key['category']})")


def _cmd_export(args) -> int:
    from hermes_cli.move_bundle import build_bundle
    home, profile = _home_and_profile()
    out = Path(args.output or f"hermes-move-{profile}.zip").expanduser().resolve()
    report = build_bundle(home, profile, out, include_history=not args.no_history,
                          history_limit=args.history_limit, dry_run=args.dry_run)
    inv = report.as_dict()
    if args.json:
        print(json.dumps(inv, indent=2))
        return 0
    print(("Would carry" if args.dry_run else f"Wrote {out} ({report.bytes:,} bytes)") + f" from {profile}:")
    _print_inventory(inv)
    return 0


def _sessions_writer(home: Path):
    """Conversation writes for a CLI receive/undo: through the live gateway when one owns the
    store, else straight into the store under the owner lock (a stopped install, a fresh one)."""
    from hermes_cli.gateway_runtime import discover_gateway_endpoint
    from hermes_cli.move_sessions import OfflineSessions, OwnerSessions, RevisionConflict

    if discover_gateway_endpoint(home, timeout=1.5).endpoint is None:
        return OfflineSessions(home)

    def mutate(session_id, request_id, operation, payload, revision, generation):
        from hermes_cli.gateway_client import GatewayClientError
        from hermes_cli.gateway_session_store import session_store
        params = dict(session_id=session_id, request_id=request_id, expected_revision=revision,
                      operation=operation, payload=payload)
        if generation is not None:
            params["expected_generation"] = generation
        try:
            return session_store().call("session.mutate", **params)
        except GatewayClientError as exc:
            if "revision_conflict" in str(exc):
                raise RevisionConflict(str(exc)) from None
            raise

    return OwnerSessions(home, mutate)


def _parts(raw: str) -> set[str]:
    parts = {p.strip() for p in (raw or "").split(",") if p.strip()}
    if "all" in parts:
        return set(STEP_NAMES)
    unknown = parts - set(STEP_NAMES)
    if unknown:
        raise SystemExit(f"Unknown part(s): {', '.join(sorted(unknown))}. Choose from {', '.join(STEP_NAMES)}.")
    return parts


def _print_replace_warning(preview: dict, replace: set[str]) -> None:
    target = preview.get("target") or {}
    lines = {
        "memory": "Memory notes on this agent are overwritten by the incoming ones.",
        "soul": "The persona (SOUL.md) is overwritten.",
        "skills": f"{len(preview['skills']['keep_existing'])} same-named skill(s) are swapped for the incoming copy.",
        "cron": f"All {target.get('cron_jobs', 0)} scheduled job(s) here are removed; the incoming ones arrive paused.",
        "sessions": f"All {target.get('conversations', 0)} visible conversation(s) here are archived (kept, not deleted).",
        "config": "Settings the bundle carries replace this agent's (model, sign-ins and hosting stay).",
    }
    print("This REPLACES, on this agent:")
    for part in STEP_NAMES:
        if part in replace:
            print(f"  - {lines[part]}")
    print("Undo restores all of it: `hermes move undo <id>`.")


def _cmd_receive(args) -> int:
    from hermes_cli import move_receive
    home, _profile = _home_and_profile()
    path = Path(args.bundle).expanduser()
    data = path.read_bytes()
    state = move_receive.begin(home, size=len(data), sha256=hashlib.sha256(data).hexdigest())
    import base64
    step = move_receive.MAX_CHUNK_BYTES
    for offset in range(state["received"], len(data), step):
        state = move_receive.chunk(home, state["id"], offset=offset,
                                   data_b64=base64.b64encode(data[offset:offset + step]).decode())
    state = move_receive.finish(home, state["id"])
    preview = state["preview"]
    replace, only = _parts(args.replace), _parts(args.only) or None
    if not args.json:
        src = preview.get("source") or {}
        print(f"From {src.get('host', '?')} ({src.get('profile', '?')}):")
        print(f"  New skills: {len(preview['skills']['add'])}, same-named here: {len(preview['skills']['keep_existing'])}")
        print(f"  Scheduled jobs: {len(preview['cron']['add'])} (paused)")
        print(f"  Conversations: {preview['sessions']}")
        print(f"  Persona: {preview['soul']['action'].replace('_', ' ')}")
        if replace:
            _print_replace_warning(preview, replace & (only or set(STEP_NAMES)))
    if not args.yes:
        if input("Apply? [y/N] ").strip().lower() not in {"y", "yes"}:
            print(f"Not applied. Run `hermes move receive {path}` again to continue.")
            return 1
    state = move_receive.apply(home, state["id"], sessions=_sessions_writer(home), include=only, replace=replace)
    if args.json:
        print(json.dumps(state, indent=2))
    else:
        print(f"Move {state['id']}: {state['status']}. Undo with `hermes move undo {state['id']}`.")
    return 0 if state["status"] == "applied" else 1


def _cmd_undo(args) -> int:
    from hermes_cli import move_receive
    home, _ = _home_and_profile()
    state = move_receive.undo(home, args.move_id, sessions=_sessions_writer(home))
    print(f"Move {state['id']}: {state['status']}.")
    return 0


def _cmd_list(args) -> int:
    from hermes_cli import move_receive
    home, _ = _home_and_profile()
    for move in move_receive.list_moves(home):
        print(f"{move['id']}  {move['status']:<10} {(move.get('preview') or {}).get('source', {}).get('host', '')}")
    return 0
