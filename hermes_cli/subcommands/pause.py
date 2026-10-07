"""``hermes pause`` / ``hermes resume`` — the global emergency stop.

``pause`` writes the ESTOP sentinel at ``$HERMES_HOME/ESTOP``; cron, kanban and new gateway
turns halt on their next check (in-flight work is never killed). ``resume`` removes it and
operation resumes on the next tick — no restart. Ported from gastownhall/gastown estop.go (MIT).

Single-user mode: ``--allow-user`` keeps the operator working THROUGH the pause (its
authenticated id); ``--allow-profile`` NARROWS that to the named serving profiles (it never
admits anyone on its own); ``--ttl`` arms the deadman that lifts the pause if the window job
dies before its release. Omitting the allowlist/ttl on a re-arm KEEPS the standing sentinel's,
so ``hermes pause`` cannot silently strip authority it did not set.
"""

from __future__ import annotations

import argparse


def cmd_pause(args: argparse.Namespace) -> int:
    """Engage the global emergency stop."""
    from agent.estop import engage, get_state, is_engaged, parse_duration

    reason = getattr(args, "reason", None)
    allow_user = list(getattr(args, "allow_user", None) or [])
    allow_profile = list(getattr(args, "allow_profile", None) or [])
    if allow_profile and not allow_user:
        print(
            "⛔ --allow-profile requires --allow-user — a serving profile NARROWS the "
            "allowlist and cannot admit anyone on its own. NOT pausing."
        )
        return 2
    # None (not an empty dict) says "not supplied": estop.engage then KEEPS the standing
    # allowlist, so a no-flag re-arm cannot wipe authority the previous arm granted.
    allow = (
        {"user_ids": allow_user, "profiles": allow_profile} if (allow_user or allow_profile) else None
    )
    ttl = getattr(args, "ttl", None)
    if ttl and parse_duration(ttl) is None:
        print(f"⛔ Invalid --ttl {ttl!r} — use a duration such as 45m, 90m or 2h. NOT pausing.")
        return 2
    already = is_engaged()
    path = engage(reason=reason, allow=allow, ttl=ttl)
    state = get_state() or {}
    verb = "Still paused" if already else "Hermes paused"
    detail = f" — reason: {state['reason']}" if state.get("reason") else ""
    print(f"⏸️  {verb}{detail}")
    print(f"    sentinel: {path}")
    allowed = state.get("allow") or {}
    if allowed.get("user_ids") or allowed.get("profiles"):
        who = ", ".join(
            list(allowed.get("user_ids") or [])
            + [f"profile:{name}" for name in (allowed.get("profiles") or [])])
        print(f"    allowlist: {who} (their new turns are served through the pause)")
    if state.get("expires_at"):
        print(f"    deadman: auto-resumes at {state['expires_at']} (--ttl {ttl})")
    elif ttl:
        print(
            f"    ⚠️  --ttl {ttl} was NOT applied (out of range) — this pause has NO "
            "auto-resume. Nothing will lift it on its own: run `hermes resume`.")
    print(
        "    Cron dispatch, kanban dispatch, and new gateway turns are on hold.\n"
        "    In-flight work keeps running. Run `hermes resume` to lift the pause.")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    """Disengage the global emergency stop."""
    from agent.estop import disengage, sentinel_path

    if disengage():
        print("▶️  Hermes resumed — dispatch picks up on the next tick.")
    else:
        print(f"Hermes is not paused (no sentinel at {sentinel_path()}).")
    return 0


def build_pause_parser(subparsers) -> None:
    """Attach the ``pause`` and ``resume`` subcommands to ``subparsers``."""
    pause_parser = subparsers.add_parser(
        "pause", help="Emergency stop: pause cron/kanban dispatch and new gateway turns",
        description="Engage the global emergency stop. Halts NEW work only — cron "
            "dispatch, kanban dispatch, and new gateway turns — until "
            "`hermes resume`. In-flight work is never killed. Use --allow-user "
            "to keep working through the pause yourself.")
    pause_parser.add_argument(
        "--reason", default=None, help="Optional reason stored in the sentinel and shown to users")
    pause_parser.add_argument(
        "--allow-user", action="append", default=None, metavar="ID",
        help="Authenticated user id exempt from the pause (repeatable) — normally the operator's")
    pause_parser.add_argument(
        "--allow-profile", action="append", default=None, metavar="PROFILE",
        help="Narrow --allow-user to these serving profiles (repeatable); never admits anyone alone")
    pause_parser.add_argument(
        "--ttl", default=None, metavar="DUR",
        help="Deadman: auto-resume after this long (e.g. 45m, 90m, 2h, or seconds). "
            "Bounds a window job that dies between arm and release.")
    pause_parser.set_defaults(func=cmd_pause)

    resume_parser = subparsers.add_parser(
        "resume", help="Lift the emergency stop set by `hermes pause`",
        description="Remove the ESTOP sentinel; dispatch resumes on the next tick.")
    resume_parser.set_defaults(func=cmd_resume)
