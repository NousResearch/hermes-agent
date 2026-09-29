"""``hermes away-bridge`` — CLI for the Away-Mode Telegram Bridge.

Subcommands (design §8.1, §4.3, §4.7):
  arm        — arm the current session (dedupe: refresh on lineage match)
  disarm     — chat-side disarm: flip every lineage-matching armed row
  prepare    — create a prepared request bound to the arm-time key
  publish    — prepared → awaiting after the waiter is confirmed started
  cancel     — cancel a prepared/awaiting request
  claim      — chat-side claim of a pending request (first-answer-wins)
  fetch      — idempotent answer fetch (delivered flag)
  status     — dump broker rows for debugging
  reconcile  — reporter-only reconciliation report (§4.7)
  sweep      — rate-limited re-ask candidates + orphan-row GC (§4.4)
  wait       — the managed waiter process (one-line stdout envelope, §4.3)

The waiter prints exactly one line:
  AWAY-BRIDGE outcome=<OUTCOME> req_key=<full-id>
and never prints answer text (2,000-char completion cap, §4.3).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Optional

from tools.away_bridge_broker import (
    AlreadyAwaitingError,
    AwayBridgeBroker,
    BrokerError,
    DEFAULT_POLL_SECONDS,
    InvalidStateError,
    NotArmedError,
    UnknownRequestError,
    current_session_key,
    load_away_bridge_config,
)


def _broker(args) -> AwayBridgeBroker:
    return AwayBridgeBroker(
        db_path=getattr(args, "db_path", None) or None,
        busy_ms=getattr(args, "busy_ms", 5000),
    )


def _print_json(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# subcommand implementations
# ---------------------------------------------------------------------------

def cmd_arm(args) -> int:
    key = current_session_key(getattr(args, "session", None))
    if not key:
        print(
            "away-bridge arm: no session key (pass --session or run inside a session)",
            file=sys.stderr,
        )
        return 2
    result = _broker(args).arm(key, surface=getattr(args, "surface", "chat") or "chat")
    _print_json(result)
    return 0


def cmd_disarm(args) -> int:
    result = _broker(args).disarm_chat(getattr(args, "session", None))
    _print_json(result)
    return 0


def cmd_prepare(args) -> int:
    try:
        result = _broker(args).prepare(getattr(args, "session", None), args.question)
    except (NotArmedError, AlreadyAwaitingError) as exc:
        print("away-bridge prepare: %s" % exc, file=sys.stderr)
        return 3
    _print_json(result)
    return 0


def cmd_publish(args) -> int:
    broker = _broker(args)
    start_time = None
    if args.waiter_pid:
        start_time = broker.waiter_start_time_for(args.waiter_pid)
        if start_time is None:
            print(
                "away-bridge publish: could not read kernel start time for pid %s; "
                "refusing to publish without identity (§4.3)" % args.waiter_pid,
                file=sys.stderr,
            )
            return 4
    try:
        result = broker.publish(
            args.request, waiter_pid=args.waiter_pid, waiter_start_time=start_time
        )
    except (UnknownRequestError, InvalidStateError) as exc:
        print("away-bridge publish: %s" % exc, file=sys.stderr)
        return 3
    _print_json(result)
    return 0


def cmd_cancel(args) -> int:
    try:
        _broker(args).cancel(args.request)
    except (UnknownRequestError, InvalidStateError) as exc:
        print("away-bridge cancel: %s" % exc, file=sys.stderr)
        return 3
    _print_json({"req_key": args.request, "state": "cancelled"})
    return 0


def cmd_claim(args) -> int:
    try:
        result = _broker(args).claim(args.request, args.channel, args.answer)
    except BrokerError as exc:
        print("away-bridge claim: %s" % exc, file=sys.stderr)
        return 3
    _print_json(result)
    acked = False
    if result["outcome"] in ("lost", "reject"):
        acked = _broker(args).ack_duplicate(args.request, args.channel)
    _print_json({"ack_sent": acked})
    return 0


def cmd_fetch(args) -> int:
    try:
        result = _broker(args).fetch(args.request)
    except UnknownRequestError as exc:
        print("away-bridge fetch: %s" % exc, file=sys.stderr)
        return 3
    _print_json(result)
    return 0


def cmd_status(args) -> int:
    _print_json(_broker(args).snapshot())
    return 0


def cmd_reconcile(args) -> int:
    _print_json(_broker(args).reconcile())
    return 0


def cmd_sweep(args) -> int:
    broker = _broker(args)
    cfg = load_away_bridge_config()
    due = broker.sweep_candidates(float(cfg.get("sweep_window_hours", 1.0)))
    gc = broker.gc(float(cfg.get("retention_days", 30)))
    _print_json({"sweep_candidates": due, "gc": gc})
    return 0


def cmd_wait(args) -> int:
    """The managed waiter (§4.3). One poll per interval; one-line exit envelope."""
    cfg = load_away_bridge_config()
    try:
        poll = float(
            getattr(args, "poll_seconds", None)
            or cfg.get("waiter_poll_seconds")
            or DEFAULT_POLL_SECONDS
        )
    except (TypeError, ValueError):
        poll = DEFAULT_POLL_SECONDS
    broker = _broker(args)
    outcome = "ERROR"
    deadline = None
    if getattr(args, "max_wait_seconds", None):
        deadline = time.time() + float(args.max_wait_seconds)
    while True:
        try:
            result = broker.poll_request(args.request)
        except BrokerError as exc:
            outcome = "ERROR"
            if getattr(args, "debug", False):
                print("wait: %s" % exc, file=sys.stderr)
            break
        outcome = result.get("outcome", "ERROR")
        if outcome != "KEEP_WAITING":
            break
        if deadline is not None and time.time() > deadline:
            outcome = "ERROR"
            break
        time.sleep(poll)
    # Exactly one stdout line — banner-free, ANSI-free, no answer text (§4.3).
    sys.stdout.write("AWAY-BRIDGE outcome=%s req_key=%s\n" % (outcome, args.request))
    sys.stdout.flush()
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def register_away_bridge_subparser(subparsers) -> None:
    """Register the away-bridge subcommands onto a subparsers group.

    Callable two ways: (a) directly on a ``hermes away-bridge`` subparser
    object (has add_argument), or (b) onto a subparsers action from
    ``hermes_cli.main``'s parent ``away-bridge`` parser (no add_argument —
    the parent owns shared flags).
    """
    p = subparsers
    if hasattr(p, "add_argument"):
        p.add_argument("--db-path", dest="db_path", default=None)
        p.add_argument("--busy-ms", dest="busy_ms", type=int, default=5000)
    sub = p

    p_arm = sub.add_parser("arm", help="arm the current session")
    p_arm.add_argument("--session", default=None, help="session key (default: $HERMES_SESSION_KEY)")
    p_arm.add_argument("--surface", default="chat", choices=["chat", "telegram"])
    p_arm.set_defaults(func=cmd_arm)

    p_disarm = sub.add_parser("disarm", help="chat-side disarm (flips every lineage-matching row)")
    p_disarm.add_argument("--session", default=None)
    p_disarm.set_defaults(func=cmd_disarm)

    p_prepare = sub.add_parser("prepare", help="create a prepared request")
    p_prepare.add_argument("--session", default=None)
    p_prepare.add_argument("--question", required=True)
    p_prepare.set_defaults(func=cmd_prepare)

    p_publish = sub.add_parser("publish", help="prepared -> awaiting (after waiter start)")
    p_publish.add_argument("--request", required=True)
    p_publish.add_argument("--waiter-pid", dest="waiter_pid", type=int, default=None)
    p_publish.set_defaults(func=cmd_publish)

    p_cancel = sub.add_parser("cancel", help="cancel a prepared/awaiting request")
    p_cancel.add_argument("--request", required=True)
    p_cancel.set_defaults(func=cmd_cancel)

    p_claim = sub.add_parser("claim", help="claim a pending request (first-answer-wins)")
    p_claim.add_argument("--request", required=True)
    p_claim.add_argument("--channel", required=True, choices=["chat", "telegram"])
    p_claim.add_argument("--answer", required=True)
    p_claim.set_defaults(func=cmd_claim)

    p_fetch = sub.add_parser("fetch", help="idempotent answer fetch")
    p_fetch.add_argument("--request", required=True)
    p_fetch.set_defaults(func=cmd_fetch)

    p_wait = sub.add_parser("wait", help="waiter process: poll until resolution")
    p_wait.add_argument("--request", required=True)
    p_wait.add_argument("--poll-seconds", dest="poll_seconds", type=float, default=None)
    p_wait.add_argument("--max-wait-seconds", dest="max_wait_seconds", type=float, default=None)
    p_wait.add_argument("--debug", action="store_true")
    p_wait.set_defaults(func=cmd_wait)

    p_status = sub.add_parser("status", help="dump broker rows")
    p_status.set_defaults(func=cmd_status)

    p_reconcile = sub.add_parser("reconcile", help="reconciliation report (reporter-only)")
    p_reconcile.set_defaults(func=cmd_reconcile)

    p_sweep = sub.add_parser("sweep", help="re-ask candidates + orphan GC")
    p_sweep.set_defaults(func=cmd_sweep)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="hermes away-bridge")
    parser.add_argument("--db-path", dest="db_path", default=None)
    parser.add_argument("--busy-ms", dest="busy_ms", type=int, default=5000)
    ab_sub = parser.add_subparsers(dest="away_bridge_command")
    register_away_bridge_subparser(ab_sub)
    args = parser.parse_args(argv)
    if getattr(args, "away_bridge_command", None):
        func = getattr(args, "func", None)
        if func is None:
            parser.print_help()
            return 2
        return func(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
