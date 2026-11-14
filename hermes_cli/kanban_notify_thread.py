"""Which thread a ``kanban notify-subscribe`` without ``--thread-id`` should target (#87281).

An omitted ``--thread-id`` meant the chat's root lane, so a task dispatched from a Telegram DM topic
pinged the DM root the operator is not watching. Resolution order: the caller's own session thread
when the target is the caller's own chat, else the thread of an existing subscription of this task to
the same chat; a Telegram DM target with neither gets a stderr warning.
"""

from __future__ import annotations

import sys


def _session_thread_for(platform: str, chat_id: str) -> str | None:
    from gateway.session_context import get_session_env

    if (str(get_session_env("HERMES_SESSION_PLATFORM", "")).lower() == platform.lower()
            and str(get_session_env("HERMES_SESSION_CHAT_ID", "")) == str(chat_id)):
        return get_session_env("HERMES_SESSION_THREAD_ID", "") or None
    return None


def resolve_subscribe_thread(conn, args) -> str | None:
    """``args.thread_id`` when given; otherwise the inferred thread (or None), reporting on stderr."""
    from hermes_cli import kanban_db_notify as kbn

    if args.thread_id is not None:
        return args.thread_id
    if thread_id := _session_thread_for(args.platform, args.chat_id):
        print(f"(no --thread-id given; using this session's thread_id={thread_id})", file=sys.stderr)
        return thread_id
    # Several threaded rows per chat are normal (the PK is the full 4-tuple) and list_notify_subs is
    # unordered: sort for a deterministic pick and name the alternatives.
    threads = sorted(
        str(sub["thread_id"]) for sub in kbn.list_notify_subs(conn, args.task_id)
        if str(sub.get("platform", "")).lower() == args.platform.lower()
        and str(sub.get("chat_id", "")) == str(args.chat_id) and sub.get("thread_id"))
    if threads:
        alternatives = f" (others: {', '.join(threads[1:])}; pass --thread-id to choose)" if len(threads) > 1 else ""
        print(f"(no --thread-id given; inherited thread_id={threads[0]} from this task's existing "
              f"{args.platform}:{args.chat_id} subscription{alternatives})", file=sys.stderr)
        return threads[0]
    if args.platform.lower() == "telegram" and args.chat_type in (None, "dm"):
        print("warning: no --thread-id given and none to infer; the completion ping will go to the "
              "chat's DM root, not a topic", file=sys.stderr)
    return None
