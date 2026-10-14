"""``/rollback`` on the shared gateway session (``rollback.list`` / ``rollback.diff`` / ``rollback.restore``).

The gateway's tools take the checkpoints (its ``CheckpointManager``, keyed by the session's launch
cwd), so the CLI process has no manager of its own: it renders main's text over the owner's
answers. A whole-tree restore also rewinds the newest user turn on the owner (``history_removed``);
the CLI then re-reads the transcript so its warm history matches.
"""
from __future__ import annotations

import os

from agent.i18n import t
from hermes_cli.gateway_client import GatewayClientError


def _t(key, **kw):
    return t(f"cli.commands.{key}", **kw)


def _checkpoint_rows(result):
    """``rollback.list`` rows as ``format_checkpoint_list`` reads them."""
    return [{"hash": c.get("hash", ""), "short_hash": (c.get("hash") or "")[:7], "timestamp": c.get("timestamp", ""),
             "reason": c.get("message", "")} for c in result.get("checkpoints") or []]


def rollback(cli, connection, command):
    from tools.checkpoint_manager import format_checkpoint_list
    args = [a for a in command.split()[1:] if a.lower() not in ("--all", "--force")]
    cwd = os.getenv("TERMINAL_CWD", os.getcwd())
    sid = cli.session_id
    try:
        if not args:
            listed = connection.call("rollback.list", session_id=sid)
            if not listed.get("enabled"):
                return print("\n".join(f"  {line}" if line else "" for line in _t("rollback.not_enabled").splitlines()))
            return print(format_checkpoint_list(_checkpoint_rows(listed), cwd))
        if args[0].lower() == "diff":
            if len(args) < 2:
                return print(f"  {_t('rollback.usage_diff')}")
            return _diff(connection.call("rollback.diff", session_id=sid, hash=args[1]))
        params = {"session_id": sid, "hash": args[0]}
        if len(args) > 1:
            params["file_path"] = args[1]
        _restored(cli, connection, connection.call("rollback.restore", **params), params.get("file_path"))
    except GatewayClientError as exc:
        print(f"  ❌ {exc}")


def _diff(result):
    stat, diff = result.get("stat", ""), result.get("diff", "")
    if not stat and not diff:
        return print(f"  {_t('rollback.no_changes_since')}")
    if stat:
        print(f"\n{stat}")
    if diff:
        lines = diff.splitlines()
        if len(lines) > 80:  # main's terminal-flood limit
            print("\n".join(lines[:80]))
            print(f"\n  {_t('rollback.more_lines', count=len(lines) - 80)}")
        else:
            print(f"\n{diff}")


def _restored(cli, connection, result, file_path):
    if not result.get("success"):
        return print(f"  ❌ {result.get('error')}")
    restored = {"hash": result.get("restored_to", ""), "reason": result.get("reason", "")}
    print(f"  {_t('rollback.restored_file', file=file_path, **restored)}" if file_path
          else f"  {_t('rollback.restored_all', **restored)}")
    print(f"  {_t('rollback.snapshot_saved')}")
    if result.get("history_removed"):
        snapshot = connection.call("session.resume", session_id=cli.session_id)
        connection.state(cli.session_id).adopt_snapshot(snapshot)
        cli.conversation_history = [{k: v for k, v in m.items() if k != "row_id"}
                                    for m in snapshot.get("messages") or []]
        print(f"  {_t('rollback.chat_turn_undone')}")
