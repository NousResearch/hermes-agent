"""Normal classic/one-shot launch through the canonical gateway, never AIAgent."""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import sys
import uuid

from hermes_cli.gateway_client import GatewayClientError, connect_gateway

# These options change execution or require frontend facilities not yet exposed by
# the authority. Reject them, rather than mutate process-wide gateway settings.
_UNSUPPORTED = (
    "image", "skills", "worktree", "w", "checkpoints", "pass_session_id",
    "yolo", "accept_hooks",
    "no_restore_cwd", "usage_file",
    "run_budget", "verbose", "compact",
    "list_tools", "list_toolsets",
)
_POLICY = ("model", "provider", "reasoning", "toolsets", "max_turns", "base_url", "ignore_rules", "api_key",
           "safe_mode", "ignore_user_config")
# Where each refused option lives now; the refusal names it so the user is not left guessing.
_RELOCATED = {
    "image": "attach the image in `hermes --tui` or the Desktop app",
    "skills": "`hermes --tui -s <skill>`",
    "worktree": "`hermes --tui -w`",
    "w": "`hermes --tui -w`",
    "checkpoints": "`checkpoints.enabled: true` in config.yaml, or `hermes --tui --checkpoints`",
    "pass_session_id": "`hermes --tui --pass-session-id`",
    "yolo": "`approvals.mode: off` in config.yaml, or `/yolo` inside the session",
    "accept_hooks": "`hooks_auto_accept: true` in config.yaml, or `hermes --tui --accept-hooks`",
    "no_restore_cwd": "`--in <dir>` (the gateway keeps the session's frozen cwd)",
    "usage_file": "`hermes sessions stats` / `hermes insights` after the run",
    "run_budget": "`agent.run_budget_seconds` in config.yaml",
    "verbose": "`hermes logs --follow`, or `hermes --tui -v`",
    "compact": "`display.compact: true` in config.yaml",
    "list_tools": "`hermes tools list`",
    "list_toolsets": "`hermes tools list`",
    "resume latest": "`hermes --tui --resume latest`, or `hermes sessions list` then `--resume <id>`",
    "continue": "`-c <name>` (or `hermes --tui -c` for the most recent session)",
    "create-if-missing without -c <name>": "`-c <name> --create-if-missing`",
}
_SAFE_MODE_EXAMPLE = 'hermes chat --safe-mode --provider openrouter --model anthropic/claude-sonnet-4 -q "hello"'


def bypass_launch(args) -> bool:
    """--safe-mode / --ignore-user-config: the owner freezes code defaults, the client reads no profile."""
    return bool(getattr(args, "safe_mode", False) or getattr(args, "ignore_user_config", False))


def continue_title(args):
    """``-c <name>`` (classic precedence: ignored when ``--resume`` is given). Bare ``-c`` needs the
    breadcrumb/MRU lookup the authority does not expose, so it stays refused like ``--resume latest``."""
    name = getattr(args, "continue_last", None)
    return name if isinstance(name, str) and not getattr(args, "resume", None) else None


def validate_options(args):
    unsupported = [name for name in _UNSUPPORTED if getattr(args, name, None)]
    if getattr(args, "resume", None) == "latest":
        unsupported.append("resume latest")
    if getattr(args, "continue_last", None) is True:
        unsupported.append("continue")
    if getattr(args, "create_if_missing", False) and not continue_title(args):
        unsupported.append("create-if-missing without -c <name>")
    if unsupported:
        flags = ", ".join("--" + name.replace("_", "-") for name in unsupported)
        where = "".join(f"\n  --{name.replace('_', '-')}: use {_RELOCATED[name]}" for name in unsupported)
        raise GatewayClientError(
            f"Unsupported gateway CLI options: {flags}. No local fallback or policy changes were made.{where}")
    resuming = getattr(args, "resume", None) or (continue_title(args) and not getattr(args, "create_if_missing", False))
    if resuming and (getattr(args, "in_dir", None) or getattr(args, "source", None) or
            any(getattr(args, name, None) not in (None, False) for name in _POLICY)):
        raise GatewayClientError("Resume retains gateway session policy; creation overrides are unsupported on resume.")
    if bypass_launch(args) and not getattr(args, "model", None):
        raise GatewayClientError("--safe-mode / --ignore-user-config read no profile default: pass --model explicitly."
                                 f"\n  Example: {_SAFE_MODE_EXAMPLE}")


async def run_gateway_chat(args):
    from hermes_cli.gateway_chat_view import GatewayChatView
    async with connect_gateway() as client:
        description = await client.rpc("runtime.describe")
        title = continue_title(args)
        create_if_missing = bool(title and getattr(args, "create_if_missing", False))
        if getattr(args, "resume", None) or (title and not create_if_missing):
            name = getattr(args, "resume", None) or title
            try:
                # Exact id first, then title (latest lineage continuation), as the classic CLI did.
                snapshot = await client.rpc("session.resume", session_id=name)
            except GatewayClientError as exc:
                if str(exc) != "not_found":
                    raise
                try:
                    snapshot = await client.rpc("session.resume", title=name)
                except GatewayClientError as exc:
                    if str(exc) != "not_found":
                        raise
                    raise GatewayClientError(
                        f"No session found matching '{name}'. Use 'hermes sessions list' to see available "
                        "sessions, or pass -c <name> --create-if-missing to start a new session with that title.")
        else:
            contract = description.get("session_create", {})
            source = getattr(args, "source", None) or "cli"
            if source not in contract.get("sources", []):
                raise GatewayClientError(f"Gateway does not support source {source!r}")
            parameters = contract.get("parameters", [])
            policy = {key: getattr(args, key) for key in _POLICY if getattr(args, key, None) not in (None, False)}
            if create_if_missing:
                policy["title"] = title
            if isinstance(policy.get("toolsets"), str):
                policy["toolsets"] = [name.strip() for name in policy["toolsets"].split(",") if name.strip()]
            cwd = str(Path(getattr(args, "in_dir", None) or os.getcwd()).expanduser().resolve())
            if "cwd" in parameters:
                policy["cwd"] = cwd
            elif getattr(args, "in_dir", None):
                raise GatewayClientError("Gateway does not support --in / caller cwd; update the gateway")
            else:
                print("Warning: this gateway cannot preserve caller cwd; it uses its configured execution directory.", file=sys.stderr)
            missing = sorted(set(policy) - set(parameters))
            if missing:
                raise GatewayClientError("Gateway does not support creation options: " + ", ".join(missing))
            snapshot = await client.rpc("session.create", request_id=uuid.uuid4().hex, source=source, **policy)
        print("Session: " + snapshot["stored_session_id"], file=sys.stderr, flush=True)
        query = getattr(args, "query", None) or getattr(args, "q", None)
        oneshot_prompt = getattr(args, "oneshot", None)
        if isinstance(oneshot_prompt, str):
            query = oneshot_prompt
        quiet = bool(getattr(args, "quiet", False) or oneshot_prompt)
        oneshot = bool(oneshot_prompt or getattr(args, "oneshot_exit", False) or quiet or
                       (query and not (sys.stdin.isatty() and sys.stdout.isatty())))
        view = GatewayChatView(client, snapshot, quiet=quiet)
        if (getattr(args, "resume", None) or title) and not quiet:
            for row in snapshot.get("messages", []):
                if row.get("role") in {"user", "assistant"} and isinstance(row.get("content"), str):
                    print(f"{row['role']}: {row['content']}")
        return await view.run(query, oneshot=oneshot)


def launch_from_args(args) -> int:
    from websockets.exceptions import WebSocketException
    try:
        validate_options(args)
        from hermes_cli.gateway_chat_startup import ensure_launch_provider
        if not ensure_launch_provider(args):
            return 0
        query_file = getattr(args, "query_file", None)
        if query_file:
            args.query = sys.stdin.read() if query_file == "-" else Path(query_file).read_text(encoding="utf-8")
            if not args.query.strip():
                raise GatewayClientError("--query-file is empty")
        if not (getattr(args, "query", None) or getattr(args, "q", None) or getattr(args, "oneshot", None) or sys.stdin.isatty()):
            raise GatewayClientError("Noninteractive chat requires --query or --oneshot")
        return asyncio.run(run_gateway_chat(args))
    except (GatewayClientError, OSError, TimeoutError, WebSocketException) as exc:
        # WebSocket errors can embed credential URLs/remote bodies.
        message = str(exc) if isinstance(exc, GatewayClientError) else "Gateway connection/read failed; no local fallback"
        print("Error: " + message, file=sys.stderr)
        return 2 if isinstance(exc, GatewayClientError) and ("Unsupported" in message or "unsupported" in message) else 1
    except KeyboardInterrupt:
        print("Detached; accepted work continues at the gateway.", file=sys.stderr)
        return 130


def launch_from_kwargs(options) -> int:
    return launch_from_args(argparse.Namespace(**options))
