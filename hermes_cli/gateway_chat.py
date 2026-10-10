"""Normal classic/one-shot launch through the canonical gateway, never AIAgent."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
from pathlib import Path
import sys
import uuid

from hermes_cli.gateway_client import GatewayClientError, connect_gateway

# These options change execution or require frontend facilities not yet exposed by
# the authority. Reject them, rather than mutate process-wide gateway settings.
_UNSUPPORTED = (
    "image", "worktree", "w",
    "no_restore_cwd",
    "run_budget", "verbose", "compact",
)
_POLICY = ("model", "provider", "reasoning", "toolsets", "max_turns", "base_url", "ignore_rules", "api_key",
           "yolo", "safe_mode", "ignore_user_config", "skills", "checkpoints", "accept_hooks", "pass_session_id")
# Where each refused option lives now; the refusal names it so the user is not left guessing.
_RELOCATED = {
    "image": "`hermes --tui` or the Desktop app to attach the image",
    "worktree": "`hermes --tui -w`",
    "w": "`hermes --tui -w`",
    "no_restore_cwd": "`--in <dir>` (the gateway keeps the session's frozen cwd)",
    "run_budget": "`agent.run_budget_seconds` in config.yaml",
    "verbose": "`hermes logs --follow`, or `hermes chat --tui -v`",
    "compact": "`display.compact: true` in config.yaml",
    "create-if-missing without -c <name>": "`-c <name> --create-if-missing`",
}
_SAFE_MODE_EXAMPLE = 'hermes chat --safe-mode --provider openrouter --model anthropic/claude-sonnet-4 -q "hello"'


# The documented environment equivalents of creation flags (the set Ink's localCreationOptions
# maps). The auto-started owner no longer inherits them (R14 scrub), so they only take effect
# by riding THIS session's session.create; a resume keeps the frozen route.
_ENV_LAUNCH_FLAGS = (("yolo", "HERMES_YOLO_MODE"), ("ignore_rules", "HERMES_IGNORE_RULES"),
                     ("safe_mode", "HERMES_SAFE_MODE"), ("ignore_user_config", "HERMES_IGNORE_USER_CONFIG"),
                     ("accept_hooks", "HERMES_ACCEPT_HOOKS"))


def env_launch_flags() -> dict:
    from utils import is_truthy_value
    return {field: True for field, name in _ENV_LAUNCH_FLAGS if is_truthy_value(os.environ.get(name))}


def bypass_launch(args) -> bool:
    """--safe-mode / --ignore-user-config (or their env opt-ins): the owner freezes code defaults,
    the client reads no profile."""
    env = env_launch_flags()
    return any(getattr(args, name, False) or env.get(name) for name in ("safe_mode", "ignore_user_config"))


def continue_title(args):
    """``-c <name>`` (classic precedence: ignored when ``--resume`` is given)."""
    name = getattr(args, "continue_last", None)
    return name if isinstance(name, str) and not getattr(args, "resume", None) else None


def wants_latest(args):
    """Bare ``-c`` or ``--resume latest`` (the keyword wins over a session titled "latest", which
    stays reachable by id or ``-c latest``)."""
    resume = getattr(args, "resume", None)
    if isinstance(resume, str):
        return resume.strip().lower() == "latest"
    return getattr(args, "continue_last", None) is True


def validate_options(args):
    unsupported = [name for name in _UNSUPPORTED if getattr(args, name, None)]
    if getattr(args, "create_if_missing", False) and not continue_title(args):
        unsupported.append("create-if-missing without -c <name>")
    if unsupported:
        # A one-letter alias (``cli.main(w=True)``) is spelled ``-w``, never the nonexistent ``--w``.
        spell = {name: ("-" if len(name) == 1 else "--") + name.replace("_", "-") for name in unsupported}
        flags = ", ".join(spell[name] for name in unsupported)
        where = "".join(f"\n  {spell[name]}: use {_RELOCATED[name]}" for name in unsupported)
        raise GatewayClientError(
            f"Unsupported gateway CLI options: {flags}. No local fallback or policy changes were made.{where}")
    # Creation flags repeated on resume are checked against the frozen route once the snapshot is
    # back (`check_resume_policy`): the same flags again are fine, a different value is refused.
    # A resume reuses the frozen route and reads no default (the TUI launcher's rule too).
    resuming = getattr(args, "resume", None) or getattr(args, "continue_last", None)
    if bypass_launch(args) and not getattr(args, "model", None) and not resuming:
        raise GatewayClientError("--safe-mode / --ignore-user-config (or HERMES_SAFE_MODE / HERMES_IGNORE_USER_CONFIG) "
                                 "read no profile default: pass --model explicitly."
                                 f"\n  Example: {_SAFE_MODE_EXAMPLE}")


def _caller_cwd(args) -> str:
    """The caller's working directory (``--in`` wins), resolved; filesystem work, so off the loop."""
    return str(Path(getattr(args, "in_dir", None) or os.getcwd()).expanduser().resolve())


def _workspace_key(cwd):
    """The classic CLI's workspace identity for ``-c`` (git root, else the cwd)."""
    from hermes_cli.main import _resolve_workspace_key
    previous = os.getcwd()
    try:
        os.chdir(cwd)
        return _resolve_workspace_key()
    finally:
        os.chdir(previous)


async def _resume(client, args, **params):
    """``session.resume``; ``--api-key`` re-supplies a launch-only key an owner restart revoked.
    The owner binds it only to a session launched with that same key (never durable); anything
    else is a route override and refused like every other creation flag on resume."""
    key = getattr(args, "api_key", None)
    try:
        return await client.rpc("session.resume", **params, **({"api_key": key} if key else {}))
    except GatewayClientError as exc:
        if key and str(exc) == "admission_conflict":
            raise GatewayClientError("--api-key on resume must be the key this session was launched with "
                                     "(a session launched without --api-key keeps its profile credentials).") from None
        raise


async def _resume_latest(client, args):
    """Bare ``-c``: this terminal's breadcrumb session when the owner still has it, else (and for
    ``--resume latest``) the owner's most recent CLI session, this workspace first."""
    from hermes_cli.terminal_breadcrumbs import read_breadcrumb
    crumb = (read_breadcrumb() or {}).get("session_id") if getattr(args, "continue_last", None) is True else None
    if isinstance(crumb, str) and crumb:
        try:
            return await _resume(client, args, session_id=crumb)
        except GatewayClientError as exc:
            if str(exc) not in {"not_found", "permission_denied"}:
                raise
    workspace = await asyncio.to_thread(_workspace_key, _caller_cwd(args))
    try:
        return await _resume(client, args, latest="cli", **({"workspace": workspace} if workspace else {}))
    except GatewayClientError as exc:
        if str(exc) != "not_found":
            raise
        raise GatewayClientError("No previous CLI session to continue. Start a new one with `hermes`, "
                                 "or list sessions with `hermes sessions list`.") from None


async def run_gateway_chat(args, emitter=None):
    from hermes_cli.gateway_chat_view import GatewayChatView
    query = getattr(args, "query", None) or getattr(args, "q", None)
    oneshot_prompt = getattr(args, "oneshot", None)
    if isinstance(oneshot_prompt, str):
        query = oneshot_prompt
    quiet = bool(getattr(args, "quiet", False) or oneshot_prompt or emitter is not None)
    # Finite: answer one prompt and exit (-z, --oneshot, -Q, stream-json, or -q off a TTY).
    oneshot = bool(oneshot_prompt or getattr(args, "oneshot_exit", False) or quiet or
                   (query and not (sys.stdin.isatty() and sys.stdout.isatty())))
    async with connect_gateway() as client:
        description = await client.rpc("runtime.describe")
        title = continue_title(args)
        create_if_missing = bool(title and getattr(args, "create_if_missing", False))
        latest = wants_latest(args)
        if latest:
            snapshot = await _resume_latest(client, args)
            check_resume_policy(args, snapshot)
        elif getattr(args, "resume", None) or (title and not create_if_missing):
            name = getattr(args, "resume", None) or title
            try:
                # Exact id first, then title (latest lineage continuation), as the classic CLI did.
                snapshot = await _resume(client, args, session_id=name)
            except GatewayClientError as exc:
                if str(exc) != "not_found":
                    raise
                try:
                    snapshot = await _resume(client, args, title=name)
                except GatewayClientError as exc:
                    if str(exc) != "not_found":
                        raise
                    raise GatewayClientError(
                        f"No session found matching '{name}'. Use 'hermes sessions list' to see available "
                        "sessions, or pass -c <name> --create-if-missing to start a new session with that title.")
            check_resume_policy(args, snapshot)
        else:
            contract = description.get("session_create", {})
            # Finite runs are stored as ``oneshot`` (hidden from human pickers, still CLI history);
            # an explicit ``--source`` always wins. A remote gateway predating the label stores ``cli``.
            sources = contract.get("sources", [])
            source = getattr(args, "source", None) or ("oneshot" if oneshot and "oneshot" in sources else "cli")
            if source not in sources:
                raise GatewayClientError(f"Gateway does not support source {source!r}")
            parameters = contract.get("parameters", [])
            policy = _requested_policy(args)
            policy.pop("source", None)
            if create_if_missing:
                policy["title"] = title
            policy.update(env_launch_flags())
            cwd = await asyncio.to_thread(_caller_cwd, args)
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
        if not oneshot:
            from hermes_cli.terminal_breadcrumbs import write_breadcrumb
            await asyncio.to_thread(write_breadcrumb, snapshot["stored_session_id"])
        if emitter is not None:
            emitter.bind_session(snapshot["stored_session_id"])
        view = GatewayChatView(client, snapshot, quiet=quiet, emitter=emitter,
                               usage_file=getattr(args, "usage_file", None))
        view.launch_api_key = getattr(args, "api_key", None)  # /new keeps this launch's key
        view.unattended = isinstance(oneshot_prompt, str)
        view.resume_footer = oneshot and not quiet
        if (getattr(args, "resume", None) or title or latest) and not quiet:
            for row in snapshot.get("messages", []):
                if row.get("role") in {"user", "assistant"} and isinstance(row.get("content"), str):
                    print(f"{row['role']}: {row['content']}")
        return await view.run(query, oneshot=oneshot)


_RESUME_POLICY_MISMATCH = "Resume retains gateway session policy; creation overrides are unsupported on resume."


def _launch_flags(args):
    """The creation flags the user passed. Identity, not equality: ``--max-turns 0`` (unlimited)
    is a value, while ``0 == False`` would drop it to the profile default."""
    return {key: value for key in _POLICY if (value := getattr(args, key, None)) is not None and value is not False}


def _requested_policy(args):
    policy = _launch_flags(args)
    if isinstance(policy.get("toolsets"), str):
        policy["toolsets"] = [name.strip() for name in policy["toolsets"].split(",") if name.strip()]
    if "skills" in policy:
        # `-s a,b -s c` (argparse append) or cli.main's comma string: one deduplicated list.
        raw = policy["skills"] if isinstance(policy["skills"], (list, tuple)) else [policy["skills"]]
        names = [name.strip() for item in raw for name in str(item).split(",") if name.strip()]
        policy["skills"] = list(dict.fromkeys(names))
        if not policy["skills"]:
            policy.pop("skills")
    if getattr(args, "source", None):
        policy["source"] = args.source
    return policy


def check_resume_policy(args, snapshot):
    """A resume may repeat the flags the session was created with (scripts re-run one command
    line); anything that would CHANGE the frozen route is refused, as before."""
    requested = _requested_policy(args)
    if getattr(args, "in_dir", None):
        requested["cwd"] = str(Path(args.in_dir).expanduser().resolve())
    if not requested:
        return
    info = snapshot.get("info") if isinstance(snapshot, dict) else None
    frozen = dict((info or {}).get("launch_request") or {})
    if info and "cwd" in info:
        frozen["cwd"] = info["cwd"]
    for key, value in requested.items():
        if key == "api_key":
            continue  # never durable; a repeated key is not a policy change
        if frozen.get(key) != value:
            raise GatewayClientError(_RESUME_POLICY_MISMATCH)


def _register_terminal_process() -> None:
    """The chat client is still a terminal on this HERMES_HOME: record it in the process ledger
    (purpose ``cli`` is never update-reapable) and warn once when another install shares the home."""
    from hermes_cli.process_identity import register_self
    from hermes_cli.shared_profile_warning import shared_profile_warning

    register_self("cli")
    warning = shared_profile_warning()
    if warning:
        print(f"Warning: {warning}", file=sys.stderr)


def launch_from_args(args) -> int:
    from websockets.exceptions import WebSocketException
    emitter = None
    if getattr(args, "output_format", "text") == "stream-json":
        # Built before validation/connection so any failed start still closes the protocol
        # (init + result) instead of exiting with an empty stdout.
        from hermes_cli.stream_json import StreamJsonEmitter
        emitter = StreamJsonEmitter(model=getattr(args, "model", None) or "")

    def failed(message, code):
        print("Error: " + message, file=sys.stderr)
        if emitter is not None:
            return emitter.emit_result({"failed": True, "error": message}, exit_code=code)
        return code

    try:
        validate_options(args)
        if getattr(args, "list_tools", False) or getattr(args, "list_toolsets", False):
            # A catalog listing needs no session and no gateway (main printed it and exited).
            from hermes_cli.gateway_chat_listing import print_tool_listing
            return print_tool_listing(args)
        _register_terminal_process()
        from hermes_cli.gateway_chat_startup import ensure_launch_provider
        if emitter is None:
            if not ensure_launch_provider(args):
                return 0
        else:
            # Setup guidance is human text and the guard exits on a non-TTY: both must become
            # stderr diagnostics + a failed ``result`` when stdout is machine-readable.
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    configured = ensure_launch_provider(args)
            except SystemExit:
                configured = False
            if not configured:
                return failed("credentials or agent init failed", 1)
        query_file = getattr(args, "query_file", None)
        if query_file:
            args.query = sys.stdin.read() if query_file == "-" else Path(query_file).read_text(encoding="utf-8-sig")
            if not args.query.strip():
                raise GatewayClientError("--query-file is empty")
        if not (getattr(args, "query", None) or getattr(args, "q", None) or getattr(args, "oneshot", None) or sys.stdin.isatty()):
            raise GatewayClientError("Noninteractive chat requires --query or --oneshot")
        return asyncio.run(run_gateway_chat(args, emitter=emitter))
    except (GatewayClientError, OSError, TimeoutError, WebSocketException) as exc:
        # WebSocket errors can embed credential URLs/remote bodies.
        message = str(exc) if isinstance(exc, GatewayClientError) else "Gateway connection/read failed; no local fallback"
        return failed(message, 2 if isinstance(exc, GatewayClientError) and ("Unsupported" in message or "unsupported" in message) else 1)
    except KeyboardInterrupt:
        print("Detached; accepted work continues at the gateway.", file=sys.stderr)
        if emitter is not None:
            return emitter.emit_result({"interrupted": True}, exit_code=130)
        return 130


def launch_from_kwargs(options) -> int:
    return launch_from_args(argparse.Namespace(**options))
