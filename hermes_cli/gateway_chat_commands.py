"""Classic CLI slash commands on the shared gateway, beyond the view's own controls.

``GatewayChatView.command`` answers the controls it owns (/stop /approve /answer /discard
/branch /model /compress /yolo /quit). This module adds the session commands main's classic
CLI ran in-process, mapped onto what the owner serves -- the mapping Ink uses
(``ui-tui/src/app/slash/canonicalSessionCommands.ts``):

- ``/title <name>`` -> ``session.mutate rename``; bare ``/title`` -> the owner's report
- ``/undo [N]`` / ``/retry`` -> ``session.mutate rewind`` to the Nth-newest user turn
  (``/undo`` puts the removed text back in the composer, ``/retry`` resubmits it)
- ``/new [title]`` / ``/reset`` -> ``session.create`` with this session's frozen launch request
- ``/usage`` -> the owner's session report plus this view's last committed turn
- ``/tools`` -> the toolsets frozen at launch
- every other registry command -> ``slash.exec``: the owner runs its reviewed reads and
  refuses the rest, which prints where the command stands (the parity table below).
"""
import sys
import uuid

from hermes_cli.gateway_client import GatewayClientError

PARITY_DOC = "https://hermes-agent.nousresearch.com/docs/developer-guide/gateway-command-parity"

# Registry command (canonical name) -> how the classic view answers it itself. Every other
# registry command goes to the owner through ``slash.exec``. Read by the parity inventory
# (scripts/gen_gateway_command_parity.py).
VIEW_ROUTES = {
    "quit": "detach", "stop": "session.interrupt", "approve": "approval.respond (prompt id)",
    "branch": "session.mutate branch", "model": "session.mutate model", "compress": "session.mutate compress",
    "yolo": "config.set yolo", "help": "local",
    "title": "session.mutate rename", "undo": "session.mutate rewind", "retry": "session.mutate rewind + resubmit",
    "new": "session.create", "clear": "session.create", "usage": "slash.exec + last-turn receipt",
    "tools": "session.info launch toolsets",
}


def refusal(name):
    return f"/{name} is not available on the shared gateway yet. Parity table: {PARITY_DOC}"


def _user_turns(snapshot):
    """Human-authored user rows (``row_id`` = durable ``messages.id``), oldest first: no display-only
    kind other than a typed /steer, never a compaction handoff carrier."""
    from agent.context_compressor import is_user_originated_turn
    return [row for row in snapshot.get("messages", [])
            if type(row.get("row_id")) is int and row.get("display_kind") in (None, "", "steer")
            and is_user_originated_turn(row)]


def _rewind_to(turns_back):
    def payload(snapshot):
        turns = _user_turns(snapshot)
        return {"target_message_id": turns[-turns_back]["row_id"]} if len(turns) >= turns_back else None
    return payload


async def _mutate(view, operation, payload):
    result = await view.mutations.apply(view.client, view.session_id, operation, payload)
    view.mutations.acknowledge(view.session_id, operation, payload)
    return result


async def _rewind(view, turns_back):
    """The rewound user turn's text, or None when there was nothing to rewind."""
    result = await _mutate(view, "rewind", _rewind_to(turns_back))
    if result.get("status") == "nothing" or not result.get("rewound_count"):
        return None
    snapshot = await view.client.rpc("session.resume", session_id=view.session_id)
    view.generation = snapshot["execution_generation"]
    content = (result.get("target_message") or {}).get("content")
    return content if isinstance(content, str) else "", result["rewound_count"]


async def _title(view, rest):
    if not rest:
        return await forward(view, "title", "")
    result = await _mutate(view, "rename", {"title": rest})
    print(f"Session title set: {result.get('title') or rest}")


async def _undo(view, rest):
    from agent.i18n import t
    if rest and not (rest.isdigit() and int(rest) > 0):
        raise GatewayClientError("Usage: /undo [N]")
    turns = int(rest or 1)
    rewound = await _rewind(view, turns)
    if rewound is None:
        print(t("cli.session.undo_no_user_message"))
        return
    text, count = rewound
    print(f"Undid {turns} turn{'s' if turns != 1 else ''} ({count} message(s)); the removed message is in the composer.")
    view.prefill = text


async def _retry(view, rest):
    from agent.i18n import t
    rewound = await _rewind(view, 1)
    if rewound is None or not rewound[0].strip():
        print(t("cli.session.retry_no_user_message"))
        return
    print(f"\n{'─' * 40}\n● {rewound[0]}", flush=True)
    await view.submit(rewound[0])


async def _new(view, rest):
    """A fresh session with this one's frozen launch request (model, toolsets, cwd, ...); the old
    session stays resumable by its id. A title is a separate rename so an existing title is a
    conflict, never a silent attach to that other session."""
    from agent.i18n import t
    info = await view.client.rpc("session.info", session_id=view.session_id)
    accepted = set((await view.client.rpc("runtime.describe")).get("session_create", {}).get("parameters", []))
    request = {key: value for key, value in (info.get("launch_request") or {}).items()
               if key in accepted and key not in {"request_id", "title"}}
    if info.get("cwd") and "cwd" in accepted:
        request["cwd"] = info["cwd"]
    # The frozen request never holds the launch key; the one this terminal supplied rides along.
    if getattr(view, "launch_api_key", None) and "api_key" in accepted:
        request["api_key"] = view.launch_api_key
    snapshot = await view.client.rpc("session.create", request_id=uuid.uuid4().hex, **request)
    await view.adopt(snapshot)
    print(t("cli.session.new_session"))
    print(f"Session: {view.session_id}", file=sys.stderr, flush=True)
    if rest:
        await _title(view, rest)


async def _usage(view, rest):
    if rest:  # `/usage reset` redeems a credit: the owner decides (and refuses it for local sessions)
        return await forward(view, "usage", rest)
    await forward(view, "usage", "")
    admission = next(reversed(view.completions), None)
    if admission is None:
        return
    receipt = await view.client.rpc("prompt.receipt", session_id=view.session_id,
                                    admission_id=admission, include_result=True)
    result = receipt.get("result") or {}
    if not isinstance(result, dict) or not result:
        return
    lines = ["", "Last turn:", f"  Model: {result.get('model') or view.model}"]
    for label, key in (("Input tokens", "input_tokens"), ("Output tokens", "output_tokens"),
                       ("Total tokens", "total_tokens"), ("API calls", "api_calls")):
        lines.append(f"  {label}: {int(result.get(key) or 0):,}")
    if isinstance(result.get("estimated_cost_usd"), (int, float)):
        lines.append(f"  Cost: ${result['estimated_cost_usd']:.4f}")
    print("\n".join(lines))


async def _tools(view, rest):
    info = await view.client.rpc("session.info", session_id=view.session_id)
    toolsets = (info.get("launch_request") or {}).get("toolsets")
    if isinstance(toolsets, list) and toolsets:
        print("Toolsets frozen at launch: " + ", ".join(map(str, toolsets)))
    else:
        print("This session runs the profile's CLI toolsets as they were at launch (`hermes tools list`).")
    print("Changing toolsets mid-session (/tools enable|disable) " + refusal("tools").split(" ", 1)[1])


async def forward(view, name, rest):
    """Run a registry command (or a skill) on the owner; its refusal names where the command stands."""
    from hermes_cli.gateway_client import GatewayRPCError
    try:
        reply = await view.client.rpc("slash.exec", session_id=view.session_id,
                                      command=name + (" " + rest if rest else ""))
    except GatewayRPCError as exc:
        if str(exc) != "unsupported_command":
            raise
        from hermes_cli.commands import resolve_command
        print(refusal(name) if resolve_command(name) else f"Unknown command: /{name} (use /help)")
        return
    if reply.get("type") == "skill":
        # A skill directive is the client's durable submit (same contract as Ink and Desktop).
        print(f"\n{'─' * 40}\n● {reply.get('display') or '/' + name}", flush=True)
        await view.submit(reply["message"])
    else:
        print(reply.get("output") or f"/{name}: no output")


_VIEW_CONTROLS = frozenset({"quit", "stop", "approve", "branch", "model", "compress", "yolo", "help"})
# /clear is main's "new session + clear screen"; the gateway view keeps scrollback and starts fresh.
_HANDLERS = {"title": _title, "undo": _undo, "retry": _retry, "new": _new, "clear": _new, "usage": _usage,
             "tools": _tools}


async def run_command(view, command, rest):
    """Any slash command the view's own controls did not take."""
    from hermes_cli.commands import resolve_command
    name = command.lstrip("/")
    definition = resolve_command(name)
    canonical = definition.name if definition else name
    if canonical in _VIEW_CONTROLS and canonical != name:
        # An alias of a control the view owns (/fork, /compact, /exit): answer it as the canonical.
        return await view.command(f"/{canonical} {rest}".rstrip())
    handler = _HANDLERS.get(canonical)
    if handler is not None:
        await handler(view, rest.strip())
    else:
        await forward(view, canonical, rest.strip())
    return True


def help_text():
    from gateway.session_commands import _READ_COMMANDS
    reports = " ".join(f"/{name}" for name in sorted(_READ_COMMANDS - {"help"}))
    return ("/stop, /approve <id> <choice>, /answer <id> <text>, /discard <admission_id> (turn lost during "
            "restart), /yolo [on|off], /quit (detach). /branch [title], /model <model> [--provider name], "
            "/compress [here [N] | <focus>] [--preview]. /title [name], /undo [N], /retry, /new [title] "
            f"(/reset), /usage, /tools.\nReports run by the gateway: {reports}.\nParity table: {PARITY_DOC}")
