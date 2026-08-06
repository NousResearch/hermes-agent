"""Claude Agent SDK runtime — whole-turn path for ``api_mode="claude_agent_sdk"``.

Hands one user turn to the official ``claude-agent-sdk`` (which drives the
Claude Code CLI) and projects the SDK's message stream back into Hermes'
canonical callbacks and message shape, so a Claude turn renders identically
on the CLI, TUI, Desktop, gateway, and ACP without surface-specific code.

The ownership split is the whole point (see
``docs/design/claude-subscription-via-agent-sdk.md``):

* **Hermes keeps** the system prompt, the tool implementations, approvals,
  environments, checkpoints, plugins, memory, skills, and UI history.  Every
  tool call comes back through the in-process MCP bridge in
  :mod:`agent.transports.claude_tool_bridge`, which funnels into
  ``execute_one_tool``.
* **The SDK keeps** Claude-native context, prompt-cache continuity,
  automatic compaction, alternation, the resume cursor, and turn usage.

Called from ``run_conversation()`` via ``AIAgent._run_claude_agent_sdk_turn``.
Returns the same dict shape as the chat_completions path.

The runtime ships default-off behind ``claude_subscription.enabled``; see
:mod:`hermes_cli.claude_subscription`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from agent.transports.claude_tool_bridge import (
    MCP_SERVER_NAME,
    MCP_TOOL_PREFIX,
    bridged_allowed_tools,
    build_hermes_sdk_mcp_server,
)

logger = logging.getLogger(__name__)

RUNTIME_LABEL = "claude_agent_sdk"

# The turn is bounded by the session facade's deadline, not by Hermes'
# per-call timeout: the SDK runs an entire agentic loop inside one query.
DEFAULT_TURN_TIMEOUT_SECONDS = 1800.0

# Claude's stderr is diagnostic, not user-facing: it goes to agent.log at DEBUG
# so it is there when someone asks "why did the CLI die", without spamming
# errors.log. The first few error-shaped lines are also escalated to WARNING —
# capped, because a wedged CLI can emit thousands.
MAX_ESCALATED_STDERR_LINES = 10
_STDERR_ERROR_PATTERN = re.compile(
    r"\b(error|fatal|panic|traceback|exception|econnrefused)\b", re.IGNORECASE
)

stderr_logger = logging.getLogger(f"{__name__}.claude_stderr")


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def claude_runtime_preflight(config: Optional[dict] = None) -> Optional[str]:
    """Return an actionable error string, or None when the runtime may start.

    Four independent gates, each with its own fix, because "it didn't work"
    is useless to a user who has to know *which* of them to act on: the config
    gate, the optional extra, the billing source, and the login.

    The billing gate comes before the login probe because it is free — it only
    reads variable *names* out of the environment — and because a credential
    that outranks the user's subscription means this turn would be billed to a
    different account, which is a refusal no matter what the login says.  The
    exact subprocess probe runs once per session in
    :func:`verify_claude_billing_for_agent`.
    """
    from hermes_cli.claude_subscription import (
        claude_agent_sdk_available,
        claude_subscription_enabled,
    )

    if config is None:
        try:
            from hermes_cli.config import load_config_readonly

            config = load_config_readonly()
        except Exception:
            logger.debug("claude runtime preflight config load failed", exc_info=True)
            config = {}

    if not claude_subscription_enabled(config):
        return (
            "The Claude subscription runtime is turned off. Set "
            "`claude_subscription.enabled: true` in config.yaml to use it."
        )

    if not claude_agent_sdk_available():
        return (
            "The Claude subscription runtime needs the optional `claude-code` "
            "extra. Install it with: pip install 'hermes-agent[claude-code]'"
        )

    from agent.claude_billing import static_billing_refusal

    refusal = static_billing_refusal()
    if refusal is not None:
        return refusal

    from hermes_cli.claude_code import CLAUDE_LOGIN_COMMAND, probe_claude_auth_cached

    probe = probe_claude_auth_cached()
    if not probe.get("logged_in"):
        return (
            probe.get("message")
            or f"Not signed in to Claude — run `{CLAUDE_LOGIN_COMMAND}`."
        )
    return None


_UNSET = object()


def verify_claude_billing_for_agent(agent) -> Optional[str]:
    """Prove this agent's session bills the user's plan. Cached per session.

    The proof is the Claude Code CLI's own initialize response, read through a
    throwaway SDK connection that never writes a user message — so the child
    completes its local startup, reports the account it resolved, and exits
    without ever issuing a model request.  Costs no tokens and no quota.

    Cached on the agent because it spawns a subprocess: it must not run per
    turn.  ``_retire_session`` clears it, so a rebuilt session re-proves.
    """
    cached = getattr(agent, "_claude_billing_refusal", _UNSET)
    if cached is not _UNSET:
        return cached

    from agent.claude_billing import verify_claude_billing_source

    refusal = verify_claude_billing_source()
    agent._claude_billing_refusal = refusal
    if refusal is None:
        logger.info("claude_agent_sdk billing source confirmed as the user's plan")
    return refusal


def _make_stderr_logger() -> Callable[[str], None]:
    """Route the Claude child's stderr into Hermes' logs.

    The SDK inherits the parent's stderr unless ``options.stderr`` is set, and
    under Electron that handle can be closed or unusable — so a callback is
    always registered, both to keep the child's diagnostics and to keep it off
    a handle it cannot write to.
    """
    escalated = [0]

    def _on_stderr(line: str) -> None:
        text = (line or "").rstrip()
        if not text:
            return
        stderr_logger.debug("claude: %s", text)
        if (
            escalated[0] < MAX_ESCALATED_STDERR_LINES
            and _STDERR_ERROR_PATTERN.search(text)
        ):
            escalated[0] += 1
            stderr_logger.warning("claude: %s", text)

    return _on_stderr


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


# Appended to the claude_code preset in subscription mode.
#
# Identity has to live HERE, not only in the first-turn context, because the
# system prompt is the only channel that persists for the life of the session.
# A one-time user-turn injection binds identity on turn 1 and then decays: by
# turn 3 of a real conversation the model drifts back to the preset's "You are
# Claude Code" (measured). The full operating instructions still arrive as the
# first user turn — this is the short, durable anchor that keeps pointing at
# them.
#
# Keep it short. The billing classifier reads appended system-prompt content:
# ~700 chars of identity and persona bills to the plan, while Hermes' full
# prompt does not (decision record §11). This text is byte-stable for the life
# of the conversation, so prompt caching is preserved.
def hermes_identity_anchor(agent=None) -> str:
    """Hermes' own identity text, verbatim — nothing authored here.

    The subscription runtime cannot put Hermes' full prompt in the system
    slot (the billing classifier re-bills that to extra usage, decision
    record §11), and the full prompt delivered as a first user turn decays:
    by turn 3 the preset's own identity line wins and the model reintroduces
    itself as the CLI rather than as Hermes (measured).

    So the system slot carries exactly one thing: the same identity section
    Hermes already puts at the top of its own prompt for every other
    provider — the user's ``SOUL.md`` when they have one, otherwise
    ``DEFAULT_AGENT_IDENTITY``. This adds no provider-specific persona: a
    user who customises their identity gets their customisation here too,
    and the rest of the prompt is unchanged and delivered in full one
    message later.
    """
    try:
        from agent.prompt_builder import DEFAULT_AGENT_IDENTITY, load_soul_md

        soul = None
        # Honour SOUL.md only when this agent would load it for its own
        # prompt, so the anchor never contradicts the prompt it anchors.
        if agent is None or getattr(agent, "load_soul_identity", False) or not getattr(
            agent, "skip_context_files", False
        ):
            soul = load_soul_md()
        identity = (soul or "").strip() or DEFAULT_AGENT_IDENTITY
    except Exception:
        logger.debug("identity anchor lookup failed; using the default", exc_info=True)
        from agent.prompt_builder import DEFAULT_AGENT_IDENTITY

        identity = DEFAULT_AGENT_IDENTITY
    return identity.strip()


# Operational fact about this runtime, not persona: the tools genuinely carry
# the mcp__hermes__ prefix here, and the built-ins genuinely are denied (the
# PreToolUse hook). Hermes' own prompt references its tools by bare name, so
# without this line the model only discovers the routing when a denial bounces
# it — measured in a clean container as the model complaining about blocked
# tools instead of switching. Keeps the append far below the classifier
# threshold (§11).
CLAUDE_TOOL_ROUTING_NOTE = (
    "In this session your tools are the mcp__hermes__* tools (for example "
    "mcp__hermes__read_file, mcp__hermes__terminal); use them for file, "
    "terminal, web, and delegation work. The built-in tools are disabled and "
    "their calls will be denied."
)


def claude_subscription_append(agent=None) -> str:
    """The system-prompt append: Hermes' own identity section, plus one
    factual note about this runtime's tool naming. No persona is authored."""
    return hermes_identity_anchor(agent) + "\n\n" + CLAUDE_TOOL_ROUTING_NOTE


def _bridge_only_hooks() -> Dict[str, Any]:
    """PreToolUse hook that pins execution to the Hermes MCP bridge.

    The billing classifier requires the built-in toolset to stay in context
    (see ``build_claude_agent_options``), but the built-ins must never run —
    the SDK's Bash/Read/Edit would execute outside Hermes' environments,
    checkpoints, approvals, and plugins.  A hook is the reliable choke point:
    unlike ``can_use_tool`` it fires for every call, including tools the CLI
    would auto-allow without a permission prompt (Read/Glob/Grep), which
    matters when Hermes routes execution to a remote backend and a local read
    would leak host files into the session.
    """
    from claude_agent_sdk import HookMatcher

    bridge_prefix = f"mcp__{MCP_SERVER_NAME}__"

    async def _deny_non_bridge_tools(hook_input, _tool_use_id, _context):
        name = str((hook_input or {}).get("tool_name") or "")
        if name.startswith(bridge_prefix):
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    f"Hermes owns tool execution. Use the {bridge_prefix}* "
                    f"equivalent of {name or 'this tool'} instead."
                ),
            }
        }

    return {"PreToolUse": [HookMatcher(matcher=None, hooks=[_deny_non_bridge_tools])]}


def build_claude_agent_options(
    agent,
    *,
    system_prompt: str,
    effective_task_id: Callable[[], str],
    cwd: str,
) -> Any:
    """Build ``ClaudeAgentOptions`` for this agent's session.

    Every field below is load-bearing:

    ``system_prompt`` (preset + Hermes' own identity section)
        The ``claude_code`` preset with Hermes' identity section appended —
        the same ``SOUL.md``-or-``DEFAULT_AGENT_IDENTITY`` text Hermes already
        puts at the top of its prompt for every other provider. Nothing is
        authored for this provider.

        Two measured constraints force this shape (decision record §11):
        replacing the preset, or appending Hermes' *full* prompt, is billed to
        the plan's extra-usage pool; and the full prompt delivered only as a
        first user turn decays — by turn 3 the preset's own identity line
        wins. The identity section is short enough to bill to the plan and
        durable enough to hold, and the rest of the prompt is unchanged and
        delivered in full one message later.
    ``tools`` unset (built-ins stay in context)
        Required by the same classifier finding.  The built-ins are *visible*
        but never *run*: a PreToolUse hook denies every non-bridge tool call
        and redirects Claude to the ``mcp__hermes__*`` equivalent, so Hermes'
        environments, checkpoints, approvals, and plugins still own all
        execution.  Denial happens client-side in the CLI and does not change
        the API-visible request shape.
    ``setting_sources=[]``
        Stops the CLI loading CLAUDE.md, skills, hooks, settings, and plugins
        a second time and competing with Hermes' context assembly.
    ``strict_mcp_config=True``
        Pins the toolset to the bridge; a user/project ``.mcp.json`` cannot
        widen it mid-conversation.
    ``stderr``
        The SDK inherits the parent's stderr unless a callback is registered,
        and under Electron that handle can be closed or unusable.
    ``session_store`` + ``session_store_flush="eager"``
        Mirrors Claude's own transcript into ``state.db`` so the session
        survives a process restart and can be rewound or branched.  Eager
        because Hermes' rewind boundary is a message UUID: a batched flush
        that lands after the turn would leave the most recent turn unmappable
        until the next one.  Deliberately NOT combined with
        ``enable_file_checkpointing`` — the SDK rejects the pair, and Hermes
        has its own checkpoint system (see the decision record).
    ``resume``
        The SDK session bound to this Hermes session, when there is one.  This
        is what keeps Claude's context, compaction state, and prompt cache
        alive across restarts instead of replaying history every turn.
    """
    from claude_agent_sdk import ClaudeAgentOptions

    # A zero-arg callable, so a long-lived server follows the turn's task id
    # and therefore the execution environment (local / Docker / SSH / Modal /
    # Daytona / Singularity) the tools run in.
    mcp_server = build_hermes_sdk_mcp_server(agent, effective_task_id)

    # Deliberately empty. The child environment is NOT assembled here: the SDK
    # merges `options.env` over a copy of os.environ, so it can override a key
    # but never delete one, and `ANTHROPIC_API_KEY=""` is still a set key.
    # Removal happens in agent.transports.claude_sanitized_transport, which
    # spawns the CLI from an explicitly-filtered environment. Anything added
    # here is re-filtered against the blocklist before the spawn.
    child_env: Dict[str, str] = {}

    from agent.claude_session_store import build_claude_session_store

    store = build_claude_session_store(getattr(agent, "_session_db", None))
    # Resolved on the turn thread by _prepare_sdk_session() before the session
    # is built, because a pending rewind/branch has to fork the stored
    # transcript first and that is async work the loop thread must not own.
    resume = getattr(agent, "_claude_sdk_resume_id", None) if store else None

    options: Dict[str, Any] = dict(
        system_prompt={
            "type": "preset",
            "preset": "claude_code",
            # Resolved once per session; byte-stable for its lifetime, which
            # is what the prompt cache requires.
            "append": claude_subscription_append(agent),
        },
        allowed_tools=bridged_allowed_tools(agent),
        mcp_servers={MCP_SERVER_NAME: mcp_server},
        strict_mcp_config=True,
        setting_sources=[],
        cwd=cwd,
        env=child_env,
        stderr=_make_stderr_logger(),
        model=getattr(agent, "model", None) or None,
        include_partial_messages=True,
        hooks=_bridge_only_hooks(),
    )
    if store is not None:
        options["session_store"] = store
        options["session_store_flush"] = "eager"
        if resume:
            options["resume"] = resume
    return ClaudeAgentOptions(**options)


# ---------------------------------------------------------------------------
# Durable sessions: binding, resume, fork, bootstrap, recovery
# ---------------------------------------------------------------------------
#
# Hermes owns the visible transcript; the SDK owns Claude's context and the
# resume cursor. Durability is therefore NOT "replay Hermes' history into
# Claude every turn" — that would rebuild the exact state the SDK is supposed
# to own and would throw away the upstream prompt cache on every message,
# which AGENTS.md treats as sacred. It is: mirror Claude's own transcript into
# state.db, remember which SDK session belongs to which Hermes session, and
# resume it.
#
# Canonical history is replayed exactly ONCE, and only when there is nothing
# to resume: a user switching into Claude mid-conversation, or a session that
# predates the mirror. After that bootstrap the binding exists and every
# later turn resumes.

#: Hard cap on stale-session recoveries for one Hermes session. A binding that
#: keeps going stale is a broken binding, not a transient one — after this
#: many attempts the runtime stops bootstrapping and just runs fresh turns,
#: so a permanently-unresumable session cannot spin forever.
MAX_SESSION_RECOVERIES = 3

#: How much canonical history a one-time bootstrap replays. Bounded because
#: this text becomes part of a single Claude user message.
BOOTSTRAP_MAX_MESSAGES = 200
BOOTSTRAP_MAX_CHARS = 60_000

# Substrings the CLI/SDK use when a --resume target no longer exists. Matched
# case-insensitively and only as a *second* signal — the primary one is
# structural (a connect that failed while a resume id was set).
_STALE_SESSION_MARKERS = (
    "no conversation found",
    "session not found",
    "no session found",
    "could not find session",
    "invalid session id",
)


def _session_cwd(agent) -> str:
    from agent.runtime_cwd import resolve_agent_cwd

    return getattr(agent, "session_cwd", None) or str(resolve_agent_cwd())


def claude_project_key(cwd: str) -> str:
    """The SDK's ``project_key`` for *cwd*.

    Must match what the SDK derives internally, or mirrored entries land under
    one key and resume looks for them under another.
    """
    try:
        from claude_agent_sdk import project_key_for_directory

        return project_key_for_directory(cwd)
    except Exception:  # pragma: no cover - optional extra absent
        return str(cwd)


def _run_async(coro: Any) -> Any:
    """Run *coro* to completion from Hermes' synchronous turn thread.

    A private loop, not the session's: the session's loop owns anyio streams
    and a subprocess and must not be borrowed for unrelated work, and this
    runs *before* that loop exists. The store only touches SQLite (through
    ``asyncio.to_thread``), so it has no loop affinity of its own.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        try:
            loop.close()
        finally:
            asyncio.set_event_loop(None)


def _fork_stored_session(
    store: Any, source_session_id: str, cwd: str, up_to_message_id: Optional[str]
) -> Optional[str]:
    """Fork a mirrored SDK session; return the new session id or None.

    Delegated to the SDK's ``fork_session_via_store`` rather than copying rows:
    a fork is not a byte copy.  It remaps every message UUID, rewrites
    ``sessionId`` on each entry, and stamps ``forkedFrom``, so the data has to
    pass through that transform once.  The source is left untouched, which is
    what keeps a rewound parent transcript immutable.
    """
    try:
        from claude_agent_sdk import fork_session_via_store
    except Exception:  # pragma: no cover - optional extra absent
        return None
    try:
        result = _run_async(
            fork_session_via_store(
                store,
                source_session_id,
                directory=cwd,
                up_to_message_id=up_to_message_id,
            )
        )
    except Exception as exc:
        logger.warning(
            "claude_agent_sdk fork of %s failed (%s); falling back to a fresh "
            "session bootstrapped from Hermes history",
            source_session_id,
            exc,
        )
        return None
    return getattr(result, "session_id", None)


def prepare_claude_sdk_session(agent, cwd: str) -> Dict[str, Any]:
    """Resolve what this turn should resume from, forking first if asked to.

    Returns ``{"resume": str|None, "bootstrap": bool, "recoveries": int}`` and
    caches the resume id on the agent, where
    :func:`build_claude_agent_options` reads it.  Runs on the turn thread
    before the session is built, because a pending rewind or branch has to
    rewrite the stored transcript before the CLI is pointed at it.
    """
    state: Dict[str, Any] = {"resume": None, "bootstrap": False, "recoveries": 0}
    db = getattr(agent, "_session_db", None)
    hermes_session = getattr(agent, "session_id", None)
    if db is None or not hermes_session:
        # No durable store (persistence disabled, a background-review fork, a
        # bare harness). The turn still runs; it just is not resumable.
        agent._claude_sdk_resume_id = None
        return state

    from agent.claude_session_store import RUNTIME, build_claude_session_store

    project_key = claude_project_key(cwd)
    agent._claude_project_key = project_key

    binding = db.get_provider_runtime_session(hermes_session, RUNTIME) or {}
    state["recoveries"] = int(binding.get("recovery_count") or 0)
    resume = binding.get("provider_session_id") or None
    pending_rewind = binding.get("pending_rewind_ordinal")
    pending_fork = binding.get("pending_fork_source")

    store = build_claude_session_store(db)
    if store is None:
        # Resuming requires materializing the stored transcript, so without a
        # store there is nothing to resume from — and a stale pending marker
        # must not survive as a resume id pointing at discarded history.
        agent._claude_sdk_resume_id = None
        return state
    if pending_fork or pending_rewind is not None:
        # A live session is already connected to the OLD transcript; the fork
        # below only changes what a *new* connection resumes. Retire it so the
        # next turn reconnects against the branch the user asked for.
        _retire_session(agent)

    if pending_fork:
        # Branch: an SDK fork, so the child keeps Claude's context and prompt
        # cache but cannot see or affect the parent's future turns.
        forked = _fork_stored_session(store, pending_fork, cwd, None)
        db.clear_provider_pending(hermes_session, RUNTIME)
        if forked:
            db.bind_provider_runtime_session(
                hermes_session, RUNTIME, forked, project_key=project_key
            )
            resume = forked
        else:
            resume = None

    elif pending_rewind is not None and resume:
        ordinal = int(pending_rewind)
        forked = None
        if ordinal > 0:
            boundary = db.get_provider_message_binding(hermes_session, ordinal) or {}
            boundary_uuid = boundary.get("fork_boundary_uuid")
            if boundary_uuid:
                forked = _fork_stored_session(store, resume, cwd, boundary_uuid)
            else:
                logger.info(
                    "claude_agent_sdk rewind to turn %s has no mirrored boundary "
                    "UUID; starting a fresh Claude session instead of forking",
                    ordinal,
                )
        db.clear_provider_pending(hermes_session, RUNTIME)
        db.prune_provider_message_bindings(hermes_session, ordinal)
        if forked:
            db.bind_provider_runtime_session(
                hermes_session, RUNTIME, forked, project_key=project_key
            )
            resume = forked
        else:
            # Rewinding to zero surviving turns, or no usable boundary: the
            # binding is dropped and the next turn starts clean. Not a
            # recovery — the user asked for this — so recovery_count is
            # untouched.
            db.clear_provider_runtime_session(hermes_session, RUNTIME)
            resume = None

    state["resume"] = resume
    # Bootstrap exactly once, and only when there is nothing to resume: with a
    # resume cursor the SDK already has the context, and replaying history on
    # top of it would duplicate the conversation AND break the prompt cache.
    if not resume and not binding.get("bootstrapped"):
        state["bootstrap"] = True
    agent._claude_sdk_resume_id = resume
    return state


def _render_bootstrap_message(message: Dict[str, Any]) -> Optional[str]:
    role = message.get("role")
    if role not in ("user", "assistant"):
        return None
    content = message.get("content")
    if isinstance(content, list):
        content = "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    text = str(content or "").strip()
    if not text:
        return None
    return f"{'User' if role == 'user' else 'Assistant'}: {text}"


def _hermes_system_prompt(agent) -> str:
    """The one Hermes prompt string this conversation uses, built at most once.

    AGENTS.md requires it to be byte-stable for the life of a conversation, so
    it is cached on the agent and shared by the session builder and the
    first-turn context bootstrap.
    """
    cached = getattr(agent, "_cached_system_prompt", None)
    if not cached:
        cached = agent._build_system_prompt()
        agent._cached_system_prompt = cached
    return cached


def claude_context_prefix(agent, system_prompt: str) -> str:
    """Deliver Hermes' identity, memory, and skills as the first user turn.

    In subscription mode the system prompt belongs to the ``claude_code``
    preset — appending Hermes' full prompt there re-bills the session to extra
    usage (decision record §11).  Hermes already has an established pattern for
    exactly this: ``agent/skill_commands.py`` injects skill content as a user
    message rather than the system prompt.  We reuse it, so nothing about
    Hermes' behavior is lost — persona, memory instructions, context files, and
    skills all still reach the model.

    Sent once per session, inside the same bootstrap turn as any prior history,
    so role alternation is untouched.  Every later turn resumes the SDK session,
    which already holds this content — so it is never re-sent and the prompt
    cache stays warm.
    """
    body = (system_prompt or "").strip()
    if not body:
        return ""
    # The framing has to be explicit about superseding the harness identity.
    # The system prompt belongs to the claude_code preset, and a mild "here is
    # some context" wrapper loses to it on identity questions once the body is
    # large enough for the identity line to be buried — measured at ~86k chars,
    # which is the real size once context files, skills, and memory load.
    return (
        "<operating_instructions>\n"
        "The system prompt that started this process describes the harness, "
        "not your role. For this conversation you are the assistant defined "
        "below: adopt its identity, persona, and instructions for every turn, "
        "including all later ones and including how you answer questions about "
        "who you are. Do not describe yourself as Claude Code.\n\n"
        f"{body}\n"
        "</operating_instructions>\n\n"
    )


def claude_bootstrap_prefix(messages: List[Dict[str, Any]]) -> str:
    """Render prior Hermes history for a one-time bootstrap into Claude.

    Deterministic and bounded.  Used only when a conversation reaches the
    Claude runtime with history but no SDK session to resume — a provider
    switch, or a session older than the mirror.  It is sent as part of the
    first user message so role alternation is untouched, and it happens once:
    the binding recorded after this turn makes every later turn a resume.

    Tool calls and tool results are omitted on purpose. Replaying them would
    imply Claude issued them, which it did not, and Claude's own transcript
    from here on is the SDK's to own.
    """
    rendered: List[str] = []
    for message in messages[-BOOTSTRAP_MAX_MESSAGES:]:
        line = _render_bootstrap_message(message)
        if line:
            rendered.append(line)
    if not rendered:
        return ""
    body = "\n\n".join(rendered)
    if len(body) > BOOTSTRAP_MAX_CHARS:
        body = "…\n\n" + body[-BOOTSTRAP_MAX_CHARS:]
    return (
        "<prior_conversation>\n"
        "This conversation started with a different model. The exchange so far "
        "is reproduced below for context; it is history, not a new request.\n\n"
        f"{body}\n"
        "</prior_conversation>\n\n"
    )


def _looks_stale(error: Optional[str]) -> bool:
    if not error:
        return False
    lowered = error.lower()
    return any(marker in lowered for marker in _STALE_SESSION_MARKERS)


def record_claude_session_binding(
    agent, projector: "ClaudeEventProjector", *, cwd: str, user_ordinal: int,
    watermark: int,
) -> None:
    """Persist the SDK session id and this turn's rewind boundary.

    Called after the turn.  Two writes:

    * the Hermes-session → SDK-session binding, which is what a later process
      resumes from, and
    * the visible user row → SDK message UUID mapping, resolved from the
      mirrored entries appended past *watermark*, which is what a later
      rewind forks at.

    Both are best-effort: a Claude turn that succeeded must not be reported as
    failed because bookkeeping did not land.
    """
    db = getattr(agent, "_session_db", None)
    hermes_session = getattr(agent, "session_id", None)
    sdk_session = projector.session_id
    if db is None or not hermes_session or not sdk_session:
        return

    from agent.claude_session_store import RUNTIME, is_visible_user_entry

    project_key = getattr(agent, "_claude_project_key", None) or claude_project_key(cwd)
    try:
        # bootstrapped=True in the same write: whatever this turn was, the
        # session now exists and every later turn resumes it.
        db.bind_provider_runtime_session(
            hermes_session,
            RUNTIME,
            sdk_session,
            project_key=project_key,
            bootstrapped=True,
        )
    except Exception:
        logger.debug("claude_agent_sdk session binding failed", exc_info=True)
        return

    if user_ordinal < 0:
        return
    try:
        db.record_provider_message_binding(
            hermes_session,
            user_ordinal,
            RUNTIME,
            sdk_session,
            project_key=project_key,
            after_entry_id=watermark,
        )
        # The mirror is asynchronous, so the UUID that opened this turn only
        # exists in the store now that the turn is over.
        entries = db.provider_transcript_entries_after(
            RUNTIME, project_key, sdk_session, watermark
        )
        for row_id, entry_uuid, entry in entries:
            if not is_visible_user_entry(entry) or not entry_uuid:
                continue
            db.set_provider_message_binding_uuids(
                hermes_session,
                user_ordinal,
                provider_message_uuid=entry_uuid,
                fork_boundary_uuid=db.provider_transcript_uuid_before(
                    RUNTIME, project_key, sdk_session, row_id
                ),
            )
            break
    except Exception:
        logger.debug("claude_agent_sdk message binding failed", exc_info=True)


# ---------------------------------------------------------------------------
# Event projection
# ---------------------------------------------------------------------------


def display_tool_name(name: str) -> str:
    """Strip the bridge's MCP namespacing for display.

    Users think of ``mcp__hermes__terminal`` as the ``terminal`` tool; the
    codex runtime makes the same call for its internal MCP server.
    """
    if name.startswith(MCP_TOOL_PREFIX):
        return name[len(MCP_TOOL_PREFIX):]
    return name


def _blocks(message: Any) -> List[Any]:
    content = getattr(message, "content", None)
    if isinstance(content, list):
        return content
    return []


def _block_kind(block: Any) -> str:
    return type(block).__name__


def _tool_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _tool_result_content(agent, tool_name: str, content: Any) -> Any:
    """Convert an SDK tool-result payload into Hermes tool-message content.

    MCP carries images as ``{"type": "image", "data", "mimeType"}`` blocks.
    Rebuild the OpenAI-style multimodal envelope so vision-capable models see
    the screenshot and text-only providers get the summary — the same
    downgrade the normal tool path applies.
    """
    if not isinstance(content, list):
        return _tool_result_text(content)

    parts: List[Dict[str, Any]] = []
    has_image = False
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            parts.append({"type": "text", "text": str(part.get("text", ""))})
        elif part.get("type") == "image" and part.get("data"):
            has_image = True
            mime = str(part.get("mimeType") or "image/png")
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{part['data']}",
                    },
                }
            )
    if not has_image:
        return _tool_result_text(content)

    envelope = {
        "_multimodal": True,
        "content": parts,
        "text_summary": "\n".join(
            p["text"] for p in parts if p.get("type") == "text"
        ),
    }
    resolve = getattr(agent, "_tool_result_content_for_active_model", None)
    if callable(resolve):
        try:
            return resolve(tool_name, envelope)
        except Exception:
            logger.debug("tool-result content resolution failed", exc_info=True)
    return envelope["text_summary"]


class ClaudeEventProjector:
    """Project one SDK response stream into Hermes callbacks + messages.

    Mirrors ``make_codex_app_server_event_bridge``: every Hermes-visible
    effect is fired from here so no surface needs Claude-specific rendering.
    Callback invocations are individually guarded — a buggy display callback
    must not tear down the turn.

    Instances are single-turn.  ``finalize()`` is idempotent so the projected
    messages can only ever be spliced into the conversation once, including
    when trailing events arrive after ``ResultMessage``.
    """

    def __init__(self, agent) -> None:
        self._agent = agent
        self.projected_messages: List[Dict[str, Any]] = []
        self.tool_iterations = 0
        self.session_id: Optional[str] = None
        self.terminal_reason: Optional[str] = None
        self.usage: Optional[Dict[str, Any]] = None
        self.total_cost_usd: Optional[float] = None
        self.result_text: Optional[str] = None
        self.compacted = False
        self.is_error = False
        self.error: Optional[str] = None
        self.stop_reason: Optional[str] = None
        self.mirror_errors: List[str] = []

        self._assistant_text_parts: List[str] = []
        self._streamed_text = False
        self._streamed_thinking = False
        self._pending_calls: List[Dict[str, Any]] = []
        self._pending_results: Dict[str, Any] = {}
        self._tool_started_at: Dict[str, float] = {}
        self._finalized = False

    # -- entry point -------------------------------------------------------

    def __call__(self, message: Any) -> None:
        try:
            self._dispatch(message)
        except Exception:
            logger.debug(
                "claude_agent_sdk projection failed for %s",
                type(message).__name__,
                exc_info=True,
            )

    def _dispatch(self, message: Any) -> None:
        # Dispatch on class name rather than isinstance so the projector
        # behaves identically against the real optional extra and the stand-in
        # the suite installs when it is absent.
        kind = type(message).__name__
        if kind == "StreamEvent":
            self._on_stream_event(message)
        elif kind == "AssistantMessage":
            self._on_assistant(message)
        elif kind == "UserMessage":
            self._on_user(message)
        elif kind == "MirrorErrorMessage":
            # A SystemMessage SUBCLASS, so a name-based dispatch has to name it
            # explicitly. Dispatched before the SystemMessage branch for the
            # same reason.
            self._on_mirror_error(message)
        elif kind == "SystemMessage":
            self._on_system(message)
        elif kind == "ResultMessage":
            self._on_result(message)

    # -- callbacks ---------------------------------------------------------

    def _fire(self, name: str, *args: Any, **kwargs: Any) -> None:
        callback = getattr(self._agent, name, None)
        if callback is None:
            return
        try:
            callback(*args, **kwargs)
        except Exception:
            logger.debug("%s raised on claude_agent_sdk turn", name, exc_info=True)

    # -- handlers ----------------------------------------------------------

    def _on_stream_event(self, message: Any) -> None:
        self._note_session_id(getattr(message, "session_id", None))
        event = getattr(message, "event", None)
        if not isinstance(event, dict):
            return
        if event.get("type") != "content_block_delta":
            return
        delta = event.get("delta")
        if not isinstance(delta, dict):
            return
        delta_type = delta.get("type")
        if delta_type == "text_delta":
            text = delta.get("text") or ""
            if text:
                self._streamed_text = True
                self._fire("_fire_stream_delta", text)
        elif delta_type == "thinking_delta":
            thinking = delta.get("thinking") or ""
            if thinking:
                self._streamed_thinking = True
                self._fire("_fire_reasoning_delta", thinking)

    def _on_assistant(self, message: Any) -> None:
        self._note_session_id(getattr(message, "session_id", None))
        stop_reason = getattr(message, "stop_reason", None)
        if stop_reason:
            self.stop_reason = str(stop_reason)
        error = getattr(message, "error", None)
        if error:
            self.is_error = True
            self.error = str(error)

        # A new assistant turn closes the previous one's tool round.
        self._flush_pending_tools()

        text_parts: List[str] = []
        thinking_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        for block in _blocks(message):
            kind = _block_kind(block)
            if kind == "TextBlock":
                text = getattr(block, "text", "") or ""
                if text:
                    text_parts.append(text)
            elif kind == "ThinkingBlock":
                thinking = getattr(block, "thinking", "") or ""
                if thinking:
                    thinking_parts.append(thinking)
            elif kind == "ToolUseBlock":
                tool_calls.append(
                    {
                        "id": str(getattr(block, "id", "") or f"claude_{uuid.uuid4().hex[:16]}"),
                        "name": str(getattr(block, "name", "") or "unknown"),
                        "input": getattr(block, "input", None) or {},
                    }
                )

        if thinking_parts and not self._streamed_thinking:
            self._fire("_fire_reasoning_delta", "\n".join(thinking_parts))
        text = "".join(text_parts)
        if text:
            self._assistant_text_parts.append(text)
            if not self._streamed_text:
                self._fire("_fire_stream_delta", text)
            if tool_calls and getattr(self._agent, "show_commentary", True):
                # Mid-turn narration alongside a tool call — same contract as
                # the codex runtime's interim commentary channel.
                self._fire(
                    "_emit_interim_assistant_message",
                    {"role": "assistant", "content": text},
                )

        self._append_assistant_message(
            text,
            tool_calls,
            reasoning="\n".join(thinking_parts) if thinking_parts else None,
        )

        for call in tool_calls:
            self._start_tool(call)

    def _on_user(self, message: Any) -> None:
        for block in _blocks(message):
            if _block_kind(block) != "ToolResultBlock":
                continue
            self._complete_tool(block)
        if self._pending_calls and all(
            call["id"] in self._pending_results for call in self._pending_calls
        ):
            self._flush_pending_tools()

    def _on_system(self, message: Any) -> None:
        data = getattr(message, "data", None)
        if isinstance(data, dict):
            self._note_session_id(data.get("session_id"))
        if getattr(message, "subtype", "") == "mirror_error":
            # The stand-in SDK the suite installs when the extra is absent may
            # deliver this as a plain SystemMessage.
            self._on_mirror_error(message)
            return
        if getattr(message, "subtype", "") == "compact_boundary":
            self.compacted = True

    def _on_mirror_error(self, message: Any) -> None:
        """A transcript-mirror batch was dropped after its retries.

        Not fatal and NOT a turn failure: Claude's own local JSONL is already
        durable, so the conversation continues intact — what is lost is a slice
        of Hermes' copy, which degrades a later resume or rewind. Surfaced as a
        Hermes warning rather than swallowed, because silently losing
        durability is exactly the failure mode a mirror is supposed to prevent.
        """
        detail = getattr(message, "error", "") or ""
        if not detail:
            data = getattr(message, "data", None)
            if isinstance(data, dict):
                detail = str(data.get("error") or "")
        self.mirror_errors.append(detail or "unknown mirror error")
        logger.warning(
            "claude_agent_sdk transcript mirror dropped a batch (session=%s): %s "
            "— this turn is unaffected, but the durable copy is now incomplete",
            self.session_id or "unknown",
            detail or "no detail reported",
        )
        self._fire(
            "_emit_status",
            "Claude transcript mirror failed — this session may not resume cleanly.",
        )

    def _on_result(self, message: Any) -> None:
        self._note_session_id(getattr(message, "session_id", None))
        self._flush_pending_tools()
        self.usage = getattr(message, "usage", None)
        self.total_cost_usd = getattr(message, "total_cost_usd", None)
        self.terminal_reason = getattr(message, "terminal_reason", None) or getattr(
            message, "subtype", None
        )
        result = getattr(message, "result", None)
        if isinstance(result, str) and result.strip():
            self.result_text = result
        if getattr(message, "is_error", False):
            self.is_error = True
            errors = getattr(message, "errors", None)
            if not self.error:
                self.error = (
                    "; ".join(str(e) for e in errors)
                    if isinstance(errors, list) and errors
                    else (result or "Claude Agent SDK reported an error")
                )

    # -- tool lifecycle ----------------------------------------------------

    def _start_tool(self, call: Dict[str, Any]) -> None:
        name = display_tool_name(call["name"])
        args = call["input"] if isinstance(call["input"], dict) else {}
        self._tool_started_at[call["id"]] = time.monotonic()
        self.tool_iterations += 1
        try:
            preview = json.dumps(args, ensure_ascii=False)[:120] if args else None
        except (TypeError, ValueError):
            preview = None
        self._fire("tool_progress_callback", "tool.started", name, preview, args)
        self._fire("tool_start_callback", call["id"], name, args)

    def _complete_tool(self, block: Any) -> None:
        tool_use_id = str(getattr(block, "tool_use_id", "") or "")
        call = next(
            (c for c in self._pending_calls if c["id"] == tool_use_id),
            None,
        )
        name = display_tool_name(call["name"]) if call else tool_use_id or "tool"
        args = (call or {}).get("input") or {}
        content = _tool_result_content(self._agent, name, getattr(block, "content", None))
        self._pending_results[tool_use_id] = content
        is_error = bool(getattr(block, "is_error", False))
        started = self._tool_started_at.pop(tool_use_id, None)
        duration = time.monotonic() - started if started is not None else None
        self._fire(
            "tool_progress_callback",
            "tool.completed",
            name,
            None,
            None,
            duration=duration,
            is_error=is_error,
            result=content,
        )
        self._fire("tool_complete_callback", tool_use_id, name, args, content)

    # -- message projection ------------------------------------------------

    def _append_assistant_message(
        self,
        text: str,
        tool_calls: List[Dict[str, Any]],
        *,
        reasoning: Optional[str] = None,
    ) -> None:
        if not text and not tool_calls:
            return
        if not tool_calls and self._can_merge_assistant_text():
            previous = self.projected_messages[-1]
            previous["content"] = f"{previous.get('content') or ''}\n\n{text}".strip()
            return
        message: Dict[str, Any] = {
            "role": "assistant",
            "content": text or None,
        }
        if reasoning:
            message["reasoning"] = reasoning
        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": display_tool_name(call["name"]),
                        "arguments": _dump_args(call["input"]),
                    },
                }
                for call in tool_calls
            ]
            self._pending_calls = list(tool_calls)
            self._pending_results = {}
        self.projected_messages.append(message)

    def _can_merge_assistant_text(self) -> bool:
        """Two assistant messages in a row would break role alternation."""
        if not self.projected_messages:
            return False
        last = self.projected_messages[-1]
        return last.get("role") == "assistant" and not last.get("tool_calls")

    def _flush_pending_tools(self) -> None:
        """Emit one tool message per pending call, in tool_calls order.

        Results can arrive out of order (parallel tools land as they finish),
        but a provider replaying this transcript expects each ``tool_calls``
        entry to be answered in the order it was issued, and every one of them
        to be answered at all — an unanswered id breaks the next request.
        """
        if not self._pending_calls:
            return
        for call in self._pending_calls:
            content = self._pending_results.get(call["id"])
            if content is None:
                content = "[no result returned by the Claude Agent SDK]"
            self.projected_messages.append(
                {
                    "role": "tool",
                    "name": display_tool_name(call["name"]),
                    "tool_name": display_tool_name(call["name"]),
                    "tool_call_id": call["id"],
                    "content": content,
                }
            )
        self._pending_calls = []
        self._pending_results = {}

    def _note_session_id(self, session_id: Any) -> None:
        if not session_id:
            return
        session_id = str(session_id)
        if session_id == self.session_id:
            return
        self.session_id = session_id
        # PR5 resumes from this. Stored on the agent so a rebuilt session
        # facade (or a durable SessionStore mirror) can pick it back up.
        self._agent._claude_sdk_session_id = session_id

    # -- terminal ----------------------------------------------------------

    def finalize(self) -> List[Dict[str, Any]]:
        """Close out the turn and return the projected messages, once.

        Idempotent: a second call returns an empty list so a caller that
        finalizes on both the normal and the error path cannot splice the same
        messages into the conversation twice.
        """
        if self._finalized:
            return []
        self._finalized = True
        self._flush_pending_tools()
        return self.projected_messages

    @property
    def final_text(self) -> str:
        if self.result_text:
            return self.result_text
        return "".join(self._assistant_text_parts)


def _dump_args(args: Any) -> str:
    try:
        return json.dumps(args or {}, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps({"_raw": str(args)}, ensure_ascii=False)


def make_claude_event_projector(agent) -> ClaudeEventProjector:
    """Build the per-turn projector bound to *agent*."""
    return ClaudeEventProjector(agent)


# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------


def _coerce_usage_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return max(int(value), 0)
    if isinstance(value, str):
        try:
            return max(int(value), 0)
        except ValueError:
            return 0
    return 0


def record_claude_usage(agent, projector: ClaudeEventProjector) -> Dict[str, Any]:
    """Translate the SDK's turn usage into Hermes accounting.

    Claude reports Anthropic-shaped usage (``input_tokens``,
    ``output_tokens``, ``cache_creation_input_tokens``,
    ``cache_read_input_tokens``).  Hermes' canonical prompt bucket is
    uncached input + cached input, same as every other Anthropic path.

    A turn with no usage still counts as one API call for session/status
    accounting, mirroring the codex app-server runtime.
    """
    agent.session_api_calls += 1

    usage = projector.usage
    if not isinstance(usage, dict) or not usage:
        compressor = getattr(agent, "context_compressor", None)
        if compressor is not None and getattr(
            compressor, "awaiting_real_usage_after_compression", False
        ):
            compressor.update_from_response({})
        _persist_api_call_only(agent)
        return {}

    from agent.usage_pricing import CanonicalUsage, estimate_usage_cost

    canonical_usage = CanonicalUsage(
        input_tokens=_coerce_usage_int(usage.get("input_tokens")),
        output_tokens=_coerce_usage_int(usage.get("output_tokens")),
        cache_read_tokens=_coerce_usage_int(usage.get("cache_read_input_tokens")),
        cache_write_tokens=_coerce_usage_int(usage.get("cache_creation_input_tokens")),
        reasoning_tokens=0,
        raw_usage=usage,
    )
    usage_dict = {
        "prompt_tokens": canonical_usage.prompt_tokens,
        "completion_tokens": canonical_usage.output_tokens,
        "total_tokens": canonical_usage.total_tokens,
        "input_tokens": canonical_usage.input_tokens,
        "output_tokens": canonical_usage.output_tokens,
        "cache_read_tokens": canonical_usage.cache_read_tokens,
        "cache_write_tokens": canonical_usage.cache_write_tokens,
        "reasoning_tokens": canonical_usage.reasoning_tokens,
    }

    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        try:
            compressor.update_from_response(usage_dict)
        except Exception:
            logger.debug("claude_agent_sdk usage update failed", exc_info=True)

    agent.session_prompt_tokens += canonical_usage.prompt_tokens
    agent.session_completion_tokens += canonical_usage.output_tokens
    agent.session_total_tokens += canonical_usage.total_tokens
    agent.session_input_tokens += canonical_usage.input_tokens
    agent.session_output_tokens += canonical_usage.output_tokens
    agent.session_cache_read_tokens += canonical_usage.cache_read_tokens
    agent.session_cache_write_tokens += canonical_usage.cache_write_tokens
    agent.session_reasoning_tokens += canonical_usage.reasoning_tokens

    cost_result = estimate_usage_cost(
        agent.model,
        canonical_usage,
        provider=agent.provider,
        base_url=agent.base_url,
        api_key="",
    )
    if cost_result.amount_usd is not None:
        agent.session_estimated_cost_usd += float(cost_result.amount_usd)
    agent.session_cost_status = cost_result.status
    agent.session_cost_source = cost_result.source

    if agent._session_db and agent.session_id:
        try:
            if not agent._session_db_created:
                agent._ensure_db_session()
            agent._session_db.queue_token_counts(
                agent.session_id,
                input_tokens=canonical_usage.input_tokens,
                output_tokens=canonical_usage.output_tokens,
                cache_read_tokens=canonical_usage.cache_read_tokens,
                cache_write_tokens=canonical_usage.cache_write_tokens,
                reasoning_tokens=canonical_usage.reasoning_tokens,
                estimated_cost_usd=float(cost_result.amount_usd)
                if cost_result.amount_usd is not None
                else None,
                cost_status=cost_result.status,
                cost_source=cost_result.source,
                billing_provider=agent.provider,
                billing_base_url=agent.base_url,
                billing_mode="subscription_included"
                if cost_result.status == "included"
                else None,
                model=agent.model,
                api_call_count=1,
            )
        except Exception as exc:
            logger.debug(
                "claude_agent_sdk token persistence failed (session=%s): %s",
                agent.session_id,
                exc,
            )

    return {
        **usage_dict,
        "last_prompt_tokens": canonical_usage.prompt_tokens,
        "estimated_cost_usd": float(cost_result.amount_usd)
        if cost_result.amount_usd is not None
        else None,
        "cost_status": cost_result.status,
        "cost_source": cost_result.source,
        # What the SDK says the same turn would have cost on the API. Under a
        # subscription this is not what the user is charged; it is reported so
        # the two billing sources can be compared, never summed.
        "claude_sdk_reported_cost_usd": projector.total_cost_usd,
    }


def _persist_api_call_only(agent) -> None:
    if not (agent._session_db and agent.session_id):
        return
    try:
        if not agent._session_db_created:
            agent._ensure_db_session()
        agent._session_db.queue_token_counts(
            agent.session_id,
            model=agent.model,
            billing_provider=agent.provider,
            billing_base_url=agent.base_url,
            billing_mode="subscription_included",
            api_call_count=1,
        )
    except Exception as exc:
        logger.debug(
            "claude_agent_sdk api-call persistence failed (session=%s): %s",
            agent.session_id,
            exc,
        )


def record_claude_compaction(agent, projector: ClaudeEventProjector) -> bool:
    """Record a Claude-native compaction boundary in Hermes state.

    The SDK owns the compacted context, so Hermes does not rewrite local
    transcript rows here — it records the boundary while preserving the
    visible transcript, exactly as the codex app-server runtime does.
    """
    if not projector.compacted:
        return False

    logger.info(
        "claude_agent_sdk compaction observed: session=%s sdk_session=%s",
        getattr(agent, "session_id", None) or "none",
        projector.session_id or "none",
    )
    try:
        from agent.conversation_compression import COMPACTION_STATUS

        agent._emit_status(COMPACTION_STATUS)
    except Exception:
        pass

    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        compressor.compression_count = getattr(compressor, "compression_count", 0) + 1
        record_boundary = getattr(type(compressor), "record_completed_compaction", None)
        if callable(record_boundary):
            # Claude owns this summary; a prior Hermes deterministic-fallback
            # flag must not leak into the native boundary's quality verdict.
            record_boundary(compressor, used_fallback=False)
        elif hasattr(compressor, "_verify_compaction_cleared_threshold"):
            compressor._verify_compaction_cleared_threshold = True

    agent._last_compaction_in_place = False
    try:
        if getattr(agent, "event_callback", None):
            agent.event_callback(
                "session:compress",
                {
                    "platform": getattr(agent, "platform", None) or "",
                    "session_id": getattr(agent, "session_id", None) or "",
                    "old_session_id": "",
                    "in_place": False,
                    "compression_count": getattr(compressor, "compression_count", 0)
                    if compressor is not None
                    else 0,
                    "runtime": RUNTIME_LABEL,
                    "claude_session_id": projector.session_id or "",
                },
            )
    except Exception:
        logger.debug("event_callback error on claude session:compress", exc_info=True)
    return True


# ---------------------------------------------------------------------------
# Turn runner
# ---------------------------------------------------------------------------


def _ensure_session(agent, effective_task_id: str) -> Any:
    """Return this agent's ``ClaudeAgentSession``, building it on first turn.

    ``effective_task_id`` is captured by reference through a mutable holder so
    the long-lived MCP server follows the *current* turn's task id, not the
    one that happened to be active when the session was built.
    """
    from agent.transports.claude_agent_session import ClaudeAgentSession

    holder = getattr(agent, "_claude_task_id_holder", None)
    if holder is None:
        holder = {"task_id": effective_task_id}
        agent._claude_task_id_holder = holder
    else:
        holder["task_id"] = effective_task_id

    session = getattr(agent, "_claude_session", None)
    if session is not None and not session.closed:
        return session

    cwd = _session_cwd(agent)

    # Reuse the exact string the normal path already built for this
    # conversation — never a second render. AGENTS.md requires the system
    # prompt to be byte-stable for the life of a conversation, and the SDK
    # keeps whatever we hand it at connect time for every later turn. The
    # rebuild below only covers a caller that reached here without going
    # through build_turn_context, and it caches back so it stays the one
    # string this session uses.
    system_prompt = getattr(agent, "_cached_system_prompt", None)
    if not system_prompt:
        system_prompt = agent._build_system_prompt()
        agent._cached_system_prompt = system_prompt

    def _options_factory() -> Any:
        return build_claude_agent_options(
            agent,
            system_prompt=system_prompt,
            effective_task_id=lambda: holder["task_id"],
            cwd=cwd,
        )

    from agent.transports.claude_sanitized_transport import build_sanitized_transport

    session = ClaudeAgentSession(
        options_factory=_options_factory,
        # The CLI is spawned from a sanitized environment, not from a copy of
        # os.environ — see the transport module for why options.env cannot do
        # this.
        transport_factory=build_sanitized_transport,
    )
    agent._claude_session = session
    # Connect here, not lazily inside the first run_turn: a connect that
    # fails while a resume id is set is the *structural* stale-session
    # signal, and the caller's `at_connect=True` handler is wrapped around
    # this function. Deferring the connect would route that failure through
    # the mid-turn handler instead, leaving recovery to depend entirely on
    # matching the CLI's error strings. Idempotent for an already-started
    # session.
    session.ensure_started()
    return session


def _retire_session(agent) -> None:
    # The next session re-proves its billing source rather than trusting a
    # verdict taken before the environment could have changed.
    agent._claude_billing_refusal = _UNSET
    session = getattr(agent, "_claude_session", None)
    if session is None:
        return
    try:
        session.close()
    except Exception:
        logger.debug("claude_agent_sdk session close failed", exc_info=True)
    agent._claude_session = None


def _mirror_watermark(agent, resume: Optional[str]) -> int:
    """Highest mirrored entry id before this turn (0 for a fresh session)."""
    db = getattr(agent, "_session_db", None)
    if db is None or not resume:
        return 0
    from agent.claude_session_store import RUNTIME

    try:
        return db.provider_transcript_watermark(
            RUNTIME,
            getattr(agent, "_claude_project_key", "") or "",
            resume,
        )
    except Exception:
        logger.debug("claude_agent_sdk watermark read failed", exc_info=True)
        return 0


def _can_recover(
    agent,
    sdk_state: Dict[str, Any],
    already_tried: bool,
    error: str,
    *,
    at_connect: bool,
) -> bool:
    """Should this failure be retried as a stale-session recovery?

    Four independent conditions, all required, so a broken binding can never
    loop: one recovery per turn, a persisted per-session ceiling, evidence
    that we were actually resuming something (a fresh session that fails is a
    real failure, not a stale cursor), and evidence that the resume is what
    broke — a connect that fails while resuming, or a CLI error that names a
    missing session. A mid-turn network blip does not qualify and is reported
    as the failure it is.
    """
    if already_tried or not sdk_state.get("resume"):
        return False
    if not (at_connect or _looks_stale(error)):
        return False
    if int(sdk_state.get("recoveries") or 0) >= MAX_SESSION_RECOVERIES:
        logger.warning(
            "claude_agent_sdk session recovery cap (%s) reached for session %s; "
            "not retrying",
            MAX_SESSION_RECOVERIES,
            getattr(agent, "session_id", None) or "none",
        )
        return False
    return True


def _recover_stale_session(
    agent,
    sdk_state: Dict[str, Any],
    messages: List[Dict[str, Any]],
    user_message: str,
) -> str:
    """Drop a stale binding, start clean, and bootstrap history once.

    Returns the prompt for the retry.  The user's turn is submitted exactly
    once overall — this replaces the failed attempt, it does not add one.
    """
    db = getattr(agent, "_session_db", None)
    hermes_session = getattr(agent, "session_id", None)
    logger.warning(
        "claude_agent_sdk session %s could not be resumed; starting a fresh "
        "Claude session and replaying Hermes history once",
        sdk_state.get("resume"),
    )
    if db is not None and hermes_session:
        from agent.claude_session_store import RUNTIME

        try:
            sdk_state["recoveries"] = db.clear_provider_runtime_session(
                hermes_session, RUNTIME
            )
        except Exception:
            logger.debug("claude_agent_sdk binding clear failed", exc_info=True)
    sdk_state["resume"] = None
    agent._claude_sdk_resume_id = None
    agent._claude_sdk_session_id = None
    _retire_session(agent)
    prefix = claude_bootstrap_prefix(messages[:-1] if messages else [])
    from agent.claude_sdk_input import prepend_text_to_prompt

    return prepend_text_to_prompt(user_message, prefix)


def _failure_result(
    agent,
    messages: List[Dict[str, Any]],
    *,
    final_response: str,
    error: str,
) -> Dict[str, Any]:
    user_interrupted = bool(getattr(agent, "_interrupt_requested", False))
    interrupt_message = (
        getattr(agent, "_interrupt_message", None) if user_interrupted else None
    )
    if user_interrupted:
        agent.clear_interrupt()
    return {
        "final_response": final_response,
        "messages": messages,
        "api_calls": 0,
        "completed": False,
        "partial": True,
        "failed": True,
        "interrupted": user_interrupted,
        **({"interrupt_message": interrupt_message} if interrupt_message else {}),
        "error": error,
    }


def run_claude_agent_sdk_turn(
    agent,
    *,
    user_message: str,
    original_user_message: Any,
    messages: List[Dict[str, Any]],
    effective_task_id: str,
    should_review_memory: bool = False,
) -> Dict[str, Any]:
    """Claude Agent SDK runtime path — hands the whole turn to the SDK.

    Called from ``run_conversation()`` when ``agent.api_mode ==
    "claude_agent_sdk"``.  Returns the same dict shape as the
    chat_completions path.
    """
    preflight_error = claude_runtime_preflight()
    if preflight_error:
        logger.info("claude_agent_sdk turn refused: %s", preflight_error)
        return _failure_result(
            agent,
            messages,
            final_response=preflight_error,
            error=preflight_error,
        )

    # Refuse rather than mis-bill: a turn that would be charged to an API
    # account instead of the user's plan does not run at all.
    billing_refusal = verify_claude_billing_for_agent(agent)
    if billing_refusal:
        logger.info("claude_agent_sdk turn refused: %s", billing_refusal)
        return _failure_result(
            agent,
            messages,
            final_response=billing_refusal,
            error=billing_refusal,
        )

    cwd = _session_cwd(agent)
    # Resolve the durable binding BEFORE the session is built: a pending
    # rewind or branch forks the mirrored transcript first, and the resume id
    # that comes out of it is what build_claude_agent_options() reads.
    sdk_state = prepare_claude_sdk_session(agent, cwd)

    # NOTE: the user message is ALREADY appended to messages by the standard
    # run_conversation() flow before the early return reaches us. Do NOT
    # append again — that would duplicate it.
    #
    # ``user_ordinal`` is this turn's position among the visible user rows;
    # it is the anchor a later rewind looks the SDK message UUID up by. Taken
    # from the in-memory transcript so it matches what the DB holds after the
    # turn-start flush.
    user_ordinal = (
        sum(
            1
            for m in messages
            if m.get("role") == "user" and not m.get("display_kind")
        )
        - 1
    )

    projector = make_claude_event_projector(agent)
    turn_error: Optional[str] = None
    should_retire = False
    attempted_recovery = False
    prompt = user_message
    if sdk_state.get("bootstrap"):
        # Exactly once per conversation: a provider switch into Claude, or a
        # session older than the mirror. Everything after this resumes.
        prefix = claude_bootstrap_prefix(messages[:-1] if messages else [])
        context = claude_context_prefix(agent, _hermes_system_prompt(agent))
        if prefix or context:
            logger.info(
                "claude_agent_sdk bootstrapping Hermes context%s into a new "
                "session (session=%s)",
                " and canonical history" if prefix else "",
                getattr(agent, "session_id", None) or "none",
            )
            from agent.claude_sdk_input import prepend_text_to_prompt

            prompt = prepend_text_to_prompt(user_message, context + prefix)

    while True:
        try:
            session = _ensure_session(agent, effective_task_id)
        except Exception as exc:
            # A connect that fails while a resume id is set is the structural
            # signature of a stale or deleted SDK session: the CLI was asked
            # to resume a transcript that no longer resolves.
            if _can_recover(
                agent, sdk_state, attempted_recovery, str(exc), at_connect=True
            ):
                attempted_recovery = True
                prompt = _recover_stale_session(
                    agent, sdk_state, messages, user_message
                )
                continue
            logger.exception("claude_agent_sdk session construction failed")
            detail = f"Claude Agent SDK could not start: {exc}"
            _retire_session(agent)
            return _failure_result(
                agent, messages, final_response=detail, error=str(exc)
            )

        watermark = _mirror_watermark(agent, sdk_state.get("resume"))
        turn_error = None
        try:
            session.run_turn(
                prompt,
                on_message=projector,
                timeout=DEFAULT_TURN_TIMEOUT_SECONDS,
            )
        except TimeoutError as exc:
            turn_error = str(exc)
            should_retire = True
            logger.warning("claude_agent_sdk turn timed out: %s", exc)
        except Exception as exc:
            turn_error = str(exc)
            should_retire = True
            if _can_recover(
                agent, sdk_state, attempted_recovery, str(exc), at_connect=False
            ):
                attempted_recovery = True
                should_retire = False
                projector = make_claude_event_projector(agent)
                prompt = _recover_stale_session(
                    agent, sdk_state, messages, user_message
                )
                continue
            logger.exception("claude_agent_sdk turn failed")
        break

    projected = projector.finalize()
    if projector.session_id:
        session.note_session_id(projector.session_id)
        record_claude_session_binding(
            agent,
            projector,
            cwd=cwd,
            user_ordinal=user_ordinal,
            watermark=watermark,
        )
    if turn_error is None and projector.error:
        turn_error = projector.error

    # A wedged or crashed client must not be reused: the next turn respawns
    # the CLI from scratch rather than riding a broken subprocess.
    if should_retire:
        _retire_session(agent)

    user_interrupted = bool(getattr(agent, "_interrupt_requested", False))
    interrupt_message = (
        getattr(agent, "_interrupt_message", None) if user_interrupted else None
    )
    if user_interrupted:
        agent.clear_interrupt()

    if projected:
        messages.extend(projected)
        # This path is an early return that bypasses conversation_loop, whose
        # per-step _persist_session() calls would otherwise flush these rows.
        # The inbound user turn was already flushed at turn start and the
        # flush dedups via _DB_PERSISTED_MARKER, so this writes ONLY the new
        # rows — which is what lets us report agent_persisted=True below and
        # keep the gateway from re-INSERTing the user turn (#860 / #42039).
        if getattr(agent, "_session_db", None) is not None:
            try:
                flushed = agent._flush_messages_to_session_db(messages)
            except Exception:
                flushed = False
                logger.warning(
                    "claude_agent_sdk projected-message flush failed", exc_info=True
                )
            if flushed is False:
                logger.warning(
                    "claude_agent_sdk turn was delivered but could NOT be "
                    "persisted to the session DB (session=%s) — this turn "
                    "will be missing after restart/resume",
                    getattr(agent, "session_id", None),
                )

    # _turns_since_memory / _user_turn_count are already incremented in the
    # run_conversation() pre-loop block; only the skill counter needs an
    # explicit bump because the tool-iteration loop is bypassed here.
    agent._iters_since_skill = (
        getattr(agent, "_iters_since_skill", 0) + projector.tool_iterations
    )
    record_claude_compaction(agent, projector)
    usage_result = record_claude_usage(agent, projector)

    should_review_skills = False
    if (
        agent._skill_nudge_interval > 0
        and agent._iters_since_skill >= agent._skill_nudge_interval
        and "skill_manage" in agent.valid_tool_names
    ):
        should_review_skills = True
        agent._iters_since_skill = 0

    final_text = projector.final_text
    completed = turn_error is None and not user_interrupted and not projector.is_error

    if completed:
        try:
            agent._sync_external_memory_for_turn(
                original_user_message=original_user_message,
                final_response=final_text,
                interrupted=False,
                messages=messages,
            )
        except Exception:
            logger.debug("external memory sync raised", exc_info=True)

    if final_text and completed and (should_review_memory or should_review_skills):
        try:
            agent._spawn_background_review(
                messages_snapshot=list(messages),
                review_memory=should_review_memory,
                review_skills=should_review_skills,
            )
        except Exception:
            logger.debug("background review spawn raised", exc_info=True)

    if turn_error and not final_text:
        final_text = f"Claude Agent SDK turn failed: {turn_error}"

    return {
        "final_response": final_text,
        "messages": messages,
        "api_calls": 1,
        "completed": completed,
        "partial": not completed,
        "interrupted": user_interrupted,
        **({"interrupt_message": interrupt_message} if interrupt_message else {}),
        "error": turn_error,
        "agent_persisted": True,
        "claude_session_id": projector.session_id,
        "claude_terminal_reason": projector.terminal_reason,
        **usage_result,
    }


__all__ = [
    "BOOTSTRAP_MAX_CHARS",
    "BOOTSTRAP_MAX_MESSAGES",
    "MAX_SESSION_RECOVERIES",
    "RUNTIME_LABEL",
    "ClaudeEventProjector",
    "build_claude_agent_options",
    "claude_bootstrap_prefix",
    "claude_project_key",
    "claude_runtime_preflight",
    "display_tool_name",
    "make_claude_event_projector",
    "prepare_claude_sdk_session",
    "record_claude_compaction",
    "record_claude_session_binding",
    "record_claude_usage",
    "run_claude_agent_sdk_turn",
    "verify_claude_billing_for_agent",
]
