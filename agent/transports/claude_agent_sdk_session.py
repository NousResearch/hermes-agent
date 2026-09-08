"""Session adapter for the claude-agent-sdk runtime.

Owns one Claude Agent SDK client per Hermes session — the structural twin of
``codex_app_server_session.py``, with the Codex JSON-RPC subprocess replaced
by Anthropic's official ``claude-agent-sdk`` (which manages the Claude Code
CLI subprocess, its agent loop, and — critically — **subscription OAuth**:
``CLAUDE_CODE_OAUTH_TOKEN`` / the ``~/.claude`` credential store by default;
known metered sources and Extra Usage fail closed unless the operator opts in).
See GitHub issue #25267.

Lifecycle:
    session = ClaudeAgentSdkSession(cwd="/home/x/proj", model="claude-opus-4-8")
    session.ensure_started()                       # loop thread + SDK connect
    result = session.run_turn(user_input="hello")  # blocks until ResultMessage
    session.close()                                # disconnect + stop loop

Threading model: the SDK is async-first, but AIAgent.run_conversation() is
synchronous (the same constraint that made CodexAppServerClient thread-based).
The adapter owns a dedicated background thread running one asyncio event loop
for the whole session lifetime; every SDK coroutine is marshaled onto it with
``asyncio.run_coroutine_threadsafe`` and awaited with a timeout, so the SDK
client keeps stable loop affinity and ``run_turn`` stays blocking.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any, Callable, Optional

from agent.transports.claude_agent_sdk_session_availability import (
    _safe_sdk_error_text,
    check_claude_sdk_available,
)
from agent.transports.claude_agent_sdk_session_sanitize import (
    safe_sdk_tool_presentation_from_canonical,
    validate_canonical_sdk_request_serialization,
)
from agent.transports.claude_agent_sdk_session_child import (
    ClaudeSdkChildProcessMixin,
    _SDK_DISCONNECT_TIMEOUT_S,
    _force_kill_sdk_child,
    _own_sdk_child_process,
    _sdk_child_pid,
)
from agent.transports.claude_agent_sdk_session_input import (
    _coerce_turn_input,
    _sdk_image_content_block,
    _sdk_user_message_stream,
)
from agent.transports.claude_agent_sdk_session_config import (
    _HERMES_TO_SDK_PERMISSION_MODE,
    _build_hermes_tools_mcp_config,
    _configured_hybrid_exclude,
    _configured_max_buffer_size,
    _configured_permission_mode,
    _configured_setting_sources,
    _http_mcp_entries_from_config,
    _is_subscription_oauth_token,
    _mcp_registry_tool_specs,
    _provider_config,
    _provider_flag,
    _sdk_env_overrides,
)
from agent.transports.claude_agent_sdk_session_watchdog import (
    _StreamEnd,
    _TurnWatch,
    _swallow_interrupt_result,
    _swallow_steer_result,
)
from agent.transports.claude_agent_sdk_session_billing import (
    ClaudeSdkBillingMixin,
)
from agent.transports.claude_agent_sdk_session_compaction import (
    ClaudeSdkCompactionMixin,
)
from agent.transports.claude_agent_sdk_session_notify import (
    ClaudeSdkNotifyMixin,
)
from agent.transports.claude_agent_sdk_session_permissions import (
    ClaudeSdkPermissionsMixin,
)
from agent.transports.claude_agent_sdk_session_turn import (
    ClaudeSdkTurnMixin,
)

logger = logging.getLogger(__name__)

# The names other packages import through this facade (the class, the runtime's and the
# auxiliary client's config/input entry points, the approval gateway's validators).
__all__ = [
    "ClaudeAgentSdkSession",
    "_coerce_turn_input",
    "_provider_config",
    "_provider_flag",
    "_sdk_env_overrides",
    "_sdk_image_content_block",
    "_sdk_user_message_stream",
    "safe_sdk_tool_presentation_from_canonical",
    "validate_canonical_sdk_request_serialization",
]


class ClaudeAgentSdkSession(ClaudeSdkTurnMixin, ClaudeSdkPermissionsMixin, ClaudeSdkNotifyMixin, ClaudeSdkCompactionMixin, ClaudeSdkBillingMixin, ClaudeSdkChildProcessMixin):
    """One SDK client per Hermes session, lifetime owned by AIAgent.

    Not thread-safe from the caller's side — one caller drives it at a time,
    matching AIAgent.run_conversation(). Internally owns a loop thread."""

    def __init__(
        self,
        *,
        cwd: Optional[str] = None,
        model: Optional[str] = None,
        permission_mode: Optional[str] = None,
        system_prompt_append: Optional[str] = None,
        approval_callback: Optional[Callable[..., str]] = None,
        approval_bypass_provider: Optional[Callable[[], bool]] = None,
        on_tool_started: Optional[Callable[[str, str, dict], None]] = None,
        max_budget_usd: Optional[float] = None,
        client_factory: Optional[Callable[..., Any]] = None,
        include_hermes_tools: bool = True,
        hermes_session_id: Optional[str] = None,
        resume_session_id: Optional[str] = None,
        on_stream_delta: Optional[Callable[[str], None]] = None,
        on_interim_assistant: Optional[Callable[[str], None]] = None,
        on_tool_iteration: Optional[Callable[[], None]] = None,
        on_unsolicited_result: Optional[Callable[[list[str]], None]] = None,
        on_compaction: Optional[Callable[[str], None]] = None,
        on_compact_boundary: Optional[Callable[[str], None]] = None,
        # Hybrid MCP bridge (ported from PR #56413): the explicit config
        # opt-in plus both `agent` and `tools` activate in-process servers
        # exposing the full Hermes registry (including proxified third-party
        # MCPs). Missing either input or the opt-in preserves fcava's original
        # stdio-only behavior.
        agent: Optional[Any] = None,
        tools: Optional[list[dict]] = None,
    ) -> None:
        self._cwd = cwd or os.getcwd()
        self._model = model
        requested_permission_mode = (
            permission_mode
            or _configured_permission_mode()
            or _HERMES_TO_SDK_PERMISSION_MODE.get(
                os.environ.get("HERMES_TERMINAL_SECURITY_MODE", "auto"),
                "default",
            )
        )
        self._sdk_approval_bypass_requested = (
            requested_permission_mode == "bypassPermissions"
        )
        # The SDK auto-approves before can_use_tool in bypassPermissions mode.
        # Keep Hermes in callback-capable mode and carry the bypass intent into
        # this mandatory session wrapper, where canonical validation and
        # immutable hardline/sudo/user-deny floors run before every selected
        # gateway, CLI, ACP, plugin, or custom callback.
        self._permission_mode = (
            "default"
            if self._sdk_approval_bypass_requested
            else requested_permission_mode
        )
        self._system_prompt_append = system_prompt_append
        self._approval_callback = approval_callback
        self._approval_bypass_provider = approval_bypass_provider
        self._on_tool_started = on_tool_started
        self._on_compaction = on_compaction
        self._on_compact_boundary = on_compact_boundary
        self._max_budget_usd = max_budget_usd
        self._client_factory = client_factory  # test seam
        self._include_hermes_tools = include_hermes_tools
        # Hermes-side session id, exported to the hermes-tools MCP subprocess
        # so the stateless session_search shim can exclude its own lineage.
        self._hermes_session_id = hermes_session_id
        # SDK-side session id to resume (#25267 continuity). Verified live:
        # resume restores the model context and keeps the SAME session id; a
        # stale id fails the session start (the caller retires + retries
        # fresh).
        self._resume_session_id = resume_session_id
        # Display-only partial-text consumer (W4 streaming). Deltas never
        # enter the projected transcript; the gateway's stream consumer
        # handles rate limiting and the already_sent final-send dedup.
        self._turn_callback_lock = threading.RLock()
        self._on_stream_delta = on_stream_delta
        # Hybrid MCP bridge inputs (see param docstring above).
        self._agent = agent
        self._tools = tools
        # Completed assistant prose that accompanies a tool call is a true
        # interim status, not final-answer text. It follows the gateway's
        # existing commentary callback so platforms can render it separately.
        # These callbacks are refreshed before EVERY run_turn: a session may
        # safely span several Hermes turns, while visibility must stay scoped
        # to the current one.
        self._on_interim_assistant = on_interim_assistant
        self._on_tool_iteration = on_tool_iteration

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._client: Any = None
        self._session_id: Optional[str] = None
        self._interrupt_event = threading.Event()
        self._closed = False
        # Activity evidence for the in-flight turn (None between turns).
        self._turn_watch: Optional[_TurnWatch] = None
        # Snapshot the streaming posture once: the quiet-watchdog default is
        # derived from it, and a mid-session config edit must not flip the
        # watchdog's semantics while the live client still has the old
        # include_partial_messages option.
        self._streaming = _provider_flag("streaming")
        # Stream ownership (see _reader_loop). The reader task is the ONLY
        # consumer of the SDK stream; `_turn_inbox` is non-None exactly while a
        # turn is in flight, which is what makes "unsolicited" decidable.
        self._reader_task: Any = None
        self._turn_inbox: Any = None
        self._turn_claims: Any = None
        self._turn_claim_requested = False
        self._unsolicited_results = 0
        self._stream_ended: Optional[_StreamEnd] = None
        # Delivery half of the stream-ownership fix (dasbrow-hermes-coder#2):
        # a finished background Agent task's answer is captured here and
        # handed to the callback — it must never enter a turn's result, but
        # dropping it entirely left completed work silently undelivered
        # (observed live 2026-07-29: answers sat in the CLI session until the
        # operator poked). No callback wired = the historical drop semantics.
        self._on_unsolicited_result = on_unsolicited_result
        self._unsolicited_text: list[str] = []
        self._unsolicited_delivered: set[str] = set()
        # One immutable billing posture per child. The startup environment,
        # post-start evidence guard, and accounting label must never disagree
        # because config.yaml changed while a long-lived SDK session was live.
        self._allow_metered = _provider_flag("allow_metered_key")
        self._billing_evidence: dict[str, Any] = {}
        self._billing_guard_error: Optional[str] = None

    # ---------- lifecycle ----------

    def ensure_started(self) -> str:
        """Start the loop thread, build the SDK client, connect. Idempotent —
        returns the session marker (SDK session ids arrive on first result)."""
        if self._client is not None:
            return self._session_id or "pending"
        # Hard default, enforced fail-closed: this provider targets the Claude
        # SUBSCRIPTION. If a metered ANTHROPIC_API_KEY is present the
        # underlying CLI would silently prefer it — refuse to start instead.
        # ANTHROPIC_TOKEN is in the set because it alone authenticates
        # Hermes' NATIVE Anthropic lane (x-api-key, metered) — but Hermes
        # also persists subscription setup tokens there, so only an
        # API-key-shaped value counts as a metered vector.
        for metered_var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_TOKEN"):
            value = os.environ.get(metered_var)
            if not value or self._allow_metered:
                continue
            if metered_var == "ANTHROPIC_TOKEN" and _is_subscription_oauth_token(value):
                continue
            raise RuntimeError(
                f"claude-agent-sdk runtime refuses to start: {metered_var} "
                "is set, which would silently switch billing from the "
                "Claude subscription to metered API usage. Unset it, or "
                "set agent.claude_agent_sdk.allow_metered_key: true in "
                "config.yaml to explicitly allow it."
            )
        if self._client_factory is None:
            ok, msg = check_claude_sdk_available()
            if not ok:
                raise RuntimeError(msg)

        self._start_loop_thread()
        client = self._build_client()
        # Assign BEFORE connect: a connect timeout/cancel leaves a
        # half-connected client whose CLI subprocess close() must still reap
        # — a None _client would skip disconnect and orphan it.
        self._client = client
        self._run_coro(client.connect(), timeout=60.0)
        # From here on exactly ONE consumer owns the SDK stream (_reader_loop).
        # Started after connect so the client is live, before any turn so no
        # message can arrive unowned.
        self._start_reader()
        logger.info(
            "claude-agent-sdk session started: model=%s mode=%s cwd=%s",
            self._model or "cli-default",
            self._permission_mode,
            self._cwd,
        )
        return self._session_id or "pending"

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Cancel the reader BEFORE disconnect so it unwinds on a live stream
        # instead of raising against a torn-down one.
        self._stop_reader()
        if self._client is not None and self._loop is not None:
            # Budget must exceed the SDK transport's own escalation ladder
            # (stdin lock + graceful wait + SIGTERM + SIGKILL); on timeout the
            # loop thread dies below and its shielded reap never finishes, so
            # fall back to killing the child directly.
            pid = _sdk_child_pid(self._client)
            # Capture the psutil identity before waiting on disconnect. If the
            # original child exits and its PID is reused during the timeout,
            # this object will fail psutil's identity check instead of
            # resolving the reused PID as a fresh kill target.
            child_process = _own_sdk_child_process(pid) if pid else None
            try:
                self._run_coro(
                    self._client.disconnect(), timeout=_SDK_DISCONNECT_TIMEOUT_S
                )
            except Exception:  # pragma: no cover - best-effort cleanup
                if child_process is not None:
                    _force_kill_sdk_child(pid, process=child_process)
            self._client = None
        self._stop_loop_thread()

    def __enter__(self) -> "ClaudeAgentSdkSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------- interrupt ----------

    def consume_interrupt(self) -> None:
        """Clear a pending interrupt signal — the caller honored it through
        another path (e.g. the runtime's cold-agent short-circuit)."""
        self._interrupt_event.clear()

    def request_interrupt(self) -> None:
        """Idempotent: signal the active turn loop to interrupt and unwind."""
        self._interrupt_event.set()
        if self._client is not None and self._loop is not None:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._client.interrupt(), self._loop
                )
                future.add_done_callback(_swallow_interrupt_result)
            except Exception as exc:  # pragma: no cover
                logger.debug(
                    "SDK interrupt scheduling failed: %s",
                    _safe_sdk_error_text(exc),
                )

    def steer(self, text: str) -> bool:
        """Inject a mid-turn user message using the SDK's own streaming input.

        Why this exists: on this lane the SDK owns tool execution, so Hermes'
        tool-batch drain points (agent/tool_executor.py) are never reached and
        a /steer would strand in ``_pending_steer`` until the turn finalizer
        handed it back — where the gateway only redelivers it if no other
        message is queued. Steers were therefore silently lost.

        ``ClaudeSDKClient.query()`` is documented as "send a new request in
        streaming mode", and we already connect in streaming mode (``connect()``
        with no prompt), so a second query() on the LIVE session is the SDK's
        own steering contract — not an interrupt and not a session restart,
        which is what the claude-cli-live transport has to do.

        Verified 2026-08-15 against SDK 0.2.120: a query() issued while a tool
        call was in flight did not raise and was honored at the next turn
        boundary ~3s later. Note it also produces a SECOND ResultMessage once
        the superseded work unwinds; that arrives unclaimed and is handled by
        the existing unsolicited-result path (``_on_unsolicited_result``), so
        ``deliver_background_results`` is load-bearing here.

        Returns True only when the steer was actually scheduled onto a live
        turn. False means "not applicable" — the caller must fall back to the
        ordinary pending-steer stash. Crucially this is never both: returning
        True suppresses the stash, so a steer cannot be delivered twice.
        """
        if not text or not text.strip():
            return False
        # No claimed turn means there is nothing to steer INTO. Sending anyway
        # would open a fresh unclaimed turn on this session whose output the
        # reader routes to the unsolicited path — a reply appearing from
        # nowhere. Decline instead and let the caller queue it normally.
        if self._turn_inbox is None:
            return False
        client = self._client
        loop = self._loop
        if client is None or loop is None:
            return False
        cleaned = text.strip()
        try:
            future = asyncio.run_coroutine_threadsafe(client.query(cleaned), loop)
            future.add_done_callback(_swallow_steer_result)
        except Exception:
            logger.debug("SDK steer scheduling failed", exc_info=True)
            return False
        logger.info(
            "claude-agent-sdk: steered live turn via streaming input (%d chars)",
            len(cleaned),
        )
        return True

    def build_option_fields(self) -> dict[str, Any]:
        """The ClaudeAgentOptions field dict — plain data so tests can assert
        on it without importing the SDK."""
        mcp_servers: dict[str, Any] = {}
        # Hybrid in-process MCP bridge (ported from PR #56413) — exposes the
        # full Hermes tool registry, including proxified third-party MCP
        # servers, which the stdio `hermes-tools` wrapper cannot reach
        # because credentials live in the gateway process env and don't
        # propagate to the subprocess. Activation requires ALL of:
        #   1. the operator opted in via
        #      ``agent.claude_agent_sdk.hybrid_mcp_bridge: true``
        #      (checked here as well as at the runtime call site);
        #   2. the caller supplied both ``agent`` and ``tools``.
        # The bridge splits into two in-process servers:
        #   - ``hermes-tools`` — stdio-legacy names (see
        #     ``HERMES_TOOLS_LEGACY_NAMES``). Preserves operator grants
        #     stored in ``~/.claude/settings.json`` that key on
        #     ``mcp__hermes-tools__<tool>``: a box on
        #     ``permission_mode: default`` would otherwise face an
        #     approval storm for tools it already granted.
        #   - ``hermes-hybrid`` — everything else (proxified MCPs +
        #     agent-level tools).
        # The exclude list from config is applied to BOTH buckets so an
        # operator can keep the wide bridge for proxified MCPs without
        # inheriting a specific tool (delegate_task, cron_*, terminal, ...).
        hybrid_active = False
        hybrid_opted_in = _provider_flag("hybrid_mcp_bridge", default=False)
        if hybrid_opted_in and self._agent is not None and self._tools:
            try:
                from agent.transports.hermes_hybrid_mcp import (
                    HERMES_TOOLS_SERVER,
                    HYBRID_SERVER,
                    build_hybrid_mcp_server,
                )
                from agent.transports.hermes_tool_exposure import (
                    HERMES_TOOLS_LEGACY_NAMES,
                )

                exclude_names = _configured_hybrid_exclude()
                mcp_servers[HERMES_TOOLS_SERVER] = build_hybrid_mcp_server(
                    self._agent,
                    self._tools,
                    server_name=HERMES_TOOLS_SERVER,
                    only_names=HERMES_TOOLS_LEGACY_NAMES,
                    exclude_names=exclude_names,
                )
                # Registry diff rides the "extras" bucket only: the legacy
                # bucket stays byte-identical so ``mcp__hermes-tools__*``
                # grants keep matching. When the diff is empty the snapshot
                # object is forwarded untouched, so a deployment with nothing
                # late-registered builds exactly the bridge it built before.
                registry_extra = _mcp_registry_tool_specs(
                    self._agent, self._tools
                )
                hybrid_tools = (
                    list(self._tools) + registry_extra
                    if registry_extra
                    else self._tools
                )
                mcp_servers[HYBRID_SERVER] = build_hybrid_mcp_server(
                    self._agent,
                    hybrid_tools,
                    server_name=HYBRID_SERVER,
                    exclude_names=(
                        list(exclude_names) + list(HERMES_TOOLS_LEGACY_NAMES)
                    ),
                )
                hybrid_active = True
            except Exception as exc:  # noqa: BLE001
                # Never break session start on hybrid bridge failure — the
                # stdio wrapper still provides the curated tool set.
                logger.warning(
                    "hybrid MCP bridge failed to build (%s) — falling back "
                    "to stdio hermes-tools only",
                    exc,
                )
        # The stdio ``hermes-tools`` wrapper exposes ~25 curated tools that
        # the hybrid bridge already re-exposes under the same server name
        # (``hermes-tools``) when active — registering both concurrently
        # would send Claude two copies of every curated tool under the same
        # server. Skip stdio when hybrid is active. (Hybrid failure above
        # falls back to stdio via ``hybrid_active`` staying False.)
        if self._include_hermes_tools and not hybrid_active:
            mcp_servers["hermes-tools"] = _build_hermes_tools_mcp_config(
                hermes_session_id=self._hermes_session_id
            )

        # Headerless third-party HTTP MCPs configured in Hermes (config.yaml
        # mcp_servers.<name>.url) can be exposed directly to the SDK because
        # the registry snapshot may not contain their late/proxified tools.
        # This is part of the SAME wide-surface security choice as the hybrid
        # bridge, never an independent back door: discovery runs only after
        # the explicit opt-in passed and both in-process bridge buckets built
        # successfully. Header-bearing entries are refused by the loader
        # because the SDK puts its MCP config in the Claude CLI argv.
        if hybrid_active:
            for entry_name, entry_cfg in _http_mcp_entries_from_config().items():
                if entry_name in mcp_servers:
                    continue
                mcp_servers[entry_name] = entry_cfg

        system_prompt: Any = {"type": "preset", "preset": "claude_code"}
        if self._system_prompt_append:
            system_prompt = {
                "type": "preset",
                "preset": "claude_code",
                "append": self._system_prompt_append,
            }

        # This wrapper is Hermes' mandatory permission-policy owner, including
        # when runtime callback selection legitimately produced no downstream
        # callback.  Callbackless non-bypass sessions intentionally fail closed
        # here instead of falling through to SDK-native prompting/permission
        # behavior; bounded readers and trusted bypass still resolve only after
        # canonical validation, correlation validation, and immutable floors.
        can_use_tool = self._make_can_use_tool()

        # Metered-vector scrub for the spawned CLI (see _METERED_ENV_DENYLIST).
        # agent.claude_agent_sdk.allow_metered_key: true is the operator's
        # explicit "bill me metered" opt-in (the same flag the startup guard
        # honors), so it disables the scrub too — otherwise the documented
        # escape hatch would hand the CLI an environment with the key blanked.
        env_overrides = _sdk_env_overrides(
            metered_allowed=self._allow_metered
        )

        fields = {
            "model": self._model,
            "cwd": self._cwd,
            "permission_mode": self._permission_mode,
            "system_prompt": system_prompt,
            "mcp_servers": mcp_servers,
            "max_budget_usd": self._max_budget_usd,
            "can_use_tool": can_use_tool,
            "env": env_overrides,
            # Explicit SDK isolation. None (the SDK default, verified against
            # claude-agent-sdk 0.2.120) means the CLI loads ALL filesystem
            # settings — ~/.claude/settings.json, .claude/settings.json,
            # .claude/settings.local.json — so ambient operator/project
            # settings could re-permission tools or install hooks UNDERNEATH
            # the permission_mode/can_use_tool posture configured above. The
            # empty list is the SDK's documented isolation mode: settings
            # come from Hermes config only. Accepted side effect: "project"
            # is also what loads CLAUDE.md, which this runtime doesn't want —
            # it composes its own system-prompt append. Operators whose
            # deployment stores tool grants in ~/.claude/settings.json
            # (unattended cron turns with nobody to answer a prompt) opt
            # back in via agent.claude_agent_sdk.setting_sources — see
            # _configured_setting_sources.
            "setting_sources": _configured_setting_sources(),
            # Explicit, because the SDK's 1 MiB default is below what this
            # lane's own tool results routinely produce and overflowing it
            # kills the turn outright — see _configured_max_buffer_size.
            "max_buffer_size": _configured_max_buffer_size(),
            # AskUserQuestion has no native answer bridge. Native Read stays
            # behind the protected-path-aware, bounded Hermes MCP surface in
            # every supported SDK permission mode.
            "disallowed_tools": ["AskUserQuestion", "Read"],
        }
        # The CLI owns compaction on this lane, so its PreCompact hook is the
        # only honest signal that a turn stalled to compact. Registered only
        # when a consumer asked for it, so the default option set is unchanged.
        _compaction_hooks = self._build_compaction_hooks()
        if _compaction_hooks is not None:
            fields["hooks"] = _compaction_hooks
        if self._resume_session_id:
            fields["resume"] = self._resume_session_id
        # Default OFF (upstream-conservative): partial messages only when the
        # operator opts in via agent.claude_agent_sdk.streaming in config.yaml.
        # Reads the __init__ snapshot so option and quiet-watchdog semantics
        # can never diverge across a mid-session config edit.
        if self._streaming:
            fields["include_partial_messages"] = True
        return fields

    def _build_client(self) -> Any:
        fields = self.build_option_fields()
        if self._client_factory is not None:
            return self._client_factory(options=fields)
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

        return ClaudeSDKClient(options=ClaudeAgentOptions(**fields))
