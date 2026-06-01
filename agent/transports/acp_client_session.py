"""Session adapter for ACP client runtime.

Owns one ACP session per Hermes session. Drives ``session/new`` + ``session/prompt``,
consumes streaming ``session/update`` notifications (AgentMessageChunk), handles
server-initiated requests (fs/*, terminal/*, permission — allowed-once by default),
and returns a TurnResult that acp_runtime.run_acp_client_turn() can splice into
the ``messages`` list.

Lifecycle:
    session = ACPClientSession(command="claude-agent-acp", model="claude-haiku-4-5")
    session.ensure_started(cwd="/home/x/proj")      # spawns + initialize + session/new
    result = session.run_turn("hello")               # blocks until session/prompt returns
    # result.final_text          → assistant text returned to caller
    # result.projected_messages  → list of {role, content} for messages list
    # result.tool_iterations     → count of tool-shaped update events (skill nudge)
    # result.should_retire       → True if session wedged (timeout, crash)
    session.close()                                  # session/close + subprocess teardown

Threading model: single-threaded from the caller's perspective.
The underlying ACPClient owns its own reader threads but exposes
blocking-with-timeout queues that this adapter polls in a loop.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agent.transports.acp_client import ACPClient, ACPClientError

logger = logging.getLogger(__name__)

# ACP wire method names (from acp.meta)
_METHOD_INITIALIZE = "initialize"
_METHOD_SESSION_NEW = "session/new"
_METHOD_SESSION_PROMPT = "session/prompt"
_METHOD_SESSION_CLOSE = "session/close"
_METHOD_SESSION_CANCEL = "session/cancel"
_METHOD_SESSION_UPDATE = "session/update"
_METHOD_SESSION_SET_CONFIG = "session/set_config_option"

# Server-initiated (client-side) methods we receive
_METHOD_FS_READ = "fs/read_text_file"
_METHOD_FS_WRITE = "fs/write_text_file"
_METHOD_PERMISSION = "session/request_permission"
_METHOD_TERMINAL_CREATE = "terminal/create"
_METHOD_TERMINAL_OUTPUT = "terminal/output"
_METHOD_TERMINAL_RELEASE = "terminal/release"
_METHOD_TERMINAL_WAIT = "terminal/wait_for_exit"
_METHOD_TERMINAL_KILL = "terminal/kill"

# ACP session/update discriminator values for streaming chunks.
# Only agent_message_chunk carries user-facing text — agent_thought_chunk is
# the model's internal reasoning and MUST NOT be merged into the reply (Fix 2).
_UPDATE_AGENT_MESSAGE = "agent_message_chunk"
_UPDATE_AGENT_THOUGHT = "agent_thought_chunk"
_UPDATE_TOOL_CALL = "tool_call_update"
_UPDATE_TOOL_CALL_START = "tool_call"

# How many trailing stderr lines to show in error messages
_STDERR_TAIL_LINES = 12


@dataclass
class TurnResult:
    """Result of one user->assistant turn through an ACP-compliant agent."""

    final_text: str = ""
    projected_messages: list[dict] = field(default_factory=list)
    tool_iterations: int = 0
    interrupted: bool = False
    error: Optional[str] = None
    # True when the session is wedged (timeout, crash, bad response).
    # The caller should retire and re-create the session on the next turn.
    should_retire: bool = False


def _extract_text_from_update(params: dict) -> str:
    """Extract plain text from an ACP session/update notification params.

    ``session/update`` params carry:
      { "sessionId": "...", "update": { "sessionUpdate": "agent_message_chunk",
                                        "content": { "type": "text", "text": "..." } } }

    Fix 2 -- ONLY extract text from ``agent_message_chunk`` updates. The server
    also emits ``agent_thought_chunk`` for the model's internal reasoning; those
    must NOT be included in the user-facing reply. Keying on the discriminator
    instead of content.type avoids silently leaking future reasoning variants.
    """
    update = params.get("update") or {}
    # Support both camelCase (sessionUpdate) and snake_case (session_update) keys
    # to match whatever the server emits -- mirrors _is_tool_iteration's approach.
    kind = update.get("sessionUpdate") or update.get("session_update") or ""
    if kind != _UPDATE_AGENT_MESSAGE:
        # Intentionally skip agent_thought_chunk and anything else that is not
        # a confirmed user-facing text chunk (YAGNI -- do not whitelist speculatively).
        return ""
    content = update.get("content") or {}
    if isinstance(content, dict) and content.get("type") == "text":
        return content.get("text") or ""
    return ""


def _is_tool_iteration(params: dict) -> bool:
    """Return True if the update represents a tool call completion."""
    update = params.get("update") or {}
    kind = update.get("sessionUpdate") or update.get("session_update") or ""
    return kind in {_UPDATE_TOOL_CALL, _UPDATE_TOOL_CALL_START}


class ACPClientSession:
    """One ACP session per Hermes session, lifetime owned by AIAgent.

    Not thread-safe -- one caller drives it at a time, matching how AIAgent's
    run_conversation() loop is structured today.
    """

    def __init__(
        self,
        *,
        command: str,
        args: Optional[list[str]] = None,
        env: Optional[dict[str, str]] = None,
        model: Optional[str] = None,
        mcp_servers: Optional[list[dict]] = None,
        setting_sources: Optional[tuple] = ("project", "local"),
        on_delta: Optional[Callable[[str], None]] = None,
        client_factory: Optional[Callable[..., ACPClient]] = None,
    ) -> None:
        """
        Args:
            command: ACP agent binary to spawn (e.g. "claude-agent-acp").
            args: Additional arguments to pass to the command.
            env: Extra environment variables for the subprocess.
            model: Model identifier to pin on the ACP session after session/new
                (Fix 1). Sent via ``session/set_config_option`` so the ACP server
                does not default to its own heavy model (e.g. Opus). Only sent
                when set; servers that do not support set_config_option are
                tolerated -- the call is wrapped in a try/except and a warning
                is logged rather than hard-failing the session.
            mcp_servers: Pre-translated ACP McpServer dicts to forward in
                session/new.  Build with ``_translate_mcp_servers()`` from
                Hermes' mcp_servers config.  Hermes does NOT open these
                connections in-process -- the external ACP agent owns them.
                None or [] → send an empty list (current default behaviour).
            setting_sources: Claude Code setting scopes forwarded to the native
                ACP server via _meta.claudeCode.options.settingSources in
                session/new.  The native server (claude-agent-acp) hardcodes
                settingSources:["user","project","local"] (dist/acp-agent.js:1522)
                and then spreads userProvidedOptions on top (line 1524), so this
                override takes effect.  Default ("project","local") mirrors
                --setting-sources project,local used by the claude_daemon transport
                and excludes the "user" scope, preventing ~/.claude plugins and
                MCP servers (playwright, context7, slack-channels, etc.) from
                leaking into the delegated agent.  Non-Claude ACP servers ignore
                unknown _meta keys, so it is safe to always send.
                Pass None or [] to omit _meta entirely (raw server defaults).
            on_delta: Optional callback invoked with each text delta during streaming.
                      Bridges to Hermes' ``_fire_stream_delta`` for live output.
            client_factory: Inject a custom ACPClient constructor for testing.
        """
        self._command = command
        self._args = list(args or [])
        self._env = env
        self._model = model
        self._mcp_servers: list[dict] = list(mcp_servers or [])
        self._setting_sources: Optional[list[str]] = (
            list(setting_sources) if setting_sources else None
        )
        self._on_delta = on_delta
        self._client_factory = client_factory or ACPClient

        self._client: Optional[ACPClient] = None
        self._session_id: Optional[str] = None
        self._closed = False

    # ---------- lifecycle ----------

    def ensure_started(self, cwd: Optional[str] = None) -> str:
        """Spawn the subprocess, do the initialize handshake, and start a
        session. Returns the ACP session_id. Idempotent -- repeated calls
        return the same session_id."""
        if self._session_id is not None:
            return self._session_id
        if self._client is None:
            self._client = self._client_factory(
                command=self._command,
                args=self._args,
                env=self._env,
            )
        self._client.initialize(
            client_name="hermes",
            client_version=_get_hermes_version(),
        )
        # Build session/new params.  _meta.claudeCode.options is a documented
        # passthrough in the native ACP server (acp-agent.js:1500) spread AFTER
        # its hardcoded settingSources (line 1524), so our value wins.
        # Sending settingSources:["project","local"] (the default) prevents
        # ~/.claude user-scope plugins and MCP servers (playwright, context7,
        # slack-channels, etc.) from leaking into the delegated agent -- the
        # same isolation the claude_daemon transport enforces via
        # --setting-sources project,local.  Generic ACP servers ignore unknown
        # _meta keys, so it is safe to always include when configured.
        session_new_params: dict = {
            "cwd": cwd or os.getcwd(),
            "mcpServers": self._mcp_servers,
        }
        if self._setting_sources:
            session_new_params["_meta"] = {
                "claudeCode": {
                    "options": {
                        "settingSources": self._setting_sources,
                    }
                }
            }
        result = self._client.request(
            _METHOD_SESSION_NEW,
            session_new_params,
            timeout=15,
        )
        session_id = result.get("sessionId") or result.get("session_id") or ""
        if not session_id:
            raise ACPClientError(
                code=-32603,
                message=(
                    "ACP session/new returned no sessionId "
                    f"(payload keys: {sorted(result.keys())})"
                ),
            )
        self._session_id = session_id
        logger.info(
            "ACP client session started: id=%s command=%r cwd=%s",
            self._session_id[:8],
            self._command,
            cwd or os.getcwd(),
        )

        # Fix 1 -- Model pin: send session/set_config_option to override the
        # ACP server's default model (native claude-agent-acp defaults to Opus,
        # which silently drains quota on every turn). Only sent when a model is
        # explicitly configured; servers that don't support set_config_option are
        # tolerated; servers that support it but reject the value raise loud.
        if self._model:
            try:
                self._send_model_config(self._session_id, self._model)
            except ACPClientError:
                # Mismatch detected: clear session so ensure_started does not
                # short-circuit on the next call (idempotency guard at top of
                # method checks self._session_id is not None).  Re-raise so
                # run_turn can surface the config error without retiring.
                self._session_id = None
                raise

        return self._session_id

    def _send_model_config(self, session_id: str, model: str) -> None:
        """Send session/set_config_option to pin the model on the ACP session, then
        verify the server honoured the value.

        The native claude-agent-acp server (v0.39) accepts only its own short aliases:
          "haiku"   → Haiku 4.5   (cheapest)
          "sonnet"  → Sonnet 4.6
          "default" → Opus 4.8    (most expensive — the server default)
        Full API IDs like "claude-haiku-4-5-20251001" are normalised to the alias by
        the server; "claude-sonnet-4-5" is rejected with -32603.  Configure the model
        as the alias ("haiku", "sonnet") in Hermes' acp_model config key.

        Verification is REQUIRED because a wrong value silently falls back to the
        server default (Opus 4.8), burning quota on every turn without any warning.

        Two-layer exception strategy:
          • transport/protocol failure (request() raises) → TOLERATE: server may
            not implement set_config_option at all; warn and continue.
          • server supported but value rejected or silently ignored (request() succeeds
            but currentValue != requested) → FAIL LOUD: raise ACPClientError so the
            caller knows the pin didn't take.  Do NOT swallow this — billing impact.
        """
        assert self._client is not None

        # Tolerance layer: wraps only the wire call.  Servers that do not implement
        # set_config_option return a -32601 method-not-found error; we log and return.
        try:
            result = self._client.request(
                _METHOD_SESSION_SET_CONFIG,
                {
                    "sessionId": session_id,
                    "configId": "model",
                    "value": model,
                },
                timeout=5,
            )
        except (ACPClientError, TimeoutError, RuntimeError) as exc:
            # Server does not support set_config_option (e.g. -32601 method-not-found)
            # or timed out.  Tolerate: not all ACP servers expose config options.
            logger.warning(
                "ACP client: session/set_config_option not supported or timed out "
                "(model=%r, session=%s): %s -- session continues with server default",
                model,
                session_id[:8],
                exc,
            )
            return

        # Verification layer: the server responded successfully.  Extract the
        # "model" configOption's currentValue from the response.  The response
        # shape is: {"configOptions": [{"id": "model", "currentValue": "haiku", ...}]}
        config_opts = result.get("configOptions") or []
        model_opt = next((o for o in config_opts if o.get("id") == "model"), None)

        if model_opt is None:
            # Server returned a successful response but no "model" configOption.
            # Generic ACP server -- cannot verify; proceed without confirmation.
            logger.warning(
                "ACP client: set_config_option succeeded but response carried no "
                "'model' configOption -- cannot verify pin (session=%s)",
                session_id[:8],
            )
            return

        current_value = model_opt.get("currentValue")
        if current_value == model:
            # Pin confirmed.
            logger.info(
                "ACP client: model pinned and verified: %r on session %s",
                model,
                session_id[:8],
            )
            return

        # BILLING-CRITICAL: server accepted the call but currentValue does not match.
        # Continuing would silently run every turn on the server's default model
        # (Opus 4.8 as of claude-agent-acp v0.39), burning expensive quota.
        # Accepted model aliases for the native server: "haiku", "sonnet", "default".
        accepted = [o.get("value") for o in model_opt.get("options", [])]
        raise ACPClientError(
            code=1,  # positive = config rejection, not a transport crash (see run_turn)
            message=(
                f"ACP model pin rejected: requested {model!r} but server "
                f"currentValue={current_value!r}. "
                f"Continuing would silently run on {current_value!r} (server default), "
                f"burning expensive model quota. "
                f"Set acp_model to one of the accepted aliases: {accepted}."
            ),
        )

    def close(self) -> None:
        """Send session/close and tear down the subprocess."""
        if self._closed:
            return
        self._closed = True
        if self._client is not None and self._session_id is not None:
            try:
                self._client.request(
                    _METHOD_SESSION_CLOSE,
                    {"sessionId": self._session_id},
                    timeout=5,
                )
            except Exception:
                pass  # best-effort
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        self._session_id = None

    def __enter__(self) -> "ACPClientSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------- turn ----------

    def run_turn(
        self,
        user_input: Any,
        *,
        cwd: Optional[str] = None,
        turn_timeout: float = 600.0,
        notification_poll_timeout: float = 0.25,
    ) -> TurnResult:
        """Send a user message and block until session/prompt returns.

        Streams session/update notifications to on_delta as they arrive.
        Projects streamed content into projected_messages so memory/skill
        review keep working.

        Returns a TurnResult. Sets should_retire=True on crash/timeout.
        """
        result = TurnResult()
        # Ensure session is open (lazy start on first turn)
        try:
            self.ensure_started(cwd=cwd)
        except ACPClientError as exc:
            result.error = f"ACP client session startup failed: {exc}"
            # Positive error code = config rejection (model pin mismatch, not a
            # session crash).  Do NOT retire: retiring would respawn the session and
            # hit the same mismatch every turn, creating an infinite retry loop.
            # The caller must fix the configured model alias and redeploy.
            result.should_retire = exc.code <= 0
            return result
        except (TimeoutError, RuntimeError) as exc:
            result.error = f"ACP client session startup failed: {exc}"
            result.should_retire = True
            return result

        assert self._client is not None and self._session_id is not None

        user_text = _coerce_user_input(user_input)

        # Build ACP prompt request
        prompt_params = {
            "sessionId": self._session_id,
            "prompt": [{"type": "text", "text": user_text}],
        }

        # session/prompt is a request that blocks until the agent returns
        # PromptResponse. While waiting, the agent sends session/update
        # notifications which arrive in the _notifications queue.
        # We poll both in a deadline loop.
        deadline = time.monotonic() + turn_timeout
        text_chunks: list[str] = []

        # Send session/prompt in a background thread so we can drain
        # notifications concurrently. The result arrives via a shared dict.
        _response: dict = {}
        _error: list = []  # [exc] if the request raised

        def _do_request() -> None:
            try:
                r = self._client.request(
                    _METHOD_SESSION_PROMPT,
                    prompt_params,
                    timeout=turn_timeout,
                )
                _response["result"] = r
            except (ACPClientError, TimeoutError, RuntimeError) as exc:
                _error.append(exc)

        req_thread = threading.Thread(target=_do_request, daemon=True)
        req_thread.start()

        def _process_notification(note: dict) -> None:
            """Apply a single session/update notification to result + text_chunks.

            Factored out so the same logic runs during the live drain loop and
            the post-join tail-drain (notifications that arrived between the
            last loop poll and req_thread completion would otherwise be lost).
            """
            if note.get("method") != _METHOD_SESSION_UPDATE:
                return
            params = note.get("params") or {}
            delta = _extract_text_from_update(params)
            if delta:
                text_chunks.append(delta)
                if self._on_delta is not None:
                    try:
                        self._on_delta(delta)
                    except Exception:
                        logger.debug("on_delta callback raised", exc_info=True)
            if _is_tool_iteration(params):
                result.tool_iterations += 1

        # Drain notifications while waiting for the prompt response.
        # session/prompt blocks for the entire turn; req_thread sends it while
        # this loop concurrently drains session/update chunks.
        # _send_lock on ACPClient ensures the two threads don't interleave
        # writes to the same BufferedWriter (see ACPClient._send).
        while req_thread.is_alive() and time.monotonic() < deadline:
            if not (self._client and self._client.is_alive()):
                result.error = self._format_error("ACP agent subprocess exited unexpectedly")
                result.should_retire = True
                break

            # Handle server-initiated requests (fs/*, permission, terminal/*)
            sreq = self._client.take_server_request(timeout=0)
            if sreq is not None:
                self._handle_server_request(sreq)
                continue

            # Drain streaming notifications (session/update)
            note = self._client.take_notification(timeout=notification_poll_timeout)
            if note is None:
                continue
            _process_notification(note)

        req_thread.join(timeout=2.0)

        # Tail-drain: consume notifications that were parsed by the reader
        # thread between the last loop poll and req_thread completing. These
        # would be silently dropped without this drain -- short responses that
        # fit in the first chunks are the most likely to be affected.
        if self._client is not None:
            while True:
                note = self._client.take_notification(timeout=0)
                if note is None:
                    break
                _process_notification(note)

        if _error:
            exc = _error[0]
            result.error = f"ACP session/prompt failed: {exc}"
            if isinstance(exc, TimeoutError) or (
                isinstance(exc, ACPClientError) and exc.code < 0
            ):
                result.should_retire = True
            return result

        if "result" not in _response and not result.should_retire:
            # Deadline hit without response
            result.error = f"ACP turn timed out after {turn_timeout}s"
            result.should_retire = True
            result.interrupted = True
            return result

        if result.should_retire:
            return result

        # Assemble final text from streamed chunks. If chunks are empty,
        # look for text in the PromptResponse itself (some implementations
        # may put content there instead of streaming).
        prompt_result = _response.get("result") or {}
        assembled = "".join(text_chunks)
        if not assembled:
            # Fallback: look for content in the PromptResponse
            for block in (prompt_result.get("content") or []):
                if isinstance(block, dict) and block.get("type") == "text":
                    assembled += block.get("text") or ""

        result.final_text = assembled

        # Project into messages so curator/memory/skill review can see the turn.
        if assembled:
            result.projected_messages.append(
                {"role": "assistant", "content": assembled}
            )

        return result

    # ---------- internals ----------

    def _handle_server_request(self, req: dict) -> None:
        """Handle server-initiated requests from the ACP agent.

        Permission requests are granted (allow_once) because this transport
        talks to a trusted local ACP agent process that Hermes itself spawned
        -- the user already consented to the agent's capabilities by configuring
        it.  A future policy knob could toggle this per-agent.

        fs/* and terminal/* are declined: Hermes controls those surfaces
        through its own tool executor.
        """
        if self._client is None:
            return
        method = req.get("method", "")
        rid = req.get("id")

        if method == _METHOD_PERMISSION:
            # Fix 3 -- ACP-correct permission response shape.
            #
            # The ACP spec (RequestPermissionResponse) requires:
            #   { "outcome": <RequestPermissionOutcome> }
            # where RequestPermissionOutcome is one of:
            #   { "outcome": "cancelled" }
            #   { "outcome": "selected", "optionId": "<id>" }
            #
            # "allow_once" / "reject_once" are PermissionOptionKind hints in the
            # INCOMING request's options[] array -- NOT valid outcome.outcome values.
            # The previously sent { "granted": false } is not a valid ACP response
            # and would wedge any turn that receives a permission request.
            #
            # Strategy: this transport talks to a trusted, locally-spawned agent
            # process; grant by echoing the first allow_once-kinded option.
            # Fall back to the first available option, then to cancelled if there
            # are no options (malformed request).
            params = req.get("params") or {}
            options = params.get("options") or []
            chosen_option_id = _pick_allow_option(options)
            if chosen_option_id is not None:
                outcome: dict = {"outcome": "selected", "optionId": chosen_option_id}
            else:
                # No options in the request (malformed) -- cancel instead of wedging.
                outcome = {"outcome": "cancelled"}
            self._client.respond(rid, {"outcome": outcome})
            logger.debug(
                "ACP client: permission request -> outcome=%r optionId=%r",
                outcome["outcome"],
                outcome.get("optionId"),
            )
        elif method in {
            _METHOD_FS_READ, _METHOD_FS_WRITE,
            _METHOD_TERMINAL_CREATE, _METHOD_TERMINAL_OUTPUT,
            _METHOD_TERMINAL_RELEASE, _METHOD_TERMINAL_WAIT,
            _METHOD_TERMINAL_KILL,
        }:
            # Decline fs/terminal proxying -- Hermes drives its own tool
            # executor. ACP agents that need fs/terminal ops should spawn
            # their own processes.
            logger.debug("ACP client: declining server request %r (not proxied in v1)", method)
            self._client.respond_error(
                rid,
                code=-32601,
                message=f"Method not supported by Hermes ACP client v1: {method}",
            )
        else:
            logger.warning("ACP client: unknown server request %r", method)
            self._client.respond_error(
                rid,
                code=-32601,
                message=f"Unknown method: {method}",
            )

    def _format_error(self, prefix: str) -> str:
        """Build a user-facing error string, appending stderr tail when available."""
        if self._client is None:
            return prefix
        try:
            tail = self._client.stderr_tail(_STDERR_TAIL_LINES)
        except Exception:
            return prefix
        if not tail:
            return prefix
        joined = "\n".join(line.rstrip() for line in tail if line)
        if not joined.strip():
            return prefix
        return f"{prefix}\nACP agent stderr (last {len(tail)} lines):\n{joined}"


def _translate_mcp_servers(servers: dict) -> list[dict]:
    """Translate Hermes mcp_servers config dict to ACP McpServer wire shapes.

    Hermes config (from config.yaml ``mcp_servers:`` key) is a dict of
    ``{name: server_cfg}`` where server_cfg contains either stdio or HTTP/SSE
    transport fields.  The ACP server expects a typed list with exact shapes
    (probed empirically against claude-agent-acp v0.39).

    Accepted ACP shapes:
      stdio (NO type field):
        {"name": str, "command": str, "args": [str], "env": [{"name": str, "value": str}]}
        env is REQUIRED and must be an array -- use [] when the config has none.
      http:
        {"type": "http", "name": str, "url": str, "headers": [{"name": str, "value": str}]}
        headers is REQUIRED array.
      sse:
        {"type": "sse", "name": str, "url": str, "headers": [{"name": str, "value": str}]}
        headers is REQUIRED array.

    Both env/headers must be [{name, value}] arrays -- dict/object shapes are
    rejected (-32602) by the native server.  They are always emitted ([] when
    empty) so the server never sees a missing required field.

    Hermes-only keys (timeout, connect_timeout, auth, sampling) are dropped;
    ACP does not accept unknown fields.  Values are already ${VAR}-interpolated
    by _load_mcp_config() -- no re-interpolation here.

    Entries with neither ``command`` nor ``url`` are skipped with a warning
    rather than crashing the session setup.
    """
    out = []
    for name, cfg in (servers or {}).items():
        if not isinstance(cfg, dict):
            logger.warning(
                "ACP MCP translate: skipping %r -- config is not a dict (%r)",
                name, type(cfg).__name__,
            )
            continue

        has_command = bool(cfg.get("command"))
        has_url = bool(cfg.get("url"))

        if has_command and has_url:
            # Prefer stdio when both are set, matching the codex translator.
            logger.debug(
                "ACP MCP translate: %r has both command and url -- using stdio", name
            )
            has_url = False

        if has_command:
            # Stdio transport.  env dict -> [{name, value}] array (always present).
            raw_env = cfg.get("env") or {}
            env_array = [{"name": str(k), "value": str(v)} for k, v in raw_env.items()]
            args = [str(a) for a in (cfg.get("args") or [])]
            out.append({
                "name": name,
                "command": str(cfg["command"]),
                "args": args,
                "env": env_array,
                # No "type" field for stdio -- ACP spec requires its absence.
            })
        elif has_url:
            # HTTP or SSE transport.  Hermes uses "transport" key ("sse" hint).
            # headers dict -> [{name, value}] array (always present).
            raw_headers = cfg.get("headers") or {}
            headers_array = [{"name": str(k), "value": str(v)} for k, v in raw_headers.items()]
            # Hermes writes "transport" (not "type") for the sse hint.
            # Honor explicit "type" too for forward-compat.
            transport_hint = cfg.get("transport") or cfg.get("type") or ""
            acp_type = "sse" if transport_hint.lower() == "sse" else "http"
            out.append({
                "type": acp_type,
                "name": name,
                "url": str(cfg["url"]),
                "headers": headers_array,
            })
        else:
            logger.warning(
                "ACP MCP translate: skipping %r -- no 'command' or 'url' field", name
            )
            continue

    return out


def _pick_allow_option(options: list) -> Optional[str]:
    """Return the optionId to grant from a permission request's options list.

    Prefers the first option whose kind is ``allow_once``; falls back to the
    first option of any kind. Returns None when the list is empty.
    """
    first_any: Optional[str] = None
    for opt in options:
        if not isinstance(opt, dict):
            continue
        option_id = opt.get("optionId")
        if option_id is None:
            continue
        if first_any is None:
            first_any = option_id
        if opt.get("kind") == "allow_once":
            return option_id
    return first_any


def _coerce_user_input(user_input: Any) -> str:
    """Collapse Hermes/OpenAI rich content into plain text for ACP session/prompt."""
    if isinstance(user_input, str):
        return user_input
    if isinstance(user_input, list):
        parts: list[str] = []
        for item in user_input:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item)
                continue
            if not isinstance(item, dict):
                if item is not None:
                    parts.append(str(item))
                continue
            item_type = item.get("type")
            if item_type in {"text", "input_text"}:
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            elif item_type in {"image", "image_url", "input_image"}:
                parts.append("[image attached]")
        text = "\n\n".join(p for p in parts if p).strip()
        return text or "What do you see in this image?"
    return "" if user_input is None else str(user_input)


def _get_hermes_version() -> str:
    """Best-effort Hermes version string for ACP initialize."""
    try:
        from importlib.metadata import version
        return version("hermes-agent")
    except Exception:  # pragma: no cover
        return "0.0.0"
