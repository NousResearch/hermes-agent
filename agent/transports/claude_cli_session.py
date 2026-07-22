"""Session adapter for the claude_cli runtime (Phase 2b multi-turn).

Owns the Hermes conversation → Claude session_id mapping and drives one
``claude -p`` subprocess **per Hermes turn** (Claude is one-shot per spawn;
multi-turn context lives in Claude's on-disk session, not a long-lived
process). Mirrors ``CodexAppServerSession``'s "one session per AIAgent /
Hermes conversation" lifetime:

    session = ClaudeCliSession(oauth_token=..., model=..., cwd=...)
    t1 = session.run_turn("Remember X")   # --session-id <uuid>
    t2 = session.run_turn("What is X?")    # --resume <same uuid>
    session.close()

Phase 2a (still on every spawn): Hermes MCP tools via ``--mcp-config`` +
``hermes_tools_mcp_server``. Tool round-trip is internal to ``claude -p``.

Phase 2b (this module):
  * First turn: generate a UUID, pass ``--session-id``, record mapping.
  * Subsequent turns in the same Hermes conversation: ``--resume`` + only
    the new user message (Claude already has history + native compaction).
  * Stable ``cwd`` so Claude finds session files under
    ``~/.claude/projects/<cwd-encoded>/<uuid>.jsonl``.
  * Missing/expired resume → fall back to a fresh ``--session-id`` once.
  * Pre-existing Hermes history on first turn: send latest user message
    only (full transcript pre-seed is a follow-up).

OpenClaw parallel: sessionMode always, ``--session-id`` / ``--resume``.
Codex parallel: ``agent._codex_session`` reuse across turns on one AIAgent.
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

from agent.redact import redact_sensitive_text
from agent.transports.claude_cli import (
    ClaudeCliClient,
    ClaudeCliConcurrencyError,
    ClaudeCliError,
    ClaudeCliSpawnConfig,
    build_claude_cli_clean_env,
    resolve_claude_bin,
)
from agent.transports.claude_cli_concurrency import (
    claude_cli_slot,
    is_force_unbounded,
)
from agent.transports.claude_event_projector import ClaudeEventProjector
from agent.transports.types import NormalizedResponse, Usage

logger = logging.getLogger(__name__)

_STDERR_TAIL_LINES = 20

# Error substrings that mean the Claude session file is missing / unusable.
# When resume hits one of these, we create a fresh session once.
_RESUME_MISSING_HINTS: tuple[str, ...] = (
    "no conversation found",
    "session not found",
    "could not find session",
    "unable to resume",
    "failed to resume",
    "unknown session",
    "invalid session",
    "does not exist",
    "no such session",
    "session file",
    "cannot resume",
)


@dataclass
class ClaudeCliTurnResult:
    """Result of one user→assistant turn through ``claude -p``."""

    final_text: str = ""
    projected_messages: list[dict] = field(default_factory=list)
    tool_iterations: int = 0  # completed Hermes MCP tool calls this turn
    interrupted: bool = False
    error: Optional[str] = None
    is_error: bool = False
    session_id: Optional[str] = None  # Claude session UUID
    usage: Optional[dict[str, Any]] = None
    total_cost_usd: Optional[float] = None
    # Mirror codex TurnResult field names used by the agent-loop recorder.
    token_usage_last: Optional[dict[str, Any]] = None
    # False while the mapping is healthy (reuse next turn). True only when
    # the runtime should drop agent._claude_cli_session (hard failure).
    should_retire: bool = False
    result_event: Optional[dict[str, Any]] = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # Phase 2b diagnostics
    resumed: bool = False
    created_session: bool = False
    resume_fallback: bool = False
    history_seed_note: Optional[str] = None

    def to_normalized_response(self) -> NormalizedResponse:
        """Map to the shared NormalizedResponse so the loop treats this as a normal turn."""
        usage_obj = None
        if isinstance(self.usage, dict) and self.usage:
            prompt = int(
                self.usage.get("input_tokens")
                or self.usage.get("prompt_tokens")
                or 0
            )
            completion = int(
                self.usage.get("output_tokens")
                or self.usage.get("completion_tokens")
                or 0
            )
            cache_read = int(
                self.usage.get("cache_read_input_tokens")
                or self.usage.get("cache_read_tokens")
                or 0
            )
            usage_obj = Usage(
                prompt_tokens=prompt + cache_read,
                completion_tokens=completion,
                total_tokens=prompt + cache_read + completion,
                cached_tokens=cache_read,
            )
        finish = "stop" if not self.is_error and not self.error else "error"
        return NormalizedResponse(
            content=self.final_text or None,
            tool_calls=None,
            finish_reason=finish if finish != "error" else "stop",
            usage=usage_obj,
            provider_data={
                "claude_cli_session_id": self.session_id,
                "claude_cli_total_cost_usd": self.total_cost_usd,
                "claude_cli_is_error": self.is_error,
                "claude_cli_usage": self.usage,
                "claude_cli_resumed": self.resumed,
                "claude_cli_created_session": self.created_session,
                "claude_cli_resume_fallback": self.resume_fallback,
            },
        )


def _looks_like_claude_setup_or_oauth_token(token: str) -> bool:
    """True for setup tokens (sk-ant-oat…) or Claude Code OAuth tokens (cc-…)."""
    stripped = (token or "").strip()
    return stripped.startswith("sk-ant-oat") or stripped.startswith("cc-")


def _read_setup_token_keys_from_env_file(path) -> Optional[str]:
    """Read CLAUDE_CODE_OAUTH_TOKEN (or legacy ANTHROPIC_TOKEN) from a .env file.

    Read-only and crash-safe: missing/unreadable files return None.
    Does **not** load values into ``os.environ``.
    """
    try:
        from pathlib import Path

        env_path = Path(path)
        if not env_path.is_file():
            return None
    except OSError:
        return None

    values: dict = {}
    try:
        from dotenv import dotenv_values

        parsed = dotenv_values(env_path) or {}
        values = {str(k): (v if v is not None else "") for k, v in parsed.items()}
    except Exception:
        try:
            text = env_path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            return None
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            values[key] = value

    for key in ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_TOKEN"):
        val = str(values.get(key) or "").strip()
        if val:
            return val
    return None


def _read_canonical_root_setup_token() -> Optional[str]:
    """Fleet fallback: setup token from the platform Hermes root ``.env``.

    Uses :func:`hermes_constants.get_default_hermes_root` so a profile with its
    own ``HERMES_HOME`` still resolves the **one** shared non-rotating setup
    token from ``~/.hermes/.env`` (platform-native root), not the profile dir.

    Why root ``.env`` (not a new shared-auth file): it is already Hermes'
    canonical secret store, needs no new machinery, and the setup token is a
    secret (never ``config.yaml``). Fork-safe / non-rotating → no lock needed.
    """
    try:
        from hermes_constants import get_default_hermes_root

        root_env = get_default_hermes_root() / ".env"
    except Exception as exc:
        logger.debug("claude_cli: canonical root path failed: %s", exc)
        return None
    try:
        return _read_setup_token_keys_from_env_file(root_env)
    except Exception as exc:
        logger.debug("claude_cli: canonical root .env read failed: %s", exc)
        return None


def resolve_claude_cli_oauth_token(
    *,
    explicit: Optional[str] = None,
    agent: Any = None,
) -> str:
    """Resolve the non-rotating setup token for CLAUDE_CODE_OAUTH_TOKEN injection.

    Final resolution order (first hit wins):

      1. Explicit / passed token (``explicit=`` argument, or agent-held
         setup/OAuth-shaped key ``sk-ant-oat`` / ``cc-``)
      2. Profile / process env ``CLAUDE_CODE_OAUTH_TOKEN`` (then legacy
         ``ANTHROPIC_TOKEN``). When a profile's ``.env`` was loaded into the
         process env at startup, that is this step.
      3. Profile credential_pool OAuth entries (``claude_code`` /
         ``env:CLAUDE_CODE_OAUTH_TOKEN`` and other anthropic OAuth pool rows)
      4. **Canonical Hermes root** ``~/.hermes/.env``
         (``get_default_hermes_root() / ".env"``) — fleet fallback so any
         profile can run Claude on demand without copying the token into its
         own ``.env``
      5. **Never** the rotating ``~/.claude`` login credentials (keychain /
         ``~/.claude/.credentials.json``)

    Those login credentials are not fork-safe for a clean ``claude -p``
    subprocess. The setup token (``sk-ant-oat…``, ~1yr) is fork-safe like an
    API key — no shared store or lock.

    Only raises when no source yields a token.
    """
    if explicit and str(explicit).strip():
        return str(explicit).strip()

    # (1b) Passed via agent constructor when already setup/OAuth-shaped.
    if agent is not None:
        key = getattr(agent, "api_key", None) or getattr(agent, "_anthropic_api_key", None)
        if isinstance(key, str) and key.strip():
            stripped = key.strip()
            if _looks_like_claude_setup_or_oauth_token(stripped):
                return stripped

    # (2) Profile / process env — first hit of the two keys.
    for env_var in ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_TOKEN"):
        val = (os.environ.get(env_var) or "").strip()
        if val:
            return val

    # (3) Profile-scoped credential_pool (HERMES_HOME auth.json).
    # Read-only enumerate — never refresh/network. Intentionally does NOT
    # call resolve_anthropic_token() (that path also reads ~/.claude login).
    try:
        from agent.anthropic_adapter import _resolve_anthropic_pool_token

        pool_token = _resolve_anthropic_pool_token()
        if pool_token and str(pool_token).strip():
            return str(pool_token).strip()
    except Exception as exc:
        logger.debug("claude_cli: credential_pool resolve failed: %s", exc)

    # (4) Canonical platform-root ~/.hermes/.env (fleet shared setup token).
    root_token = _read_canonical_root_setup_token()
    if root_token:
        return root_token

    raise ClaudeCliError(
        message=(
            "claude_cli: no Anthropic setup token found. Put the non-rotating "
            "setup token once in the canonical Hermes root ~/.hermes/.env as "
            "CLAUDE_CODE_OAUTH_TOKEN=... (any profile resolves it), or set it in "
            "this profile's env / anthropic credential_pool "
            "(env:CLAUDE_CODE_OAUTH_TOKEN or claude_code). Generate with "
            "`claude setup-token`. Do NOT rely on the rotating `claude /login` "
            "session for the subprocess."
        )
    )


def new_claude_session_id() -> str:
    """Generate a Claude-compatible session UUID (required by ``--session-id``)."""
    return str(uuid.uuid4())


def is_resume_missing_error(message: str) -> bool:
    """True when a failed resume looks like a missing/expired Claude session."""
    text = (message or "").lower()
    if not text:
        return False
    return any(hint in text for hint in _RESUME_MISSING_HINTS)


def hermes_history_has_prior_turns(messages: Optional[list] = None) -> bool:
    """True if Hermes already has prior user/assistant turns (resumed chat).

    Phase 2b MVP does not pre-seed that transcript into a brand-new Claude
    session; only the latest user message is sent.
    """
    if not messages:
        return False
    count = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role in ("user", "assistant"):
            count += 1
            if count > 1:
                return True
    return False


class ClaudeCliSession:
    """Multi-turn Claude CLI session (Phase 2b — create + resume).

    One instance maps one Hermes conversation to one Claude session_id.
    Each ``run_turn`` still spawns a fresh ``claude -p`` (print mode is
    one-shot), but the Claude session file carries history across spawns.
    """

    def __init__(
        self,
        *,
        oauth_token: Optional[str] = None,
        model: str = "claude-opus-4-8",
        claude_bin: Optional[str] = None,
        cwd: Optional[str] = None,
        on_event: Optional[Callable[[dict], None]] = None,
        env: Optional[dict[str, str]] = None,
        client_factory: Optional[Callable[..., ClaudeCliClient]] = None,
        turn_timeout: float = 600.0,
        enable_hermes_mcp: bool = True,
        hermes_mcp_profile: str = "claude",
        max_turns: Optional[int] = None,
        hermes_conversation_id: Optional[str] = None,
        claude_session_id: Optional[str] = None,
    ) -> None:
        self._oauth_token = oauth_token
        self._model = model
        self._claude_bin = resolve_claude_bin(claude_bin)
        self._cwd = cwd
        self._on_event = on_event
        self._env = env
        self._client_factory = client_factory or ClaudeCliClient
        self._turn_timeout = turn_timeout
        self._enable_hermes_mcp = enable_hermes_mcp
        self._hermes_mcp_profile = hermes_mcp_profile
        self._max_turns = max_turns
        self._client: Optional[ClaudeCliClient] = None
        self._closed = False
        self._workspace: Optional[str] = None
        # Hermes conversation id (agent.session_id) — mapping key / logs.
        self._hermes_conversation_id = hermes_conversation_id
        # Claude session UUID used with --session-id / --resume.
        self._claude_session_id: Optional[str] = claude_session_id
        self._turn_index = 0
        self._history_seed_note_emitted = False

    # ---------- public mapping surface ----------

    @property
    def claude_session_id(self) -> Optional[str]:
        return self._claude_session_id

    @property
    def hermes_conversation_id(self) -> Optional[str]:
        return self._hermes_conversation_id

    @property
    def turn_index(self) -> int:
        return self._turn_index

    @property
    def workspace(self) -> Optional[str]:
        """Stable cwd used for Claude session file resolution."""
        if self._cwd:
            return self._cwd
        return self._workspace

    def _ensure_workspace(self) -> str:
        if self._cwd:
            return self._cwd
        if self._workspace is None:
            self._workspace = tempfile.mkdtemp(prefix="hermes-claude-cli-ws-")
            logger.info(
                "claude_cli: created stable workspace cwd=%s hermes_id=%s",
                self._workspace,
                self._hermes_conversation_id,
            )
        return self._workspace

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        # Keep Claude session files on disk (resume-friendly). Workspace is
        # ephemeral when we created it; OS reaps temp dirs. Do not delete
        # Claude's ~/.claude/projects session jsonl here.

    def __enter__(self) -> "ClaudeCliSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def reset_claude_session(self) -> None:
        """Drop the Claude session mapping so the next turn creates a fresh one."""
        logger.info(
            "claude_cli: resetting claude session mapping hermes_id=%s old=%s",
            self._hermes_conversation_id,
            (self._claude_session_id or "")[:8] or None,
        )
        self._claude_session_id = None

    def run_turn(
        self,
        user_input: Any,
        *,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        turn_timeout: Optional[float] = None,
        messages: Optional[list] = None,
    ) -> ClaudeCliTurnResult:
        """Spawn ``claude -p`` (create or resume), stream events, return result.

        Raises ClaudeCliError when the terminal result has is_error=true so
        the agent-loop / fallback chain can classify the failure. Soft
        failures (spawn/timeout/nonzero exit without a result event) also
        raise ClaudeCliError with a stderr tail.

        On a missing/expired resume target, clears the mapping and retries
        once as a fresh ``--session-id`` create.
        """
        if self._closed:
            raise ClaudeCliError(message="claude_cli: session is closed")

        prompt = _coerce_user_text(user_input)
        if not prompt.strip():
            raise ClaudeCliError(message="claude_cli: empty user prompt")

        history_note: Optional[str] = None
        if (
            self._claude_session_id is None
            and not self._history_seed_note_emitted
            and hermes_history_has_prior_turns(messages)
        ):
            history_note = (
                "claude_cli Phase 2b: Hermes conversation already has prior "
                "history, but Claude session is new — only the latest user "
                "message is sent (full transcript pre-seed deferred)."
            )
            self._history_seed_note_emitted = True
            logger.warning(history_note)

        # Decide create vs resume *before* the attempt so create failures
        # never look like resume-missing fallback candidates.
        attempting_resume = bool(self._claude_session_id)
        try:
            return self._run_turn_once(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                turn_timeout=turn_timeout,
                history_note=history_note,
                resume_fallback=False,
            )
        except ClaudeCliConcurrencyError:
            # Host slot saturated — do not treat as resume-missing; let the
            # conversation loop activate the profile fallback chain.
            raise
        except ClaudeCliError as exc:
            if attempting_resume and is_resume_missing_error(str(exc)):
                logger.warning(
                    "claude_cli: resume failed (missing session); "
                    "falling back to fresh --session-id hermes_id=%s err=%s",
                    self._hermes_conversation_id,
                    str(exc)[:200],
                )
                self.reset_claude_session()
                return self._run_turn_once(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=model,
                    turn_timeout=turn_timeout,
                    history_note=history_note,
                    resume_fallback=True,
                )
            raise

    def _run_turn_once(
        self,
        *,
        prompt: str,
        system_prompt: Optional[str],
        model: Optional[str],
        turn_timeout: Optional[float],
        history_note: Optional[str],
        resume_fallback: bool = False,
    ) -> ClaudeCliTurnResult:
        result = ClaudeCliTurnResult(history_seed_note=history_note)
        try:
            token = self._oauth_token or resolve_claude_cli_oauth_token()
        except ClaudeCliError:
            raise
        except Exception as exc:
            raise ClaudeCliError(
                message=f"claude_cli: token resolution failed: {exc}"
            ) from exc

        env = self._env or build_claude_cli_clean_env(oauth_token=token)
        workspace = self._ensure_workspace()

        # Create vs resume decision (OpenClaw sessionMode: always).
        resume_id: Optional[str] = None
        create_id: Optional[str] = None
        if self._claude_session_id:
            resume_id = self._claude_session_id
            result.resumed = True
        else:
            create_id = new_claude_session_id()
            result.created_session = True
            # Pre-bind so a crash mid-turn still leaves a stable id if Claude
            # wrote the session; overwritten from result event when present.
            self._claude_session_id = create_id

        result.resume_fallback = resume_fallback

        # System prompt: only on create. Re-appending on every resume would
        # stack append-system-prompt into Claude's session each turn.
        effective_system = system_prompt if create_id else None

        spawn_kwargs: dict[str, Any] = {
            "model": model or self._model,
            "prompt": prompt,
            "system_prompt": effective_system,
            "claude_bin": self._claude_bin,
            "cwd": workspace,
            "enable_hermes_mcp": self._enable_hermes_mcp,
            "hermes_mcp_profile": self._hermes_mcp_profile,
            "timeout_seconds": float(
                turn_timeout if turn_timeout is not None else self._turn_timeout
            ),
            "session_id": create_id,
            "resume": resume_id,
        }
        if self._max_turns is not None:
            spawn_kwargs["max_turns"] = int(self._max_turns)
        cfg = ClaudeCliSpawnConfig(**spawn_kwargs)

        client = self._client_factory(
            oauth_token=token,
            env=env,
            claude_bin=self._claude_bin,
        )
        self._client = client
        projector = ClaudeEventProjector()

        logger.info(
            "claude_cli turn hermes_id=%s mode=%s claude_id=%s cwd=%s",
            self._hermes_conversation_id,
            "resume" if resume_id else "create",
            (resume_id or create_id or "")[:8],
            workspace,
        )

        # Host-global concurrency slot: one per concurrent `claude -p` across
        # all Hermes profiles/processes. Held for the whole spawn→stream
        # lifetime so a long tool-using turn counts as one occupied slot.
        # On saturation after bounded wait → ClaudeCliConcurrencyError →
        # conversation_loop activates the profile fallback chain.
        slot_meta = {
            "hermes_conversation_id": self._hermes_conversation_id or "",
            "mode": "resume" if resume_id else "create",
            "model": str(model or self._model or ""),
        }
        # force_unbounded is a test hook; production always acquires.
        slot_cm = (
            claude_cli_slot(metadata=slot_meta)
            if not is_force_unbounded()
            else _null_slot_cm()
        )
        with slot_cm:
            return self._drive_spawned_turn(
                client=client,
                cfg=cfg,
                projector=projector,
                result=result,
                resume_id=resume_id,
                create_id=create_id,
            )

    def _drive_spawned_turn(
        self,
        *,
        client: Any,
        cfg: ClaudeCliSpawnConfig,
        projector: ClaudeEventProjector,
        result: ClaudeCliTurnResult,
        resume_id: Optional[str],
        create_id: Optional[str],
    ) -> ClaudeCliTurnResult:
        """Spawn + stream one ``claude -p`` (called while holding a host slot)."""
        try:
            client.spawn(cfg)
        except FileNotFoundError as exc:
            raise ClaudeCliError(
                message=(
                    f"claude_cli: binary not found ({self._claude_bin!r}). "
                    "Install Claude Code (npm i -g @anthropic-ai/claude-code) "
                    "or put `claude` on PATH."
                )
            ) from exc
        except ClaudeCliConcurrencyError:
            raise
        except Exception as exc:
            raise ClaudeCliError(
                message=f"claude_cli: failed to spawn subprocess: {exc}"
            ) from exc

        timeout = cfg.timeout_seconds
        try:
            for line in client.iter_stdout_lines(timeout=timeout):
                state = projector.consume_line(line)
                # Fire stream deltas + tool activity to Hermes UI bridge.
                if self._on_event is not None:
                    for delta in state.last_text_deltas:
                        try:
                            self._on_event(
                                {
                                    "method": "claude/text_delta",
                                    "params": {"delta": delta, "text": delta},
                                }
                            )
                        except Exception:
                            logger.debug(
                                "claude_cli on_event raised", exc_info=True
                            )
                    for rec in state.last_tool_started:
                        try:
                            self._on_event(
                                {
                                    "method": "claude/tool_started",
                                    "params": {
                                        "id": rec.id,
                                        "name": rec.name,
                                        "raw_name": rec.raw_name,
                                        "arguments": rec.input,
                                        "server": "hermes-tools",
                                    },
                                }
                            )
                        except Exception:
                            logger.debug(
                                "claude_cli tool_started on_event raised",
                                exc_info=True,
                            )
                    for rec in state.last_tool_completed:
                        try:
                            self._on_event(
                                {
                                    "method": "claude/tool_completed",
                                    "params": {
                                        "id": rec.id,
                                        "name": rec.name,
                                        "raw_name": rec.raw_name,
                                        "arguments": rec.input,
                                        "result": rec.result,
                                        "is_error": rec.is_error,
                                        "server": "hermes-tools",
                                    },
                                }
                            )
                        except Exception:
                            logger.debug(
                                "claude_cli tool_completed on_event raised",
                                exc_info=True,
                            )
                if state.finished:
                    break
        except ClaudeCliError:
            client.close()
            # Preserve mapping on stream timeout so caller can decide retire.
            raise
        except Exception as exc:
            tail = "\n".join(client.stderr_tail(_STDERR_TAIL_LINES))
            client.close()
            raise ClaudeCliError(
                message=f"claude_cli: stream read failed: {exc}",
                stderr_tail=redact_sensitive_text(tail, force=True) if tail else "",
            ) from exc

        state = projector.state
        exit_code = client.wait(timeout=5.0)
        stderr_tail_raw = "\n".join(client.stderr_tail(_STDERR_TAIL_LINES))
        stderr_tail = (
            redact_sensitive_text(stderr_tail_raw, force=True)
            if stderr_tail_raw
            else ""
        )
        client.close()
        self._client = None

        # Prefer session_id from Claude's init/result events when present.
        if state.session_id:
            self._claude_session_id = state.session_id
        result.session_id = self._claude_session_id
        result.final_text = state.final_text
        result.usage = state.usage
        result.token_usage_last = _usage_to_codex_shape(state.usage)
        result.total_cost_usd = state.total_cost_usd
        result.result_event = state.result_event
        result.is_error = state.is_error
        result.should_retire = False  # healthy multi-turn reuse by default
        result.tool_iterations = state.tool_iterations
        result.tool_calls = [
            {
                "id": t.id,
                "name": t.name,
                "raw_name": t.raw_name,
                "arguments": t.input,
                "result": t.result,
                "is_error": t.is_error,
                "completed": t.completed,
            }
            for t in state.tool_calls
        ]

        if state.final_text:
            result.projected_messages = [
                {"role": "assistant", "content": state.final_text}
            ]

        # Terminal result with is_error=true → raise so fallback can fire.
        if state.is_error:
            err_msg = (
                state.result_text
                or (state.result_event or {}).get("error")
                or "claude_cli result is_error=true"
            )
            if not isinstance(err_msg, str):
                err_msg = str(err_msg)
            result.error = err_msg
            # Resume-missing errors: keep mapping for outer fallback path
            # (run_turn will clear + retry). Other hard errors retire.
            if resume_id and is_resume_missing_error(err_msg):
                raise ClaudeCliError(
                    message=f"claude_cli turn error: {err_msg}",
                    is_error=True,
                    exit_code=exit_code,
                    stderr_tail=stderr_tail,
                    result_event=state.result_event,
                )
            result.should_retire = True
            raise ClaudeCliError(
                message=f"claude_cli turn error: {err_msg}",
                is_error=True,
                exit_code=exit_code,
                stderr_tail=stderr_tail,
                result_event=state.result_event,
            )

        # No result event + nonzero exit → error with stderr tail.
        if not state.finished and exit_code not in (0, None):
            combined = f"claude_cli exited without a result event (exit={exit_code})"
            if stderr_tail:
                combined = f"{combined}\n{stderr_tail}"
            if resume_id and is_resume_missing_error(combined):
                raise ClaudeCliError(
                    message=combined,
                    exit_code=exit_code,
                    stderr_tail=stderr_tail,
                )
            result.should_retire = True
            raise ClaudeCliError(
                message=(
                    f"claude_cli exited without a result event "
                    f"(exit={exit_code})"
                ),
                exit_code=exit_code,
                stderr_tail=stderr_tail,
            )

        # Finished successfully but empty text — still a valid stop (model
        # can return blank); surface as soft error only if exit nonzero.
        if not state.finished and not result.final_text:
            if exit_code not in (0, None):
                result.should_retire = True
                raise ClaudeCliError(
                    message="claude_cli produced no stream-json result",
                    exit_code=exit_code,
                    stderr_tail=stderr_tail,
                )
            logger.warning(
                "claude_cli: no result event and empty text (exit=%s)",
                exit_code,
            )

        # Fire assistant_completed for UI finalization. Prefer result_text so
        # the completed frame is clean even when intermediate partials
        # narrated tool intent. Bridge consumers should treat this as the
        # authoritative final (replace, not append) relative to stream deltas.
        if self._on_event is not None and result.final_text:
            try:
                self._on_event(
                    {
                        "method": "claude/assistant_completed",
                        "params": {
                            "text": result.final_text,
                            # Hint for UIs that already painted text_delta:
                            # replace the stream buffer rather than append.
                            "replace_stream": True,
                        },
                    }
                )
            except Exception:
                logger.debug("claude_cli on_event completed raised", exc_info=True)

        self._turn_index += 1
        return result


@contextmanager
def _null_slot_cm() -> Iterator[None]:
    """No-op slot context for tests that force unbounded concurrency."""
    yield None


def _coerce_user_text(user_input: Any) -> str:
    if isinstance(user_input, str):
        return user_input
    if isinstance(user_input, list):
        parts: list[str] = []
        for item in user_input:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") in {"text", "input_text"}:
                    text = item.get("text") or item.get("content") or ""
                    if text:
                        parts.append(str(text))
                elif item.get("type") in {"image", "image_url", "input_image"}:
                    parts.append("[image attached]")
            elif item is not None:
                parts.append(str(item))
        return "\n\n".join(p for p in parts if p).strip()
    return "" if user_input is None else str(user_input)


def _usage_to_codex_shape(usage: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Translate Claude usage keys into the shape codex_runtime recorders know."""
    if not isinstance(usage, dict) or not usage:
        return None
    input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    cache_read = int(
        usage.get("cache_read_input_tokens") or usage.get("cache_read_tokens") or 0
    )
    return {
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        "cachedInputTokens": cache_read,
        "reasoningOutputTokens": 0,
        "totalTokens": input_tokens + output_tokens + cache_read,
    }


__all__ = [
    "ClaudeCliSession",
    "ClaudeCliTurnResult",
    "hermes_history_has_prior_turns",
    "is_resume_missing_error",
    "new_claude_session_id",
    "resolve_claude_cli_oauth_token",
]
