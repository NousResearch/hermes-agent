"""OpenAI-compatible shim that forwards Hermes requests to `junie --acp=true`.

This adapter lets Hermes treat the JetBrains Junie CLI's ACP server as a
chat-style backend. Each request starts a short-lived ACP session, sends the
formatted conversation as a single prompt, collects text chunks, and converts
the result back into the minimal shape Hermes expects from an OpenAI client.

The JSON-RPC / stdio plumbing is delegated to the official Agent Client
Protocol Python SDK (``agent-client-protocol``): Hermes acts as an ACP *client*
(:class:`acp.ClientSideConnection`) driving the Junie agent subprocess, and a
small :class:`_HermesClient` implements the client-side callbacks (fs reads/
writes, permission prompts, and streamed ``session/update`` notifications).

The SDK is asyncio-native, but Hermes calls ``chat.completions.create`` on a
synchronous code path (see ``agent/chat_completion_helpers.py``). We therefore
own a dedicated asyncio event loop on a background daemon thread and bridge each
request with :func:`asyncio.run_coroutine_threadsafe`.

Junie specifics (vs Copilot):
  * launched as ``junie --acp=true`` (Copilot uses ``copilot --acp --stdio``);
  * auth is supplied via ``--auth <token>`` / ``JUNIE_API_KEY`` rather than
    being wholly owned by the CLI;
  * Junie wraps each JSON-RPC message with an extra
    ``"type": "com.agentclientprotocol.rpc.*"`` envelope field, which the SDK's
    parser simply ignores (messages are matched by ``id`` / ``method``).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import logging
import os
from collections import deque
import shlex
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent.file_safety import get_read_block_error, is_write_denied
from agent.redact import redact_sensitive_text

logger = logging.getLogger(__name__)

# The ACP SDK is an optional dependency (the ``acp`` extra). Import lazily-ish:
# module import must succeed for the pure helpers below (imported by tests and
# by callers that only touch resolution logic), but actually driving a Junie
# subprocess requires the SDK. ``_require_acp`` raises a clear, actionable error
# when it is missing.
try:  # pragma: no cover - trivial import guard
    import acp as _acp  # noqa: N813
    from acp.exceptions import RequestError
    from acp.schema import (
        AllowedOutcome,
        ClientCapabilities,
        DeniedOutcome,
        FileSystemCapabilities,
        Implementation,
        ReadTextFileResponse,
        RequestPermissionResponse,
    )

    _ACP_IMPORT_ERROR: Exception | None = None
except Exception as _exc:  # pragma: no cover - only hit without the extra
    _acp = None  # type: ignore[assignment]
    _ACP_IMPORT_ERROR = _exc


ACP_MARKER_BASE_URL = "acp://junie"
_DEFAULT_TIMEOUT_SECONDS = 900.0
# Used when the caller passes an explicit "no timeout" (httpx.Timeout(None)):
# a long-but-finite ceiling so a legitimate long coding run isn't cut at 900s
# but a truly hung subprocess still eventually unblocks.
_NO_TIMEOUT_FALLBACK_SECONDS = 3600.0
# Handshake (JVM cold start) can be slow; give initialize/session-setup room.
_HANDSHAKE_TIMEOUT_SECONDS = 90.0
# Per-call caps inside a turn so a wedged (but still "alive") reused process
# fails fast on session/new — recovering on a fresh process — instead of
# hanging the whole request until the outer turn timeout.
_SESSION_NEW_TIMEOUT_SECONDS = 60.0
_CONFIG_OPTION_TIMEOUT_SECONDS = 15.0
# stdin reader buffer for the agent subprocess. Junie streams large tool_call
# payloads (file contents) on a single line; the SDK default (64KB) would raise
# LimitOverrunError, so match the SDK's own 50MB ceiling for multimodal data.
_STDIO_LIMIT_BYTES = 50 * 1024 * 1024
# After session/prompt returns, wait until Junie has been quiet for this long
# before finalizing — the ACP completion response can precede a trailing
# agent_message_chunk / task finalization, and the latency that matters is the
# real subprocess stdout-flush / GC delay (NOT in-process dispatch), so keep the
# original conservative gap. Capped by ``_SETTLE_MAX_SECONDS``. Tests override
# both to tiny values (see tests/agent/test_junie_acp_client.py:_make_client).
_SETTLE_QUIET_GAP_SECONDS = 2.5
_SETTLE_MAX_SECONDS = 20.0

# Junie's ACP session/update kinds that report tool activity. Unlike an OpenAI
# model, Junie is an autonomous agent that EXECUTES its own tools and reports
# them here (status pending -> in_progress -> completed/failed) — these are NOT
# delegation requests for Hermes to run. We consume them for observability, not
# to fabricate OpenAI tool_calls (see JunieACPClient._create_chat_completion).
_TOOL_UPDATE_KINDS = ("tool_call", "tool_call_update")

# Model values that mean "let Junie use its own default" — the provider
# sentinel/aliases, not real Junie model ids. These are NOT forwarded to
# session/set_config_option{model}.
_MODEL_PASSTHROUGH_SENTINELS = frozenset({
    "junie-acp", "junie", "jetbrains-junie-acp", "junie-acp-agent", "",
})

# Prefer approving *once*: only grant persistent auto-approval if a request
# offers no single-shot option (see _choose_permission_option).
_ALLOW_OPTION_KINDS = ("allow_once", "allow_always")


def _client_capabilities() -> Any:
    """The ClientCapabilities Hermes advertises to Junie (fs on, no terminal)."""
    return ClientCapabilities(
        fs=FileSystemCapabilities(read_text_file=True, write_text_file=True),
        terminal=False,
    )


def _client_info() -> Any:
    return Implementation(name="hermes-agent", title="Hermes Agent", version="0.0.0")


def _require_acp() -> None:
    if _acp is None:
        raise RuntimeError(
            "The Agent Client Protocol SDK is required for the junie-acp provider. "
            "Install it with `pip install 'hermes-agent[acp]'` (or "
            "`pip install agent-client-protocol`)."
        ) from _ACP_IMPORT_ERROR


def _rough_tokens(text: str | None) -> int:
    """Rough token estimate (~4 chars/token). Used only as a fallback when
    Junie does not report usage; see JunieACPClient._create_chat_completion."""
    return len(text) // 4 if text else 0


def _resolve_command() -> str:
    return (
        os.getenv("HERMES_JUNIE_ACP_COMMAND", "").strip()
        or os.getenv("JUNIE_CLI_PATH", "").strip()
        or "junie"
    )


def _resolve_args() -> list[str]:
    raw = os.getenv("HERMES_JUNIE_ACP_ARGS", "").strip()
    args = shlex.split(raw) if raw else ["--acp=true", "--skip-update-check"]
    # Inject auth from the environment when the caller hasn't already supplied
    # a token via --auth. Junie accepts a JetBrains/Junie token (perm-...).
    if "--auth" not in args and not any(a.startswith("--auth=") for a in args):
        token = os.getenv("JUNIE_API_KEY", "").strip()
        if token:
            args = args + ["--auth", token]
    return args


def _resolve_permission_policy() -> str:
    """How the client answers Junie's session/request_permission requests.

    "deny" (default, safe): reject every request (Junie can't act without
    explicit consent — matters only when Brave Mode is OFF). "allow": approve
    once. This is the seam a future Hermes-approval bridge would replace.
    """
    val = os.getenv("HERMES_JUNIE_ACP_PERMISSION", "").strip().lower()
    return "allow" if val in ("allow", "allow_once", "yes", "1", "true") else "deny"


def _resolve_brave_override() -> bool | None:
    """Optional override for Junie's Brave Mode (auto-execute without asking).

    Returns True/False to force it via session/set_config_option, or None to
    leave Junie on its own persisted/default setting.
    """
    val = os.getenv("HERMES_JUNIE_ACP_BRAVE", "").strip().lower()
    if val in ("on", "1", "true", "yes"):
        return True
    if val in ("off", "0", "false", "no"):
        return False
    return None


def _resolve_home_dir() -> str:
    """Return a stable HOME for child ACP processes."""
    home = os.environ.get("HOME", "").strip()
    if home:
        return home

    expanded = os.path.expanduser("~")
    if expanded and expanded != "~":
        return expanded

    try:
        import pwd

        resolved = pwd.getpwuid(os.getuid()).pw_dir.strip()  # windows-footgun: ok — POSIX fallback inside try/except (pwd import fails on Windows)
        if resolved:
            return resolved
    except Exception:
        pass

    # Last resort: /tmp (writable on any POSIX system). Avoids crashing the
    # subprocess with no HOME; callers can set HERMES_HOME explicitly if they
    # need a different writable dir.
    return "/tmp"


def _build_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    home = _resolve_home_dir()
    env["HOME"] = home
    from hermes_constants import apply_subprocess_home_env
    apply_subprocess_home_env(env)
    return env


def _choose_permission_option(options: list[dict[str, Any]], policy: str) -> str | None:
    """Pick the optionId to approve for a session/request_permission request.

    Returns None to reject (``policy="deny"`` or no allow-kind option offered).
    "allow" means approve *once*: prefer an allow_once option and only fall back
    to allow_always if the request offers no single-shot option — never grant
    persistent auto-approval just because it happened to be listed first.
    """
    if policy != "allow":
        return None
    opts = [o for o in options if isinstance(o, dict)]
    for kind in _ALLOW_OPTION_KINDS:
        for opt in opts:
            if str(opt.get("kind", "")).lower() == kind:
                chosen = opt.get("optionId") or opt.get("id")
                if chosen:
                    return str(chosen)
    return None


def _format_messages_as_prompt(
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> str:
    # Junie is an autonomous coding agent: it runs its OWN tools (read/edit/
    # execute) inside the ACP session and reports them via native tool_call
    # notifications. We therefore do NOT ask it to emit OpenAI-style tool
    # calls; it should just do the work and answer. Hermes' own tool schemas
    # are irrelevant to Junie's execution, so we don't inject them.
    del tools, tool_choice  # accepted for OpenAI-client compatibility; unused
    sections: list[str] = [
        "You are being used as the active ACP coding agent backend for Hermes.",
        "Use your own tools to complete the task, then answer normally.",
    ]
    if model:
        sections.append(f"Hermes requested model hint: {model}")

    transcript: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown").strip().lower()
        if role == "tool":
            role = "tool"
        elif role not in {"system", "user", "assistant"}:
            role = "context"

        content = message.get("content")
        rendered = _render_message_content(content)
        if not rendered:
            continue

        label = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
            "tool": "Tool",
            "context": "Context",
        }.get(role, role.title())
        transcript.append(f"{label}:\n{rendered}")

    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))

    sections.append("Continue the conversation from the latest user request.")
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


def _render_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "").strip()
        if "content" in content and isinstance(content.get("content"), str):
            return str(content.get("content") or "").strip()
        return json.dumps(content, ensure_ascii=True)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def _chunk_text(content: Any) -> str:
    """Extract text from a session/update content that may be a dict or a list
    of {type,text} blocks."""
    if isinstance(content, dict):
        return str(content.get("text") or "")
    if isinstance(content, list):
        return "".join(
            str(b.get("text") or "") for b in content if isinstance(b, dict)
        )
    return ""


def _tool_update_text(update: dict[str, Any]) -> str:
    """Best-effort plain-text extraction from an ACP tool_call content list."""
    parts: list[str] = []
    for block in update.get("content") or []:
        if not isinstance(block, dict):
            continue
        inner = block.get("content")
        if isinstance(inner, dict) and isinstance(inner.get("text"), str):
            parts.append(inner["text"])
        elif isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(p for p in parts if p and p.strip()).strip()


def _merge_tool_update(store: dict[str, dict[str, Any]], update: dict[str, Any]) -> None:
    """Fold a tool_call / tool_call_update notification into ``store`` by id.

    Junie streams a ``tool_call`` (first-seen) then zero or more
    ``tool_call_update`` messages sharing the same ``toolCallId`` as the tool
    progresses (pending -> in_progress -> completed/failed). We keep the latest
    non-empty value for each field so ``store`` ends up with the final state.
    """
    tcid = str(update.get("toolCallId") or f"tool_{len(store)}")
    entry = store.setdefault(tcid, {"id": tcid})
    for field in ("title", "kind", "status"):
        val = update.get(field)
        if val:
            entry[field] = val
    text = _tool_update_text(update)
    if text:
        entry["result"] = text
    locations = update.get("locations")
    if locations:
        entry["locations"] = locations


def _render_tool_activity(tool_events: dict[str, dict[str, Any]]) -> str:
    """Render captured tool activity as a compact, human-readable summary.

    Surfaced via the assistant message's ``reasoning`` so the operator can see
    what Junie actually did, without misrepresenting completed actions as
    OpenAI tool_calls Hermes must execute.
    """
    lines: list[str] = []
    for ev in tool_events.values():
        kind = ev.get("kind", "tool")
        status = ev.get("status", "")
        title = ev.get("title", "")
        head = f"[{kind}] {title}".strip()
        if status:
            head = f"{head} ({status})"
        lines.append(head)
        result = ev.get("result")
        if result:
            snippet = result if len(result) <= 500 else result[:500] + "…"
            lines.append(f"    → {snippet}")
    return "\n".join(lines).strip()


def _ensure_path_within_cwd(path_text: str, cwd: str) -> Path:
    candidate = Path(path_text)
    if not candidate.is_absolute():
        raise PermissionError("ACP file-system paths must be absolute.")
    resolved = candidate.resolve()
    root = Path(cwd).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"Path '{resolved}' is outside the session cwd '{root}'.") from exc
    return resolved


def _read_text_file_content(
    path_text: str, cwd: str, *, line: int | None, limit: int | None
) -> str:
    """Safely read a file for an fs/read_text_file request.

    Enforces path-within-cwd, honors Hermes' read-blocklist, redacts secrets,
    and applies ACP line/limit pagination (honoring ``limit`` even from the top
    so a paginated read doesn't return the whole file).
    """
    path = _ensure_path_within_cwd(path_text, cwd)
    block_error = get_read_block_error(str(path))
    if block_error:
        raise PermissionError(block_error)
    try:
        content = path.read_text()
    except FileNotFoundError:
        content = ""
    has_line = isinstance(line, int) and line > 1
    has_limit = isinstance(limit, int) and limit > 0
    if has_line or has_limit:
        lines = content.splitlines(keepends=True)
        start = line - 1 if has_line else 0
        end = start + limit if has_limit else None
        content = "".join(lines[start:end])
    if content:
        content = redact_sensitive_text(content, force=True)
    return content


def _write_text_file(path_text: str, cwd: str, content: str) -> None:
    """Safely write a file for an fs/write_text_file request.

    Enforces path-within-cwd and Hermes' write-deny policy (protected
    system/credential files).
    """
    path = _ensure_path_within_cwd(path_text, cwd)
    if is_write_denied(str(path)):
        raise PermissionError(
            f"Write denied: '{path}' is a protected system/credential file."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


# --------------------------------------------------------------------------- #
# Live model discovery for the /model picker                                  #
# --------------------------------------------------------------------------- #
# Junie advertises its selectable models in the session/new response's
# ``config_options`` (the ``model`` Select option). Discovering them live keeps
# the picker in sync with whatever the account/CLI actually offers, without
# hardcoding ids into _PROVIDER_MODELS (which would make detect_provider_for_model
# mis-resolve claude/gemini/gpt ids to junie-acp). Cached because each probe
# spawns Junie (JVM cold start).
_MODEL_CACHE_TTL_SECONDS = 6 * 3600
# Negative results (unauthed / offline / timeout) are cached in-memory only, for
# a short window, so a wedged /model picker doesn't re-spawn Junie on every open
# — but a fresh process (or a re-auth after this window) re-probes promptly.
_MODEL_NEG_TTL_SECONDS = 300
# key -> (ts, ids); an empty ids list is a cached negative result.
_model_cache: dict[str, tuple[float, list[str]]] = {}


def _account_fingerprint(args: list[str]) -> str:
    """Short, non-reversible tag for the Junie account behind these args.

    Keys the model cache so switching accounts (different --auth token /
    JUNIE_API_KEY) doesn't serve the previous account's catalog. Only the hash
    is ever stored — never the raw token, on disk or in memory.
    """
    import hashlib

    token = ""
    for i, a in enumerate(args):
        if a == "--auth" and i + 1 < len(args):
            token = args[i + 1]
            break
        if a.startswith("--auth="):
            token = a.split("=", 1)[1]
            break
    if not token:
        token = os.getenv("JUNIE_API_KEY", "").strip()
    if not token:
        return "noauth"
    return hashlib.sha256(token.encode("utf-8", "ignore")).hexdigest()[:12]


def _model_cache_key(command: str, args: list[str]) -> str:
    return f"{command}#{_account_fingerprint(args)}"


def _extract_model_ids(config_options: Any) -> list[str]:
    """Pull the ``model`` config option's advertised value ids (current first)."""
    for opt in config_options or []:
        data = opt.model_dump(by_alias=True, exclude_none=True) if hasattr(opt, "model_dump") else dict(opt)
        if data.get("id") != "model" and data.get("configId") != "model":
            continue
        ids: list[str] = []
        for o in data.get("options") or []:
            if isinstance(o, dict):
                val = o.get("value") or o.get("valueId") or o.get("optionId")
                if val:
                    ids.append(str(val))
        current = data.get("currentValue")
        if current and current in ids:
            ids = [current] + [i for i in ids if i != current]
        return ids
    return []


def _model_disk_cache_path() -> Path:
    from hermes_constants import get_hermes_home
    return Path(get_hermes_home()) / "junie_acp_models_cache.json"


def _load_model_disk_cache(key: str) -> tuple[float, list[str]] | None:
    """Return ``(stored_ts, ids)`` for ``key`` or None. Freshness is judged by
    the caller against the original timestamp (so loading never re-stamps the
    TTL). Only positive (non-empty) results are ever on disk."""
    try:
        path = _model_disk_cache_path()
        if not path.exists():
            return None
        blob = json.loads(path.read_text())
        entry = blob.get(key)
        if not entry:
            return None
        ids = entry.get("ids")
        if not (isinstance(ids, list) and ids):
            return None
        return float(entry.get("ts", 0)), [str(i) for i in ids]
    except Exception:
        return None


def _save_model_disk_cache(key: str, ids: list[str]) -> None:
    # ``key`` = "<command>#<account-hash>": never persists the raw --auth token.
    # Negatives are NOT written (they live in-memory only) so a fresh process
    # after re-auth re-probes immediately instead of honoring a stale miss.
    if not ids:
        return
    try:
        path = _model_disk_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {}
        if path.exists():
            with contextlib.suppress(Exception):
                blob = json.loads(path.read_text())
        if not isinstance(blob, dict):
            blob = {}
        blob[key] = {"ts": time.time(), "ids": ids}
        path.write_text(json.dumps(blob))
    except Exception:
        pass


async def _afetch_junie_models(command: str, args: list[str], cwd: str) -> list[str]:
    handler = _HermesClient(cwd=cwd, permission_policy="deny")
    async with contextlib.AsyncExitStack() as stack:
        conn, _proc = await stack.enter_async_context(
            _acp.spawn_agent_process(
                handler, command, *args,
                env=_build_subprocess_env(), cwd=cwd,
                transport_kwargs={"limit": _STDIO_LIMIT_BYTES},
            )
        )
        await conn.initialize(
            protocol_version=_acp.PROTOCOL_VERSION,
            client_capabilities=_client_capabilities(),
            client_info=_client_info(),
        )
        session = await conn.new_session(cwd=cwd, mcp_servers=[])
        return _extract_model_ids(getattr(session, "config_options", None))


def fetch_junie_models(
    *,
    command: str | None = None,
    args: list[str] | None = None,
    cwd: str | None = None,
    timeout: float = 20.0,
    force_refresh: bool = False,
) -> list[str] | None:
    """Return the models Junie advertises over ACP, or None if unavailable.

    Spawns ``junie --acp=true`` and reads the ``model`` config option from the
    session/new response. Positive results are cached in-memory + on disk (6h,
    keyed per account); failures (SDK missing, CLI absent, not authed, timeout)
    return None and are negatively cached in-memory for a short window so the
    /model picker doesn't re-spawn a JVM on every open. Callers fall back to the
    curated sentinel list on None.
    """
    if _acp is None:
        return None
    command = command or _resolve_command()
    args = list(args if args is not None else _resolve_args())
    cwd = str(Path(cwd or os.getcwd()).resolve())
    key = _model_cache_key(command, args)
    now = time.time()

    def _fresh(ts: float, ids: list[str]) -> bool:
        ttl = _MODEL_CACHE_TTL_SECONDS if ids else _MODEL_NEG_TTL_SECONDS
        return (now - ts) < ttl

    if not force_refresh:
        mem = _model_cache.get(key)
        if mem and _fresh(*mem):
            return mem[1] or None
        disk = _load_model_disk_cache(key)  # positives only; original ts preserved
        if disk and _fresh(*disk):
            _model_cache[key] = disk  # keep the disk timestamp — don't re-stamp the TTL
            return disk[1] or None

    async def _runner() -> list[str]:
        return await asyncio.wait_for(_afetch_junie_models(command, args, cwd), timeout)

    try:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            ids = asyncio.run(_runner())
        else:
            # Called from within a running loop: run on a private loop thread.
            # (This still blocks the caller — provider_model_ids is a sync API —
            # but avoids the "asyncio.run() inside a running loop" error.)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                ids = ex.submit(lambda: asyncio.run(_runner())).result(timeout + 5)
    except Exception as exc:
        logger.debug("fetch_junie_models failed: %s", exc)
        ids = []  # negative cache below so we don't re-spawn Junie every open

    _model_cache[key] = (time.time(), ids)
    if ids:
        _save_model_disk_cache(key, ids)
        return ids
    return None


class _ACPChatCompletions:
    def __init__(self, client: "JunieACPClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ACPChatNamespace:
    def __init__(self, client: "JunieACPClient"):
        self.completions = _ACPChatCompletions(client)


class _HermesClient:
    """Client-side ACP callbacks Hermes exposes to the Junie agent.

    Implements the parts of :class:`acp.Client` that Junie exercises: streamed
    ``session/update`` notifications (text / thought / tool activity), the
    ``session/request_permission`` prompt, and sandboxed ``fs/read_text_file`` /
    ``fs/write_text_file``. Terminal methods are intentionally unimplemented —
    the SDK router treats them as optional and returns a null default.

    One instance is reused across turns on the client's event loop; per-turn
    collection buffers are installed by :meth:`begin_turn` and cleared by
    :meth:`end_turn`, so notifications are single-threaded on that loop and need
    no locking.
    """

    def __init__(self, *, cwd: str, permission_policy: str):
        self._cwd = cwd
        self._policy = permission_policy
        self._session_id: str | None = None
        self._text_parts: list[str] | None = None
        self._reasoning_parts: list[str] | None = None
        self._tool_events: dict[str, dict[str, Any]] | None = None
        self._last_activity = 0.0

    def begin_turn(
        self,
        session_id: str,
        text_parts: list[str],
        reasoning_parts: list[str],
        tool_events: dict[str, dict[str, Any]],
    ) -> None:
        self._session_id = session_id
        self._text_parts = text_parts
        self._reasoning_parts = reasoning_parts
        self._tool_events = tool_events
        self._last_activity = time.monotonic()

    def end_turn(self) -> None:
        self._session_id = None
        self._text_parts = None
        self._reasoning_parts = None
        self._tool_events = None

    async def settle(self, quiet_gap: float, max_wait: float) -> None:
        """Wait until Junie has been quiet for ``quiet_gap`` (capped at
        ``max_wait``) so a trailing chunk after the completion isn't lost."""
        if max_wait <= 0:
            return
        deadline = time.monotonic() + max_wait
        while time.monotonic() < deadline:
            if time.monotonic() - self._last_activity >= quiet_gap:
                return
            await asyncio.sleep(0.05)

    async def session_update(self, session_id: str, update: Any, **_: Any) -> None:
        # One reused process serves many sessions over one connection. Drop
        # updates belonging to a *different* session (e.g. a straggler chunk
        # from a previous turn) so a prior answer can't leak into this turn.
        if self._session_id is not None and session_id and session_id != self._session_id:
            return
        self._last_activity = time.monotonic()
        data = update.model_dump(by_alias=True, exclude_none=True) if hasattr(update, "model_dump") else dict(update)
        kind = str(data.get("sessionUpdate") or "").strip()
        if kind in _TOOL_UPDATE_KINDS:
            # Native structured tool activity — captured for observability, NOT
            # turned into OpenAI tool_calls.
            if self._tool_events is not None:
                _merge_tool_update(self._tool_events, data)
            return
        chunk_text = _chunk_text(data.get("content"))
        if kind == "agent_message_chunk" and chunk_text and self._text_parts is not None:
            self._text_parts.append(chunk_text)
        elif kind == "agent_thought_chunk" and chunk_text and self._reasoning_parts is not None:
            self._reasoning_parts.append(chunk_text)

    async def request_permission(
        self, options: list[Any], session_id: str, tool_call: Any, **_: Any
    ) -> Any:
        opt_dicts = [
            (o.model_dump(by_alias=True, exclude_none=True) if hasattr(o, "model_dump") else dict(o))
            for o in (options or [])
        ]
        tc = tool_call.model_dump(by_alias=True, exclude_none=True) if hasattr(tool_call, "model_dump") else dict(tool_call or {})
        title = tc.get("title") or tc.get("toolCallId") or "?"
        chosen = _choose_permission_option(opt_dicts, self._policy)
        if chosen:
            logger.info("Junie ACP permission ALLOW: %s (option=%s)", title, chosen)
            return RequestPermissionResponse(outcome=AllowedOutcome(option_id=chosen, outcome="selected"))
        logger.info("Junie ACP permission DENY: %s (policy=%s)", title, self._policy)
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    async def read_text_file(
        self, path: str, session_id: str, limit: int | None = None, line: int | None = None, **_: Any
    ) -> Any:
        try:
            content = _read_text_file_content(path, self._cwd, line=line, limit=limit)
        except Exception as exc:
            raise RequestError.invalid_params({"details": str(exc)}) from exc
        return ReadTextFileResponse(content=content)

    async def write_text_file(self, content: str, path: str, session_id: str, **_: Any) -> None:
        try:
            _write_text_file(path, self._cwd, content or "")
        except Exception as exc:
            raise RequestError.invalid_params({"details": str(exc)}) from exc
        return None

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        raise RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        return None


class JunieACPClient:
    """Minimal OpenAI-client-compatible facade for JetBrains Junie ACP."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        acp_command: str | None = None,
        acp_args: list[str] | None = None,
        acp_cwd: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        permission_policy: str | None = None,
        brave_mode: bool | None = None,
        **_: Any,
    ):
        self.api_key = api_key or "junie-acp"
        self.base_url = base_url or ACP_MARKER_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._acp_command = acp_command or command or _resolve_command()
        self._acp_args = list(acp_args or args or _resolve_args())
        self._acp_cwd = str(Path(acp_cwd or os.getcwd()).resolve())
        self._permission_policy = permission_policy or _resolve_permission_policy()
        # None => don't override Junie's own Brave Mode setting.
        self._brave_override = brave_mode if brave_mode is not None else _resolve_brave_override()
        self.chat = _ACPChatNamespace(self)
        self.is_closed = False

        # Persistent subprocess reused across requests (avoids Junie/JVM cold
        # start on every Hermes step). One process handles many independent
        # sessions; we open a fresh session per request and re-send the full
        # transcript, so we never rely on Junie's cross-turn memory matching
        # Hermes' history (no divergence risk).
        self._handler = _HermesClient(cwd=self._acp_cwd, permission_policy=self._permission_policy)
        # SDK objects live on a dedicated event loop running on a daemon thread.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._conn: Any = None  # acp.ClientSideConnection
        self._proc: Any = None  # asyncio subprocess.Process
        self._exit_stack: contextlib.AsyncExitStack | None = None
        self._initialized = False
        # Last lines of the subprocess's stderr, surfaced in failure messages so
        # a crash/auth-rejection gives an actionable reason (not an opaque
        # timeout/connection error).
        self._stderr_tail: deque[str] = deque(maxlen=40)
        # Overridable for tests (fast settle); production keeps the safe gap.
        self._settle_quiet_gap = _SETTLE_QUIET_GAP_SECONDS
        self._settle_max = _SETTLE_MAX_SECONDS
        # True once session/prompt has been dispatched this turn — gates the
        # no-replay retry policy (Junie may already have applied side effects).
        self._prompt_in_flight = False
        self._req_lock = threading.Lock()

    # ---- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self.is_closed = True
        loop = self._loop
        if loop is not None and not loop.is_closed():
            try:
                self._submit(self._aclose_proc(), timeout=5.0)
            except Exception:
                pass
            loop.call_soon_threadsafe(loop.stop)
        thread = self._loop_thread
        if thread is not None:
            thread.join(timeout=3.0)
        if loop is not None and not loop.is_closed() and not (thread and thread.is_alive()):
            # Release the loop's selector/selfpipe fds once its thread has
            # stopped (avoids ResourceWarning: unclosed event loop).
            with contextlib.suppress(Exception):
                loop.close()
        self._loop = None
        self._loop_thread = None

    def _stderr_suffix(self) -> str:
        text = "\n".join(self._stderr_tail).strip()
        return f" (Junie stderr: {text})" if text else ""

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        loop = self._loop
        if loop is not None and not loop.is_closed() and self._loop_thread and self._loop_thread.is_alive():
            return loop
        _require_acp()
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, name="junie-acp-loop", daemon=True)
        thread.start()
        self._loop = loop
        self._loop_thread = thread
        return loop

    def _submit(self, coro: Any, *, timeout: float) -> Any:
        """Run ``coro`` on the background loop and block until it finishes."""
        loop = self._loop
        if loop is None:
            raise RuntimeError("Junie ACP event loop is not running.")
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return future.result(timeout)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise TimeoutError("Timed out waiting for the Junie ACP subprocess.") from exc

    async def _aensure_proc(self) -> None:
        """Spawn + handshake the Junie subprocess on demand, reusing a live one."""
        if (
            self._conn is not None
            and self._proc is not None
            and getattr(self._proc, "returncode", 0) is None
            and self._initialized
        ):
            return

        await self._aclose_proc()

        stack = contextlib.AsyncExitStack()
        try:
            conn, proc = await stack.enter_async_context(
                _acp.spawn_agent_process(
                    self._handler,
                    self._acp_command,
                    *self._acp_args,
                    env=_build_subprocess_env(),
                    cwd=self._acp_cwd,
                    transport_kwargs={"limit": _STDIO_LIMIT_BYTES},
                )
            )
        except FileNotFoundError as exc:
            await stack.aclose()
            raise RuntimeError(
                f"Could not start Junie ACP command '{self._acp_command}'. "
                "Install the JetBrains Junie CLI or set HERMES_JUNIE_ACP_COMMAND/JUNIE_CLI_PATH."
            ) from exc

        self._exit_stack = stack
        self._conn = conn
        self._proc = proc
        self._stderr_tail = deque(maxlen=40)
        if getattr(proc, "stderr", None) is not None:
            asyncio.ensure_future(self._adrain_stderr(proc))
        # If the subprocess dies (crash / self-exit), close the connection so any
        # in-flight request is rejected promptly instead of blocking until the
        # turn's wall-clock timeout. The SDK rejects outstanding futures on
        # close(); a clean stdout EOF alone would not.
        asyncio.ensure_future(self._awatch_proc(proc, conn))
        await conn.initialize(
            protocol_version=_acp.PROTOCOL_VERSION,
            client_capabilities=_client_capabilities(),
            client_info=_client_info(),
        )
        self._initialized = True
        self.is_closed = False

    async def _adrain_stderr(self, proc: Any) -> None:
        stream = getattr(proc, "stderr", None)
        if stream is None:
            return
        try:
            async for raw in stream:
                line = raw.decode("utf-8", "replace").rstrip("\n") if isinstance(raw, (bytes, bytearray)) else str(raw).rstrip("\n")
                if line:
                    self._stderr_tail.append(line)
        except Exception:
            return

    async def _awatch_proc(self, proc: Any, conn: Any) -> None:
        try:
            await proc.wait()
        except Exception:
            return
        # Only tear down if this is still the active process (a respawn may have
        # already replaced it).
        if conn is self._conn:
            self._initialized = False
        with contextlib.suppress(Exception):
            await conn.close()

    async def _aclose_proc(self) -> None:
        self._initialized = False
        stack = self._exit_stack
        self._exit_stack = None
        self._conn = None
        self._proc = None
        if stack is not None:
            with contextlib.suppress(Exception):
                await stack.aclose()

    # ---- request path ------------------------------------------------------

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        **_: Any,
    ) -> Any:
        prompt_text = _format_messages_as_prompt(
            messages or [],
            model=model,
            tools=tools,
            tool_choice=tool_choice,
        )
        # Normalise timeout: run_agent.py may pass an httpx.Timeout object
        # (used natively by the OpenAI SDK) rather than a plain float.
        if timeout is None:
            _effective_timeout = _DEFAULT_TIMEOUT_SECONDS
        elif isinstance(timeout, (int, float)):
            _effective_timeout = float(timeout)
        else:
            _candidates = [
                getattr(timeout, attr, None)
                for attr in ("read", "write", "connect", "pool", "timeout")
            ]
            _numeric = [float(v) for v in _candidates if isinstance(v, (int, float))]
            # An httpx.Timeout with every component None means "no timeout".
            _effective_timeout = max(_numeric) if _numeric else _NO_TIMEOUT_FALLBACK_SECONDS

        response_text, reasoning_text, tool_events, usage_obj = self._run_prompt(
            prompt_text,
            timeout_seconds=_effective_timeout,
            model=model,
        )

        # Junie executes its own tools and reports them as completed activity;
        # they are NOT delegation requests, so we never surface them as OpenAI
        # tool_calls (that would make Hermes try to re-run finished work).
        # Instead we log them and fold a readable summary into `reasoning`.
        activity = _render_tool_activity(tool_events)
        if tool_events:
            for ev in tool_events.values():
                logger.info(
                    "Junie ACP tool activity: kind=%s status=%s title=%s",
                    ev.get("kind"), ev.get("status"), ev.get("title"),
                )
        combined_reasoning = "\n\n".join(
            p for p in (
                (f"Junie tool activity:\n{activity}" if activity else ""),
                reasoning_text or "",
            ) if p
        ).strip() or None

        usage = self._build_usage(prompt_text, response_text, reasoning_text, usage_obj)
        assistant_message = SimpleNamespace(
            content=response_text.strip(),
            tool_calls=[],
            reasoning=combined_reasoning,
            reasoning_content=combined_reasoning,
            reasoning_details=None,
        )
        choice = SimpleNamespace(message=assistant_message, finish_reason="stop")
        return SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=model or "junie-acp",
        )

    @staticmethod
    def _build_usage(
        prompt_text: str, response_text: str, reasoning_text: str, usage_obj: Any
    ) -> SimpleNamespace:
        """Prefer Junie's own token counts (PromptResponse.usage) and fall back
        to a ~4-chars/token estimate. ACP historically reported no counts, which
        froze the context gauge at 0% and starved the compressor; either source
        keeps both working."""
        prompt_tokens = completion_tokens = cached = 0
        if usage_obj is not None:
            prompt_tokens = int(getattr(usage_obj, "input_tokens", 0) or 0)
            completion_tokens = int(getattr(usage_obj, "output_tokens", 0) or 0)
            completion_tokens += int(getattr(usage_obj, "thought_tokens", 0) or 0)
            cached = int(getattr(usage_obj, "cached_read_tokens", 0) or 0)
        if prompt_tokens <= 0 and completion_tokens <= 0:
            prompt_tokens = _rough_tokens(prompt_text)
            completion_tokens = _rough_tokens(response_text) + _rough_tokens(reasoning_text)
        return SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
        )

    def _run_prompt(
        self, prompt_text: str, *, timeout_seconds: float, model: str | None = None
    ) -> tuple[str, str, dict[str, dict[str, Any]], Any]:
        # Serialize requests: one shared connection + process.
        with self._req_lock:
            self._ensure_loop()
            last_exc: BaseException | None = None
            for attempt in (1, 2):
                self._prompt_in_flight = False
                try:
                    self._submit(self._aensure_proc(), timeout=_HANDSHAKE_TIMEOUT_SECONDS)
                    # Give the turn a bit more wall-clock than the prompt itself
                    # so the post-prompt settle can complete before we bail.
                    turn_timeout = timeout_seconds + self._settle_max + 5.0
                    return self._submit(
                        self._ado_turn(prompt_text, timeout_seconds, model=model),
                        timeout=turn_timeout,
                    )
                except (TimeoutError, RuntimeError, OSError, ConnectionError) as exc:
                    last_exc = exc
                    self._safe_close_proc()
                    # Junie is autonomous: once session/prompt is dispatched it may
                    # already have edited files / run commands. Never replay a
                    # dispatched prompt — that would double-apply side effects.
                    # Only retry failures that happened BEFORE the prompt was sent
                    # (e.g. a stale reused subprocess died on session/new).
                    if self._prompt_in_flight:
                        logger.warning("Junie ACP turn failed after prompt dispatch; not retrying: %s", exc)
                        raise
                    logger.warning(
                        "Junie ACP turn failed before prompt (attempt %d/2), respawning: %s", attempt, exc
                    )
                except Exception as exc:
                    # RequestError (SDK) and any other agent-side failure.
                    last_exc = exc
                    self._safe_close_proc()
                    if self._prompt_in_flight:
                        logger.warning("Junie ACP turn failed after prompt dispatch; not retrying: %s", exc)
                        raise RuntimeError(f"Junie ACP prompt failed: {exc}{self._stderr_suffix()}") from exc
                    logger.warning(
                        "Junie ACP turn failed before prompt (attempt %d/2), respawning: %s", attempt, exc
                    )
            assert last_exc is not None
            # Exhausted the pre-prompt retries — surface Junie's stderr (auth
            # rejection, missing runtime, …) so the failure is actionable.
            suffix = self._stderr_suffix()
            if suffix:
                raise RuntimeError(f"Junie ACP failed to start: {last_exc}{suffix}") from last_exc
            raise last_exc

    def _safe_close_proc(self) -> None:
        with contextlib.suppress(Exception):
            self._submit(self._aclose_proc(), timeout=5.0)

    async def _ado_turn(
        self, prompt_text: str, timeout_seconds: float, model: str | None = None
    ) -> tuple[str, str, dict[str, dict[str, Any]], Any]:
        # Cap session/new so a wedged-but-alive reused process fails fast here
        # (pre-prompt → safe to respawn+retry) instead of hanging the whole turn.
        session = await asyncio.wait_for(
            self._conn.new_session(cwd=self._acp_cwd, mcp_servers=[]),
            min(_SESSION_NEW_TIMEOUT_SECONDS, timeout_seconds),
        )
        session_id = str(getattr(session, "session_id", "") or "").strip()
        if not session_id:
            raise RuntimeError("Junie ACP did not return a sessionId.")

        # Forward the requested model to Junie via its ACP config. Skip the
        # "junie-acp" provider sentinel / aliases (those mean "use Junie's own
        # default"); only real Junie model ids (gemini-3-flash-preview,
        # claude-opus-4-8, gpt-5.x, …) are set. Best-effort: an unknown id or a
        # set failure must not abort the prompt.
        requested = (model or "").strip()
        if requested and requested.lower() not in _MODEL_PASSTHROUGH_SENTINELS:
            try:
                await asyncio.wait_for(
                    self._conn.set_config_option(config_id="model", session_id=session_id, value=requested),
                    _CONFIG_OPTION_TIMEOUT_SECONDS,
                )
                logger.info("Junie ACP model set to %s", requested)
            except Exception as exc:
                logger.warning("Junie ACP set model %s failed: %s", requested, exc)

        # Optionally force Junie's Brave Mode. Junie accepts a boolean
        # (true -> ON, false -> OFF) for backward compatibility; best-effort.
        if self._brave_override is not None:
            try:
                await asyncio.wait_for(
                    self._conn.set_config_option(
                        config_id="brave_mode", session_id=session_id, value=bool(self._brave_override)
                    ),
                    _CONFIG_OPTION_TIMEOUT_SECONDS,
                )
                logger.info("Junie ACP brave_mode set to %s", self._brave_override)
            except Exception as exc:
                logger.warning("Junie ACP set brave_mode failed: %s", exc)

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_events: dict[str, dict[str, Any]] = {}
        self._handler.begin_turn(session_id, text_parts, reasoning_parts, tool_events)
        # Mark dispatch so _run_prompt won't replay a prompt Junie may have
        # already acted on (see the retry guard).
        self._prompt_in_flight = True
        usage_obj: Any = None
        try:
            result = await self._conn.prompt(
                prompt=[_acp.text_block(prompt_text)], session_id=session_id
            )
            usage_obj = getattr(result, "usage", None)
            # The ACP completion response can precede a trailing message chunk;
            # settle until Junie is quiet so we don't truncate the answer.
            await self._handler.settle(self._settle_quiet_gap, min(self._settle_max, timeout_seconds))
        finally:
            self._handler.end_turn()
        # A fresh session is opened per request (we re-send the full transcript),
        # so we don't rely on Junie's cross-turn memory; sessions are left for
        # the persistent process to reap, matching the original client.
        return "".join(text_parts), "".join(reasoning_parts), tool_events, usage_obj
