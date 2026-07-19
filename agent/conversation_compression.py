"""Context compression — extract the AIAgent methods that drive summarisation.

Three concerns live here:

* :func:`check_compression_model_feasibility` — startup probe of the
  configured auxiliary compression model.  Warns when the aux context
  window can't fit the main model's compression threshold; auto-lowers
  the session threshold when possible; hard-rejects auxes below
  ``MINIMUM_CONTEXT_LENGTH``.

* :func:`replay_compression_warning` — re-emit a stored warning through
  the gateway ``status_callback`` once it's wired up (the callback is
  set after :class:`AIAgent` construction).

* :func:`compress_context` — the actual compression call.  Runs the
  configured compressor, splits the SQLite session, rotates the
  session_id, notifies plugin context engines / memory providers, and
  returns the compressed message list and freshly-built system prompt.

* :func:`try_shrink_image_parts_in_messages` — image-too-large recovery
  helper that re-encodes ``data:image/...;base64,...`` parts at a smaller
  size so retries can fit under provider ceilings (Anthropic's 5 MB).

``run_agent`` keeps thin wrappers for each so existing call sites
(``self._compress_context(...)``) keep working.  Tests that exercise
these paths see no behavioural change.
"""

from __future__ import annotations

import inspect
import logging
import os
import re
import tempfile
import uuid
import threading
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Any, Optional, Tuple
from urllib.parse import unquote_plus, urlsplit, urlunsplit

from agent.model_metadata import estimate_request_tokens_rough
from agent.redact import redact_sensitive_text
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# Stable marker the gateway matches on to re-tag the auto-compaction lifecycle
# status as ``kind="compacting"`` (tui_gateway/server.py::_status_update), so
# drivers like the desktop app can show an explicit "Summarizing…" indicator
# instead of the transcript appearing to silently reset. Keep the marker phrase
# intact if you reword COMPACTION_STATUS.
COMPACTION_STATUS_MARKER = "Compacting context"
COMPACTION_STATUS = (
    f"🗜️ {COMPACTION_STATUS_MARKER} — summarizing earlier conversation so I can continue..."
)

_HANDOFF_MAX_BYTES = 32 * 1024
_HANDOFF_MAX_MESSAGES = 24
_HANDOFF_MAX_TODOS = 24
_HANDOFF_FIELD_INPUT_CHARS = 8192
_HANDOFF_MESSAGE_CHARS = 1200
_HANDOFF_TODO_CHARS = 320
_HANDOFF_ROLE_CHARS = 32
_HANDOFF_PATH_CHARS = 512

_ARTIFACT_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ARTIFACT_URL_RE = re.compile(r"\b(?:https?|wss?|ftp)://[^\s<>()\[\]{}]+", re.IGNORECASE)
_ARTIFACT_SENSITIVE_QUERY_KEYS = frozenset(
    {
        "access_token",
        "refresh_token",
        "id_token",
        "token",
        "api_key",
        "apikey",
        "client_secret",
        "password",
        "passwd",
        "auth",
        "authorization",
        "jwt",
        "session",
        "cookie",
        "secret",
        "key",
        "code",
        "signature",
        "x-amz-signature",
    }
)
_ARTIFACT_NAMED_SECRET_RE = re.compile(
    r"(?i)(\b(?:api[_ .-]?key|access[_ .-]?token|refresh[_ .-]?token|"
    r"client[_ .-]?secret|password|passwd|credential|authorization|auth[_ .-]?token|"
    r"cookie|session[_ .-]?token|private[_ .-]?key|secret)\b\s*[:=]\s*)"
    r"(?:['\"]?)[^\s,;&\"']+"
)
_ARTIFACT_HEADER_SECRET_RE = re.compile(
    r"(?im)^((?:proxy-)?authorization|cookie|set-cookie|x-api-key|api-key)\s*:\s*.*$"
)
_ARTIFACT_PREFIX_SECRET_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?:sk-|ghp_|github_pat_|gho_|ghu_|ghs_|ghr_|"
    r"xox[baprs]-|AIza|pplx-|fal_|gAAAA|AKIA|sk_live_|sk_test_|rk_live_|"
    r"SG\.|hf_|npm_|pypi-|dop_v1_|doo_v1_|gsk_|xai-|ntn_|fpk_)"
    r"[A-Za-z0-9_.=/-]{8,}",
    re.IGNORECASE,
)
_ARTIFACT_JWT_RE = re.compile(
    r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}(?:\.[A-Za-z0-9_=-]{4,}){0,2}"
)


def _bounded_value_repr(
    value: Any,
    *,
    max_chars: int,
    depth: int = 0,
    seen: Optional[set[int]] = None,
) -> str:
    """Represent untrusted packet input with bounded work and cycle depth."""
    if max_chars <= 0:
        return ""
    if isinstance(value, str):
        return value[:max_chars]
    if isinstance(value, bytes):
        return value[:max_chars].decode("utf-8", errors="replace")
    if value is None or isinstance(value, (bool, int, float)):
        return str(value)[:max_chars]
    if depth >= 3:
        return f"<{type(value).__name__}>"
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return "<cycle>"
    seen.add(identity)
    try:
        if isinstance(value, dict):
            chunks = []
            for key, item in islice(value.items(), 8):
                key_text = _bounded_value_repr(
                    key, max_chars=64, depth=depth + 1, seen=seen
                )
                item_text = _bounded_value_repr(
                    item, max_chars=256, depth=depth + 1, seen=seen
                )
                chunks.append(f"{key_text}: {item_text}")
            rendered = "{" + ", ".join(chunks) + "}"
        elif isinstance(value, (list, tuple)):
            chunks = [
                _bounded_value_repr(
                    item, max_chars=256, depth=depth + 1, seen=seen
                )
                for item in islice(iter(value), 8)
            ]
            rendered = "[" + ", ".join(chunks) + "]"
        else:
            # Do not call an arbitrary object's potentially expensive __str__.
            rendered = f"<{type(value).__name__}>"
        return rendered[:max_chars]
    finally:
        seen.discard(identity)


def _sanitize_artifact_url(match: re.Match[str]) -> str:
    raw_url = match.group(0)
    trailing = ""
    while raw_url and raw_url[-1] in ".,;:!?\"'":
        trailing = raw_url[-1] + trailing
        raw_url = raw_url[:-1]
    try:
        parts = urlsplit(raw_url)
        netloc = parts.netloc
        if "@" in netloc:
            netloc = "[REDACTED]@" + netloc.rsplit("@", 1)[1]
        query_parts = re.split(r"([&;])", parts.query)
        for index in range(0, len(query_parts), 2):
            field = query_parts[index]
            if not field:
                continue
            key, separator, _value = field.partition("=")
            if unquote_plus(key).strip().lower() in _ARTIFACT_SENSITIVE_QUERY_KEYS:
                query_parts[index] = key + (separator or "=") + "[REDACTED]"
        sanitized = urlunsplit(
            (parts.scheme, netloc, parts.path, "".join(query_parts), parts.fragment)
        )
        return sanitized + trailing
    except Exception:
        # A malformed URL that still contains userinfo is safer fully hidden.
        return "[REDACTED URL]" + trailing


def _sanitize_artifact_text(
    value: Any,
    *,
    max_chars: int,
    multiline: bool = True,
) -> str:
    """Force-redact and bound text crossing the handoff-artifact boundary."""
    input_limit = min(
        _HANDOFF_FIELD_INPUT_CHARS,
        max(1024, max_chars * 4),
    )
    text = _bounded_value_repr(value, max_chars=input_limit)
    text = _ARTIFACT_CONTROL_RE.sub(" ", text)
    if not multiline:
        text = re.sub(r"[\r\n\t]+", " ", text)
    text = _ARTIFACT_URL_RE.sub(_sanitize_artifact_url, text)
    text = _ARTIFACT_HEADER_SECRET_RE.sub(
        lambda match: f"{match.group(1)}: [REDACTED]", text
    )
    text = _ARTIFACT_NAMED_SECRET_RE.sub(
        lambda match: f"{match.group(1)}[REDACTED]", text
    )
    text = _ARTIFACT_PREFIX_SECRET_RE.sub("[REDACTED]", text)
    text = _ARTIFACT_JWT_RE.sub("[REDACTED]", text)
    text = redact_sensitive_text(text, force=True)
    if len(text) > max_chars:
        marker = "… [truncated]"
        text = text[: max(0, max_chars - len(marker))] + marker
    return text


def _bound_packet_bytes(packet: str) -> str:
    encoded = packet.encode("utf-8", errors="replace")
    if len(encoded) <= _HANDOFF_MAX_BYTES:
        return encoded.decode("utf-8")
    marker = "\n… [packet truncated]\n".encode()
    prefix = encoded[: _HANDOFF_MAX_BYTES - len(marker)]
    return (prefix.decode("utf-8", errors="ignore") + marker.decode()).rstrip() + "\n"


def _build_handoff_packet(
    agent: Any,
    messages: list,
    *,
    approx_tokens: Optional[int] = None,
    note: Optional[str] = None,
    active_session_id: Optional[str] = None,
    parent_session_id: Optional[str] = None,
    automatic: bool = False,
) -> str:
    """Build a bounded, force-redacted continuation packet without file I/O."""
    active_id = _sanitize_artifact_text(
        active_session_id or getattr(agent, "session_id", None) or "unbound",
        max_chars=160,
        multiline=False,
    )
    parent_id = _sanitize_artifact_text(
        parent_session_id or "none", max_chars=160, multiline=False
    )
    try:
        cwd_value = os.getcwd()
    except Exception:
        cwd_value = "unknown"
    cwd = _sanitize_artifact_text(
        cwd_value, max_chars=_HANDOFF_PATH_CHARS, multiline=False
    )
    model = _sanitize_artifact_text(
        getattr(agent, "model", "unknown"), max_chars=160, multiline=False
    )
    provider = _sanitize_artifact_text(
        getattr(agent, "provider", "unknown"), max_chars=120, multiline=False
    )
    compression_count = max(
        0,
        int(getattr(getattr(agent, "context_compressor", None), "compression_count", 0) or 0),
    )

    lines = [
        "# Hermes handoff packet",
        "",
        (
            "Automatic bounded continuation packet. Verify live repository and "
            "runtime state before editing."
            if automatic
            else "Manual packet; no new session started. Verify live state before editing."
        ),
        "",
        "## Coordinates",
        f"- Active session: `{active_id}`",
        f"- Parent session: `{parent_id}`",
        f"- Model/provider: `{model}` / `{provider}`",
        f"- Working directory: `{cwd}`",
        f"- Completed compressions before handoff: {compression_count}",
    ]
    if approx_tokens is not None:
        try:
            token_count = max(0, int(approx_tokens))
        except (TypeError, ValueError):
            token_count = 0
        lines.append(f"- Approximate request tokens: {token_count:,}")
    if note:
        lines.extend(
            [
                "",
                "## Operator note",
                _sanitize_artifact_text(
                    note, max_chars=1200, multiline=True
                ),
            ]
        )

    lines.extend(["", "## Recent bounded context"])
    bounded_messages = messages[-_HANDOFF_MAX_MESSAGES:] if isinstance(messages, list) else []
    if not bounded_messages:
        lines.append("- No conversation messages were available.")
    for message in bounded_messages:
        if not isinstance(message, dict):
            continue
        role = _sanitize_artifact_text(
            message.get("role", "unknown"),
            max_chars=_HANDOFF_ROLE_CHARS,
            multiline=False,
        )
        content = _sanitize_artifact_text(
            message.get("content", ""),
            max_chars=_HANDOFF_MESSAGE_CHARS,
            multiline=True,
        )
        lines.extend([f"### {role}", content or "(empty)"])

    lines.extend(["", "## Active todos"])
    todo_items: list = []
    try:
        raw_todos = agent._todo_store.read()
        if isinstance(raw_todos, list):
            todo_items = raw_todos[:_HANDOFF_MAX_TODOS]
    except Exception:
        todo_items = []
    active_todos = 0
    for item in todo_items:
        if not isinstance(item, dict):
            continue
        status = _sanitize_artifact_text(
            item.get("status", "pending"), max_chars=32, multiline=False
        )
        if status.lower() in {"completed", "cancelled", "canceled"}:
            continue
        todo_id = _sanitize_artifact_text(
            item.get("id", "todo"), max_chars=96, multiline=False
        )
        content = _sanitize_artifact_text(
            item.get("content", ""),
            max_chars=_HANDOFF_TODO_CHARS,
            multiline=False,
        )
        lines.append(f"- [{status}] {todo_id}: {content}")
        active_todos += 1
    if active_todos == 0:
        lines.append("- No active todos were available.")

    lines.extend(
        [
            "",
            "## Continuation rules",
            "- Re-read repository instructions and verify current git/session state.",
            "- Treat this packet as bounded context, not as authoritative live state.",
            "- Preserve tests, lineage, routing, and persistence invariants.",
            "",
        ]
    )
    return _bound_packet_bytes("\n".join(lines))


def _handoff_artifact_directory(agent: Any) -> Optional[Path]:
    home = get_hermes_home().resolve()
    configured = str(
        getattr(agent, "_auto_handoff_artifact_dir", ".hermes/handoffs")
        or ".hermes/handoffs"
    )
    candidate = Path(configured).expanduser()
    if not candidate.is_absolute():
        parts = candidate.parts
        if parts and parts[0] == ".hermes":
            candidate = Path(*parts[1:])
        candidate = home / candidate
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(home)
        relative_parts = resolved.relative_to(home).parts
        current = home
        for part in relative_parts:
            current = current / part
            if current.exists() and current.is_symlink():
                return None
        resolved.mkdir(parents=True, exist_ok=True, mode=0o700)
        resolved = resolved.resolve(strict=True)
        resolved.relative_to(home)
        return resolved
    except (OSError, RuntimeError, ValueError):
        return None


def _publish_handoff_artifact(agent: Any, packet: str) -> Path:
    """Publish *packet* atomically with 0600, no-clobber file semantics."""
    directory = _handoff_artifact_directory(agent)
    if directory is None:
        raise OSError("handoff artifact directory is outside HERMES_HOME or unsafe")
    data = _bound_packet_bytes(packet).encode("utf-8")
    directory_flags = os.O_RDONLY
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    dir_fd = os.open(directory, directory_flags)
    try:
        for _attempt in range(8):
            stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            final_name = f"handoff-{stamp}-{uuid.uuid4().hex}.md"
            temp_name = f".{final_name}.{uuid.uuid4().hex}.tmp"
            file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            file_flags |= getattr(os, "O_NOFOLLOW", 0)
            file_fd = os.open(temp_name, file_flags, 0o600, dir_fd=dir_fd)
            try:
                try:
                    os.fchmod(file_fd, 0o600)
                    written = 0
                    while written < len(data):
                        chunk_size = os.write(file_fd, data[written:])
                        if chunk_size <= 0:
                            raise OSError(
                                "handoff artifact write made no progress"
                            )
                        written += chunk_size
                    os.fsync(file_fd)
                finally:
                    os.close(file_fd)
            except BaseException:
                try:
                    os.unlink(temp_name, dir_fd=dir_fd)
                except FileNotFoundError:
                    pass
                raise
            try:
                os.link(
                    temp_name,
                    final_name,
                    src_dir_fd=dir_fd,
                    dst_dir_fd=dir_fd,
                    follow_symlinks=False,
                )
                os.fsync(dir_fd)
                return directory / final_name
            except FileExistsError:
                continue
            finally:
                try:
                    os.unlink(temp_name, dir_fd=dir_fd)
                except FileNotFoundError:
                    pass
        raise FileExistsError("could not allocate a unique handoff artifact name")
    finally:
        os.close(dir_fd)


def create_handoff_packet(
    agent: Any,
    messages: list,
    *,
    approx_tokens: Optional[int] = None,
    note: Optional[str] = None,
    active_session_id: Optional[str] = None,
    parent_session_id: Optional[str] = None,
) -> Tuple[str, Optional[Path]]:
    """Build and publish a manual packet without rotating or consuming quota."""
    packet = _build_handoff_packet(
        agent,
        messages,
        approx_tokens=approx_tokens,
        note=note,
        active_session_id=active_session_id,
        parent_session_id=parent_session_id,
        automatic=False,
    )
    try:
        return packet, _publish_handoff_artifact(agent, packet)
    except Exception as exc:
        logger.warning("Could not publish handoff packet: %s", exc)
        return packet, None


def _lock_api_is_absent_on_session_db(lock_db: Any) -> bool:
    """Whether the live in-memory SessionDB class structurally predates locks.

    In the supported hot-reload skew, this module is new while the already
    imported ``hermes_state.SessionDB`` class (and its live instances) is old.
    Only that exact class identity may fail open. Proxies, nominal lookalikes,
    non-callables, and descriptor failures must fail closed. Static lookup
    avoids invoking a present-but-broken descriptor.
    """
    try:
        from hermes_state import SessionDB

        missing = object()
        return (
            type(lock_db) is SessionDB
            and inspect.getattr_static(
                SessionDB, "try_acquire_compression_lock", missing
            ) is missing
        )
    except Exception:
        return False


def _refresh_persisted_compression_guards(compressor: Any) -> None:
    """Refresh durable automatic-compression guards on a built-in compressor."""
    method_calls = (
        ("get_active_compression_failure_cooldown", {"refresh": True}),
        ("_load_fallback_compression_streak", {}),
    )
    for method_name, kwargs in method_calls:
        method = getattr(type(compressor), method_name, None)
        if not callable(method):
            continue
        try:
            method(compressor, **kwargs)
        except Exception as exc:
            logger.debug("compression guard refresh failed (%s): %s", method_name, exc)


def _session_was_rotated_by_compression(session_db: Any, session_id: str) -> bool:
    """Return whether another path already rotated this compression parent."""
    getter = getattr(type(session_db), "get_session", None)
    if not callable(getter):
        return False
    session = getter(session_db, session_id)
    return bool(
        session
        and session.get("ended_at") is not None
        and session.get("end_reason") == "compression"
    )


def _compression_lock_holder(agent: Any) -> str:
    """Build a unique holder id for the lock: pid:tid:agent-instance:uuid.

    The pid+tid prefix lets ops tell crashed/abandoned holders apart from
    live ones (expiry-based recovery uses the timestamp, but ``holder``
    is what shows up in diagnostics + log lines). The agent instance id
    and a per-acquire uuid disambiguate two co-resident agents on the
    same thread (background_review forks run on a worker thread, but
    on machines where compression itself dispatches to a thread pool
    we want each acquire to be unique).
    """
    import threading
    return (
        f"pid={os.getpid()}"
        f":tid={threading.get_ident()}"
        f":agent={id(agent):x}"
        f":nonce={uuid.uuid4().hex[:8]}"
    )


class _CompressionLockLeaseRefresher:
    def __init__(
        self,
        db: Any,
        session_id: str,
        holder: str,
        ttl_seconds: float,
        refresh_interval_seconds: float | None = None,
    ) -> None:
        self._db = db
        self._session_id = session_id
        self._holder = holder
        self._ttl_seconds = ttl_seconds
        if refresh_interval_seconds is None:
            refresh_interval_seconds = max(1.0, min(60.0, ttl_seconds / 2.0))
        self._refresh_interval_seconds = max(0.1, float(refresh_interval_seconds))
        # Tolerate transient refresh failures for at most one lease's worth of
        # time, so the give-up window is genuinely bounded by the TTL the
        # acquirer set (a single blip recovers on the next tick; a persistent
        # failure stops before the lease could outlive its TTL). Floor of 1 so a
        # degenerate interval >= ttl still tolerates one blip.
        self._max_consecutive_failures = max(
            1, int(self._ttl_seconds / self._refresh_interval_seconds)
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="compression-lock-refresh",
            daemon=True,
        )

    def start(self) -> "_CompressionLockLeaseRefresher":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        # join() may time out while the refresher is mid-UPDATE; that's safe —
        # it's a daemon thread, and a late refresh on an already-released lock
        # matches rowcount 0 (a no-op). stop() returning does not guarantee the
        # thread has fully quiesced, only that we've signalled it and waited
        # briefly.
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        # A single falsy refresh must NOT permanently kill the lease: a
        # transient DB blip (write contention escaping _execute_write's retry
        # budget, a momentary "database is locked") returns False just like a
        # genuine lost-ownership, but only the latter should stop the loop.
        # Tolerate consecutive failures for at most one lease's worth of time
        # (_max_consecutive_failures = ttl / interval), so a one-off blip
        # recovers on the next tick while the total give-up window stays bounded
        # by the TTL the acquirer set — the lock can never be held past its TTL
        # by a stuck refresher.
        consecutive_failures = 0
        while not self._stop.wait(self._refresh_interval_seconds):
            try:
                refreshed = self._db.refresh_compression_lock(
                    self._session_id,
                    self._holder,
                    ttl_seconds=self._ttl_seconds,
                )
            except Exception as exc:
                logger.debug("compression lock refresh raised: %s", exc)
                refreshed = False
            if refreshed:
                consecutive_failures = 0
                continue
            consecutive_failures += 1
            if consecutive_failures >= self._max_consecutive_failures:
                logger.debug(
                    "compression lock refresh failed %d times in a row; "
                    "stopping lease refresher for session %s",
                    consecutive_failures, self._session_id,
                )
                break


def check_compression_model_feasibility(agent: Any) -> None:
    """Warn at session start if the auxiliary compression model's context
    window is smaller than the main model's compression threshold.

    When the auxiliary model cannot fit the content that needs summarising,
    compression will either fail outright (the LLM call errors) or produce
    a severely truncated summary.

    Called during ``AIAgent.__init__`` so CLI users see the warning
    immediately (via ``_vprint``).  The gateway sets ``status_callback``
    *after* construction, so :func:`replay_compression_warning` re-sends
    the stored warning through the callback on the first
    ``run_conversation()`` call.
    """
    if not agent.compression_enabled:
        return
    try:
        from agent.auxiliary_client import (
            _resolve_task_provider_model,
            _try_configured_fallback_for_unavailable_client,
            get_text_auxiliary_client,
        )
        from agent.model_metadata import (
            MINIMUM_CONTEXT_LENGTH,
            get_model_context_length,
        )

        # Best-effort aux provider label for the warning message. The
        # configured provider may be "auto", in which case we fall back
        # to the client's base_url hostname so the user can still tell
        # where the compression model is actually being called.
        try:
            _aux_cfg_provider, _, _, _, _ = _resolve_task_provider_model("compression")
        except Exception:
            _aux_cfg_provider = ""
        client, aux_model = get_text_auxiliary_client(
            "compression",
            main_runtime=agent._current_main_runtime(),
        )
        if client is None or not aux_model:
            fb_client, fb_model, fb_label = _try_configured_fallback_for_unavailable_client(
                "compression",
                _aux_cfg_provider,
            )
            if fb_client is not None and fb_model:
                client, aux_model = fb_client, fb_model
                if "(" in fb_label and fb_label.endswith(")"):
                    _aux_cfg_provider = fb_label.rsplit("(", 1)[1][:-1]
        if client is None or not aux_model:
            if _aux_cfg_provider and _aux_cfg_provider != "auto":
                msg = (
                    "⚠ Configured auxiliary compression provider "
                    f"'{_aux_cfg_provider}' is unavailable — context "
                    "compression will drop middle turns without a summary. "
                    "Check auxiliary.compression in config.yaml and "
                    "reauthenticate that provider."
                )
            else:
                msg = (
                    "⚠ No auxiliary LLM provider configured — context "
                    "compression will drop middle turns without a summary. "
                    "Run `hermes setup` or set OPENROUTER_API_KEY."
                )
            agent._compression_warning = msg
            agent._emit_status(msg)
            logger.warning(
                "No auxiliary LLM provider for compression — "
                "summaries will be unavailable."
            )
            return

        aux_base_url = str(getattr(client, "base_url", ""))
        # ``client.api_key`` may be a callable (Azure Foundry Entra ID
        # bearer provider). The context-length resolver chain expects a
        # string, but it only needs a key for live catalogue probes
        # (provider model lists). For Entra clients the model-metadata
        # chain still resolves via models.dev + hardcoded family
        # fallbacks, which don't require auth — pass empty string rather
        # than minting a bearer JWT just to look up a context length.
        _raw_aux_key = getattr(client, "api_key", "")
        aux_api_key = "" if (callable(_raw_aux_key) and not isinstance(_raw_aux_key, str)) else str(_raw_aux_key or "")

        aux_context = get_model_context_length(
            aux_model,
            base_url=aux_base_url,
            api_key=aux_api_key,
            config_context_length=getattr(agent, "_aux_compression_context_length_config", None),
            # Each model must be resolved with its own provider so that
            # provider-specific paths (e.g. Bedrock static table, OpenRouter API)
            # are invoked for the correct client, not inherited from the main model.
            provider=(_aux_cfg_provider if _aux_cfg_provider and _aux_cfg_provider != "auto" else getattr(agent, "provider", "")),
            custom_providers=agent._custom_providers,
        )

        # Hard floor: the auxiliary compression model must have at least
        # MINIMUM_CONTEXT_LENGTH (64K) tokens of context.  The main model
        # is already required to meet this floor (checked earlier in
        # __init__), so the compression model must too — otherwise it
        # cannot summarise a full threshold-sized window of main-model
        # content.  Mirrors the main-model rejection pattern.
        if aux_context and aux_context < MINIMUM_CONTEXT_LENGTH:
            raise ValueError(
                f"Auxiliary compression model {aux_model} has a context "
                f"window of {aux_context:,} tokens, which is below the "
                f"minimum {MINIMUM_CONTEXT_LENGTH:,} required by Hermes "
                f"Agent.  Choose a compression model with at least "
                f"{MINIMUM_CONTEXT_LENGTH // 1000}K context (set "
                f"auxiliary.compression.model in config.yaml), or set "
                f"auxiliary.compression.context_length to override the "
                f"detected value if it is wrong."
            )

        threshold = agent.context_compressor.threshold_tokens
        if aux_context < threshold:
            # Auto-correct: lower the live session threshold so
            # compression actually works this session.  The hard floor
            # above guarantees aux_context >= MINIMUM_CONTEXT_LENGTH,
            # so the new threshold is always >= 64K.
            #
            # The compression summariser sends a single user-role
            # prompt (no system prompt, no tools) to the aux model, so
            # new_threshold == aux_context is safe: the request is
            # the raw messages plus a small summarisation instruction.
            old_threshold = threshold
            new_threshold = aux_context
            agent.context_compressor.threshold_tokens = new_threshold
            # Keep threshold_percent in sync so future main-model
            # context_length changes (update_model) re-derive from a
            # sensible number rather than the original too-high value.
            main_ctx = agent.context_compressor.context_length
            if main_ctx:
                agent.context_compressor.threshold_percent = (
                    new_threshold / main_ctx
                )
            safe_pct = int((aux_context / main_ctx) * 100) if main_ctx else 50
            # Build human-readable "model (provider)" labels for both
            # the main model and the compression model so users can
            # tell at a glance which provider each side is actually
            # using. When the configured provider is empty or "auto",
            # fall back to the client's base_url hostname.
            _main_model = getattr(agent, "model", "") or "?"
            _main_provider = getattr(agent, "provider", "") or ""
            _aux_provider_label = (
                _aux_cfg_provider
                if _aux_cfg_provider and _aux_cfg_provider != "auto"
                else ""
            )
            if not _aux_provider_label:
                try:
                    from urllib.parse import urlparse
                    _aux_provider_label = (
                        urlparse(aux_base_url).hostname or aux_base_url
                    )
                except Exception:
                    _aux_provider_label = aux_base_url or "auto"
            _main_label = (
                f"{_main_model} ({_main_provider})"
                if _main_provider
                else _main_model
            )
            _aux_label = f"{aux_model} ({_aux_provider_label})"
            msg = (
                f"⚠ Compression model {_aux_label} context is "
                f"{aux_context:,} tokens, but the main model "
                f"{_main_label}'s compression threshold was "
                f"{old_threshold:,} tokens. "
                f"Auto-lowered this session's threshold to "
                f"{new_threshold:,} tokens so compression can run.\n"
                f"  To make this permanent, edit config.yaml — either:\n"
                f"  1. Use a larger compression model:\n"
                f"       auxiliary:\n"
                f"         compression:\n"
                f"           model: <model-with-{old_threshold:,}+-context>\n"
                f"  2. Lower the compression threshold:\n"
                f"       compression:\n"
                f"         threshold: 0.{safe_pct:02d}"
            )
            agent._compression_warning = msg
            agent._emit_status(msg)
            logger.warning(
                "Auxiliary compression model %s has %d token context, "
                "below the main model's compression threshold of %d "
                "tokens — auto-lowered session threshold to %d to "
                "keep compression working.",
                aux_model,
                aux_context,
                old_threshold,
                new_threshold,
            )
    except ValueError:
        # Hard rejections (aux below minimum context) must propagate
        # so the session refuses to start.
        raise
    except Exception as exc:
        logger.debug(
            "Compression feasibility check failed (non-fatal): %s", exc
        )


def replay_compression_warning(agent: Any) -> None:
    """Re-send the compression warning through ``status_callback``.

    During ``__init__`` the gateway's ``status_callback`` is not yet
    wired, so ``_emit_status`` only reaches ``_vprint`` (CLI).  This
    method is called once at the start of the first
    ``run_conversation()`` — by then the gateway has set the callback,
    so every platform (Telegram, Discord, Slack, etc.) receives the
    warning.
    """
    msg = getattr(agent, "_compression_warning", None)
    if msg and agent.status_callback:
        try:
            agent.status_callback("lifecycle", msg)
        except Exception:
            pass


def _new_compression_session_id() -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def _sync_rotated_session_context(agent: Any, session_id: str) -> None:
    """Synchronize tool/gateway and logging coordinates after DB commit."""
    try:
        from gateway.session_context import set_current_session_id

        set_current_session_id(session_id)
    except Exception:
        os.environ["HERMES_SESSION_ID"] = session_id
    try:
        from hermes_logging import set_session_context

        set_session_context(session_id)
    except Exception:
        pass


def _mark_child_messages_persisted(agent: Any, messages: list) -> None:
    agent._last_flushed_db_idx = len(messages)
    agent._flushed_db_message_session_id = agent.session_id
    agent._flushed_db_message_ids = {
        id(message) for message in messages if isinstance(message, dict)
    }


def _authoritative_rotation_committed(
    session_db: Any,
    parent_session_id: str,
    child_session_id: str,
) -> bool:
    """Resolve ambiguous post-commit exceptions from authoritative rows."""
    try:
        parent = session_db.get_session(parent_session_id)
        child = session_db.get_session(child_session_id)
    except Exception:
        return False
    return bool(
        parent
        and child
        and parent.get("ended_at") is not None
        and parent.get("end_reason") == "compression"
        and child.get("ended_at") is None
        and child.get("parent_session_id") == parent_session_id
    )


def _rotate_compression_session(
    agent: Any,
    *,
    parent_session_id: str,
    child_session_id: str,
    child_messages: list,
    system_prompt: str,
    consume_auto_handoff: bool,
) -> Tuple[bool, int, Optional[Exception]]:
    """Invoke the atomic SessionDB API and classify ambiguous exceptions."""
    session_db = getattr(agent, "_session_db", None)
    if session_db is None:
        return False, 0, RuntimeError("SessionDB unavailable")
    child_config = dict(getattr(agent, "_session_init_model_config", {}) or {})
    try:
        count = session_db.rotate_session_for_compression(
            parent_session_id=parent_session_id,
            child_session_id=child_session_id,
            source=getattr(agent, "platform", None)
            or os.environ.get("HERMES_SESSION_SOURCE", "cli"),
            model=getattr(agent, "model", None),
            model_config=child_config,
            system_prompt=system_prompt,
            messages=child_messages,
            consume_auto_handoff=consume_auto_handoff,
            max_auto_handoffs=getattr(
                agent, "_auto_handoff_max_auto_handoffs", 0
            ),
        )
        return True, max(0, int(count)), None
    except Exception as exc:
        # _execute_write commits before best-effort post-write maintenance. A
        # wrapper/observer can therefore raise after the transaction committed.
        # Never "compensate" a committed child by reopening its parent: inspect
        # canonical rows and adopt the child when the transition is complete.
        if _authoritative_rotation_committed(
            session_db, parent_session_id, child_session_id
        ):
            try:
                count = session_db.get_auto_handoff_count(child_session_id)
            except Exception:
                count = getattr(agent, "_auto_handoff_count", 0)
            logger.warning(
                "Compression rotation raised after its DB transition committed "
                "(%s); adopting authoritative child %s.",
                exc,
                child_session_id,
            )
            return True, max(0, int(count or 0)), exc
        return False, getattr(agent, "_auto_handoff_count", 0), exc


def _reset_agent_for_fresh_handoff(
    agent: Any,
    *,
    parent_session_id: str,
    child_session_id: str,
    system_prompt: str,
    child_messages: list,
    handoff_count: int,
) -> None:
    """Re-baseline all in-memory state owned by a fresh continuation child."""
    agent.session_id = child_session_id
    agent._parent_session_id = parent_session_id
    agent._session_db_created = True
    agent._cached_system_prompt = system_prompt
    agent._compression_feasibility_checked = False
    agent._auto_handoff_count = handoff_count
    child_config = dict(getattr(agent, "_session_init_model_config", {}) or {})
    child_config["_auto_handoff_count"] = handoff_count
    agent._session_init_model_config = child_config
    _mark_child_messages_persisted(agent, child_messages)
    _sync_rotated_session_context(agent, child_session_id)

    reset = getattr(
        getattr(agent, "context_compressor", None), "on_session_reset", None
    )
    if callable(reset):
        try:
            reset()
        except Exception as exc:
            logger.debug("context engine fresh-handoff reset failed: %s", exc)

    # Reset child-owned usage/verdict state without touching the in-flight
    # parent registry slot; note_turn_end intentionally clears that old slot.
    for attr in (
        "session_prompt_tokens",
        "session_completion_tokens",
        "session_total_tokens",
        "session_api_calls",
        "session_input_tokens",
        "session_output_tokens",
        "session_cache_read_tokens",
        "session_cache_write_tokens",
        "session_reasoning_tokens",
        "_user_turn_count",
        "_turns_since_memory",
        "_iters_since_skill",
    ):
        setattr(agent, attr, 0)
    agent.session_estimated_cost_usd = 0.0
    agent.session_cost_status = "unknown"
    agent.session_cost_source = "none"
    agent._session_messages = []
    agent._last_aux_fallback_warning_key = None
    agent._compression_warning = None
    agent._gateway_turn_context_notes = ""
    agent._current_api_request_id = ""
    task_id = getattr(agent, "_current_task_id", None) or "default"
    agent._current_turn_id = f"{child_session_id}:{task_id}:handoff"


def _remove_failed_auto_artifact(path: Optional[Path]) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except OSError:
        logger.debug("Could not remove unused handoff artifact %s", path)


def conversation_history_after_compression(agent: Any, messages: list) -> Optional[list]:
    """Return the correct flush baseline after a compression boundary.

    Legacy compression rotates to a fresh child session. That child has not
    seen the compacted transcript through the normal same-turn flush path yet,
    so callers must clear ``conversation_history`` to ``None`` and let the next
    persistence call write the whole compacted list.

    In-place compaction is different: ``archive_and_compact()`` has already
    soft-archived the previous active rows and inserted ``messages`` as the new
    active live transcript under the same session id. If the same agent turn
    continues with ``conversation_history=None``, the identity-based flush path
    treats those already-persisted compacted dicts as new and appends them a
    second time, doubling the active context and retriggering compression.

    A shallow copy is intentional: it captures the current compacted dict
    identities as history while allowing later same-turn appends to remain new.
    """
    if bool(getattr(agent, "_last_compaction_in_place", False)):
        return list(messages)
    return None


def persist_rejoined_partial_compression(agent: Any, messages: list) -> bool:
    """Persist the full transcript after a manual head-only compression.

    ``compress_context`` sees only the head selected by ``/compress here`` and
    therefore cannot include the caller-owned verbatim tail in its boundary
    write. Once the caller rejoins that tail, replace only the active rows. This
    preserves the pre-compaction archive created by in-place compaction while
    making both in-place and legacy-child sessions immediately resumable with
    the exact rejoined transcript.
    """
    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    if session_db is None or not session_id:
        return False
    session_db.replace_messages(session_id, messages, active_only=True)
    _mark_child_messages_persisted(agent, messages)
    return True


_SYNTHETIC_USER_PREFIXES = (
    "[System: Your previous response was truncated",
    "[System: The previous response was cut off",
    "[System: Your previous tool call",
    "[Your active task list was preserved across context compression]",
    "[IMPORTANT: Background process ",
)


def _message_text(message: Any) -> str:
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text") or part.get("content") or "")
            for part in content
            if isinstance(part, dict)
        )
    return ""


_SYNTHETIC_USER_FLAGS = (
    "_todo_snapshot_synthetic",
    "_empty_recovery_synthetic",
    "_verification_stop_synthetic",
    "_pre_verify_synthetic",
)


def _is_real_user_message(message: Any) -> bool:
    """Distinguish human intent from user-role runtime scaffolding.

    A compaction summary pinned to ``role="user"`` (the compressor flips the
    summary role to preserve alternation when the tail starts with an
    assistant message) is scaffolding too: treating it as human intent would
    short-circuit anchor restoration with a message the model is explicitly
    told NOT to act on.
    """
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    if any(message.get(flag) for flag in _SYNTHETIC_USER_FLAGS):
        return False
    text = _message_text(message).strip()
    if not text:
        return False
    if text.startswith(_SYNTHETIC_USER_PREFIXES):
        return False
    from agent.context_compressor import ContextCompressor

    return not ContextCompressor._is_context_summary_content(text)


def _merge_anchor_into_user_message(target: dict, anchor: dict) -> None:
    """Fold the human anchor into an existing user-role scaffolding turn.

    Used only when every insertion slot would create two consecutive
    user-role messages. The anchor text leads (it is the active task), the
    scaffolding content is preserved after it, and the synthetic flags are
    cleared because the merged turn now carries real human intent.
    """
    anchor_content = anchor.get("content")
    target_content = target.get("content")
    if isinstance(anchor_content, list) or isinstance(target_content, list):
        anchor_parts = (
            list(anchor_content)
            if isinstance(anchor_content, list)
            else [{"type": "text", "text": str(anchor_content or "")}]
        )
        target_parts = (
            list(target_content)
            if isinstance(target_content, list)
            else [{"type": "text", "text": str(target_content or "")}]
        )
        target["content"] = anchor_parts + target_parts
    else:
        merged = f"{anchor_content or ''}\n\n{target_content or ''}".strip()
        target["content"] = merged
    for flag in _SYNTHETIC_USER_FLAGS:
        target.pop(flag, None)


def _insert_real_user_anchor(messages: list, anchor: dict) -> None:
    """Insert the latest human turn without breaking role alternation."""

    def _role(msg: Any) -> Optional[str]:
        return msg.get("role") if isinstance(msg, dict) else None

    # Preferred: the summary boundary — before the first assistant message
    # not already preceded by a user turn. The left neighbour is then
    # non-user by construction and the right neighbour is an assistant.
    for index, message in enumerate(messages):
        if _role(message) != "assistant":
            continue
        previous_role = _role(messages[index - 1]) if index > 0 else None
        if previous_role != "user":
            messages.insert(index, anchor)
            return
    # Every assistant is user-preceded (or there are none). Appending is
    # safe whenever the transcript does not already end with a user turn.
    if not messages or _role(messages[-1]) != "user":
        messages.append(anchor)
        return
    # The transcript ends with a user-role message and no slot avoids
    # user/user adjacency.
    from agent.context_compressor import ContextCompressor

    if ContextCompressor._is_context_summary_content(
        _message_text(messages[-1])
    ):
        # Never merge into a compaction summary: the summary prefix must
        # stay at the start of its message for downstream summary detection.
        # Appending after it makes the anchor "the latest user message after
        # the summary" — exactly what the handoff prefix instructs — and the
        # adjacent user turns are merged summary-first by
        # repair_message_sequence before the next API call.
        messages.append(anchor)
        return
    # Trailing user-role scaffolding (e.g. the todo snapshot): merge instead
    # of inserting a consecutive same-role message (#55677 strict templates).
    _merge_anchor_into_user_message(messages[-1], anchor)


def _ensure_compressed_has_user_turn(original_messages: list, compressed: list) -> None:
    """Preserve human intent, not merely a synthetic user-role placeholder."""
    if any(_is_real_user_message(message) for message in compressed):
        return
    from agent.context_compressor import _fresh_compaction_message_copy

    for message in reversed(original_messages):
        if _is_real_user_message(message):
            _insert_real_user_anchor(
                compressed,
                _fresh_compaction_message_copy(message),
            )
            return
    compressed.append({
        "role": "user",
        "content": (
            "Continue from the compressed conversation context above. "
            "This marker exists because no human user turn was available."
        ),
    })


def compress_context(
    agent: Any,
    messages: list,
    system_message: str,
    *,
    approx_tokens: Optional[int] = None,
    task_id: str = "default",
    focus_topic: Optional[str] = None,
    force: bool = False,
) -> Tuple[list, str]:
    """Compress conversation context and split the session in SQLite.

    Args:
        agent: The owning :class:`AIAgent`.
        messages: Current message history (will be summarised).
        system_message: Current system prompt; rebuilt after compression.
        approx_tokens: Pre-compression token estimate, logged for ops.
        task_id: Tool task scope (used for clearing file-read dedup state).
        focus_topic: Optional focus string for guided compression — the
            summariser will prioritise preserving information related to
            this topic.  Inspired by Claude Code's ``/compact <focus>``.
        force: If True, bypass any active summary-failure cooldown.  Set
            by the manual ``/compress`` slash command so users can retry
            immediately after an auto-compress abort.  Auto-compress
            callers use the default ``False``.

    Returns:
        ``(compressed_messages, new_system_prompt)`` tuple.  When
        compression aborts (aux LLM failed to produce a usable summary),
        returns the original messages unchanged and the existing system
        prompt — the session is NOT rotated.  Callers should detect the
        no-op via ``len(returned) == len(input)`` and stop the retry loop.
    """
    # Codex app-server sessions: the codex agent owns the real thread context;
    # Hermes' summarizer would only rewrite a local mirror without shrinking
    # the actual thread (#36801). Route compaction to the app server's own
    # thread/compact mechanism. Behavior is controlled by
    # ``compression.codex_app_server_auto`` (native|hermes|off).
    if getattr(agent, "api_mode", None) == "codex_app_server":
        return _compress_context_via_codex_app_server(
            agent,
            messages,
            system_message,
            approx_tokens=approx_tokens,
            task_id=task_id,
            force=force,
        )

    # Every automatic entrypoint must honor compressor-owned cooldown and
    # breaker state. Gateway hygiene constructs a fresh AIAgent, so the
    # persisted fallback streak is loaded by bind_session_state() before this.
    if not force:
        _refresh_persisted_compression_guards(agent.context_compressor)
        blocked = getattr(
            type(agent.context_compressor),
            "_automatic_compression_blocked",
            None,
        )
        if callable(blocked) and blocked(agent.context_compressor):
            existing_prompt = getattr(agent, "_cached_system_prompt", None)
            if not existing_prompt:
                existing_prompt = agent._build_system_prompt(system_message)
            return messages, existing_prompt

    # Lazy feasibility check — run the auxiliary-provider probe + context
    # length lookup just-in-time on the first compression attempt instead of
    # at AIAgent.__init__. Saves ~400ms cold off every short session that
    # never reaches the threshold (the vast majority of ``chat -q`` runs).
    # The check itself sets ``agent._compression_warning`` so the
    # status-callback replay machinery still emits the warning to the user
    # the first time it would matter.
    if not getattr(agent, "_compression_feasibility_checked", False):
        # Mark as checked only after the probe completes. If the check
        # raises (e.g. a fatal aux-context ValueError that aborts the
        # session), leaving the flag unset is harmless; a non-fatal
        # transient failure is swallowed inside the function so the flag
        # is set normally on the next successful pass.
        check_compression_model_feasibility(agent)
        agent._compression_feasibility_checked = True

    _pre_msg_count = len(messages)
    # In-place compaction (config: compression.in_place, see #38763). When True,
    # this compaction rewrites the message list + rebuilds the system prompt but
    # keeps the SAME session_id — no end_session, no parent_session_id child, no
    # `name #N` renumber, no contextvar/env/logging re-sync, no memory/context-
    # engine session-switch. The conversation keeps one durable id for life,
    # eliminating the session-rotation bug cluster. Default False during rollout.
    in_place = bool(getattr(agent, "compression_in_place", False))
    # Set True once the in-place DB write actually completes (the DB block can
    # raise and skip it). Surfaced to the gateway via agent._last_compaction_in_place.
    compacted_in_place = False
    logger.info(
        "context compression started: session=%s messages=%d tokens=~%s model=%s focus=%r",
        agent.session_id or "none", _pre_msg_count,
        f"{approx_tokens:,}" if approx_tokens else "unknown", agent.model,
        focus_topic,
    )
    agent._emit_status(COMPACTION_STATUS)

    # ── Compression lock ────────────────────────────────────────────────
    # Atomic, state.db-backed lock per session_id.  Without this, two
    # AIAgent instances that share the same session_id (most commonly the
    # parent-turn agent and its background-review fork — see
    # ``agent/background_review.py``: ``review_agent.session_id =
    # agent.session_id``) can each call compress() on overlapping
    # snapshots of the same conversation.  Both succeed, both rotate
    # ``agent.session_id`` to a fresh id, both create child sessions in
    # state.db parented to the same old id.  The gateway's SessionEntry
    # only catches one rotation, so the other child becomes an orphan
    # that silently accumulates writes — Damien's repro shape.
    #
    # Acquire keyed on the OLD session_id (the rotation target's parent),
    # because that's the id that competing paths see and read from
    # SessionEntry at the start of their own compression attempt.
    #
    # If we can't acquire the lock, another path is mid-compression on
    # this session.  Aborting is correct: the messages are unchanged, the
    # other path's rotation will produce the canonical new session_id,
    # and our caller's auto-compress loop sees ``len(returned) == len(input)``
    # and stops retrying for this cycle. The session is NOT corrupted —
    # we just sit out this round and let the winner finish.
    _lock_db = getattr(agent, "_session_db", None)
    _lock_sid = agent.session_id or ""
    _lock_holder: Optional[str] = None
    # Probe whether the lock subsystem is actually available on this
    # SessionDB instance. A process running mismatched module versions can have
    # this call site while its long-lived SessionDB instance predates the lock
    # API. Only that structural absence is safe to fail open for: compression
    # must make progress rather than spin forever after an update. Once the
    # method has been resolved, every exception from its implementation fails
    # closed because proceeding without a lock can fork the session lineage.
    _try_acquire_lock = None
    _lock_lookup_error: Optional[Exception] = None
    _legacy_session_db_without_lock_api = False
    if _lock_db is not None:
        try:
            _legacy_session_db_without_lock_api = _lock_api_is_absent_on_session_db(
                _lock_db
            )
        except Exception as exc:
            _lock_lookup_error = exc
        if _lock_lookup_error is None and not _legacy_session_db_without_lock_api:
            try:
                _try_acquire_lock = _lock_db.try_acquire_compression_lock
                if not callable(_try_acquire_lock):
                    _lock_lookup_error = TypeError(
                        "compression lock API is present but not callable"
                    )
            except Exception as exc:
                _lock_lookup_error = exc
    try:
        _lock_ttl = float(getattr(agent, "_compression_lock_ttl_seconds", 300.0) or 300.0)
    except (TypeError, ValueError):
        _lock_ttl = 300.0
    _lock_refresh_interval = getattr(agent, "_compression_lock_refresh_interval", None)
    _lock_refresher: Optional[_CompressionLockLeaseRefresher] = None
    if _lock_db is not None and _lock_sid:
        _lock_holder = _compression_lock_holder(agent)
        if _lock_lookup_error is not None:
            # Attribute lookup itself failed for a reason other than a missing
            # lock API. It is unsafe to proceed without a lock in that case.
            _lock_holder = None
            logger.warning(
                "compression lock lookup raised unexpectedly for session=%s "
                "(%s: %s) — skipping compression this cycle",
                _lock_sid, type(_lock_lookup_error).__name__, _lock_lookup_error,
            )
            _lock_acquired = False
        elif _try_acquire_lock is None:
            # The lock API itself is absent on this in-memory instance. Log once
            # and proceed unlocked so an update-version skew cannot leave the
            # outer auto-compression loop making no progress forever.
            _lock_holder = None
            if getattr(agent, "_last_compression_lock_error_sid", None) != _lock_sid:
                agent._last_compression_lock_error_sid = _lock_sid
                logger.warning(
                    "compression lock subsystem unavailable for session=%s "
                    "— proceeding without lock. This usually means a stale "
                    "in-memory module after an update; restart the process "
                    "(or `hermes update`) to resync.",
                    _lock_sid,
                )
            _lock_acquired = True  # acquired-but-unlocked compatibility path
        else:
            try:
                _lock_acquired = _try_acquire_lock(
                    _lock_sid, _lock_holder, ttl_seconds=_lock_ttl
                )
            except Exception as _lock_err:
                # The method exists and entered its implementation but failed.
                # Do not mistake an internal AttributeError or TypeError for
                # version skew: fail closed and preserve session lineage. A
                # failure after SQLite committed the acquire can leave our
                # holder row behind, so release it best-effort before returning
                # unchanged messages; release is holder-qualified and safe when
                # acquisition never succeeded.
                try:
                    _lock_db.release_compression_lock(_lock_sid, _lock_holder)
                except Exception as _release_err:
                    logger.debug(
                        "compression lock cleanup after failed acquire failed: %s",
                        _release_err,
                    )
                _lock_holder = None
                logger.warning(
                    "compression lock acquisition raised unexpectedly for "
                    "session=%s (%s: %s) — skipping compression this cycle",
                    _lock_sid, type(_lock_err).__name__, _lock_err,
                )
                _lock_acquired = False
        if not _lock_acquired:
            try:
                existing = _lock_db.get_compression_lock_holder(_lock_sid)
            except Exception:
                existing = None
            logger.warning(
                "compression skipped: another path is compressing session=%s "
                "(holder=%s) — returning messages unchanged to avoid session fork",
                _lock_sid, existing,
            )
            _lock_holder = None  # don't release a lock we don't own
            # Surface to the user once — quiet for downstream auto-compress loops
            if getattr(agent, "_last_compression_lock_warning_sid", None) != _lock_sid:
                agent._last_compression_lock_warning_sid = _lock_sid
                try:
                    agent._emit_warning(
                        "⚠ Skipping concurrent compression — another path "
                        "is already compressing this session. Will retry "
                        "after it finishes."
                    )
                except Exception:
                    pass
            _existing_sp = getattr(agent, "_cached_system_prompt", None)
            if not _existing_sp:
                _existing_sp = agent._build_system_prompt(system_message)
            return messages, _existing_sp
        if _lock_holder is not None:
            _lock_refresher = _CompressionLockLeaseRefresher(
                _lock_db,
                _lock_sid,
                _lock_holder,
                _lock_ttl,
                _lock_refresh_interval,
            ).start()

    def _release_lock() -> None:
        """Release the lock keyed on the OLD session_id (before rotation)."""
        if _lock_refresher is not None:
            _lock_refresher.stop()
        if _lock_db is not None and _lock_sid and _lock_holder:
            try:
                _lock_db.release_compression_lock(_lock_sid, _lock_holder)
            except Exception as _rel_err:
                logger.debug("compression lock release failed: %s", _rel_err)

    # A delayed contender can acquire the parent lock after the winning path
    # has released it and completed rotation. The lock serializes work but does
    # not by itself prove that this stale agent still owns a live parent.
    if _lock_db is not None and _lock_sid:
        try:
            _parent_already_rotated = _session_was_rotated_by_compression(
                _lock_db, _lock_sid
            )
        except Exception as _session_err:
            logger.warning(
                "compression session ownership lookup failed for session=%s "
                "(%s: %s) - skipping compression this cycle",
                _lock_sid,
                type(_session_err).__name__,
                _session_err,
            )
            _release_lock()
            _existing_sp = getattr(agent, "_cached_system_prompt", None)
            if not _existing_sp:
                _existing_sp = agent._build_system_prompt(system_message)
            return messages, _existing_sp
        if _parent_already_rotated:
            logger.info(
                "compression skipped: session=%s was already rotated by "
                "another compression path",
                _lock_sid,
            )
            _release_lock()
            _existing_sp = getattr(agent, "_cached_system_prompt", None)
            if not _existing_sp:
                _existing_sp = agent._build_system_prompt(system_message)
            return messages, _existing_sp

    # The agent may have been constructed before another path completed an
    # in-place compaction on the same session. Re-read durable breaker state
    # after acquiring the session lock so this final gate cannot act on the
    # stale snapshot loaded by bind_session_state().
    if not force:
        compressor = agent.context_compressor
        _refresh_persisted_compression_guards(compressor)
        blocked = getattr(
            type(compressor),
            "_automatic_compression_blocked",
            None,
        )
        if callable(blocked) and blocked(compressor):
            _release_lock()
            existing_prompt = getattr(agent, "_cached_system_prompt", None)
            if not existing_prompt:
                existing_prompt = agent._build_system_prompt(system_message)
            return messages, existing_prompt

    # Notify external memory provider before compression discards context
    if agent._memory_manager:
        try:
            agent._memory_manager.on_pre_compress(messages)
        except Exception:
            pass

    try:
        compressed = agent.context_compressor.compress(messages, current_tokens=approx_tokens, focus_topic=focus_topic, force=force)
    except TypeError:
        # Plugin context engine with strict signature that doesn't accept
        # focus_topic / force — fall back to calling without them.
        try:
            compressed = agent.context_compressor.compress(messages, current_tokens=approx_tokens)
        except BaseException:
            _release_lock()
            raise
    except BaseException:
        # ANY exception during compress() must release the lock so the
        # session isn't permanently blocked from future compression.
        _release_lock()
        raise

    # Capture boundary quality before session-rotation callbacks run. Built-in
    # and plugin lifecycle hooks may reset per-session compressor fields while
    # rebinding to the child id; the completed attempt's verdict must survive
    # that rebind and be recorded only after the full boundary commits.
    _compression_made_progress = bool(
        getattr(agent.context_compressor, "_last_compression_made_progress", False)
    )
    _compression_used_fallback = bool(
        getattr(agent.context_compressor, "_last_summary_fallback_used", False)
    )

    # If compression aborted (aux LLM failed to produce a usable summary)
    # the compressor returns the input messages unchanged.  Surface the
    # error to the user, skip the session-rotation work entirely (no
    # session has logically ended), and let auto-compress callers detect
    # the no-op via len(returned) == len(input).
    if getattr(agent.context_compressor, "_last_compress_aborted", False):
        try:
            _err = getattr(agent.context_compressor, "_last_summary_error", None) or "unknown error"
            if getattr(agent, "_last_compression_summary_warning", None) != _err:
                agent._last_compression_summary_warning = _err
                agent._emit_warning(
                    f"⚠ Compression aborted: {_err}. "
                    "No messages were dropped — conversation continues unchanged. "
                    "Run /compress to retry, or /new to start a fresh session."
                )
            _existing_sp = getattr(agent, "_cached_system_prompt", None)
            if not _existing_sp:
                _existing_sp = agent._build_system_prompt(system_message)
            return messages, _existing_sp
        finally:
            _release_lock()

    # A compressor that returns the exact input object made no structural
    # progress. Do not rotate/rewrite the session or arm post-compression
    # deferral in that case; its own anti-thrash counter records the no-op.
    if compressed is messages:
        logger.info(
            "Compression made no progress (session=%s) — skipping boundary rewrite.",
            agent.session_id or "none",
        )
        _existing_sp = getattr(agent, "_cached_system_prompt", None)
        if not _existing_sp:
            _existing_sp = agent._build_system_prompt(system_message)
        _release_lock()
        return messages, _existing_sp

    if not compressed:
        logger.error(
            "context compression returned an empty transcript; refusing to "
            "rotate session=%s so the parent remains resumable",
            agent.session_id or "none",
        )
        try:
            agent._emit_warning(
                "⚠ Compression returned an empty transcript. "
                "No session split was performed; conversation continues unchanged."
            )
        except Exception:
            pass
        _existing_sp = getattr(agent, "_cached_system_prompt", None)
        if not _existing_sp:
            _existing_sp = agent._build_system_prompt(system_message)
        _release_lock()
        return messages, _existing_sp

    try:
        summary_error = getattr(agent.context_compressor, "_last_summary_error", None)
        if summary_error:
            if getattr(agent, "_last_compression_summary_warning", None) != summary_error:
                agent._last_compression_summary_warning = summary_error
                agent._emit_warning(
                    f"⚠ Compression summary failed: {summary_error}. "
                    "Inserted a fallback context marker."
                )
        else:
            # No hard failure — but did the configured aux model error out
            # and get recovered by retrying on main?  Surface that so users
            # know their auxiliary.compression.model setting is broken even
            # though compression succeeded.
            _aux_fail_model = getattr(agent.context_compressor, "_last_aux_model_failure_model", None)
            _aux_fail_err = getattr(agent.context_compressor, "_last_aux_model_failure_error", None)
            if _aux_fail_model:
                # Dedup on (model, error) so we don't spam on every compaction
                _aux_key = (_aux_fail_model, _aux_fail_err)
                if getattr(agent, "_last_aux_fallback_warning_key", None) != _aux_key:
                    agent._last_aux_fallback_warning_key = _aux_key
                    agent._emit_warning(
                        f"ℹ Configured compression model '{_aux_fail_model}' failed "
                        f"({_aux_fail_err or 'unknown error'}). Recovered using main model — "
                        "check auxiliary.compression.model in config.yaml."
                    )

        todo_snapshot = agent._todo_store.format_for_injection()
        if todo_snapshot:
            compressed.append({
                "role": "user",
                "content": todo_snapshot,
                "_todo_snapshot_synthetic": True,
            })
        _ensure_compressed_has_user_turn(messages, compressed)

        _prior_system_prompt = (
            getattr(agent, "_cached_system_prompt", None) or system_message
        )
        agent._invalidate_system_prompt()
        new_system_prompt = agent._build_system_prompt(system_message)
        agent._cached_system_prompt = new_system_prompt

        old_session_id: Optional[str] = None
        boundary_completed = False
        fresh_handoff_completed = False
        _handoff_artifact: Optional[Path] = None
        _handoff_packet: Optional[str] = None
        _handoff_fallback_packet: Optional[str] = None
        _planned_child_id: Optional[str] = None
        _rotation_failure: Optional[Exception] = None
        _fresh_parent_title: Optional[str] = None

        session_db = getattr(agent, "_session_db", None)
        parent_session_id = getattr(agent, "session_id", None)
        try:
            completed_compressions = max(
                0, int(agent.context_compressor.compression_count or 0)
            )
        except (TypeError, ValueError):
            completed_compressions = 0
        try:
            durable_handoffs = max(
                0,
                int(
                    session_db.get_auto_handoff_count(parent_session_id)
                    if session_db is not None and parent_session_id
                    else getattr(agent, "_auto_handoff_count", 0)
                ),
            )
        except Exception:
            durable_handoffs = max(
                0, int(getattr(agent, "_auto_handoff_count", 0) or 0)
            )
        agent._auto_handoff_count = durable_handoffs
        try:
            handoff_after = max(
                1, int(getattr(agent, "_auto_handoff_after_compressions", 2))
            )
            handoff_max = max(
                0, int(getattr(agent, "_auto_handoff_max_auto_handoffs", 1))
            )
        except (TypeError, ValueError):
            handoff_after, handoff_max = 2, 1
        handoff_mode = str(
            getattr(agent, "_auto_handoff_mode", "prompt_user")
            or "prompt_user"
        ).lower()
        handoff_eligible = bool(
            getattr(agent, "_auto_handoff_on_compression_enabled", False)
            and not force
            and focus_topic is None
            and session_db is not None
            and parent_session_id
            and completed_compressions >= handoff_after
            and durable_handoffs < handoff_max
            and handoff_mode in {"fresh_session", "prompt_user"}
        )
        if handoff_eligible:
            _planned_child_id = (
                _new_compression_session_id()
                if handoff_mode == "fresh_session" or not in_place
                else parent_session_id
            )
            try:
                _handoff_packet = _build_handoff_packet(
                    agent,
                    messages,
                    approx_tokens=approx_tokens,
                    active_session_id=_planned_child_id,
                    parent_session_id=(
                        parent_session_id
                        if _planned_child_id != parent_session_id
                        else None
                    ),
                    automatic=True,
                )
                if (
                    handoff_mode == "prompt_user"
                    and _planned_child_id != parent_session_id
                ):
                    # Build the rollback coordinate variant before attempting
                    # legacy rotation. If the transaction fails and compaction
                    # persists in place, publication remains nonthrowing packet
                    # selection rather than post-transition packet construction.
                    _handoff_fallback_packet = _build_handoff_packet(
                        agent,
                        messages,
                        approx_tokens=approx_tokens,
                        active_session_id=parent_session_id,
                        parent_session_id=None,
                        automatic=True,
                    )
            except Exception as packet_error:
                logger.warning(
                    "Could not build bounded handoff packet; continuing with "
                    "ordinary compression: %s",
                    packet_error,
                )
                agent._emit_warning(
                    f"⚠️ Could not build bounded handoff packet: {packet_error}"
                )
                handoff_eligible = False

        # Memory extraction is independent from the DB transition. It must not
        # be able to strand a committed child or prevent the safe persistence
        # fallback from running.
        if session_db is not None:
            try:
                agent.commit_memory_session(messages)
            except Exception as memory_error:
                logger.debug(
                    "Pre-compression memory extraction failed: %s", memory_error
                )

        # A fresh child's active context is the packet itself. Publish it before
        # committing the transition so every potentially failing packet step is
        # complete. A failed write consumes no quota and falls through to the
        # current ordinary compaction behavior.
        if (
            handoff_eligible
            and handoff_mode == "fresh_session"
            and _handoff_packet
            and _planned_child_id
            and isinstance(new_system_prompt, str)
            and new_system_prompt.strip()
        ):
            try:
                _handoff_artifact = _publish_handoff_artifact(
                    agent, _handoff_packet
                )
            except Exception as artifact_error:
                logger.warning(
                    "Could not publish bounded handoff packet; continuing with "
                    "ordinary compression: %s",
                    artifact_error,
                )
                agent._emit_warning(
                    f"⚠️ Could not write bounded handoff packet: {artifact_error}"
                )
                handoff_eligible = False

        if (
            handoff_eligible
            and handoff_mode == "fresh_session"
            and _handoff_packet
            and _handoff_artifact
            and _planned_child_id
        ):
            assert session_db is not None
            assert parent_session_id is not None
            try:
                _fresh_parent_title = session_db.get_session_title(
                    parent_session_id
                )
            except Exception:
                _fresh_parent_title = None
            packet_messages = [{"role": "user", "content": _handoff_packet}]
            rotated, consumed, _rotation_failure = _rotate_compression_session(
                agent,
                parent_session_id=parent_session_id,
                child_session_id=_planned_child_id,
                child_messages=packet_messages,
                system_prompt=new_system_prompt,
                consume_auto_handoff=True,
            )
            if rotated:
                old_session_id = parent_session_id
                boundary_completed = True
                fresh_handoff_completed = True
                compressed = packet_messages
                _reset_agent_for_fresh_handoff(
                    agent,
                    parent_session_id=parent_session_id,
                    child_session_id=_planned_child_id,
                    system_prompt=new_system_prompt,
                    child_messages=packet_messages,
                    handoff_count=consumed,
                )
                try:
                    from hermes_cli.goals import migrate_goal_to_session

                    migrate_goal_to_session(
                        parent_session_id,
                        _planned_child_id,
                        reason="compression",
                    )
                except Exception as goal_error:
                    logger.debug(
                        "Could not migrate goal on fresh handoff: %s",
                        goal_error,
                    )
                if _fresh_parent_title:
                    try:
                        child_title = session_db.get_next_title_in_lineage(
                            _fresh_parent_title
                        )
                        session_db.set_session_title(
                            _planned_child_id, child_title
                        )
                    except Exception as title_error:
                        logger.debug(
                            "Could not propagate title on fresh handoff: %s",
                            title_error,
                        )
                agent._emit_status(
                    f"📦 Bounded handoff {consumed}/{handoff_max} started "
                    f"session {agent.session_id}. Packet: {_handoff_artifact}"
                )
            else:
                _remove_failed_auto_artifact(_handoff_artifact)
                _handoff_artifact = None
                logger.warning(
                    "Fresh bounded handoff rolled back; preserving parent %s "
                    "and using in-place compaction: %s",
                    parent_session_id,
                    _rotation_failure,
                )
                agent._emit_warning(
                    "⚠️ Fresh handoff could not rotate atomically; continuing "
                    "on the original session with in-place compaction."
                )

        if not fresh_handoff_completed and session_db is not None and parent_session_id:
            # A failed fresh rotation always falls back in-place, even when the
            # legacy config requested child rotation. This keeps routing on the
            # active parent and gives the compressed transcript a coherent DB
            # baseline rather than appending it over uncompressed rows.
            persist_in_place = in_place or _rotation_failure is not None
            if persist_in_place:
                try:
                    session_db.archive_and_compact(parent_session_id, compressed)
                    try:
                        session_db.update_system_prompt(
                            parent_session_id, new_system_prompt
                        )
                    except Exception as prompt_error:
                        logger.debug(
                            "Could not refresh in-place system prompt: %s",
                            prompt_error,
                        )
                    agent._flushed_db_message_ids = set()
                    agent._last_flushed_db_idx = 0
                    agent._flushed_db_message_session_id = parent_session_id
                    compacted_in_place = True
                    boundary_completed = True
                except Exception as fallback_error:
                    logger.warning(
                        "In-place compression persistence failed for active "
                        "session %s: %s",
                        parent_session_id,
                        fallback_error,
                    )
                    agent._cached_system_prompt = _prior_system_prompt
                    return messages, _prior_system_prompt
            else:
                # Legacy child rotation now uses the same single transaction as
                # fresh handoff: strict child insert + transcript + parent close.
                try:
                    agent._flush_messages_to_session_db(messages)
                except Exception as flush_error:
                    logger.warning(
                        "Could not flush the parent before compression rotation; "
                        "falling back in-place: %s",
                        flush_error,
                    )
                    _rotation_failure = flush_error

                old_title = None
                try:
                    old_title = session_db.get_session_title(parent_session_id)
                except Exception:
                    pass
                child_id = _planned_child_id or _new_compression_session_id()
                if _rotation_failure is None:
                    rotated, inherited_count, _rotation_failure = (
                        _rotate_compression_session(
                            agent,
                            parent_session_id=parent_session_id,
                            child_session_id=child_id,
                            child_messages=compressed,
                            system_prompt=new_system_prompt,
                            consume_auto_handoff=False,
                        )
                    )
                else:
                    rotated, inherited_count = False, durable_handoffs

                if rotated:
                    old_session_id = parent_session_id
                    boundary_completed = True
                    agent.session_id = child_id
                    agent._parent_session_id = parent_session_id
                    agent._session_db_created = True
                    agent._auto_handoff_count = inherited_count
                    child_config = dict(
                        getattr(agent, "_session_init_model_config", {}) or {}
                    )
                    child_config["_auto_handoff_count"] = inherited_count
                    agent._session_init_model_config = child_config
                    _mark_child_messages_persisted(agent, compressed)
                    _sync_rotated_session_context(agent, child_id)
                    try:
                        from hermes_cli.goals import migrate_goal_to_session

                        migrate_goal_to_session(
                            parent_session_id, child_id, reason="compression"
                        )
                    except Exception as goal_error:
                        logger.debug(
                            "Could not migrate goal on compression: %s", goal_error
                        )
                    if old_title:
                        try:
                            new_title = session_db.get_next_title_in_lineage(old_title)
                            session_db.set_session_title(child_id, new_title)
                        except Exception as title_error:
                            logger.debug(
                                "Could not propagate title on compression: %s",
                                title_error,
                            )
                else:
                    logger.warning(
                        "Atomic compression rotation rolled back; preserving "
                        "parent %s and using in-place compaction: %s",
                        parent_session_id,
                        _rotation_failure,
                    )
                    try:
                        session_db.archive_and_compact(
                            parent_session_id, compressed
                        )
                        try:
                            session_db.update_system_prompt(
                                parent_session_id, new_system_prompt
                            )
                        except Exception:
                            pass
                        agent.session_id = parent_session_id
                        agent._session_db_created = True
                        _sync_rotated_session_context(agent, parent_session_id)
                        agent._flushed_db_message_ids = set()
                        agent._last_flushed_db_idx = 0
                        agent._flushed_db_message_session_id = parent_session_id
                        compacted_in_place = True
                        boundary_completed = True
                    except Exception as fallback_error:
                        logger.warning(
                            "Compression rotation and in-place fallback both "
                            "failed for %s: %s",
                            parent_session_id,
                            fallback_error,
                        )
                        agent._cached_system_prompt = _prior_system_prompt
                        return messages, _prior_system_prompt

        # Prompt-only mode never changes the normal persistence semantics. The
        # packet was built before any legacy rotation and is published only
        # after final session coordinates are authoritative. Consume quota only
        # after successful publication; delete the artifact if DB consumption
        # loses a race or fails, so failed writes/updates cannot burn quota.
        if (
            handoff_eligible
            and handoff_mode == "prompt_user"
            and _handoff_packet
            and boundary_completed
            and session_db is not None
            and agent.session_id
        ):
            try:
                if (
                    _handoff_fallback_packet
                    and agent.session_id != _planned_child_id
                ):
                    _handoff_packet = _handoff_fallback_packet
                _handoff_artifact = _publish_handoff_artifact(
                    agent, _handoff_packet
                )
                consumed = session_db.consume_auto_handoff(
                    agent.session_id,
                    max_auto_handoffs=handoff_max,
                )
                agent._auto_handoff_count = consumed
                child_config = dict(
                    getattr(agent, "_session_init_model_config", {}) or {}
                )
                child_config["_auto_handoff_count"] = consumed
                agent._session_init_model_config = child_config
                agent._emit_status(
                    f"📦 Session quality threshold reached. Review the "
                    f"bounded packet before /new: {_handoff_artifact}"
                )
            except Exception as artifact_error:
                _remove_failed_auto_artifact(_handoff_artifact)
                _handoff_artifact = None
                logger.warning(
                    "Could not write/record prompt-only handoff packet: %s",
                    artifact_error,
                )
                agent._emit_warning(
                    f"⚠️ Could not write bounded handoff packet: {artifact_error}"
                )

        # Compaction-boundary bookkeeping uses explicit completed persistence,
        # never merely the configured in_place flag. This keeps null-DB and
        # rolled-back transitions from publishing false gateway coordinates.
        _old_sid = old_session_id
        _is_boundary = boundary_completed
        _boundary_parent = _old_sid or agent.session_id or ""

        # Notify the context engine that a compaction boundary occurred. Plugin
        # engines (e.g. hermes-lcm) use boundary_reason="compression" to preserve
        # DAG lineage / checkpoint per-session state across the boundary instead of
        # re-initializing fresh. See hermes-lcm#68. Built-in ContextCompressor
        # ignores kwargs. Fires in BOTH modes: rotation passes old→new ids; in-place
        # passes the SAME id (the boundary is real even though the id didn't move).
        try:
            if _is_boundary and hasattr(agent.context_compressor, "on_session_start"):
                agent.context_compressor.on_session_start(
                    agent.session_id or "",
                    boundary_reason="compression",
                    old_session_id=_boundary_parent,
                    fresh_handoff=fresh_handoff_completed,
                    platform=getattr(agent, "platform", None) or "cli",
                    conversation_id=getattr(agent, "_gateway_session_key", None),
                )
        except Exception as _ce_err:
            logger.debug("context engine on_session_start (compression): %s", _ce_err)

        # Notify memory providers of the compaction boundary so provider-cached
        # per-session state (Hindsight's _document_id, accumulated turn buffers,
        # counters) refreshes. reset=False because the logical conversation
        # continues. See #6672. Fires in BOTH modes: in-place uses the same id as
        # parent (the conversation didn't fork, but the buffer must still be told
        # the transcript was compacted so it doesn't double-count dropped turns).
        try:
            if _is_boundary and agent._memory_manager:
                agent._memory_manager.on_session_switch(
                    agent.session_id or "",
                    parent_session_id=_boundary_parent,
                    reset=False,
                    reason="compression",
                )
        except Exception as _me_err:
            logger.debug("memory manager on_session_switch (compression): %s", _me_err)

        # Warn on repeated compressions (quality degrades with each pass).
        # Route through _emit_status (like the other compression warnings above)
        # so the warning reaches the TUI / Telegram / Discord via status_callback,
        # not just CLI stdout. _emit_status still _vprints for the CLI, and
        # storing it on _compression_warning lets replay_compression_warning
        # re-deliver it once a late-bound gateway status_callback is wired (#36908).
        _cc = completed_compressions
        if _cc >= 2:
            _cc_msg = (
                f"{agent.log_prefix}⚠️  Session compressed {_cc} times — "
                f"accuracy may degrade. Consider /new to start fresh."
            )
            agent._compression_warning = _cc_msg
            agent._emit_status(_cc_msg)

        # Emit session:compress event so hooks (e.g. MemPalace sync) can ingest
        # the completed old session before its details are lost. In in-place mode
        # there is no old id (same session); ``in_place=True`` tells hooks the
        # transcript was compacted on the same id rather than rotated.
        if getattr(agent, "event_callback", None):
            try:
                agent.event_callback("session:compress", {
                    "platform": agent.platform or "",
                    "session_id": agent.session_id,
                    "old_session_id": _old_sid or "",
                    "in_place": compacted_in_place,
                    "compression_count": completed_compressions,
                })
            except Exception as e:
                logger.debug("event_callback error on session:compress: %s", e)

        # Surface the compaction mode to the caller (run_conversation / gateway)
        # via a rotation-independent flag. The gateway uses this — NOT an
        # id-change diff — to re-baseline transcript handling (history_offset=0 +
        # rewrite on the same id) when compaction happened in place. See #38763.
        agent._last_compaction_in_place = compacted_in_place

        # Keep the post-compression rough estimate for diagnostics, but do not
        # treat it as provider-reported prompt usage. Schema-heavy rough estimates
        # can remain above threshold even after the next real API request fits.
        _compressed_est = estimate_request_tokens_rough(
            compressed,
            system_prompt=new_system_prompt or "",
            tools=agent.tools or None,
        )
        if not fresh_handoff_completed:
            agent.context_compressor.last_compression_rough_tokens = _compressed_est
            agent.context_compressor.last_prompt_tokens = -1
            agent.context_compressor.last_completion_tokens = 0
            agent.context_compressor.awaiting_real_usage_after_compression = True
        # Arm the effectiveness verdict only after a completed rewrite crosses
        # the full compaction boundary. Exceptions, aborts, and no-op attempts
        # leave this false, so unrelated later usage cannot be charged to an
        # attempt that never changed the transcript.
        if _compression_made_progress and not fresh_handoff_completed:
            record_boundary = getattr(
                type(agent.context_compressor),
                "record_completed_compaction",
                None,
            )
            if callable(record_boundary):
                record_boundary(
                    agent.context_compressor,
                    used_fallback=_compression_used_fallback,
                )
            else:
                agent.context_compressor._verify_compaction_cleared_threshold = True

        # Clear the file-read dedup cache.  After compression the original
        # read content is summarised away — if the model re-reads the same
        # file it needs the full content, not a "file unchanged" stub.
        try:
            from tools.file_tools import reset_file_dedup
            reset_file_dedup(task_id)
        except Exception:
            pass

        logger.info(
            "context compression done: session=%s messages=%d->%d rough_tokens=~%s awaiting_real_usage=%s",
            agent.session_id or "none", _pre_msg_count, len(compressed),
            f"{_compressed_est:,}",
            not fresh_handoff_completed,
        )
        return compressed, new_system_prompt
    finally:
        # Release the lock on the OLD session_id only AFTER rotation completed
        # and all post-rotation bookkeeping (memory manager, context engine,
        # file dedup) ran. A concurrent path that wakes up the moment we
        # release will see the NEW session_id in state.db / SessionEntry and
        # acquire on that — no race against our just-finished work.
        _release_lock()


def _compress_context_via_codex_app_server(
    agent: Any,
    messages: list,
    system_message: Optional[str],
    *,
    approx_tokens: Optional[int] = None,
    task_id: str = "default",
    force: bool = False,
) -> Tuple[list, str]:
    """Route compaction to Codex app-server for Codex-owned threads.

    Hermes' normal compressor rewrites the local OpenAI-style transcript.
    That does not shrink the actual Codex app-server thread context. For this
    runtime, ask Codex to compact its own thread and keep Hermes' transcript
    unchanged.
    """
    auto_mode = str(
        getattr(agent, "codex_app_server_auto_compaction", "native") or "native"
    ).lower()
    if auto_mode not in {"native", "hermes", "off"}:
        auto_mode = "native"
    if not force and auto_mode != "hermes":
        logger.info(
            "codex app-server compaction skipped: mode=%s force=false "
            "(session=%s messages=%d tokens=~%s)",
            auto_mode,
            getattr(agent, "session_id", None) or "none",
            len(messages),
            f"{approx_tokens:,}" if approx_tokens else "unknown",
        )
        existing_prompt = getattr(agent, "_cached_system_prompt", None)
        if not existing_prompt:
            existing_prompt = agent._build_system_prompt(system_message)
        return messages, existing_prompt

    codex_session = getattr(agent, "_codex_session", None)
    if codex_session is None:
        logger.info(
            "codex app-server compaction skipped: no active codex thread "
            "(session=%s messages=%d tokens=~%s)",
            getattr(agent, "session_id", None) or "none",
            len(messages),
            f"{approx_tokens:,}" if approx_tokens else "unknown",
        )
        existing_prompt = getattr(agent, "_cached_system_prompt", None)
        if not existing_prompt:
            existing_prompt = agent._build_system_prompt(system_message)
        return messages, existing_prompt

    logger.info(
        "codex app-server compaction started: session=%s messages=%d tokens=~%s",
        getattr(agent, "session_id", None) or "none",
        len(messages),
        f"{approx_tokens:,}" if approx_tokens else "unknown",
    )
    try:
        agent._emit_status(COMPACTION_STATUS)
    except Exception:
        pass

    result = codex_session.compact_thread()
    if getattr(result, "should_retire", False):
        try:
            codex_session.close()
        except Exception:
            pass
        agent._codex_session = None

    if getattr(result, "interrupted", False) or getattr(result, "error", None):
        try:
            agent._emit_warning(
                f"⚠ Codex app-server compaction failed: {result.error}"
            )
        except Exception:
            pass
        existing_prompt = getattr(agent, "_cached_system_prompt", None)
        if not existing_prompt:
            existing_prompt = agent._build_system_prompt(system_message)
        return messages, existing_prompt

    try:
        from agent.codex_runtime import (
            _record_codex_app_server_compaction,
            _record_codex_app_server_usage,
        )

        _record_codex_app_server_compaction(
            agent,
            result,
            approx_tokens=approx_tokens,
            force=True,
        )
        # An empty usage report must consume the pending post-compaction verdict
        # rather than leaving preflight deferral armed until some unrelated later
        # Codex turn supplies usage. Minimal external test engines may not expose
        # the ContextEngine update hook; preserve their existing bookkeeping.
        if hasattr(agent.context_compressor, "update_from_response"):
            _record_codex_app_server_usage(agent, result)
    except Exception:
        logger.debug("codex compaction bookkeeping failed", exc_info=True)

    try:
        from tools.file_tools import reset_file_dedup

        reset_file_dedup(task_id)
    except Exception:
        pass

    logger.info(
        "codex app-server compaction done: session=%s thread=%s turn=%s",
        getattr(agent, "session_id", None) or "none",
        getattr(result, "thread_id", None) or "",
        getattr(result, "turn_id", None) or "",
    )
    existing_prompt = getattr(agent, "_cached_system_prompt", None)
    if not existing_prompt:
        existing_prompt = agent._build_system_prompt(system_message)
    return messages, existing_prompt


def try_shrink_image_parts_in_messages(
    api_messages: list,
    *,
    max_dimension: int = 8000,
) -> bool:
    """Re-encode all native image parts at a smaller size to recover from
    image-too-large errors (Anthropic 5 MB, unknown other providers).

    Mutates ``api_messages`` in place. Returns True if any image part was
    actually replaced, False if there were no image parts to shrink or
    Pillow couldn't help (caller should surface the original error).

    Strategy: look for ``image_url`` / ``input_image`` parts carrying a
    ``data:image/...;base64,...`` payload, plus Anthropic-native
    ``{"type": "image", "source": {"type": "base64", ...}}`` blocks.
    For each one whose encoded size exceeds 4 MB (a safe target that slides
    under Anthropic's 5 MB ceiling with header overhead) or whose longest side
    exceeds ``max_dimension``, write the base64 to a tempfile, call
    ``vision_tools._resize_image_for_vision`` to produce a smaller data
    URL, and substitute it in place.

    Non-data-URL images (http/https URLs) are not touched — the provider
    fetches those itself and the size limit is different.
    """
    if not api_messages:
        return False

    try:
        from tools.vision_tools import _resize_image_for_vision
    except Exception as exc:
        logger.warning("image-shrink recovery: vision_tools unavailable — %s", exc)
        return False

    # 4 MB target leaves comfortable headroom under Anthropic's 5 MB.
    # Non-Anthropic providers we haven't observed rejecting are fine with
    # much larger; shrinking to 4 MB here loses quality but only fires
    # after a confirmed provider rejection, so the alternative is failure.
    target_bytes = 4 * 1024 * 1024
    # Anthropic enforces an 8000px per-side dimension cap independently of
    # the 5 MB byte cap.  In many-image requests, the provider can report a
    # lower cap (observed: 2000px).  The caller passes that parsed ceiling
    # when the rejection includes it.
    changed_count = 0
    # Track parts that are over the target but could NOT be shrunk under it.
    # If any survive, retrying is pointless — the same oversized payload will
    # be re-sent and rejected again, wasting the single retry budget.  We only
    # report success (caller retries) when every over-threshold image was
    # actually brought under the target.
    unshrinkable_oversized = 0

    def _decode_pixels(data_url: str) -> Optional[tuple]:
        """Return ``(width, height)`` of a base64 data URL, or None on failure.

        Soft-depends on Pillow; returns None (caller falls back to a
        bytes-only check) if Pillow is missing or the payload is corrupt.
        """
        try:
            import base64 as _b64_dim
            import io as _io_dim
            header_d, _, data_d = data_url.partition(",")
            if not data_d or not data_url.startswith("data:"):
                return None
            from PIL import Image as _PILImage
            with _PILImage.open(_io_dim.BytesIO(_b64_dim.b64decode(data_d))) as _img:
                return _img.size
        except Exception:
            return None

    def _shrink_data_url(url: str) -> tuple:
        """Return ``(resized_url, unshrinkable)`` for a data URL.

        ``resized_url`` is a smaller/dimension-correct data URL, or None when
        no rewrite was applied.  ``unshrinkable`` is True only when the image
        exceeded a constraint (byte-size or dimensions) and the resize failed
        to satisfy *that same* constraint — so the caller knows retrying is
        pointless even if a different image in the request shrank.
        """
        if not isinstance(url, str) or not url.startswith("data:"):
            return None, False

        # Determine which constraint is binding.  The accept/reject gate below
        # MUST be checked against the same axis that triggered the shrink: a
        # downscaled screenshot PNG routinely re-encodes to *more* bytes than
        # the original (PNG compression is non-monotonic in image size — a
        # smaller raster with LANCZOS resampling noise compresses worse than a
        # larger smooth one).  Rejecting a pixel-correct downscale purely
        # because its bytes grew permanently wedges sessions on the Anthropic
        # many-image 2000px path (#48013).
        needs_shrink = len(url) > target_bytes  # over byte budget
        triggered_by = "bytes" if needs_shrink else None
        if not needs_shrink:
            # Bytes are fine — check pixel dimensions against the provider's
            # reported per-side cap.  A screenshot can be tiny in bytes yet
            # too large in pixels.
            dims = _decode_pixels(url)
            if dims is None:
                # Pillow missing or corrupt data — fall back to byte-only.
                return None, False
            if max(dims) <= max_dimension:
                return None, False  # both bytes and pixels are within limits
            needs_shrink = True
            triggered_by = "dimension"

        try:
            header, _, data = url.partition(",")
            mime = "image/jpeg"
            if header.startswith("data:"):
                mime_part = header[len("data:"):].split(";", 1)[0].strip()
                if mime_part.startswith("image/"):
                    mime = mime_part
            import base64 as _b64
            raw = _b64.b64decode(data)
            suffix = {
                "image/png": ".png", "image/gif": ".gif", "image/webp": ".webp",
                "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/bmp": ".bmp",
            }.get(mime, ".jpg")
            tmp = tempfile.NamedTemporaryFile(
                prefix="hermes_shrink_", suffix=suffix, delete=False,
            )
            try:
                tmp.write(raw)
                tmp.close()
                resized = _resize_image_for_vision(
                    Path(tmp.name),
                    mime_type=mime,
                    max_base64_bytes=target_bytes,
                    max_dimension=max_dimension,
                )
            finally:
                try:
                    Path(tmp.name).unlink(missing_ok=True)
                except Exception:
                    pass
            if not resized:
                # Resize returned nothing — Pillow couldn't help.
                return None, True
            if triggered_by == "bytes":
                # Byte budget is the binding constraint — bytes must shrink.
                if len(resized) >= len(url):
                    return None, True  # re-encode made it bigger
                # The per-side dimension cap is ALSO an active provider
                # constraint on this request (the caller passes the parsed cap
                # to both this helper and the resizer).  _resize_image_for_vision
                # returns a best-effort, possibly-over-cap blob when it
                # exhausts its halving budget — it freezes the long side once
                # the short side hits its 64px floor, so a very-high-aspect
                # image can stay over the cap even after bytes shrank.  If the
                # output is still over the cap, retrying would re-400 on
                # dimensions; treat it as unshrinkable.  (Skip when dims can't
                # be decoded — preserves historical byte-only behaviour.)
                new_dims = _decode_pixels(resized)
                if new_dims is not None and max(new_dims) > max_dimension:
                    return None, True
                return resized, False
            # triggered_by == "dimension": the per-side cap is binding.  The
            # re-encode may have grown in bytes; accept it as long as it is now
            # within the dimension cap.  Verify the new dimensions when we can.
            new_dims = _decode_pixels(resized)
            if new_dims is not None:
                if max(new_dims) <= max_dimension:
                    return resized, False
                # Still over the per-side cap — the resize didn't satisfy it.
                return None, True
            # Couldn't verify the re-encode's dimensions (corrupt output or
            # Pillow gone mid-call).  Fall back to the historical "bytes must
            # shrink" gate so we never accept an unverifiable, byte-larger blob.
            if len(resized) >= len(url):
                return None, True
            return resized, False
        except Exception as exc:
            logger.warning("image-shrink recovery: re-encode failed — %s", exc)
            return None, triggered_by is not None

    def _source_to_data_url(source: Any) -> Optional[str]:
        if not isinstance(source, dict) or source.get("type") != "base64":
            return None
        data = source.get("data")
        if not isinstance(data, str) or not data:
            return None
        media_type = str(source.get("media_type") or "image/jpeg").strip()
        if not media_type.startswith("image/"):
            media_type = "image/jpeg"
        return f"data:{media_type};base64,{data}"

    def _write_data_url_to_source(source: dict, data_url: str) -> None:
        header, _, data = data_url.partition(",")
        media_type = "image/jpeg"
        if header.startswith("data:"):
            candidate = header[len("data:"):].split(";", 1)[0].strip()
            if candidate.startswith("image/"):
                media_type = candidate
        source["type"] = "base64"
        source["media_type"] = media_type
        source["data"] = data

    for msg in api_messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "image":
                source = part.get("source")
                url = _source_to_data_url(source)
                resized, unshrinkable = _shrink_data_url(url or "")
                if resized and isinstance(source, dict):
                    _write_data_url_to_source(source, resized)
                    changed_count += 1
                elif unshrinkable:
                    unshrinkable_oversized += 1
                continue
            if ptype not in {"image_url", "input_image"}:
                continue
            image_value = part.get("image_url")
            # OpenAI chat.completions: {"image_url": {"url": "data:..."}}
            # OpenAI Responses: {"image_url": "data:..."}
            if isinstance(image_value, dict):
                url = image_value.get("url", "")
                resized, unshrinkable = _shrink_data_url(url)
                if resized:
                    image_value["url"] = resized
                    changed_count += 1
                elif unshrinkable:
                    unshrinkable_oversized += 1
            elif isinstance(image_value, str):
                resized, unshrinkable = _shrink_data_url(image_value)
                if resized:
                    part["image_url"] = resized
                    changed_count += 1
                elif unshrinkable:
                    unshrinkable_oversized += 1

    if changed_count:
        logger.info(
            "image-shrink recovery: re-encoded %d image part(s) to fit under %.0f MB",
            changed_count, target_bytes / (1024 * 1024),
        )
    if unshrinkable_oversized:
        # At least one oversized image could not be shrunk under the target.
        # Retrying would re-send it and fail identically, so signal "no
        # progress" even if other parts shrank — the caller will surface the
        # original error rather than burning its single retry on a no-op.
        logger.warning(
            "image-shrink recovery: %d oversized image part(s) could not be "
            "shrunk under %.0f MB — not retrying (would re-send rejected payload)",
            unshrinkable_oversized, target_bytes / (1024 * 1024),
        )
        return False
    return changed_count > 0


__all__ = [
    "COMPACTION_STATUS",
    "COMPACTION_STATUS_MARKER",
    "check_compression_model_feasibility",
    "replay_compression_warning",
    "compress_context",
    "try_shrink_image_parts_in_messages",
]
