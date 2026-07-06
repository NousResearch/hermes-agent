"""KarinAI /v1/runs logic for the api_server platform (gateway bridge).

Everything /v1/runs-KarinAI that used to live inline in
``gateway/platforms/api_server.py`` — attachment surfacing/inlining, the
image-mode decision, the managed toolset override, and the app-tool-gateway
request validation — moved here verbatim to shrink the recurring
upstream-sync conflict surface (see ``docs/karinai-gateway-bridge-design.md``).
The adapter keeps thin, clearly-marked call-site hooks and owns all HTTP
concerns (this module returns error strings for the 400 paths instead of
building responses).

Import rule: ``gateway.*`` / ``agent.*`` imports must stay lazy inside
functions (as they were at the original call sites) — the adapter imports this
module at module level.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# --- /v1/runs `attachments` (KarinAI managed runs) ------------------------------
# The KarinAI backend materializes a conversation's uploaded files into the
# workspace BEFORE dispatch and sends a manifest of confirmed-present files in
# body["attachments"] ({file_id, safe_name, mime, size, local_path, purpose}).
# The agent must be TOLD about them (a context note per file, mirroring how the
# messaging platforms handle inbound documents) or it has no idea they exist.
#
# Context model note (drives the design below): /v1/runs builds the model's
# context SOLELY from the request body — the backend stores the CLEAN user input
# and resends clean history, so nothing the gateway prepends here survives into
# later runs' context. Path notes are therefore re-injected on EVERY run (they
# are small and never duplicated in backend history); only the expensive content
# INLINING is deduped per session (first surfacing inlines, later runs carry the
# path note and the agent tool-reads on demand).
#
# Per-file inline cap matches the platform adapters' MAX_TEXT_INJECT_BYTES:
# all-or-nothing, no truncation — an over-cap text file degrades to a path note.
MAX_ATTACHMENT_INLINE_BYTES = 100 * 1024
# Aggregate inline budget per run: a many-file manifest must not blow the prompt
# (inlined bytes come from DISK, so MAX_REQUEST_BYTES does not bound them).
# Files past the budget get path notes and stay eligible for a later run.
MAX_ATTACHMENT_INLINE_TOTAL_BYTES = 256 * 1024
# Inlining reads file bytes into the prompt, so it is restricted to paths under
# the managed workspace mount (the same dir the runtime validates at startup).
# Anything else still gets a path note (the agent's own tools enforce their own
# sandboxing when it goes to read the file).
ATTACHMENT_INLINE_ROOT = (
    os.getenv("KARINAI_ATTACHMENT_INLINE_ROOT") or os.getenv("KARINAI_WORKSPACE_DIR") or "/workspace"
)
# Cap on sessions tracked for inline dedup.
MAX_ATTACHMENT_DEDUP_SESSIONS = 2048
# Native image attachment (vision models): bound the COUNT per run; images past
# the cap get a path note and stay eligible for a later run. Like text inlining,
# pixels attach once per session per file.
MAX_ATTACHMENT_NATIVE_IMAGES_PER_RUN = 3
# Per-image byte gate: the read + base64 encode happen synchronously in the
# request handler and the payload rides every subsequent model call of the run,
# so oversized images degrade to a path note (the agent's vision tool applies
# its own proactive embed capping when it goes to look). 5 MB also matches the
# strictest known provider ceiling, avoiding a guaranteed shrink-retry cycle.
MAX_ATTACHMENT_NATIVE_IMAGE_BYTES = 5 * 1024 * 1024

_IMAGE_ATTACHMENT_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"})


def validate_run_attachments(raw: Any) -> Optional[str]:
    """Validate body["attachments"] shape; return a human-readable error or None.

    Mirrors the conversation_history validation style: per-index messages, fail
    the request loudly — the manifest comes from the trusted runtime-manager, so
    a malformed entry is a programming error, not user input to tolerate."""
    if not isinstance(raw, list):
        return "'attachments' must be an array of attachment objects"
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            return f"attachments[{i}] must be an object"
        for key in ("safe_name", "local_path"):
            if not isinstance(entry.get(key), str) or not entry.get(key, "").strip():
                return f"attachments[{i}] must have non-empty '{key}'"
    return None


def _attachment_dedup_key(entry: Dict[str, Any]) -> str:
    file_id = entry.get("file_id")
    return str(file_id) if file_id else str(entry.get("local_path", ""))


def _is_text_attachment(safe_name: str, mime: str) -> bool:
    # Same gate the platform adapters use: extension allowlist OR text/* MIME —
    # deliberately NOT a blind UTF-8 sniff (PDF/zip can start decodable).
    from gateway.platforms.base import _TEXT_INJECT_EXTENSIONS

    ext = os.path.splitext(safe_name)[1].lower()
    return ext in _TEXT_INJECT_EXTENSIONS or (mime or "").startswith("text/")


def is_image_attachment(safe_name: str, mime: str) -> bool:
    ext = os.path.splitext(safe_name)[1].lower()
    return (mime or "").startswith("image/") or ext in _IMAGE_ATTACHMENT_EXTENSIONS


def _build_image_attachment_note(display_name: str, agent_path: str) -> str:
    """Context note for an image whose pixels are NOT attached to this turn
    (text-mode model, per-run cap reached, or already attached in an earlier
    run). Instructs tool-based viewing instead of punting back to the user —
    the same lesson as the platforms' binary-document wording."""
    return (
        f"[The user sent an image: '{display_name}'. It is saved at: {agent_path}. "
        f"Its pixels are not attached to this message. If the request involves "
        f"what the image shows, view or analyze it yourself with your available "
        f"image/vision tools instead of asking the user to describe it.]"
    )


_IMAGE_MODE_CACHE: Dict[str, Tuple[str, float]] = {}
_IMAGE_MODE_CACHE_TTL_SECONDS = 300.0


def decide_run_image_mode() -> str:
    """"native" (attach pixels on the user turn) or "text" (path note only) for
    the active model — same resolution the gateway runner uses (config +
    model-capability lookup); any failure degrades to "text".

    Cached (5 min TTL): the capability lookup can hit a cold models.dev fetch
    (up to ~15s) and this runs synchronously in the request handler; the model
    config effectively never changes within a container's lifetime. (Managed
    runs short-circuit on the explicit supports_vision override and never
    fetch.)"""
    cached = _IMAGE_MODE_CACHE.get("mode")
    now = time.monotonic()
    if cached is not None and now - cached[1] < _IMAGE_MODE_CACHE_TTL_SECONDS:
        return cached[0]
    try:
        from agent.auxiliary_client import _read_main_model, _read_main_provider
        from agent.image_routing import decide_image_input_mode
        from hermes_cli.config import load_config

        mode = decide_image_input_mode(_read_main_provider(), _read_main_model(), load_config())
    except Exception as exc:  # noqa: BLE001 — degraded mode, never a crash
        logger.debug("run attachments: image mode decision failed, using text — %s", exc)
        mode = "text"
    _IMAGE_MODE_CACHE["mode"] = (mode, now)
    return mode


def workspace_relative_path(local_path: str) -> str:
    """Render a workspace-rooted path relative to the workspace root, for the
    attachment note. The agent's tools run with the workspace as cwd, so a
    relative `inputs/...` path resolves for it; the absolute sandbox path is
    internal detail that would otherwise flow into user-visible transcripts and
    public run events. Paths outside the root are returned unchanged
    (non-managed callers)."""
    try:
        root = Path(ATTACHMENT_INLINE_ROOT).resolve()
        resolved = Path(local_path).resolve()
        if root in resolved.parents:
            return str(resolved.relative_to(root))
    except (OSError, ValueError):
        pass
    return local_path


def _resolve_under_inline_root(local_path: str) -> Optional[Path]:
    """Resolve a path iff it is contained under the workspace inline root.
    Reading file bytes into the prompt (text inlining, image pixels) is gated on
    this; anything else degrades to a path note."""
    try:
        resolved = Path(local_path).resolve()
        root = Path(ATTACHMENT_INLINE_ROOT).resolve()
        if root in resolved.parents:
            return resolved
    except (OSError, ValueError):
        pass
    return None


def _read_inline_attachment_text(local_path: str) -> Optional[str]:
    """Read a text attachment for prompt inlining, or None when it must degrade
    to a path note: outside the workspace inline root, missing, over the 100 KB
    all-or-nothing cap (matching platform adapters — no truncation), or not
    valid UTF-8."""
    resolved = _resolve_under_inline_root(local_path)
    if resolved is None:
        return None
    try:
        if resolved.stat().st_size > MAX_ATTACHMENT_INLINE_BYTES:
            return None
        return resolved.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError, ValueError):
        return None


def prepare_run_attachment_blocks(
    attachments: List[Dict[str, Any]],
    already_inlined: Set[str],
    image_mode: str = "text",
) -> Tuple[List[str], Set[str], List[Tuple[str, str]]]:
    """Map the backend's materialized-attachment manifest to prompt context blocks.

    EVERY file gets a context note on EVERY run (reusing the platforms' document
    note builder): /v1/runs context is rebuilt per request from backend-clean
    history, so a note injected only once would vanish from turn 2 onward and
    the model would forget the file exists.

    Content INLINING is the deduped part: a text file's content (platforms'
    "[Content of X]:" shape) is inlined the first time this session surfaces it
    — answerable without a tool round-trip — and later runs carry the path note
    instead (the agent tool-reads on demand). An aggregate per-run budget
    (MAX_ATTACHMENT_INLINE_TOTAL_BYTES) bounds many-file manifests; files past
    the budget degrade to path notes and stay eligible for a later run. Binary,
    oversized, unreadable, or non-UTF-8 files get the self-extraction note (the
    instruction the platforms use so the model reads the file with its tools
    instead of punting back to the user).

    IMAGES (image_mode="native", i.e. the active model has vision): pixels are
    attached to the user turn as OpenAI-style image parts — once per session per
    file, at most MAX_ATTACHMENT_NATIVE_IMAGES_PER_RUN per run, and only for
    files under the inline root. Every other case (text-mode model, cap reached,
    already attached earlier, outside root) gets an image path note instructing
    tool-based viewing. Natively attached images get NO note block here —
    build_native_content_parts appends its own "[Image attached at: ...]" hint.

    Returns (blocks, newly_inlined_keys, native_images) where native_images is
    [(dedup_key, absolute_path)] for the caller to hand to
    build_native_content_parts. The CALLER must commit newly_inlined into the
    session's dedup set only after the run actually executes — marking at
    request time would let a pre-flight failure permanently swallow the content
    (the backend retries with the same manifest).
    """
    from gateway.run import _build_document_context_note

    blocks: List[str] = []
    newly_inlined: Set[str] = set()
    native_images: List[Tuple[str, str]] = []
    seen_this_run: Set[str] = set()
    inline_budget = MAX_ATTACHMENT_INLINE_TOTAL_BYTES
    for entry in attachments:
        key = _attachment_dedup_key(entry)
        if key in seen_this_run:
            continue
        seen_this_run.add(key)
        safe_name = str(entry.get("safe_name", ""))
        local_path = str(entry.get("local_path", ""))
        display_name = re.sub(r"[^\w.\- ]", "_", safe_name)
        mime = str(entry.get("mime") or "") or "application/octet-stream"
        # Present workspace paths RELATIVE to the workspace root: the agent's
        # tools run with the workspace as cwd, so `inputs/...` resolves — and the
        # note text flows into user-visible transcripts/events, where an absolute
        # sandbox path is internal detail (the platform-tests leak scrubber
        # rightly flags '/workspace/...' in public payloads). Paths outside the
        # root (non-managed callers) stay as sent.
        note_path = workspace_relative_path(local_path)

        if is_image_attachment(safe_name, mime):
            resolved = _resolve_under_inline_root(local_path)
            try:
                size_ok = resolved is not None and resolved.is_file() and resolved.stat().st_size <= MAX_ATTACHMENT_NATIVE_IMAGE_BYTES
            except OSError:
                size_ok = False
            if (
                image_mode == "native"
                and key not in already_inlined
                and len(native_images) < MAX_ATTACHMENT_NATIVE_IMAGES_PER_RUN
                and size_ok
            ):
                # The RESOLVED path: containment was checked on it, so the read
                # must use the same object (no symlink-swap window), matching how
                # text inlining reads.
                native_images.append((key, str(resolved)))
                newly_inlined.add(key)
            else:
                blocks.append(_build_image_attachment_note(display_name, note_path))
            continue

        inline_text: Optional[str] = None
        if key not in already_inlined and inline_budget > 0 and _is_text_attachment(safe_name, mime):
            candidate = _read_inline_attachment_text(local_path)
            if candidate is not None:
                candidate_bytes = len(candidate.encode("utf-8"))
                if candidate_bytes <= inline_budget:
                    inline_text = candidate
                    inline_budget -= candidate_bytes
                else:
                    logger.info(
                        "Run attachment %s (%d bytes) skipped inlining: per-run budget exhausted; path note only.",
                        display_name,
                        candidate_bytes,
                    )

        if inline_text is not None:
            newly_inlined.add(key)
            # Force a text/* mtype so the note says "content has been included
            # below" — which is now actually true.
            note_mime = mime if mime.startswith("text/") else "text/plain"
            blocks.append(_build_document_context_note(display_name, note_path, note_mime))
            blocks.append(f"[Content of {display_name}]:\n{inline_text}")
        else:
            # Path note only. Force the non-text wording: it instructs the agent
            # to extract the content itself with its tools, which is right for
            # binary files, for text files that failed the inline gate, AND for
            # files whose content was inlined in an earlier run (the note must
            # never claim content was included when it wasn't — in THIS message).
            blocks.append(_build_document_context_note(display_name, note_path, "application/octet-stream"))
    return blocks, newly_inlined, native_images


class RunAttachmentInlineDedup:
    """Per-session dedup of /v1/runs attachment content INLINING.

    Attachment INLINE dedup: session_id -> keys of files whose content was
    already inlined into a run that executed. Path notes are re-injected on
    every run (per-request context — see prepare_run_attachment_blocks);
    only the expensive content inlining happens once per session. In-memory:
    after a container restart a file is re-inlined once, which is harmless.
    """

    def __init__(self) -> None:
        self._inlined_run_attachments: Dict[str, Set[str]] = {}

    def for_session(self, session_id: str) -> Set[str]:
        """The dedup set of attachment keys whose content was already inlined
        into an executed run of this session.

        Bounded: when the map exceeds MAX_ATTACHMENT_DEDUP_SESSIONS, the oldest
        sessions (insertion order) are dropped — re-inlining a file once in a
        very old resumed session is harmless, unbounded growth is not."""
        inlined = self._inlined_run_attachments.get(session_id)
        if inlined is None:
            while len(self._inlined_run_attachments) >= MAX_ATTACHMENT_DEDUP_SESSIONS:
                oldest = next(iter(self._inlined_run_attachments))
                del self._inlined_run_attachments[oldest]
            inlined = set()
            self._inlined_run_attachments[session_id] = inlined
        return inlined

    def commit(self, session_id: str, keys: Set[str]) -> None:
        """Commit newly-inlined keys AFTER the run actually executed.

        run_conversation executed, so the enriched message (incl. any inlined
        attachment content) reached the agent turn — only NOW is the inline
        dedup committed. Committing at request time would let a pre-flight
        failure (_create_agent throwing, early stop) permanently swallow the
        content for the session: the backend retries with the same manifest
        and dedup would skip it.
        """
        if keys:
            self.for_session(session_id).update(keys)


def managed_toolset_override(enabled_toolsets: List[str]) -> Tuple[List[str], List[str]]:
    """KarinAI managed-runtime tool policy for the api_server platform.

    In managed mode (``KARINAI_MANAGED_RUNTIME`` truthy) the runtime-manager's
    tool policy replaces the user-editable platform toolset config for private
    /v1/runs containers, returning ``(enabled, disabled)``. Outside managed
    mode the caller's resolved toolsets pass through unchanged with an empty
    disabled list.
    """
    from karinai.runtime.config import parse_bool

    if not parse_bool(os.getenv("KARINAI_MANAGED_RUNTIME")):
        return enabled_toolsets, []
    from karinai.runtime.managed import load_managed_runtime_config, managed_agent_toolsets

    managed_cfg = load_managed_runtime_config()
    return managed_agent_toolsets(managed_cfg)


def parse_app_tool_gateway(body: Dict[str, Any]) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """Validate body["app_tool_gateway"] (run-scoped app-tool gateway creds).

    Returns ``(parsed_or_None, error_or_None)`` — the adapter owns HTTP, so
    the error is a message string for its 400 path, not a response object.
    """
    raw_app_tool_gateway = body.get("app_tool_gateway")
    if raw_app_tool_gateway is None:
        return None, None
    if not isinstance(raw_app_tool_gateway, dict):
        return None, "'app_tool_gateway' must be an object"
    gateway_url = str(raw_app_tool_gateway.get("url") or "").strip()
    gateway_token = str(raw_app_tool_gateway.get("token") or "").strip()
    gateway_expires_at = str(raw_app_tool_gateway.get("expires_at") or "").strip()
    if not gateway_url or not gateway_token:
        return None, "'app_tool_gateway' requires url and token fields"
    return {
        "url": gateway_url,
        "token": gateway_token,
        "expires_at": gateway_expires_at,
    }, None


def enrich_run_user_message(
    body: Dict[str, Any],
    user_message: Any,
    dedup: RunAttachmentInlineDedup,
    session_id: str,
) -> Tuple[Any, Set[str], Optional[str]]:
    """KarinAI managed runs: surface backend-materialized conversation files
    to the agent. The backend confirms each file on disk before dispatch and
    sends the manifest in body["attachments"]; without this step the agent is
    never told the files exist (see prepare_run_attachment_blocks). Notes are
    injected on EVERY run (per-request context); content inlining is deduped
    per session — the CALLER commits the returned key set via
    ``dedup.commit(session_id, keys)`` only after the run executes.

    Returns ``(user_message, newly_inlined_keys, error_or_None)``. On a
    manifest-validation error the message is returned unmodified and the
    adapter owns the 400 response.
    """
    newly_inlined_attachments: Set[str] = set()
    raw_attachments = body.get("attachments")
    if not raw_attachments:
        return user_message, newly_inlined_attachments, None
    attachments_error = validate_run_attachments(raw_attachments)
    if attachments_error:
        return user_message, newly_inlined_attachments, attachments_error
    has_images = any(
        is_image_attachment(str(entry.get("safe_name", "")), str(entry.get("mime") or ""))
        for entry in raw_attachments
    )
    attachment_blocks, newly_inlined_attachments, native_images = prepare_run_attachment_blocks(
        raw_attachments,
        dedup.for_session(session_id),
        image_mode=decide_run_image_mode() if has_images else "text",
    )
    if attachment_blocks:
        context_block = "\n\n".join(attachment_blocks)
        if isinstance(user_message, list):
            # Content-parts input: prepend the context as a text part.
            user_message = [{"type": "text", "text": context_block}, *user_message]
        else:
            user_message = f"{context_block}\n\n{user_message}"
    if native_images:
        # Vision model: attach the pixels to the user turn as OpenAI-style
        # image parts, plus a "[Image attached at: <relative path>]" hint
        # per image (same handle the platforms' native path provides, so
        # tools can be invoked on the file without a round-trip). Parts
        # are built directly — build_native_content_parts fabricates a
        # default caption for empty text, which must not be injected into
        # a content-parts user turn.
        from agent.image_routing import _file_to_data_url

        image_parts: List[Dict[str, Any]] = []
        hint_lines: List[str] = []
        for img_key, img_path in native_images:
            data_url = _file_to_data_url(Path(img_path))
            if data_url is None:
                # Neither pixels nor a note this run; un-mark so a later
                # run re-surfaces the file (self-healing).
                newly_inlined_attachments.discard(img_key)
                continue
            image_parts.append({"type": "image_url", "image_url": {"url": data_url}})
            hint_lines.append(f"[Image attached at: {workspace_relative_path(img_path)}]")
        if image_parts:
            hints = "\n".join(hint_lines)
            if isinstance(user_message, list):
                user_message = [*user_message, {"type": "text", "text": hints}, *image_parts]
            else:
                user_message = [
                    {"type": "text", "text": f"{user_message}\n\n{hints}"},
                    *image_parts,
                ]
    return user_message, newly_inlined_attachments, None
