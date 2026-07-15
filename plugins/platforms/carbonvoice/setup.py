"""Hermes plugin registration entry points.

Everything Hermes calls during plugin discovery lives here:

    check_requirements   — verify import-time deps (httpx mandatory, socketio optional)
    validate_config      — runtime check that PAT is present
    is_connected         — quick "is this plugin usable?" probe
    _env_enablement      — seed PlatformConfig.extra from environment
    interactive_setup    — terminal wizard for ``hermes setup``
    register             — wire everything into the gateway plugin registry
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from gateway.config import PlatformConfig

from .adapter import CarbonVoiceAdapter
from .api import standalone_send
from .constants import DEFAULT_BASE_URL, DEFAULT_POLL_INTERVAL_MS, DEFAULT_WS_RETRY_MAX_MS

logger = logging.getLogger(__name__)

try:
    import httpx  # noqa: F401
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import socketio  # noqa: F401
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False


def check_requirements() -> bool:
    if not HTTPX_AVAILABLE:
        logger.error("carbonvoice: httpx not installed")
        return False
    if not SOCKETIO_AVAILABLE:
        logger.warning(
            "carbonvoice: python-socketio not installed — running in polling-only mode "
            "(install with: pip install 'python-socketio[asyncio_client]')"
        )
    return True


def validate_config(config: PlatformConfig) -> bool:
    extra = getattr(config, "extra", {}) or {}
    pat = os.getenv("CARBONVOICE_PAT") or config.token or extra.get("pat", "")
    return bool(pat)


def is_connected(config: PlatformConfig) -> bool:
    extra = getattr(config, "extra", {}) or {}
    return bool(config.token or extra.get("pat"))


def _bool_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("true", "1", "yes", "on")


def _env_enablement() -> Optional[Dict[str, Any]]:
    """Seed PlatformConfig.extra from env vars before adapter construction.

    Returns a *flat* dict — Hermes core merges this into ``extra`` via
    ``config.platforms[platform].extra.update(seed)``, so nesting here would
    end up as ``extra["extra"]`` and the keys would never reach the adapter.
    """
    pat = os.getenv("CARBONVOICE_PAT")
    if not pat:
        return None

    seed: Dict[str, Any] = {
        "pat": pat,
        "base_url": (os.getenv("CARBONVOICE_BASE_URL") or DEFAULT_BASE_URL).rstrip("/"),
        "poll_interval_ms": int(
            os.getenv("CARBONVOICE_POLL_INTERVAL_MS") or DEFAULT_POLL_INTERVAL_MS
        ),
        "ws_retry_max_ms": int(
            os.getenv("CARBONVOICE_WS_RETRY_MAX_MS") or DEFAULT_WS_RETRY_MAX_MS
        ),
        "creator_id": os.getenv("CARBONVOICE_CREATOR_ID") or None,
        "state_path": os.getenv("CARBONVOICE_STATE_PATH") or None,
        "reaction_id": os.getenv("CARBONVOICE_REACTION_ID") or None,
        "disable_ack_reaction": _bool_env("CARBONVOICE_DISABLE_ACK_REACTION"),
        "disable_mark_read": _bool_env("CARBONVOICE_DISABLE_MARK_READ"),
        "ignored_senders_log": os.getenv("CARBONVOICE_IGNORED_SENDERS_LOG") or None,
        # When true, every inbound MessageEvent is marked as
        # ``MessageType.VOICE`` regardless of whether the user typed or
        # spoke. This unlocks Hermes core's auto-TTS path
        # (``base.py:3493``) and the ``voice_mode`` dispatch
        # (``run.py:11142``), so the agent's text reply is auto-
        # converted to audio and sent via :meth:`send_voice` →
        # ``/v5/messages/audio``. Carbon Voice transcribes the audio
        # server-side, so the recipient sees a voice memo bubble with
        # transcript — the symmetric voice-first experience expected on
        # this platform.
        #
        # Required companion config on the operator side:
        #   - ``voice.auto_tts: true`` in ``config.yaml`` (or use
        #     ``/voice on`` per chat if Hermes core ever wires slash
        #     commands for CV)
        #   - A configured TTS provider in ``config.yaml`` under
        #     ``tts.provider`` (``edge`` works with no API key)
        "voice_out": _bool_env("CARBONVOICE_VOICE_OUT"),
        # PR 7 — inbound multimodal: max attachment size (MB) the
        # adapter will download from CV and forward to Hermes core's
        # multimodal pipeline. Anything larger is logged and skipped
        # (the agent still sees the text part of the message; only the
        # attachment is dropped). Default 10 MB balances Claude /
        # OpenAI vision recommendations against typical CV usage.
        "max_attachment_mb": int(
            os.getenv("CARBONVOICE_MAX_ATTACHMENT_MB") or "10"
        ),
    }

    # CARBONVOICE_SHARED_GROUP_SESSIONS=true → flip
    # ``group_sessions_per_user`` to False so every participant in a
    # group channel shares one session, *regardless of thread_id*. Use
    # for bot-room channels where strict per-user isolation isn't
    # wanted. Default behavior already shares sessions within a thread
    # (because SessionSource.thread_id is now populated for groups);
    # this knob extends sharing to non-threaded conversations too.
    # See DEVELOPMENT.md §7.4 / §7.9 for the design rationale.
    if _bool_env("CARBONVOICE_SHARED_GROUP_SESSIONS"):
        seed["group_sessions_per_user"] = False

    home_channel_id = os.getenv("CARBONVOICE_HOME_CHANNEL")
    if home_channel_id:
        seed["home_channel"] = {
            "chat_id": home_channel_id,
            "name": os.getenv("CARBONVOICE_HOME_CHANNEL_NAME") or home_channel_id,
        }
    return seed


def interactive_setup() -> Optional[Dict[str, str]]:
    """Lightweight wizard: gather PAT."""
    try:
        pat = input("Carbon Voice PAT (cv_pat_...): ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not pat:
        return None
    return {"CARBONVOICE_PAT": pat}


async def _standalone_send(
    pconfig: PlatformConfig,
    chat_id: str,
    content: str,
    **_kwargs: Any,
) -> Dict[str, Any]:
    """Adapter for Hermes' cron delivery hook — unwraps PlatformConfig and calls api.standalone_send."""
    extra = pconfig.extra or {}
    pat = pconfig.token or extra.get("pat")
    base_url = (extra.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
    if not pat:
        return {"success": False, "error": "missing CARBONVOICE_PAT"}
    return await standalone_send(pat, base_url, chat_id, content)


def register(ctx) -> None:
    """Called by the Hermes plugin system on discovery."""
    # DENY-BY-DEFAULT (security): we no longer force
    # CARBONVOICE_ALLOW_ALL_USERS=true. Previously this opened both our
    # AllowlistGate and Hermes core's parallel check to everyone — the hole
    # this whole feature closes. Now: the adapter authorizes the owner
    # (whoami.created_by) and mirrors them + any /cv-allow approvals into
    # core's PairingStore (which core's own check always consults), so the
    # gate stays closed without forcing allow-all. The operator opts back
    # into open access explicitly with CARBONVOICE_ALLOW_ALL_USERS=true.

    ctx.register_platform(
        name="carbonvoice",
        label="Carbon Voice",
        adapter_factory=lambda cfg: CarbonVoiceAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["CARBONVOICE_PAT"],
        install_hint=(
            "pip install httpx 'python-socketio[asyncio_client]' "
            "(python-socketio is optional — polling-only without it)"
        ),
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="CARBONVOICE_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="CARBONVOICE_ALLOWED_USERS",
        allow_all_env="CARBONVOICE_ALLOW_ALL_USERS",
        max_message_length=8000,
        emoji="🎙️",
        allow_update_command=True,
        platform_hint=(
            "You are chatting via Carbon Voice.\n\n"
            "## Sending file attachments — ALWAYS use the MEDIA: directive\n"
            "To attach ANY file (.md, .pdf, .png, audio, etc.) to your reply, "
            "include this exact line in your response text:\n\n"
            "    MEDIA:/absolute/path/to/file.ext\n\n"
            "Example reply that sends a markdown report:\n\n"
            "    Aquí está el resumen.\n"
            "    MEDIA:/tmp/report.md\n\n"
            "The plugin parses that line out, uploads the file to Carbon "
            "Voice's S3 storage, and attaches it to the message — the user "
            "sees your text + a downloadable file in one bubble.\n\n"
            "Routing by extension:\n"
            "- .wav, .mp3, .opus, .m4a, .ogg, .flac → sent as a voice memo "
            "(Carbon Voice transcribes server-side, user gets a play button "
            "with transcript)\n"
            "- Everything else (.md, .pdf, .png, .jpg, .zip, ...) → sent as a "
            "downloadable native attachment\n\n"
            "Allowed paths: the operator's HERMES_MEDIA_ALLOW_DIRS plus the "
            "default ~/.hermes/{document,audio,image,video}_cache roots. If "
            "unsure where to write, use ~/.hermes/document_cache/ for text "
            "and PDFs, ~/.hermes/audio_cache/ for audio.\n\n"
            "## Anti-patterns — DO NOT do these\n"
            "- DO NOT call the send_message tool to attach files in THIS "
            "conversation. send_message is for cross-channel proactive "
            "sends to OTHER platforms / channels. For attachments in the "
            "current Carbon Voice chat, the MEDIA: directive is the only "
            "correct path.\n"
            "- DO NOT just describe the file's path in prose ('the file is "
            "at /tmp/foo.md'); that ships as plain text, not an attachment. "
            "You must emit the literal `MEDIA:/tmp/foo.md` line.\n\n"
            "## Voice-out (auto-TTS)\n"
            "If the operator has enabled voice-out for this conversation "
            "(env ``CARBONVOICE_VOICE_OUT=true`` + ``voice.auto_tts: true`` "
            "in config), Hermes core automatically converts your text "
            "reply into a voice memo via a TTS provider and ships it via "
            "the audio endpoint — Carbon Voice then transcribes the audio "
            "server-side so the user sees a voice-memo bubble with the "
            "transcript inline. You don't need to call any TTS tool or "
            "emit MEDIA: for an audio path — just write your reply as "
            "text and Hermes handles the conversion.\n\n"
            "When voice-out is active, optimize the reply for spoken "
            "delivery:\n"
            "- Conversational tone, short sentences\n"
            "- Avoid markdown that doesn't translate to speech (no "
            "bullet lists with bare ``-``, no tables, no code fences — "
            "they'll be read aloud verbatim and sound awful)\n"
            "- Spell out symbols you'd expect read literally (use "
            "\"and\" instead of ``&``, \"percent\" instead of ``%``)\n"
            "- Keep under ~30 seconds of audio (~120 words) unless the "
            "user explicitly asked for a long answer; long voice memos "
            "are skimmed, not listened to\n\n"
            "If you specifically need to send code, JSON, a table, or "
            "any structured artifact in this conversation, attach it as "
            "a file via the MEDIA: directive instead — that keeps the "
            "voice-memo bubble short and the artifact downloadable.\n\n"
            "## Inbound attachments (multimodal)\n"
            "Two attachment types are wired through to you:\n"
            "- **Images** (.jpg/.png/.webp/...): downloaded and "
            "surfaced via Claude vision — reference what you see "
            "naturally (\"in the screenshot you sent\", \"the photo "
            "shows...\") without asking the user to describe it.\n"
            "- **Links** (CV's link-share UI): the URL is prepended "
            "to the user's message text as a line like ``[Attached "
            "link: https://...]``. Use your existing browser / fetch "
            "tools (``browser_navigate``, ``fetch_url``, etc.) to "
            "open it the same way you would for any URL the user "
            "types inline.\n\n"
            "Other attachment types are NOT delivered to you in this "
            "plugin version: PDFs, text files, code files, archives, "
            "and audio attachments arrive only as a notice in the "
            "logs — the text part of the user's message still reaches "
            "you, but the file contents do not. If the user attaches "
            "something other than an image / link and asks about its "
            "contents, tell them honestly that you only received the "
            "image / link / text and don't have the file body. Do NOT "
            "reach for ``terminal``, ``execute_code``, or ``read_file`` "
            "to try to extract the file yourself — those tools require "
            "operator approval and produce a worse UX than just "
            "being upfront about the limitation.\n\n"
            "Voice memos arrive as transcript (server-side STT by CV) "
            "in the text channel as usual.\n\n"
            "## Other notes\n"
            "Carbon Voice transcribes inbound voice → text before "
            "delivery. Plain text and lightweight markdown render best "
            "for text-mode replies — avoid complex tables, multi-column "
            "layouts, or raw HTML. Keep responses conversational and "
            "concise either way."
        ),
    )
