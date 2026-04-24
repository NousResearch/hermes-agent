# -*- coding: utf-8 -*-
"""QQ Bot attachment downloading and processing pipeline.

Provides three composable classes:

- :class:`AttachmentDownloader` — downloads CDN URLs to a local cache directory.
- :class:`STTPipeline` — transcribes voice attachments via a configurable STT backend.
- :class:`AttachmentProcessor` — orchestrates the full attachment pipeline,
  returning a list of :class:`ProcessedAttachment` for each inbound message.

All classes are **injected** with their dependencies at construction time,
carrying zero hermes-specific imports.

Usage (hermes adapter)::

    downloader = AttachmentDownloader(
        http_client=httpx_client,
        cache_dir="/path/to/cache",
        media_headers_fn=adapter.media_headers,
    )
    stt = STTPipeline(
        http_client=httpx_client,
        stt_config_fn=lambda: resolve_stt_config(config.extra),
        log_tag="QQBot",
    )
    processor = AttachmentProcessor(downloader=downloader, stt_pipeline=stt)
    results = await processor.process(attachments)
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import unquote, urlparse

from .audio import (
    call_stt,
    convert_audio_to_wav,
    guess_audio_ext,
    is_voice_content_type,
    resolve_stt_config,
)
from .dto import MessageAttachment

logger = logging.getLogger(__name__)


# ── Result type ───────────────────────────────────────────────────────

@dataclass
class ProcessedAttachment:
    """Processed attachment descriptor (no hermes types).

    :param kind: ``'image'`` | ``'voice'`` | ``'video'`` | ``'document'``.
    :param local_path: Absolute path to the locally cached file.
    :param content_type: MIME type of the attachment.
    :param transcript: Voice transcript (populated when ``kind='voice'``).
    :param description: Human-readable description for non-image attachments.
    """

    kind: str
    local_path: str
    content_type: str
    transcript: str = ""
    description: str = ""


# ── AttachmentDownloader ─────────────────────────────────────────────

class AttachmentDownloader:
    """Download QQ CDN URLs to a local cache directory.

    Dependency-injected: requires only an HTTP client, a cache directory
    path, and an optional callable that returns authentication headers.

    :param http_client: Any async HTTP client with ``.get()`` method.
    :param cache_dir: Root directory for cached files.
    :param media_headers_fn: Callable returning auth headers for CDN requests.
    :param log_tag: Log prefix.
    """

    def __init__(
        self,
        http_client: Any,
        cache_dir: str,
        media_headers_fn: Optional[Callable[[], Dict[str, str]]] = None,
        log_tag: str = "QQBot",
    ) -> None:
        self._http_client = http_client
        self._cache_dir = Path(cache_dir)
        self._media_headers_fn = media_headers_fn or (lambda: {})
        self._log_tag = log_tag

    def update_http_client(self, http_client: Any) -> None:
        """Replace the HTTP client (called after connect())."""
        self._http_client = http_client

    async def download_image(self, url: str, content_type: str) -> Optional[str]:
        """Download an image URL using MD5-based deduplication.

        :param url: CDN URL to download.
        :param content_type: MIME type (used to derive file extension).
        :returns: Local file path, or ``None`` on failure.
        """
        if not self._is_safe_url(url):
            logger.warning("[%s] Blocked unsafe image URL: %s", self._log_tag, url[:80])
            return None
        ext = mimetypes.guess_extension(content_type) or ".jpg"
        url_hash = hashlib.md5(url.encode()).hexdigest()[:16]
        cached_path = self._cache_dir / f"img_{url_hash}{ext}"

        if cached_path.exists():
            logger.debug("[%s] Image cache hit: %s", self._log_tag, cached_path.name)
            return str(cached_path)

        data = await self._fetch(url)
        if data is None:
            return None

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cached_path.write_bytes(data)
        logger.debug("[%s] Image cached: %s (%d bytes)", self._log_tag, cached_path.name, len(data))
        return str(cached_path)

    async def download_audio(self, url: str, filename: str = "") -> Optional[str]:
        """Download an audio URL and convert to WAV.

        No deduplication (audio is converted to a new format on each download).

        :param url: CDN URL or pre-converted WAV URL.
        :param filename: Hint for extension detection.
        :returns: Local WAV file path, or ``None`` on failure.
        """
        if not self._is_safe_url(url):
            logger.warning("[%s] Blocked unsafe audio URL: %s", self._log_tag, url[:80])
            return None

        data = await self._fetch(url)
        if data is None:
            return None

        return await self._convert_and_cache_audio(data, filename or url)

    async def download_document(self, url: str, original_name: str = "") -> Optional[str]:
        """Download a document/file URL with MD5-based deduplication.

        :param url: CDN URL to download.
        :param original_name: Original filename for naming the cached file.
        :returns: Local file path, or ``None`` on failure.
        """
        if not self._is_safe_url(url):
            logger.warning("[%s] Blocked unsafe document URL: %s", self._log_tag, url[:80])
            return None

        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        name = original_name or _extract_filename_from_url(url) or "qq_attachment"
        cached_path = self._cache_dir / f"doc_{url_hash}_{name}"

        if cached_path.exists():
            logger.debug("[%s] Document cache hit: %s", self._log_tag, cached_path.name)
            return str(cached_path)

        data = await self._fetch(url)
        if data is None:
            return None

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cached_path.write_bytes(data)
        return str(cached_path)

    async def download(self, url: str, content_type: str) -> Optional[str]:
        """Dispatch to the correct download method based on *content_type*.

        :param url: CDN URL to download.
        :param content_type: MIME type used for routing.
        :returns: Local file path, or ``None`` on failure.
        """
        ct = content_type.lower()
        if ct.startswith("image/"):
            return await self.download_image(url, content_type)
        if ct.startswith("audio/") or ct == "voice":
            return await self.download_audio(url)
        return await self.download_document(url)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch(self, url: str) -> Optional[bytes]:
        """Perform an HTTP GET and return raw bytes."""
        if not self._http_client:
            logger.warning("[%s] No HTTP client available", self._log_tag)
            return None
        try:
            resp = await self._http_client.get(
                url,
                timeout=30.0,
                headers=self._media_headers_fn(),
                follow_redirects=True,
            )
            resp.raise_for_status()
            return resp.content
        except Exception as exc:
            logger.debug("[%s] Download failed for %s: %s", self._log_tag, url[:80], exc)
            return None

    async def _convert_and_cache_audio(
        self,
        audio_data: bytes,
        source_hint: str,
    ) -> Optional[str]:
        """Convert audio bytes to WAV and write to cache."""
        wav_path = await convert_audio_to_wav(
            audio_data,
            source_hint=source_hint,
            log_tag=self._log_tag,
        )
        if not wav_path:
            ext = guess_audio_ext(audio_data)
            return self._write_to_cache(audio_data, ext)

        try:
            wav_data = Path(wav_path).read_bytes()
            os.unlink(wav_path)
            return self._write_to_cache(wav_data, ".wav")
        except Exception as exc:
            logger.debug("[%s] Failed to read converted WAV: %s", self._log_tag, exc)
            return None

    def _write_to_cache(self, data: bytes, ext: str) -> str:
        """Write *data* to a uniquely named file in the cache directory."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=ext,
            dir=self._cache_dir,
            delete=False,
        ) as tmp:
            tmp.write(data)
            return tmp.name

    @staticmethod
    def _is_safe_url(url: str) -> bool:
        """Check URL safety (basic scheme validation).

        The hermes adapter may override this by injecting a stricter check
        via a subclass or by pre-validating URLs before passing them in.
        """
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)


# ── STTPipeline ───────────────────────────────────────────────────────

class STTPipeline:
    """Voice-to-text transcription pipeline.

    Encapsulates the four-step process:

    1. Use QQ's built-in ``asr_refer_text`` if available (free, no API call).
    2. Resolve the download URL (prefer ``voice_wav_url`` — already WAV).
    3. Download and convert audio to WAV.
    4. Call the configured STT API.

    Each step is a separate method (CC ≤ 5 each).

    :param http_client: Async HTTP client for downloading audio and calling STT.
    :param stt_config_fn: Callable returning the STT config dict, or ``None``
        if STT is not configured.
    :param log_tag: Log prefix.
    """

    def __init__(
        self,
        http_client: Any,
        stt_config_fn: Callable[[], Optional[Dict[str, str]]],
        downloader: AttachmentDownloader,
        log_tag: str = "QQBot",
    ) -> None:
        self._http_client = http_client
        self._stt_config_fn = stt_config_fn
        self._downloader = downloader
        self._log_tag = log_tag

    def update_http_client(self, http_client: Any) -> None:
        """Replace the HTTP client (called after connect())."""
        self._http_client = http_client

    async def transcribe(self, att: MessageAttachment) -> Optional[str]:
        """Transcribe a voice attachment to text.

        :param att: The voice :class:`~dto.MessageAttachment`.
        :returns: Transcript string, or ``None`` if transcription fails.
        """
        # Step 1: built-in ASR (free, always preferred)
        builtin = self._use_builtin_asr(att)
        if builtin is not None:
            return builtin

        # Step 2: resolve download URL
        url = self._resolve_download_url(att)
        if not url:
            return None

        # Step 3: download + convert to WAV
        wav_path = await self._download_and_convert(url, att.filename)
        if not wav_path:
            return None

        # Step 4: call STT API
        return await self._call_stt_api(wav_path)

    def _use_builtin_asr(self, att: MessageAttachment) -> Optional[str]:
        """Return QQ's built-in ASR text if available."""
        text = att.asr_refer_text.strip()
        if text:
            logger.debug("[%s] STT: using QQ asr_refer_text", self._log_tag)
            return text
        return None

    def _resolve_download_url(self, att: MessageAttachment) -> Optional[str]:
        """Return the best URL to download audio from.

        Prefers ``voice_wav_url`` (pre-converted WAV) over the raw URL.
        """
        if att.voice_wav_url.strip():
            wav_url = att.voice_wav_url.strip()
            if wav_url.startswith("//"):
                wav_url = f"https:{wav_url}"
            logger.debug("[%s] STT: using voice_wav_url (pre-converted WAV)", self._log_tag)
            return wav_url
        url = att.resolved_url
        if not url:
            logger.warning("[%s] STT: attachment has no downloadable URL", self._log_tag)
        return url or None

    async def _download_and_convert(self, url: str, filename: str) -> Optional[str]:
        """Download audio and convert to WAV; returns local WAV path."""
        wav_path = await self._downloader.download_audio(url, filename)
        if not wav_path:
            logger.warning("[%s] STT: audio download/conversion failed", self._log_tag)
        return wav_path

    async def _call_stt_api(self, wav_path: str) -> Optional[str]:
        """Call the configured STT API and clean up the temp file."""
        stt_cfg = self._stt_config_fn()
        if not stt_cfg:
            logger.warning(
                "[%s] STT not configured (no stt config or QQ_STT_API_KEY)",
                self._log_tag,
            )
            self._cleanup(wav_path)
            return None

        try:
            transcript = await call_stt(
                self._http_client,
                wav_path,
                stt_cfg,
                self._log_tag,
            )
        finally:
            self._cleanup(wav_path)

        if transcript:
            logger.debug("[%s] STT success: %r", self._log_tag, transcript[:100])
        else:
            logger.warning("[%s] STT: API returned empty transcript", self._log_tag)
        return transcript

    @staticmethod
    def _cleanup(path: str) -> None:
        try:
            os.unlink(path)
        except OSError:
            pass


# ── AttachmentProcessor ───────────────────────────────────────────────

class AttachmentProcessor:
    """Process a list of :class:`~dto.MessageAttachment` objects.

    Dispatches each attachment to the appropriate handler based on content
    type, returning a flat list of :class:`ProcessedAttachment` results.

    :param downloader: :class:`AttachmentDownloader` instance.
    :param stt_pipeline: :class:`STTPipeline` instance (optional; voice
        attachments are skipped if ``None``).
    """

    def __init__(
        self,
        downloader: AttachmentDownloader,
        stt_pipeline: Optional[STTPipeline] = None,
    ) -> None:
        self._downloader = downloader
        self._stt = stt_pipeline

    async def process(
        self,
        attachments: List[MessageAttachment],
    ) -> List[ProcessedAttachment]:
        """Process all attachments and return results.

        :param attachments: List of inbound :class:`~dto.MessageAttachment`.
        :returns: List of :class:`ProcessedAttachment` (one per processed item).
        """
        results: List[ProcessedAttachment] = []
        for att in attachments:
            url = att.resolved_url
            if not url:
                continue
            ct = att.content_type.strip().lower()
            result = await self._process_one(att, url, ct)
            if result is not None:
                results.append(result)
        return results

    async def _process_one(
        self,
        att: MessageAttachment,
        url: str,
        ct: str,
    ) -> Optional[ProcessedAttachment]:
        """Dispatch a single attachment to the correct handler."""
        if is_voice_content_type(ct, att.filename):
            return await self._handle_voice(att)
        if ct.startswith("image/"):
            return await self._handle_image(url, ct)
        if ct.startswith("video/"):
            return await self._handle_video(url, ct, att.filename)
        return await self._handle_document(url, ct, att.filename)

    async def _handle_voice(
        self,
        att: MessageAttachment,
    ) -> Optional[ProcessedAttachment]:
        if self._stt is None:
            return None
        transcript = await self._stt.transcribe(att)
        if transcript:
            return ProcessedAttachment(
                kind="voice",
                local_path="",
                content_type=att.content_type,
                transcript=transcript,
            )
        return ProcessedAttachment(
            kind="voice",
            local_path="",
            content_type=att.content_type,
            transcript="[语音识别失败]",
        )

    async def _handle_image(
        self,
        url: str,
        ct: str,
    ) -> Optional[ProcessedAttachment]:
        local_path = await self._downloader.download_image(url, ct)
        if not local_path:
            return None
        return ProcessedAttachment(kind="image", local_path=local_path, content_type=ct)

    async def _handle_video(
        self,
        url: str,
        ct: str,
        filename: str,
    ) -> Optional[ProcessedAttachment]:
        local_path = await self._downloader.download_document(url, filename)
        desc = f"[video: {filename} ({local_path})]" if local_path else f"[video: {filename}]"
        return ProcessedAttachment(
            kind="video",
            local_path=local_path or "",
            content_type=ct,
            description=desc,
        )

    async def _handle_document(
        self,
        url: str,
        ct: str,
        filename: str,
    ) -> Optional[ProcessedAttachment]:
        local_path = await self._downloader.download_document(url, filename)
        name = filename or ct
        desc = f"[file: {name} ({local_path})]" if local_path else f"[file: {name}]"
        return ProcessedAttachment(
            kind="document",
            local_path=local_path or "",
            content_type=ct,
            description=desc,
        )


# ── Helpers ───────────────────────────────────────────────────────────

def _extract_filename_from_url(url: str) -> str:
    """Extract the filename component from a URL path."""
    try:
        return Path(unquote(urlparse(url).path)).name
    except Exception:
        return ""
