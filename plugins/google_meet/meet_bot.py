"""Headless Google Meet bot — Playwright + live-caption scraping.

Runs as a standalone subprocess spawned by ``process_manager.py``. Reads config
from env vars, writes status + transcript to files under
``$HERMES_HOME/workspace/meetings/<meeting-id>/``. The main hermes process
reads those files via the ``meet_*`` tools — no IPC beyond filesystem.

The scraping strategy mirrors OpenUtter (sumansid/openutter): we don't parse
WebRTC audio, we enable Google Meet's built-in live captions and observe the
captions container in the DOM via a MutationObserver. This is lossy and
English-biased but it is:

* deterministic (no API keys, no STT billing),
* works behind Meet's normal login / admission,
* survives Meet UI rewrites fairly well because the caption container has a
  stable ARIA role.

Run standalone for debugging::

    HERMES_MEET_URL=https://meet.google.com/abc-defg-hij \\
    HERMES_MEET_OUT_DIR=/tmp/meet-debug \\
    HERMES_MEET_HEADED=1 \\
    python -m plugins.google_meet.meet_bot

No meet.google.com URL → exits non-zero. Any URL that doesn't start with
``https://meet.google.com/`` is rejected (explicit-by-design).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import sys
import threading
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

# Match ``https://meet.google.com/abc-defg-hij`` or ``.../lookup/...`` — the
# short three-segment code or a lookup URL. Anything else is rejected.
MEET_URL_RE = re.compile(
    r"^https://meet\.google\.com/("
    r"[a-z0-9]{3,}-[a-z0-9]{3,}-[a-z0-9]{3,}"
    r"|lookup/[^/?#]+"
    r"|new"
    r")(?:[/?#].*)?$"
)


# Filenames the bot reads/writes in ``HERMES_MEET_OUT_DIR``.
SAY_QUEUE_FILENAME = "say_queue.jsonl"
SAY_PCM_FILENAME = "speaker.pcm"
CALL_ERROR_STRIKE_LIMIT = 3
MAX_TRANSCRIPT_TEXT_LEN = 500
MEET_MEDIA_PROXY_BYPASS = "74.125.250.0/24,74.125.247.128,142.250.82.0/24"
MEET_WEBRTC_PROXY_POLICY = "--force-webrtc-ip-handling-policy=disable_non_proxied_udp"


def _debug_status_enabled() -> bool:
    return os.environ.get("HERMES_MEET_DEBUG_STATUS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _is_safe_meet_url(url: str) -> bool:
    """Return True if *url* is a Google Meet URL we're willing to navigate to."""
    if not isinstance(url, str):
        return False
    return bool(MEET_URL_RE.match(url.strip()))


def _meeting_id_from_url(url: str) -> str:
    """Extract the 3-segment meeting code from a Meet URL.

    For ``https://meet.google.com/abc-defg-hij`` → ``abc-defg-hij``.
    For ``.../lookup/<id>`` or ``/new`` we fall back to a timestamped id — the
    bot won't know the real code until after redirect, and callers pass this
    through to filename anyway.
    """
    m = re.search(
        r"meet\.google\.com/([a-z0-9]{3,}-[a-z0-9]{3,}-[a-z0-9]{3,})",
        url or "",
    )
    if m:
        return m.group(1)
    return f"meet-{int(time.time())}"


# ---------------------------------------------------------------------------
# Status + transcript file writers
# ---------------------------------------------------------------------------

class _BotState:
    """Single-process mutable state, flushed to ``status.json`` on each change."""

    def __init__(self, out_dir: Path, meeting_id: str, url: str):
        self.out_dir = out_dir
        self.meeting_id = meeting_id
        self.url = url
        self.in_call = False
        self.captioning = False
        self.captions_enabled_attempted = False
        self.lobby_waiting = False
        # Consecutive observations of Meet's transient "couldn't start the
        # video call" banner. Reset whenever a probe no longer reports it; the
        # bot only exits once it persists past the strike limit.
        self.call_error_strikes = 0
        # Sticky: set once the UI looks admitted. It is useful progress, but is
        # not by itself enough to prove a healthy call because Meet can briefly
        # expose roster text while returning to an error/landing page.
        self.ever_admitted = False
        self.join_attempted_at: Optional[float] = None
        self.joined_at: Optional[float] = None
        self.last_caption_at: Optional[float] = None
        self.transcript_lines = 0
        self.error: Optional[str] = None
        self.exited = False
        now = time.time()
        self.phase = "starting"
        self.last_heartbeat_at = now
        self.last_progress_at = now
        self.stalled_reason: Optional[str] = None
        self.last_ui_text: Optional[str] = None
        self.last_url: Optional[str] = None
        self.last_speaker_source: Optional[str] = None
        self.last_speaker_candidates: list = []
        self.local_microphone_on: Optional[bool] = None
        self.local_camera_on: Optional[bool] = None
        # v2 realtime fields.
        self.realtime = False
        self.realtime_ready = False
        self.realtime_device: Optional[str] = None
        self.realtime_audio_pump_status = "disabled"
        self.realtime_audio_pump_tool: Optional[str] = None
        self.realtime_audio_pump_pid: Optional[int] = None
        self.realtime_audio_pump_return_code: Optional[int] = None
        self.realtime_audio_pump_error: Optional[str] = None
        self.audio_bytes_out: int = 0
        self.last_audio_out_at: Optional[float] = None
        self.last_barge_in_at: Optional[float] = None
        self.leave_reason: Optional[str] = None
        self.unresolved_caption_drops = 0
        self.unresolved_caption_lines = 0
        self.caption_ui_noise_drops = 0
        self.last_unresolved_caption_at: Optional[float] = None
        # Scraped captions, in order, deduped. Each entry is a dict of
        # {"ts": <epoch>, "speaker": str, "text": str}.
        self._seen: set = set()
        self._transcript_entries: list[tuple[str, str, str, str]] = []
        self._caption_groups: list[dict] = []
        self._caption_group_by_key: dict[str, dict] = {}
        self._next_caption_group_id = 1
        out_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_path = out_dir / "transcript.txt"
        self.caption_debug_path = out_dir / "caption_debug.jsonl"
        self.status_path = out_dir / "status.json"
        self._flush()

    # -------- transcript ------------------------------------------------

    @staticmethod
    def _common_prefix_len(left: str, right: str) -> int:
        limit = min(len(left), len(right))
        for idx in range(limit):
            if left[idx] != right[idx]:
                return idx
        return limit

    @classmethod
    def _is_caption_revision(cls, previous: str, current: str) -> bool:
        previous_norm = re.sub(r"\s+", " ", previous or "").strip()
        current_norm = re.sub(r"\s+", " ", current or "").strip()
        if not previous_norm or not current_norm or previous_norm == current_norm:
            return False
        previous_lower = previous_norm.lower()
        current_lower = current_norm.lower()
        if current_lower.startswith(previous_lower):
            return True
        prefix_len = cls._common_prefix_len(previous_lower, current_lower)
        if prefix_len >= min(24, max(4, len(previous_lower) // 2)):
            return True
        ratio = SequenceMatcher(None, previous_lower, current_lower).ratio()
        return ratio >= 0.82 and prefix_len >= 8

    def _touch_caption_progress(self) -> str:
        self.last_caption_at = time.time()
        self.last_progress_at = self.last_caption_at
        self.phase = "capturing"
        self.stalled_reason = None
        if not self.in_call:
            self.in_call = True
            self.lobby_waiting = False
            self.joined_at = self.last_caption_at
        return time.strftime("%H:%M:%S", time.localtime(self.last_caption_at))

    def _rewrite_transcript(self) -> None:
        rendered: list[tuple[str, str, str, str]] = []
        for group in self._caption_groups:
            speaker = str(group.get("speaker") or "")
            text = str(group.get("text") or "")
            ts = str(group.get("ts") or "")
            group_id = str(group.get("id") or "")
            for chunk in self._split_caption_text(text):
                rendered.append((ts, speaker, chunk, group_id))
        self._transcript_entries = rendered
        with self.transcript_path.open("w", encoding="utf-8") as f:
            for ts, speaker, text, _group_id in rendered:
                f.write(f"[{ts}] {speaker}: {text}\n")
        self.transcript_lines = len(rendered)

    @staticmethod
    def _caption_suffix(previous: str, current: str) -> str:
        if not current.startswith(previous):
            return ""
        return current[len(previous):].lstrip(" \t\r\n,.;:!?-")

    @staticmethod
    def _same_caption_text(left: str, right: str) -> bool:
        return re.sub(r"\s+", " ", left or "").strip().lower() == re.sub(
            r"\s+", " ", right or ""
        ).strip().lower()

    @staticmethod
    def _caption_tokens(text: str) -> list[str]:
        return re.findall(r"[\w']+", (text or "").lower())

    @classmethod
    def _caption_token_prefix_len(cls, previous: str, current: str) -> int:
        previous_tokens = cls._caption_tokens(previous)
        current_tokens = cls._caption_tokens(current)
        prefix_len = 0
        for left, right in zip(previous_tokens, current_tokens):
            if left != right:
                break
            prefix_len += 1
        return prefix_len

    @staticmethod
    def _caption_token_lcs_len(previous_tokens: list[str], current_tokens: list[str]) -> int:
        if not previous_tokens or not current_tokens:
            return 0
        previous_len = len(previous_tokens)
        current_len = len(current_tokens)
        row = [0] * (current_len + 1)
        for previous_idx in range(previous_len - 1, -1, -1):
            next_row = row[:]
            previous_token = previous_tokens[previous_idx]
            for current_idx in range(current_len - 1, -1, -1):
                if previous_token == current_tokens[current_idx]:
                    next_row[current_idx] = 1 + row[current_idx + 1]
                else:
                    next_row[current_idx] = max(row[current_idx], next_row[current_idx + 1])
            row = next_row
        return row[0]

    @classmethod
    def _is_caption_token_prefix(cls, previous: str, current: str) -> bool:
        previous_tokens = cls._caption_tokens(previous)
        current_tokens = cls._caption_tokens(current)
        if not previous_tokens or len(current_tokens) <= len(previous_tokens):
            return False
        return current_tokens[: len(previous_tokens)] == previous_tokens

    @classmethod
    def _is_cross_caption_key_revision(cls, previous: str, current: str) -> bool:
        previous_norm = re.sub(r"\s+", " ", previous or "").strip()
        current_norm = re.sub(r"\s+", " ", current or "").strip()
        if not previous_norm or not current_norm:
            return False
        previous_lower = previous_norm.lower()
        current_lower = current_norm.lower()
        if current_lower == previous_lower:
            return True
        if current_lower.startswith(previous_lower):
            return True
        if cls._is_caption_token_prefix(previous_norm, current_norm):
            return True
        normalized_previous = " ".join(cls._caption_tokens(previous_norm))
        normalized_current = " ".join(cls._caption_tokens(current_norm))
        if normalized_previous and normalized_current.startswith(normalized_previous):
            return True
        previous_tokens = cls._caption_tokens(previous_norm)
        current_tokens = cls._caption_tokens(current_norm)
        token_prefix_len = cls._caption_token_prefix_len(previous_norm, current_norm)
        if (
            len(previous_tokens) == len(current_tokens)
            and token_prefix_len >= 4
            and token_prefix_len >= len(previous_tokens) - 1
        ):
            return True
        if len(current_tokens) > len(previous_tokens):
            if len(previous_tokens) <= 2 and token_prefix_len >= 1:
                return True
            if token_prefix_len >= 4 and token_prefix_len >= len(previous_tokens) - 1:
                return True
            lcs_len = cls._caption_token_lcs_len(previous_tokens, current_tokens)
            suffix_overlap = lcs_len - token_prefix_len
            if (
                token_prefix_len >= 4
                and suffix_overlap >= 2
                and lcs_len >= len(previous_tokens) - 2
            ):
                return True
        prefix_len = cls._common_prefix_len(previous_lower, current_lower)
        return prefix_len >= 80 and cls._is_caption_revision(previous_norm, current_norm)

    @staticmethod
    def _split_caption_text(text: str) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []
        chunks: list[str] = []
        remaining = text
        while len(remaining) > MAX_TRANSCRIPT_TEXT_LEN:
            split_at = remaining.rfind(" ", 0, MAX_TRANSCRIPT_TEXT_LEN + 1)
            if split_at <= 0:
                split_at = MAX_TRANSCRIPT_TEXT_LEN
            chunks.append(remaining[:split_at].strip())
            remaining = remaining[split_at:].strip()
        if remaining:
            chunks.append(remaining)
        return chunks

    def _new_caption_group(self, speaker: str, text: str, ts: str, caption_key: str) -> dict:
        group_id = f"caption-group-{self._next_caption_group_id}"
        self._next_caption_group_id += 1
        group = {
            "id": group_id,
            "speaker": speaker,
            "text": text,
            "ts": ts,
            "caption_keys": {caption_key},
        }
        self._caption_groups.append(group)
        self._caption_group_by_key[caption_key] = group
        return group

    def _caption_group_revision_score(self, previous: str, current: str) -> int:
        previous_norm = re.sub(r"\s+", " ", previous or "").strip().lower()
        current_norm = re.sub(r"\s+", " ", current or "").strip().lower()
        char_score = self._common_prefix_len(previous_norm, current_norm)
        token_score = self._caption_token_prefix_len(previous_norm, current_norm) * 1000
        return token_score + char_score

    def _revision_caption_group_for_speaker(
        self,
        speaker: str,
        text: str,
        *,
        exclude_caption_key: str,
    ) -> Optional[dict]:
        candidates: list[tuple[int, dict]] = []
        for group in reversed(self._caption_groups[-24:]):
            if group.get("speaker") != speaker:
                continue
            caption_keys = group.get("caption_keys")
            if isinstance(caption_keys, set) and exclude_caption_key in caption_keys:
                continue
            previous = str(group.get("text") or "")
            if not self._is_cross_caption_key_revision(previous, text):
                continue
            candidates.append((self._caption_group_revision_score(previous, text), group))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
            return None
        return candidates[0][1]

    def _resolve_caption_group(self, speaker: str, text: str, caption_key: str) -> Optional[dict]:
        group = self._caption_group_by_key.get(caption_key)
        if group:
            previous = str(group.get("text") or "")
            if (
                self._same_caption_text(previous, text)
                or self._is_cross_caption_key_revision(previous, text)
                or self._is_caption_revision(previous, text)
            ):
                return group
            return None
        return self._revision_caption_group_for_speaker(
            speaker,
            text,
            exclude_caption_key=caption_key,
        )

    def _upsert_caption_group(self, speaker: str, text: str, ts: str, caption_key: str) -> bool:
        group = self._resolve_caption_group(speaker, text, caption_key)
        if group is None:
            self._new_caption_group(speaker, text, ts, caption_key)
            self._rewrite_transcript()
            return True

        previous = str(group.get("text") or "")
        caption_keys = group.get("caption_keys")
        if not isinstance(caption_keys, set):
            caption_keys = set()
            group["caption_keys"] = caption_keys
        caption_keys.add(caption_key)
        self._caption_group_by_key[caption_key] = group
        if self._same_caption_text(previous, text):
            return False
        group["text"] = text
        group["speaker"] = speaker
        group["ts"] = ts
        self._rewrite_transcript()
        return True

    def record_caption(
        self,
        speaker: str,
        text: str,
        *,
        speaker_source: Optional[str] = None,
        speaker_debug: Optional[dict] = None,
        caption_id: Optional[str] = None,
    ) -> None:
        """Append a caption line if we haven't seen this exact (speaker, text)."""
        speaker = (speaker or "").strip()
        text = (text or "").strip()
        if not text:
            return
        if speaker_source:
            self.last_speaker_source = speaker_source
        debug_enabled = _debug_status_enabled()
        unresolved_speaker = not speaker or speaker.lower() == "unknown"
        if debug_enabled and isinstance(speaker_debug, dict):
            candidates = speaker_debug.get("candidates")
            if isinstance(candidates, list):
                self.last_speaker_candidates = candidates[:20]
            if unresolved_speaker:
                debug_line = {
                    "ts": time.time(),
                    "text": text[:300],
                    "speakerSource": speaker_source,
                    "speakerDebug": speaker_debug,
                }
                with self.caption_debug_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(debug_line, ensure_ascii=True) + "\n")
        if _caption_text_is_ui_noise(text):
            self.caption_ui_noise_drops += 1
            if unresolved_speaker:
                self.unresolved_caption_drops += 1
                self.last_unresolved_caption_at = time.time()
            self._flush()
            return
        if unresolved_speaker:
            now = time.time()
            self.last_unresolved_caption_at = now
            self.unresolved_caption_lines += 1
            speaker = "Unresolved speaker"

        caption_id = (caption_id or "").strip()
        if caption_id:
            caption_key = f"row:{caption_id}"
        elif unresolved_speaker:
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
            caption_key = f"unresolved:{digest}"
        else:
            caption_key = f"speaker:{speaker}"
        key = f"{caption_key}|{speaker}|{text}"
        if key in self._seen:
            return
        self._seen.add(key)
        ts = self._touch_caption_progress()
        self._upsert_caption_group(speaker, text, ts, caption_key)
        self._flush()

    # -------- status file ----------------------------------------------

    def _flush(self) -> None:
        debug_enabled = _debug_status_enabled()
        data = {
            "meetingId": self.meeting_id,
            "url": self.url,
            "inCall": self.in_call,
            "captioning": self.captioning,
            "captionsEnabledAttempted": self.captions_enabled_attempted,
            "lobbyWaiting": self.lobby_waiting,
            "joinAttemptedAt": self.join_attempted_at,
            "joinedAt": self.joined_at,
            "lastCaptionAt": self.last_caption_at,
            "transcriptLines": self.transcript_lines,
            "transcriptPath": str(self.transcript_path),
            "error": self.error,
            "exited": self.exited,
            "pid": os.getpid(),
            "phase": self.phase,
            "lastHeartbeatAt": self.last_heartbeat_at,
            "lastProgressAt": self.last_progress_at,
            "stalledReason": self.stalled_reason,
            "lastUiText": self.last_ui_text if debug_enabled else None,
            "lastUrl": self.last_url,
            "lastSpeakerSource": self.last_speaker_source,
            "lastSpeakerCandidates": self.last_speaker_candidates if debug_enabled else [],
            "captionDebugPath": str(self.caption_debug_path) if debug_enabled else None,
            "localMicrophoneOn": self.local_microphone_on,
            "localCameraOn": self.local_camera_on,
            # v2 realtime telemetry.
            "realtime": self.realtime,
            "realtimeReady": self.realtime_ready,
            "realtimeDevice": self.realtime_device,
            "realtimeAudioPumpStatus": self.realtime_audio_pump_status,
            "realtimeAudioPumpTool": self.realtime_audio_pump_tool,
            "realtimeAudioPumpPid": self.realtime_audio_pump_pid,
            "realtimeAudioPumpReturnCode": self.realtime_audio_pump_return_code,
            "realtimeAudioPumpError": self.realtime_audio_pump_error,
            "audioBytesOut": self.audio_bytes_out,
            "lastAudioOutAt": self.last_audio_out_at,
            "lastBargeInAt": self.last_barge_in_at,
            "leaveReason": self.leave_reason,
            "unresolvedCaptionDrops": self.unresolved_caption_drops,
            "unresolvedCaptionLines": self.unresolved_caption_lines,
            "captionUiNoiseDrops": self.caption_ui_noise_drops,
            "lastUnresolvedCaptionAt": self.last_unresolved_caption_at,
        }
        tmp = self.status_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.status_path)

    def set(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
        if any(k in kwargs for k in ("join_attempted_at", "joined_at", "last_caption_at")):
            self.last_progress_at = time.time()
        self._flush()

    def heartbeat(
        self,
        *,
        phase: Optional[str] = None,
        stalled_reason: Optional[str] = None,
        last_ui_text: Optional[str] = None,
        last_url: Optional[str] = None,
    ) -> None:
        if phase:
            self.phase = phase
        self.stalled_reason = stalled_reason
        if last_ui_text is not None:
            if _debug_status_enabled():
                text = " ".join(str(last_ui_text).split())
                self.last_ui_text = text[:1000]
            else:
                self.last_ui_text = None
        if last_url is not None:
            self.last_url = str(last_url)
        self.last_heartbeat_at = time.time()
        self._flush()


# ---------------------------------------------------------------------------
# Playwright bot entry point
# ---------------------------------------------------------------------------

_CAPTION_UI_NOISE_PATTERNS = (
    "audio settings",
    "caption settings",
    "chat with everyone",
    "getting ready",
    "join now",
    "jump to bottom",
    "jump to most recent captions",
    "leave call",
    "meeting tools",
    "more options",
    "open caption settings",
    "present now",
    "return home",
    "turn off camera",
    "turn off microphone",
    "turn on camera",
    "turn on microphone",
)


def _caption_text_is_ui_noise(text: str) -> bool:
    normalized = " ".join((text or "").split()).strip().lower()
    if not normalized:
        return True
    if (
        "open caption settings" in normalized
        and (
            "font size" in normalized
            or "font colour settings" in normalized
            or "font color settings" in normalized
            or "format_size" in normalized
        )
    ):
        return True
    return normalized in _CAPTION_UI_NOISE_PATTERNS


# JavaScript injected into the Meet tab to observe captions. Captures
# {speaker, text} tuples via a MutationObserver on the caption container,
# and exposes ``window.__hermesMeetDrain()`` to pull new entries. This
# mirrors the OpenUtter caption scraping approach.
_CAPTION_OBSERVER_JS = r"""
(() => {
  if (window.__hermesMeetInstalled) return;
  window.__hermesMeetInstalled = true;
  window.__hermesMeetQueue = [];
  window.__hermesMeetLastSpeaker = '';
  window.__hermesMeetLastSpeakerAt = 0;
  window.__hermesMeetLastFallbackText = '';
  window.__hermesMeetKnownSpeakers = [];
  window.__hermesMeetCaptionRegionAttached = false;
  window.__hermesMeetLastTextByCaptionKey = {};
  window.__hermesMeetCaptionRowIds = new WeakMap();
  window.__hermesMeetNextCaptionRowId = 1;

  const captionSelector = '[role="region"][aria-label*="aption" i], ' +
                          'div[jsname="YSxPC"], ' +  // legacy
                          'div[jsname="tgaKEf"]';    // current (Apr 2026)

  function cleanSpeakerName(raw) {
    let value = (raw || '').replace(/\s+/g, ' ').trim();
    if (!value) return '';
    const lower = value.toLowerCase();
    if (
      lower.includes('switch account') ||
      lower.includes('getting ready') ||
      lower.includes("you'll be able to join in just a moment") ||
      lower.includes('meet.google.com') ||
      lower.includes('@') ||
      lower === 'you' ||
      lower === 'unknown'
    ) {
      return '';
    }
    const patterns = [
      /^(.+?)(?:,|\s+)(?:is\s+)?speaking\b/i,
      /^(.+?)\s+\((?:speaking|is speaking)\)$/i,
      /^(.+?)\s+(?:is\s+)?presenting\b/i,
    ];
    for (const pattern of patterns) {
      const match = value.match(pattern);
      if (match && match[1]) {
        value = match[1].replace(/\s+/g, ' ').trim();
        break;
      }
    }
    value = value.replace(/\b(?:is\s+)?speaking\b/ig, '').trim();
    value = value.replace(/\b(?:microphone|camera)\s+(?:is\s+)?(?:off|muted)\b/ig, '').trim();
    value = value.replace(/[,.:\-]+$/g, '').trim();
    if (!value || value.length > 80) return '';
    return value;
  }

  function captionRowId(row) {
    if (!row || typeof row !== 'object') return '';
    if (!window.__hermesMeetCaptionRowIds.has(row)) {
      const nextId = `caption-row-${window.__hermesMeetNextCaptionRowId++}`;
      window.__hermesMeetCaptionRowIds.set(row, nextId);
    }
    return window.__hermesMeetCaptionRowIds.get(row);
  }

  function captionRowForLabel(label) {
    if (!label) return null;
    if (typeof label.closest === 'function') {
      const row = label.closest('div[jsname="dsyhDe"], div.CNusmb, div.TBMuR, div.nMcdL');
      if (row) return row;
    }
    for (let node = label.parentElement; node; node = node.parentElement) {
      const text = node.innerText || '';
      const children = Array.from(node.children || []);
      if (children.length >= 2 && text) return node;
    }
    return null;
  }

  function escapeRegExp(value) {
    return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  function rememberSpeaker(raw) {
    const speaker = cleanSpeakerName(raw);
    if (!speaker) return '';
    if (!window.__hermesMeetKnownSpeakers.includes(speaker)) {
      window.__hermesMeetKnownSpeakers.push(speaker);
      window.__hermesMeetKnownSpeakers = window.__hermesMeetKnownSpeakers.slice(-30);
    }
    window.__hermesMeetLastSpeaker = speaker;
    window.__hermesMeetLastSpeakerAt = Date.now();
    return speaker;
  }

  function inferParticipantNames() {
    const text = document.body ? document.body.innerText || '' : '';
    const names = new Set(window.__hermesMeetKnownSpeakers || []);
    const patterns = [
      /Pin\s+(.+?)\s+to your main screen/g,
      /More options for\s+(.+?)(?:\n|$)/g,
    ];
    for (const pattern of patterns) {
      let match;
      while ((match = pattern.exec(text)) !== null) {
        const speaker = cleanSpeakerName(match[1]);
        if (speaker) names.add(speaker);
      }
    }
    const labelNodes = Array.from(document.querySelectorAll('span.NWpY1d, .NWpY1d, div.KcIKyf')).slice(0, 80);
    for (const node of labelNodes) {
      const speaker = cleanSpeakerName(node.innerText);
      if (speaker) names.add(speaker);
    }
    return Array.from(names).sort((a, b) => b.length - a.length);
  }

  function trimCaptionChrome(text) {
    let value = (text || '').replace(/\s+/g, ' ').trim();
    const markers = [
      'keyboard_arrow_up Audio settings',
      'Audio settings mic_off',
      'Audio settings videocam',
      'Turn on microphone',
      'Turn off microphone',
      'Video settings',
      'Share screen',
      'Send a reaction',
      'Turn off captions',
      'Turn on captions',
      'Raise hand',
      'Leave call',
      'Chat with everyone',
      'Meeting tools',
      'Camera not found',
      'Make sure that your camera is plugged in',
    ];
    let end = value.length;
    for (const marker of markers) {
      const idx = value.indexOf(marker);
      if (idx >= 0) end = Math.min(end, idx);
    }
    return value.slice(0, end).trim();
  }

  function collectSpeakerCandidates() {
    const selectors = [
      { selector: '[aria-label*="speaking" i]', attrs: ['aria-label', 'innerText'] },
      { selector: '[aria-label*="is speaking" i]', attrs: ['aria-label', 'innerText'] },
      { selector: '[data-participant-name]', attrs: ['data-participant-name', 'aria-label', 'innerText'] },
      { selector: '[data-self-name]', attrs: ['data-self-name', 'aria-label', 'innerText'] },
      { selector: '[aria-label]', attrs: ['aria-label'], diagnosticOnly: true },
    ];
    const candidates = [];
    for (const spec of selectors) {
      const nodes = Array.from(document.querySelectorAll(spec.selector)).slice(0, 80);
      for (const node of nodes) {
        for (const attr of spec.attrs) {
          const raw = attr === 'innerText' ? node.innerText : node.getAttribute(attr);
          const clean = cleanSpeakerName(raw);
          candidates.push({
            selector: spec.selector,
            attr,
            raw: (raw || '').replace(/\s+/g, ' ').trim().slice(0, 200),
            clean,
            diagnosticOnly: !!spec.diagnosticOnly,
          });
        }
      }
    }
    return candidates;
  }

  function inferActiveSpeaker() {
    const candidates = collectSpeakerCandidates();
    for (const candidate of candidates) {
      if (candidate.clean && !candidate.diagnosticOnly) {
        rememberSpeaker(candidate.clean);
        return {
          speaker: candidate.clean,
          source: candidate.selector,
          candidates,
        };
      }
    }
    if (
      window.__hermesMeetLastSpeaker &&
      (Date.now() - window.__hermesMeetLastSpeakerAt) < 8000
    ) {
      return {
        speaker: window.__hermesMeetLastSpeaker,
        source: 'lastSpeaker',
        candidates,
      };
    }
    return {
      speaker: '',
      source: 'unresolved',
      candidates,
    };
  }

  function scanDocumentFallback() {
    if (window.__hermesMeetCaptionRegionAttached) return;
    const bodyText = document.body ? document.body.innerText || '' : '';
    if (!bodyText) return;
    const markers = [
      'Open caption settings',
      'Caption settings',
      'Font colour settings',
      'Font color settings',
    ];
    let start = -1;
    for (const marker of markers) {
      const idx = bodyText.lastIndexOf(marker);
      if (idx >= 0) start = Math.max(start, idx + marker.length);
    }
    if (start < 0) return;
    const tail = bodyText.slice(start).replace(/\s+/g, ' ').trim();
    if (!tail || tail === window.__hermesMeetLastFallbackText) return;
    window.__hermesMeetLastFallbackText = tail;

    const names = inferParticipantNames();
    if (!names.length) {
      const text = trimCaptionChrome(tail);
      if (text) pushEntry('', text);
      return;
    }
    if (!pushSpeakerSegments(tail, names)) {
      const text = trimCaptionChrome(tail);
      if (text) pushEntry('', text);
    }
  }

  function pushSpeakerSegments(rawText, names, captionIdPrefix = 'segment') {
    const text = trimCaptionChrome(rawText);
    if (!text) return false;
    const knownNames = names && names.length ? names : inferParticipantNames();
    if (!knownNames.length) return false;
    const matches = [];
    for (const name of knownNames) {
      const re = new RegExp(`(?:^|\\s)(${escapeRegExp(name)})(?=\\s)`, 'g');
      let match;
      while ((match = re.exec(text)) !== null) {
        matches.push({
          index: match.index + match[0].indexOf(match[1]),
          end: match.index + match[0].indexOf(match[1]) + match[1].length,
          speaker: name,
        });
      }
    }
    if (!matches.length) return false;
    matches.sort((a, b) => a.index - b.index || b.end - a.end);
    const deduped = [];
    for (const match of matches) {
      const prev = deduped[deduped.length - 1];
      if (prev && prev.index === match.index) continue;
      deduped.push(match);
    }
    let emitted = false;
    for (let i = 0; i < deduped.length; i += 1) {
      const current = deduped[i];
      const next = deduped[i + 1];
      const segment = trimCaptionChrome(text.slice(current.end, next ? next.index : undefined));
      if (segment) {
        pushEntry(current.speaker, segment, `${captionIdPrefix}:${current.index}:${current.speaker}`);
        emitted = true;
      }
    }
    return emitted;
  }

  function textAfterSpeaker(rawText, speaker) {
    let text = (rawText || '').replace(/\s+/g, ' ').trim();
    if (!text || !speaker) return '';
    if (text === speaker) return '';
    const idx = text.indexOf(speaker);
    if (idx < 0) return '';
    text = text.slice(idx + speaker.length).trim();
    return trimCaptionChrome(text);
  }

  function trimSeenCaptionPrefix(speaker, rawText, captionId) {
    const text = trimCaptionChrome(rawText);
    if (!text) return '';
    const key = captionId ? `row:${captionId}` : (cleanSpeakerName(speaker) || '__unknown__');
    const previous = window.__hermesMeetLastTextByCaptionKey[key] || '';
    if (text === previous) return '';
    window.__hermesMeetLastTextByCaptionKey[key] = text;
    return text;
  }

  function containsNode(root, target) {
    for (let node = target; node; node = node.parentElement) {
      if (node === root) return true;
    }
    return false;
  }

  function nearbyCaptionText(label, speaker) {
    for (let node = label.parentElement; node; node = node.parentElement) {
      const children = Array.from(node.children || []);
      if (children.length >= 2 && children.some((child) => containsNode(child, label))) {
        const text = children
          .filter((child) => !containsNode(child, label))
          .map((child) => child.innerText || '')
          .join(' ');
        const caption = trimCaptionChrome(text);
        if (caption && caption !== speaker) return caption;
      }
    }
    return '';
  }

  function scanVisibleSpeakerLabels(root) {
    const labels = Array.from(root.querySelectorAll('span.NWpY1d, .NWpY1d')).slice(0, 40);
    let emitted = false;
    for (const label of labels) {
      const speaker = cleanSpeakerName(label.innerText);
      if (!speaker) continue;
      const captionId = captionRowId(captionRowForLabel(label));

      const nearbyText = nearbyCaptionText(label, speaker);
      if (nearbyText) {
        pushEntry(speaker, nearbyText, captionId);
        emitted = true;
        continue;
      }

      let node = label;
      for (let depth = 0; node && depth < 7; depth += 1) {
        const text = textAfterSpeaker(node.innerText || '', speaker);
        if (text) {
          pushEntry(speaker, text, captionId);
          emitted = true;
          break;
        }
        node = node.parentElement;
      }
    }
    return emitted;
  }

  function pushEntry(speaker, text, captionId = '') {
    if (!text || !text.trim()) return;
    const rowSpeaker = cleanSpeakerName(speaker);
    if (rowSpeaker) rememberSpeaker(rowSpeaker);
    const inferred = rowSpeaker
      ? { speaker: rowSpeaker, source: 'captionRow', candidates: [] }
      : inferActiveSpeaker();
    const captionText = trimSeenCaptionPrefix(inferred.speaker, text, captionId);
    if (!captionText) return;
    const entry = {
      ts: Date.now(),
      speaker: inferred.speaker,
      speakerSource: inferred.source,
      speakerDebug: { candidates: inferred.candidates },
      text: captionText,
    };
    if (captionId) entry.captionId = captionId;
    window.__hermesMeetQueue.push(entry);
  }

  function scan(root) {
    // Meet captions render as a list of rows; each row contains a speaker
    // label and a text block. Selectors vary across Meet rewrites; we try
    // a few shapes and fall back to raw text.
    const rows = root.querySelectorAll('div[jsname="dsyhDe"], div.CNusmb, div.TBMuR');
    if (rows.length) {
      rows.forEach((row) => {
        const spkEl = row.querySelector('div.KcIKyf, span.NWpY1d, div.zs7s8d, span[jsname="YSxPC"]');
        const txtEl = row.querySelector('div.bh44bd, span[jsname="tgaKEf"], div.iTTPOb');
        const speaker = spkEl ? spkEl.innerText : '';
        const text = txtEl ? txtEl.innerText : row.innerText;
        pushEntry(speaker, text, captionRowId(row));
      });
      return;
    }
    if (scanVisibleSpeakerLabels(root)) return;
    // Fallback: split participant-prefixed live caption history before
    // treating the whole region's innerText as one anonymous line.
    if (pushSpeakerSegments(root.innerText || '')) return;
    const text = (root.innerText || '').split('\n').filter(Boolean).pop();
    pushEntry('', text);
  }

  function attach() {
    const el = document.querySelector(captionSelector);
    if (!el) return false;
    window.__hermesMeetCaptionRegionAttached = true;
    const obs = new MutationObserver(() => scan(el));
    obs.observe(el, { childList: true, subtree: true, characterData: true });
    scan(el);
    return true;
  }

  // Try now and retry on interval — the caption region only appears after
  // captions are enabled and someone speaks.
  if (!attach()) {
    const iv = setInterval(() => { if (attach()) clearInterval(iv); }, 1500);
  }
  setInterval(() => scanDocumentFallback(), 1500);

  window.__hermesMeetDrain = () => {
    const out = window.__hermesMeetQueue.slice();
    window.__hermesMeetQueue = [];
    return out;
  };
})();
"""


def _enable_captions_js() -> str:
    """Return a small JS snippet that tries to click the 'Turn on captions' button.

    Best-effort — Meet's caption toggle is keyboard-accessible via ``c``. We
    dispatch that keystroke as a cheap fallback. Real click targeting is too
    brittle to rely on.
    """
    return r"""
    (() => {
      const ev = new KeyboardEvent('keydown', {
        key: 'c', code: 'KeyC', keyCode: 67, which: 67, bubbles: true,
      });
      document.body.dispatchEvent(ev);
      return true;
    })();
    """


def _first_visible(locator):
    """Return the first visible element in a Playwright locator-like object."""
    try:
        count = locator.count()
    except Exception:
        count = 0
    if count <= 0:
        return None

    if hasattr(locator, "nth"):
        for idx in range(count):
            try:
                candidate = locator.nth(idx)
                if candidate.is_visible():
                    return candidate
            except Exception:
                continue

    try:
        first = locator.first
        if first.is_visible():
            return first
    except Exception:
        pass
    return None


def _captions_are_enabled(page) -> bool:
    """Return True when the Meet UI proves captions are currently enabled."""
    locators = (
        lambda: page.get_by_role(
            "button",
            name=re.compile(r"turn off captions", re.IGNORECASE),
        ),
        lambda: page.locator('button[aria-label*="Turn off captions" i]'),
        lambda: page.locator('[role="button"][aria-label*="Turn off captions" i]'),
        lambda: page.locator('[role="region"][aria-label*="aption" i]'),
        lambda: page.locator('div[jsname="YSxPC"], div[jsname="tgaKEf"]'),
    )
    for make_locator in locators:
        try:
            locator = make_locator()
            if _first_visible(locator) is not None:
                return True
        except Exception:
            continue
    return False


def _wait_for_ui(page, ms: int = 500) -> None:
    try:
        page.wait_for_timeout(ms)
    except Exception:
        pass


def _open_meet_hover_tray(page) -> None:
    """Reveal Meet's bottom controls when the in-call toolbar is hidden."""
    try:
        page.keyboard.press("ArrowDown")
        _wait_for_ui(page, 250)
    except Exception:
        pass


def _enable_captions(page, *, allow_shortcut: bool = True) -> bool:
    """Best-effort caption toggle without clicking an already-on control."""
    if _captions_are_enabled(page):
        return True

    locators = (
        lambda: page.get_by_role(
            "button",
            name=re.compile(r"turn on captions", re.IGNORECASE),
        ),
        lambda: page.locator('button[aria-label*="Turn on captions" i]'),
        lambda: page.locator('[role="button"][aria-label*="Turn on captions" i]'),
    )
    for make_locator in locators:
        try:
            btn = _first_visible(make_locator())
            if btn is not None:
                btn.click(timeout=3_000)
                return True
        except Exception:
            continue

    if not allow_shortcut:
        return False

    _open_meet_hover_tray(page)
    for make_locator in locators:
        try:
            btn = _first_visible(make_locator())
            if btn is not None:
                btn.click(timeout=3_000)
                return True
        except Exception:
            continue

    try:
        page.keyboard.press("c")
        return True
    except Exception:
        pass

    try:
        page.evaluate(_enable_captions_js())
        return True
    except Exception:
        return False


def _disable_local_media(
    page,
    *,
    disable_microphone: bool = True,
    disable_camera: bool = True,
) -> int:
    """Turn off the bot's local mic/camera if Meet left either enabled."""
    clicked = 0
    current = _probe_local_media_state(page)
    controls = [
        (
            "local_microphone_on",
            re.compile(r"(turn off|mute)\s+(microphone|mic)", re.IGNORECASE),
            "microphone",
            "Control+D",
        ),
        (
            "local_camera_on",
            re.compile(r"(turn off|stop|disable)\s+(camera|video)", re.IGNORECASE),
            "camera",
            "Control+E",
        ),
    ]
    if not disable_microphone:
        controls = [control for control in controls if control[2] != "microphone"]
    if not disable_camera:
        controls = [control for control in controls if control[2] != "camera"]
    for state_key, label, kind, shortcut in controls:
        if current.get(state_key) is False:
            continue
        toggled = False
        try:
            btn = _first_visible(page.get_by_role("button", name=label))
            if btn is not None:
                btn.click(timeout=3_000)
                toggled = True
        except Exception:
            pass
        if not toggled and _click_local_media_control_js(page, kind):
            toggled = True
        if not toggled and current.get(state_key) is True:
            try:
                page.keyboard.press(shortcut)
                toggled = True
            except Exception:
                pass
        if toggled:
            clicked += 1
            _wait_for_ui(page, 250)
    return clicked


def _enable_local_microphone(page) -> int:
    """Turn on the local Meet microphone only when an explicit off control is visible."""
    current = _probe_local_media_state(page)
    if current.get("local_microphone_on") is True:
        return 0
    toggled = False
    try:
        btn = _first_visible(
            page.get_by_role(
                "button",
                name=re.compile(r"(turn on|unmute)\s+(microphone|mic)", re.IGNORECASE),
            )
        )
        if btn is not None:
            btn.click(timeout=3_000)
            toggled = True
    except Exception:
        pass
    if not toggled and _click_local_media_control_js(page, "microphone_on"):
        toggled = True
    if toggled:
        _wait_for_ui(page, 250)
        return 1
    return 0


def _click_local_media_control_js(page, kind: str) -> bool:
    """Click a local Meet media-off control when role locators miss it."""
    if kind == "microphone":
        labels = ("turn off microphone", "mute microphone", "turn off mic", "mute mic")
    elif kind == "microphone_on":
        labels = ("turn on microphone", "unmute microphone", "turn on mic", "unmute mic")
    else:
        labels = ("turn off camera", "stop camera", "disable camera", "turn off video", "stop video")
    label_js = "|".join(re.escape(label) for label in labels)
    script = rf"""
    (() => {{
      const wanted = new RegExp({json.dumps(label_js)}, 'i');
      const reject = /you can't|someone else|more options for|pin .* to your main screen/i;
      const isVisible = (el) => {{
        const style = window.getComputedStyle(el);
        const rect = el.getBoundingClientRect();
        return style && style.visibility !== 'hidden' && style.display !== 'none' &&
          rect.width > 0 && rect.height > 0;
      }};
      const nodes = Array.from(document.querySelectorAll('button, [role="button"]'));
      for (const node of nodes) {{
        const label = [
          node.getAttribute('aria-label') || '',
          node.getAttribute('title') || '',
          node.innerText || '',
          node.textContent || '',
        ].join(' ');
        if (wanted.test(label) && !reject.test(label) && isVisible(node)) {{
          node.click();
          return true;
        }}
      }}
      return false;
    }})();
    """
    try:
        return bool(page.evaluate(script))
    except Exception:
        return False


def _probe_local_media_state(page) -> dict:
    """Infer local mic/camera state from visible Meet control labels."""
    controls = {
        "local_microphone_on": (
            re.compile(r"(turn off|mute)\s+(microphone|mic)", re.IGNORECASE),
            re.compile(r"(turn on|unmute)\s+(microphone|mic)", re.IGNORECASE),
        ),
        "local_camera_on": (
            re.compile(r"(turn off|stop|disable)\s+(camera|video)", re.IGNORECASE),
            re.compile(r"(turn on|start|enable)\s+(camera|video)", re.IGNORECASE),
        ),
    }
    result = {}
    for key, (on_pattern, off_pattern) in controls.items():
        state = None
        try:
            if _first_visible(page.get_by_role("button", name=on_pattern)) is not None:
                state = True
            elif _first_visible(page.get_by_role("button", name=off_pattern)) is not None:
                state = False
        except Exception:
            state = None
        result[key] = state
    js_result = _probe_local_media_state_js(page)
    for key, value in js_result.items():
        if result.get(key) is None and value is not None:
            result[key] = value
    return result


def _local_media_state_is_safe(
    media_state: dict,
    *,
    realtime_enabled: bool,
    realtime_route_ready: bool = False,
) -> bool:
    """Return True when local media state satisfies the selected privacy mode."""
    camera_off = media_state.get("local_camera_on") is False
    if realtime_enabled:
        return (
            realtime_route_ready
            and camera_off
            and media_state.get("local_microphone_on") is True
        )
    return camera_off and media_state.get("local_microphone_on") is False


def _ensure_local_media_before_join(
    page,
    state: _BotState,
    *,
    realtime_enabled: bool,
    realtime_route_ready: bool = False,
    attempts: int = 3,
) -> bool:
    """Disable local media and fail closed if the state cannot be proven safe."""
    for _idx in range(max(1, attempts)):
        if realtime_enabled and not realtime_route_ready:
            _disable_local_media(page, disable_microphone=True, disable_camera=True)
        else:
            _disable_local_media(
                page,
                disable_microphone=not realtime_enabled,
                disable_camera=True,
            )
            if realtime_enabled:
                _enable_local_microphone(page)
        media_state = _probe_local_media_state(page)
        state.set(**media_state)
        if _local_media_state_is_safe(
            media_state,
            realtime_enabled=realtime_enabled,
            realtime_route_ready=realtime_route_ready,
        ):
            return True
        _wait_for_ui(page, 250)

    state.set(
        error="local media state unsafe before join",
        leave_reason="unsafe_media_state",
        exited=True,
        phase="exited",
    )
    return False


def _probe_local_media_state_js(page) -> dict:
    script = r"""
    (() => {
      const text = document.body ? document.body.innerText || '' : '';
      const reject = /you can't|someone else|more options for|pin .* to your main screen/i;
      const isVisible = (el) => {
        const style = window.getComputedStyle(el);
        const rect = el.getBoundingClientRect();
        return style && style.visibility !== 'hidden' && style.display !== 'none' &&
          rect.width > 0 && rect.height > 0;
      };
      const buttonText = Array.from(document.querySelectorAll('button, [role="button"]'))
        .filter((node) => isVisible(node))
        .map((node) => [
          node.getAttribute('aria-label') || '',
          node.getAttribute('title') || '',
          node.innerText || '',
          node.textContent || '',
        ].join(' '))
        .filter((label) => !reject.test(label))
        .join('\n');
      const infer = (onPatterns, offPatterns) => {
        for (const pattern of offPatterns) {
          if (pattern.test(text) || pattern.test(buttonText)) return false;
        }
        for (const pattern of onPatterns) {
          if (pattern.test(text) || pattern.test(buttonText)) return true;
        }
        return null;
      };
      const localMicrophoneOn = infer(
        [/your microphone is turned on/i, /\bturn off microphone\b/i, /\bmute microphone\b/i],
        [/your microphone is turned off/i, /\bturn on microphone\b/i, /\bunmute microphone\b/i]
      );
      const localCameraOn = infer(
        [/your camera is turned on/i, /\bturn off camera\b/i, /\bstop camera\b/i, /\bturn off video\b/i],
        [/your camera is turned off/i, /\bturn on camera\b/i, /\bstart camera\b/i, /\bturn on video\b/i]
      );
      return {localMicrophoneOn, localCameraOn};
    })();
    """
    try:
        result = page.evaluate(script)
    except Exception:
        return {}
    if not isinstance(result, dict):
        return {}
    return {
        "local_microphone_on": result.get("localMicrophoneOn"),
        "local_camera_on": result.get("localCameraOn"),
    }


def _should_retry_caption_enable(
    state: _BotState,
    *,
    now: float,
    last_caption_enable_check: float,
) -> bool:
    if state.last_caption_at or (now - last_caption_enable_check) <= 3.0:
        return False
    return bool(state.in_call or state.join_attempted_at)


def _retry_caption_enable(page, state: _BotState) -> bool:
    """Retry enabling captions and update status only when an attempt succeeds.

    Before admission we avoid keyboard shortcuts because ``c`` can toggle
    captions off if Meet already has them enabled but has not exposed the
    control yet. After admission, failing to use the shortcut leaves headless
    runs stuck with no visible "Turn on captions" button and no transcript.
    """
    if _captions_are_enabled(page):
        state.set(captioning=True, captions_enabled_attempted=True)
        return True

    attempted = _enable_captions(page, allow_shortcut=state.in_call)
    if attempted:
        state.set(captions_enabled_attempted=True)
        _wait_for_ui(page, 500)
        if _captions_are_enabled(page):
            state.set(captioning=True, captions_enabled_attempted=True)
            return True
    if state.in_call:
        try:
            page.evaluate(_enable_captions_js())
            attempted = True
        except Exception:
            pass
        if attempted:
            state.set(captions_enabled_attempted=True)
            _wait_for_ui(page, 500)
    if not attempted or not _captions_are_enabled(page):
        return False
    state.set(captioning=True, captions_enabled_attempted=True)
    return True


def _should_probe_admission(
    state: _BotState,
    *,
    now: float,
    last_admission_check: float,
) -> bool:
    if (now - last_admission_check) <= 3.0:
        return False
    return bool((not state.in_call) or not state.last_caption_at)


def _apply_admission_probe(
    state: _BotState,
    ui_probe: dict,
    *,
    now: float,
    lobby_deadline: float,
    call_error_strike_limit: int = 3,
) -> tuple[bool, bool]:
    """Apply a Meet UI probe to admission state.

    Returns ``(admitted, terminal)``. ``terminal`` means the bot should exit.
    """
    if ui_probe.get("waitingLobby") and not state.lobby_waiting:
        state.set(lobby_waiting=True)
    if ui_probe.get("preJoin") and state.in_call and not state.last_caption_at:
        state.set(
            in_call=False,
            joined_at=None,
            phase="joining",
        )
    # Meet's "couldn't start the video call because of an error" banner is
    # frequently transient and can flash while the call is healthy. Treating a
    # single observation as fatal kills live sessions, but treating a
    # false-positive admission as proof of health leaves status stuck at
    # ``in_call`` on an error page. In-call controls or caption/transcript
    # evidence prove that a persistent call-error overlay is safe to ignore.
    has_caption_evidence = bool(state.last_caption_at or state.transcript_lines > 0)
    if not ui_probe.get("callError") or ui_probe.get("inCall") or has_caption_evidence:
        state.call_error_strikes = 0
    elif state.join_attempted_at:
        state.call_error_strikes += 1
        if state.call_error_strikes >= call_error_strike_limit:
            state.set(
                in_call=False,
                joined_at=None,
                error="meet call error before captions",
                leave_reason="meet_error",
                phase="exited",
            )
            return False, True
    if ui_probe.get("landing") and not state.last_caption_at:
        state.set(
            in_call=False,
            joined_at=None,
            error="meet returned to landing before captions",
            leave_reason="meet_landing",
            phase="exited",
        )
        return False, True

    admitted = bool(ui_probe.get("inCall"))
    if admitted:
        state.ever_admitted = True
        state.set(
            in_call=True,
            lobby_waiting=False,
            joined_at=state.joined_at or now,
        )
        return True, False
    if now > lobby_deadline:
        state.set(
            error=(
                "lobby timeout — host never admitted the bot "
                f"within {int(lobby_deadline - state.join_attempted_at) if state.join_attempted_at else 0}s"
            ),
            leave_reason="lobby_timeout",
            phase="exited",
        )
        return False, True
    if bool(ui_probe.get("denied")) and (state.join_attempted_at or ui_probe.get("terminalDenied")):
        state.set(
            error="host denied admission",
            leave_reason="denied",
            phase="exited",
        )
        return False, True
    return False, False


def _compute_meet_phase(
    state: _BotState,
    *,
    now: float,
    stall_after: float,
) -> tuple[str, Optional[str]]:
    if state.exited:
        return "exited", state.leave_reason or state.error
    if state.transcript_lines > 0 or state.last_caption_at:
        return "capturing", None
    if state.in_call:
        return "in_call", None
    if state.join_attempted_at:
        age = now - state.join_attempted_at
        if age > stall_after:
            return "stalled", f"no admission progress for {int(age)}s"
        if state.lobby_waiting:
            return "waiting_lobby", None
        return "joining", None
    if state.captioning:
        return "joining", None
    return "starting", None


def _start_pcm_file_pump(
    *,
    rt: dict,
    pcm_path: Path,
    proc,
    stop_flag: dict,
    state: "_BotState",
    error_prefix: str,
) -> None:
    """Stream newly appended PCM bytes into a subprocess stdin."""

    def _pcm_pump_loop():
        offset = 0
        ready_reported = False
        try:
            while not stop_flag.get("stop", False):
                return_code = proc.poll()
                if return_code is not None:
                    state.set(
                        realtime_ready=False,
                        realtime_audio_pump_status="exited",
                        realtime_audio_pump_return_code=return_code,
                    )
                    break
                if not ready_reported:
                    state.set(
                        realtime_ready=True,
                        realtime_audio_pump_status="ready",
                        realtime_audio_pump_pid=getattr(proc, "pid", None),
                        realtime_audio_pump_return_code=None,
                        realtime_audio_pump_error=None,
                    )
                    ready_reported = True
                try:
                    with pcm_path.open("rb") as fh:
                        fh.seek(offset)
                        chunk = fh.read()
                except FileNotFoundError:
                    chunk = b""
                if chunk:
                    offset += len(chunk)
                    if proc.stdin is None:
                        state.set(
                            realtime_ready=False,
                            realtime_audio_pump_status="exited",
                            realtime_audio_pump_error=f"{error_prefix} pcm pump stdin unavailable",
                        )
                        break
                    try:
                        proc.stdin.write(chunk)
                        proc.stdin.flush()
                    except (BrokenPipeError, OSError) as e:
                        state.set(
                            realtime_ready=False,
                            realtime_audio_pump_status="exited",
                            realtime_audio_pump_error=f"{error_prefix} pcm pump write failed: {e}",
                            realtime_audio_pump_return_code=proc.poll(),
                        )
                        break
                else:
                    time.sleep(0.05)
        except Exception as e:
            state.set(
                realtime_ready=False,
                realtime_audio_pump_status="exited",
                realtime_audio_pump_error=f"{error_prefix} pcm pump crashed: {e}",
                error=f"{error_prefix} pcm pump crashed: {e}",
            )
        finally:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except Exception:
                pass

    t_pump = threading.Thread(
        target=_pcm_pump_loop,
        name="meet-pcm-pump",
        daemon=True,
    )
    t_pump.start()
    rt["pcm_pump_thread"] = t_pump


def _teardown_realtime(rt: dict, state: Optional[_BotState] = None) -> None:
    """Best-effort cleanup for realtime session, pump, and audio bridge."""
    if state is not None and (
        rt.get("enabled") or rt.get("session") or rt.get("pcm_pump")
    ):
        state.set(
            realtime_ready=False,
            realtime_audio_pump_status="teardown",
        )
    if rt.get("speaker_stop"):
        try:
            rt["speaker_stop"]()
        except Exception:
            pass
    if rt.get("pcm_pump_thread") is not None:
        try:
            rt["pcm_pump_thread"].join(timeout=3.0)
        except Exception:
            pass
    if rt.get("pcm_pump"):
        try:
            if rt["pcm_pump"].poll() is None:
                rt["pcm_pump"].terminate()
            rt["pcm_pump"].wait(timeout=3)
        except Exception:
            pass
    if rt.get("speaker_thread") is not None:
        try:
            rt["speaker_thread"].join(timeout=5.0)
        except Exception:
            pass
    if rt.get("session"):
        try:
            rt["session"].close()
        except Exception:
            pass
    if rt.get("bridge"):
        try:
            rt["bridge"].teardown()
        except Exception:
            pass


def _start_realtime_speaker(
    *,
    rt: dict,
    out_dir: Path,
    bridge_info: dict,
    api_key: str,
    model: str,
    voice: str,
    instructions: str,
    stop_flag: dict,
    state: "_BotState",
) -> None:
    """Wire up the OpenAI Realtime session + speaker thread + PCM pump.

    The speaker thread reads text lines from ``say_queue.jsonl``, sends each
    to OpenAI Realtime, and writes PCM audio into ``speaker.pcm``. A
    separate *pump* thread forwards that PCM into the OS audio sink so
    Chrome's fake mic picks it up. On Linux we pipe to ``paplay`` against
    the null-sink; on macOS the caller is expected to have the BlackHole
    device selected as default input.
    """
    try:
        from plugins.google_meet.realtime.openai_client import (
            RealtimeSession,
            RealtimeSpeaker,
        )
    except Exception as e:
        state.set(error=f"realtime import failed: {e}")
        return

    pcm_path = out_dir / SAY_PCM_FILENAME
    queue_path = out_dir / SAY_QUEUE_FILENAME
    processed_path = out_dir / "say_processed.jsonl"
    # Reset the sink file so we start clean each session.
    pcm_path.write_bytes(b"")
    # Make sure the queue exists so the speaker poller doesn't error on
    # first iteration.
    queue_path.touch()

    try:
        session = RealtimeSession(
            api_key=api_key,
            model=model,
            voice=voice,
            instructions=instructions,
            audio_sink_path=pcm_path,
            sample_rate=24000,
        )
        session.connect()
    except Exception as e:
        state.set(error=f"realtime connect failed: {e}")
        return

    rt["session"] = session

    def _stop_fn():
        return stop_flag.get("stop", False)

    rt["speaker_stop"] = lambda: stop_flag.__setitem__("stop", True)

    speaker = RealtimeSpeaker(
        session=session,
        queue_path=queue_path,
        processed_path=processed_path,
    )

    def _speaker_loop():
        try:
            speaker.run_until_stopped(_stop_fn)
        except Exception as e:
            state.set(error=f"realtime speaker crashed: {e}")

    t_speaker = threading.Thread(target=_speaker_loop, name="meet-speaker", daemon=True)
    t_speaker.start()
    rt["speaker_thread"] = t_speaker

    # PCM pump: feeds speaker.pcm (24kHz s16le mono) into the OS audio
    # device that Chrome's fake mic reads from. Different tools per
    # platform, but the contract is the same — block-read the growing
    # PCM file and stream it to the device in near-real-time.
    platform_tag = (bridge_info or {}).get("platform")
    if platform_tag == "linux":
        import subprocess as _sp

        sink = (bridge_info or {}).get("write_target") or "hermes_meet_sink"
        state.set(
            realtime_ready=False,
            realtime_audio_pump_status="starting",
            realtime_audio_pump_tool="paplay",
            realtime_audio_pump_pid=None,
            realtime_audio_pump_return_code=None,
            realtime_audio_pump_error=None,
        )
        try:
            proc = _sp.Popen(
                [
                    "paplay",
                    "--raw",
                    "--rate=24000",
                    "--format=s16le",
                    "--channels=1",
                    f"--device={sink}",
                ],
                stdin=_sp.PIPE,
                stdout=_sp.DEVNULL,
                stderr=_sp.DEVNULL,
            )
            rt["pcm_pump"] = proc
            state.set(realtime_audio_pump_pid=getattr(proc, "pid", None))
            _start_pcm_file_pump(
                rt=rt,
                pcm_path=pcm_path,
                proc=proc,
                stop_flag=stop_flag,
                state=state,
                error_prefix="linux",
            )
        except FileNotFoundError:
            state.set(
                realtime_ready=False,
                realtime_audio_pump_status="missing_tool",
                realtime_audio_pump_error="paplay not found",
                error="paplay not found — install pulseaudio-utils for realtime on Linux",
            )
    elif platform_tag == "darwin":
        # macOS: use ffmpeg to tail-read speaker.pcm and write it to the
        # BlackHole output device. The user must have BlackHole selected
        # as the default input in System Settings → Sound for Chrome to
        # pick it up. We prefer ffmpeg because it's scriptable and can
        # target AVFoundation devices by name; fall back to afplay-ing
        # the file in a tight loop if ffmpeg is absent.
        import shutil as _shutil
        import subprocess as _sp

        device_name = (bridge_info or {}).get("write_target") or "BlackHole 2ch"
        if _shutil.which("ffmpeg"):
            state.set(
                realtime_ready=False,
                realtime_audio_pump_status="starting",
                realtime_audio_pump_tool="ffmpeg",
                realtime_audio_pump_pid=None,
                realtime_audio_pump_return_code=None,
                realtime_audio_pump_error=None,
            )
            try:
                audio_device_index = _mac_audio_device_index(device_name)
                if audio_device_index is None:
                    state.set(
                        realtime_ready=False,
                        realtime_audio_pump_status="missing_tool",
                        realtime_audio_pump_error=f"macOS audio device not found: {device_name}",
                        error=f"macOS audio device not found: {device_name}",
                    )
                    return
                proc = _sp.Popen(
                    [
                        "ffmpeg",
                        "-hide_banner", "-loglevel", "error",
                        "-f", "s16le", "-ar", "24000", "-ac", "1",
                        "-i", "pipe:0",
                        "-f", "audiotoolbox",
                        "-audio_device_index", audio_device_index,
                        "-",
                    ],
                    stdin=_sp.PIPE,
                    stdout=_sp.DEVNULL,
                    stderr=_sp.DEVNULL,
                )
                rt["pcm_pump"] = proc
                state.set(realtime_audio_pump_pid=getattr(proc, "pid", None))
                _start_pcm_file_pump(
                    rt=rt,
                    pcm_path=pcm_path,
                    proc=proc,
                    stop_flag=stop_flag,
                    state=state,
                    error_prefix="macOS",
                )
            except FileNotFoundError:
                state.set(
                    realtime_ready=False,
                    realtime_audio_pump_status="missing_tool",
                    realtime_audio_pump_error="ffmpeg not found",
                    error="ffmpeg not found — install via `brew install ffmpeg` for realtime on macOS",
                )
            except Exception as e:
                state.set(
                    realtime_ready=False,
                    realtime_audio_pump_status="exited",
                    realtime_audio_pump_error=f"macOS pcm pump failed to start: {e}",
                    error=f"macOS pcm pump failed to start: {e}",
                )
        else:
            state.set(
                realtime_ready=False,
                realtime_audio_pump_status="missing_tool",
                realtime_audio_pump_tool="ffmpeg",
                realtime_audio_pump_error="ffmpeg not found",
                error="ffmpeg not found — install via `brew install ffmpeg` for realtime on macOS",
            )


def _mac_audio_device_index(device_name: str) -> Optional[str]:
    """Return the ffmpeg ``-audio_device_index`` for *device_name*, as a string.

    Probes ``ffmpeg -f avfoundation -list_devices true -i ''`` (which prints
    the device table on stderr) and matches *device_name* case-insensitively.
    Returns ``None`` when the device cannot be found so realtime speech fails
    closed instead of routing audio to an arbitrary device.
    """
    import subprocess as _sp

    try:
        out = _sp.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    # ffmpeg prints the table on stderr. Lines look like:
    #   [AVFoundation indev @ 0x...] [0] BlackHole 2ch
    import re as _re

    needle = device_name.strip().lower()
    for line in (out.stderr or "").splitlines():
        m = _re.search(r"\[(\d+)\]\s+(.+)$", line)
        if not m:
            continue
        if m.group(2).strip().lower() == needle:
            return m.group(1)
    return None


def _realtime_route_ready(state: _BotState) -> bool:
    return bool(
        state.realtime
        and state.realtime_ready
        and state.realtime_audio_pump_status == "ready"
    )


def _wait_for_realtime_route_ready(state: _BotState, *, timeout_s: float = 15.0) -> bool:
    deadline = time.time() + max(0.1, timeout_s)
    terminal_statuses = {"exited", "missing_tool"}
    while time.time() < deadline:
        if _realtime_route_ready(state):
            return True
        if state.error or state.realtime_audio_pump_status in terminal_statuses:
            break
        time.sleep(0.05)
    state.set(
        realtime_ready=False,
        error=state.error or "realtime audio route not ready before join",
        leave_reason="realtime_not_ready",
        exited=True,
        phase="exited",
    )
    return False


def _apply_meet_proxy_args(chrome_args: list[str]) -> None:
    """Append deterministic Chromium proxy/media args when a Meet proxy is set."""
    proxy_server = os.environ.get("HERMES_MEET_PROXY_SERVER", "").strip()
    if not proxy_server:
        return

    chrome_args.append(f"--proxy-server={proxy_server}")
    proxy_bypass = os.environ.get("HERMES_MEET_PROXY_BYPASS", MEET_MEDIA_PROXY_BYPASS).strip()
    if proxy_bypass:
        chrome_args.append(f"--proxy-bypass-list={proxy_bypass}")
    chrome_args.append(MEET_WEBRTC_PROXY_POLICY)


def _build_browser_launch_config(*, realtime_enabled: bool) -> tuple[list[str], list[str]]:
    """Return Chromium args and browser permissions for the selected Meet mode."""
    chrome_args = [
        "--use-fake-ui-for-media-stream",
        "--disable-blink-features=AutomationControlled",
    ]
    if not realtime_enabled:
        chrome_args.insert(0, "--use-fake-device-for-media-stream")
    permissions = ["microphone", "camera"]

    _apply_meet_proxy_args(chrome_args)
    return chrome_args, permissions


def run_bot() -> int:  # noqa: C901 — orchestration, explicit branches
    url = os.environ.get("HERMES_MEET_URL", "").strip()
    out_dir_env = os.environ.get("HERMES_MEET_OUT_DIR", "").strip()
    headed = os.environ.get("HERMES_MEET_HEADED", "").lower() in {"1", "true", "yes"}
    auth_state = os.environ.get("HERMES_MEET_AUTH_STATE", "").strip()
    guest_name = os.environ.get("HERMES_MEET_GUEST_NAME", "Hermes Agent")
    duration_s = _parse_duration(os.environ.get("HERMES_MEET_DURATION", ""))
    # v2: optional realtime mode. Enabled when HERMES_MEET_MODE=realtime.
    mode = os.environ.get("HERMES_MEET_MODE", "transcribe").strip().lower()
    realtime_model = os.environ.get("HERMES_MEET_REALTIME_MODEL", "gpt-realtime")
    realtime_voice = os.environ.get("HERMES_MEET_REALTIME_VOICE", "alloy")
    realtime_instructions = os.environ.get("HERMES_MEET_REALTIME_INSTRUCTIONS", "")
    realtime_api_key = os.environ.get("HERMES_MEET_REALTIME_KEY") or os.environ.get("OPENAI_API_KEY", "")

    if not url or not _is_safe_meet_url(url):
        sys.stderr.write(
            "google_meet bot: refusing to launch — HERMES_MEET_URL must be a "
            "meet.google.com URL. got: %r\n" % url
        )
        return 2
    if not out_dir_env:
        sys.stderr.write("google_meet bot: HERMES_MEET_OUT_DIR is required\n")
        return 2

    out_dir = Path(out_dir_env)
    meeting_id = _meeting_id_from_url(url)
    state = _BotState(out_dir=out_dir, meeting_id=meeting_id, url=url)

    # SIGTERM → exit cleanly so the parent ``meet_leave`` gets a finalized
    # transcript. We set a flag instead of raising so the Playwright context
    # teardown runs in the finally block below.
    stop_flag = {"stop": False}

    def _on_signal(_sig, _frame):
        stop_flag["stop"] = True

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    # v2 realtime: provision virtual audio device + start speaker thread.
    # We track these in a dict so teardown runs regardless of how we exit. If
    # realtime setup cannot be proven ready, fail closed before joining.
    rt = {
        "enabled": mode == "realtime",
        "bridge": None,            # AudioBridge | None
        "bridge_info": None,       # dict | None
        "session": None,           # RealtimeSession | None
        "speaker_thread": None,    # threading.Thread | None
        "speaker_stop": None,      # callable | None
    }
    if rt["enabled"]:
        if not realtime_api_key:
            state.set(
                error="realtime mode requested but no API key in HERMES_MEET_REALTIME_KEY/OPENAI_API_KEY",
                leave_reason="realtime_not_ready",
                exited=True,
                phase="exited",
            )
            return 6
        else:
            bridge = None
            try:
                from plugins.google_meet.audio_bridge import AudioBridge
                bridge = AudioBridge()
                rt["bridge_info"] = bridge.setup()
                rt["bridge"] = bridge
                state.set(realtime=True, realtime_device=rt["bridge_info"].get("device_name"))
            except Exception as e:
                if bridge is not None:
                    try:
                        bridge.teardown()
                    except Exception:
                        pass
                state.set(
                    error=f"audio bridge setup failed: {e}",
                    leave_reason="realtime_not_ready",
                    exited=True,
                    phase="exited",
                )
                return 6

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        state.set(error=f"playwright not installed: {e}", exited=True)
        sys.stderr.write(
            "google_meet bot: playwright is not installed. Run "
            "`pip install playwright && python -m playwright install chromium`\n"
        )
        _teardown_realtime(rt)
        return 3

    # Chrome env: if realtime is live on Linux, point PULSE_SOURCE at the
    # virtual source so Chrome's fake mic reads the audio we generate.
    chrome_env = os.environ.copy()
    chrome_args, permissions = _build_browser_launch_config(realtime_enabled=rt["enabled"])
    if rt["enabled"] and rt["bridge_info"] and rt["bridge_info"].get("platform") == "linux":
        chrome_env["PULSE_SOURCE"] = rt["bridge_info"].get("device_name", "")

    browser = None
    context = None
    page = None

    try:
        with sync_playwright() as pw:
            # Playwright's launch() doesn't take env; we set PULSE_SOURCE
            # via the process env before launch so the child Chrome inherits it.
            for k, v in chrome_env.items():
                os.environ[k] = v
            browser = pw.chromium.launch(
                headless=not headed,
                args=chrome_args,
            )
            context_args = {
                "viewport": {"width": 1280, "height": 800},
                "user_agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                ),
            }
            if auth_state and Path(auth_state).is_file():
                context_args["storage_state"] = auth_state
            if permissions:
                context_args["permissions"] = permissions
            context = browser.new_context(**context_args)
            page = context.new_page()

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            except Exception as e:
                state.set(error=f"navigate failed: {e}", exited=True)
                return 4

            if rt["enabled"]:
                _disable_local_media(page, disable_microphone=True, disable_camera=True)
                state.set(**_probe_local_media_state(page))
                _start_realtime_speaker(
                    rt=rt,
                    out_dir=out_dir,
                    bridge_info=rt["bridge_info"],
                    api_key=realtime_api_key,
                    model=realtime_model,
                    voice=realtime_voice,
                    instructions=realtime_instructions,
                    stop_flag=stop_flag,
                    state=state,
                )
                timeout_s = float(os.environ.get("HERMES_MEET_REALTIME_READY_TIMEOUT", "15"))
                if not _wait_for_realtime_route_ready(state, timeout_s=timeout_s):
                    return 6

            # Guest-mode: Meet shows a name field before "Ask to join". When
            # we're authed, we instead see "Join now".
            if not _ensure_local_media_before_join(
                page,
                state,
                realtime_enabled=rt["enabled"],
                realtime_route_ready=_realtime_route_ready(state),
            ):
                return 5
            _try_guest_name(page, guest_name)
            join_clicked = _click_join(page, state)

            # Install caption observer and attempt to enable captions.
            _retry_caption_enable(page, state)
            try:
                page.evaluate(_CAPTION_OBSERVER_JS)
            except Exception as e:
                state.set(error=f"caption observer install failed: {e}")

            # Note: in_call=False until admission is confirmed (we detect
            # either the Leave button or the caption region, signalling we
            # made it past the lobby).
            if join_clicked:
                state.set(join_attempted_at=time.time())

            # Admission + drain loop. Runs until SIGTERM, duration expiry,
            # or the page detects "You were removed / you left the
            # meeting". Responsible for:
            #   * detecting admission (Leave button visible → in_call=True)
            #   * timing out stuck-in-lobby (default 5 minutes)
            #   * draining scraped captions into the transcript
            #   * triggering realtime barge-in when a human speaks while
            #     the bot is generating audio
            #   * periodically flushing realtime counters into status.json
            deadline = (time.time() + duration_s) if duration_s else None
            lobby_deadline = time.time() + float(
                os.environ.get("HERMES_MEET_LOBBY_TIMEOUT", "300")
            )
            stall_after = float(os.environ.get("HERMES_MEET_STALL_AFTER", "90"))
            call_error_strike_limit = CALL_ERROR_STRIKE_LIMIT
            last_admission_check = 0.0
            last_caption_enable_check = 0.0
            last_media_disable_check = 0.0
            last_heartbeat_check = 0.0
            while not stop_flag["stop"]:
                now = time.time()
                ui_probe: dict = {}
                if deadline and now > deadline:
                    state.set(leave_reason="duration_expired", phase="exited")
                    break

                # Admission detection every ~3s until captions prove capture.
                if _should_probe_admission(
                    state,
                    now=now,
                    last_admission_check=last_admission_check,
                ):
                    last_admission_check = now
                    if not state.join_attempted_at:
                        if not _ensure_local_media_before_join(
                            page,
                            state,
                            realtime_enabled=rt["enabled"],
                            realtime_route_ready=_realtime_route_ready(state),
                        ):
                            return 5
                        _try_guest_name(page, guest_name)
                        if _click_join(page, state):
                            state.set(join_attempted_at=now)
                    ui_probe = _probe_meet_ui(page)
                    _, terminal = _apply_admission_probe(
                        state,
                        ui_probe,
                        now=now,
                        lobby_deadline=lobby_deadline,
                        call_error_strike_limit=call_error_strike_limit,
                    )
                    if terminal:
                        break

                if state.in_call and (now - last_media_disable_check) > 3.0:
                    last_media_disable_check = now
                    if rt["enabled"]:
                        if not _ensure_local_media_before_join(
                            page,
                            state,
                            realtime_enabled=True,
                            realtime_route_ready=_realtime_route_ready(state),
                            attempts=1,
                        ):
                            return 5
                    else:
                        _disable_local_media(page, disable_microphone=True)
                        state.set(**_probe_local_media_state(page))

                if _should_retry_caption_enable(state, now=now, last_caption_enable_check=last_caption_enable_check):
                    last_caption_enable_check = now
                    _retry_caption_enable(page, state)

                try:
                    queued = page.evaluate("window.__hermesMeetDrain && window.__hermesMeetDrain()")
                    if isinstance(queued, list):
                        for entry in queued:
                            if not isinstance(entry, dict):
                                continue
                            speaker = str(entry.get("speaker", ""))
                            text = str(entry.get("text", ""))
                            speaker_source = str(entry.get("speakerSource", ""))
                            speaker_debug = entry.get("speakerDebug")
                            caption_id = str(entry.get("captionId", ""))
                            state.record_caption(
                                speaker=speaker,
                                text=text,
                                speaker_source=speaker_source,
                                speaker_debug=speaker_debug if isinstance(speaker_debug, dict) else None,
                                caption_id=caption_id,
                            )
                            # Barge-in: if the bot is currently generating
                            # audio AND a real human just spoke, cancel the
                            # in-flight response so we don't talk over them.
                            if rt["enabled"] and rt["session"] is not None:
                                if _looks_like_human_speaker(speaker, guest_name):
                                    try:
                                        cancelled = rt["session"].cancel_response()
                                        if cancelled:
                                            state.set(last_barge_in_at=now)
                                    except Exception:
                                        pass
                except Exception:
                    # Meet reloaded or we got booted — try to detect and
                    # exit gracefully rather than spinning.
                    if page.is_closed():
                        state.set(leave_reason="page_closed")
                        break

                # Fold the realtime session's byte/timestamp counters into
                # the status file so meet_status can surface them.
                if rt["session"] is not None:
                    state.set(
                        audio_bytes_out=getattr(rt["session"], "audio_bytes_out", 0),
                        last_audio_out_at=getattr(rt["session"], "last_audio_out_at", None),
                    )

                if (now - last_heartbeat_check) > 5.0:
                    last_heartbeat_check = now
                    phase, stalled_reason = _compute_meet_phase(
                        state,
                        now=now,
                        stall_after=stall_after,
                    )
                    try:
                        current_url = page.url
                    except Exception:
                        current_url = None
                    state.heartbeat(
                        phase=phase,
                        stalled_reason=stalled_reason,
                        last_ui_text=ui_probe.get("text") if ui_probe else None,
                        last_url=ui_probe.get("url") if ui_probe else current_url,
                    )

                time.sleep(1.0)

            # Try to leave cleanly — click "Leave call" button if present.
            try:
                page.evaluate(
                    "() => { const b = document.querySelector('button[aria-label*=\"eave call\"]');"
                    " if (b) b.click(); }"
                )
            except Exception:
                pass

            state.set(in_call=False, captioning=False, exited=True, phase="exited")
            return 0

    except Exception as e:
        state.set(error=f"unhandled: {e}", exited=True, phase="exited")
        return 1
    finally:
        if context is not None:
            try:
                context.close()
            except Exception:
                pass
        if browser is not None:
            try:
                browser.close()
            except Exception:
                pass
        _teardown_realtime(rt, state)


def _try_guest_name(page, guest_name: str) -> None:
    """If Meet is showing a guest-name input, type *guest_name* into it."""
    selectors = (
        'input[aria-label*="name" i]',
        'input[placeholder*="name" i]',
        'input[name*="name" i]',
        'textarea[aria-label*="name" i]',
        '[contenteditable="true"][aria-label*="name" i]',
    )
    for selector in selectors:
        try:
            locator = _first_visible(page.locator(selector))
            if locator is not None:
                locator.fill(guest_name, timeout=2_000)
                return
        except Exception:
            continue


def _classify_meet_ui(
    text: str,
    *,
    leave: bool = False,
    caption_region: bool = False,
    in_call_control: bool = False,
    in_call_text: bool = False,
    waiting_lobby: bool = False,
    denied: bool = False,
    pre_join: bool = False,
    url: str = "",
) -> dict:
    """Classify a Meet page snapshot into admission-related states."""
    text = text or ""
    call_error = bool(
        re.search(r"couldn['’]?t start the video call because of an error", text, re.IGNORECASE)
        or re.search(r"could not start the video call because of an error", text, re.IGNORECASE)
    )
    landing = bool(
        re.search(r"/landing(?:[?#]|$)", url, re.IGNORECASE)
        or re.search(r"^https://workspace\.google\.com/products/meet/?(?:[?#]|$)", url, re.IGNORECASE)
        or (
            re.search(r"secure video conferencing for everyone", text, re.IGNORECASE)
            and re.search(r"\bnew meeting\b", text, re.IGNORECASE)
            and re.search(r"\bjoin\b", text, re.IGNORECASE)
        )
    )
    terminal_denied = bool(
        re.search(r"You can't join this video call", text, re.IGNORECASE)
        and (
            re.search(r"Returning to home screen", text, re.IGNORECASE)
            or re.search(r"Return to home screen", text, re.IGNORECASE)
        )
        or re.search(
            r"No one can join a meeting unless invited or admitted by the host",
            text,
            re.IGNORECASE,
        )
        or re.search(r"You were removed from the meeting", text, re.IGNORECASE)
        or re.search(r"No one responded to your request to join", text, re.IGNORECASE)
    )
    denied = bool(
        denied
        or terminal_denied
        or re.search(r"You can't join this video call", text, re.IGNORECASE)
        or re.search(r"You were removed from the meeting", text, re.IGNORECASE)
        or re.search(r"No one responded to your request to join", text, re.IGNORECASE)
        or re.search(
            r"No one can join a meeting unless invited or admitted by the host",
            text,
            re.IGNORECASE,
        )
    )
    waiting_lobby = bool(
        waiting_lobby
        or re.search(r"asking to be let in", text, re.IGNORECASE)
        or re.search(r"waiting for.*let you in", text, re.IGNORECASE)
        or re.search(r"you'll join.*when someone lets you in", text, re.IGNORECASE)
        or re.search(r"please wait until a meeting host brings you into the call", text, re.IGNORECASE)
        or re.search(r"ask to join", text, re.IGNORECASE)
    )
    pre_join = bool(
        pre_join
        or re.search(r"getting ready", text, re.IGNORECASE)
        or re.search(r"you'll be able to join in just a moment", text, re.IGNORECASE)
        or re.search(r"\bready to join\?", text, re.IGNORECASE)
        or re.search(r"continue without microphone and camera", text, re.IGNORECASE)
        or re.search(r"do you want people to see and hear you", text, re.IGNORECASE)
    )
    in_call = bool(leave or caption_region or in_call_control or in_call_text)
    in_call = in_call and not (waiting_lobby or denied or pre_join or landing)
    return {
        "inCall": in_call,
        "waitingLobby": waiting_lobby,
        "denied": denied,
        "terminalDenied": terminal_denied,
        "preJoin": pre_join,
        "callError": call_error,
        "landing": landing,
        "text": text[:1000],
        "url": url,
    }


def _probe_meet_ui(page) -> dict:
    """Return a best-effort snapshot of the current Meet UI."""
    probe = r"""
    (() => {
      const text = document.body ? document.body.innerText || '' : '';
      const url = location.href;
      const has = (selector) => !!document.querySelector(selector);
      const leave = has(
        'button[aria-label*="eave call" i], ' +
        'button[aria-label*="leave meeting" i], ' +
        '[role="button"][aria-label*="eave call" i]'
      );
      const inCallControl = has(
        '[aria-label*="meeting details" i], ' +
        '[aria-label*="show everyone" i], ' +
        '[aria-label*="people" i], ' +
        '[aria-label*="chat with everyone" i], ' +
        '[aria-label*="present now" i]'
      );
      let captionRegion = false;
      if (window.__hermesMeetInstalled) {
        captionRegion = has(
          '[role="region"][aria-label*="aption" i], ' +
          'div[jsname="YSxPC"], div[jsname="tgaKEf"]'
        );
      }
      const denied = (
        /You can't join this video call/i.test(text) ||
        /You were removed from the meeting/i.test(text) ||
        /No one responded to your request to join/i.test(text) ||
        /No one can join a meeting unless invited or admitted by the host/i.test(text)
      );
      const waitingLobby = (
        /asking to be let in/i.test(text) ||
        /waiting for.*let you in/i.test(text) ||
        /you'll join.*when someone lets you in/i.test(text) ||
        /ask to join/i.test(text)
      );
      const preJoin = (
        /getting ready/i.test(text) ||
        /you'll be able to join in just a moment/i.test(text) ||
        /\bready to join\?/i.test(text) ||
        /continue without microphone and camera/i.test(text) ||
        /do you want people to see and hear you/i.test(text)
      );
      const inCallText = (
        /you're the only one here/i.test(text) ||
        /you are the only one here/i.test(text) ||
        /meeting details/i.test(text) ||
        /chat with everyone/i.test(text)
      );
      return {
        leave,
        captionRegion,
        inCallControl,
        inCallText,
        waitingLobby,
        denied,
        preJoin,
        text: text.slice(0, 1000),
        url,
      };
    })();
    """
    try:
        result = page.evaluate(probe)
        if isinstance(result, dict):
            has_raw_signals = any(
                key in result
                for key in ("leave", "captionRegion", "inCallControl", "inCallText")
            )
            return _classify_meet_ui(
                str(result.get("text", "")),
                leave=bool(result.get("leave")),
                caption_region=bool(result.get("captionRegion")),
                in_call_control=bool(
                    result.get("inCallControl")
                    or (result.get("inCall") and not has_raw_signals)
                ),
                in_call_text=bool(result.get("inCallText")),
                waiting_lobby=bool(result.get("waitingLobby")),
                denied=bool(result.get("denied")),
                pre_join=bool(result.get("preJoin")),
                url=str(result.get("url", "")),
            )
        return {"inCall": bool(result), "waitingLobby": False, "denied": False}
    except Exception as e:
        return {"inCall": False, "waitingLobby": False, "denied": False, "error": str(e)}


def _detect_admission(page) -> bool:
    """True if we're clearly past the lobby and in the call itself."""
    return bool(_probe_meet_ui(page).get("inCall"))


def _detect_denied(page) -> bool:
    """True when Meet is showing a 'you were denied' / 'no one admitted' page."""
    return bool(_probe_meet_ui(page).get("denied"))


def _looks_like_human_speaker(speaker: str, bot_guest_name: str) -> bool:
    """Whether a caption line's speaker is probably a human, not our bot echo.

    Meet attributes captions to the speaker's display name. When Chrome is
    reading our fake mic, Meet still attributes captions to *our* bot name
    (because the bot is the one "speaking"). We don't want those to trigger
    barge-in. Anything else — real participant names — does.

    Conservative: unknown / blank speakers (common when caption scraping
    falls back to raw text) do NOT trigger barge-in, because we can't tell
    whether it was a human or us.
    """
    if not speaker or not speaker.strip():
        return False
    spk = speaker.strip().lower()
    if spk in {"unknown", "you", bot_guest_name.strip().lower()}:
        return False
    return True


def _click_join(page, state: _BotState) -> bool:
    """Click 'Join now' or 'Ask to join' if either button is visible.

    Flags ``lobby_waiting`` when we hit the "waiting for host to admit you"
    state so the agent can surface that in status.
    """
    try:
        continue_btn = page.get_by_role(
            "button",
            name="Continue without microphone and camera",
            exact=False,
        )
        continue_btn = _first_visible(continue_btn)
        if continue_btn is not None:
            continue_btn.click(timeout=3_000)
            page.wait_for_timeout(500)
    except Exception:
        pass

    for label in ("Join now", "Ask to join"):
        try:
            btn = _first_visible(page.get_by_role("button", name=label, exact=False))
            if btn is None and hasattr(page, "locator"):
                btn = _first_visible(
                    page.locator(
                        f"button:has-text('{label}'), [role='button']:has-text('{label}')"
                    )
                )
            if btn is None and hasattr(page, "get_by_text"):
                btn = _first_visible(page.get_by_text(label, exact=True))
            if btn is not None:
                btn.click(timeout=3_000)
                if label == "Ask to join":
                    state.set(lobby_waiting=True)
                return True
        except Exception:
            continue
    return False


def _parse_duration(raw: str) -> Optional[float]:
    """Parse ``30m`` / ``2h`` / ``90`` (seconds) → float seconds, or None."""
    if not raw:
        return None
    raw = raw.strip().lower()
    try:
        if raw.endswith("h"):
            return float(raw[:-1]) * 3600
        if raw.endswith("m"):
            return float(raw[:-1]) * 60
        if raw.endswith("s"):
            return float(raw[:-1])
        return float(raw)
    except ValueError:
        return None


if __name__ == "__main__":  # pragma: no cover — subprocess entry point
    sys.exit(run_bot())
