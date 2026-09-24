"""Headless Google Meet bot — Playwright + live-caption capture.

The bot is a standalone child process. It receives resolved ``HERMES_MEET_*``
child settings and uses the session directory as its only IPC surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from utils import atomic_json_write, atomic_write_text


MEET_URL_RE = re.compile(
    r"^https://meet\.google\.com/([a-z0-9]{3,}-[a-z0-9]{3,}-[a-z0-9]{3,}|lookup/[^/?#]+|new)"
    r"(?:[/?#].*)?$"
)
SAY_QUEUE_FILENAME = "say_queue.jsonl"
SAY_PCM_FILENAME = "speaker.pcm"
CAPTION_REVISION_WINDOW_SECONDS = 2.0
MAX_TRANSCRIPT_TEXT_LEN = 500
CALL_ERROR_STRIKE_LIMIT = 3
MEET_MEDIA_PROXY_BYPASS = "74.125.250.0/24,74.125.247.128,142.250.82.0/24"
MEET_WEBRTC_PROXY_POLICY = "--force-webrtc-ip-handling-policy=disable_non_proxied_udp"


def _is_safe_meet_url(url: str) -> bool:
    """Return whether *url* is a Google Meet URL the bot may navigate to."""
    return isinstance(url, str) and bool(MEET_URL_RE.match(url.strip()))


def _meeting_id_from_url(url: str) -> str:
    """Extract a Meet code or create an identifier for lookup/new URLs."""
    match = re.search(
        r"meet\.google\.com/([a-z0-9]{3,}-[a-z0-9]{3,}-[a-z0-9]{3,})", url or ""
    )
    return match.group(1) if match else f"meet-{int(time.time())}"


def _quiet(fn, *args, **kwargs):
    """Run a best-effort browser or teardown operation."""
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def _clicked(locator, *, timeout: int = 3_000) -> bool:
    return bool(_quiet(lambda: (locator.click(timeout=timeout), True)))


def _filled(locator, value: str) -> bool:
    return bool(_quiet(lambda: (locator.fill(value, timeout=2_000), True)))


def _debug_status_enabled() -> bool:
    return os.environ.get("HERMES_MEET_DEBUG_STATUS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _float_env(value: str, default: float, *, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(value))
    except (TypeError, ValueError):
        return default


_STATUS_FIELDS = (
    ("meetingId", "meeting_id", None),
    ("url", "url", None),
    ("inCall", "in_call", False),
    ("captioning", "captioning", False),
    ("captionsEnabledAttempted", "captions_enabled_attempted", False),
    ("lobbyWaiting", "lobby_waiting", False),
    ("joinAttemptedAt", "join_attempted_at", None),
    ("joinedAt", "joined_at", None),
    ("lastCaptionAt", "last_caption_at", None),
    ("transcriptLines", "transcript_lines", 0),
    ("error", "error", None),
    ("exited", "exited", False),
    ("phase", "phase", "starting"),
    ("lastHeartbeatAt", "last_heartbeat_at", None),
    ("lastProgressAt", "last_progress_at", None),
    ("stalledReason", "stalled_reason", None),
    ("lastUiText", "last_ui_text", None),
    ("lastUrl", "last_url", None),
    ("lastSpeakerSource", "last_speaker_source", None),
    ("lastSpeakerCandidates", "last_speaker_candidates", None),
    ("localMicrophoneOn", "local_microphone_on", None),
    ("localCameraOn", "local_camera_on", None),
    ("realtime", "realtime", False),
    ("realtimeReady", "realtime_ready", False),
    ("realtimeDevice", "realtime_device", None),
    ("realtimeAudioPumpStatus", "realtime_audio_pump_status", "disabled"),
    ("realtimeAudioPumpTool", "realtime_audio_pump_tool", None),
    ("realtimeAudioPumpPid", "realtime_audio_pump_pid", None),
    ("realtimeAudioPumpReturnCode", "realtime_audio_pump_return_code", None),
    ("realtimeAudioPumpError", "realtime_audio_pump_error", None),
    ("audioBytesOut", "audio_bytes_out", 0),
    ("lastAudioOutAt", "last_audio_out_at", None),
    ("lastBargeInAt", "last_barge_in_at", None),
    ("leaveReason", "leave_reason", None),
    ("micState", "mic_state", None),
    ("unresolvedCaptionDrops", "unresolved_caption_drops", 0),
    ("unresolvedCaptionLines", "unresolved_caption_lines", 0),
    ("captionUiNoiseDrops", "caption_ui_noise_drops", 0),
    ("lastUnresolvedCaptionAt", "last_unresolved_caption_at", None),
)

_CAPTION_UI_NOISE = {
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
}


@dataclass
class _CaptionGroup:
    """One mutable logical utterance, independent of DOM-row churn."""

    group_id: str
    speaker: str
    text: str
    timestamp: str
    updated_at: float
    row_keys: set[str] = field(default_factory=set)


def _caption_text_is_ui_noise(text: str) -> bool:
    """Filter only recognized UI chrome; missing speakers remain legitimate captions."""
    normalized = " ".join((text or "").split()).strip().lower()
    if not normalized or normalized in _CAPTION_UI_NOISE:
        return True
    return "open caption settings" in normalized and any(
        marker in normalized
        for marker in ("font size", "font colour", "font color", "format_size")
    )


class _BotState:
    """Thread-safe status and mutable canonical transcript state."""

    out_dir: Path
    transcript_path: Path
    status_path: Path
    caption_debug_path: Path
    meeting_id: str
    url: str
    in_call: bool
    captioning: bool
    captions_enabled_attempted: bool
    lobby_waiting: bool
    join_attempted_at: Optional[float]
    joined_at: Optional[float]
    last_caption_at: Optional[float]
    transcript_lines: int
    error: Optional[str]
    exited: bool
    phase: str
    last_heartbeat_at: Optional[float]
    last_progress_at: Optional[float]
    stalled_reason: Optional[str]
    last_ui_text: Optional[str]
    last_url: Optional[str]
    last_speaker_source: Optional[str]
    last_speaker_candidates: list[str]
    local_microphone_on: Optional[bool]
    local_camera_on: Optional[bool]
    realtime: bool
    realtime_ready: bool
    realtime_device: Optional[str]
    realtime_audio_pump_status: str
    realtime_audio_pump_tool: Optional[str]
    realtime_audio_pump_pid: Optional[int]
    realtime_audio_pump_return_code: Optional[int]
    realtime_audio_pump_error: Optional[str]
    audio_bytes_out: int
    last_audio_out_at: Optional[float]
    last_barge_in_at: Optional[float]
    leave_reason: Optional[str]
    mic_state: Optional[str]
    unresolved_caption_drops: int
    unresolved_caption_lines: int
    caption_ui_noise_drops: int
    last_unresolved_caption_at: Optional[float]
    call_error_strikes: int
    ever_admitted: bool
    _caption_groups: list[_CaptionGroup]
    _caption_group_by_key: dict[str, _CaptionGroup]
    _seen_caption_versions: set[str]
    _next_caption_group_id: int

    def __init__(self, out_dir: Path, meeting_id: str, url: str):
        self._lock = threading.RLock()
        self.__dict__.update(
            {
                attribute: default
                for _, attribute, default in _STATUS_FIELDS
                if attribute
            }
        )
        now = time.time()
        self.__dict__.update(
            out_dir=Path(out_dir),
            meeting_id=meeting_id,
            url=url,
            transcript_path=Path(out_dir) / "transcript.txt",
            status_path=Path(out_dir) / "status.json",
            caption_debug_path=Path(out_dir) / "caption_debug.jsonl",
            last_heartbeat_at=now,
            last_progress_at=now,
            last_speaker_candidates=[],
            call_error_strikes=0,
            ever_admitted=False,
            _caption_groups=[],
            _caption_group_by_key={},
            _seen_caption_versions=set(),
            _next_caption_group_id=1,
        )
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._flush()

    @staticmethod
    def _normalized_text(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    @classmethod
    def _caption_tokens(cls, text: str) -> list[str]:
        return re.findall(r"[\w']+", cls._normalized_text(text).lower())

    @staticmethod
    def _common_prefix_len(left: str, right: str) -> int:
        for index, (left_char, right_char) in enumerate(zip(left, right)):
            if left_char != right_char:
                return index
        return min(len(left), len(right))

    @staticmethod
    def _matching_token_prefix(previous: list[str], current: list[str]) -> int:
        matched = 0
        for index, (left, right) in enumerate(zip(previous, current)):
            if left == right:
                matched += 1
                continue
            if (
                index == min(len(previous), len(current)) - 1
                and min(len(left), len(right)) >= 3
                and (left.startswith(right) or right.startswith(left))
            ):
                matched += 1
            break
        return matched

    @staticmethod
    def _matching_token_suffix(previous: list[str], current: list[str]) -> int:
        matched = 0
        for left, right in zip(reversed(previous), reversed(current)):
            if left != right:
                break
            matched += 1
        return matched

    @classmethod
    def _is_caption_revision(cls, previous: str, current: str) -> bool:
        previous = cls._normalized_text(previous)
        current = cls._normalized_text(current)
        if not previous or not current:
            return False
        previous_lower, current_lower = previous.lower(), current.lower()
        if previous_lower == current_lower or current_lower.startswith(previous_lower):
            return True
        previous_tokens, current_tokens = (
            cls._caption_tokens(previous),
            cls._caption_tokens(current),
        )
        if not previous_tokens or not current_tokens:
            return False
        prefix = cls._matching_token_prefix(previous_tokens, current_tokens)
        suffix = cls._matching_token_suffix(previous_tokens, current_tokens)
        shortest = min(len(previous_tokens), len(current_tokens))
        if prefix == len(previous_tokens):
            return True
        if prefix >= max(3, (shortest * 4 + 4) // 5):
            return True
        if prefix >= 3 and suffix >= 3 and prefix + suffix >= (shortest * 3 + 3) // 4:
            return True
        matching_tokens = sum(
            block.size
            for block in SequenceMatcher(
                None, previous_tokens, current_tokens, autojunk=False
            ).get_matching_blocks()
        )
        if (
            len(current_tokens) > len(previous_tokens)
            and prefix >= 3
            and matching_tokens > prefix
            and matching_tokens * 5 >= len(previous_tokens) * 4
        ):
            return True
        if prefix >= 12 and len(current_tokens) > len(previous_tokens):
            return True
        char_prefix = cls._common_prefix_len(previous_lower, current_lower)
        return char_prefix >= max(4, min(24, len(previous_lower) // 2)) and (
            SequenceMatcher(None, previous_lower, current_lower).ratio() >= 0.88
        )

    @classmethod
    def _revision_score(cls, previous: str, current: str) -> int:
        previous_tokens, current_tokens = (
            cls._caption_tokens(previous),
            cls._caption_tokens(current),
        )
        prefix = cls._matching_token_prefix(previous_tokens, current_tokens)
        suffix = cls._matching_token_suffix(previous_tokens, current_tokens)
        matching_tokens = sum(
            block.size
            for block in SequenceMatcher(
                None, previous_tokens, current_tokens, autojunk=False
            ).get_matching_blocks()
        )
        ratio = int(
            SequenceMatcher(None, previous.lower(), current.lower()).ratio() * 100
        )
        return (
            prefix * 10_000
            + suffix * 1_000
            + matching_tokens * 100
            + cls._common_prefix_len(previous.lower(), current.lower())
            + ratio
        )

    @staticmethod
    def _split_caption_text(text: str) -> list[str]:
        text = (text or "").strip()
        chunks: list[str] = []
        while len(text) > MAX_TRANSCRIPT_TEXT_LEN:
            split_at = text.rfind(" ", 0, MAX_TRANSCRIPT_TEXT_LEN + 1)
            if split_at <= 0:
                split_at = MAX_TRANSCRIPT_TEXT_LEN
            chunks.append(text[:split_at].strip())
            text = text[split_at:].strip()
        return [*chunks, text] if text else chunks

    def _new_caption_group(
        self, speaker: str, text: str, timestamp: str, row_key: str
    ) -> _CaptionGroup:
        group = _CaptionGroup(
            f"caption-group-{self._next_caption_group_id}",
            speaker,
            text,
            timestamp,
            time.monotonic(),
            {row_key},
        )
        self._next_caption_group_id += 1
        self._caption_groups.append(group)
        self._caption_group_by_key[row_key] = group
        return group

    def _recent_revision_group(
        self, speaker: str, text: str, row_key: str
    ) -> Optional[_CaptionGroup]:
        # An unresolved label cannot establish identity across separate DOM rows.
        if speaker == "Unresolved speaker":
            return None
        now = time.monotonic()
        candidates = [
            (self._revision_score(group.text, text), group)
            for group in reversed(self._caption_groups[-24:])
            if group.speaker == speaker
            and row_key not in group.row_keys
            and now - group.updated_at <= CAPTION_REVISION_WINDOW_SECONDS
            and self._is_caption_revision(group.text, text)
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        return (
            None
            if len(candidates) > 1 and candidates[0][0] == candidates[1][0]
            else candidates[0][1]
        )

    def _resolve_caption_group(
        self, speaker: str, text: str, row_key: str
    ) -> Optional[_CaptionGroup]:
        group = self._caption_group_by_key.get(row_key)
        if group is not None:
            return group if self._is_caption_revision(group.text, text) else None
        return self._recent_revision_group(speaker, text, row_key)

    def _rewrite_transcript_locked(self) -> None:
        lines = [
            f"[{group.timestamp}] {group.speaker}: {chunk}\n"
            for group in self._caption_groups
            for chunk in self._split_caption_text(group.text)
        ]
        atomic_write_text(self.transcript_path, "".join(lines))
        self.transcript_lines = len(lines)

    def _touch_caption_progress_locked(self) -> str:
        self.last_caption_at = self.last_progress_at = time.time()
        self.phase, self.stalled_reason = "capturing", None
        if not self.in_call:
            self.in_call, self.lobby_waiting, self.joined_at = (
                True,
                False,
                self.last_caption_at,
            )
        return time.strftime("%H:%M:%S", time.localtime(self.last_caption_at))

    def _upsert_caption_group_locked(
        self, speaker: str, text: str, timestamp: str, row_key: str
    ) -> None:
        group = self._resolve_caption_group(speaker, text, row_key)
        if group is None:
            self._new_caption_group(speaker, text, timestamp, row_key)
            self._rewrite_transcript_locked()
            return
        group.row_keys.add(row_key)
        self._caption_group_by_key[row_key] = group
        if (
            self._normalized_text(group.text).lower()
            == self._normalized_text(text).lower()
        ):
            return
        group.speaker, group.text, group.timestamp, group.updated_at = (
            speaker,
            text,
            timestamp,
            time.monotonic(),
        )
        self._rewrite_transcript_locked()

    def record_caption(
        self,
        speaker: str,
        text: str,
        *,
        speaker_source: Optional[str] = None,
        speaker_debug: Optional[dict] = None,
        caption_id: Optional[str] = None,
    ) -> None:
        """Reconcile one partial, interleaved, or replacement caption update."""
        with self._lock:
            text, speaker = self._normalized_text(text), self._normalized_text(speaker)
            unresolved = not speaker or speaker.lower() == "unknown"
            if speaker_source:
                self.last_speaker_source = speaker_source
            if not text:
                return
            if _caption_text_is_ui_noise(text):
                self.caption_ui_noise_drops += 1
                if unresolved:
                    self.unresolved_caption_drops += 1
                    self.last_unresolved_caption_at = time.time()
                self._flush_locked()
                return
            if _debug_status_enabled() and isinstance(speaker_debug, dict):
                candidates = speaker_debug.get("candidates")
                if isinstance(candidates, list):
                    self.last_speaker_candidates = candidates[:20]
                with self.caption_debug_path.open("a", encoding="utf-8") as debug_file:
                    debug_file.write(
                        json.dumps(
                            {
                                "ts": time.time(),
                                "text": text[:300],
                                "speakerSource": speaker_source,
                                "speakerDebug": speaker_debug,
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )
            if unresolved:
                speaker = "Unresolved speaker"
                self.unresolved_caption_lines += 1
                self.last_unresolved_caption_at = time.time()
            caption_id = (caption_id or "").strip()
            row_key = (
                f"row:{caption_id}"
                if caption_id
                else "unresolved:" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
                if unresolved
                else f"speaker:{speaker}"
            )
            version_key = f"{row_key}\0{text.lower()}"
            if version_key in self._seen_caption_versions:
                return
            self._seen_caption_versions.add(version_key)
            self._upsert_caption_group_locked(
                speaker, text, self._touch_caption_progress_locked(), row_key
            )
            self._flush_locked()

    def heartbeat(
        self,
        *,
        phase: str,
        stalled_reason: Optional[str],
        last_ui_text: Optional[str],
        last_url: Optional[str],
    ) -> None:
        with self._lock:
            self.phase, self.stalled_reason, self.last_heartbeat_at = (
                phase,
                stalled_reason,
                time.time(),
            )
            self.last_ui_text = (
                self._normalized_text(str(last_ui_text))[:1000]
                if _debug_status_enabled() and last_ui_text
                else None
            )
            if last_url is not None:
                self.last_url = str(last_url)
            self._flush_locked()

    def _flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        debug_enabled = _debug_status_enabled()
        data = {
            key: getattr(self, attribute)
            for key, attribute, _ in _STATUS_FIELDS
            if attribute
        }
        data.update(
            transcriptPath=str(self.transcript_path),
            captionDebugPath=str(self.caption_debug_path) if debug_enabled else None,
            lastUiText=self.last_ui_text if debug_enabled else None,
            lastSpeakerCandidates=self.last_speaker_candidates if debug_enabled else [],
            pid=os.getpid(),
        )
        atomic_json_write(self.status_path, data)

    def set(self, **kwargs) -> None:
        with self._lock:
            self.__dict__.update(kwargs)
            self._flush_locked()


_CAPTION_OBSERVER_JS = r"""
(() => {
  if (window.__hermesMeetInstalled) return;
  window.__hermesMeetInstalled = true;
  window.__hermesMeetQueue = [];
  window.__hermesMeetCaptionRows = new WeakMap();
  window.__hermesMeetCaptionLast = {};
  window.__hermesMeetFallbackIds = {};
  window.__hermesMeetCaptionNextId = 1;
  window.__hermesMeetCaptionRegionAttached = false;
  const rootSelector = '[role="region"][aria-label*="aption" i], div[jsname="YSxPC"], div[jsname="tgaKEf"]';
  const rowSelector = 'div[jsname="dsyhDe"], div.CNusmb, div.TBMuR, div.nMcdL';
  const speakerSelector = 'div.KcIKyf, div.zs7s8d, span.NWpY1d, span[jsname="YSxPC"]';
  const textSelector = 'div.bh44bd, span[jsname="tgaKEf"], div.iTTPOb';
  const normalize = value => (value || '').replace(/\s+/g, ' ').trim();
  const escapeRegExp = value => value.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
  function cleanSpeaker(value) {
    const speaker = normalize(value).replace(/^(?:Pin|More options for)\s+/i, '');
    const lower = speaker.toLowerCase();
    if (!speaker || lower === 'unknown' || lower === 'you' || lower.includes('switch account')) return '';
    return speaker.replace(/(?:,|\s+)(?:is\s+)?speaking\b.*$/i, '').replace(/\s+to your main screen$/i, '').trim();
  }
  function rowId(row) {
    if (!window.__hermesMeetCaptionRows.has(row)) {
      window.__hermesMeetCaptionRows.set(row, 'caption-row-' + window.__hermesMeetCaptionNextId++);
    }
    return window.__hermesMeetCaptionRows.get(row);
  }
  function fallbackId(speaker, ordinal) {
    const key = (speaker || 'unresolved') + '\\u0000' + ordinal;
    if (!window.__hermesMeetFallbackIds[key]) {
      window.__hermesMeetFallbackIds[key] = 'caption-fallback-' + window.__hermesMeetCaptionNextId++;
    }
    return window.__hermesMeetFallbackIds[key];
  }
  function push(id, speaker, text, source, debug, emitCaptionId = true) {
    text = normalize(text);
    if (!text) return;
    const previous = window.__hermesMeetCaptionLast[id];
    if (previous && previous.speaker === speaker && previous.text === text) return;
    window.__hermesMeetCaptionLast[id] = {speaker, text};
    const entry = {
      ts: Date.now(), speaker, text,
      speakerSource: source || (speaker ? 'captionRow' : 'unresolved'),
      speakerDebug: debug || {candidates: []},
    };
    if (emitCaptionId) entry.captionId = id;
    window.__hermesMeetQueue.push(entry);
  }
  function visibleRows(root) {
    if (!root || !root.querySelectorAll) return [];
    const rows = Array.from(root.querySelectorAll(rowSelector));
    const known = new Set(rows);
    Array.from(root.querySelectorAll('span.NWpY1d, .NWpY1d')).forEach(label => {
      const row = (label.closest && label.closest(rowSelector)) || label.parentElement?.parentElement || label.parentElement;
      if (row) known.add(row);
    });
    return Array.from(known);
  }
  function rowCaption(row) {
    const labels = row.querySelectorAll ? Array.from(row.querySelectorAll(speakerSelector)) : [];
    const label = labels.find(node => cleanSpeaker(node.innerText || ''));
    const speaker = cleanSpeaker(label ? label.innerText : '');
    const textNode = row.querySelector ? row.querySelector(textSelector) : null;
    let text = normalize(textNode && textNode !== label ? textNode.innerText : '');
    if (!text && row.children) {
      const children = Array.from(row.children).map(node => normalize(node.innerText)).filter(Boolean);
      text = children.find(value => value !== speaker) || '';
    }
    if (!text) text = normalize(row.innerText);
    if (speaker && text.toLowerCase().startsWith(speaker.toLowerCase())) {
      text = normalize(text.slice(speaker.length));
    }
    return {speaker, text};
  }
  function speakerCandidates() {
    const candidates = [], selectors = [
      ['[aria-label*="speaking" i]', false],
      ['[aria-label*="is speaking" i]', false],
      ['[aria-label]', true],
    ];
    let chosen = null;
    selectors.forEach(([selector, diagnosticOnly]) => {
      Array.from(document.querySelectorAll ? document.querySelectorAll(selector) : []).forEach(node => {
        const raw = node.getAttribute ? node.getAttribute('aria-label') || '' : '';
        const clean = cleanSpeaker(raw);
        candidates.push({selector, attr: 'aria-label', raw, clean, diagnosticOnly});
        if (!diagnosticOnly) {
          const innerText = normalize(node.innerText);
          candidates.push({selector, attr: 'innerText', raw: innerText, clean: cleanSpeaker(innerText), diagnosticOnly});
        }
        if (!chosen && clean && !diagnosticOnly) chosen = {speaker: clean, source: selector};
      });
    });
    return {speaker: chosen ? chosen.speaker : '', source: chosen ? chosen.source : 'unresolved', candidates};
  }
  function namesFromBody() {
    const names = [], body = document.body ? document.body.innerText || '' : '';
    const expression = /(?:Pin|More options for)\s+(.+?)(?:\s+to your main screen|\n|$)/gi;
    let match;
    while ((match = expression.exec(body))) {
      const name = cleanSpeaker(match[1]);
      if (name && !names.includes(name)) names.push(name);
    }
    return names;
  }
  function stripChrome(text) {
    let value = normalize(text), match = /open caption settings\s*/i.exec(value);
    if (match) value = value.slice(match.index + match[0].length);
    match = /\b(?:keyboard_arrow_up|audio settings|mic_off|turn on microphone|turn on camera)\b/i.exec(value);
    if (match) value = value.slice(0, match.index);
    return normalize(value);
  }
  function fallbackSegments(root) {
    const text = stripChrome(root ? root.innerText : document.body ? document.body.innerText : '');
    const names = namesFromBody(), inferred = speakerCandidates();
    if (!text) return [];
    if (names.length) {
      const expression = new RegExp('(' + names.map(escapeRegExp).join('|') + ')', 'gi');
      const hits = [], lowerNames = names.map(name => name.toLowerCase());
      let match;
      while ((match = expression.exec(text))) {
        const index = lowerNames.indexOf(match[0].toLowerCase());
        if (index >= 0) hits.push({index: match.index, end: expression.lastIndex, speaker: names[index]});
      }
      if (hits.length) {
        return hits.map((hit, index) => ({
          speaker: hit.speaker,
          text: normalize(text.slice(hit.end, index + 1 < hits.length ? hits[index + 1].index : text.length)),
          source: 'captionRow', debug: {candidates: []}, ordinal: index, stableId: true,
        })).filter(segment => segment.text);
      }
      return [{speaker: names[0], text, source: 'captionRow', debug: {candidates: []}, ordinal: 0, stableId: true}];
    }
    return [{speaker: inferred.speaker, text, source: inferred.source, debug: {candidates: inferred.candidates}, ordinal: 0, stableId: false}];
  }
  function scan(root) {
    const rows = visibleRows(root);
    if (rows.length) {
      rows.forEach(row => {
        const caption = rowCaption(row);
        push(rowId(row), caption.speaker, caption.text, caption.speaker ? 'captionRow' : 'unresolved', {candidates: []});
      });
      return;
    }
    fallbackSegments(root).forEach(segment => {
      push(fallbackId(segment.speaker, segment.ordinal), segment.speaker, segment.text, segment.source, segment.debug, segment.stableId);
    });
  }
  let observed = null;
  function attach() {
    const root = document.querySelector ? document.querySelector(rootSelector) : null;
    if (root) {
      window.__hermesMeetCaptionRoot = root;
      window.__hermesMeetCaptionRegionAttached = true;
    }
    const target = root || document.body;
    if (target && target !== observed) {
      observed = target;
      new MutationObserver(() => scan(window.__hermesMeetCaptionRoot || null)).observe(target, {childList: true, subtree: true, characterData: true});
    }
    scan(root || null);
    return !!root;
  }
  attach();
  setInterval(() => attach(), 1500);
  window.__hermesMeetDrain = () => {
    const entries = window.__hermesMeetQueue.slice();
    window.__hermesMeetQueue = [];
    return entries;
  };
})();
"""

_ENABLE_CAPTIONS_JS = "(() => { document.body.dispatchEvent(new KeyboardEvent('keydown', {key: 'c', code: 'KeyC', bubbles: true})); return true; })();"
_LEAVE_CALL_JS = "() => { const button = document.querySelector('button[aria-label*=\"eave call\"]'); if (button) button.click(); }"
_ADMISSION_PROBE_JS = r"""(() => !!document.querySelector('button[aria-label*="eave call" i], [aria-label*="participants" i], [aria-label*="show everyone" i]'))();"""
_DENIED_PROBE_JS = r"""(() => /You can't join this video call|You were removed from the meeting|No one responded to your request to join/i.test(document.body ? document.body.innerText || '' : ''))();"""
_MEET_UI_PROBE_JS = r"""
(() => {
  const text = document.body ? document.body.innerText || '' : '', has = selector => !!document.querySelector(selector);
  return {leave: has('button[aria-label*="eave call" i], [role="button"][aria-label*="eave call" i]'), captionRegion: !!window.__hermesMeetCaptionRegionAttached && has('[role="region"][aria-label*="aption" i], div[jsname="YSxPC"], div[jsname="tgaKEf"]'), inCallControl: has('[aria-label*="meeting details" i], [aria-label*="show everyone" i], [aria-label*="chat with everyone" i], [aria-label*="present now" i]'), text: text.slice(0, 1000), url: location.href};
})();
"""


def _page_call(page, method: str, *args, **kwargs):
    target = getattr(page, method, None)
    return _quiet(target, *args, **kwargs) if callable(target) else None


def _probe(page, script: str) -> bool:
    return bool(_page_call(page, "evaluate", script))


def _visible(locator):
    def first_visible():
        count = int(locator.count())
        for index in range(count):
            candidate = (
                locator.nth(index)
                if callable(getattr(locator, "nth", None))
                else locator.first
                if index == 0
                else None
            )
            if candidate is not None and candidate.is_visible():
                return candidate
        return None

    return _quiet(first_visible)


def _try_guest_name(page, guest_name: str) -> None:
    for selector in (
        'input[aria-label*="name" i]',
        'input[placeholder*="name" i]',
        'input[name*="name" i]',
        'textarea[aria-label*="name" i]',
        '[contenteditable="true"][aria-label*="name" i]',
    ):
        locator = _page_call(page, "locator", selector)
        if (
            locator is not None
            and (visible := _visible(locator)) is not None
            and _filled(visible, guest_name)
        ):
            return


def _click_join(page, state: _BotState) -> bool:
    continue_button = _page_call(
        page,
        "get_by_role",
        "button",
        name="Continue without microphone and camera",
        exact=False,
    )
    if (
        continue_button is not None
        and (visible := _visible(continue_button)) is not None
    ):
        _clicked(visible)
        _page_call(page, "wait_for_timeout", 500)
    for label in ("Join now", "Ask to join"):
        button = _page_call(page, "get_by_role", "button", name=label, exact=False)
        visible = _visible(button) if button is not None else None
        if visible is None:
            by_text = _page_call(page, "get_by_text", label, exact=True)
            visible = _visible(by_text) if by_text is not None else None
        if visible is None:
            locator = _page_call(
                page,
                "locator",
                f"button:has-text('{label}'), [role='button']:has-text('{label}')",
            )
            visible = _visible(locator) if locator is not None else None
        if visible is not None and _clicked(visible):
            if label == "Ask to join":
                state.set(lobby_waiting=True, phase="waiting_lobby")
            return True
    return False


def _join(page, cfg: "_BotConfig", state: _BotState, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        _try_guest_name(page, cfg.guest_name)
        if _click_join(page, state):
            return True
        time.sleep(0.5)
    return False


def _classify_meet_ui(
    text: str,
    *,
    leave: bool = False,
    caption_region: bool = False,
    in_call_control: bool = False,
    waiting_lobby: bool = False,
    denied: bool = False,
    pre_join: bool = False,
    url: str = "",
) -> dict:
    text = text or ""
    terminal_denied = bool(
        re.search(r"You can't join this video call", text, re.IGNORECASE)
        and re.search(
            r"returning to home screen|return to home screen", text, re.IGNORECASE
        )
        or re.search(
            r"No one can join a meeting unless invited or admitted by the host|You were removed from the meeting|No one responded to your request to join",
            text,
            re.IGNORECASE,
        )
    )
    waiting_lobby = bool(
        waiting_lobby
        or re.search(
            r"asking to be let in|waiting for.*let you in|you'll join.*when someone lets you in|ask to join|please wait until a meeting host brings you into the call|waiting for (?:the )?meeting host|waiting for host",
            text,
            re.IGNORECASE,
        )
    )
    pre_join = bool(
        pre_join
        or re.search(
            r"getting ready|you'll be able to join in just a moment|\bready to join\?|continue without microphone and camera",
            text,
            re.IGNORECASE,
        )
    )
    denied = bool(
        denied
        or terminal_denied
        or re.search(
            r"You can't join this video call|You were removed from the meeting|No one responded to your request to join",
            text,
            re.IGNORECASE,
        )
    )
    landing = bool(
        re.search(r"/landing(?:[?#]|$)", url, re.IGNORECASE)
        or re.search(
            r"^https://workspace\.google\.com/products/meet/?(?:[?#]|$)",
            url,
            re.IGNORECASE,
        )
    )
    return {
        "inCall": bool(leave or caption_region or in_call_control)
        and not (waiting_lobby or denied or pre_join or landing),
        "waitingLobby": waiting_lobby,
        "denied": denied,
        "terminalDenied": terminal_denied,
        "preJoin": pre_join,
        "landing": landing,
        "callError": bool(
            re.search(
                r"couldn['’]?t start the video call because of an error",
                text,
                re.IGNORECASE,
            )
        ),
        "text": text[:1000],
        "url": url,
    }


def _probe_meet_ui(page) -> dict:
    result = _page_call(page, "evaluate", _MEET_UI_PROBE_JS)
    if isinstance(result, dict):
        return _classify_meet_ui(
            str(result.get("text", "")),
            leave=bool(result.get("leave")),
            caption_region=bool(result.get("captionRegion")),
            in_call_control=bool(result.get("inCallControl")),
            waiting_lobby=bool(result.get("waitingLobby")),
            denied=bool(result.get("denied")),
            pre_join=bool(result.get("preJoin")),
            url=str(result.get("url", "")),
        )
    return _classify_meet_ui(
        "",
        in_call_control=bool(result) or _probe(page, _ADMISSION_PROBE_JS),
        denied=_probe(page, _DENIED_PROBE_JS),
        url=str(getattr(page, "url", "") or ""),
    )


def _captions_are_enabled(page) -> bool:
    for selector, name in (
        (
            'button[aria-label*="turn off captions" i]',
            re.compile(r"turn off captions", re.IGNORECASE),
        ),
        ('[aria-label*="captions on" i]', re.compile(r"captions on", re.IGNORECASE)),
    ):
        button = _page_call(page, "get_by_role", "button", name=name)
        if button is not None and _visible(button) is not None:
            return True
        locator = _page_call(page, "locator", selector)
        if locator is not None and _visible(locator) is not None:
            return True
    return False


def _page_effect(page, method: str, *args, **kwargs) -> bool:
    target = getattr(page, method, None)
    return (
        bool(_quiet(lambda: (target(*args, **kwargs), True)))
        if callable(target)
        else False
    )


def _caption_enable_control(page) -> bool:
    button = _page_call(
        page,
        "get_by_role",
        "button",
        name=re.compile(r"turn on captions", re.IGNORECASE),
        exact=False,
    )
    if (
        button is not None
        and (visible := _visible(button)) is not None
        and _clicked(visible)
    ):
        return True
    locator = _page_call(page, "locator", '[aria-label*="turn on captions" i]')
    return bool(
        locator is not None
        and (visible := _visible(locator)) is not None
        and _clicked(visible)
    )


def _retry_caption_enable(page, state: _BotState, *, after_join: bool = False) -> bool:
    if not (state.in_call or after_join):
        return False
    if _captions_are_enabled(page):
        state.set(captioning=True, captions_enabled_attempted=True)
        return True
    attempted = _caption_enable_control(page)
    keyboard = getattr(page, "keyboard", None)
    if not attempted and keyboard is not None:
        _quiet(keyboard.press, "ArrowDown")
        _page_call(page, "wait_for_timeout", 250)
        attempted = _caption_enable_control(page)
    if not attempted:
        attempted = _page_effect(keyboard, "press", "c")
    if attempted:
        state.set(captions_enabled_attempted=True)
        _page_call(page, "wait_for_timeout", 500)
    if _captions_are_enabled(page):
        state.set(captioning=True, captions_enabled_attempted=True)
        return True
    if state.in_call and _page_effect(page, "evaluate", _ENABLE_CAPTIONS_JS):
        state.set(captions_enabled_attempted=True)
        _page_call(page, "wait_for_timeout", 500)
        if _captions_are_enabled(page):
            state.set(captioning=True, captions_enabled_attempted=True)
            return True
    return False


_MEDIA_STATE_PROBE_JS = r"""
(() => {
  const body = (document.body && document.body.innerText || '').toLowerCase();
  const buttons = Array.from(document.querySelectorAll('button,[role="button"]'));
  const has = phrase => buttons.some(button => (button.getAttribute('aria-label') || '').toLowerCase().includes(phrase)) || body.includes(phrase);
  const state = (on, off) => has(on) ? true : has(off) ? false : null;
  return {localMicrophoneOn: state('turn off microphone', 'turn on microphone'), localCameraOn: state('turn off camera', 'turn on camera')};
})();
"""


def _probe_local_media_state(page) -> dict:
    controls = {
        "local_microphone_on": (
            "localMicrophoneOn",
            re.compile(r"turn off microphone", re.IGNORECASE),
            re.compile(r"turn on microphone", re.IGNORECASE),
            'button[aria-label*="turn off microphone" i]',
            'button[aria-label*="turn on microphone" i]',
        ),
        "local_camera_on": (
            "localCameraOn",
            re.compile(r"turn off camera", re.IGNORECASE),
            re.compile(r"turn on camera", re.IGNORECASE),
            'button[aria-label*="turn off camera" i]',
            'button[aria-label*="turn on camera" i]',
        ),
    }
    result = {}
    for key, (
        live_key,
        on_name,
        off_name,
        on_selector,
        off_selector,
    ) in controls.items():
        on = _page_call(page, "get_by_role", "button", name=on_name)
        if on is None or _visible(on) is None:
            on = _page_call(page, "locator", on_selector)
        off = _page_call(page, "get_by_role", "button", name=off_name)
        if off is None or _visible(off) is None:
            off = _page_call(page, "locator", off_selector)
        result[key] = (
            True
            if on is not None and _visible(on) is not None
            else False
            if off is not None and _visible(off) is not None
            else None
        )
    live_state = _page_call(page, "evaluate", _MEDIA_STATE_PROBE_JS)
    if isinstance(live_state, dict):
        for key, control in controls.items():
            live_key = control[0]
            if result[key] is None and isinstance(live_state.get(live_key), bool):
                result[key] = live_state[live_key]
    return result


def _click_media_control(page, selector: str) -> bool:
    locator = _page_call(page, "locator", selector)
    return bool(
        locator is not None
        and (visible := _visible(locator)) is not None
        and _clicked(visible)
    )


def _toggle_media_control(page, selector: str, shortcut: str) -> bool:
    """Use the visible control first; Meet's keyboard shortcut covers hidden trays."""
    return _click_media_control(page, selector) or _page_effect(
        getattr(page, "keyboard", None), "press", shortcut
    )


def _ensure_local_media_before_join(
    page,
    state: _BotState,
    *,
    realtime_enabled: bool,
    realtime_route_ready: bool,
    attempts: int = 3,
) -> bool:
    """Verify privacy-safe media state before joining; realtime requires a live route and mic."""
    if realtime_enabled and not realtime_route_ready:
        state.set(
            error="realtime audio route is not ready",
            leave_reason="realtime_not_ready",
            in_call=False,
            lobby_waiting=False,
            joined_at=None,
            phase="exited",
            exited=True,
        )
        return False
    for _ in range(max(1, attempts)):
        media = _probe_local_media_state(page)
        if media.get("local_camera_on") is True:
            _toggle_media_control(
                page, 'button[aria-label*="turn off camera" i]', "Control+E"
            )
        if realtime_enabled and media.get("local_microphone_on") is False:
            _toggle_media_control(
                page, 'button[aria-label*="turn on microphone" i]', "Control+D"
            )
        elif not realtime_enabled and media.get("local_microphone_on") is True:
            _toggle_media_control(
                page, 'button[aria-label*="turn off microphone" i]', "Control+D"
            )
        _page_call(page, "wait_for_timeout", 250)
        media = _probe_local_media_state(page)
        state.set(**media)
        safe = media.get("local_camera_on") is False and (
            media.get("local_microphone_on") is True
            if realtime_enabled
            else media.get("local_microphone_on") is False
        )
        if safe:
            state.set(mic_state="unmuted" if realtime_enabled else "muted")
            return True
    state.set(
        error="local media state unsafe before join",
        leave_reason="unsafe_media_state",
        in_call=False,
        lobby_waiting=False,
        joined_at=None,
        phase="exited",
        exited=True,
    )
    return False


def _pcm_tail_loop(
    proc, pcm_path: Path, stop_flag: dict, state: _BotState, poll_interval: float = 0.05
) -> None:
    """Forward PCM continuously and revoke readiness when the stream fails."""
    failure = None
    try:
        with pcm_path.open("rb") as pcm_file:
            while not stop_flag.get("stop") and proc.poll() is None:
                chunk = pcm_file.read(65536)
                if not chunk:
                    time.sleep(poll_interval)
                    continue
                proc.stdin.write(chunk)
                proc.stdin.flush()
    except (BrokenPipeError, OSError) as exc:
        failure = str(exc)
    finally:
        _quiet(proc.stdin.close)
        if not stop_flag.get("stop"):
            return_code = proc.poll()
            state.set(
                realtime_ready=False,
                realtime_audio_pump_status="failed",
                realtime_audio_pump_return_code=return_code,
                realtime_audio_pump_error=failure or f"PCM pump exited ({return_code})",
            )


def _mac_audio_device_index(device_name: str) -> Optional[str]:
    result = _quiet(
        subprocess.run,
        ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
    )
    for line in (result.stderr if result else "").splitlines():
        match = re.search(r"\[(\d+)\]\s+(.+)$", line)
        if match and match.group(2).strip().lower() == device_name.strip().lower():
            return match.group(1)
    return None


def _start_pcm_pump(
    rt: dict, bridge_info: dict, pcm_path: Path, state: _BotState, stop_flag: dict
) -> bool:
    bridge_info = bridge_info or {}
    platform_tag, target = bridge_info.get("platform"), bridge_info.get("write_target")
    if platform_tag == "linux":
        command = [
            "paplay",
            "--raw",
            "--rate=24000",
            "--format=s16le",
            "--channels=1",
            f"--device={target or 'hermes_meet_sink'}",
            "-",
        ]
    elif platform_tag == "darwin":
        device_name = target or "BlackHole 2ch"
        device_index = _mac_audio_device_index(device_name)
        if device_index is None:
            state.set(
                realtime_audio_pump_status="missing_device",
                realtime_audio_pump_tool="ffmpeg",
                realtime_audio_pump_error=f"macOS audio device not found: {device_name}",
            )
            return False
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            "24000",
            "-ac",
            "1",
            "-i",
            "-",
            "-f",
            "audiotoolbox",
            "-audio_device_index",
            device_index,
            "-",
        ]
    else:
        state.set(
            realtime_audio_pump_status="unsupported",
            realtime_audio_pump_error=f"unsupported realtime audio host: {platform_tag or 'unknown'}",
        )
        return False
    try:
        pump = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        state.set(
            realtime_audio_pump_status="missing_tool",
            realtime_audio_pump_tool=command[0],
            realtime_audio_pump_error=f"{command[0]} not found",
        )
        return False
    except OSError as exc:
        state.set(
            realtime_audio_pump_status="failed",
            realtime_audio_pump_tool=command[0],
            realtime_audio_pump_error=str(exc),
        )
        return False
    rt["pcm_pump"] = pump
    rt["pcm_tail_thread"] = threading.Thread(
        target=_pcm_tail_loop,
        args=(pump, pcm_path, stop_flag, state),
        name="meet-pcm-tail",
        daemon=True,
    )
    state.set(
        realtime_audio_pump_status="ready",
        realtime_audio_pump_tool=command[0],
        realtime_audio_pump_pid=pump.pid,
        realtime_audio_pump_error=None,
    )
    rt["pcm_tail_thread"].start()
    return True


def _setup_realtime(rt: dict, api_key: str, state: _BotState) -> bool:
    """Provision a supported bridge or fail closed before a browser can join."""
    if not api_key:
        state.set(
            error="realtime mode requested but no API key in HERMES_MEET_REALTIME_KEY/OPENAI_API_KEY",
            leave_reason="realtime_not_ready",
            phase="exited",
            exited=True,
        )
        return False
    bridge = None
    try:
        from plugins.google_meet.audio_bridge import AudioBridge

        bridge = AudioBridge()
        bridge_info = bridge.setup()
    except Exception as exc:
        if bridge is not None:
            _quiet(bridge.teardown)
        state.set(
            error=f"audio bridge setup failed: {exc}",
            leave_reason="realtime_not_ready",
            phase="exited",
            exited=True,
        )
        return False
    rt["bridge"], rt["bridge_info"] = bridge, bridge_info
    state.set(realtime=True, realtime_device=bridge_info.get("device_name"))
    return True


def _start_realtime_speaker(
    rt: dict, cfg: "_BotConfig", stop_flag: dict, state: _BotState
) -> bool:
    """Connect Realtime and prove the PCM route before accepting a Meet join."""
    pcm_path, queue_path = (
        cfg.out_dir / SAY_PCM_FILENAME,
        cfg.out_dir / SAY_QUEUE_FILENAME,
    )
    pcm_path.write_bytes(b"")
    queue_path.touch()
    try:
        from plugins.google_meet.realtime.openai_client import (
            RealtimeSession,
            RealtimeSpeaker,
        )

        session = RealtimeSession(
            api_key=cfg.realtime_api_key,
            model=cfg.realtime_model,
            voice=cfg.realtime_voice,
            instructions=cfg.realtime_instructions,
            audio_sink_path=pcm_path,
            sample_rate=24000,
        )
        session.connect()
    except Exception as exc:
        state.set(
            error=f"realtime connect failed: {exc}",
            leave_reason="realtime_not_ready",
            phase="exited",
            exited=True,
        )
        return False
    rt["session"], rt["stop_flag"] = session, stop_flag
    if not _start_pcm_pump(rt, rt.get("bridge_info") or {}, pcm_path, state, stop_flag):
        _quiet(session.close)
        rt["session"] = None
        state.set(
            error=state.realtime_audio_pump_error or "realtime PCM pump failed",
            leave_reason="realtime_not_ready",
            phase="exited",
            exited=True,
        )
        return False
    speaker = RealtimeSpeaker(
        session=session,
        queue_path=queue_path,
        processed_path=cfg.out_dir / "say_processed.jsonl",
    )

    def speaker_loop() -> None:
        try:
            speaker.run_until_stopped(lambda: stop_flag.get("stop", False))
        except Exception as exc:
            state.set(error=f"realtime speaker crashed: {exc}", realtime_ready=False)

    rt["speaker_thread"] = threading.Thread(
        target=speaker_loop, name="meet-speaker", daemon=True
    )
    rt["speaker_thread"].start()
    state.set(realtime_ready=True)
    return True


def _realtime_route_ready(state: _BotState, rt: Optional[dict] = None) -> bool:
    if not (
        state.realtime
        and state.realtime_ready
        and state.realtime_audio_pump_status == "ready"
    ):
        return False
    if rt is None:
        return True
    pump, tail = rt.get("pcm_pump"), rt.get("pcm_tail_thread")
    return bool(
        rt.get("bridge")
        and rt.get("session")
        and pump is not None
        and pump.poll() is None
        and tail
        and tail.is_alive()
    )


def _wait_for_realtime_route_ready(
    state: _BotState, *, rt: Optional[dict] = None, timeout_s: float = 15.0
) -> bool:
    deadline = time.time() + max(0.1, timeout_s)
    while time.time() < deadline:
        if _realtime_route_ready(state, rt):
            return True
        if state.realtime_audio_pump_status in {
            "failed",
            "missing_tool",
            "unsupported",
        }:
            return False
        time.sleep(0.1)
    state.set(
        error="realtime audio route did not become ready",
        leave_reason="realtime_not_ready",
        phase="exited",
        exited=True,
    )
    return False


def _teardown_realtime(rt: dict, state: Optional[_BotState] = None) -> None:
    stop_flag = rt.get("stop_flag")
    if isinstance(stop_flag, dict):
        stop_flag["stop"] = True
    pump = rt.get("pcm_pump")
    if pump is not None:
        _quiet(pump.terminate)
        _quiet(pump.wait, timeout=3)
    for key, method, kwargs in (
        ("pcm_tail_thread", "join", {"timeout": 1.0}),
        ("speaker_thread", "join", {"timeout": 5.0}),
        ("session", "close", {}),
        ("bridge", "teardown", {}),
    ):
        if (value := rt.get(key)) is not None:
            _quiet(getattr(value, method), **kwargs)
    if state is not None and (
        rt.get("enabled") or pump is not None or rt.get("bridge") is not None
    ):
        state.set(
            realtime_ready=False,
            realtime_audio_pump_return_code=pump.poll() if pump is not None else None,
            realtime_audio_pump_status="stopped"
            if state.realtime_audio_pump_status == "ready"
            else state.realtime_audio_pump_status,
        )


def _apply_meet_proxy_args(cfg: "_BotConfig", chrome_args: list[str]) -> None:
    if not cfg.proxy_server:
        return
    chrome_args.append(f"--proxy-server={cfg.proxy_server}")
    if cfg.proxy_bypass:
        chrome_args.append(f"--proxy-bypass-list={cfg.proxy_bypass}")
    chrome_args.append(MEET_WEBRTC_PROXY_POLICY)


def _build_browser_launch_config(cfg: "_BotConfig") -> tuple[list[str], list[str]]:
    chrome_args = [
        "--use-fake-ui-for-media-stream",
        "--disable-blink-features=AutomationControlled",
    ]
    if not cfg.realtime:
        chrome_args.insert(1, "--use-fake-device-for-media-stream")
    _apply_meet_proxy_args(cfg, chrome_args)
    return chrome_args, ["microphone", "camera"]


_BotConfig = SimpleNamespace


def _config_from_env() -> _BotConfig:
    env = os.environ.get
    out_raw, mode = (
        env("HERMES_MEET_OUT_DIR", "").strip(),
        env("HERMES_MEET_MODE", "transcribe").strip().lower(),
    )
    return _BotConfig(
        url=env("HERMES_MEET_URL", "").strip(),
        out_dir=Path(out_raw) if out_raw else None,
        headed=env("HERMES_MEET_HEADED", "").strip().lower() in {"1", "true", "yes"},
        auth_state=env("HERMES_MEET_AUTH_STATE", "").strip(),
        guest_name=env("HERMES_MEET_GUEST_NAME", "Hermes Agent").strip()
        or "Hermes Agent",
        duration_s=_parse_duration(env("HERMES_MEET_DURATION", "")),
        lobby_timeout=_float_env(
            env("HERMES_MEET_LOBBY_TIMEOUT", "300"), 300.0, minimum=1.0
        ),
        realtime=mode == "realtime",
        realtime_api_key=env("HERMES_MEET_REALTIME_KEY", "")
        or env("OPENAI_API_KEY", ""),
        realtime_model=env("HERMES_MEET_REALTIME_MODEL", "gpt-realtime"),
        realtime_voice=env("HERMES_MEET_REALTIME_VOICE", "alloy"),
        realtime_instructions=env("HERMES_MEET_REALTIME_INSTRUCTIONS", ""),
        proxy_server=env("HERMES_MEET_PROXY_SERVER", "").strip(),
        proxy_bypass=env("HERMES_MEET_PROXY_BYPASS", MEET_MEDIA_PROXY_BYPASS).strip(),
        realtime_ready_timeout=_float_env(
            env("HERMES_MEET_REALTIME_READY_TIMEOUT", "15"), 15.0, minimum=0.1
        ),
        stall_after=_float_env(env("HERMES_MEET_STALL_AFTER", "90"), 90.0, minimum=1.0),
    )


def _apply_admission_probe(
    state: _BotState, ui_probe: dict, *, now: float, lobby_deadline: float
) -> tuple[bool, bool]:
    if ui_probe.get("waitingLobby"):
        state.set(
            in_call=False, joined_at=None, lobby_waiting=True, phase="waiting_lobby"
        )
    if ui_probe.get("preJoin") and state.in_call and not state.last_caption_at:
        state.set(in_call=False, joined_at=None, phase="joining")
    has_caption_evidence = bool(state.last_caption_at or state.transcript_lines)
    if (
        ui_probe.get("callError")
        and not ui_probe.get("inCall")
        and not has_caption_evidence
        and state.join_attempted_at
    ):
        state.call_error_strikes += 1
        if state.call_error_strikes >= CALL_ERROR_STRIKE_LIMIT:
            state.set(
                error="meet call error before captions",
                leave_reason="meet_error",
                in_call=False,
                lobby_waiting=False,
                joined_at=None,
                phase="exited",
                exited=True,
            )
            return False, True
    else:
        state.call_error_strikes = 0
    if ui_probe.get("landing") and not has_caption_evidence:
        state.set(
            error="meet returned to landing before captions",
            leave_reason="meet_landing",
            in_call=False,
            lobby_waiting=False,
            joined_at=None,
            phase="exited",
            exited=True,
        )
        return False, True
    if ui_probe.get("inCall"):
        state.ever_admitted = True
        state.set(
            in_call=True,
            lobby_waiting=False,
            joined_at=state.joined_at or now,
            phase="in_call",
        )
        return True, False
    if state.join_attempted_at and now > lobby_deadline:
        state.set(
            error="lobby timeout — host never admitted the bot",
            leave_reason="lobby_timeout",
            in_call=False,
            lobby_waiting=False,
            joined_at=None,
            phase="exited",
            exited=True,
        )
        return False, True
    if ui_probe.get("denied") and (
        state.join_attempted_at or ui_probe.get("terminalDenied")
    ):
        state.set(
            error="host denied admission",
            leave_reason="denied",
            in_call=False,
            lobby_waiting=False,
            joined_at=None,
            phase="exited",
            exited=True,
        )
        return False, True
    return False, False


def _compute_meet_phase(
    state: _BotState, *, now: float, stall_after: float
) -> tuple[str, Optional[str]]:
    if state.exited:
        return "exited", state.leave_reason or state.error
    if state.transcript_lines or state.last_caption_at:
        return "capturing", None
    if state.in_call:
        return "in_call", None
    if state.join_attempted_at:
        age = now - state.join_attempted_at
        if age > stall_after:
            return "stalled", f"no admission progress for {int(age)}s"
        return ("waiting_lobby" if state.lobby_waiting else "joining"), None
    return "starting", None


def _looks_like_human_speaker(speaker: str, bot_guest_name: str) -> bool:
    return bool(speaker and speaker.strip()) and speaker.strip().lower() not in {
        "unknown",
        "unresolved speaker",
        "you",
        bot_guest_name.strip().lower(),
    }


def _drain_loop(
    page, cfg: _BotConfig, state: _BotState, rt: dict, stop_flag: dict
) -> None:
    deadline = (
        time.time() + cfg.duration_s if getattr(cfg, "duration_s", None) else None
    )
    lobby_deadline, last_admission_probe, last_caption_retry, last_heartbeat = (
        time.time() + getattr(cfg, "lobby_timeout", 300.0),
        0.0,
        0.0,
        0.0,
    )
    while not stop_flag.get("stop"):
        now = time.time()
        if state.realtime and not _realtime_route_ready(state, rt):
            state.set(
                realtime_ready=False,
                error="realtime audio route stopped",
                leave_reason="realtime_audio_route_failed",
                in_call=False,
                lobby_waiting=False,
                joined_at=None,
                phase="exited",
                exited=True,
            )
            return
        if deadline is not None and now > deadline:
            state.set(leave_reason="duration_expired", phase="exited")
            return
        ui_probe = {}
        if now - last_admission_probe > 3.0:
            last_admission_probe = now
            if not state.join_attempted_at and not state.in_call:
                if not _ensure_local_media_before_join(
                    page,
                    state,
                    realtime_enabled=bool(getattr(cfg, "realtime", False)),
                    realtime_route_ready=_realtime_route_ready(state, rt),
                ):
                    return
                if _join(page, cfg, state, timeout=0.5):
                    state.set(join_attempted_at=now, phase="joining")
            ui_probe = _probe_meet_ui(page)
            admitted, terminal = _apply_admission_probe(
                state, ui_probe, now=now, lobby_deadline=lobby_deadline
            )
            if terminal:
                return
            if admitted and not _ensure_local_media_before_join(
                page,
                state,
                realtime_enabled=bool(getattr(cfg, "realtime", False)),
                realtime_route_ready=_realtime_route_ready(state, rt),
                attempts=1,
            ):
                return
        if (
            (state.in_call or state.join_attempted_at)
            and not state.last_caption_at
            and now - last_caption_retry > 3.0
        ):
            last_caption_retry = now
            _retry_caption_enable(page, state, after_join=bool(state.join_attempted_at))
        try:
            queued = page.evaluate(
                "window.__hermesMeetDrain && window.__hermesMeetDrain()"
            )
            if isinstance(queued, list):
                for entry in queued:
                    if not isinstance(entry, dict):
                        continue
                    speaker = str(entry.get("speaker", ""))
                    state.record_caption(
                        speaker,
                        str(entry.get("text", "")),
                        speaker_source=str(entry.get("speakerSource", "")),
                        speaker_debug=entry.get("speakerDebug")
                        if isinstance(entry.get("speakerDebug"), dict)
                        else None,
                        caption_id=str(entry.get("captionId", "")),
                    )
                    if (
                        (session := rt.get("session")) is not None
                        and _looks_like_human_speaker(speaker, cfg.guest_name)
                        and _quiet(session.cancel_response)
                    ):
                        state.set(last_barge_in_at=now)
        except Exception:
            if _quiet(page.is_closed):
                state.set(leave_reason="page_closed", phase="exited")
                return
        if (session := rt.get("session")) is not None:
            state.set(
                audio_bytes_out=getattr(session, "audio_bytes_out", 0),
                last_audio_out_at=getattr(session, "last_audio_out_at", None),
            )
        if now - last_heartbeat > 5.0:
            last_heartbeat = now
            phase, stalled_reason = _compute_meet_phase(
                state, now=now, stall_after=getattr(cfg, "stall_after", 90.0)
            )
            last_ui_text = ui_probe.get("text")
            last_url = ui_probe.get("url") or _quiet(lambda: page.url)
            state.heartbeat(
                phase=phase,
                stalled_reason=stalled_reason,
                last_ui_text=last_ui_text if isinstance(last_ui_text, str) else None,
                last_url=last_url if isinstance(last_url, str) else None,
            )
        time.sleep(1.0)


_CONTEXT_ARGS = {
    "viewport": {"width": 1280, "height": 800},
    "user_agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
}


def _parse_duration(raw: str) -> Optional[float]:
    if not raw:
        return None
    raw, multiplier = (
        raw.strip().lower(),
        {"h": 3600.0, "m": 60.0, "s": 1.0}.get(raw.strip().lower()[-1:]),
    )
    try:
        return float(raw[:-1]) * multiplier if multiplier else float(raw)
    except ValueError:
        return None


def run_bot() -> int:
    cfg = _config_from_env()
    if not _is_safe_meet_url(cfg.url):
        sys.stderr.write(
            "google_meet bot: refusing to launch — HERMES_MEET_URL must be a meet.google.com URL. got: %r\n"
            % cfg.url
        )
        return 2
    if cfg.out_dir is None:
        sys.stderr.write("google_meet bot: HERMES_MEET_OUT_DIR is required\n")
        return 2
    state = _BotState(cfg.out_dir, _meeting_id_from_url(cfg.url), cfg.url)
    stop_flag = {"stop": False}

    def on_signal(*unused) -> None:
        _ = unused
        stop_flag["stop"] = True

    # Signal setup follows.

    for signal_number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signal_number, on_signal)
    rt = {
        "enabled": cfg.realtime,
        "bridge": None,
        "bridge_info": None,
        "session": None,
        "speaker_thread": None,
        "pcm_pump": None,
        "pcm_tail_thread": None,
        "stop_flag": stop_flag,
    }
    browser = context = page = None
    try:
        if cfg.realtime and not _setup_realtime(rt, cfg.realtime_api_key, state):
            return 6
        try:
            from playwright.sync_api import sync_playwright  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            state.set(error=f"playwright not installed: {exc}")
            sys.stderr.write(
                "google_meet bot: playwright is not installed. Install Playwright and Chromium.\n"
            )
            return 3
        chrome_args, permissions = _build_browser_launch_config(cfg)
        chrome_env = {
            key: value
            for key, value in os.environ.items()
            if key not in {"HERMES_MEET_REALTIME_KEY", "OPENAI_API_KEY"}
        }
        if cfg.realtime and rt["bridge_info"].get("platform") == "linux":
            chrome_env["PULSE_SOURCE"] = rt["bridge_info"].get("device_name", "")
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=not cfg.headed, args=chrome_args, env=chrome_env
            )
            context_args = dict(_CONTEXT_ARGS, permissions=permissions)
            if cfg.auth_state and Path(cfg.auth_state).is_file():
                context_args["storage_state"] = cfg.auth_state
            context = browser.new_context(**context_args)
            page = context.new_page()
            try:
                page.goto(cfg.url, wait_until="domcontentloaded", timeout=30_000)
            except Exception as exc:
                state.set(error=f"navigate failed: {exc}")
                return 4
            if cfg.realtime:
                if not _start_realtime_speaker(rt, cfg, stop_flag, state):
                    return 6
                if not _wait_for_realtime_route_ready(
                    state, rt=rt, timeout_s=cfg.realtime_ready_timeout
                ):
                    return 6
            if not _ensure_local_media_before_join(
                page,
                state,
                realtime_enabled=cfg.realtime,
                realtime_route_ready=_realtime_route_ready(state, rt),
            ):
                return 5
            if _join(page, cfg, state):
                state.set(join_attempted_at=time.time(), phase="joining")
                _retry_caption_enable(page, state, after_join=True)
            try:
                page.evaluate(_CAPTION_OBSERVER_JS)
            except Exception as exc:
                state.set(error=f"caption observer install failed: {exc}")
            _drain_loop(page, cfg, state, rt, stop_flag)
            return 0
    except Exception as exc:
        state.set(error=f"unhandled: {exc}")
        return 1
    finally:
        if page is not None:
            _quiet(page.evaluate, _LEAVE_CALL_JS)
        if context is not None:
            _quiet(context.close)
        if browser is not None:
            _quiet(browser.close)
        _teardown_realtime(rt, state)
        state.set(in_call=False, captioning=False, exited=True, phase="exited")


if __name__ == "__main__":  # pragma: no cover - subprocess entry point
    sys.exit(run_bot())


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
SAY_PCM_FILENAME = "speaker.pcm"
SAY_QUEUE_FILENAME = "say_queue.jsonl"
# ---- END PLUGIN-COMPAT ----
