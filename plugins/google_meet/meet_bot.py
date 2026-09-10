"""Headless Google Meet bot — Playwright + live-caption scraping.

Standalone subprocess spawned by ``process_manager.py``; configured via ``HERMES_MEET_*`` env,
status + transcript written under ``$HERMES_MEET_OUT_DIR`` (filesystem is the only IPC).
No WebRTC audio parsing: Meet's live captions are watched via a MutationObserver — lossy and
English-biased, but deterministic (no STT billing) and stable thanks to the ARIA role.
Debug: ``HERMES_MEET_URL=... HERMES_MEET_OUT_DIR=/tmp/x HERMES_MEET_HEADED=1 \\
    python -m plugins.google_meet.meet_bot``
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from plugins.google_meet._jsonfile import write_json_atomic

# Short three-segment code, a lookup URL, or /new. Anything else is rejected.
MEET_URL_RE = re.compile(
    r"^https://meet\.google\.com/([a-z0-9]{3,}-[a-z0-9]{3,}-[a-z0-9]{3,}|lookup/[^/?#]+|new)"
    r"(?:[/?#].*)?$")

_FFMPEG_MISSING = "ffmpeg not found — install via `brew install ffmpeg` for realtime on macOS"


def _is_safe_meet_url(url: str) -> bool:
    """True if *url* is a Google Meet URL we're willing to navigate to."""
    return isinstance(url, str) and bool(MEET_URL_RE.match(url.strip()))


def _meeting_id_from_url(url: str) -> str:
    """3-segment meeting code, or a timestamped id for ``/lookup/...`` and ``/new``."""
    m = re.search(r"meet\.google\.com/([a-z0-9]{3,}-[a-z0-9]{3,}-[a-z0-9]{3,})", url or "")
    return m.group(1) if m else f"meet-{int(time.time())}"


def _quiet(fn, *args, **kwargs):
    """Call *fn*, swallowing any exception (best-effort teardown steps)."""
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


# status.json keys in file order → _BotState attribute + initial value.
_STATUS_FIELDS = (
    ("meetingId", "meeting_id", None), ("url", "url", None), ("inCall", "in_call", False),
    ("captioning", "captioning", False), ("captionsEnabledAttempted", "captions_enabled_attempted", False),
    ("lobbyWaiting", "lobby_waiting", False), ("joinAttemptedAt", "join_attempted_at", None),
    ("joinedAt", "joined_at", None), ("lastCaptionAt", "last_caption_at", None),
    ("transcriptLines", "transcript_lines", 0), ("transcriptPath", "transcript_path", None),
    ("error", "error", None), ("exited", "exited", False), ("pid", None, None),
    # realtime telemetry
    ("realtime", "realtime", False), ("realtimeReady", "realtime_ready", False),
    ("realtimeDevice", "realtime_device", None), ("audioBytesOut", "audio_bytes_out", 0),
    ("lastAudioOutAt", "last_audio_out_at", None), ("lastBargeInAt", "last_barge_in_at", None),
    ("leaveReason", "leave_reason", None))


class _BotState:
    """Single-process mutable state, flushed to ``status.json`` on each change."""

    def __init__(self, out_dir: Path, meeting_id: str, url: str):
        self.__dict__.update({attr: default for _, attr, default in _STATUS_FIELDS if attr})
        self.__dict__.update(out_dir=out_dir, meeting_id=meeting_id, url=url, _seen=set(),  # seen "speaker|text"
                             transcript_path=out_dir / "transcript.txt", status_path=out_dir / "status.json")
        out_dir.mkdir(parents=True, exist_ok=True)
        self._flush()

    def record_caption(self, speaker: str, text: str) -> None:
        """Append a caption line unless this exact (speaker, text) was already seen."""
        speaker, text = (speaker or "").strip() or "Unknown", (text or "").strip()
        key = f"{speaker}|{text}"
        if not text or key in self._seen:
            return
        self._seen.add(key)
        self.transcript_lines += 1
        self.last_caption_at = time.time()
        ts = time.strftime("%H:%M:%S", time.localtime(self.last_caption_at))
        with self.transcript_path.open("a", encoding="utf-8") as f:
            f.write(f"[{ts}] {speaker}: {text}\n")
        self._flush()

    def _flush(self) -> None:
        data = {key: getattr(self, attr) if attr else None for key, attr, _ in _STATUS_FIELDS}
        data.update(transcriptPath=str(self.transcript_path), pid=os.getpid())  # keeps table key order
        write_json_atomic(self.status_path, data)

    def set(self, **kwargs) -> None:
        self.__dict__.update(kwargs)
        self._flush()


# JS injected into the Meet tab: MutationObserver on the caption container
# collects {speaker, text}; ``window.__hermesMeetDrain()`` pulls new entries.
_CAPTION_OBSERVER_JS = r"""
(() => {
  if (window.__hermesMeetInstalled) return;
  window.__hermesMeetInstalled = true;
  window.__hermesMeetQueue = [];

  const captionSelector = '[role="region"][aria-label*="aption" i], ' +
                          'div[jsname="YSxPC"], ' +  // legacy
                          'div[jsname="tgaKEf"]';    // current (Apr 2026)

  function pushEntry(speaker, text) {
    if (!text || !text.trim()) return;
    window.__hermesMeetQueue.push({
      ts: Date.now(),
      speaker: (speaker || '').trim(),
      text: text.trim(),
    });
  }

  function scan(root) {
    // Meet captions render as rows of speaker label + text block. Selectors
    // vary across Meet rewrites; try a few shapes and fall back to raw text.
    const rows = root.querySelectorAll('div[jsname="dsyhDe"], div.CNusmb, div.TBMuR');
    if (rows.length) {
      rows.forEach((row) => {
        const spkEl = row.querySelector('div.KcIKyf, div.zs7s8d, span[jsname="YSxPC"]');
        const txtEl = row.querySelector('div.bh44bd, span[jsname="tgaKEf"], div.iTTPOb');
        pushEntry(spkEl ? spkEl.innerText : '', txtEl ? txtEl.innerText : row.innerText);
      });
      return;
    }
    // Fallback: treat the whole region's innerText as one anonymous line.
    pushEntry('', (root.innerText || '').split('\n').filter(Boolean).pop());
  }

  function attach() {
    const el = document.querySelector(captionSelector);
    if (!el) return false;
    new MutationObserver(() => scan(el)).observe(el, { childList: true, subtree: true, characterData: true });
    scan(el);
    return true;
  }

  // Retry on interval — the caption region only appears after captions are
  // enabled and someone speaks.
  if (!attach()) {
    const iv = setInterval(() => { if (attach()) clearInterval(iv); }, 1500);
  }

  window.__hermesMeetDrain = () => {
    const out = window.__hermesMeetQueue.slice();
    window.__hermesMeetQueue = [];
    return out;
  };
})();
"""

# Best-effort caption toggle: Meet binds it to the ``c`` key; click targeting is too brittle.
_ENABLE_CAPTIONS_JS = (
    "(() => { document.body.dispatchEvent(new KeyboardEvent('keydown', "
    "{ key: 'c', code: 'KeyC', keyCode: 67, which: 67, bubbles: true })); return true; })();")

_LEAVE_CALL_JS = (
    "() => { const b = document.querySelector('button[aria-label*=\"eave call\"]');"
    " if (b) b.click(); }")

# True once past the lobby: leave button, caption region (once our observer is installed)
# or participant list visible.
_ADMISSION_PROBE_JS = r"""
    (() => {
      if (document.querySelector('button[aria-label*="eave call" i]')) return true;
      if (window.__hermesMeetInstalled && document.querySelector(
          '[role="region"][aria-label*="aption" i], div[jsname="YSxPC"], div[jsname="tgaKEf"]')) return true;
      return !!document.querySelector('[aria-label*="articipants" i]');
    })();
    """

# English only — what Meet shows when the host denies or removes a guest.
_DENIED_PROBE_JS = r"""
    (() => {
      const text = document.body ? document.body.innerText || '' : '';
      return /You can't join this video call|You were removed from the meeting|No one responded to your request to join/i.test(text);
    })();
    """


def _probe(page, js: str) -> bool:
    """Evaluate a boolean JS probe; conservative — False on any error."""
    return bool(_quiet(page.evaluate, js))


def _visible(locator):
    """``locator.first`` if it exists and is visible, else None (swallows Playwright errors)."""
    return _quiet(lambda: locator.first if locator.first.count() and locator.first.is_visible() else None)


def _start_pcm_pump(rt: dict, bridge_info: dict, pcm_path: Path, state: "_BotState") -> None:
    """Stream the growing ``speaker.pcm`` (24kHz s16le mono) into the device Chrome's fake mic reads."""
    bridge_info = bridge_info or {}
    platform_tag = bridge_info.get("platform")
    target = bridge_info.get("write_target")
    if platform_tag == "linux":
        cmd = ["paplay", "--raw", "--rate=24000", "--format=s16le", "--channels=1",
               f"--device={target or 'hermes_meet_sink'}", str(pcm_path)]
        missing = "paplay not found — install pulseaudio-utils for realtime on Linux"
    elif platform_tag == "darwin":
        # User must have BlackHole as default input; ffmpeg targets it by audiotoolbox index.
        if not shutil.which("ffmpeg"):
            state.set(error=_FFMPEG_MISSING)
            return
        cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-re",
               "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", str(pcm_path), "-f", "audiotoolbox",
               "-audio_device_index", _mac_audio_device_index(target or "BlackHole 2ch"), "-"]
        missing = _FFMPEG_MISSING
    else:
        return
    try:
        rt["pcm_pump"] = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        state.set(error=missing)
    except Exception as e:
        if platform_tag != "darwin":
            raise
        state.set(error=f"macOS pcm pump failed to start: {e}")


def _start_realtime_speaker(rt: dict, cfg: "_BotConfig", stop_flag: dict, state: "_BotState") -> None:
    """Wire up the OpenAI Realtime session, the say-queue speaker thread and the PCM pump."""
    pcm_path, queue_path = cfg.out_dir / "speaker.pcm", cfg.out_dir / "say_queue.jsonl"
    pcm_path.write_bytes(b"")  # clean sink file per session
    queue_path.touch()  # so the speaker poller doesn't error on first iteration
    phase = "import"
    try:
        from plugins.google_meet.realtime.openai_client import RealtimeSession, RealtimeSpeaker
        phase = "connect"
        session = RealtimeSession(
            api_key=cfg.realtime_api_key, model=cfg.realtime_model, voice=cfg.realtime_voice,
            instructions=cfg.realtime_instructions, audio_sink_path=pcm_path, sample_rate=24000)
        session.connect()
    except Exception as e:
        state.set(error=f"realtime {phase} failed: {e}")
        return
    rt["session"] = session
    speaker = RealtimeSpeaker(session=session, queue_path=queue_path,
                              processed_path=cfg.out_dir / "say_processed.jsonl")

    def _speaker_loop():
        try:
            speaker.run_until_stopped(lambda: stop_flag.get("stop", False))
        except Exception as e:
            state.set(error=f"realtime speaker crashed: {e}")

    rt["speaker_thread"] = threading.Thread(target=_speaker_loop, name="meet-speaker", daemon=True)
    rt["speaker_thread"].start()
    _start_pcm_pump(rt, rt["bridge_info"], pcm_path, state)
    state.set(realtime_ready=True)


def _mac_audio_device_index(device_name: str) -> str:
    """ffmpeg ``-audio_device_index`` for *device_name* (case-insensitive; ``"0"`` if not found).
    ffmpeg prints the avfoundation device table on stderr as ``[N] Name``."""
    out = _quiet(subprocess.run, ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                 capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=10)
    needle = device_name.strip().lower()
    for line in (out.stderr if out else "").splitlines():
        m = re.search(r"\[(\d+)\]\s+(.+)$", line)
        if m and m.group(2).strip().lower() == needle:
            return m.group(1)
    return "0"


def _setup_realtime(rt: dict, api_key: str, state: _BotState) -> None:
    """Provision the virtual audio bridge; on any failure fall back to transcribe mode."""
    if not api_key:
        state.set(error="realtime mode requested but no API key in HERMES_MEET_REALTIME_KEY/OPENAI_API_KEY — falling back to transcribe")
        rt["enabled"] = False
        return
    try:
        from plugins.google_meet.audio_bridge import AudioBridge
        rt["bridge"] = AudioBridge()
        rt["bridge_info"] = rt["bridge"].setup()
        state.set(realtime=True, realtime_device=rt["bridge_info"].get("device_name"))
    except Exception as e:
        state.set(error=f"audio bridge setup failed: {e} — falling back to transcribe")
        rt["enabled"] = False


def _teardown_realtime(rt: dict) -> None:
    if rt.get("pcm_pump"):
        _quiet(rt["pcm_pump"].terminate)
        _quiet(rt["pcm_pump"].wait, timeout=3)
    for key, method, kw in (("speaker_thread", "join", {"timeout": 5.0}), ("session", "close", {}),
                            ("bridge", "teardown", {})):
        if rt[key] is not None:
            _quiet(getattr(rt[key], method), **kw)


_BotConfig = SimpleNamespace  # everything the bot reads from ``HERMES_MEET_*`` env vars


def _config_from_env() -> _BotConfig:
    env = os.environ.get
    out_raw = env("HERMES_MEET_OUT_DIR", "").strip()
    return _BotConfig(
        url=env("HERMES_MEET_URL", "").strip(),
        out_dir=Path(out_raw) if out_raw else None,
        headed=env("HERMES_MEET_HEADED", "").lower() in {"1", "true", "yes"},
        auth_state=env("HERMES_MEET_AUTH_STATE", "").strip(),
        guest_name=env("HERMES_MEET_GUEST_NAME", "Hermes Agent"),
        duration_s=_parse_duration(env("HERMES_MEET_DURATION", "")),
        realtime=env("HERMES_MEET_MODE", "transcribe").strip().lower() == "realtime",
        # HERMES_MEET_REALTIME_KEY is resolved by process_manager.start() via the parent's
        # profile secret scope; OPENAI_API_KEY only serves standalone `python -m` runs.
        realtime_api_key=env("HERMES_MEET_REALTIME_KEY") or env("OPENAI_API_KEY", ""),
        realtime_model=env("HERMES_MEET_REALTIME_MODEL", "gpt-realtime"),
        realtime_voice=env("HERMES_MEET_REALTIME_VOICE", "alloy"),
        realtime_instructions=env("HERMES_MEET_REALTIME_INSTRUCTIONS", ""),
        lobby_timeout=float(env("HERMES_MEET_LOBBY_TIMEOUT", "300")))


def _join(page, cfg: _BotConfig, state: _BotState) -> None:
    """Fill the guest-name field and click 'Join now' / 'Ask to join' (the latter → lobby_waiting)."""
    name_box = _visible(page.locator('input[aria-label*="name" i]'))
    if name_box is not None:
        _quiet(name_box.fill, cfg.guest_name, timeout=2_000)
    for label in ("Join now", "Ask to join"):
        btn = _visible(page.get_by_role("button", name=label, exact=False))
        if btn is not None and _quiet(lambda: (btn.click(timeout=3_000), True)):
            if label == "Ask to join":
                state.set(lobby_waiting=True)
            break


def _drain_loop(page, cfg: _BotConfig, state: _BotState, rt: dict, stop_flag: dict) -> None:
    """Admission + caption drain loop until SIGTERM, duration expiry, lobby timeout/denial or page loss.
    Sets ``leave_reason`` for every exit but SIGTERM; triggers barge-in; mirrors realtime counters."""
    deadline = (time.time() + cfg.duration_s) if cfg.duration_s else None
    lobby_deadline = time.time() + cfg.lobby_timeout
    last_admission_check = 0.0
    while not stop_flag["stop"]:
        now = time.time()
        if deadline and now > deadline:
            state.set(leave_reason="duration_expired")
            return
        if not state.in_call and (now - last_admission_check) > 3.0:
            last_admission_check = now
            if _probe(page, _ADMISSION_PROBE_JS):
                state.set(in_call=True, lobby_waiting=False, joined_at=now)
            elif now > lobby_deadline:
                waited = int(lobby_deadline - state.join_attempted_at) if state.join_attempted_at else 0
                state.set(error=f"lobby timeout — host never admitted the bot within {waited}s",
                          leave_reason="lobby_timeout")
                return
            elif _probe(page, _DENIED_PROBE_JS):
                state.set(error="host denied admission", leave_reason="denied")
                return
        try:
            queued = page.evaluate("window.__hermesMeetDrain && window.__hermesMeetDrain()")
            for entry in (e for e in (queued if isinstance(queued, list) else ()) if isinstance(e, dict)):
                speaker = str(entry.get("speaker", ""))
                state.record_caption(speaker=speaker, text=str(entry.get("text", "")))
                # Barge-in: a real human spoke while we may be generating audio.
                if (rt["session"] is not None and _looks_like_human_speaker(speaker, cfg.guest_name)
                        and _quiet(rt["session"].cancel_response)):
                    state.set(last_barge_in_at=now)
        except Exception:
            if page.is_closed():  # Meet reloaded or we got booted — exit rather than spin
                state.set(leave_reason="page_closed")
                return
        if rt["session"] is not None:
            state.set(audio_bytes_out=rt["session"].audio_bytes_out,
                      last_audio_out_at=rt["session"].last_audio_out_at)
        time.sleep(1.0)


_CONTEXT_ARGS = {
    "viewport": {"width": 1280, "height": 800},
    "user_agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "permissions": ["microphone", "camera"]}


def run_bot() -> int:
    cfg = _config_from_env()
    if not _is_safe_meet_url(cfg.url):
        sys.stderr.write("google_meet bot: refusing to launch — HERMES_MEET_URL must be a "
                         "meet.google.com URL. got: %r\n" % cfg.url)
        return 2
    if cfg.out_dir is None:
        sys.stderr.write("google_meet bot: HERMES_MEET_OUT_DIR is required\n")
        return 2
    state = _BotState(out_dir=cfg.out_dir, meeting_id=_meeting_id_from_url(cfg.url), url=cfg.url)
    # SIGTERM sets a flag (not an exception) so the Playwright teardown below still runs
    # and ``meet_leave`` gets a finalized transcript.
    stop_flag = {"stop": False}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda _sig, _frame: stop_flag.__setitem__("stop", True))
    # Realtime resources in one dict so teardown works however we exit.
    rt = dict(enabled=cfg.realtime, bridge=None, bridge_info=None, session=None, speaker_thread=None)
    if rt["enabled"]:
        _setup_realtime(rt, cfg.realtime_api_key, state)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        state.set(error=f"playwright not installed: {e}", exited=True)
        sys.stderr.write("google_meet bot: playwright is not installed. Run "
                         "`pip install playwright && python -m playwright install chromium`\n")
        if rt["bridge"]:
            rt["bridge"].teardown()
        return 3
    chrome_args = ["--use-fake-ui-for-media-stream", "--disable-blink-features=AutomationControlled"]
    if not rt["enabled"]:
        chrome_args.insert(1, "--use-fake-device-for-media-stream")  # silent fake mic
    elif rt["bridge_info"] and rt["bridge_info"].get("platform") == "linux":
        # Playwright's launch() takes no env: set PULSE_SOURCE on ourselves so Chrome inherits it.
        os.environ["PULSE_SOURCE"] = rt["bridge_info"].get("device_name", "")
    context_args = dict(_CONTEXT_ARGS)
    if cfg.auth_state and Path(cfg.auth_state).is_file():
        context_args["storage_state"] = cfg.auth_state
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=not cfg.headed, args=chrome_args)
            context = browser.new_context(**context_args)
            page = context.new_page()
            try:
                page.goto(cfg.url, wait_until="domcontentloaded", timeout=30_000)
            except Exception as e:
                state.set(error=f"navigate failed: {e}", exited=True)
                return 4
            _join(page, cfg, state)
            if _quiet(page.evaluate, _ENABLE_CAPTIONS_JS):
                state.set(captions_enabled_attempted=True)
            try:
                page.evaluate(_CAPTION_OBSERVER_JS)
            except Exception as e:
                state.set(error=f"caption observer install failed: {e}")
            # in_call stays False until admission is confirmed by the drain loop.
            state.set(captioning=True, join_attempted_at=time.time())
            if rt["enabled"]:
                _start_realtime_speaker(rt, cfg, stop_flag, state)
            _drain_loop(page, cfg, state, rt, stop_flag)
            _quiet(page.evaluate, _LEAVE_CALL_JS)
            context.close()
            browser.close()
            _teardown_realtime(rt)
            state.set(in_call=False, captioning=False, exited=True)
            return 0
    except Exception as e:
        state.set(error=f"unhandled: {e}", exited=True)
        return 1


def _looks_like_human_speaker(speaker: str, bot_guest_name: str) -> bool:
    """Whether a caption's speaker is probably a human rather than our own echo (Meet attributes
    our fake-mic audio to the bot's name; blank/unknown speakers are ambiguous — no barge-in)."""
    return bool(speaker and speaker.strip()) and (
        speaker.strip().lower() not in {"unknown", "you", bot_guest_name.strip().lower()})


_DURATION_UNITS = {"h": 3600.0, "m": 60.0, "s": 1.0}


def _parse_duration(raw: str) -> Optional[float]:
    """Parse ``30m`` / ``2h`` / ``90`` (seconds) → float seconds, or None."""
    if not raw:
        return None
    raw = raw.strip().lower()
    mult = _DURATION_UNITS.get(raw[-1:])
    try:
        return float(raw[:-1]) * mult if mult else float(raw)
    except ValueError:
        return None


if __name__ == "__main__":  # pragma: no cover — subprocess entry point
    sys.exit(run_bot())


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import json  # noqa: F401,E402

SAY_PCM_FILENAME = "speaker.pcm"

SAY_QUEUE_FILENAME = "say_queue.jsonl"
# ---- END PLUGIN-COMPAT ----
