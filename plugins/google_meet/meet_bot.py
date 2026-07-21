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

import json
import os
import re
import signal
import sys
import threading
import time
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
        self.join_attempted_at: Optional[float] = None
        self.joined_at: Optional[float] = None
        self.last_caption_at: Optional[float] = None
        self.transcript_lines = 0
        self.error: Optional[str] = None
        self.exited = False
        # v2 realtime fields.
        self.realtime = False
        self.realtime_ready = False
        self.realtime_device: Optional[str] = None
        self.audio_bytes_out: int = 0
        self.last_audio_out_at: Optional[float] = None
        self.last_barge_in_at: Optional[float] = None
        self.leave_reason: Optional[str] = None
        # Scraped captions, in order, deduped. Each entry is a dict of
        # {"ts": <epoch>, "speaker": str, "text": str}.
        self._seen: set = set()
        out_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_path = out_dir / "transcript.txt"
        self.status_path = out_dir / "status.json"
        self._flush()

    # -------- transcript ------------------------------------------------

    def record_caption(self, speaker: str, text: str) -> None:
        """Append a caption line if we haven't seen this exact (speaker, text)."""
        speaker = (speaker or "").strip() or "Unknown"
        text = (text or "").strip()
        if not text:
            return
        key = f"{speaker}|{text}"
        if key in self._seen:
            return
        self._seen.add(key)
        self.transcript_lines += 1
        self.last_caption_at = time.time()
        ts = time.strftime("%H:%M:%S", time.localtime(self.last_caption_at))
        line = f"[{ts}] {speaker}: {text}\n"
        # Atomic-ish append — good enough for a single-writer.
        with self.transcript_path.open("a", encoding="utf-8") as f:
            f.write(line)
        self._flush()

    # -------- status file ----------------------------------------------

    def _flush(self) -> None:
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
            # v2 realtime telemetry.
            "realtime": self.realtime,
            "realtimeReady": self.realtime_ready,
            "realtimeDevice": self.realtime_device,
            "audioBytesOut": self.audio_bytes_out,
            "lastAudioOutAt": self.last_audio_out_at,
            "lastBargeInAt": self.last_barge_in_at,
            "leaveReason": self.leave_reason,
        }
        tmp = self.status_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.status_path)

    def set(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._flush()


# ---------------------------------------------------------------------------
# Playwright bot entry point
# ---------------------------------------------------------------------------

# JavaScript injected into the Meet tab to observe captions. Captures
# {speaker, text} tuples via a MutationObserver on the caption container,
# and exposes ``window.__hermesMeetDrain()`` to pull new entries. This
# mirrors the OpenUtter caption scraping approach.
_CAPTION_OBSERVER_JS = r"""
(() => {
  window.__hermesMeetInstalled = true;
  window.__hermesMeetQueue = window.__hermesMeetQueue || [];
  window.__hermesMeetAttached = window.__hermesMeetAttached || new WeakSet();

  const captionSelector = [
    '[role="region"][aria-label*="aption" i]',
    '[role="region"][aria-label*="ottotitol" i]',
    '[aria-live="polite"]',
    '[aria-live="assertive"]',
    'div[jsname="YSxPC"]',      // legacy speaker/caption node
    'div[jsname="tgaKEf"]',     // current caption text node
    'span[jsname="YSxPC"]',
    'span[jsname="tgaKEf"]',
    'div.iTTPOb',
    'div.bh44bd',
  ].join(', ');

  function visible(e) {
    return !!(e && (e.offsetWidth || e.offsetHeight || e.getClientRects().length));
  }

  function pushEntry(speaker, text) {
    text = (text || '').replace(/\s+/g, ' ').trim();
    speaker = (speaker || '').replace(/\s+/g, ' ').trim();
    if (!text || text.length < 2 || text.length > 500) return;
    const low = text.toLowerCase();
    const uiNoise = [
      'attendi finché', 'tentativo di partecipazione', 'chiedi di partecipare',
      'continua senza microfono', 'impostazioni audio', 'impostazioni video',
      'vuoi che le persone ti vedano', 'usa gemini', 'condividi note',
      'meeting details', 'people', 'partecipanti', 'presenta ora',
    ];
    if (uiNoise.some((x) => low.includes(x))) return;
    window.__hermesMeetQueue.push({ts: Date.now(), speaker, text});
  }

  function scan(root) {
    if (!root || !root.querySelectorAll) return;
    const rows = root.querySelectorAll('div[jsname="dsyhDe"], div.CNusmb, div.TBMuR');
    if (rows.length) {
      rows.forEach((row) => {
        if (!visible(row)) return;
        const spkEl = row.querySelector('div.KcIKyf, div.zs7s8d, span[jsname="YSxPC"], div[jsname="YSxPC"]');
        const txtEl = row.querySelector('div.bh44bd, span[jsname="tgaKEf"], div[jsname="tgaKEf"], div.iTTPOb');
        pushEntry(spkEl ? spkEl.innerText : '', txtEl ? txtEl.innerText : row.innerText);
      });
      return;
    }

    // Modern Meet sometimes exposes only aria-live text nodes. Scan all likely
    // caption/live regions instead of attaching once to the first stale region.
    root.querySelectorAll(captionSelector).forEach((el) => {
      if (!visible(el)) return;
      const text = (el.innerText || '').split('\n').map((x) => x.trim()).filter(Boolean).pop();
      pushEntry('', text);
    });
  }

  function attachAll() {
    const candidates = [document.body, ...Array.from(document.querySelectorAll(captionSelector))].filter(Boolean);
    candidates.forEach((el) => {
      if (window.__hermesMeetAttached.has(el)) return;
      window.__hermesMeetAttached.add(el);
      try {
        new MutationObserver(() => scan(el)).observe(el, {childList: true, subtree: true, characterData: true});
      } catch (_) {}
      scan(el);
    });
  }

  attachAll();
  if (!window.__hermesMeetCaptionInterval) {
    window.__hermesMeetCaptionInterval = setInterval(attachAll, 1500);
  }

  window.__hermesMeetDrain = () => {
    attachAll();
    const out = window.__hermesMeetQueue.slice();
    window.__hermesMeetQueue = [];
    return out;
  };
})();
"""


def _build_chrome_args(*, rt_enabled: bool) -> list[str]:
    """Return Chromium launch args for Meet.

    Transcribe-only mode must not publish fake camera/microphone media. Chromium's
    fake media device is useful for tests, but in real Meet calls it appears as
    an animated test-pattern tile and a loud ticking microphone. Keep it behind
    an explicit debug env var.
    """
    args = ["--disable-blink-features=AutomationControlled"]
    if not rt_enabled:
        args.extend([
            "--mute-audio",
            "--disable-audio-input",
            "--disable-video-capture",
        ])
        if os.environ.get("HERMES_MEET_FAKE_MEDIA", "").lower() in ("1", "true", "yes"):
            args.extend([
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
            ])
    else:
        args.append("--use-fake-ui-for-media-stream")
    return args


def _build_context_args(*, rt_enabled: bool, auth_state: str = "") -> dict:
    """Return Playwright browser-context args for Meet."""
    context_args = {
        "viewport": {"width": 1280, "height": 800},
        "user_agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
    }
    if rt_enabled:
        context_args["permissions"] = ["microphone", "camera"]
    if auth_state and Path(auth_state).is_file():
        context_args["storage_state"] = auth_state
    elif auth_state:
        # Tests and remote nodes may pass an auth path that is not present in the
        # current process. Preserve the value for caller-visible config checks;
        # run_bot only calls this helper after checking the file normally.
        context_args["storage_state"] = auth_state
    return context_args


def _enable_captions_js() -> str:
    """Return a small JS snippet that tries to click the 'Turn on captions' button.

    Best-effort — Meet's caption toggle is keyboard-accessible via ``c``. We
    dispatch that keystroke as a cheap fallback. Real click targeting is too
    brittle to rely on.
    """
    return r"""
    (() => {
      const visible = (e) => !!(e && (e.offsetWidth || e.offsetHeight || e.getClientRects().length));
      const onLabels = ['turn on captions', 'show captions', 'enable captions', 'attiva sottotitoli', 'mostra sottotitoli'];
      const genericLabels = ['caption', 'captions', 'subtitle', 'subtitles', 'sottotitol', 'sottotitoli'];
      const offLabels = ['turn off captions', 'hide captions', 'disable captions', 'disattiva sottotitoli', 'nascondi sottotitoli'];
      const buttons = Array.from(document.querySelectorAll('button,[role="button"]')).filter(visible);
      for (const b of buttons) {
        const label = ((b.innerText || '') + ' ' + (b.getAttribute('aria-label') || '') + ' ' + (b.getAttribute('data-tooltip') || '')).toLowerCase();
        if (offLabels.some((x) => label.includes(x))) return 'already_on';
        if (onLabels.some((x) => label.includes(x))) {
          b.click();
          return 'clicked';
        }
      }
      for (const b of buttons) {
        const label = ((b.innerText || '') + ' ' + (b.getAttribute('aria-label') || '') + ' ' + (b.getAttribute('data-tooltip') || '')).toLowerCase();
        if (genericLabels.some((x) => label.includes(x)) && !offLabels.some((x) => label.includes(x))) {
          b.click();
          return 'clicked';
        }
      }
      const ev = new KeyboardEvent('keydown', {
        key: 'c', code: 'KeyC', keyCode: 67, which: 67, bubbles: true,
      });
      document.body.dispatchEvent(ev);
      return 'keydown';
    })();
    """


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

    rt["speaker_stop"] = lambda: stop_flag.__setitem__("stop", stop_flag.get("stop", False))

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
        try:
            proc = _sp.Popen(
                [
                    "paplay",
                    "--raw",
                    "--rate=24000",
                    "--format=s16le",
                    "--channels=1",
                    f"--device={sink}",
                    str(pcm_path),
                ],
                stdin=_sp.DEVNULL,
                stdout=_sp.DEVNULL,
                stderr=_sp.DEVNULL,
            )
            rt["pcm_pump"] = proc
        except FileNotFoundError:
            state.set(error="paplay not found — install pulseaudio-utils for realtime on Linux")
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
            try:
                # -re: read input at native frame rate.
                # -f avfoundation -i: speaker path as raw PCM.
                # -f s16le -ar 24000 -ac 1 -i <pcm>: interpret the file.
                # -f audiotoolbox -audio_device_index: write to BlackHole.
                # Simpler: output as raw via coreaudio using "-f audiotoolbox".
                # ffmpeg's audiotoolbox output picks the current default
                # output device, which isn't what we want. Instead we use
                # -f avfoundation with the named device as OUTPUT via
                # -vn and the device name.
                proc = _sp.Popen(
                    [
                        "ffmpeg",
                        "-nostdin", "-hide_banner", "-loglevel", "error",
                        "-re",
                        "-f", "s16le", "-ar", "24000", "-ac", "1",
                        "-i", str(pcm_path),
                        "-f", "audiotoolbox",
                        "-audio_device_index", _mac_audio_device_index(device_name),
                        "-",
                    ],
                    stdin=_sp.DEVNULL,
                    stdout=_sp.DEVNULL,
                    stderr=_sp.DEVNULL,
                )
                rt["pcm_pump"] = proc
            except FileNotFoundError:
                state.set(error="ffmpeg not found — install via `brew install ffmpeg` for realtime on macOS")
            except Exception as e:
                state.set(error=f"macOS pcm pump failed to start: {e}")
        else:
            state.set(error="ffmpeg not found — install via `brew install ffmpeg` for realtime on macOS")


def _mac_audio_device_index(device_name: str) -> str:
    """Return the ffmpeg ``-audio_device_index`` for *device_name*, as a string.

    Probes ``ffmpeg -f avfoundation -list_devices true -i ''`` (which prints
    the device table on stderr) and matches *device_name* case-insensitively.
    Defaults to ``"0"`` if the device can't be found — caller will get a
    misrouted stream but not a crash, and the error will be obvious.
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
        return "0"
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
    return "0"


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
    # We track these in a dict so the finally block can tear them down
    # regardless of how we exit. If anything in the realtime setup fails we
    # fall back to transcribe mode with a status flag.
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
            state.set(error="realtime mode requested but no API key in HERMES_MEET_REALTIME_KEY/OPENAI_API_KEY — falling back to transcribe")
            rt["enabled"] = False
        else:
            try:
                from plugins.google_meet.audio_bridge import AudioBridge
                bridge = AudioBridge()
                rt["bridge_info"] = bridge.setup()
                rt["bridge"] = bridge
                state.set(realtime=True, realtime_device=rt["bridge_info"].get("device_name"))
            except Exception as e:
                state.set(error=f"audio bridge setup failed: {e} — falling back to transcribe")
                rt["enabled"] = False

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        state.set(error=f"playwright not installed: {e}", exited=True)
        sys.stderr.write(
            "google_meet bot: playwright is not installed. Run "
            "`pip install playwright && python -m playwright install chromium`\n"
        )
        if rt["bridge"]:
            rt["bridge"].teardown()
        return 3

    # Chrome env: if realtime is live on Linux, point PULSE_SOURCE at the
    # virtual source so Chrome's fake mic reads the audio we generate.
    chrome_env = os.environ.copy()
    chrome_args = _build_chrome_args(rt_enabled=bool(rt["enabled"]))
    if rt["enabled"] and rt["bridge_info"] and rt["bridge_info"].get("platform") == "linux":
        chrome_env["PULSE_SOURCE"] = rt["bridge_info"].get("device_name", "")

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
            context_args = _build_context_args(
                rt_enabled=bool(rt["enabled"]),
                auth_state=auth_state if auth_state and Path(auth_state).is_file() else "",
            )
            context = browser.new_context(**context_args)
            page = context.new_page()

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            except Exception as e:
                state.set(error=f"navigate failed: {e}", exited=True)
                return 4

            # Guest-mode: Meet shows a name field before "Ask to join". When
            # we're authed, we instead see "Join now".
            _try_guest_name(page, guest_name)
            _click_join(page, state)

            # Install caption observer and attempt to enable captions.
            try:
                page.evaluate(_enable_captions_js())
                state.set(captions_enabled_attempted=True)
            except Exception:
                pass
            try:
                page.evaluate(_CAPTION_OBSERVER_JS)
            except Exception as e:
                state.set(error=f"caption observer install failed: {e}")

            # Note: in_call=False until admission is confirmed (we detect
            # either the Leave button or the caption region, signalling we
            # made it past the lobby).
            state.set(captioning=True, join_attempted_at=time.time())

            # v2 realtime: start the speaker thread reading from the
            # plugin-side say queue. The thread reads JSONL lines written by
            # meet_say, calls OpenAI Realtime, and streams the audio PCM to
            # the virtual sink that Chrome's fake-mic is pointed at.
            if rt["enabled"]:
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
                if rt["session"] is not None:
                    state.set(realtime_ready=True)

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
            last_admission_check = 0.0
            last_caption_enable = 0.0
            while not stop_flag["stop"]:
                now = time.time()
                if deadline and now > deadline:
                    state.set(leave_reason="duration_expired")
                    break

                # Admission detection every ~3s until admitted.
                if not state.in_call and (now - last_admission_check) > 3.0:
                    last_admission_check = now
                    admitted = _detect_admission(page)
                    if admitted:
                        state.set(
                            in_call=True,
                            lobby_waiting=False,
                            joined_at=now,
                        )
                    elif now > lobby_deadline:
                        state.set(
                            error=(
                                "lobby timeout — host never admitted the bot "
                                f"within {int(lobby_deadline - state.join_attempted_at) if state.join_attempted_at else 0}s"
                            ),
                            leave_reason="lobby_timeout",
                        )
                        break
                    elif _detect_denied(page):
                        state.set(
                            error="host denied admission",
                            leave_reason="denied",
                        )
                        break

                # Caption controls only reliably exist after admission. Retry
                # enabling them in-call; if they are already on, the JS returns
                # without toggling them off.
                if state.in_call and (now - last_caption_enable) > 10.0:
                    last_caption_enable = now
                    try:
                        page.evaluate(_enable_captions_js())
                        page.evaluate(_CAPTION_OBSERVER_JS)
                        state.set(captions_enabled_attempted=True, captioning=True)
                    except Exception:
                        pass

                try:
                    queued = page.evaluate("window.__hermesMeetDrain && window.__hermesMeetDrain()")
                    if isinstance(queued, list):
                        for entry in queued:
                            if not isinstance(entry, dict):
                                continue
                            speaker = str(entry.get("speaker", ""))
                            text = str(entry.get("text", ""))
                            state.record_caption(speaker=speaker, text=text)
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

                time.sleep(1.0)

            # Try to leave cleanly — click "Leave call" button if present.
            try:
                page.evaluate(
                    "() => { const b = document.querySelector('button[aria-label*=\"eave call\"]');"
                    " if (b) b.click(); }"
                )
            except Exception:
                pass

            context.close()
            browser.close()
            # v2: teardown PCM pump, speaker thread, and audio bridge.
            if rt.get("pcm_pump"):
                try:
                    rt["pcm_pump"].terminate()
                    rt["pcm_pump"].wait(timeout=3)
                except Exception:
                    pass
            if rt["speaker_stop"]:
                try:
                    rt["speaker_stop"]()
                except Exception:
                    pass
            if rt["speaker_thread"] is not None:
                try:
                    rt["speaker_thread"].join(timeout=5.0)
                except Exception:
                    pass
            if rt["session"]:
                try:
                    rt["session"].close()
                except Exception:
                    pass
            if rt["bridge"]:
                try:
                    rt["bridge"].teardown()
                except Exception:
                    pass
            state.set(in_call=False, captioning=False, exited=True)
            return 0

    except Exception as e:
        state.set(error=f"unhandled: {e}", exited=True)
        return 1


def _try_guest_name(page, guest_name: str) -> None:
    """If Meet is showing a guest-name input, type *guest_name* into it."""
    try:
        # Meet's guest name input has placeholder "Your name".
        locator = page.locator('input[aria-label*="name" i]').first
        if locator.count() and locator.is_visible():
            locator.fill(guest_name, timeout=2_000)
    except Exception:
        pass


def _admission_probe_js() -> str:
    """Return JS that detects whether Meet is past the pre-join/lobby UI."""
    return r"""
    (() => {
      const text = document.body ? (document.body.innerText || '').toLowerCase() : '';
      const hasText = (...needles) => needles.some((n) => text.includes(n));
      const visible = (e) => !!(e && (e.offsetWidth || e.offsetHeight || e.getClientRects().length));
      const controls = Array.from(document.querySelectorAll('button,[role="button"]')).filter(visible);
      const labels = controls.map((b) => ((b.innerText || '') + ' ' + (b.getAttribute('aria-label') || '') + ' ' + (b.getAttribute('data-tooltip') || '')).toLowerCase());
      const hasLabel = (...needles) => labels.some((label) => needles.some((needle) => label.includes(needle)));

      // Lobby/waiting pages can also show an "leave/exit call" button; reject
      // those states before checking in-call controls.
      if (hasText(
        'waiting to be admitted',
        'waiting for the host',
        'ask to join',
        'attendi finché',
        'organizzatore della riunione',
        'tentativo di partecipazione',
        'chiedi di partecipare'
      )) return false;

      // In-call controls. Include localized labels observed in non-English Meet UIs.
      if (hasLabel('leave call', 'eave call', 'hang up', 'abbandona', 'termina chiamata', 'esci dalla chiamata', 'salir de la llamada', 'quitter l')) return true;
      if (hasText('meeting details', 'dettagli della riunione', 'dettagli chiamata')) return true;
      if (hasLabel('participants', 'partecipanti', 'personas', 'partecipantes')) return true;

      // Do NOT treat caption containers as admission by themselves: Meet can
      // render caption/sidebar regions on pre-join pages too. A real admission
      // signal must be an in-call control such as leave/participants/details.
      return false;
    })();
    """


def _denied_probe_js() -> str:
    """Return JS that detects a Meet denial/block page."""
    return r"""
    (() => {
      const text = document.body ? (document.body.innerText || '').toLowerCase() : '';
      if (text.includes("you can't join this video call")) return true;
      if (text.includes('you were removed from the meeting')) return true;
      if (text.includes('no one responded to your request to join')) return true;
      if (text.includes('non puoi partecipare') || text.includes('non puoi accedere')) return true;
      if (text.includes('sei stato rimosso')) return true;
      if (text.includes('nessuno ha risposto')) return true;
      return false;
    })();
    """


def _detect_admission(page) -> bool:
    """True if we're clearly past the lobby and in the call itself.

    Uses a JS-side probe because Meet's DOM structure varies by client
    version. We check several high-signal indicators and declare admission
    on the first hit:

      1. Leave-call button is present (``aria-label`` contains "eave call").
      2. Caption region has appeared (we installed the observer and it attached).
      3. The participant list container is visible.

    Conservative by default — returns False on any error.
    """
    try:
        return bool(page.evaluate(_admission_probe_js()))
    except Exception:
        return False


def _detect_denied(page) -> bool:
    """True when Meet is showing a 'you were denied' / 'no one admitted' page."""
    try:
        return bool(page.evaluate(_denied_probe_js()))
    except Exception:
        return False


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


def _click_join(page, state: _BotState) -> None:
    """Click 'Join now' or 'Ask to join' if either button is visible.

    Flags ``lobby_waiting`` when we hit the "waiting for host to admit you"
    state so the agent can surface that in status.
    """
    join_labels = (
        # Ask/lobby buttons first; some localized ask labels contain the generic
        # join verb, so matching generic labels first can click the right button
        # but fail to mark lobby_waiting.
        ("Ask to join", True),
        ("Chiedi di partecipare", True),
        ("Pedir unirse", True),
        ("Demander à participer", True),
        ("Teilnahme anfragen", True),
        # Immediate join buttons.
        ("Join now", False),
        ("Partecipa", False),
        ("Unirse ahora", False),
        ("Participer", False),
        ("Jetzt teilnehmen", False),
    )

    # In transcribe mode we do not grant mic/camera. Meet may show an explicit
    # confirmation button before the join/ask request is fully sent; click it
    # first when present.
    no_media_labels = (
        "Continue without microphone and camera",
        "Continue without mic and camera",
        "Continua senza microfono e videocamera",
        "Continua senza microfono e fotocamera",
    )

    # Meet renders the pre-join UI after domcontentloaded; checking only once
    # races and misses the localized button on slower loads. Poll briefly.
    deadline = time.time() + 25.0
    while time.time() < deadline:
        for label in no_media_labels:
            try:
                btn = page.get_by_role("button", name=label, exact=False).first
                if btn.count() and btn.is_visible():
                    btn.click(timeout=3_000)
                    page.wait_for_timeout(500)
                    break
            except Exception:
                continue

        for label, waits_for_lobby in join_labels:
            try:
                btn = page.get_by_role("button", name=label, exact=True).first
                if not (btn.count() and btn.is_visible()):
                    btn = page.get_by_role("button", name=label, exact=False).first
                if btn.count() and btn.is_visible():
                    btn.click(timeout=3_000)
                    if waits_for_lobby:
                        state.set(lobby_waiting=True)
                    return
            except Exception:
                continue

        # Last-resort DOM/text fallback for localized Meet UIs.
        try:
            clicked = page.evaluate(
                r"""
                () => {
                  const ask = ['ask to join','chiedi di partecipare','pedir unirse','demander à participer','teilnahme anfragen'];
                  const join = ['join now','partecipa','unirse ahora','participer','jetzt teilnehmen'];
                  const noMedia = ['continue without microphone and camera','continue without mic and camera','continua senza microfono e videocamera','continua senza microfono e fotocamera'];
                  const visible = (e) => !!(e.offsetWidth || e.offsetHeight || e.getClientRects().length);
                  const norm = (s) => (s || '').trim().toLowerCase();
                  const buttons = Array.from(document.querySelectorAll('button')).filter(visible);
                  for (const b of buttons) {
                    const t = norm(b.innerText || b.getAttribute('aria-label'));
                    if (noMedia.some(x => t.includes(x))) { b.click(); return 'continue'; }
                  }
                  for (const b of buttons) {
                    const t = norm(b.innerText || b.getAttribute('aria-label'));
                    if (ask.some(x => t.includes(x))) { b.click(); return 'ask'; }
                  }
                  for (const b of buttons) {
                    const t = norm(b.innerText || b.getAttribute('aria-label'));
                    if (join.some(x => t.includes(x))) { b.click(); return 'join'; }
                  }
                  return null;
                }
                """
            )
            if clicked == "continue":
                time.sleep(0.5)
                continue
            if clicked == "ask":
                state.set(lobby_waiting=True)
                return
            if clicked == "join":
                return
        except Exception:
            pass
        time.sleep(0.5)
    state.set(error="join button not found within 25s")


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
