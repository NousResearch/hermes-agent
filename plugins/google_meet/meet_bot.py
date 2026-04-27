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
        self.joined_at: Optional[float] = None
        self.last_caption_at: Optional[float] = None
        self.transcript_lines = 0
        self.error: Optional[str] = None
        self.exited = False
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
            "joinedAt": self.joined_at,
            "lastCaptionAt": self.last_caption_at,
            "transcriptLines": self.transcript_lines,
            "transcriptPath": str(self.transcript_path),
            "error": self.error,
            "exited": self.exited,
            "pid": os.getpid(),
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
    // Meet captions render as a list of rows; each row contains a speaker
    // label and a text block. Selectors vary across Meet rewrites; we try
    // a few shapes and fall back to raw text.
    const rows = root.querySelectorAll('div[jsname="dsyhDe"], div.CNusmb, div.TBMuR');
    if (rows.length) {
      rows.forEach((row) => {
        const spkEl = row.querySelector('div.KcIKyf, div.zs7s8d, span[jsname="YSxPC"]');
        const txtEl = row.querySelector('div.bh44bd, span[jsname="tgaKEf"], div.iTTPOb');
        const speaker = spkEl ? spkEl.innerText : '';
        const text = txtEl ? txtEl.innerText : row.innerText;
        pushEntry(speaker, text);
      });
      return;
    }
    // Fallback: treat the whole region's innerText as one anonymous line.
    const text = (root.innerText || '').split('\n').filter(Boolean).pop();
    pushEntry('', text);
  }

  function attach() {
    const el = document.querySelector(captionSelector);
    if (!el) return false;
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

    # PCM pump: on Linux we feed speaker.pcm into the null-sink via
    # `paplay --raw --device=<sink> --rate=24000 --format=s16le --channels=1`.
    # paplay will block reading from the file as more PCM is appended.
    # We run it as a subprocess and track the pid for teardown.
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
    # macOS: the caller must route BlackHole manually; we write to disk
    # only. A future helper could spawn `afplay` or `sox play` in a loop.


def run_bot() -> int:  # noqa: C901 — orchestration, explicit branches
    url = os.environ.get("HERMES_MEET_URL", "").strip()
    out_dir_env = os.environ.get("HERMES_MEET_OUT_DIR", "").strip()
    headed = os.environ.get("HERMES_MEET_HEADED", "").lower() in ("1", "true", "yes")
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
    chrome_args = [
        "--use-fake-ui-for-media-stream",
        "--disable-blink-features=AutomationControlled",
    ]
    if not rt["enabled"]:
        # v1-style fake device (silence) — we don't care about mic content
        # when we're not speaking.
        chrome_args.insert(1, "--use-fake-device-for-media-stream")
    elif rt["bridge_info"] and rt["bridge_info"].get("platform") == "linux":
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
            context_args = {
                "viewport": {"width": 1280, "height": 800},
                "user_agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                ),
                "permissions": ["microphone", "camera"],
            }
            if auth_state and Path(auth_state).is_file():
                context_args["storage_state"] = auth_state
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

            state.set(in_call=True, captioning=True, joined_at=time.time())

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

            # Drain loop — pull queued captions every ~1s until SIGTERM or
            # duration expiry. Also poll the page for the "You've left the
            # meeting" state.
            deadline = (time.time() + duration_s) if duration_s else None
            while not stop_flag["stop"]:
                if deadline and time.time() > deadline:
                    break
                try:
                    queued = page.evaluate("window.__hermesMeetDrain && window.__hermesMeetDrain()")
                    if isinstance(queued, list):
                        for entry in queued:
                            if not isinstance(entry, dict):
                                continue
                            state.record_caption(
                                speaker=str(entry.get("speaker", "")),
                                text=str(entry.get("text", "")),
                            )
                except Exception:
                    # Meet reloaded or we got booted — try to detect and
                    # exit gracefully rather than spinning.
                    if page.is_closed():
                        break
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
            # v2: teardown realtime speaker + audio bridge.
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


def _click_join(page, state: _BotState) -> None:
    """Click 'Join now' or 'Ask to join' if either button is visible.

    Flags ``lobby_waiting`` when we hit the "waiting for host to admit you"
    state so the agent can surface that in status.
    """
    for label in ("Join now", "Ask to join"):
        try:
            btn = page.get_by_role("button", name=label, exact=False).first
            if btn.count() and btn.is_visible():
                btn.click(timeout=3_000)
                if label == "Ask to join":
                    state.set(lobby_waiting=True)
                break
        except Exception:
            continue


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
