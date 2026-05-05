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

# Local-first default: most of Joohyun's meetings are Korean, but a small
# number of fixed recurring meetings are predictably English. Keep this as a
# transparent heuristic instead of guessing per caption line. Override with
# HERMES_MEET_ENGLISH_HINTS as a JSON/string list separated by |, comma, or
# newlines.
DEFAULT_ENGLISH_CAPTION_HINTS = (
    "Data & Hyspire Weekly Sync",
    "Jeremy Kang",
    "Phuong Luong",
)


def _language_hint_terms() -> list[str]:
    raw = os.environ.get("HERMES_MEET_ENGLISH_HINTS", "").strip()
    if not raw:
        return list(DEFAULT_ENGLISH_CAPTION_HINTS)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    return [part.strip() for part in re.split(r"[|,\n]", raw) if part.strip()]


def _surface_text_for_language(surface: Optional[dict]) -> str:
    if not isinstance(surface, dict):
        return ""
    chunks = [
        str(surface.get("title") or ""),
        str(surface.get("textSample") or ""),
        str(surface.get("peopleText") or ""),
    ]
    for button in surface.get("buttons") or []:
        if isinstance(button, dict):
            chunks.append(str(button.get("aria") or ""))
            chunks.append(str(button.get("text") or ""))
    return " ".join(chunks)


def _resolve_caption_language(requested: str, surface: Optional[dict]) -> tuple[str, dict]:
    requested_norm = _normalize_caption_language(requested or "Korean")
    # Explicit non-Korean choices always win. The English-hint policy only
    # overrides Mina's Korean default.
    if requested_norm.lower() not in ("", "auto", "default", "korean"):
        return requested_norm, {"source": "explicit"}
    text = _surface_text_for_language(surface)
    matches = [hint for hint in _language_hint_terms() if hint.lower() in text.lower()]
    if matches:
        return "English", {"source": "english_hint", "matches": matches[:8]}
    return "Korean", {"source": "default"}


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

def _is_meet_ui_announcement(text: str) -> bool:
    """Filter Google Meet UI live-region announcements from transcript lines."""
    t = (text or "").strip().lower()
    if not t:
        return True
    noisy_exact = {
        "your camera is off.",
        "your camera is off",
        "your hand is lowered.",
        "your hand is lowered",
        "gemini is taking notes",
        "your camera is on. your microphone is on.",
        "your camera is on. your microphone is on",
        "ask gemini is on. meeting language set to korean.",
        "ask gemini is on. meeting language set to korean",
        "your microphone is off.",
        "your microphone is off",
    }
    if t in noisy_exact:
        return True
    noisy_fragments = (
        "you have joined the call",
        "you are the first one here",
        "you’re the only one here",
        "you're the only one here",
        "waiting for others to join",
    )
    if any(fragment in t for fragment in noisy_fragments):
        return True
    if re.search(r"^(video for .+ was (added to|removed from) the main screen|.+ is on the main screen|the main screen now has)", t):
        return True
    return False


class _BotState:
    """Single-process mutable state, flushed to ``status.json`` on each change."""

    def __init__(self, out_dir: Path, meeting_id: str, url: str):
        self.out_dir = out_dir
        self.meeting_id = meeting_id
        self.url = url
        self.in_call = False
        self.captioning = False
        self.captions_enabled_attempted = False
        self.caption_enable_result: Optional[str] = None
        self.caption_region_found = False
        self.caption_observer_attached = False
        self.last_caption_probe_at: Optional[float] = None
        self.caption_debug_path: Optional[str] = None
        self.caption_language_requested: Optional[str] = None
        self.caption_language_set_result: Optional[str] = None
        self.caption_language_evidence: dict = {}
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
        self.meet_surface: dict = {}
        self.meet_surface_at: Optional[float] = None
        self.admission_evidence: Optional[dict] = None
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
        if not text or _is_meet_ui_announcement(text):
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
            "captionEnableResult": self.caption_enable_result,
            "captionRegionFound": self.caption_region_found,
            "captionObserverAttached": self.caption_observer_attached,
            "lastCaptionProbeAt": self.last_caption_probe_at,
            "captionDebugPath": self.caption_debug_path,
            "captionLanguageRequested": self.caption_language_requested,
            "captionLanguageSetResult": self.caption_language_set_result,
            "captionLanguageEvidence": self.caption_language_evidence,
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
            "meetSurfaceAt": self.meet_surface_at,
            "meetSurface": self.meet_surface,
            "admissionEvidence": self.admission_evidence,
        }
        tmp = self.status_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.status_path)

    def set(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._flush()

    def set_caption_probe(
        self,
        *,
        result: Optional[str] = None,
        region_found: bool = False,
        observer_attached: bool = False,
        debug_path: Optional[str] = None,
    ) -> None:
        self.captions_enabled_attempted = True
        if result is not None:
            self.caption_enable_result = result
        self.caption_region_found = bool(region_found)
        self.caption_observer_attached = bool(observer_attached)
        self.captioning = bool(observer_attached)
        self.last_caption_probe_at = time.time()
        if debug_path is not None:
            self.caption_debug_path = debug_path
        self._flush()

    def set_meet_surface(self, surface: dict, *, admission_evidence: Optional[dict] = None) -> None:
        self.meet_surface = surface or {}
        self.meet_surface_at = time.time()
        if admission_evidence is not None:
            self.admission_evidence = admission_evidence
        self._flush()

    def set_caption_language(self, requested: str, result: str, evidence: Optional[dict] = None) -> None:
        self.caption_language_requested = requested
        self.caption_language_set_result = result
        self.caption_language_evidence = evidence or {}
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
  window.__hermesMeetQueue = window.__hermesMeetQueue || [];
  window.__hermesMeetAttached = false;

  const captionSelector = '[role="region"][aria-label*="aption" i], ' +
                          '[aria-live="polite"], ' +
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

  function scan(root, allowFallback = true) {
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
    if (!allowFallback) return;
    const text = (root.innerText || '').split('\n').filter(Boolean).pop();
    pushEntry('', text);
  }

  function attach() {
    const el = document.querySelector(captionSelector);
    if (el && !window.__hermesMeetObserver) {
      window.__hermesMeetObserver = new MutationObserver(() => scan(el, true));
      window.__hermesMeetObserver.observe(el, { childList: true, subtree: true, characterData: true });
    }
    if (!window.__hermesMeetBodyObserver && document.body) {
      // Same-account/second-tab Meet surfaces sometimes emit actual caption rows
      // outside the first aria-live region we find. Observe body too, but only
      // harvest known caption row structures to avoid arbitrary UI noise.
      window.__hermesMeetBodyObserver = new MutationObserver(() => scan(document.body, false));
      window.__hermesMeetBodyObserver.observe(document.body, { childList: true, subtree: true, characterData: true });
    }
    window.__hermesMeetAttached = Boolean(el || window.__hermesMeetBodyObserver);
    if (el) scan(el, true);
    if (document.body) scan(document.body, false);
    return Boolean(window.__hermesMeetAttached);
  }

  if (!window.__hermesMeetDrain) {
    window.__hermesMeetDrain = () => {
      const out = window.__hermesMeetQueue.slice();
      window.__hermesMeetQueue = [];
      return out;
    };
  }

  if (!attach() && !window.__hermesMeetAttachInterval) {
    window.__hermesMeetAttachInterval = setInterval(() => {
      if (attach()) clearInterval(window.__hermesMeetAttachInterval);
    }, 1500);
  }

  return Boolean(window.__hermesMeetAttached);
})();
"""


def _enable_captions_js() -> str:
    """Return JS that tries to enable Meet captions.

    Prefer clicking the visible captions control first; the keyboard shortcut
    can fail in headless/persistent-profile mode when focus is inside Meet's
    call surface or a modal. Keep the ``c`` shortcut as a fallback.
    """
    return r"""
    (() => {
      const candidates = Array.from(document.querySelectorAll('button,[role="button"]'));
      const direct = candidates.find((b) => {
        const label = `${b.getAttribute('aria-label') || ''} ${b.innerText || ''}`;
        if (!/turn on captions|자막.*켜|자막.*사용/i.test(label)) return false;
        if (/turn off captions|자막.*끄|자막.*중지/i.test(label)) return false;
        return b.offsetParent !== null || b.getClientRects().length > 0;
      });
      if (direct) {
        direct.click();
        return 'clicked_direct';
      }
      const selectors = [
        'button[aria-label*="caption" i]',
        'button[aria-label*="자막" i]',
        '[role="button"][aria-label*="caption" i]',
        '[role="button"][aria-label*="자막" i]',
      ];
      for (const selector of selectors) {
        const buttons = Array.from(document.querySelectorAll(selector));
        const btn = buttons.find((b) => {
          const label = (b.getAttribute('aria-label') || '').toLowerCase();
          const pressed = b.getAttribute('aria-pressed');
          if (pressed === 'true') return false;
          if (/turn off captions|자막 사용 중지/i.test(label)) return false;
          return b.offsetParent !== null || b.getClientRects().length > 0;
        });
        if (btn) {
          btn.click();
          return 'clicked';
        }
      }
      const ev = new KeyboardEvent('keydown', {
        key: 'c', code: 'KeyC', keyCode: 67, which: 67, bubbles: true,
      });
      document.body.dispatchEvent(ev);
      return 'shortcut';
    })();
    """




def _caption_probe_js() -> str:
    """Return JS that reports whether Meet caption DOM is actually present."""
    return r"""
    (() => {
      const el = document.querySelector(
        '[role="region"][aria-label*="aption" i], ' +
        '[aria-live="polite"], ' +
        'div[jsname="YSxPC"], div[jsname="tgaKEf"]'
      );
      return {
        regionFound: Boolean(el),
        observerAttached: Boolean(window.__hermesMeetAttached),
        text: el ? (el.innerText || '').slice(0, 500) : '',
        buttons: Array.from(document.querySelectorAll('button,[role="button"]'))
          .map((b) => b.getAttribute('aria-label') || b.innerText || '')
          .filter(Boolean)
          .slice(0, 80),
      };
    })();
    """


def _write_caption_debug_artifacts(page, out_dir: Path, *, reason: str) -> Optional[str]:
    """Persist lightweight DOM/button evidence when captions do not attach."""
    try:
        data = page.evaluate(_caption_probe_js())
        html_text = page.evaluate("() => document.body ? document.body.innerText.slice(0, 12000) : ''")
        debug = out_dir / "caption_debug.json"
        debug.write_text(json.dumps({"reason": reason, "probe": data, "bodyText": html_text}, indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            page.screenshot(path=str(out_dir / "caption_debug.png"), full_page=True)
        except Exception:
            pass
        return str(debug)
    except Exception:
        return None


def _normalize_caption_language(language: str) -> str:
    """Return Meet's display label for common caption language aliases."""
    raw = (language or "").strip()
    if not raw:
        return ""
    aliases = {
        "ko": "Korean",
        "ko-kr": "Korean",
        "kr": "Korean",
        "korean": "Korean",
        "한국어": "Korean",
        "en": "English",
        "en-us": "English",
        "english": "English",
        "영어": "English",
    }
    return aliases.get(raw.lower(), raw)


def _set_caption_language_js(language: str) -> str:
    """Best-effort JS to set Google Meet's meeting/caption language.

    Meet's settings UI is a custom Material surface that changes frequently.
    This script intentionally uses text/ARIA evidence and returns a structured
    result instead of assuming one stable selector. It is safe to call every few
    seconds: if the requested language already appears in the visible settings
    surface it returns ``already_visible`` without toggling captions.
    """
    target = json.dumps(_normalize_caption_language(language))
    return f"""
    (() => {{
      const target = {target};
      const visible = (el) => !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
      const labelOf = (el) => `${{el.getAttribute('aria-label') || ''}} ${{el.innerText || el.textContent || ''}}`.replace(/\s+/g, ' ').trim();
      const clickFirst = (patterns) => {{
        const els = Array.from(document.querySelectorAll('button,[role=\"button\"],[role=\"menuitem\"],[role=\"tab\"]'));
        for (const pattern of patterns) {{
          const re = new RegExp(pattern, 'i');
          const hit = els.find((el) => visible(el) && re.test(labelOf(el)));
          if (hit) {{ hit.click(); return {{clicked: labelOf(hit).slice(0, 160)}}; }}
        }}
        return null;
      }};
      const bodyText = (document.body && document.body.innerText || '').replace(/\s+/g, ' ').trim();
      const evidence = {{bodyTextTail: bodyText.slice(-1200)}};
      // Do NOT treat Ask Gemini's "Meeting language set to Korean" banner as
      // proof that Live Captions are using Korean. The caption settings panel
      // exposes the actual setting as e.g. "language English closed_caption
      // Live captions". Only that caption-language-local evidence counts.
      const captionLangRe = new RegExp(`language\\s+${{target}}\\s+closed_caption\\s+Live captions|Live captions.{{0,120}}language\\s+${{target}}`, 'i');
      if (target && captionLangRe.test(bodyText)) return {{result: 'already_visible', evidence}};

      const directCaptionSettings = clickFirst(['open caption settings', 'caption settings', '자막 설정']);
      if (directCaptionSettings) evidence.directCaptionSettings = directCaptionSettings.clicked;
      const more = clickFirst(['more options', 'more actions', '기타 옵션', '더보기']);
      if (more) evidence.moreOptions = more.clicked;
      const settings = clickFirst(['^settings$', 'settings', '^설정$', '설정']);
      if (settings) evidence.settings = settings.clicked;
      const captions = clickFirst(['^captions$', 'captions', '^자막$', '자막']);
      if (captions) evidence.captions = captions.clicked;

      const afterOpenText = (document.body && document.body.innerText || '').replace(/\s+/g, ' ').trim();
      evidence.afterOpenTail = afterOpenText.slice(-1600);
      if (target && captionLangRe.test(afterOpenText)) return {{result: 'already_visible_after_open', evidence}};

      const controls = Array.from(document.querySelectorAll('button,[role=\"button\"],[role=\"option\"],[role=\"menuitemradio\"],[role=\"menuitem\"],[role=\"listbox\"],[role=\"combobox\"]'));
      const languageControl = controls.find((el) => {{
        const label = labelOf(el);
        return visible(el)
          && /(meeting language|language of the meeting|spoken language|caption language|언어)/i.test(label)
          && !/ask gemini|gemini/i.test(label);
      }});
      if (languageControl) {{
        evidence.languageControl = labelOf(languageControl).slice(0, 200);
        languageControl.click();
      }}
      const options = Array.from(document.querySelectorAll('button,[role=\"button\"],[role=\"option\"],[role=\"menuitemradio\"],[role=\"menuitem\"],li,div[role=\"listitem\"]'));
      const targetOption = options.find((el) => visible(el) && new RegExp(`(^|\\b)${{target}}($|\\b)`, 'i').test(labelOf(el)));
      if (targetOption) {{
        evidence.targetOption = labelOf(targetOption).slice(0, 200);
        targetOption.click();
        return {{result: 'clicked_language_option', evidence}};
      }}
      return {{result: languageControl ? 'opened_language_control_but_option_not_found' : (settings || captions ? 'settings_opened_option_not_found' : 'settings_not_found'), evidence}};
    }})();
    """


def _set_caption_language(page, state: "_BotState", language: str) -> bool:
    """Try to set Meet's caption/meeting language and record telemetry."""
    requested = _normalize_caption_language(language)
    if not requested or requested.lower() in ("auto", "default", "none", "off", "skip"):
        state.set_caption_language(requested, "skipped", {})
        return False
    try:
        outcome = page.evaluate(_set_caption_language_js(requested)) or {}
        if not isinstance(outcome, dict):
            outcome = {"result": str(outcome)}
        result = str(outcome.get("result") or "unknown")
        evidence = outcome.get("evidence") if isinstance(outcome.get("evidence"), dict) else outcome
        if result == "opened_language_control_but_option_not_found":
            # Meet's language list can be virtualized: after opening the
            # language control, the requested language may not be in the DOM
            # until searched. Try locator and keyboard search before declaring
            # failure, and keep telemetry even when it does not succeed.
            keyboard_evidence = {"attempted": True}
            try:
                try:
                    loc = page.get_by_text(re.compile(r"Korean|한국어", re.I)).first
                    if loc.count():
                        try:
                            loc.click(timeout=2_000, force=True)
                            keyboard_evidence["locatorClicked"] = True
                        except Exception as fe:
                            keyboard_evidence["locatorClickError"] = str(fe)
                        try:
                            loc.evaluate("""
                                (el) => {
                                  let node = el;
                                  for (let i = 0; node && i < 8; i++, node = node.parentElement) {
                                    const label = `${node.getAttribute('aria-label') || ''} ${node.innerText || node.textContent || ''}`;
                                    if (/Korean|한국어/i.test(label)) {
                                      node.click();
                                      const radio = node.querySelector && node.querySelector('input[type=radio], input[type=checkbox]');
                                      if (radio) radio.click();
                                      return;
                                    }
                                  }
                                  el.click();
                                }
                            """)
                            keyboard_evidence["locatorEvaluatedClick"] = True
                        except Exception as ee:
                            keyboard_evidence["locatorEvaluateError"] = str(ee)
                        page.wait_for_timeout(500)
                except Exception as le:
                    keyboard_evidence["locatorError"] = str(le)
                # Typeahead fallback for virtualized language menus.
                page.keyboard.press("KeyK")
                page.wait_for_timeout(100)
                page.keyboard.type("orean", delay=20)
                page.wait_for_timeout(250)
                page.keyboard.press("Enter")
                page.wait_for_timeout(500)
                confirm = page.evaluate(_set_caption_language_js(requested)) or {}
                keyboard_evidence["confirm"] = confirm
                confirm_blob = json.dumps(confirm, ensure_ascii=False) if isinstance(confirm, dict) else str(confirm)
                if isinstance(confirm, dict) and (
                    str(confirm.get("result") or "") in ("already_visible", "already_visible_after_open", "clicked_language_option")
                    or re.search(r"language\s+%s\s+closed_caption\s+Live captions" % re.escape(requested), confirm_blob, re.I)
                ):
                    result = "keyboard_search_selected"
                evidence = {"initial": evidence, "keyboardFallback": keyboard_evidence}
            except Exception as e:
                evidence = {"initial": evidence, "keyboardFallback": {**keyboard_evidence, "error": str(e)}}
        state.set_caption_language(requested, result, evidence)
        return result in ("already_visible", "already_visible_after_open", "clicked_language_option", "keyboard_search_selected")
    except Exception as e:
        state.set_caption_language(requested, f"failed:{type(e).__name__}", {"error": str(e)})
        return False


def _enable_and_probe_captions(page, state: "_BotState", out_dir: Path) -> bool:
    """Try to turn on captions, install observer, and verify attachment."""
    result: Optional[str] = None
    try:
        result = page.evaluate(_enable_captions_js())
        if result not in ("clicked", "clicked_direct"):
            try:
                page.keyboard.press("c")
            except Exception:
                pass
    except Exception as e:
        result = f"failed:{type(e).__name__}"
    observer_attached = False
    try:
        observer_attached = bool(page.evaluate(_CAPTION_OBSERVER_JS))
    except Exception as e:
        state.set(error=f"caption observer install failed: {e}")
    probe = {}
    try:
        probe = page.evaluate(_caption_probe_js()) or {}
    except Exception:
        probe = {}
    region_found = bool(probe.get("regionFound"))
    observer_attached = bool(probe.get("observerAttached") or observer_attached)
    debug_path = None
    if not observer_attached:
        debug_path = _write_caption_debug_artifacts(page, out_dir, reason="caption_observer_not_attached")
    state.set_caption_probe(
        result=str(result or "unknown"),
        region_found=region_found,
        observer_attached=observer_attached,
        debug_path=debug_path,
    )
    return observer_attached

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


def _open_browser_context(pw, *, headed: bool, chrome_args: list, context_args: dict, chrome_profile: str = ""):
    """Open a Playwright browser context for Meet.

    Without a dedicated profile we use the existing ephemeral Chromium launch +
    new_context path. With ``chrome_profile`` we use a persistent user data dir
    so Google login/session state survives bot restarts without reusing the
    user's everyday Chrome profile.

    Returns ``(context, browser)``. ``browser`` is None for persistent contexts,
    because Playwright's persistent context owns the browser lifecycle.
    """
    if chrome_profile:
        persistent_args = dict(context_args)
        # storage_state is only valid for browser.new_context(); persistent
        # contexts load cookies/local storage from the user data dir itself.
        persistent_args.pop("storage_state", None)
        persistent_args.update({
            "headless": not headed,
            "args": chrome_args,
        })
        return pw.chromium.launch_persistent_context(chrome_profile, **persistent_args), None

    browser = pw.chromium.launch(
        headless=not headed,
        args=chrome_args,
    )
    return browser.new_context(**context_args), browser


def run_bot() -> int:  # noqa: C901 — orchestration, explicit branches
    url = os.environ.get("HERMES_MEET_URL", "").strip()
    out_dir_env = os.environ.get("HERMES_MEET_OUT_DIR", "").strip()
    headed = os.environ.get("HERMES_MEET_HEADED", "").lower() in ("1", "true", "yes")
    auth_state = os.environ.get("HERMES_MEET_AUTH_STATE", "").strip()
    chrome_profile = os.environ.get("HERMES_MEET_CHROME_PROFILE", "").strip()
    guest_name = os.environ.get("HERMES_MEET_GUEST_NAME", "Hermes Agent")
    join_style = os.environ.get("HERMES_MEET_JOIN_STYLE", "normal").strip().lower()
    caption_language = _normalize_caption_language(os.environ.get("HERMES_MEET_CAPTION_LANGUAGE", "Korean"))
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
            context_args = {
                "viewport": {"width": 1280, "height": 800},
                "user_agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                ),
                "permissions": ["microphone", "camera"],
            }
            if auth_state and Path(auth_state).is_file() and not chrome_profile:
                context_args["storage_state"] = auth_state
            context, browser = _open_browser_context(
                pw,
                headed=headed,
                chrome_args=chrome_args,
                context_args=context_args,
                chrome_profile=chrome_profile,
            )
            page = context.new_page()

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            except Exception as e:
                state.set(error=f"navigate failed: {e}", exited=True)
                return 4

            # Optional companion mode: use Google's low-presence second-screen
            # join path when available. This keeps mic/audio/video off and is
            # less visually intrusive than a normal participant tile.
            if join_style == "companion":
                _try_companion_mode(page)

            # Guest-mode: Meet shows a name field before "Ask to join". When
            # we're authed, we instead see "Join now".
            _try_guest_name(page, guest_name)
            _click_join(page, state)
            if state.error:
                state.set(exited=True, leave_reason="join_button_not_found")
                return 6

            # Install caption observer and attempt to enable captions. We do
            # not report captioning=true until the Meet caption region exists
            # and the MutationObserver is actually attached. Korean is the
            # default because most of Joohyun's meetings are Korean; recurring
            # English meetings can be recognized by title/participant hints.
            initial_surface = _probe_meet_surface(page)
            state.set_meet_surface(initial_surface)
            caption_language, language_policy = _resolve_caption_language(caption_language, initial_surface)
            state.set_caption_language(caption_language, f"policy:{language_policy.get('source', 'unknown')}", language_policy)
            _set_caption_language(page, state, caption_language)
            _enable_and_probe_captions(page, state, out_dir)

            # Note: in_call=False until admission is confirmed (we detect
            # either the Leave button or the caption region, signalling we
            # made it past the lobby).
            state.set(join_attempted_at=time.time())

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
            last_caption_retry = 0.0
            last_alone_check = 0.0
            last_surface_refresh = 0.0
            alone_since: Optional[float] = None
            alone_grace_s = float(os.environ.get("HERMES_MEET_ALONE_GRACE", "120"))
            # Companion mode is often a second-screen session for the same
            # signed-in Google account. Meet may not count the user's primary
            # tab as "someone else" from that companion session, which makes
            # our old auto-leave heuristic fire during valid tests. Keep
            # normal joins protected by default, but do not auto-exit companion
            # sessions unless explicitly requested.
            alone_exit_enabled = os.environ.get(
                "HERMES_MEET_ALONE_EXIT",
                "0" if join_style == "companion" else "1",
            ).lower() not in ("0", "false", "no", "off")
            while not stop_flag["stop"]:
                now = time.time()
                if deadline and now > deadline:
                    state.set(leave_reason="duration_expired")
                    break

                # Admission detection every ~3s until admitted.
                if not state.in_call and (now - last_admission_check) > 3.0:
                    last_admission_check = now
                    surface = _probe_meet_surface(page)
                    evidence = _admission_evidence(surface)
                    state.set_meet_surface(surface, admission_evidence=evidence)
                    if evidence:
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

                if state.in_call:
                    _dismiss_got_it(page)
                    _ensure_prejoin_listen_only(page)
                    _try_stay_in_call(page)
                    if state.caption_language_set_result not in ("already_visible", "already_visible_after_open", "clicked_language_option", "skipped"):
                        _set_caption_language(page, state, caption_language)
                    if (now - last_surface_refresh) > 5.0:
                        last_surface_refresh = now
                        surface = _probe_meet_surface(page)
                        state.set_meet_surface(surface)
                        resolved_language, language_policy = _resolve_caption_language(caption_language, surface)
                        if resolved_language != caption_language:
                            caption_language = resolved_language
                            state.set_caption_language(caption_language, f"policy:{language_policy.get('source', 'unknown')}", language_policy)
                            _set_caption_language(page, state, caption_language)
                        surface_labels = " ".join(
                            f"{button.get('aria', '')} {button.get('text', '')}"
                            for button in (surface.get("buttons") or [])
                            if isinstance(button, dict)
                        )
                        if re.search(r"turn on captions|자막.*켜|자막.*사용", surface_labels, re.I):
                            _enable_and_probe_captions(page, state, out_dir)

                # Once admitted, leave automatically after everyone else is
                # gone for a short grace period. This supports the intended
                # transcribe-only mode for normal joins. In Companion mode this
                # is disabled by default because a same-account companion
                # session can see the primary Joohyun tab as "self", not as an
                # other participant.
                if alone_exit_enabled and state.in_call and (now - last_alone_check) > 10.0:
                    last_alone_check = now
                    if _detect_no_other_participants(page):
                        if alone_since is None:
                            alone_since = now
                        elif (now - alone_since) >= alone_grace_s:
                            state.set(leave_reason="everyone_left")
                            break
                    else:
                        alone_since = None

                # Retry caption enable/observer attachment while in-call; Meet
                # sometimes renders controls only after admission or focus.
                if state.in_call and (not state.caption_region_found or not state.caption_observer_attached) and (now - last_caption_retry) > 5.0:
                    last_caption_retry = now
                    _enable_and_probe_captions(page, state, out_dir)

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
            if browser is not None:
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


def _try_companion_mode(page) -> bool:
    """Best-effort: switch Meet prejoin into Companion Mode.

    Google exposes this either as a direct "Use Companion mode" action or
    behind "Other ways to join". The prejoin UI is lazy-rendered, and the
    option often appears only after dismissing the "Switch here" hint, so poll
    briefly and try both role-based and text-based locators.
    """
    companion_labels = (
        "Use Companion mode",
        "Use companion mode",
        "Companion mode",
        "동반 모드 사용",
        "컴패니언 모드",
    )
    other_ways_labels = (
        "Other ways to join",
        "Other joining options",
        "다른 참여 방법",
        "기타 참여 옵션",
    )

    def _click_visible(label: str, *, include_text: bool = True) -> bool:
        pattern = re.compile(re.escape(label), re.I)
        locators = []
        for role in ("button", "link"):
            try:
                locators.append(page.get_by_role(role, name=pattern).first)
            except Exception:
                pass
        if include_text:
            try:
                locators.append(page.get_by_text(pattern).first)
            except Exception:
                pass
        for loc in locators:
            try:
                if loc.count() and loc.is_visible():
                    loc.click(timeout=3_000)
                    return True
            except Exception:
                continue
        return False

    deadline = time.time() + float(os.environ.get("HERMES_MEET_COMPANION_TIMEOUT", "20"))
    expanded_other_ways = False
    while time.time() < deadline:
        # Existing-call hint can cover the alternate join controls.
        _dismiss_got_it(page)
        _ensure_prejoin_listen_only(page)

        for label in companion_labels:
            if _click_visible(label):
                return True

        if not expanded_other_ways:
            for label in other_ways_labels:
                if _click_visible(label, include_text=False):
                    expanded_other_ways = True
                    try:
                        page.wait_for_timeout(500)
                    except Exception:
                        time.sleep(0.5)
                    break

        try:
            page.wait_for_timeout(500)
        except Exception:
            time.sleep(0.5)
    return False


def _dismiss_got_it(page) -> bool:
    """Dismiss Meet transient hints/panels if they are covering call controls."""
    clicked = False
    try:
        got_it = page.get_by_role("button", name="Got it", exact=True).first
        if got_it.count() and got_it.is_visible():
            got_it.click(timeout=3_000)
            clicked = True
    except Exception:
        pass
    try:
        has_ask_gemini_panel = page.evaluate(
            "() => /New Ask Gemini|Gemini is available to answer questions/i.test(document.body ? document.body.innerText || '' : '')"
        )
        if has_ask_gemini_panel:
            close = page.get_by_role("button", name="Close", exact=True).first
            if close.count() and close.is_visible():
                close.click(timeout=3_000)
                clicked = True
    except Exception:
        pass
    return clicked


def _try_guest_name(page, guest_name: str) -> None:
    """If Meet is showing a guest-name input, type *guest_name* into it."""
    try:
        # Meet's guest name input has placeholder "Your name".
        locator = page.locator('input[aria-label*="name" i]').first
        if locator.count() and locator.is_visible():
            locator.fill(guest_name, timeout=2_000)
    except Exception:
        pass


def _probe_meet_surface(page) -> dict:
    """Return compact visible Meet UI evidence for status/debugging."""
    probe = r"""
    (() => {
      const text = (document.body && document.body.innerText || '').replace(/\s+/g, ' ').trim();
      const buttons = Array.from(document.querySelectorAll('button[aria-label], button'))
        .map((button) => ({
          aria: button.getAttribute('aria-label') || '',
          text: (button.innerText || '').replace(/\s+/g, ' ').trim(),
          visible: !!(button.offsetWidth || button.offsetHeight || button.getClientRects().length),
        }))
        .filter((button) => button.visible && (button.aria || button.text))
        .slice(0, 80);
      const callControls = buttons.filter((button) => {
        const label = `${button.aria} ${button.text}`;
        return /leave call|hang up|raise hand|captions|microphone|camera|present now|more options|통화에서 나가기|마이크|카메라|자막|발표/i.test(label);
      });
      const peopleText = text.match(/(?:Show everyone|People|참여자|Everyone|모든 사용자).{0,120}/i);
      return {
        href: location.href,
        title: document.title || '',
        textSample: text.slice(0, 1200),
        buttons,
        callControls,
        hasVideoCallPath: /\/[^/]*[a-z]{3}-[a-z]{4}-[a-z]{3}/i.test(location.pathname),
        hasPrejoinText: /ready to join|ask to join|join now|other ways to join|use companion mode|참여 요청|지금 참여|다른 참여|컴패니언/i.test(text),
        hasLobbyText: /asking to be let in|waiting.*admit|host.*let you in|참여.*요청|승인.*대기|대기 중/i.test(text),
        hasCallText: /you’re the only one here|you're the only one here|you are the only one here|show everyone|present now|raise hand|captions|참여자|발표|손들기|자막/i.test(text),
        peopleText: peopleText ? peopleText[0] : '',
      };
    })();
    """
    try:
        value = page.evaluate(probe)
        return value if isinstance(value, dict) else {}
    except Exception as e:
        return {"probeError": str(e)}


def _admission_evidence(surface: dict) -> Optional[dict]:
    """Return evidence dict only when UI strongly indicates in-call state."""
    controls = surface.get("callControls") or []
    labels = " ".join(
        f"{control.get('aria', '')} {control.get('text', '')}" for control in controls if isinstance(control, dict)
    )
    has_leave = bool(re.search(r"leave call|hang up|통화에서 나가기", labels, re.I))
    has_second_control = bool(re.search(r"raise hand|present now|captions|microphone|camera|손들기|발표|자막|마이크|카메라", labels, re.I))
    button_labels = " ".join(
        f"{button.get('aria', '')} {button.get('text', '')}" for button in (surface.get("buttons") or []) if isinstance(button, dict)
    )
    text_sample = f"{surface.get('textSample') or ''} {button_labels}"
    has_stay_prompt = bool(re.search(r"call is ending soon|stay in the call|통화.*곧.*종료|계속.*통화", text_sample, re.I))
    # Same-account second-tab joins can show a "Your call is ending soon" modal
    # with a "Join now" button after the browser is already in the call. That
    # leaves Meet's text looking prejoin-ish even though the high-signal call
    # controls are present. Treat that as admitted so the main loop can click
    # the stay prompt and attach captions.
    if has_leave and has_second_control and not surface.get("hasLobbyText") and (not surface.get("hasPrejoinText") or has_stay_prompt):
        return {
            "method": "leave_plus_call_controls" + ("_with_stay_prompt" if has_stay_prompt else ""),
            "labels": labels[:500],
            "href": surface.get("href"),
            "title": surface.get("title"),
        }
    return None


def _detect_admission(page) -> bool:
    surface = _probe_meet_surface(page)
    return _admission_evidence(surface) is not None


def _detect_denied(page) -> bool:
    """True when Meet is showing a 'you were denied' / 'no one admitted' page."""
    probe = r"""
    (() => {
      const text = document.body ? document.body.innerText || '' : '';
      // English only — matches what shows up when the host denies or
      // removes a guest.
      if (/You can't join this video call/i.test(text)) return true;
      if (/You were removed from the meeting/i.test(text)) return true;
      if (/No one responded to your request to join/i.test(text)) return true;
      return false;
    })();
    """
    try:
        return bool(page.evaluate(probe))
    except Exception:
        return False


def _try_stay_in_call(page) -> bool:
    """Click Meet's auto-end prompt when the tab looks alone.

    Same-account Companion Mode can show "Your call is ending soon" even while
    the user's primary tab is present. If Meet offers an affirmative stay
    action, click it so transcription tests do not die before captions arrive.
    """
    probe = r"""
    (() => {
      const text = document.body ? document.body.innerText || '' : '';
      const buttons = Array.from(document.querySelectorAll('button'));
      const buttonText = buttons.map((button) => `${button.getAttribute('aria-label') || ''} ${button.innerText || ''}`).join(' ');
      if (!/call is ending soon|stay in the call|통화.*곧.*종료|계속.*통화/i.test(`${text} ${buttonText}`)) return false;
      for (const button of buttons) {
        const label = `${button.getAttribute('aria-label') || ''} ${button.innerText || ''}`;
        if (/stay|keep|continue|join now|계속|유지|지금 참여/i.test(label)) {
          button.click();
          return true;
        }
      }
      return false;
    })();
    """
    try:
        return bool(page.evaluate(probe))
    except Exception:
        return False


def _detect_no_other_participants(page) -> bool:
    """True when Meet indicates the bot is alone in the call.

    This is intentionally conservative and text-based because Meet's people
    panel DOM changes often. The strings below are the high-signal empty-room
    prompts shown after everyone else has left or before anyone else joins.
    """
    probe = r"""
    (() => {
      const text = document.body ? document.body.innerText || '' : '';
      if (/you(?:'|’)?re the only one here/i.test(text)) return true;
      if (/you're the only one in this call/i.test(text)) return true;
      if (/no one else is here/i.test(text)) return true;
      if (/waiting for others to join/i.test(text)) return true;
      if (/다른 참여자가 없습니다/i.test(text)) return true;
      if (/나만 참여 중/i.test(text)) return true;
      return false;
    })();
    """
    try:
        return bool(page.evaluate(probe))
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
    if spk in ("unknown", "you", bot_guest_name.strip().lower()):
        return False
    return True


def _ensure_prejoin_listen_only(page) -> None:
    """Best-effort: make Meet prejoin controls listen-only before joining.

    We only click positive "Turn off ..." controls. If Meet already shows
    "Turn on microphone" / "Turn on camera", that means the device is already
    muted/off and must be left alone. This keeps the bot safe for the user's
    expected mode: camera off, microphone off, transcribe-only.
    """
    for label in ("Turn off microphone", "Turn off camera"):
        try:
            btn = page.get_by_role("button", name=label, exact=False).first
            if btn.count() and btn.is_visible():
                btn.click(timeout=3_000)
        except Exception:
            pass


def _click_join(page, state: _BotState) -> None:
    """Click 'Join now' or 'Ask to join' once Meet's pre-join UI is ready.

    Meet often renders the pre-join controls after ``domcontentloaded`` by a
    few seconds. A single immediate probe can silently miss the button, leaving
    the bot alive but not actually requesting admission. Poll briefly and make
    the failure visible in status.json so smoke tests don't look like a stuck
    lobby.

    Flags ``lobby_waiting`` when we hit the "waiting for host to admit you"
    state so the agent can surface that in status.
    """
    deadline = time.time() + float(os.environ.get("HERMES_MEET_JOIN_BUTTON_TIMEOUT", "30"))
    last_error: Optional[str] = None
    expanded_other_ways = False
    dismissed_switch_hint = False
    while time.time() < deadline:
        # https://meet.google.com/new can auto-create and enter a meeting for a
        # signed-in user, skipping the pre-join lobby entirely. Treat that as
        # success instead of waiting for a non-existent Join button.
        if _detect_admission(page):
            return

        if not dismissed_switch_hint:
            try:
                got_it = page.get_by_role("button", name="Got it", exact=True).first
                if got_it.count() and got_it.is_visible():
                    got_it.click(timeout=3_000)
                    dismissed_switch_hint = True
            except Exception as e:
                last_error = str(e)

        _ensure_prejoin_listen_only(page)

        # For a signed-in same-account second tab, "Join here too" is the
        # safe path: it adds this browser as another participant while keeping
        # the user's primary tab in the call. Prefer it over "Join now" when
        # both are visible.
        for label in ("Join here too", "Join now", "Ask to join"):
            try:
                btn = page.get_by_role("button", name=label, exact=False).first
                if btn.count() and btn.is_visible():
                    btn.click(timeout=3_000)
                    if label == "Ask to join":
                        state.set(lobby_waiting=True)
                    return
            except Exception as e:
                last_error = str(e)
                continue

        # If the same Google account is already in the meeting elsewhere,
        # Meet may hide the safe second-device path behind "Other ways to
        # join" and show a prominent "Switch here" button instead. Do not
        # click "Switch here" because it would steal the user's active call;
        # expand the menu and prefer "Join here too".
        if not expanded_other_ways:
            try:
                other = page.get_by_role("button", name="Other ways to join", exact=False).first
                if other.count() and other.is_visible():
                    other.click(timeout=3_000)
                    expanded_other_ways = True
            except Exception as e:
                last_error = str(e)

        try:
            page.wait_for_timeout(500)
        except Exception as e:
            last_error = str(e)
            break
    state.set(error=("join button not found" + (f": {last_error}" if last_error else "")))


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
