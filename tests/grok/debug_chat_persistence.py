"""Debug script: instrument what happens to grok.com page state across sends.

Sends 3 messages in sequence using the SAME _BrowserThread instance (same as
real Hermes session). Logs page.url, tab count, .ProseMirror count, and
_in_conversation state at every step so we can see EXACTLY where the
chat-persistence assumption breaks.

Run:
  cd ~/Projects/hermes-agent && python3 tests/grok/debug_chat_persistence.py

Output:
  /tmp/grok_chat_persistence.log  — full timeline of state transitions
  /tmp/grok_chat_persistence_*.png — screenshots after each send

What to look for in the log:
  - Did self._page.url change between sends? (expected: yes after turn 1, then stable)
  - Did context.pages count grow? (unexpected: would mean new tabs)
  - How many .ProseMirror elements on the page after turn 1? (>1 = ambiguous)
  - Did _in_conversation stay True? (must)
  - After typing turn 2, did URL stay same or did SPA navigate to a new chat?
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.patchright_client import (
    _BrowserThread,
    _GROK_URL,
    _EDITOR_SELECTOR,
    _get_profile_dir,
)

LOG_PATH = Path("/tmp/grok_chat_persistence.log")
SHOTS_DIR = Path("/tmp")
log_buffer = []


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    log_buffer.append(line)


def snapshot(bt, label: str):
    """Capture full state of the browser thread."""
    page = bt._page
    ctx = bt._context
    try:
        url = page.url
    except Exception as e:
        url = f"<error: {e}>"
    try:
        page_count = len(ctx.pages) if ctx else "?"
    except Exception as e:
        page_count = f"<error: {e}>"
    try:
        prosemirror_count = len(page.query_selector_all(_EDITOR_SELECTOR))
    except Exception as e:
        prosemirror_count = f"<error: {e}>"
    try:
        # How many bubbles already on page
        bubble_count = page.evaluate(
            "() => document.querySelectorAll('.message-bubble').length"
        )
    except Exception as e:
        bubble_count = f"<error: {e}>"
    try:
        # What page text says about the active mode
        mode_label = page.evaluate(
            """() => {
                const triggers = Array.from(document.querySelectorAll('button[aria-haspopup="menu"]'));
                const known = ['Auto', 'Fast', 'Expert', 'Heavy', 'Grok 4.3 (beta)'];
                for (const b of triggers) {
                    const t = (b.innerText || '').trim();
                    if (known.includes(t)) return t;
                }
                return '<no mode trigger found>';
            }"""
        )
    except Exception as e:
        mode_label = f"<error: {e}>"

    log(f"--- STATE: {label} ---")
    log(f"  URL:                  {url}")
    log(f"  context.pages count:  {page_count}")
    log(f"  .ProseMirror count:   {prosemirror_count}")
    log(f"  .message-bubble count: {bubble_count}")
    log(f"  mode trigger label:   {mode_label}")
    log(f"  _in_conversation:     {bt._in_conversation}")
    log("")


def detect_popups(bt):
    """Attach a listener to detect new tabs/pages opening."""
    def on_popup(popup):
        log(f"!!! POPUP DETECTED: new page opened with URL: {popup.url}")
        log(f"    Now total pages: {len(bt._context.pages)}")

    bt._context.on("page", on_popup)
    log("[detect_popups] popup/page listener attached")


def main():
    profile_dir = _get_profile_dir()
    if not profile_dir.exists():
        print(f"[ERROR] Profile dir not found at {profile_dir}")
        print("        Run a regular hermes session first to log in.")
        sys.exit(1)

    log(f"=== Debug: chat persistence across 3 sends ===")
    log(f"Profile: {profile_dir}")

    # Build _BrowserThread without using the singleton (don't pollute real state)
    bt = _BrowserThread()
    bt.start()
    if not bt._ready.wait(timeout=30):
        log("[ERROR] Browser thread not ready in 30s")
        sys.exit(1)

    log("[OK] Browser thread launched")
    detect_popups(bt)
    snapshot(bt, "after launch (no nav yet)")

    # === MESSAGE 1 ===
    log("\n=== MESSAGE 1: 'hello, what hostname am i on?' ===")
    snapshot(bt, "before _send_message #1")
    try:
        # Use _send_to_browser-style dispatch via cmd queue
        bt._cmd_queue.put((
            "send_message",
            {
                "message": "hello, what hostname am i on?",
                "mode": "grok-fast",
                "timeout": 60,
            },
        ))
        status, result = bt._result_queue.get(timeout=120)
        log(f"  result status: {status}")
        if status == "error":
            log(f"  ERROR: {result}")
        else:
            log(f"  response (first 200 chars): {result[:200]!r}")
    except Exception as e:
        log(f"  EXCEPTION during send #1: {e}")

    snapshot(bt, "after _send_message #1")
    try:
        bt._page.screenshot(path=str(SHOTS_DIR / "grok_chat_persistence_1.png"))
        log("  screenshot: /tmp/grok_chat_persistence_1.png")
    except Exception as e:
        log(f"  screenshot failed: {e}")

    # === MESSAGE 2 ===
    log("\n=== MESSAGE 2: 'now show me the OS version' ===")
    snapshot(bt, "before _send_message #2")
    try:
        bt._cmd_queue.put((
            "send_message",
            {
                "message": "now show me the OS version on this same machine",
                "mode": "grok-fast",
                "timeout": 60,
            },
        ))
        status, result = bt._result_queue.get(timeout=120)
        log(f"  result status: {status}")
        if status == "error":
            log(f"  ERROR: {result}")
        else:
            log(f"  response (first 200 chars): {result[:200]!r}")
    except Exception as e:
        log(f"  EXCEPTION during send #2: {e}")

    snapshot(bt, "after _send_message #2")
    try:
        bt._page.screenshot(path=str(SHOTS_DIR / "grok_chat_persistence_2.png"))
        log("  screenshot: /tmp/grok_chat_persistence_2.png")
    except Exception as e:
        log(f"  screenshot failed: {e}")

    # === MESSAGE 3 ===
    log("\n=== MESSAGE 3: 'and the kernel version?' ===")
    snapshot(bt, "before _send_message #3")
    try:
        bt._cmd_queue.put((
            "send_message",
            {
                "message": "and the kernel version on the same machine?",
                "mode": "grok-fast",
                "timeout": 60,
            },
        ))
        status, result = bt._result_queue.get(timeout=120)
        log(f"  result status: {status}")
        if status == "error":
            log(f"  ERROR: {result}")
        else:
            log(f"  response (first 200 chars): {result[:200]!r}")
    except Exception as e:
        log(f"  EXCEPTION during send #3: {e}")

    snapshot(bt, "after _send_message #3")
    try:
        bt._page.screenshot(path=str(SHOTS_DIR / "grok_chat_persistence_3.png"))
        log("  screenshot: /tmp/grok_chat_persistence_3.png")
    except Exception as e:
        log(f"  screenshot failed: {e}")

    # === ANALYSIS ===
    log("\n=== ANALYSIS ===")
    log("Compare URLs across snapshots:")
    log("  - If URL stayed the same after #1 → SAME chat used (good)")
    log("  - If URL changed each time → NEW chat each turn (BUG)")
    log("Compare context.pages count:")
    log("  - If always 1 → single tab (good)")
    log("  - If grew → new tabs opened (BUG)")
    log("Compare .ProseMirror count:")
    log("  - If 1 → single editor (good)")
    log("  - If >1 → ambiguous (.ProseMirror selector matches sidebar + main)")

    # Save log
    LOG_PATH.write_text("\n".join(log_buffer))
    log(f"\n[DONE] Full log saved to {LOG_PATH}")
    log(f"[DONE] Screenshots in {SHOTS_DIR}/grok_chat_persistence_*.png")
    LOG_PATH.write_text("\n".join(log_buffer))  # rewrite with last lines

    # Cleanup
    log("\nClosing browser thread in 5s (Ctrl+C to inspect first)...")
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        log("Inspection requested — leaving browser open. Ctrl+C again to exit.")
        try:
            time.sleep(60)
        except KeyboardInterrupt:
            pass
    LOG_PATH.write_text("\n".join(log_buffer))

    bt._cmd_queue.put(("close", {}))
    try:
        bt._result_queue.get(timeout=10)
    except Exception:
        pass


if __name__ == "__main__":
    main()
