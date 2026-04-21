"""Deep diagnostic for chat-persistence bug — captures EVERYTHING.

Thread-safe (runs Patchright in main thread, not via _BrowserThread which
caused greenlet errors yesterday). Simulates real Hermes flow by using the
actual _GrokPromptFormatter to build prompts identical to what production
sends, then drives the browser directly.

Captures:
  - DOM snapshots at every state (ProseMirror count, parents, placeholders,
    visibility, data-testid, aria attrs, text preview)
  - URL transitions (via framenavigated event + explicit snapshots)
  - context.pages count (catches new tab spawns)
  - NEW PAGE events (spawned tabs/popups)
  - ALL HTTP requests to grok.com (URL + method + POST body)
  - HTTP response status codes
  - Screenshots at each major state

Flow:
  1. Launch browser + attach listeners
  2. Navigate to grok.com/ — snapshot STATE 1
  3. Send prompt 1 (real Hermes first-turn format with tools)
  4. Wait for response — snapshot STATE 2
  5. Send prompt 2 (real Hermes CONTINUATION format — simulates tool result flowing back)
  6. Before clicking editor: snapshot STATE 3 (which ProseMirror are we about to type into?)
  7. After response: snapshot STATE 4
  8. Dump all to /tmp/grok_deep_trace.log + network log + screenshots

Run:
  cd ~/Projects/hermes-agent && python3 tests/grok/debug_chat_deep.py

Outputs:
  /tmp/grok_deep_trace.log       — full timeline (DOM, URLs, events)
  /tmp/grok_deep_network.log     — HTTP requests to grok.com
  /tmp/grok_deep_1.png           — after initial nav
  /tmp/grok_deep_2.png           — after prompt 1 + response
  /tmp/grok_deep_3.png           — before prompt 2 send
  /tmp/grok_deep_4.png           — after prompt 2 + response
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.patchright_client import _GROK_URL, _get_profile_dir, _GrokPromptFormatter, _MODE_LABELS

TRACE_LOG = Path("/tmp/grok_deep_trace.log")
NETWORK_LOG = Path("/tmp/grok_deep_network.log")
SHOTS_DIR = Path("/tmp")

trace_lines = []
network_lines = []


def ts():
    return time.strftime("%H:%M:%S")


def log(msg):
    line = f"[{ts()}] {msg}"
    print(line)
    trace_lines.append(line)


def net_log(msg):
    line = f"[{ts()}] {msg}"
    network_lines.append(line)


def snapshot_dom(page, label):
    """Snapshot DOM state — ProseMirror details, bubble count, URL."""
    log(f"\n━━━ DOM SNAPSHOT: {label} ━━━")
    try:
        log(f"  URL: {page.url}")
    except Exception as e:
        log(f"  URL: <error: {e}>")

    try:
        prosemirrors = page.query_selector_all(".ProseMirror")
        log(f"  .ProseMirror count: {len(prosemirrors)}")
        for i, pm in enumerate(prosemirrors):
            try:
                info = pm.evaluate("""el => {
                    let parent = el.closest('nav, aside, main, [role="main"], footer, header');
                    let parentId = parent?.id || '';
                    let parentCls = (parent?.className || '').slice(0, 60);
                    return {
                        parent: parent?.tagName || '(none)',
                        parentId,
                        parentCls,
                        placeholder: el.getAttribute('placeholder') || '',
                        ariaLabel: el.getAttribute('aria-label') || '',
                        ariaPlaceholder: el.getAttribute('aria-placeholder') || '',
                        testId: el.getAttribute('data-testid') || '',
                        visible: el.offsetParent !== null,
                        width: el.offsetWidth,
                        height: el.offsetHeight,
                        text: (el.innerText || '').slice(0, 50),
                    };
                }""")
                placeholder = (
                    info["placeholder"]
                    or info["ariaPlaceholder"]
                    or info["ariaLabel"]
                    or "(none)"
                )
                log(
                    f"  PM[{i}]: parent=<{info['parent']} id={info['parentId']!r} cls={info['parentCls']!r}> "
                    f"placeholder={placeholder!r} "
                    f"visible={info['visible']} size={info['width']}x{info['height']} "
                    f"testId={info['testId']!r} "
                    f"text={info['text']!r}"
                )
            except Exception as e:
                log(f"  PM[{i}]: <error: {e}>")
    except Exception as e:
        log(f"  ProseMirror inspection error: {e}")

    try:
        bubble_count = page.evaluate(
            "() => document.querySelectorAll('.message-bubble').length"
        )
        log(f"  .message-bubble count: {bubble_count}")
    except Exception as e:
        log(f"  .message-bubble error: {e}")


def snapshot_context(context, label):
    """Snapshot browser context — how many pages/tabs exist."""
    log(f"  context.pages count: {len(context.pages)}")
    for i, p in enumerate(context.pages):
        try:
            url = p.url
        except Exception:
            url = "<error>"
        log(f"    page[{i}]: url={url}")


def send_message(page, message: str, label: str, wait_timeout: int = 90):
    """Send a message via the ProseMirror editor — mirrors production code path.

    Uses the EXACT same logic as _BrowserThread._send_message:
      1. query_selector('.ProseMirror') — first match in DOM order
      2. click it
      3. paste content via document.execCommand
      4. press Enter
      5. wait for .message-bubble response
    """
    log(f"\n▶ Sending: {label}")
    log(f"  message length: {len(message)} chars")
    log(f"  message preview: {message[:120]!r}...")

    editor = page.query_selector(".ProseMirror")
    if not editor:
        log(f"  [ERROR] No .ProseMirror found!")
        return None

    # Log WHICH ProseMirror we're about to click (production uses first in DOM)
    try:
        editor_info = editor.evaluate("""el => {
            let parent = el.closest('nav, aside, main, [role="main"], footer, header');
            return {
                parent: parent?.tagName || '(none)',
                parentId: parent?.id || '',
                parentCls: (parent?.className || '').slice(0, 60),
                placeholder: el.getAttribute('placeholder') || el.getAttribute('aria-placeholder') || el.getAttribute('aria-label') || '',
            };
        }""")
        log(
            f"  Clicking PM[0]: parent=<{editor_info['parent']} "
            f"id={editor_info['parentId']!r} cls={editor_info['parentCls']!r}> "
            f"placeholder={editor_info['placeholder']!r}"
        )
    except Exception as e:
        log(f"  [WARN] Could not inspect PM[0]: {e}")

    editor.click()
    time.sleep(0.3)

    # Paste (production code uses this for messages > 500 chars)
    page.evaluate(
        """(text) => {
        const editor = document.querySelector('.ProseMirror');
        if (editor) {
            editor.focus();
            document.execCommand('insertText', false, text);
        }
    }""",
        message,
    )
    time.sleep(0.5)
    page.keyboard.press("Enter")
    log("  Enter pressed. Waiting for response...")

    start = time.time()
    while time.time() - start < wait_timeout:
        time.sleep(1)
        try:
            result = page.evaluate(
                """() => {
                const bubbles = document.querySelectorAll('.message-bubble');
                if (bubbles.length < 2) return {text: '', done: false, count: bubbles.length};
                const last = bubbles[bubbles.length - 1];
                const md = last.querySelector('.response-content-markdown');
                const text = ((md || last).innerText || '').trim();
                const done = !!last.parentElement?.querySelector('.action-buttons');
                return {text, done, count: bubbles.length};
            }"""
            )
            if result.get("text") and result.get("done"):
                elapsed = time.time() - start
                log(
                    f"  Response captured after {elapsed:.1f}s "
                    f"(bubble count: {result.get('count')})"
                )
                log(f"    preview: {result['text'][:200]!r}")
                return result["text"]
        except Exception:
            pass

    log(f"  [WARN] Response wait timed out after {wait_timeout}s")
    return None


def main():
    from patchright.sync_api import sync_playwright

    profile_dir = _get_profile_dir()
    if not profile_dir.exists():
        log(f"[ERROR] Profile dir not found at {profile_dir}")
        log("        Run a regular hermes session first to log in.")
        sys.exit(1)

    log("═══ DEEP CHAT PERSISTENCE DIAGNOSTIC ═══")
    log(f"Profile: {profile_dir}")
    log(f"Target URL: {_GROK_URL}")

    pw = sync_playwright().start()
    context = pw.chromium.launch_persistent_context(
        user_data_dir=str(profile_dir),
        headless=False,  # show window so we can see
        viewport={"width": 1280, "height": 900},
        args=["--disable-blink-features=AutomationControlled"],
    )
    page = context.pages[0] if context.pages else context.new_page()

    # ─── Event listeners ─────────────────────────────────────────
    def on_frame_nav(frame):
        if frame == page.main_frame:
            log(f"  ⚡ FRAME NAVIGATED: {frame.url}")

    def on_new_page(new_page):
        log(f"  ⚡ NEW PAGE created in context (total={len(context.pages)}): url={new_page.url}")

    def on_popup(popup):
        log(f"  ⚡ POPUP detected: url={popup.url}")

    def on_request(request):
        url = request.url
        # Filter to grok.com + api requests to reduce noise
        if "grok.com" in url or "/api/" in url or "x.ai" in url:
            method = request.method
            net_log(f"REQ  {method:4s} {url}")
            if method == "POST":
                try:
                    post_data = request.post_data
                    if post_data:
                        # Truncate huge bodies
                        body_preview = post_data[:600]
                        net_log(f"     BODY[{len(post_data)}]: {body_preview}")
                except Exception:
                    pass

    def on_response(response):
        url = response.url
        if "grok.com" in url or "/api/" in url or "x.ai" in url:
            net_log(f"RES  {response.status}  {url}")

    page.on("framenavigated", on_frame_nav)
    page.on("popup", on_popup)
    page.on("request", on_request)
    page.on("response", on_response)
    context.on("page", on_new_page)

    log("Event listeners attached (framenavigated, popup, request, response, new page)")
    # ────────────────────────────────────────────────────────────

    # Navigate to grok.com (domcontentloaded is faster + more reliable than networkidle
    # on grok.com — the SPA has persistent connections that keep network "busy")
    log(f"\n▶ Navigating to {_GROK_URL}")
    page.goto(_GROK_URL, wait_until="domcontentloaded", timeout=60000)
    time.sleep(5)  # let SPA hydrate

    # Dismiss cookie banner
    try:
        btn = page.query_selector("#onetrust-accept-btn-handler")
        if btn:
            btn.click()
            time.sleep(1)
            log("  Cookie banner dismissed")
    except Exception:
        pass

    # Verify login
    if "Sign in" in page.content() and "Ask anything" not in page.content():
        log("[ERROR] Not logged in — please log in at grok.com first")
        context.close()
        pw.stop()
        sys.exit(1)

    snapshot_dom(page, "STATE 1: after nav to grok.com/ (fresh homepage)")
    snapshot_context(context, "STATE 1")
    try:
        page.screenshot(path=str(SHOTS_DIR / "grok_deep_1.png"))
        log("  screenshot: /tmp/grok_deep_1.png")
    except Exception as e:
        log(f"  screenshot error: {e}")

    # ─── Build REAL Hermes prompts using the actual formatter ────
    formatter = _GrokPromptFormatter()

    # Prompt 1: first turn with Hermes system + tools + user message
    first_turn_msgs = [
        {
            "role": "system",
            "content": (
                "You are Hermes Agent, an intelligent AI assistant created by Nous Research. "
                "# Tool-use enforcement\n"
                "You MUST use your tools to take action — do not describe what you would do. "
                "Conversation started: Sunday, April 20, 2026 02:30 PM\n"
                "Model: grok-auto\nProvider: xai-grok"
            ),
        },
        {"role": "user", "content": "what is my hostname"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "Run shell commands on the user's machine.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "shell command"}
                    },
                    "required": ["command"],
                },
            },
        },
    ]
    prompt1 = formatter.format(first_turn_msgs, tools)
    log(f"\n▶ Prompt 1 built (first-turn Hermes format): {len(prompt1)} chars")

    # ─── SEND 1 ──────────────────────────────────────────────────
    response1 = send_message(page, prompt1, "PROMPT 1 (first-turn with tools)", wait_timeout=90)
    time.sleep(2)  # SPA settle

    snapshot_dom(page, "STATE 2: after prompt 1 + response")
    snapshot_context(context, "STATE 2")
    try:
        page.screenshot(path=str(SHOTS_DIR / "grok_deep_2.png"))
        log("  screenshot: /tmp/grok_deep_2.png")
    except Exception as e:
        log(f"  screenshot error: {e}")

    # ─── Build continuation prompt (simulates tool result flowing back) ──
    continuation_msgs = [
        {
            "role": "system",
            "content": (
                "You are Hermes Agent, an intelligent AI assistant created by Nous Research. "
                "# Tool-use enforcement\n"
                "You MUST use your tools to take action — do not describe what you would do. "
                "Conversation started: Sunday, April 20, 2026 02:30 PM\n"
                "Model: grok-auto\nProvider: xai-grok"
            ),
        },
        {"role": "user", "content": "what is my hostname"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": '{"command": "hostname"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "terminal", "content": "my-machine"},
    ]
    prompt2 = formatter.format(continuation_msgs, tools)
    log(f"\n▶ Prompt 2 built (continuation with tool result): {len(prompt2)} chars")

    # ─── EXPERIMENTAL: reproduce _set_mode() behavior inline ─────────
    # If _set_mode is the culprit, calling it here will cause prompt 2
    # to create a NEW chat instead of appending to f4f4a3de-...
    log("\n▶ ⚗️  EXPERIMENTAL: calling _set_mode_inline BEFORE prompt 2")
    log("   Target: 'grok-heavy' (mirrors the previously-failing test config)")

    target_mode = "grok-heavy"
    target_label = _MODE_LABELS.get(target_mode)
    log(f"   Target label: {target_label!r}")

    try:
        # This is the EXACT logic from _BrowserThread._set_mode
        triggers = page.query_selector_all('button[aria-haspopup="menu"]')
        log(f"   Found {len(triggers)} button[aria-haspopup='menu'] candidates")
        trigger = None
        current_label = None
        for btn in triggers:
            try:
                txt = (btn.inner_text() or "").strip()
                if txt in _MODE_LABELS.values():
                    trigger = btn
                    current_label = txt
                    break
            except Exception:
                continue

        if not trigger:
            log("   [WARN] Mode dropdown trigger not found — simulating warning path")
        elif current_label == target_label:
            log(f"   Mode '{target_label}' already selected — no click (matches _set_mode)")
        else:
            log(f"   Current label: {current_label!r}, clicking trigger to open dropdown...")
            trigger.click()
            time.sleep(0.5)
            option = page.query_selector(f'text="{target_label}"')
            if option:
                log(f"   Clicking option '{target_label}'...")
                option.click()
                time.sleep(0.5)
                log(f"   ⚡ Mode switched: {current_label} → {target_label}")
            else:
                log(f"   [WARN] Option '{target_label}' not found in dropdown")
                try:
                    trigger.click()  # close
                except Exception:
                    pass
    except Exception as e:
        log(f"   [WARN] Mode selection failed: {e}")

    time.sleep(1)

    # Take a screenshot IMMEDIATELY after mode switch to see state
    try:
        page.screenshot(path=str(SHOTS_DIR / "grok_deep_after_mode_switch.png"))
        log("   screenshot: /tmp/grok_deep_after_mode_switch.png")
    except Exception as e:
        log(f"   screenshot error: {e}")

    # Before send 2 — snapshot to see what editor we're about to hit
    snapshot_dom(page, "STATE 3: before prompt 2 send (AFTER _set_mode_inline, BEFORE editor click)")

    # ─── SEND 2 ──────────────────────────────────────────────────
    response2 = send_message(page, prompt2, "PROMPT 2 (continuation)", wait_timeout=60)
    time.sleep(2)

    snapshot_dom(page, "STATE 4: after prompt 2 + response")
    snapshot_context(context, "STATE 4")
    try:
        page.screenshot(path=str(SHOTS_DIR / "grok_deep_4.png"))
        log("  screenshot: /tmp/grok_deep_4.png")
    except Exception as e:
        log(f"  screenshot error: {e}")

    # ─── SEND 3: new user turn (simulates 2nd user prompt) ───────
    # This is the key test. In real Hermes, after user prompt 1 returned a
    # final answer, user types prompt 2. Hermes builds a continuation with
    # the full history (prev user msg + prev assistant final answer + new user msg).
    # Does this still stay in the same chat?
    turn3_msgs = continuation_msgs + [
        {"role": "assistant", "content": "Your hostname is my-machine."},
        {"role": "user", "content": "now show me how much free disk space i have"},
    ]
    prompt3 = formatter.format(turn3_msgs, tools)
    log(f"\n▶ Prompt 3 built (new user turn 2 after assistant finalized): {len(prompt3)} chars")

    snapshot_dom(page, "STATE 5: before prompt 3 send")
    response3 = send_message(page, prompt3, "PROMPT 3 (second user turn)", wait_timeout=90)
    time.sleep(2)

    snapshot_dom(page, "STATE 6: after prompt 3 + response")
    snapshot_context(context, "STATE 6")
    try:
        page.screenshot(path=str(SHOTS_DIR / "grok_deep_5.png"))
        log("  screenshot: /tmp/grok_deep_5.png")
    except Exception as e:
        log(f"  screenshot error: {e}")

    # ─── SEND 4: tool result for prompt 3 (simulates 2nd tool round trip) ───
    turn4_msgs = turn3_msgs + [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_2", "type": "function",
             "function": {"name": "terminal", "arguments": '{"command": "df -h /"}'}}]},
        {"role": "tool", "tool_call_id": "call_2", "name": "terminal",
         "content": "Filesystem  Size  Used Avail Use%\n/dev/nvme0n1p7  92G  39G  49G  45%  /"},
    ]
    prompt4 = formatter.format(turn4_msgs, tools)
    log(f"\n▶ Prompt 4 built (2nd tool result continuation): {len(prompt4)} chars")

    snapshot_dom(page, "STATE 7: before prompt 4 send")
    response4 = send_message(page, prompt4, "PROMPT 4 (2nd tool result)", wait_timeout=60)
    time.sleep(2)

    snapshot_dom(page, "STATE 8: after prompt 4 + response")
    snapshot_context(context, "STATE 8")
    try:
        page.screenshot(path=str(SHOTS_DIR / "grok_deep_6.png"))
        log("  screenshot: /tmp/grok_deep_6.png")
    except Exception as e:
        log(f"  screenshot error: {e}")

    # ─── ANALYSIS HINTS ──────────────────────────────────────────
    log("\n═══ ANALYSIS HINTS ═══")
    log("Compare STATE 2 vs STATE 4:")
    log("  - URL same? (same /c/<id> = same chat)")
    log("  - .message-bubble count GROWING (same chat) or RESET to 2 (new chat)?")
    log("  - context.pages count — stayed at 1 or grew?")
    log("")
    log("Look at STATE 3 PM[0]:")
    log("  - Parent = <main> or [role=main] → good, typing in chat compose")
    log("  - Parent = <nav>/<aside> or sidebar-related → BUG: typing in sidebar")
    log("  - Parent = <body> or <div> with unclear container → needs network log to confirm")
    log("")
    log("Network log: look for POST requests after prompt 2 — does the URL contain:")
    log("  - '/c/<id>/messages' or '/conversations/<id>' → APPEND to same chat (good)")
    log("  - '/new' or '/create' or empty chat id → NEW chat (bug)")

    # Save logs
    TRACE_LOG.write_text("\n".join(trace_lines))
    NETWORK_LOG.write_text("\n".join(network_lines))
    log(f"\n✓ Trace:     {TRACE_LOG}")
    log(f"✓ Network:   {NETWORK_LOG}")
    log(f"✓ Screenshots: {SHOTS_DIR}/grok_deep_*.png")

    log("\nBrowser stays open 30s for manual inspection. Ctrl+C to exit sooner.")
    try:
        time.sleep(30)
    except KeyboardInterrupt:
        log("Early exit requested.")

    try:
        context.close()
    except Exception:
        pass
    try:
        pw.stop()
    except Exception:
        pass

    # Final save (in case any trace lines added after last write)
    TRACE_LOG.write_text("\n".join(trace_lines))
    NETWORK_LOG.write_text("\n".join(network_lines))


if __name__ == "__main__":
    main()
