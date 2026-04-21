"""Reproduction test: use PatchrightGrokClient (singleton _BrowserThread)
exactly like real Hermes does. Attempt to reproduce the 4-new-chats bug.

This hits the code path:
  PatchrightGrokClient.chat.completions.create()
    → _PatchrightCompletionsAdapter.create()
      → _send_to_browser("send_message", ...)
        → _BrowserThread.run() cmd loop
          → _send_message(msg, mode, timeout)
            → _ensure_grok() + _set_mode() + click editor + type + Enter

If the bug reproduces: 4 sends → 4 different chat URLs observed (via response.model
or by inspecting grok.com sidebar afterward).

If it does NOT reproduce: the bug is elsewhere (maybe Hermes' outer flow calls close(),
rebuilds the client, or does something we haven't traced yet).

Run:
  cd ~/Projects/hermes-agent && python3 tests/grok/debug_client_flow.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.patchright_client import PatchrightGrokClient, _get_thread

LOG_PATH = Path("/tmp/grok_client_flow.log")
log_lines = []


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    log_lines.append(line)


def main():
    log("═══ CLIENT-FLOW REPRODUCTION TEST ═══")
    log("Uses PatchrightGrokClient exactly like real Hermes does.")

    client = PatchrightGrokClient(model="grok-auto")
    log("[OK] Client instantiated (warmup in progress)")

    time.sleep(3)  # let warmup complete

    # Get reference to the browser thread's page for URL inspection
    # (we can read .url from any thread, just not call sync methods)
    bt = _get_thread()
    initial_url = "<unknown>"
    try:
        initial_url = bt._page.url
    except Exception as e:
        log(f"  [WARN] Could not read initial URL: {e}")
    log(f"  Initial page URL: {initial_url}")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "Run shell commands.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
    ]

    # Simulate Hermes' exact message flow for 2 user prompts with tool round trips
    system_msg = {
        "role": "system",
        "content": (
            "You are Hermes Agent. "
            "# Tool-use enforcement\n"
            "You MUST use your tools to take action."
        ),
    }

    # ─── SEND 1: first user prompt ───────────────────────────────
    log("\n▶ SEND 1: first user message (expect: /conversations/new)")
    msgs1 = [system_msg, {"role": "user", "content": "what is my hostname"}]
    r1 = client.chat.completions.create(messages=msgs1, tools=tools, model="grok-auto")
    url_after_1 = bt._page.url
    log(f"  URL after send 1: {url_after_1}")
    log(f"  Response: {r1.choices[0].message.content[:120] if r1.choices[0].message.content else '(tool call)'}!r")
    if r1.choices[0].message.tool_calls:
        tc = r1.choices[0].message.tool_calls[0]
        log(f"  Tool call: {tc.function.name}({tc.function.arguments[:80]})")

    time.sleep(1)

    # ─── SEND 2: tool result continuation ───────────────────────
    log("\n▶ SEND 2: tool result (expect: /conversations/<same-id>/responses)")
    msgs2 = msgs1 + [
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
    r2 = client.chat.completions.create(messages=msgs2, tools=tools, model="grok-auto")
    url_after_2 = bt._page.url
    log(f"  URL after send 2: {url_after_2}")
    log(f"  Response: {(r2.choices[0].message.content or '')[:120]!r}")

    time.sleep(1)

    # ─── SEND 3: second user prompt ───────────────────────────────
    log("\n▶ SEND 3: SECOND user message (expect: append to same chat)")
    msgs3 = msgs2 + [
        {"role": "assistant", "content": "Your hostname is my-machine."},
        {"role": "user", "content": "now show me how much free disk space i have"},
    ]
    r3 = client.chat.completions.create(messages=msgs3, tools=tools, model="grok-auto")
    url_after_3 = bt._page.url
    log(f"  URL after send 3: {url_after_3}")
    log(f"  Response: {(r3.choices[0].message.content or '')[:120]!r}")
    if r3.choices[0].message.tool_calls:
        tc = r3.choices[0].message.tool_calls[0]
        log(f"  Tool call: {tc.function.name}({tc.function.arguments[:80]})")

    time.sleep(1)

    # ─── SEND 4: second tool result ───────────────────────────────
    log("\n▶ SEND 4: 2nd tool result (expect: append to same chat)")
    msgs4 = msgs3 + [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": '{"command": "df -h /"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "name": "terminal",
            "content": "Filesystem  Size  Used Avail Use%\n/dev/nvme0n1p7  92G  39G  49G  45%  /",
        },
    ]
    r4 = client.chat.completions.create(messages=msgs4, tools=tools, model="grok-auto")
    url_after_4 = bt._page.url
    log(f"  URL after send 4: {url_after_4}")
    log(f"  Response: {(r4.choices[0].message.content or '')[:120]!r}")

    # ─── ANALYSIS ──────────────────────────────────────────────
    log("\n═══ URL COMPARISON (all 4 sends) ═══")
    log(f"  after send 1: {url_after_1}")
    log(f"  after send 2: {url_after_2}")
    log(f"  after send 3: {url_after_3}")
    log(f"  after send 4: {url_after_4}")
    log("")

    # Extract chat IDs (between /c/ and ?)
    def chat_id(url):
        if "/c/" not in url:
            return None
        start = url.find("/c/") + 3
        end = url.find("?", start)
        if end == -1:
            end = len(url)
        return url[start:end]

    ids = [chat_id(url_after_1), chat_id(url_after_2), chat_id(url_after_3), chat_id(url_after_4)]
    unique_ids = set(filter(None, ids))

    log(f"Chat IDs per send: {ids}")
    log(f"Unique chat IDs: {len(unique_ids)}")
    if len(unique_ids) == 1:
        log("✅ SAME CHAT — bug did NOT reproduce here.")
        log("   The bug must be somewhere else in Hermes' outer flow.")
    else:
        log(f"❌ {len(unique_ids)} DIFFERENT CHAT IDs — BUG REPRODUCED!")
        log("   Each send created a new chat — confirms the bug is in")
        log("   _BrowserThread/_send_message path when called via real client.")

    log(f"\n✓ Log saved: {LOG_PATH}")
    LOG_PATH.write_text("\n".join(log_lines))

    # Clean shutdown
    log("\nClosing client...")
    try:
        client.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
