"""Verification test for Layer 9 fix: lifecycle-aware multi-send.

Simulates Hermes' actual request lifecycle (from run_agent.py:5456-5463):
  for each send:
    client = _create_request_openai_client()  # PatchrightGrokClient(...)
    response = client.chat.completions.create(...)
    client.close()  # ← was killing browser thread, now no-op

Verifies that all 4 sends land on the SAME grok.com chat URL, proving the
Layer 9 fix prevents the browser thread from being torn down between requests.

Expected (with fix applied):
  - All 4 sends → same chat ID in URL
  - context has ONE browser thread throughout
  - grok.com sidebar grows by exactly ONE chat (not 4)

Without the fix (the bug we fixed):
  - Each close() killed the browser thread
  - Next create() spawned new browser → grok.com/ homepage → new chat
  - 4 sends = 4 different chat IDs = 4 sidebar entries

Run:
  cd ~/Projects/hermes-agent && python3 tests/grok/debug_client_lifecycle.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.patchright_client import PatchrightGrokClient, _get_thread

LOG_PATH = Path("/tmp/grok_client_lifecycle.log")
log_lines = []


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line)
    log_lines.append(line)


def chat_id(url: str) -> str:
    """Extract chat ID (between /c/ and ?) from a grok.com URL."""
    if "/c/" not in url:
        return "<no-chat-id>"
    start = url.find("/c/") + 3
    end = url.find("?", start)
    if end == -1:
        end = len(url)
    return url[start:end]


def main():
    log("═══ LAYER 9 VERIFICATION: LIFECYCLE-AWARE MULTI-SEND ═══")
    log("Mirrors Hermes' per-request create/close cycle (run_agent.py:5456-5463).\n")

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

    system_msg = {
        "role": "system",
        "content": "You are Hermes Agent. # Tool-use enforcement\nYou MUST use your tools.",
    }

    urls_seen = []
    thread_ids_seen = []

    # Build full message history progressively (like Hermes does)
    full_history = [system_msg]

    # ─── Simulate 4 request/close cycles ─────────────────────────
    test_prompts = [
        # (role, content, tool_call?)
        {"role": "user", "content": "what is my hostname"},
        # After response from send 1, we simulate the tool round trip:
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
        # Then prompt 2:
        {"role": "assistant", "content": "Your hostname is my-machine."},
        {"role": "user", "content": "now show me how much free disk space i have"},
    ]

    # SEND 1 through SEND 4
    sends = [
        # send_num, description, messages_to_use (subset of full_history)
        (1, "first user msg", [system_msg, {"role": "user", "content": "what is my hostname"}]),
        (
            2,
            "tool result continuation",
            [
                system_msg,
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
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "terminal",
                    "content": "my-machine",
                },
            ],
        ),
        (
            3,
            "second user msg",
            [
                system_msg,
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
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "terminal",
                    "content": "my-machine",
                },
                {"role": "assistant", "content": "Your hostname is my-machine."},
                {"role": "user", "content": "now show me how much free disk space i have"},
            ],
        ),
        (
            4,
            "2nd tool result",
            [
                system_msg,
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
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "terminal",
                    "content": "my-machine",
                },
                {"role": "assistant", "content": "Your hostname is my-machine."},
                {"role": "user", "content": "now show me how much free disk space i have"},
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
            ],
        ),
    ]

    for send_num, desc, msgs in sends:
        log(f"\n▶ SEND {send_num}: {desc}")
        log(f"  [lifecycle] Creating new PatchrightGrokClient (mirrors Hermes per-request)...")

        # CREATE new client (what Hermes does every request)
        client = PatchrightGrokClient(model="grok-auto")

        # Track which browser thread we're on (python id of the thread object)
        bt = _get_thread()
        thread_id = id(bt)
        thread_ids_seen.append(thread_id)
        log(f"  [lifecycle] Browser thread id={thread_id} alive={bt.is_alive()}")

        # SEND via client
        try:
            response = client.chat.completions.create(
                messages=msgs, tools=tools, model="grok-auto"
            )
            url_after = bt._page.url
            urls_seen.append(url_after)
            log(f"  URL after send: {url_after}")
            log(f"  Chat ID: {chat_id(url_after)}")
            content = response.choices[0].message.content or ""
            tool_calls = response.choices[0].message.tool_calls or []
            if tool_calls:
                tc = tool_calls[0]
                log(f"  Response: tool_call {tc.function.name}({tc.function.arguments[:60]})")
            else:
                log(f"  Response: {content[:100]!r}")
        except Exception as e:
            log(f"  [ERROR] send {send_num} failed: {e}")
            urls_seen.append(f"<error: {e}>")

        # CLOSE (what Hermes does after every request — Layer 9: now no-op)
        log(f"  [lifecycle] Calling client.close() (Layer 9 fix makes this a no-op)...")
        client.close()
        log(f"  [lifecycle] After close: browser thread still alive? {bt.is_alive()}")

        time.sleep(1)

    # ─── ANALYSIS ──────────────────────────────────────────────
    log("\n═══ VERDICT ═══")
    log(f"Browser thread IDs across 4 sends: {thread_ids_seen}")
    unique_threads = set(thread_ids_seen)
    if len(unique_threads) == 1:
        log(f"  ✅ SAME thread object across all sends (singleton preserved)")
    else:
        log(f"  ❌ {len(unique_threads)} different threads — singleton got recreated")

    log(f"\nURLs per send:")
    for i, url in enumerate(urls_seen, 1):
        log(f"  send {i}: {url}")

    ids = [chat_id(u) for u in urls_seen]
    unique_ids = set(i for i in ids if not i.startswith("<"))
    log(f"\nUnique chat IDs: {len(unique_ids)}")
    if len(unique_ids) == 1:
        log(f"  ✅ LAYER 9 FIX WORKING — all sends hit same chat {list(unique_ids)[0]}")
    elif len(unique_ids) == len(sends):
        log(f"  ❌ BUG PRESENT — {len(sends)} different chats created (one per send)")
        log(f"     This means close() is still killing the browser thread.")
    else:
        log(f"  ⚠️  Partial — {len(unique_ids)} chats for {len(sends)} sends")

    log(f"\n✓ Log saved: {LOG_PATH}")
    LOG_PATH.write_text("\n".join(log_lines))

    # Process will exit; daemon browser thread dies automatically
    log("\nExiting (daemon browser thread will auto-cleanup)...")


if __name__ == "__main__":
    main()
