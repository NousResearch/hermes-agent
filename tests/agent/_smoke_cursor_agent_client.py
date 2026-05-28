"""Live integration smoke test against the real ``cursor-agent`` CLI.

Not a pytest test — run directly. Requires ``cursor-agent`` on PATH and a
logged-in Cursor session (``cursor-agent login`` or ``CURSOR_API_KEY`` env).

Usage:
    /path/to/python tests/agent/_smoke_cursor_agent_client.py
"""

from __future__ import annotations

import json
import os
import sys
import time

# Make the repo importable when invoked directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent.cursor_agent_client import CursorAgentClient


def main() -> int:
    print("--- live smoke: CursorAgentClient ---")
    client = CursorAgentClient(mode="ask")
    try:
        info = client.whoami()
        if info:
            print(f"  logged-in: {info.get('email', '?')}")
        else:
            print("  warning: whoami returned empty (no cursor-agent or not logged in)")

        prompts = [
            ("auto", "Reply with just the single word: ALPHA"),
            ("composer-2.5-fast", "Reply with just the single word: BETA"),
        ]
        for model, prompt in prompts:
            print(f"\n>>> model={model}  prompt={prompt!r}")
            t0 = time.monotonic()
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "Be terse. Reply with exactly one word and nothing else."},
                        {"role": "user", "content": prompt},
                    ],
                )
            except Exception as exc:
                print(f"    ERROR after {time.monotonic() - t0:.2f}s: {exc}")
                return 1
            dt = time.monotonic() - t0
            text = resp.choices[0].message.content
            usage = resp.usage
            print(f"    response: {text!r}")
            print(f"    finish_reason: {resp.choices[0].finish_reason}")
            print(f"    duration: {dt:.2f}s")
            print(
                f"    usage: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, "
                f"total={usage.total_tokens}, cached={usage.prompt_tokens_details.cached_tokens}"
            )
            print(f"    model: {resp.model}, id: {resp.id}")
            if not text or len(text.strip()) > 200:
                print(f"    UNEXPECTED response length / empty — text was {text!r}")
                return 2
        print("\n--- smoke PASSED ---")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
