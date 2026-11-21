"""Copy a live reply through real Ink mouse events and observe its OSC 52 payload.

No desktop clipboard is needed: the PTY records the exact terminal clipboard write.
The real local provider keeps streaming while the receipt appears and expires.
"""

from __future__ import annotations

import base64
import re
import shutil
from pathlib import Path

import pytest

from tests.e2e.core.terminal._pty import REPO_ROOT, PtyHermes, canon
from tests.fakes.fake_llm_provider import FakeLLMServer, Text
from tests.waiting import wait_until

pytestmark = [
    pytest.mark.platforms("linux"),
    pytest.mark.live_system_guard_bypass,
]


def test_copy_receipt_and_clipboard_during_stream(tmp_path: Path) -> None:
    if shutil.which("node") is None or not (REPO_ROOT / "ui-tui/dist/entry.js").is_file():
        pytest.fail("Build the TUI and install Node before running clipboard PTY coverage")

    reply = "COPYME " + "streaming " * 240 + "STREAM-END"
    with FakeLLMServer([Text(reply, chunk_chars=8, delay_per_chunk=0.05)],
                       aux=lambda _req: Text("Clipboard test")) as llm:
        term = PtyHermes(tmp_path, ["--tui", "--yolo"], llm, rows=60, cols=160)
        try:
            term.wait_ready(timeout=45)
            term.submit("stream a reply for clipboard testing")
            term.wait_for_text("COPYME streaming", timeout=30)
            with term._lock:
                rows = term.screen.display()
                row, line = next((i, line) for i, line in enumerate(rows) if "COPYME" in line)
                col = line.index("COPYME")
            # SGR coordinates are one-based; selection endpoints are inclusive.
            term.write(f"\x1b[<0;{col + 1};{row + 1}M")
            term.write(f"\x1b[<32;{col + 6};{row + 1}M")
            term.write(f"\x1b[<0;{col + 6};{row + 1}m")
            term.wait_for_text("copied 6 characters", timeout=10)

            with term._lock:
                payloads = re.findall(rb"\x1b\]52;c;([A-Za-z0-9+/=]+)(?:\x07|\x1b\\)", bytes(term.raw))
            assert b"COPYME" in [base64.b64decode(payload) for payload in payloads]
            receipt_line = next(line for line in term.lines() if "copied 6 characters" in line)
            assert re.search(r"sessions?\s*│\s*copied 6 characters", receipt_line), term.dump()
            assert "STREAM-END" not in term.text(), "copy happened only after the stream finished"
            wait_until(lambda: canon("copied 6 characters") not in term.text(), timeout=10,
                       what="copy receipt to expire while streaming")
            assert "STREAM-END" not in term.text(), "receipt expiry did not occur during streaming"

            term.resize(60, 100)
            term.wait_for_text("STREAM-END", timeout=45)
            term.wait_turns_persisted(1)
            term.wait_quiet()
            assert term.count(reply) == 1, term.dump()
            assert term.exit() == 0
            assert not term.leftover_processes()
        finally:
            term.close()
