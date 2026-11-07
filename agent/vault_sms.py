"""Opt-in SMS source for ``browser_vault`` ``enter_code``.

Without an authenticator seed, ``enter_code`` asks the user for the code their phone shows. On a Mac that
receives their texts (Text Message Forwarding puts them in ``~/Library/Messages/chat.db``) the code can be read
instead, so nobody is asked. Off by default; enable with ``browser_vault.sms_code_source`` in config.yaml:

- ``messages``: read the newest inbound one-time code from this Mac's Messages archive (macOS; needs Full
  Disk Access for the Hermes process).
- anything else: a shell command run with ``HERMES_SMS_SINCE`` (epoch seconds) set; it prints the code on
  stdout and exits. Use it to fetch from another Mac, e.g. ``ssh hub python -m agent.vault_sms --since
  $HERMES_SMS_SINCE``.

The code is returned to the caller only; it is never logged or printed except by the CLI below, whose stdout
is a pipe, not a conversation.
"""

from __future__ import annotations

import os
import re
import shlex
import sqlite3
import subprocess
import sys
import time
from typing import Optional

APPLE_EPOCH = 978307200
GRACE_SECONDS = 90  # the site texts the code a moment before enter_code is called
_CODEWORD = re.compile(r"code|verif|otp|passcode|one-time|security|confirm", re.IGNORECASE)
_CODE = re.compile(r"(?<![\d-])(?:G-)?(\d{4,8})(?!\d)")


def extract_code(text: str) -> Optional[str]:
    """The one-time code in an SMS body, or None. Requires a code-ish word so a plain number is never taken."""
    if not text or not _CODEWORD.search(text):
        return None
    match = _CODE.search(text)
    return match.group(1) if match else None


def newest_message_code(since: float, db_path: Optional[str] = None) -> Optional[str]:
    """Newest inbound one-time code received at/after ``since`` (epoch seconds), or None."""
    db_path = db_path or os.path.expanduser("~/Library/Messages/chat.db")  # macOS system DB, not HERMES_HOME
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2)
    except sqlite3.Error:
        return None
    try:
        rows = conn.execute(
            "select text from message where is_from_me = 0 and text is not null "
            "and date / 1000000000 + ? >= ? order by date desc limit 10",
            (APPLE_EPOCH, since),
        ).fetchall()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    for (text,) in rows:
        code = extract_code(text)
        if code:
            return code
    return None


def _command_code(command: str, since: float) -> Optional[str]:
    try:
        out = subprocess.run(
            shlex.split(command), capture_output=True, text=True, timeout=20, check=False,
            env={**os.environ, "HERMES_SMS_SINCE": str(int(since))},
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return out if re.fullmatch(r"\d{4,8}", out) else None


def wait_for_code(source: str, since: float, timeout: float = 90.0, poll: float = 3.0) -> Optional[str]:
    """Poll ``source`` until a code newer than ``since`` appears or ``timeout`` runs out."""
    source = (source or "").strip()
    if not source:
        return None
    deadline = time.time() + max(0.0, timeout)
    while True:
        code = newest_message_code(since) if source == "messages" else _command_code(source, since)
        if code or time.time() >= deadline:
            return code
        time.sleep(poll)


def main(argv: Optional[list] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    since = time.time() - GRACE_SECONDS
    if argv[:1] == ["--since"] and len(argv) > 1:
        since = float(argv[1])
    code = newest_message_code(since)
    if code:
        print(code)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
