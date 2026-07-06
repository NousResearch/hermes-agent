#!/usr/bin/env python3

import json
import re
import subprocess
import sys
from pathlib import Path


SPEAK_MARKER_RE = re.compile(r"<!--\s*speak:\s*(.*?)\s*-->")


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:
        return

    if payload.get("stop_hook_active") is True:
        return

    transcript_path = payload.get("transcript_path")
    if not isinstance(transcript_path, str) or not transcript_path:
        return

    assistant_text = last_assistant_text(Path(transcript_path))
    if not assistant_text:
        return

    matches = SPEAK_MARKER_RE.findall(assistant_text)
    if not matches:
        return

    text = " ".join(matches[-1].strip().split())[:500]
    if not text:
        return

    speak_script = Path(__file__).resolve().parent / "speak.sh"
    subprocess.Popen(
        [str(speak_script), text],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )


def last_assistant_text(transcript_path: Path) -> str:
    try:
        lines = transcript_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""

    for line in reversed(lines):
        try:
            item = json.loads(line)
        except Exception:
            continue

        if item.get("type") != "assistant":
            continue

        message = item.get("message")
        if not isinstance(message, dict):
            return ""

        content = message.get("content")
        if not isinstance(content, list):
            return ""

        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "text":
                continue
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)

    return ""


try:
    main()
except Exception:
    pass

sys.exit(0)
