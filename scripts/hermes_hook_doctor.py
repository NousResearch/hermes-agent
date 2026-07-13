#!/usr/bin/env python3
"""Self-test the three owner-facing response gates."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


HOME = Path.home()
CLAUDE_HOOKS = HOME / ".claude" / "hooks"
CODEX_HOOKS = HOME / ".codex" / "hooks"


def call(path: Path, payload: dict) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(path)],
        input=json.dumps(payload, ensure_ascii=False),
        text=True,
        capture_output=True,
    )


def transcript(path: Path, tool_name: str, final: str, command: str = "") -> None:
    rows = [
        {"type": "user", "message": {"content": [{"type": "text", "text": "- แก้ระบบ\n- เพิ่ม test"}]}},
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": tool_name, "input": {"file_path": "/tmp/app.py", "command": command}}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": final}]}},
    ]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> int:
    results = []
    thai = call(CLAUDE_HOOKS / "validate-thai-language.py", {"last_assistant_message": "leverage utilize synergy seamless robust scalable optimize"})
    results.append({"gate": "plain_language", "ok": thai.returncode == 2, "exit": thai.returncode})

    with tempfile.TemporaryDirectory() as tmp:
        review_file = Path(tmp) / "review.jsonl"
        transcript(review_file, "apply_patch", "เสร็จแล้วครับ")
        review = call(CODEX_HOOKS / "enforce-codex-review.py", {"transcript_path": str(review_file), "last_assistant_message": "เสร็จแล้วครับ"})
        results.append({"gate": "independent_review", "ok": review.returncode == 2, "exit": review.returncode})

        evidence_file = Path(tmp) / "evidence.jsonl"
        transcript(evidence_file, "apply_patch", "เสร็จครบ 100% แล้วครับ")
        evidence = call(CLAUDE_HOOKS / "enforce-prompt-evidence.py", {"transcript_path": str(evidence_file), "last_assistant_message": "เสร็จครบ 100% แล้วครับ"})
        results.append({"gate": "prompt_evidence", "ok": evidence.returncode == 2, "exit": evidence.returncode})

    ok = all(row["ok"] for row in results)
    print(json.dumps({"ok": ok, "gates": results}, ensure_ascii=False))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
