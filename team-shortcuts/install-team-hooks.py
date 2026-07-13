#!/usr/bin/env python3
"""Install team response gates without replacing existing AI settings."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys


HOME = Path.home()
SOURCE = Path(__file__).resolve().parent / "hooks"
HOOK_NAMES = (
    "validate-thai-language.py",
    "enforce-codex-review.py",
    "enforce-prompt-evidence.py",
    "team-stop-gates.py",
)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ไฟล์ตั้งค่า JSON เสียที่ {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"ไฟล์ตั้งค่าต้องเป็น JSON object: {path}")
    return value


def install_files(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in HOOK_NAMES:
        src = SOURCE / name
        if not src.is_file():
            raise SystemExit(f"ไม่พบไฟล์ Hook ในชุดติดตั้ง: {src}")
        dst = target / name
        if not dst.exists() or src.read_bytes() != dst.read_bytes():
            shutil.copy2(src, dst)
        dst.chmod(0o755)


def install_stop_entry(settings_path: Path, runner: Path) -> None:
    data = load_json(settings_path)
    hooks = data.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise SystemExit(f"ช่อง hooks ผิดรูปแบบใน {settings_path}")
    stop = hooks.setdefault("Stop", [])
    if not isinstance(stop, list):
        raise SystemExit(f"ช่อง hooks.Stop ผิดรูปแบบใน {settings_path}")

    existing_commands = [
        str(hook.get("command", ""))
        for entry in stop
        if isinstance(entry, dict)
        for hook in entry.get("hooks", [])
        if isinstance(hook, dict)
    ]
    has_native_bundle = (
        any("validate-all-stop.py" in command for command in existing_commands)
        and any("enforce-codex-review.py" in command for command in existing_commands)
    )

    if has_native_bundle:
        cleaned = []
        for entry in stop:
            if not isinstance(entry, dict):
                cleaned.append(entry)
                continue
            hooks_in_entry = entry.get("hooks", [])
            kept = [
                hook for hook in hooks_in_entry
                if not (isinstance(hook, dict) and "team-stop-gates.py" in str(hook.get("command", "")))
            ]
            if kept:
                updated = dict(entry)
                updated["hooks"] = kept
                cleaned.append(updated)
        hooks["Stop"] = cleaned
        stop = cleaned

    command = str(runner)
    found = False
    for entry in stop:
        if not isinstance(entry, dict):
            continue
        for hook in entry.get("hooks", []):
            if isinstance(hook, dict) and "team-stop-gates.py" in str(hook.get("command", "")):
                hook.update({"type": "command", "command": command, "timeout": 12})
                found = True
    if not found and not has_native_bundle:
        stop.append({"hooks": [{"type": "command", "command": command, "timeout": 12}]})

    settings_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = settings_path.with_suffix(settings_path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(settings_path)


def main() -> int:
    claude_hooks = HOME / ".claude" / "hooks"
    codex_hooks = HOME / ".codex" / "hooks"
    install_files(claude_hooks)
    install_files(codex_hooks)
    install_stop_entry(HOME / ".claude" / "settings.json", claude_hooks / "team-stop-gates.py")
    install_stop_entry(HOME / ".codex" / "hooks.json", codex_hooks / "team-stop-gates.py")
    print("ติดตั้ง Hook ทีมให้ Claude Code และ Codex แล้ว")
    return 0


if __name__ == "__main__":
    sys.exit(main())
