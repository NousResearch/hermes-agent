#!/usr/bin/env python3
"""Stop hook ตรวจกฎภาษาไทย (global-ai-behavior) — v4 blocking 2026-07-12

โหมดบล็อกจริง:
  - ถ้าเจอการละเมิด จะ exit 2 และบันทึกลง ~/.claude/hooks-violations.log
  - validate-all-stop จัด hook นี้อยู่ในกลุ่ม selective block

อ่าน stdin JSON จาก Claude Code ตรวจ last_assistant_message
บันทึกการละเมิดเป็น JSONL บรรทัดต่อบรรทัด เพื่ออ่านย้อนหลังและนับได้

ปิดฉุกเฉินเฉพาะเจ้าของอนุมัติ: export VALIDATE_THAI_DISABLED=owner-approved
"""
import json
import os
import re
import sys
from datetime import datetime

VIOLATION_LOG = os.path.expanduser("~/.claude/hooks-violations.log")

BANNED_WORDS = (
    "leverage|utilize|synergy|seamless|robust|scalable|optimize"
    "|deprecate|paradigm|idempotent|refactor|dispatch|orchestrate"
)

WHITELIST = frozenset(
    w.lower()
    for w in (
        "session", "sessions",
        "watchdog", "watchdogs",
        "container", "containers",
        "silence", "silences", "silenced",
        "cron", "crons", "crontab", "cronjob",
        "pm2", "docker", "nginx", "apache",
        "restart", "reload", "reboot",
        "deploy", "deployment", "rollback",
        "status", "backup", "backups", "restore",
        "log", "logs", "config", "configs",
        "port", "ports", "host", "hostname", "domain",
        "service", "services", "alert", "alerts",
        "grafana", "prometheus", "alertmanager", "loki", "tempo",
        "probe", "probes", "kernel", "systemd",
        "ssh", "git", "pid",
        "exit", "exited", "running", "stopped",
        "verified", "tested", "done", "ready", "working",
        "server", "servers", "browser", "file", "files", "folder", "folders",
        "code", "bug", "test", "install", "save", "click",
        "login", "password", "email", "url", "api", "pdf", "csv", "json",
        "user", "users", "admin", "root", "path", "node", "npm", "pnpm",
        "build", "builds", "error", "errors", "warning", "warnings",
        "hook", "hooks", "trigger", "triggers",
        "tier", "tiers",
        "min", "mins", "sec", "secs", "hr", "hrs",
        "up", "down",
        "vm", "lxc", "proxmox", "ubuntu", "debian", "linux",
        "exporter", "exporters", "metric", "metrics",
    )
)


def clean_noise(text: str) -> str:
    """ลบ code block, inline code, url, path, filename, identifier"""
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[~/][A-Za-z0-9._/\-]+", " ", text)
    text = re.sub(r"\b[A-Za-z0-9_\-]+\.[A-Za-z0-9]{1,5}\b", " ", text)
    text = re.sub(r"\b[A-Za-z]+[_\-][A-Za-z0-9_\-]+\b", " ", text)
    return text


def strip_whitelist(text: str) -> str:
    """แทนคำใน WHITELIST ด้วย space เพื่อไม่นับเป็น chain"""
    pattern = r"\b(" + "|".join(re.escape(w) for w in WHITELIST) + r")\b"
    return re.sub(pattern, " ", text, flags=re.I)


def is_technical_report(raw: str) -> bool:
    """รายงานเทคนิค = มี fenced code block ≥ 1 หรือ inline code ≥ 5"""
    fenced = len(re.findall(r"```[\s\S]*?```", raw))
    inline = len(re.findall(r"`[^`]+`", raw))
    return fenced >= 1 or inline >= 5


def find_violations(msg: str):
    violations = []
    # table size check ถูกถอดออก 2026-04-24
    # เหตุผล: Stop hook block = ตอบซ้ำเสมอ (ข้อความเก่าสตรีมไปแล้วดึงกลับไม่ได้)
    # ตารางใหญ่ false positive บ่อยสำหรับรายงานข้อมูลจริง
    # CLAUDE.md ยังบอกกฎตาราง ≤ 8/12 บรรทัด ให้ AI ทำตามเอง

    clean = clean_noise(msg)

    # Chain check: ข้ามทั้งหมดถ้าเป็นรายงานเทคนิค
    # ปกติใช้ clean_nowl (ลบคำ whitelist ออกก่อน) เพื่อไม่เจอ false positive
    clean_nowl = strip_whitelist(clean)
    chains = re.findall(r"[A-Za-z]{2,}(?:\s+[A-Za-z]{2,}){3,}", clean_nowl)
    if chains:
        sample = chains[0][:80]
        violations.append(
            f"คำอังกฤษติดกันเกิน 3 คำ ตัวอย่าง {sample!r} "
            "ต้องแทรกคำไทยหรือวงเล็บแปล"
        )

    # คำต้องห้าม: เข้มเสมอทุกโหมด
    found = sorted({m.group().lower() for m in re.finditer(BANNED_WORDS, clean, re.I)})
    if found:
        violations.append(
            f"พบคำต้องห้าม {', '.join(found)} ต้องใช้คำไทยแทน"
        )

    return violations


def main() -> int:
    if os.environ.get("VALIDATE_THAI_DISABLED") == "owner-approved":
        return 0

    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0

    if data.get("stop_hook_active"):
        return 0

    msg = data.get("last_assistant_message") or data.get("response") or data.get("message") or ""
    if not msg:
        return 0

    violations = find_violations(msg)
    if not violations:
        return 0

    # บันทึกก่อนบล็อก เพื่อให้ตรวจย้อนหลังได้
    try:
        entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "hook": "validate-thai-language",
            "violations": violations,
            "msg_preview": msg[:200].replace("\n", " "),
        }
        with open(VIOLATION_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass
    print(
        "BLOCK · คำตอบยังไม่เป็นภาษาคนตามกฎเจ้าของ ·\n"
        + "\n".join(f"  - {item}" for item in violations)
        + "\nแก้ข้อความแล้วส่งใหม่ โดยใช้ภาษาไทยและแปลศัพท์ครั้งแรก",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
