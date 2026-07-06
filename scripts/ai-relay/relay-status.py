#!/usr/bin/env python3
"""relay-status — ตรวจสดว่า AI Relay ตัวไหนพร้อมใช้ ณ ตอนนี้

อ่านทะเบียนจาก relay-call.py เท่านั้น แล้วเติมสถานะที่เปลี่ยนตามเครื่องจริง:
โปรแกรมมีไหม, กำลังพักไหม, และถ้าเจ้าของใส่ --probe ค่อยเรียกตรวจ login/quota จริง
"""
import argparse
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("relay_call", SCRIPT_DIR / "relay-call.py")
relay_call = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(relay_call)

NOT_PROBED = "ยังไม่เช็ค (ใส่ --probe เพื่อเช็คจริง)"
UNKNOWN = "ไม่ทราบ"


def bin_for(name, meta):
    """หาโปรแกรมที่ควรมีบนเครื่องจากข้อมูลทะเบียนกลาง"""
    meta = meta or {}
    configured = meta.get("bin")
    if configured:
        return str(configured)
    vendor = meta.get("vendor")
    if vendor == "anthropic":
        return "claude"
    if vendor == "local":
        return "ollama"
    return str(name)


def _login_hint(meta, fallback):
    return (meta or {}).get("login_hint") or f"ติดตั้งหรือล็อกอิน {fallback}"


def _cooldown_entry(cooldown_map, name):
    if not cooldown_map:
        return None
    return cooldown_map.get(name, cooldown_map.get("*"))


def _cooldown_state(cooldown_map, name):
    entry = _cooldown_entry(cooldown_map, name)
    if entry is None:
        return False, None, ""
    if isinstance(entry, bool):
        return entry, None, ""
    if isinstance(entry, dict):
        state = entry.get("cooldown", False)
        if state in (True, False):
            return state, entry.get("until"), entry.get("reason", "")
        return UNKNOWN, entry.get("until"), entry.get("reason", "")
    return UNKNOWN, None, "อ่านไฟล์สถานะพักไม่ได้"


def _fmt_until(until):
    if not until:
        return ""
    try:
        return datetime.fromtimestamp(float(until)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(until)


def _probe_hint(status, meta, bin_name):
    login = _login_hint(meta, bin_name)
    if status == "auth":
        return f"ยังไม่ล็อกอิน — {login}"
    if status == "quota":
        return f"โควตาเต็มหรือโดนจำกัดการใช้ — {login}"
    if status == "not_found":
        return f"ยังไม่พบโปรแกรม `{bin_name}` — {login}"
    if status == "timeout":
        return "ตรวจจริงแล้วค้างหรือหมดเวลา"
    if status == "limit_exceeded":
        return "ชนเพดานรอบงานของ Relay"
    if status == "crash":
        return f"ตรวจจริงแล้วพัง — {login}"
    return f"ตรวจจริงคืนค่า {status} — {login}"


def tool_status(name, meta, which_fn, cooldown_map, probe_result=None):
    """คำนวณสถานะต่อหนึ่ง AI แบบไม่อ่านไฟล์เอง เพื่อให้ทดสอบง่าย"""
    meta = meta or {}
    bin_name = bin_for(name, meta)
    installed = which_fn(bin_name) is not None
    cooldown, until, cd_reason = _cooldown_state(cooldown_map, name)
    live = probe_result if probe_result is not None else NOT_PROBED

    hint = ""
    if not installed:
        hint = f"ยังไม่พบโปรแกรม `{bin_name}` — {_login_hint(meta, bin_name)}"
    if cooldown is True:
        end = _fmt_until(until)
        hint = f"กำลังพัก" + (f"ถึง {end}" if end else "")
    elif cooldown == UNKNOWN:
        hint = cd_reason or "อ่านสถานะพักไม่ได้"
    if probe_result is not None and probe_result != "ok":
        hint = _probe_hint(probe_result, meta, bin_name)

    ready = installed and cooldown is False and (probe_result is None or probe_result == "ok")
    return {
        "name": name,
        "vendor": meta.get("vendor") or "",
        "bin": bin_name,
        "installed": installed,
        "cooldown": cooldown,
        "cooldown_until": until,
        "live": live,
        "hint": "" if ready else hint,
        "ready": ready,
    }


def _as_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _normalize_cooldown_value(value, now):
    """รองรับทั้ง {tool: epoch} และ {tool: {until: epoch}}; แบบอื่นคืนไม่ทราบ"""
    until = None
    if isinstance(value, (int, float, str)):
        until = _as_float(value)
    elif isinstance(value, dict):
        until = _as_float(value.get("until"))
    if until is None:
        return {"cooldown": UNKNOWN, "reason": "โครงไฟล์สถานะพักไม่ตรงที่รองรับ"}
    return {"cooldown": until > now, "until": until}


def read_cooldown_map(cwd, now=None):
    cwd = pathlib.Path(cwd)
    path = cwd / ".hermes" / "ai-relay" / ".cooldown.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"*": {"cooldown": UNKNOWN, "reason": "อ่านไฟล์สถานะพักไม่ได้"}}
    if not isinstance(data, dict):
        return {"*": {"cooldown": UNKNOWN, "reason": "โครงไฟล์สถานะพักไม่ใช่ object"}}
    now = float(now if now is not None else datetime.now().timestamp())
    return {str(tool): _normalize_cooldown_value(value, now) for tool, value in data.items()}


def _last_json_line(text):
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except Exception:
            continue
    return {}


def run_probe(cwd, name):
    """เรียก relay-call ด้วย prompt เล็กมาก เฉพาะเมื่อเจ้าของใส่ --probe"""
    prompt_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
            f.write("reply OK")
            prompt_path = pathlib.Path(f.name)
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "relay-call.py"),
            "--tool",
            name,
            "--task-id",
            "STATUS-PROBE",
            "--prompt-file",
            str(prompt_path),
            "--cwd",
            str(cwd),
        ]
        # timeout ชั้นนอกกันค้าง (relay-call มี timeout ในตัวแล้ว แต่ probe เป็นงานจิ๋ว 180 วิพอ)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        payload = _last_json_line(proc.stdout)
        return payload.get("status") or "crash"
    except subprocess.TimeoutExpired:
        return "timeout"
    except Exception:
        return "crash"
    finally:
        if prompt_path:
            try:
                prompt_path.unlink()
            except Exception:
                pass


def collect_status(cwd, probe=False, which_fn=shutil.which):
    cwd = pathlib.Path(cwd).expanduser().resolve()
    reg = relay_call.load_registry(cwd)
    enabled = relay_call.registry_enabled(reg)
    cooldown_map = read_cooldown_map(cwd)
    rows = []
    for name, meta in enabled.items():
        probe_result = run_probe(cwd, name) if probe else None
        rows.append(tool_status(name, meta, which_fn, cooldown_map, probe_result=probe_result))
    return rows


def _cooldown_label(row):
    if row["cooldown"] is True:
        end = _fmt_until(row.get("cooldown_until"))
        return "กำลังพัก" + (f"ถึง {end}" if end else "")
    if row["cooldown"] is False:
        return "ไม่พัก"
    return UNKNOWN


def _installed_label(row):
    return "มี" if row["installed"] else "ไม่มี"


def _live_label(row, probed):
    if not probed:
        return row["live"]
    if row["live"] == "ok":
        return "พร้อม"
    return f"ไม่พร้อม ({row['live']})"


def _summary(rows):
    ready = [r["name"] for r in rows if r["ready"]]
    need = [f"{r['name']} ({r['hint'] or 'ไม่พร้อม'})" for r in rows if not r["ready"]]
    return (
        "พร้อมใช้ตอนนี้: " + (", ".join(ready) if ready else "ไม่มี")
        + " · ต้องแก้: " + (", ".join(need) if need else "ไม่มี")
    )


def print_table(cwd, rows, probed):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("═══ AI Relay · สถานะพร้อมใช้ ณ ตอนนี้ ═══")
    print(f"เวลา: {now}")
    print(f"โปรเจกต์: {cwd}")
    if probed:
        print("คำเตือน: --probe จะเรียก AI จริงเพื่อเช็ค login/quota และอาจมีค่าใช้จ่าย")
    print()
    print("| ชื่อ | ค่าย | ติดตั้ง | สถานะพัก(cooldown) | พร้อมใช้จริง(ถ้า --probe) | login_hint(ถ้าไม่พร้อม) |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        hint = r["hint"] or "—"
        print(
            f"| {r['name']} | {r['vendor']} | {_installed_label(r)} | "
            f"{_cooldown_label(r)} | {_live_label(r, probed)} | {hint} |"
        )
    print()
    print(_summary(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cwd", default=os.getcwd(), help="โฟลเดอร์โปรเจกต์ ค่าเริ่มต้นคือโฟลเดอร์ปัจจุบัน")
    ap.add_argument("--probe", action="store_true", help="เช็ค login/quota จริงผ่าน relay-call มีโอกาสเสียค่าใช้จ่าย")
    ap.add_argument("--json", action="store_true", help="คืน JSON แทนตารางภาษาคน")
    args = ap.parse_args()

    cwd = pathlib.Path(args.cwd).expanduser().resolve()
    rows = collect_status(cwd, probe=args.probe)
    if args.json:
        out = {
            "root": str(cwd),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "probe": bool(args.probe),
            "probe_warning": "มีโอกาสเสียค่าใช้จ่าย" if args.probe else "",
            "tools": rows,
            "summary_human": _summary(rows),
        }
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return
    print_table(cwd, rows, args.probe)


if __name__ == "__main__":
    main()
