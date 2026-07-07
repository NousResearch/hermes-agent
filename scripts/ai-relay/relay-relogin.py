#!/usr/bin/env python3
"""relay-relogin — คำสั่งเดียว: บอกว่า "ตอนนี้ AI ตัวไหนหมด/หลุด login" + พา login ในที่เดียว

ปัญหาที่แก้: id/โควตาหมดแล้วต้องไล่กด login ทีละ app · ตัวนี้รวมให้จบในคำสั่งเดียว
ใช้:
  relay-relogin --cwd <repo>          # แค่บอกว่าต้อง login ตัวไหน + คำสั่งที่ต้องรัน (ปลอดภัย ไม่รันเอง)
  relay-relogin --cwd <repo> --run    # พา login ตัวที่หมดทีละตัว (รันคำสั่ง login ให้ · เปิดเบราว์เซอร์)
อ่านทะเบียนจาก relay-call · เช็คสถานะสดจาก relay-status (probe จริง)
"""
import argparse, importlib.util, os, pathlib, shlex, shutil, subprocess, sys

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def _load(mod_name, filename):
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_DIR / filename)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def relogin_plan(reg, status_map):
    """ฟังก์ชันบริสุทธิ์: คืนรายการ AI ที่ "ต้อง login ใหม่" (enabled + ยังไม่ ready)
    เรียงให้ตัวที่ 'login ได้ด้วยคำสั่ง' (มี login_cmd) มาก่อน เพื่อ --run พาทำได้เลย"""
    plan = []
    for name, meta in (reg or {}).items():
        if not isinstance(meta, dict) or meta.get("enabled") is not True:
            continue
        st = status_map.get(name) or {}
        if st.get("ready") is True:
            continue
        plan.append({
            "tool": name,
            "login_cmd": meta.get("login_cmd"),           # คำสั่งรันได้ (ถ้ามี)
            "login_hint": meta.get("login_hint") or "",   # คำอธิบายวิธี login (สำรอง)
            "reason": st.get("live") or st.get("hint") or "ยังไม่พร้อม",
        })
    plan.sort(key=lambda p: (0 if p["login_cmd"] else 1, p["tool"]))
    return plan


def safe_login_argv(login_cmd, allowed_bins):
    # กันไฟล์ registry ถูกแก้ให้ --run รันคำสั่งอันตราย (เช่น rm) · โปรแกรมตัวแรกต้องอยู่ในรายชื่ออนุญาต
    if not login_cmd:
        return None
    try:
        argv = shlex.split(login_cmd)
    except ValueError:
        return None
    if not argv:
        return None
    if pathlib.Path(argv[0]).name not in set(allowed_bins or []):
        return None
    return argv


def _live_status_map(relay_call, relay_status, reg, cwd, do_probe):
    cooldown = relay_status.read_cooldown_map(cwd)
    status_map = {}
    for name, meta in relay_call.registry_enabled(reg).items():
        probe = relay_status.run_probe(cwd, name) if do_probe else None
        status_map[name] = relay_status.tool_status(name, meta, shutil.which, cooldown, probe)
    return status_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cwd", default=os.environ.get("AI_RELAY_ROOT") or ".")
    ap.add_argument("--run", action="store_true", help="รันคำสั่ง login ให้ตัวที่หมดทีละตัว (เปิดเบราว์เซอร์)")
    ap.add_argument("--no-probe", action="store_true", help="ไม่ยิงเช็ค login จริง (ดูแค่ติดตั้ง/พัก · เร็ว/ฟรี)")
    a = ap.parse_args()
    cwd = pathlib.Path(a.cwd).expanduser().resolve()

    relay_call = _load("relay_call", "relay-call.py")
    relay_status = _load("relay_status", "relay-status.py")
    reg = relay_call.load_registry(cwd)

    status_map = _live_status_map(relay_call, relay_status, reg, cwd, do_probe=not a.no_probe)
    plan = relogin_plan(reg, status_map)

    if not plan:
        print("✅ ทุกตัวพร้อมใช้ ไม่มีตัวไหนต้อง login ใหม่")
        return

    print(f"══ ต้อง login ใหม่ {len(plan)} ตัว ══")
    for i, p in enumerate(plan, 1):
        how = p["login_cmd"] or p["login_hint"] or "(ดูวิธี login ของตัวนี้เอง)"
        print(f"  {i}. {p['tool']} — เหตุ: {p['reason']} → รัน: {how}")

    if not a.run:
        print("\nพา login ให้อัตโนมัติทีละตัว: เพิ่ม --run  (เช่น  relay-relogin --cwd <repo> --run)")
        print("ตัวที่ไม่มีคำสั่ง login อัตโนมัติ ให้ทำตามวิธีที่บอกด้านบนเอง")
        return

    # --run: พา login ตัวที่มีคำสั่ง login ทีละตัว (interactive · เปิดเบราว์เซอร์)
    allowed = relay_call.allowed_bins()
    ran = []
    for p in plan:
        argv = safe_login_argv(p["login_cmd"], allowed)
        if argv is None:
            if p["login_cmd"]:
                print(f"\n⛔ ข้าม {p['tool']} — คำสั่ง login ไม่อยู่ในรายชื่ออนุญาต (กันไฟล์ตั้งค่าถูกแก้ให้รันคำสั่งอันตราย): {p['login_cmd']}")
            else:
                print(f"\n⏭️  ข้าม {p['tool']} — ไม่มีคำสั่ง login อัตโนมัติ · ทำเองตาม: {p['login_hint']}")
            continue
        print(f"\n▶️  login {p['tool']} ... (ทำตามหน้าจอ/เบราว์เซอร์)")
        rc = subprocess.call(argv)
        ran.append((p["tool"], rc))
        print(f"   {p['tool']}: {'สำเร็จ' if rc == 0 else f'ยังไม่สำเร็จ (exit {rc})'}")

    print("\n══ สรุป ══")
    for tool, rc in ran:
        print(f"  {tool}: {'✅' if rc == 0 else '⚠️ ลองใหม่'}")
    print("เช็คซ้ำว่าพร้อมหมดยัง:  relay-status --probe --cwd", str(cwd))


if __name__ == "__main__":
    main()
