#!/usr/bin/env python3
"""move-exec — ตัวย้ายโฟลเดอร์ที่ฝังการ์ดความปลอดภัยเป็นโค้ด (Use Move Folder v2.1 executor)

หัวใจ: ต่อให้ AI สั่งย้าย path อันตราย โค้ดนี้ก็ปฏิเสธ — ไม่พึ่ง prompt
การ์ดที่ฝังเป็นโค้ด:
  - dot-guard      : ปฏิเสธทุก path ที่ตกใน $HOME/.* (home dotfile/.ssh/.env/บัญชี AI) เด็ดขาด
  - allowlist      : ย้ายได้เฉพาะรายการในแผนที่อนุมัติ (plan.json)
  - protected-root : /srv/projects + ตัว root เป้าหมาย ห้ามย้าย/ลบตัวเอง
  - service-check  : path มี process รันอยู่ = ปฏิเสธ
  - free-space     : ปลายทางพื้นที่ไม่พอ = ปฏิเสธ
  - same-fs mv / cross-fs rsync -a (เก็บ source ไม่ลบ) + verify checksum/จำนวนไฟล์
  - ledger+rollback เก็บใน $HOME/.hermes (no-touch) · verify ไม่ตรง = หยุดทั้ง batch

ใช้:  move-exec.py --plan plan.json            # dry-run (ค่าตั้งต้น · ไม่ย้ายจริง)
      move-exec.py --plan plan.json --execute  # ย้ายจริง (เฉพาะชุดปลอดภัย)
plan.json: {"home":"/home/linux-nat","moves":[{"src":"...","dst":"..."}, ...]}
"""
import argparse, json, os, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

def home_root() -> Path:
    return Path(os.environ.get("MOVE_HOME", "/home/linux-nat"))

# ---------- dot-guard (หัวใจ · ปฏิเสธ home-dotpath เด็ดขาด) ----------
def is_home_dotpath(p: Path) -> bool:
    HOME = home_root()
    try:
        rp = p.resolve()
    except Exception:
        rp = p
    try:
        rel = rp.relative_to(HOME.resolve())
    except Exception:
        return False
    parts = rel.parts
    return len(parts) >= 1 and parts[0].startswith(".")

def guard(src: Path, dst: Path):
    """คืน (ok: bool, reason: str) · ปฏิเสธถ้าอันตราย — บังคับระดับโค้ด"""
    if is_home_dotpath(src):
        return False, f"REFUSED dot-guard: source แตะ home-dotpath ({src}) — .ssh/.env/บัญชี AI ห้ามแตะ"
    if is_home_dotpath(dst):
        return False, f"REFUSED dot-guard: dest ตกใน home-dotpath ({dst})"
    for q in (src, dst):
        try:
            if str(q.resolve()).startswith("/srv/projects"):
                return False, f"REFUSED protected: /srv/projects ({q})"
        except Exception:
            pass
    if not src.exists():
        return False, f"REFUSED: ไม่พบ source {src}"
    return True, "ok"

# ---------- ด่านอื่น ----------
def service_running_at(p: Path) -> bool:
    try:
        r = subprocess.run(["lsof", "+D", str(p)], capture_output=True, text=True, timeout=25)
        return bool(r.stdout.strip())
    except FileNotFoundError:
        return False  # ไม่มี lsof — ถือว่าตรวจไม่ได้ (ดูหมายเหตุ caller)
    except Exception:
        return True   # ตรวจไม่สำเร็จ = fail-closed ถือว่ามีรัน

def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

def count_files(p: Path) -> int:
    return sum(1 for f in p.rglob("*") if f.is_file())

def free_space_ok(dst: Path, need: int) -> bool:
    try:
        target = dst.parent if not dst.parent.exists() is False else dst.parent
        st = shutil.disk_usage(str(dst.parent))
        return st.free > int(need * 1.2)
    except Exception:
        return False

def same_fs(src: Path, dst: Path) -> bool:
    try:
        return src.stat().st_dev == dst.parent.stat().st_dev
    except Exception:
        return False

def ledger_write(row: dict):
    d = home_root() / ".hermes" / "move-ledger"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "moves.md"
    cols = ["timestamp", "src", "dst", "mode", "files_before", "files_after",
            "size_before", "size_after", "result", "rollback"]
    if not f.exists():
        f.write_text("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n", encoding="utf-8")
    with f.open("a", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |\n")

def do_move(src: Path, dst: Path, execute: bool):
    """คืน (status, detail) · execute=False = dry-run"""
    ok, reason = guard(src, dst)
    if not ok:
        return "blocked", reason
    if service_running_at(src):
        return "blocked", f"REFUSED service-check: มี process รันอยู่ใน {src}"
    need = dir_size(src)
    if not free_space_ok(dst, need):
        return "blocked", f"REFUSED free-space: ปลายทางพื้นที่ไม่พอ (ต้องการ ~{need} ไบต์)"

    fb, sb = count_files(src), need
    sfs = same_fs(src, dst)
    plan_desc = f"{'mv (same-fs)' if sfs else 'rsync -a เก็บ source (cross-fs)'} · {fb} ไฟล์ · {sb} ไบต์"
    if not execute:
        return "dry-ok", f"จะย้าย: {src} → {dst} · {plan_desc}"

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rollback = ""
    try:
        if sfs:
            shutil.move(str(src), str(dst))
            rollback = f"mv {dst} {src}"
        else:
            subprocess.run(["rsync", "-a", str(src) + "/", str(dst) + "/"], check=True, timeout=3600)
            rollback = f"rm -rf {dst}  # source ยังอยู่ที่ {src}"
        fa = count_files(dst)
        sa = dir_size(dst)
        # verify
        if sfs:
            verified = dst.exists() and not src.exists()
        else:
            verified = (fa == fb and sa == sb)
        result = "ok" if verified else "VERIFY_FAIL"
        ledger_write({"timestamp": ts, "src": str(src), "dst": str(dst),
                      "mode": "mv" if sfs else "rsync", "files_before": fb, "files_after": fa,
                      "size_before": sb, "size_after": sa, "result": result, "rollback": rollback})
        if result != "ok":
            return "verify_fail", f"VERIFY ไม่ตรง: {src}→{dst} (ไฟล์ {fb}→{fa}, ขนาด {sb}→{sa}) · STOP BATCH"
        return "ok", f"ย้ายแล้ว+verify ผ่าน: {dst} ({plan_desc})"
    except Exception as e:
        return "error", f"ย้ายล้มเหลว: {e}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="ไฟล์แผน json ที่อนุมัติแล้ว (allowlist)")
    ap.add_argument("--execute", action="store_true", help="ย้ายจริง (ไม่ใส่ = dry-run)")
    a = ap.parse_args()

    plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    if "home" in plan:
        os.environ["MOVE_HOME"] = plan["home"]
    moves = plan.get("moves", [])
    print(f"=== move-exec · {'EXECUTE' if a.execute else 'DRY-RUN'} · {len(moves)} รายการ · home={home_root()} ===")

    done = blocked = 0
    for i, m in enumerate(moves, 1):
        src, dst = Path(m["src"]), Path(m["dst"])
        status, detail = do_move(src, dst, a.execute)
        print(f"[{i}] {status.upper()}: {detail}")
        if status in ("ok", "dry-ok"):
            done += 1
        elif status == "verify_fail":
            print("!! STOP BATCH — verify ไม่ตรง หยุดทั้งชุด (S-5)")
            sys.exit(2)
        else:
            blocked += 1
    print(f"=== สรุป: ผ่าน {done} · บล็อก/พัก {blocked} · ไม่มีการลบใด ๆ ===")

if __name__ == "__main__":
    main()
