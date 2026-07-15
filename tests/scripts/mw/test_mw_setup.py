"""Test scripts/mw/mw-setup.sh installs the MW tools and smoke-checks them (MW-P4/P6).

2026-07-15: ติดตั้งแบบ copy เข้า $MW_LIB_DIR ถาวรแล้วค่อย link (บทเรียนเครื่องทีม:
ต้นทาง /tmp ของตัวติดตั้ง curl ถูกลบหลังติดตั้ง → symlink ตายยกชุด) ·
mw-spec-check เป็นเครื่องมือฝั่งนักพัฒนา ไม่ติดตั้งให้ทีมแล้ว ·
ทุกเทสต์ต้องตั้ง MW_LIB_DIR ชั่วคราว — ห้ามให้เทสต์เขียน ~/.hermes/mw จริง (Codex review)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SETUP = REPO_ROOT / "scripts" / "mw" / "mw-setup.sh"

EXPECTED = {
    "mw-work-locks",
    "mw-menu-gate",
    "mw-page-check",
    "mw-doctor",
    "mw-rtm-report",
    "mw-wow-report",
    "mw-backend-check",
}

SHARED = {"flow_eval.py", "flow_gate.py", "flow-rules.yaml"}


def _env(tmp_path: Path) -> dict:
    env = os.environ.copy()
    env["MW_BIN_DIR"] = str(tmp_path / "bin")
    env["MW_LIB_DIR"] = str(tmp_path / "lib")
    env["MW_PYTHON"] = sys.executable  # the venv python running the tests
    return env


def _run(setup: Path, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(setup)], capture_output=True, text=True, env=env)


def test_setup_installs_and_smokes_all_tools(tmp_path: Path):
    env = _env(tmp_path)
    proc = _run(SETUP, env)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    bindir = tmp_path / "bin"
    installed = {p.name for p in bindir.iterdir()}
    assert EXPECTED <= installed, (EXPECTED - installed, installed)
    # symlinks resolve to persistent copies under MW_LIB_DIR (not the repo/source dir)
    libdir = (tmp_path / "lib").resolve()
    for name in EXPECTED:
        link = bindir / name
        assert link.is_symlink()
        real = link.resolve()
        assert real.is_file()
        assert real.parent == libdir, (name, real)
    # shared files for the enforce-flow-gate hook are present in LIB_DIR
    lib_names = {p.name for p in libdir.iterdir()}
    assert SHARED <= lib_names, (SHARED - lib_names, lib_names)
    # smoke summary: 0 failures
    assert "ทดสอบไม่ผ่าน 0" in proc.stdout
    # dev-only tool must NOT be installed for the team
    assert "mw-spec-check" not in installed


def test_tools_survive_source_deletion(tmp_path: Path):
    """Regression 2026-07-15: การติดตั้งผ่าน curl รันจากโฟลเดอร์ /tmp ที่ถูกลบทิ้ง —
    เครื่องมือทุกตัวต้องยังรันได้หลังต้นทางหาย (เดิม symlink ชี้ต้นทางตรง ๆ = ตายยกชุด)"""
    # จำลอง: คัดชุด scripts ไปโฟลเดอร์ชั่วคราว (เหมือน tarball ที่ดาวน์โหลด)
    staging = tmp_path / "downloaded"
    shutil.copytree(REPO_ROOT / "scripts" / "mw", staging / "scripts" / "mw")
    env = _env(tmp_path)
    proc = _run(staging / "scripts" / "mw" / "mw-setup.sh", env)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    # ลบต้นทางทิ้ง — จุดที่เคยพังจริงบน VPS
    shutil.rmtree(staging)
    for name in sorted(EXPECTED):
        link = tmp_path / "bin" / name
        run = subprocess.run(
            [str(link), "--help"], capture_output=True, text=True, env=env
        )
        assert run.returncode == 0, (name, run.stdout, run.stderr)


def test_setup_cleans_stale_spec_check_symlink(tmp_path: Path):
    """Codex review: เครื่องที่เคยติดรุ่นเก่าต้องไม่เหลือลิงก์ mw-spec-check ค้าง/เสีย"""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stale = bindir / "mw-spec-check"
    stale.symlink_to(tmp_path / "gone" / "mw-spec-check.py")  # ลิงก์เสียแบบของจริง
    env = _env(tmp_path)
    proc = _run(SETUP, env)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert not stale.exists() and not stale.is_symlink()


def test_setup_keeps_real_spec_check_file(tmp_path: Path):
    """ลบเฉพาะ symlink — ไฟล์จริงชื่อเดียวกันของผู้ใช้ห้ามแตะ"""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    real = bindir / "mw-spec-check"
    real.write_text("USER FILE", encoding="utf-8")
    env = _env(tmp_path)
    proc = _run(SETUP, env)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert real.read_text(encoding="utf-8") == "USER FILE"


def test_setup_fails_closed_when_python_missing(tmp_path: Path):
    env = _env(tmp_path)
    env["MW_PYTHON"] = "python-does-not-exist-xyz"
    proc = _run(SETUP, env)
    assert proc.returncode == 1
    assert "ไม่พบ" in (proc.stdout + proc.stderr)


def test_setup_stops_before_linking_when_copy_incomplete(tmp_path: Path):
    """Codex review: คัดลอกไม่ครบต้องหยุดก่อนสร้างคำสั่ง — กันชุดครึ่ง ๆ กลาง ๆ"""
    staging = tmp_path / "partial"
    shutil.copytree(REPO_ROOT / "scripts" / "mw", staging / "scripts" / "mw")
    (staging / "scripts" / "mw" / "backend_check.py").unlink()  # ทำให้ชุดขาด 1 ไฟล์
    env = _env(tmp_path)
    proc = _run(staging / "scripts" / "mw" / "mw-setup.sh", env)
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    bindir = tmp_path / "bin"
    linked = {p.name for p in bindir.iterdir()} if bindir.exists() else set()
    assert not (EXPECTED & linked), linked  # ห้ามมีคำสั่งโผล่มาแม้ตัวเดียว


def test_setup_refuses_to_clobber_real_file(tmp_path: Path):
    """GPT-5: ln -sf must NOT silently overwrite a user's real (non-symlink) file."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    victim = bindir / "mw-doctor"
    victim.write_text("MY REAL FILE — do not clobber", encoding="utf-8")
    env = _env(tmp_path)
    proc = _run(SETUP, env)
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    assert not victim.is_symlink()
    assert victim.read_text(encoding="utf-8") == "MY REAL FILE — do not clobber"


def test_setup_symlink_to_dir_does_not_clobber_inside(tmp_path: Path):
    """GPT-5 round 2: a pre-existing symlink-to-directory must not let ln follow into it
    and overwrite a real file inside that directory."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    realdir = tmp_path / "elsewhere"
    realdir.mkdir()
    inside = realdir / "mw-doctor"
    inside.write_text("REAL FILE INSIDE DIR", encoding="utf-8")
    (bindir / "mw-doctor").symlink_to(realdir)
    env = _env(tmp_path)
    proc = _run(SETUP, env)
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert inside.read_text(encoding="utf-8") == "REAL FILE INSIDE DIR"
    link = bindir / "mw-doctor"
    assert link.is_symlink()
    assert link.resolve().name == "mw_doctor.py"
