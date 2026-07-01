#!/usr/bin/env python3
"""gate-run — รัน quality gate ของ repo จริง จับผล เขียน ledger

แหล่งความจริงเดียวของคำว่า verified (Memory Schema v1.1) · ผลถูกรันและจดโดยโค้ดนี้ ไม่ใช่ LLM
ใช้:  python gate-run.py --cwd <worktree> --task-id <P#-I#>
คืน: JSON บรรทัดเดียว + exit (pass=0 / fail=1 / no_gate=2 / error=3)
"""
import argparse, json, os, re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

GATE_TIMEOUT = 1800

def detect_gate(cwd: Path):
    pkg = cwd / "package.json"
    if pkg.exists():
        try:
            scripts = json.loads(pkg.read_text(encoding="utf-8")).get("scripts", {})
            pm = "pnpm" if (cwd/"pnpm-lock.yaml").exists() else ("yarn" if (cwd/"yarn.lock").exists() else "npm")
            for key in ("test", "lint", "typecheck", "build"):
                if key in scripts:
                    return ([pm, "run", key], f"{pm} run {key}")
        except Exception:
            pass
    mk = cwd / "Makefile"
    if mk.exists():
        txt = mk.read_text(encoding="utf-8", errors="ignore")
        for target in ("test", "lint", "check", "build"):
            if re.search(rf"^{re.escape(target)}:", txt, re.M):
                return (["make", target], f"make {target}")
    if any((cwd/f).exists() for f in ("pyproject.toml", "pytest.ini", "tox.ini", "setup.cfg")):
        return ([sys.executable, "-m", "pytest", "-q"], "pytest -q")
    return (None, None)

_SECRET_RE = [
    re.compile(r"(https?://)[^/\s:@]+:[^/\s@]+@", re.I),
    re.compile(r"((?:token|password|secret|api[_-]?key|bearer)\s*[=:]\s*)\S+", re.I),
    re.compile(r"\b(sk-[A-Za-z0-9]{8,})\b"),
]
def redact(text: str) -> str:
    if not text: return text
    text = _SECRET_RE[0].sub(r"\1***@", text)
    text = _SECRET_RE[1].sub(r"\1***", text)
    text = _SECRET_RE[2].sub("***", text)
    return text

def git_value(cwd: Path, *args):
    try:
        r = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None

def write_ledger(cwd: Path, row: dict):
    branch = row.get("branch") or "nobranch"
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", branch)
    d = cwd / ".hermes" / "ledger"; d.mkdir(parents=True, exist_ok=True)
    ledger = d / f"{safe}.md"
    cols = ["schema_version","timestamp","machine","staff","branch","issue_id",
            "tool","gate_command","gate_exit","result","commit_sha","status","output_ref"]
    if not ledger.exists():
        ledger.write_text("| "+" | ".join(cols)+" |\n|"+"---|"*len(cols)+"\n", encoding="utf-8")
    with ledger.open("a", encoding="utf-8") as f:
        f.write("| "+" | ".join(str(row.get(c,"")) for c in cols)+" |\n")
    return str(ledger)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cwd", required=True)
    ap.add_argument("--task-id", required=True)
    a = ap.parse_args()
    cwd = Path(a.cwd).expanduser().resolve()
    if not cwd.is_dir():
        print(json.dumps({"gate_status":"error","reason":f"ไม่พบโฟลเดอร์ {cwd}"}, ensure_ascii=False)); sys.exit(3)

    sha = git_value(cwd, "rev-parse", "HEAD")
    branch = git_value(cwd, "rev-parse", "--abbrev-ref", "HEAD")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    machine = os.uname().nodename if hasattr(os,"uname") else "unknown"
    staff = os.environ.get("HERMES_STAFF", os.environ.get("USER","unknown"))
    base = {"schema_version":"relay-1","timestamp":ts,"machine":machine,"staff":staff,
            "branch":branch,"issue_id":a.task_id,"commit_sha":sha or "","tool":"gate-run"}

    cmd, label = detect_gate(cwd)
    if cmd is None:
        write_ledger(cwd, {**base,"gate_command":"","gate_exit":"","result":"no_gate","status":"no_gate","output_ref":""})
        print(json.dumps({"gate_status":"no_gate","gate_exit":None,"gate_command":None,
                          "commit_sha":sha,"output_ref":None,"ledger_written":True}, ensure_ascii=False))
        sys.exit(2)
    try:
        p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=GATE_TIMEOUT)
        exit_code, output = p.returncode, (p.stdout or "")+(p.stderr or "")
    except subprocess.TimeoutExpired:
        write_ledger(cwd, {**base,"gate_command":label,"gate_exit":"timeout","result":"error","status":"error","output_ref":""})
        print(json.dumps({"gate_status":"error","gate_command":label,"reason":"timeout","ledger_written":True}, ensure_ascii=False)); sys.exit(3)
    except Exception as e:
        print(json.dumps({"gate_status":"error","reason":str(e)}, ensure_ascii=False)); sys.exit(3)

    status = "pass" if exit_code == 0 else "fail"
    od = cwd/".hermes"/"gate-output"; od.mkdir(parents=True, exist_ok=True)
    safe_task = re.sub(r"[^A-Za-z0-9._-]","_",a.task_id)
    out_file = od/f"{safe_task}-{ts.replace(':','')}.log"
    out_file.write_text(redact(output), encoding="utf-8")
    ledger = write_ledger(cwd, {**base,"gate_command":label,"gate_exit":exit_code,
                                "result":status,"status":status,"output_ref":str(out_file)})
    print(json.dumps({"gate_status":status,"gate_exit":exit_code,"gate_command":label,
                      "commit_sha":sha,"output_ref":str(out_file),"ledger_written":True}, ensure_ascii=False))
    sys.exit(0 if status=="pass" else 1)

if __name__ == "__main__":
    main()
