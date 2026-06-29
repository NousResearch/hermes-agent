#!/usr/bin/env python3
"""relay-call — เรียก AI coder 1 ครั้ง · จับผลตายตัว · สลับบัญชี · นับงบ · เขียน ledger

ส่วนของ Use AI Relay (Memory Schema v1.1) · LLM อ่านแค่ช่อง status ที่ตัวนี้คืน ไม่ parse stderr เอง
ใช้:  python relay-call.py --tool grok --task-id P1-I2 --prompt-file brief.md --cwd <worktree>
คืน: JSON บรรทัดเดียว + exit (ok=0 not_found=10 auth=20 quota=30 crash=40 limit_exceeded=50)

หมายเหตุ: คำสั่งเรียก coder อ่านจาก .hermes/ai-relay/adapters.yaml
          บัญชี/สาย/เพดาน อ่านจาก .hermes/ai-relay/accounts.yaml
          ถ้าไม่มีไฟล์ ใช้ค่าปริยายในตัว (รองรับ ollama ได้ทันทีเพื่อทดสอบ)
"""
import argparse, glob, json, os, re, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
try:
    import yaml
except ImportError:
    yaml = None

# ---- ค่าปริยาย (ใช้เมื่อไม่มีไฟล์ตั้งค่า) ----
DEFAULT_ADAPTERS = {
    "grok":   {"cmd": ["grok","-p","{prompt}","--cwd","{cwd}","--output-format","json","--always-approve"]},
    "gemini": {"cmd": ["gemini","-p","{prompt}","-m","{model}","--approval-mode","yolo"], "run_in_cwd": True},
    "ollama": {"cmd": ["ollama","run","{model}","{prompt}"], "run_in_cwd": True},
    "codex":  {"cmd": ["codex","exec","--skip-git-repo-check","--color","never","{prompt}"], "run_in_cwd": True},
}
DEFAULT_ACCOUNTS = {
    "fallback": {"code_writing": ["grok","codex","gemini","ollama"]},
    "limits": {"max_rounds_per_issue": 3, "max_calls_per_session": 50, "budget": None},
    "ollama_models": {"default": "qwen3:8b", "code": "deepseek-r1:7b"},
}

def load_yaml(p: Path):
    if p.exists() and yaml:
        try: return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception: return {}
    return {}

def cfg_dir(cwd: Path): return cwd/".hermes"/"ai-relay"

# ---- จัดประเภท error จาก exit code + stderr (เป็นที่เดียวที่ตีความ) ----
def classify(exit_code, stderr):
    s = (stderr or "").lower()
    if exit_code == 127 or "command not found" in s or "no such file" in s:
        return "not_found"
    if re.search(r"unauthor|\b401\b|\b403\b|not logged|please login|sign ?in|credential|invalid api key", s):
        return "auth"
    if re.search(r"\b429\b|rate.?limit|quota|too many request|usage limit|insufficient", s):
        return "quota"
    if exit_code == 0:
        return "ok"
    return "crash"

REASON = {
    "ok": "เรียกสำเร็จ",
    "not_found": "ไม่มีโปรแกรม AI ตัวนี้บนเครื่องนี้ (ติดตั้งก่อน หรือสลับตัวอื่น)",
    "auth": "บัญชีหลุด/ยังไม่ล็อกอิน ต้องล็อกอินใหม่",
    "quota": "บัญชีนี้เกินโควต้า สลับบัญชีถัดไป",
    "crash": "AI ตอบไม่ได้/พัง ลองซ้ำหรือสลับตัวสำรอง",
}

_SECRET_RE = re.compile(r"((?:token|password|secret|api[_-]?key|bearer)\s*[=:]\s*)\S+", re.I)
def redact(t): return _SECRET_RE.sub(r"\1***", t or "")

def resolve_codex_bin():
    """หา codex CLI ข้ามเครื่อง · ไล่ตามลำดับ:
       1) env RELAY_CODEX_BIN / XC_CODEX_BIN  2) Cursor extension (Mac/laptop พนักงาน)
       3) codex บน PATH (เช่น /usr/bin/codex บน VPS)  4) ~/.codex/bin/codex
       คืน 'codex' เฉย ๆ ถ้าหาไม่เจอ (ให้ relay จัดเป็น not_found แล้วสลับตัวอื่น)"""
    for env in (os.environ.get("RELAY_CODEX_BIN"), os.environ.get("XC_CODEX_BIN")):
        if env and Path(env).exists():
            return env
    exts = sorted(p for p in glob.glob(str(Path.home()/".cursor"/"extensions"/"openai.chatgpt-*"/"bin"/"*"/"codex")) if Path(p).exists())
    if exts:
        return exts[-1]
    on_path = shutil.which("codex")
    if on_path:
        return on_path
    fb = Path.home()/".codex"/"bin"/"codex"
    return str(fb) if fb.exists() else "codex"

def relay_now(action, tool="", task="", phase=""):
    sh = SCRIPT_DIR/"relay-now.sh"
    if not sh.exists(): return
    args = ["bash", str(sh), action]
    if action == "set":
        args += ["--tool", tool, "--task", task, "--phase", phase]
    try: subprocess.run(args, capture_output=True, timeout=10)
    except Exception: pass

def write_ledger(cwd: Path, row: dict):
    branch = row.get("branch") or "nobranch"
    safe = re.sub(r"[^A-Za-z0-9._-]","_",branch)
    d = cfg_dir(cwd); d.mkdir(parents=True, exist_ok=True)
    f = d/f"calls-{safe}.md"
    cols = ["timestamp","issue_id","tool","account_used","rotated_from","status","calls_used","output_ref"]
    if not f.exists():
        f.write_text("| "+" | ".join(cols)+" |\n|"+"---|"*len(cols)+"\n", encoding="utf-8")
    with f.open("a", encoding="utf-8") as fh:
        fh.write("| "+" | ".join(str(row.get(c,"")) for c in cols)+" |\n")
    return str(f)

def run_once(spec, prompt, cwd, model):
    cmd = [a.replace("{prompt}",prompt).replace("{cwd}",str(cwd)).replace("{model}",model or "") for a in spec["cmd"]]
    workdir = str(cwd) if spec.get("run_in_cwd") else None
    try:
        p = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=1800,
                           stdin=subprocess.DEVNULL)  # ปิด stdin · กัน codex exec ค้างรออ่าน input
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return 127, "", "command not found"
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"

# ---- ตัวนับงบระดับ session (ไฟล์เล็กใน cfg dir) ----
def bump_calls(cwd: Path):
    f = cfg_dir(cwd)/".session-calls"; f.parent.mkdir(parents=True, exist_ok=True)
    n = int(f.read_text()) if f.exists() else 0
    n += 1; f.write_text(str(n)); return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True)
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--cwd", required=True)
    a = ap.parse_args()

    cwd = Path(a.cwd).expanduser().resolve()
    prompt = Path(a.prompt_file).read_text(encoding="utf-8") if Path(a.prompt_file).exists() else a.prompt_file
    adapters = {**DEFAULT_ADAPTERS, **(load_yaml(cfg_dir(cwd)/"adapters.yaml").get("tools") or {})}
    # ทำให้ codex พกพาข้ามเครื่อง: ถ้า adapter เรียกชื่อ "codex" ลอย ๆ แปลงเป็น path จริงที่หาเจอ
    if "codex" in adapters:
        ccmd = list(adapters["codex"].get("cmd", []))
        if ccmd and ccmd[0] == "codex":
            ccmd[0] = resolve_codex_bin()
            adapters["codex"] = {**adapters["codex"], "cmd": ccmd}
    accounts = {**DEFAULT_ACCOUNTS, **load_yaml(cfg_dir(cwd)/"accounts.yaml")}
    limits = accounts.get("limits", DEFAULT_ACCOUNTS["limits"])
    models = accounts.get("ollama_models", DEFAULT_ACCOUNTS["ollama_models"])

    # เพดาน session
    calls = bump_calls(cwd)
    if limits.get("max_calls_per_session") and calls > limits["max_calls_per_session"]:
        out = {"status":"limit_exceeded","tool":a.tool,"reason_human":"เกินเพดานจำนวนครั้งต่อรอบงาน หยุดเพื่อกันค่าใช้จ่ายบาน",
               "calls_used":calls,"ledger_written":False}
        print(json.dumps(out, ensure_ascii=False)); sys.exit(50)

    # สายลำดับ: tool ที่เลือก + สำรองจาก fallback (ตัด tool ที่ไม่มี adapter)
    chain = [a.tool] + [t for t in accounts.get("fallback",{}).get("code_writing",[]) if t != a.tool]
    chain = [t for t in chain if t in adapters]

    tried, rotated_from = [], ""
    for tool in chain:
        model = models.get("code") if tool == "ollama" else ""
        relay_now("set", tool, a.task_id, "กำลังเขียนโค้ด")
        code, out, err = run_once(adapters[tool], prompt, cwd, model)
        st = classify(code, err)
        tried.append(f"{tool}:{st}")

        if st == "ok":
            ofile = cfg_dir(cwd)/f"out-{re.sub(r'[^A-Za-z0-9]', '_', a.task_id)}.txt"
            ofile.write_text(redact(out), encoding="utf-8")
            relay_now("clear")
            ledger = write_ledger(cwd, {"timestamp":datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "issue_id":a.task_id,"tool":tool,"account_used":tool,"rotated_from":rotated_from,
                "status":"ok","calls_used":calls,"output_ref":str(ofile)})
            print(json.dumps({"status":"ok","tool":tool,"account_used":tool,"rotated_from":rotated_from,
                "reason_human":REASON["ok"],"output_ref":str(ofile),"ledger_written":True,"calls_used":calls,
                "tried":tried}, ensure_ascii=False))
            sys.exit(0)

        # ไม่ ok → ตัดสินว่าสลับหรือหยุด
        if st in ("quota", "crash"):
            rotated_from = tool
            continue   # สลับตัวถัดไปในสาย
        if st == "auth":
            relay_now("clear")
            print(json.dumps({"status":"auth","tool":tool,"reason_human":REASON["auth"],
                "hint":f"ล็อกอินใหม่: {tool} login","tried":tried}, ensure_ascii=False)); sys.exit(20)
        if st == "not_found":
            rotated_from = tool
            continue   # ไม่มีตัวนี้ → ลองตัวถัดไป

    relay_now("clear")
    print(json.dumps({"status":"crash","reason_human":"ลองครบทุกตัวในสายแล้วยังไม่สำเร็จ","tried":tried},
                     ensure_ascii=False))
    sys.exit(40)

if __name__ == "__main__":
    main()
