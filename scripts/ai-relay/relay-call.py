#!/usr/bin/env python3
"""relay-call — เรียก AI coder 1 ครั้ง · จับผลตายตัว · สลับบัญชี · นับงบ · เขียน ledger

ส่วนของ Use AI Relay (Memory Schema v1.1) · LLM อ่านแค่ช่อง status ที่ตัวนี้คืน ไม่ parse stderr เอง
ใช้:  python relay-call.py --tool grok --task-id P1-I2 --prompt-file brief.md --cwd <worktree>
คืน: JSON บรรทัดเดียว + exit (ok=0 not_found=10 auth=20 quota=30 crash=40 limit_exceeded=50)

หมายเหตุ: คำสั่งเรียก coder อ่านจาก .hermes/ai-relay/adapters.yaml
          บัญชี/สาย/เพดาน อ่านจาก .hermes/ai-relay/accounts.yaml
          ถ้าไม่มีไฟล์ ใช้ค่าปริยายในตัว (รองรับ ollama ได้ทันทีเพื่อทดสอบ)
v2.2:     เพิ่ม fable (สมองพิเศษ · --tool fable · เพดานแยก max_fable_calls_per_session ·
          ยืนเดี่ยวไม่เข้าสาย coder) · เพดานรอบต่อ issue นับข้าม coder · cooldown ตัวที่พังซ้ำ ·
          อ่าน YAML ได้แม้ไม่มี PyYAML (ตัวอ่านสำรองในตัว)
"""
import argparse, glob, json, os, re, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
try:
    import yaml
except ImportError:
    yaml = None  # ไม่มี PyYAML ก็ยังอ่าน config ได้ ด้วยตัวอ่านสำรองข้างล่าง
try:
    import fcntl  # ล็อกไฟล์กันสองโปรเซสเขียนตัวนับ/สถานะพร้อมกัน (มีบน Mac/Linux)
except ImportError:
    fcntl = None

# บัญชีโปรแกรมที่อนุญาตให้ adapter เรียกได้ (กันไฟล์ตั้งค่าใน worktree ถูกแก้ให้รันคำสั่งอันตราย)
# เพิ่มชั่วคราวได้ทาง env RELAY_EXTRA_BINS=ชื่อ:ชื่อ (ตั้งโดยคนคุมเครื่อง ไม่ใช่ไฟล์ใน worktree)
ALLOWED_BINS = {"grok", "gemini", "ollama", "codex", "claude"}
def allowed_bins():
    extra = os.environ.get("RELAY_EXTRA_BINS", "")
    return ALLOWED_BINS | {b for b in extra.split(":") if b}

# ---- ค่าปริยาย (ใช้เมื่อไม่มีไฟล์ตั้งค่า) ----
DEFAULT_ADAPTERS = {
    "grok":   {"cmd": ["grok","-p","{prompt}","--cwd","{cwd}","--output-format","json","--always-approve"]},
    "gemini": {"cmd": ["gemini","-p","{prompt}","-m","gemini-2.5-flash","--skip-trust","--approval-mode","yolo","--output-format","text"], "run_in_cwd": True},
    "ollama": {"cmd": ["ollama","run","{model}","{prompt}"], "run_in_cwd": True},
    "codex":  {"cmd": ["codex","exec","--skip-git-repo-check","--color","never","{prompt}"], "run_in_cwd": True},
    # สมองพิเศษ (brain · premium) · เฉพาะงานเกรด "ยาก" หรือบันไดส่งต่อขึ้น · ห้ามเข้าสายสำรองของ coder
    # premium=True = ตัวแพงสุด มีเพดานเรียกแยก · โดน limit/พัง → ตกไปสมองสำรอง (opus) อัตโนมัติ
    "fable":  {"cmd": ["claude","--model","claude-fable-5","-p","{prompt}"], "run_in_cwd": True, "brain": True, "premium": True},
    # สมองสำรอง (brain) · รับช่วงเมื่อ fable โดน limit หรือเรียกไม่ได้ · ทำงานร่วมกับ coder เหมือนเดิม
    "opus":   {"cmd": ["claude","--model","claude-opus-4-8","-p","{prompt}"], "run_in_cwd": True, "brain": True},
}
DEFAULT_ACCOUNTS = {
    "fallback": {"code_writing": ["grok","codex","gemini","ollama"],
                 # สายสมอง: fable โดน limit/พัง → opus 4.8 อัตโนมัติ (แก้ทีเดียวใช้ทุกเครื่อง)
                 "brain": ["opus"]},
    "limits": {"max_rounds_per_issue": 3, "max_calls_per_session": 50,
               "max_fable_calls_per_session": 3, "session_hours": 12, "budget": None},
    "cooldown": {"fail_threshold": 3, "window_seconds": 300, "minutes": 10},
    "ollama_models": {"default": "qwen3:8b", "code": "deepseek-r1:7b"},
}

# ---- ตัวอ่าน YAML สำรอง (ใช้เมื่อไม่มี PyYAML) ----
# อ่านเฉพาะโครงที่ Relay ใช้จริง: mapping ซ้อนกัน · list ของค่า · list ของ mapping (- id: x)
def _scalar(v: str):
    v = v.strip()
    if v[:1] in "\"'" and v[-1:] == v[:1] and len(v) >= 2:
        return v[1:-1]
    if " #" in v:  # ตัด comment ท้ายบรรทัด (ค่าไม่ได้ห่อ quote)
        v = v.split(" #", 1)[0].strip()
    low = v.lower()
    if low in ("null", "~", ""): return None
    if low == "true": return True
    if low == "false": return False
    for cast in (int, float):
        try: return cast(v)
        except ValueError: pass
    return v

def _mini_yaml(text: str):
    lines = [l for l in (raw.rstrip() for raw in text.splitlines())
             if l.strip() and not l.strip().startswith("#")]
    pos = 0
    def indent_of(l): return len(l) - len(l.lstrip(" "))
    def parse_block(indent):
        nonlocal pos
        result = None
        while pos < len(lines):
            line = lines[pos]
            cur = indent_of(line)
            if cur != indent:
                break
            s = line.strip()
            if s.startswith("- "):
                if result is None: result = []
                if not isinstance(result, list): break
                item = s[2:].strip()
                pos += 1
                if ":" in item and item[:1] not in "\"'":
                    k, _, v = item.partition(":")
                    d = {k.strip(): _scalar(v) if v.strip() else None}
                    while pos < len(lines):
                        nline = lines[pos]
                        ni = indent_of(nline)
                        if ni <= indent or nline.strip().startswith("- "): break
                        nk, _, nv = nline.strip().partition(":")
                        pos += 1
                        if nv.strip():
                            d[nk.strip()] = _scalar(nv)
                        elif pos < len(lines) and indent_of(lines[pos]) > ni:
                            d[nk.strip()] = parse_block(indent_of(lines[pos]))
                        else:
                            d[nk.strip()] = None
                    result.append(d)
                else:
                    result.append(_scalar(item))
            else:
                if result is None: result = {}
                if not isinstance(result, dict): break
                k, _, v = s.partition(":")
                pos += 1
                if v.strip():
                    result[k.strip()] = _scalar(v)
                elif pos < len(lines) and indent_of(lines[pos]) > cur:
                    result[k.strip()] = parse_block(indent_of(lines[pos]))
                else:
                    result[k.strip()] = None
        return result
    return parse_block(0) or {}

def load_yaml(p: Path):
    if not p.exists():
        return {}
    try:
        text = p.read_text(encoding="utf-8")
        if yaml:
            return yaml.safe_load(text) or {}
        return _mini_yaml(text)
    except Exception:
        return {}

def cfg_dir(cwd: Path): return cwd/".hermes"/"ai-relay"

def load_env_file(path: Path):
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            continue
        if os.environ.get(key):
            continue
        value = value.strip().strip("\"'")
        os.environ[key] = value

def load_relay_env(cwd: Path):
    # ให้ CLI ที่ relay-call เรียกตรง ๆ เห็น key เดียวกับ Hermes โดยไม่พิมพ์ secret ออก stdout
    load_env_file(Path.home()/".hermes"/".env")
    load_env_file(cwd/".hermes"/".env")

AUTH_RE = re.compile(r"unauthor|\b401\b|\b403\b|not logged|please login|sign ?in|credential|invalid api key|auth method|gemini_api_key|google_api_key|environment variables|not authenticated|disabled .*access|subscription access|use an anthropic api key|organization has disabled", re.I)
QUOTA_RE = re.compile(r"\b429\b|rate.?limit|quota|too many request|usage limit|insufficient", re.I)
# ข้อความ auth ที่ชัดเจนมาก (คำตอบปกติไม่มีทางพูด) · ถือเป็นล็อกอินพัง แม้ CLI จะ exit 0 · กันเอา error มาเป็นคำตอบ
STRONG_AUTH_RE = re.compile(r"you are not authenticated|organization has disabled|subscription access for claude|use an anthropic api key instead", re.I)

# ---- จัดประเภท error จาก exit code + stdout/stderr (เป็นที่เดียวที่ตีความ) ----
def classify(exit_code, stdout, stderr):
    out = stdout or ""
    err = stderr or ""
    err_low = err.lower()
    if exit_code == 127:
        return "not_found"
    if exit_code == 0:
        # บาง CLI (เช่น claude เมื่อ org ปิดสิทธิ์) พิมพ์ error ยาวแต่ exit 0 · กันเอา error มาเป็นคำตอบ
        if STRONG_AUTH_RE.search(out) or STRONG_AUTH_RE.search(err):
            return "auth"
        if len(out.strip()) < 40 and AUTH_RE.search(err):
            return "auth"
        return "ok"
    if "command not found" in err_low or "no such file" in err_low:
        return "not_found"
    if AUTH_RE.search(err):
        return "auth"
    if QUOTA_RE.search(err):
        return "quota"
    if AUTH_RE.search(out):
        return "auth"
    if QUOTA_RE.search(out):
        return "quota"
    return "crash"

REASON = {
    "ok": "เรียกสำเร็จ",
    "not_found": "ไม่มีโปรแกรม AI ตัวนี้บนเครื่องนี้ (ติดตั้งก่อน หรือสลับตัวอื่น)",
    "auth": "บัญชีหลุด/ยังไม่ล็อกอิน ต้องล็อกอินใหม่",
    "quota": "บัญชีนี้เกินโควต้า สลับบัญชีถัดไป",
    "crash": "AI ตอบไม่ได้/พัง ลองซ้ำหรือสลับตัวสำรอง",
}

# กฎบังรหัสลับชุดเดียวกับ gate-run (URL ฝัง user:pass · key=value · รหัสขึ้นต้น sk-)
_SECRET_RE = [
    re.compile(r"(https?://)[^/\s:@]+:[^/\s@]+@", re.I),
    re.compile(r"((?:token|password|secret|api[_-]?key|bearer)\s*[=:]\s*)\S+", re.I),
    re.compile(r"\b(sk-[A-Za-z0-9]{8,})\b"),
]
def redact(t):
    if not t: return t or ""
    t = _SECRET_RE[0].sub(r"\1***@", t)
    t = _SECRET_RE[1].sub(r"\1***", t)
    t = _SECRET_RE[2].sub("***", t)
    return t

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
        if fcntl: fcntl.flock(fh, fcntl.LOCK_EX)
        fh.write("| "+" | ".join(str(row.get(c,"")) for c in cols)+" |\n")
        if fcntl: fcntl.flock(fh, fcntl.LOCK_UN)
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

# ---- ตัวนับงบระดับ session (ไฟล์เล็กใน cfg dir · ล็อกไฟล์กันนับพลาดเมื่อรันพร้อมกัน) ----
def bump_counter(cwd: Path, name: str, session_hours=12):
    f = cfg_dir(cwd)/name; f.parent.mkdir(parents=True, exist_ok=True)
    with open(f, "a+", encoding="utf-8") as fh:
        if fcntl: fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        raw = (fh.read() or "").strip()
        now = time.time()
        started = now
        try:
            data = json.loads(raw) if raw else {}
            n = int(data.get("count", 0) or 0)
            started = float(data.get("started", now) or now)
        except Exception:
            try: n = int(raw or 0)
            except Exception: n = 0
            started = now
        try:
            hours = float(session_hours or 12)
        except Exception:
            hours = 12
        if hours > 0 and now - started > hours * 3600:
            n = 0
            started = now
        n += 1
        fh.seek(0); fh.truncate(); fh.write(json.dumps({"count": n, "started": started}))
        if fcntl: fcntl.flock(fh, fcntl.LOCK_UN)
    return n

def bump_calls(cwd: Path, session_hours=12):
    return bump_counter(cwd, ".session-calls", session_hours=session_hours)

# ---- cooldown: ตัวไหนพัง/ชนโควต้าซ้ำในช่วงสั้น พักตัวนั้นชั่วคราว ไม่เสียเวลาลองซ้ำทุกรอบ ----
def _cooldown_file(cwd: Path): return cfg_dir(cwd)/".cooldown.json"

def load_cooldown(cwd: Path):
    f = _cooldown_file(cwd)
    try: return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    except Exception: return {}

def save_cooldown(cwd: Path, state: dict):
    f = _cooldown_file(cwd); f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".json.tmp")   # เขียนไฟล์ชั่วคราวแล้วสลับทั้งก้อน กันไฟล์ครึ่งๆ กลางๆ
    tmp.write_text(json.dumps(state), encoding="utf-8")
    os.replace(tmp, f)

def in_cooldown(state: dict, tool: str, now: float):
    return float(state.get(tool, {}).get("until", 0)) > now

def record_fail(cwd: Path, tool: str, cd: dict, now: float):
    # อ่าน-แก้-เขียน ใต้ล็อกเดียว กันสองโปรเซสทับสถานะกัน
    lockf = cfg_dir(cwd)/".cooldown.lock"; lockf.parent.mkdir(parents=True, exist_ok=True)
    with open(lockf, "w") as lk:
        if fcntl: fcntl.flock(lk, fcntl.LOCK_EX)
        state = load_cooldown(cwd)
        e = state.setdefault(tool, {"fails": [], "until": 0})
        window = float(cd.get("window_seconds", 300))
        e["fails"] = [t for t in e.get("fails", []) if now - t <= window] + [now]
        if len(e["fails"]) >= int(cd.get("fail_threshold", 3)):
            e["until"] = now + float(cd.get("minutes", 10)) * 60
            e["fails"] = []
        save_cooldown(cwd, state)
        if fcntl: fcntl.flock(lk, fcntl.LOCK_UN)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True)
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument(
        "--cwd",
        default=os.environ.get("AI_RELAY_ROOT") or str(Path.home()),
        help="โฟลเดอร์ทำงานและที่เก็บ .hermes/ai-relay; ค่าเริ่มต้นคือ $AI_RELAY_ROOT หรือ home ของผู้ใช้",
    )
    a = ap.parse_args()

    cwd = Path(a.cwd).expanduser().resolve()
    load_relay_env(cwd)
    prompt = Path(a.prompt_file).read_text(encoding="utf-8") if Path(a.prompt_file).exists() else a.prompt_file
    adapters = {**DEFAULT_ADAPTERS, **(load_yaml(cfg_dir(cwd)/"adapters.yaml").get("tools") or {})}
    # ทำให้ codex พกพาข้ามเครื่อง: ถ้า adapter เรียกชื่อ "codex" ลอย ๆ แปลงเป็น path จริงที่หาเจอ
    if "codex" in adapters:
        ccmd = list(adapters["codex"].get("cmd", []))
        if ccmd and ccmd[0] == "codex":
            ccmd[0] = resolve_codex_bin()
            adapters["codex"] = {**adapters["codex"], "cmd": ccmd}
    accounts = {**DEFAULT_ACCOUNTS, **load_yaml(cfg_dir(cwd)/"accounts.yaml")}
    limits = {**DEFAULT_ACCOUNTS["limits"], **(accounts.get("limits") or {})}
    cd_cfg = {**DEFAULT_ACCOUNTS["cooldown"], **(accounts.get("cooldown") or {})}
    models = accounts.get("ollama_models", DEFAULT_ACCOUNTS["ollama_models"])
    is_brain = bool(adapters.get(a.tool, {}).get("brain"))
    session_hours = limits.get("session_hours", DEFAULT_ACCOUNTS["limits"]["session_hours"])

    # เพดาน session
    calls = bump_calls(cwd, session_hours=session_hours)
    if limits.get("max_calls_per_session") and calls > limits["max_calls_per_session"]:
        out = {"status":"limit_exceeded","tool":a.tool,"reason_human":"เกินเพดานจำนวนครั้งต่อรอบงาน หยุดเพื่อกันค่าใช้จ่ายบาน",
               "calls_used":calls,"ledger_written":False}
        print(json.dumps(out, ensure_ascii=False)); sys.exit(50)

    # เพดานแยกของสมองพิเศษ (fable แพงสุด · นับต่างหาก) · เกินเพดาน = ไม่หยุด แต่สลับไปสมองสำรอง (opus) อัตโนมัติ
    is_premium = bool(adapters.get(a.tool, {}).get("premium"))
    premium_over_cap = False
    if is_premium:
        fable_calls = bump_counter(cwd, ".session-fable-calls", session_hours=session_hours)
        if limits.get("max_fable_calls_per_session") and fable_calls > limits["max_fable_calls_per_session"]:
            premium_over_cap = True  # เกินเพดาน fable → ตัดออกจากสาย ใช้ opus แทน

    # เพดานรอบแก้ต่อ issue (นับข้ามทุก coder · ไม่รีเซ็ตตอนสลับ) · สมองพิเศษไม่กินรอบ coder
    safe_task = re.sub(r"[^A-Za-z0-9._-]", "_", a.task_id)
    rounds = 0
    if not is_brain:
        rounds = bump_counter(cwd, f".rounds-{safe_task}", session_hours=session_hours)
        if limits.get("max_rounds_per_issue") and rounds > limits["max_rounds_per_issue"]:
            out = {"status":"limit_exceeded","tool":a.tool,
                   "reason_human":f"issue {a.task_id} แก้เกิน {limits['max_rounds_per_issue']} รอบแล้ว หยุดเพื่อกันวนไหม้เงิน ให้ยกขึ้นสมองคิดใหม่หรือถามเจ้าของ",
                   "rounds_used":rounds,"calls_used":calls,"ledger_written":False}
            print(json.dumps(out, ensure_ascii=False)); sys.exit(50)

    # สายลำดับ:
    #  - สมอง (brain): fable → สมองสำรอง (opus) · พัง/ชนเพดาน = ตกไป opus ไม่ไหลลง coder
    #  - coder: มีสายสำรองของ coder เหมือนเดิม
    prerotated = ""
    if is_brain:
        brain_fallback = [t for t in (accounts.get("fallback", {}).get("brain") or ["opus"]) if t != a.tool]
        chain = [a.tool] + brain_fallback
        chain = [t for t in chain if t in adapters and adapters[t].get("brain")]
        if premium_over_cap:
            # fable เกินเพดาน → เอา premium ออกจากสาย เหลือแต่สมองสำรอง (opus)
            chain = [t for t in chain if not adapters[t].get("premium")]
            prerotated = a.tool
            if not chain:
                out = {"status":"limit_exceeded","tool":a.tool,
                       "reason_human":f"เกินเพดานเรียกสมองพิเศษ ({limits['max_fable_calls_per_session']} ครั้ง/รอบงาน) และไม่มีสมองสำรอง (opus) ตั้งไว้ · เพิ่ม opus ใน adapters หรือถามเจ้าของ",
                       "calls_used":calls,"ledger_written":False}
                print(json.dumps(out, ensure_ascii=False)); sys.exit(50)
    else:
        chain = [a.tool] + [t for t in accounts.get("fallback",{}).get("code_writing",[]) if t != a.tool]
        chain = [t for t in chain if t in adapters and not adapters[t].get("brain")]

    # ข้ามตัวที่ติด cooldown (พังซ้ำเมื่อกี้ ยังไม่ครบเวลาพัก)
    now = time.time()
    cooldown_state = load_cooldown(cwd)
    tried = [f"{t}:cooldown-skip" for t in chain if in_cooldown(cooldown_state, t, now)]
    chain = [t for t in chain if not in_cooldown(cooldown_state, t, now)]
    if not chain:
        print(json.dumps({"status":"crash","tool":a.tool,
            "reason_human":"ทุกตัวในสายติด cooldown (พังซ้ำเมื่อสักครู่) รอครบเวลาพักหรือถามเจ้าของ",
            "tried":tried}, ensure_ascii=False)); sys.exit(40)

    def attempt_row(tool, st, rotated_from, output_ref=""):
        return write_ledger(cwd, {"timestamp":datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "issue_id":a.task_id,"tool":tool,"account_used":tool,"rotated_from":rotated_from,
            "status":st,"calls_used":calls,"output_ref":output_ref})

    rotated_from = prerotated
    if prerotated:
        # จดว่า fable ถูก "ข้ามเพราะเกินเพดาน" (ไม่ใช่เรียกจริง) แล้วสลับไปสมองสำรอง · relay-report ไม่นับเป็นการเรียก
        attempt_row(prerotated, "skipped_by_cap", "")
    for tool in chain:
        # ด่านบัญชีโปรแกรมอนุญาต: กันไฟล์ adapters.yaml ใน worktree ถูกแก้ให้รันคำสั่งอันตราย
        bin_name = Path(str((adapters[tool].get("cmd") or ["?"])[0])).name
        if bin_name not in allowed_bins():
            tried.append(f"{tool}:blocked-bin")
            attempt_row(tool, "blocked-bin", rotated_from)
            rotated_from = tool
            continue
        model = models.get("code") if tool == "ollama" else ""
        relay_now("set", tool, a.task_id, "กำลังเขียนโค้ด")
        code, out, err = run_once(adapters[tool], prompt, cwd, model)
        st = classify(code, out, err)
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
                "rounds_used":rounds,"tried":tried}, ensure_ascii=False))
            sys.exit(0)

        # ไม่ ok → จดความพยายามลง ledger ก่อน (relay-report จะได้เห็นครั้งที่พัง/ชนโควต้าด้วย) แล้วค่อยตัดสินว่าสลับหรือหยุด
        attempt_row(tool, st, rotated_from)
        if st in ("quota", "crash"):
            record_fail(cwd, tool, cd_cfg, time.time())  # พังซ้ำครบเกณฑ์ = พักตัวนี้ชั่วคราว
            rotated_from = tool
            continue   # สลับตัวถัดไปในสาย
        if st == "auth":
            # สมอง (brain) ตัวหน้าล็อกอินหลุด + ยังมีสมองสำรองในสาย → สลับไปตัวถัดไป (fable หลุด → opus)
            if is_brain and tool != chain[-1]:
                rotated_from = tool
                continue
            relay_now("clear")
            print(json.dumps({"status":"auth","tool":tool,"reason_human":REASON["auth"],
                "hint":f"ล็อกอินใหม่: {tool} login","tried":tried,"ledger_written":True}, ensure_ascii=False)); sys.exit(20)
        if st == "not_found":
            rotated_from = tool
            continue   # ไม่มีตัวนี้ → ลองตัวถัดไป

    relay_now("clear")
    print(json.dumps({"status":"crash","reason_human":"ลองครบทุกตัวในสายแล้วยังไม่สำเร็จ","tried":tried},
                     ensure_ascii=False))
    sys.exit(40)

if __name__ == "__main__":
    main()
