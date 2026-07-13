#!/usr/bin/env python3
"""relay-call — เรียก AI coder 1 ครั้ง · จับผลตายตัว · สลับบัญชี · นับงบ · เขียน ledger

ส่วนของ Use AI Relay (Memory Schema v1.1) · LLM อ่านแค่ช่อง status ที่ตัวนี้คืน ไม่ parse stderr เอง
ใช้:  python relay-call.py --tool grok --task-id P1-I2 --prompt-file brief.md --cwd <worktree>
คืน: JSON บรรทัดเดียว + exit (ok=0 not_found=10 auth=20 quota=30 crash=40 limit_exceeded=50 off_plan=60)

หมายเหตุ: คำสั่งเรียก coder อ่านจาก .hermes/ai-relay/adapters.yaml
          บัญชี/สาย/เพดาน อ่านจาก .hermes/ai-relay/accounts.yaml
          ถ้าไม่มีไฟล์ ใช้ค่าปริยายในตัว (รองรับ ollama ได้ทันทีเพื่อทดสอบ)
สมอง (brain) = Opus 4.8 ตัวเดียว (--tool opus) · Fable ถอดออกแล้ว (เจ้าของสั่ง 2026-07-06)
เพดานรอบต่อ issue นับข้าม coder · cooldown ตัวที่พังซ้ำ · อ่าน YAML ได้แม้ไม่มี PyYAML (ตัวอ่านสำรองในตัว)
"""
import argparse, glob, json, os, re, shutil, signal, socket, subprocess, sys, threading, time
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
ALLOWED_BINS = {"grok", "gemini", "ollama", "codex", "claude", "relay-portal"}
def allowed_bins():
    extra = os.environ.get("RELAY_EXTRA_BINS", "")
    return ALLOWED_BINS | {b for b in extra.split(":") if b}

# หมายเหตุ: สมองพิเศษเดิมถูกถอดออกจาก Use AI Relay แล้ว (เจ้าของสั่ง 2026-07-06) เพื่อประหยัดโควต้า
# สมองหลัก (คิด/วิเคราะห์/วางแผน/ตรวจ) = Opus 4.8 ตัวเดียว

# ---- ค่าปริยาย (ใช้เมื่อไม่มีไฟล์ตั้งค่า) ----
DEFAULT_ADAPTERS = {
    "grok":   {"cmd": ["relay-portal","grok","--prompt","{prompt}"]},
    "gemini": {"cmd": ["gemini","-p","{prompt}","-m","gemini-2.5-flash","--skip-trust","--approval-mode","yolo","--output-format","text"], "run_in_cwd": True},
    "ollama": {"cmd": ["ollama","run","{model}","{prompt}"], "run_in_cwd": True},
    "codex":  {"cmd": ["relay-portal","codex","--prompt","{prompt}"], "run_in_cwd": True},
    # สมองหลัก (brain) · คิด/วิเคราะห์/วางแผน/ตรวจ/ตัดสิน · Opus 4.8 ตัวเดียว (Fable ถอดออกแล้ว)
    "opus":   {"cmd": ["relay-portal","claude","--model","claude-opus-4-8","--prompt","{prompt}"], "run_in_cwd": True, "brain": True},
}
DEFAULT_ACCOUNTS = {
    "fallback": {"code_writing": ["grok","codex","gemini","ollama"],
                 # สายสมอง: สมองหลัก = opus 4.8 (ถ้ามีสมองสำรองในอนาคต เติมต่อท้ายได้)
                 "brain": ["opus"]},
    "limits": {"max_rounds_per_issue": 3, "max_calls_per_session": 50,
               "session_hours": 12, "budget": None,
               # นาฬิกาปลุกต่อการเรียก 1 ครั้ง (วินาที) · เกินเวลา = ถือว่า "ค้าง" ตัดทิ้งแล้วสลับตัวถัดไป
               # ปรับได้ใน accounts.yaml (limits.call_timeout_seconds) หรือต่อ tool ใน adapters.yaml (timeout)
               # coder = 900 (15 นาที) · สมอง (brain: opus) คิดนานกว่า = 1800 (30 นาที) แยกกัน
               "call_timeout_seconds": 900, "brain_call_timeout_seconds": 1800,
               # ตัวจับเงียบ — ไม่มี output ใหม่เกินนี้ = ถือว่าค้าง ตัดเลย ไม่รอครบนาฬิกาใหญ่
               "silence_timeout_seconds": 180},
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

def _registry_list(v):
    # load_yaml มีตัวอ่านสำรองเวลาไม่มี PyYAML; ตรงนี้แปลง list แบบ [a, b]
    # ให้ผู้ใช้ได้ค่าเหมือนกันทั้งเครื่องที่มีและไม่มี PyYAML
    if isinstance(v, list):
        return v
    if isinstance(v, tuple):
        return list(v)
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            if not inner:
                return []
            return [_scalar(part.strip()) for part in inner.split(",")]
    return v

def _as_registry_bool(v):
    # กัน enabled: "true"/"false" (มี quote) หายเงียบ · คืน bool ถ้าตีความได้ ไม่งั้นคืนค่าเดิม
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        low = v.strip().lower()
        if low == "true":  return True
        if low == "false": return False
    return v

def _normalize_registry_ai(ai):
    if not isinstance(ai, dict):
        return {}
    reg = {}
    for name, meta in ai.items():
        if not isinstance(meta, dict):
            continue
        item = dict(meta)
        for key in ("roles", "good_for"):
            if key in item:
                v = _registry_list(item[key])
                # ค่าเดี่ยว (เช่น roles: coder) → ห่อเป็น list · None → [] · กันโค้ดถัดไปวนตัวอักษรทีละตัว
                if v is None:
                    item[key] = []
                elif not isinstance(v, list):
                    item[key] = [v]
                else:
                    item[key] = v
        if "enabled" in item:
            item["enabled"] = _as_registry_bool(item["enabled"])
        reg[str(name)] = item
    return reg

def load_registry(cwd):
    # ทะเบียนเป็นชั้นข้อมูลเสริมให้ doctor/suggest อ่านรอบหน้า ยังไม่เอาไปขับการเรียก AI ตอนนี้
    cwd = Path(cwd)
    local = cfg_dir(cwd)/"registry.yaml"
    source = local if local.exists() else SCRIPT_DIR/"references"/"registry.example.yaml"
    if not source.exists():
        return {}
    data = load_yaml(source)
    if not isinstance(data, dict):
        return {}
    return _normalize_registry_ai(data.get("ai"))

def registry_enabled(reg):
    return {name: meta for name, meta in (reg or {}).items()
            if isinstance(meta, dict) and meta.get("enabled") is True}

def registry_vendor(reg, tool):
    meta = (reg or {}).get(tool)
    if not isinstance(meta, dict):
        return None
    return meta.get("vendor")

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
QUOTA_RE = re.compile(r"\b429\b|rate.?limit|quota|too many request|usage limit|session limit|insufficient|hit your limit", re.I)
# ข้อความ auth ที่ชัดเจนมาก (คำตอบปกติไม่มีทางพูด) · ถือเป็นล็อกอินพัง แม้ CLI จะ exit 0 · กันเอา error มาเป็นคำตอบ
STRONG_AUTH_RE = re.compile(r"you are not authenticated|organization has disabled|subscription access for claude|use an anthropic api key instead", re.I)

# บรรทัด log ปนใน stderr ที่ "ไม่ใช่" ข้อความ error ของ CLI เอง — ต้องตัดทิ้งก่อนตีความ auth/quota
# (hook ของ session ฉีดความจำ/กฎที่ "พูดถึง" เรื่อง auth ได้ · MCP ปลั๊กอินเสริมพ่น AuthRequired ของมันเอง
#  ทั้งที่งานหลักสำเร็จ — เคสจริง 2026-07-07: mcp.cloudflare.com token หมด แต่ codex ตอบงานปกติ)
# หมายเหตุ (GPT-5 review): "warning:" เฉยๆ กว้างไป (จะกลืน "warning: not authenticated" จริง)
# → จำกัดเป็นเฉพาะ warning เรื่อง skill-budget ของ codex ที่เจอจริงเท่านั้น
NOISE_LINE_RE = re.compile(r"^\s*(hook:|mcp:|codex$|tokens used|warning: skill descriptions)|ERROR\s+rmcp::|AuthRequired\(AuthRequiredError", re.I)
def _clean_stream(text):
    return "\n".join(l for l in (text or "").splitlines() if not NOISE_LINE_RE.search(l))

# ---- จัดประเภท error จาก exit code + stdout/stderr (เป็นที่เดียวที่ตีความ) ----
def classify(exit_code, stdout, stderr):
    out = stdout or ""
    err = stderr or ""
    err_low = err.lower()
    if exit_code == 124 and TIMEOUT_MARK in err:
        return "timeout"   # ค้างจริง (run_once ตั้งป้ายนี้ตอน subprocess เกินนาฬิกาปลุก · CLI ที่บังเอิญ exit 124 เอง ไม่เข้าเงื่อนไขนี้ → ตกไป crash)
    if exit_code == 127:
        return "not_found"
    out_clean = _clean_stream(out)
    err_clean = _clean_stream(err)
    if exit_code == 0:
        # บาง CLI (เช่น claude เมื่อ org ปิดสิทธิ์) พิมพ์ error ยาวแต่ exit 0 · กันเอา error มาเป็นคำตอบ
        # แต่ "งานสำเร็จที่เนื้อหาพูดถึงเรื่อง auth" (เช่นแก้โค้ด/เอกสารระบบ login) ต้องไม่โดนจับ:
        #  - stdout ยาว (>250) = คำตอบงานจริง · error จริงของ CLI สั้น (~110 ตัว) → ไม่ตรวจ STRONG บน stdout ยาว
        #  - stderr ตรวจได้เสมอ แต่หลังตัดบรรทัด log ปน (hook:/MCP) แล้วเท่านั้น
        # เคสจริง 2026-07-05: codex ทำงาน P2/P3 เสร็จ แต่สรุปงานมีคำ "organization has disabled ..."
        # → โดนตีเป็น auth ปลอม 4 ครั้ง ทั้งที่ login ติดปกติ
        # รูที่ 2 (DEC-036 · 2026-07-11): stderr ต้องมีตัวกันความยาวแบบเดียวกับ stdout ด้วย
        #  codex CLI สะท้อนบทสนทนาทั้งหมด (prompt+คำตอบ) ลง stderr → งาน P14 ทั้งเฟสพูดเรื่อง
        #  rate limit/quota ของ API วัดผล ทำให้ err_clean ยาวเต็มไปด้วยคำต้องห้าม → quota/auth ปลอม
        #  error จริงของ CLI สั้น (~110 ตัว ≤250) ยังจับได้ · บทสนทนายาว (>250) ไม่โดนตีปลอม
        if (len(out_clean.strip()) <= 250 and STRONG_AUTH_RE.search(out_clean)) or (len(err_clean.strip()) <= 250 and STRONG_AUTH_RE.search(err_clean)):
            return "auth"
        # quota ใช้ตัวกันความยาวแบบเดียวกับ auth — เคสจริง 2026-07-10 (QAQC review):
        # งานรีวิวที่ "เนื้อหาพูดถึง" quota/rate limit (เช่นตรวจดีไซน์หมวด Quota / Rate-limit)
        # ตอบยาวปกติ exit 0 แต่โดนตีเป็น quota ปลอม → codex/gemini ถูกทิ้งคำตอบทั้งที่ทำงานสำเร็จ
        if (len(out_clean.strip()) <= 250 and QUOTA_RE.search(out_clean)) or (len(err_clean.strip()) <= 250 and QUOTA_RE.search(err_clean)):
            return "quota"
        if len(out_clean.strip()) < 40 and AUTH_RE.search(err_clean):
            return "auth"
        return "ok"
    if "command not found" in err_low or "no such file" in err_low:
        return "not_found"
    # exit != 0: ตรวจบน stream ที่ตัด log ปน (hook:/MCP) แล้ว — เหตุผลเดียวกับฝั่ง exit 0
    if AUTH_RE.search(err_clean):
        return "auth"
    if QUOTA_RE.search(err_clean):
        return "quota"
    if AUTH_RE.search(out_clean):
        return "auth"
    # ตัวกันความยาวเดียวกับฝั่ง exit 0 — คำตอบยาวที่ "พูดถึง" quota แล้วพังกลางทาง = crash ไม่ใช่ quota
    if len(out_clean.strip()) <= 250 and QUOTA_RE.search(out_clean):
        return "quota"
    return "crash"

REASON = {
    "ok": "เรียกสำเร็จ",
    "not_found": "ไม่มีโปรแกรม AI ตัวนี้บนเครื่องนี้ (ติดตั้งก่อน หรือสลับตัวอื่น)",
    "auth": "บัญชีหลุด/ยังไม่ล็อกอิน ต้องล็อกอินใหม่",
    "quota": "บัญชีนี้เกินโควต้า สลับบัญชีถัดไป",
    "crash": "AI ตอบไม่ได้/พัง ลองซ้ำหรือสลับตัวสำรอง",
}

def summarize_final_failure(tried):
    statuses = [item.rsplit(":", 1)[-1] for item in tried if ":" in item]
    if statuses and all(s in {"quota", "not_found", "cooldown-skip"} for s in statuses) and "quota" in statuses:
        return "quota", REASON["quota"], 30
    if statuses and all(s in {"not_found", "cooldown-skip"} for s in statuses):
        return "not_found", REASON["not_found"], 10
    return "crash", "ลองครบทุกตัวในสายแล้วยังไม่สำเร็จ", 40

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

def prefer_portal_adapters(adapters):
    # Employee machines should use AI Portal by default. Old per-project
    # adapters.yaml files may still call local claude/grok/codex and then fail
    # with "Not logged in". Allow local CLI only when explicitly requested.
    if os.environ.get("AI_RELAY_ALLOW_LOCAL_CLI") == "1":
        return adapters
    for tool in ("opus", "codex", "grok"):
        spec = adapters.get(tool) or {}
        cmd = spec.get("cmd") or []
        bin_name = Path(str(cmd[0])).name if cmd else ""
        if bin_name in {"claude", "codex", "grok"}:
            adapters[tool] = dict(DEFAULT_ADAPTERS[tool])
    return adapters

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

TIMEOUT_MARK = "__relay_timeout__"   # ป้ายเฉพาะบอกว่า "ค้างจริง" (กันสับสนกับ CLI ที่บังเอิญ exit 124)

def resolve_timeout(spec, limits):
    # นาฬิกาปลุกต่อการเรียก · ต่อ tool (adapters.yaml: timeout) ชนะก่อน → ค่ากลาง (accounts.yaml) → ค่าปริยาย
    # สมอง (brain: opus) คิดนานกว่า coder → ใช้ค่ากลาง+ปริยายของสมองแยก (ไม่ให้โดนตัดเร็วเกิน)
    limits = limits or {}
    is_brain = bool(spec.get("brain"))
    fallback = 1800 if is_brain else 900
    mid_key = "brain_call_timeout_seconds" if is_brain else "call_timeout_seconds"
    for v in (spec.get("timeout"), limits.get(mid_key), fallback):
        try:
            if v is not None and int(v) > 0:
                return int(v)
        except (TypeError, ValueError):
            pass
    return fallback

def resolve_silence(spec, limits):
    # ตัวจับเงียบต่อ coder · ต่อ tool ชนะก่อน → ค่ากลาง → ปริยาย 180
    # สมอง (brain) อาจคิดเงียบนานได้จริง จึงปิดไว้ เว้นแต่ตั้งต่อ tool ชัดเจน
    limits = limits or {}
    if "silence_timeout" in spec:
        v = spec.get("silence_timeout")
    elif spec.get("brain"):
        return None
    elif "silence_timeout_seconds" in limits:
        v = limits.get("silence_timeout_seconds")
    else:
        v = 180
    try:
        v = int(v)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None

def _kill_process_group(p):
    # timeout ต้องฆ่าทั้งกลุ่ม ไม่ใช่ฆ่าแค่ shell/coder ตัวหน้าแล้วปล่อยลูกหลานค้างเครื่อง
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except Exception:
        try: p.kill()
        except Exception: pass
    try:
        p.wait(timeout=3)
        return
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        return
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception:
        try: p.kill()
        except Exception: pass
    try: p.wait(timeout=1)
    except Exception: pass

def run_once(spec, prompt, cwd, model, timeout=900, silence_timeout=None):
    cmd = [a.replace("{prompt}",prompt).replace("{cwd}",str(cwd)).replace("{model}",model or "") for a in spec["cmd"]]
    workdir = str(cwd) if spec.get("run_in_cwd") else None
    env = os.environ.copy()
    if cmd and Path(cmd[0]).name == "claude":
        # ตัด token org ที่ใช้ไม่ได้ เพื่อให้ claude ใช้ login ของเครื่องแทน (จับ path เต็มด้วย)
        env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
    try:
        # ไบนารีล้วน (ไม่ text=True) · อ่านเป็น chunk ด้วย read1 ให้ last_output ขยับทุกครั้งที่มี byte จริง
        # แม้ coder พ่นบรรทัดยาวไม่มีขึ้นบรรทัดใหม่ (เช่น Grok output json ก้อนเดียว) ก็ไม่ถูกนับว่า "เงียบ"
        p = subprocess.Popen(cmd, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             stdin=subprocess.DEVNULL, env=env,
                             start_new_session=True)  # ปิด stdin · กัน codex exec ค้างรออ่าน input
    except FileNotFoundError:
        return 127, "", "command not found"

    out_buf, err_buf = [], []
    start = time.monotonic()
    last_output = [start]

    def reader(pipe, buf):
        try:
            while True:
                chunk = pipe.read1(65536) if hasattr(pipe, "read1") else pipe.read(65536)
                if not chunk:
                    break
                buf.append(chunk)
                last_output[0] = time.monotonic()  # มี byte ใหม่ = ยังไม่เงียบ (ไม่รอขึ้นบรรทัดใหม่)
        finally:
            try: pipe.close()
            except Exception: pass

    threads = [
        threading.Thread(target=reader, args=(p.stdout, out_buf), daemon=True),
        threading.Thread(target=reader, args=(p.stderr, err_buf), daemon=True),
    ]
    for t in threads: t.start()

    def _dec(buf):  # รวม chunk ไบนารี → ข้อความ (กันอักขระหลายไบต์ขาดกลาง ด้วยการ decode ทีเดียวตอนจบ)
        joined = b"".join(b if isinstance(b, bytes) else str(b).encode("utf-8", "replace") for b in buf)
        return joined.decode("utf-8", "replace")

    while True:
        rc = p.poll()
        if rc is not None:
            for t in threads: t.join(timeout=1)
            return rc, _dec(out_buf), _dec(err_buf)
        now = time.monotonic()
        if now - start > timeout:
            _kill_process_group(p)
            for t in threads: t.join(timeout=1)
            # เก็บ stderr เดิมไว้ + ต่อป้าย TIMEOUT_MARK (ไม่ทิ้ง log วินิจฉัย · classify ยังจับ timeout ได้)
            return 124, _dec(out_buf), _dec(err_buf) + TIMEOUT_MARK
        if silence_timeout and now - last_output[0] > silence_timeout:
            _kill_process_group(p)
            for t in threads: t.join(timeout=1)
            return 124, _dec(out_buf), _dec(err_buf) + TIMEOUT_MARK + ":silence"
        time.sleep(0.5)

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

OFF_PLAN_REASON = (
    "เลขงานไม่อยู่ในแผน .project/plan.md — ตรวจแผนก่อน "
    "หรือใช้ --no-plan สำหรับงานจรนอกแผน"
)

def check_plan_anchor(cwd: Path, task_id: str, no_plan: bool) -> tuple[str, str | None]:
    """คืน (action, plan_check) · action = proceed|off_plan · plan_check = error เมื่อ anchor พัง"""
    if no_plan:
        return "proceed", None
    anchor = SCRIPT_DIR / "plan-anchor.py"
    if not anchor.exists():
        return "proceed", None
    plan_path = cwd / ".project" / "plan.md"
    if not plan_path.exists():
        return "proceed", None
    try:
        proc = subprocess.run(
            [sys.executable, str(anchor), "--task-id", task_id, "--plan", str(plan_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError):
        return "proceed", "error"
    if proc.returncode == 0:
        return "proceed", None
    if proc.returncode == 1:
        return "off_plan", None
    if proc.returncode == 2:
        return "proceed", None
    return "proceed", "error"

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
    ap.add_argument(
        "--no-plan",
        action="store_true",
        help="ข้ามการตรวจแผน (จด [no-plan] ใน ledger)",
    )
    a = ap.parse_args()

    cwd = Path(a.cwd).expanduser().resolve()
    plan_check_error = None

    def with_plan_check(payload: dict) -> dict:
        if plan_check_error:
            payload = dict(payload)
            payload["plan_check"] = plan_check_error
        return payload

    def emit(payload: dict, code: int):
        print(json.dumps(with_plan_check(payload), ensure_ascii=False))
        sys.exit(code)
    load_relay_env(cwd)
    prompt = Path(a.prompt_file).read_text(encoding="utf-8") if Path(a.prompt_file).exists() else a.prompt_file
    adapters = {**DEFAULT_ADAPTERS, **(load_yaml(cfg_dir(cwd)/"adapters.yaml").get("tools") or {})}
    adapters = prefer_portal_adapters(adapters)
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

    plan_action, plan_check_error = check_plan_anchor(cwd, a.task_id, a.no_plan)

    ledger_issue_id = f"{a.task_id} [no-plan]" if a.no_plan else a.task_id

    if plan_action == "off_plan":
        write_ledger(cwd, {"timestamp":datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "issue_id":ledger_issue_id,"tool":a.tool,"account_used":a.tool,"rotated_from":"",
            "status":"off_plan","calls_used":0,"output_ref":""})
        emit({"status":"off_plan","tool":a.tool,"reason_human":OFF_PLAN_REASON,
               "task_id":a.task_id,"ledger_written":True}, 60)

    # เพดาน session
    calls = bump_calls(cwd, session_hours=session_hours)
    if limits.get("max_calls_per_session") and calls > limits["max_calls_per_session"]:
        emit({"status":"limit_exceeded","tool":a.tool,"reason_human":"เกินเพดานจำนวนครั้งต่อรอบงาน หยุดเพื่อกันค่าใช้จ่ายบาน",
               "calls_used":calls,"ledger_written":False}, 50)

    # (Fable ถอดออกแล้ว — ไม่มีสมองพิเศษ premium · สมองหลัก = Opus 4.8 ตัวเดียว)

    # เพดานรอบแก้ต่อ issue (นับข้ามทุก coder · ไม่รีเซ็ตตอนสลับ) · สมองไม่กินรอบ coder
    safe_task = re.sub(r"[^A-Za-z0-9._-]", "_", a.task_id)
    rounds = 0
    if not is_brain:
        rounds = bump_counter(cwd, f".rounds-{safe_task}", session_hours=session_hours)
        if limits.get("max_rounds_per_issue") and rounds > limits["max_rounds_per_issue"]:
            emit({"status":"limit_exceeded","tool":a.tool,
                   "reason_human":f"issue {a.task_id} แก้เกิน {limits['max_rounds_per_issue']} รอบแล้ว หยุดเพื่อกันวนไหม้เงิน ให้ยกขึ้นสมองคิดใหม่หรือถามเจ้าของ",
                   "rounds_used":rounds,"calls_used":calls,"ledger_written":False}, 50)

    # สายลำดับ:
    #  - สมอง (brain): opus + สมองสำรองในทะเบียน (ถ้ามี) · พัง = รายงานกลับ ไม่ไหลลง coder
    #  - coder: มีสายสำรองของ coder เหมือนเดิม
    if is_brain:
        brain_fallback = [t for t in (accounts.get("fallback", {}).get("brain") or ["opus"]) if t != a.tool]
        chain = [a.tool] + brain_fallback
        chain = [t for t in chain if t in adapters and adapters[t].get("brain")]
    else:
        chain = [a.tool] + [t for t in accounts.get("fallback",{}).get("code_writing",[]) if t != a.tool]
        chain = [t for t in chain if t in adapters and not adapters[t].get("brain")]

    # ข้ามตัวที่ติด cooldown (พังซ้ำเมื่อกี้ ยังไม่ครบเวลาพัก)
    now = time.time()
    cooldown_state = load_cooldown(cwd)
    tried = [f"{t}:cooldown-skip" for t in chain if in_cooldown(cooldown_state, t, now)]
    chain = [t for t in chain if not in_cooldown(cooldown_state, t, now)]
    if not chain:
        emit({"status":"crash","tool":a.tool,
            "reason_human":"ทุกตัวในสายติด cooldown (พังซ้ำเมื่อสักครู่) รอครบเวลาพักหรือถามเจ้าของ",
            "tried":tried}, 40)

    def attempt_row(tool, st, rotated_from, output_ref=""):
        return write_ledger(cwd, {"timestamp":datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "issue_id":ledger_issue_id,"tool":tool,"account_used":tool,"rotated_from":rotated_from,
            "status":st,"calls_used":calls,"output_ref":output_ref})

    rotated_from = ""
    for tool in chain:
        # ด่านบัญชีโปรแกรมอนุญาต: กันไฟล์ adapters.yaml ใน worktree ถูกแก้ให้รันคำสั่งอันตราย
        bin_name = Path(str((adapters[tool].get("cmd") or ["?"])[0])).name
        if bin_name not in allowed_bins():
            tried.append(f"{tool}:blocked-bin")
            attempt_row(tool, "blocked-bin", rotated_from)
            rotated_from = tool
            continue
        model = models.get("code") if tool == "ollama" else ""
        call_timeout = resolve_timeout(adapters[tool], limits)
        silence_timeout = resolve_silence(adapters[tool], limits)
        relay_now("set", tool, a.task_id, "กำลังเขียนโค้ด")
        code, out, err = run_once(adapters[tool], prompt, cwd, model, timeout=call_timeout, silence_timeout=silence_timeout)
        st = classify(code, out, err)
        tried.append(f"{tool}:{st}")

        if st == "ok":
            ofile = cfg_dir(cwd)/f"out-{re.sub(r'[^A-Za-z0-9]', '_', a.task_id)}.txt"
            ofile.write_text(redact(out), encoding="utf-8")
            relay_now("clear")
            ledger = write_ledger(cwd, {"timestamp":datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "issue_id":ledger_issue_id,"tool":tool,"account_used":tool,"rotated_from":rotated_from,
                "status":"ok","calls_used":calls,"output_ref":str(ofile)})
            emit({"status":"ok","tool":tool,"account_used":tool,"rotated_from":rotated_from,
                "reason_human":REASON["ok"],"output_ref":str(ofile),"ledger_written":True,"calls_used":calls,
                "rounds_used":rounds,"tried":tried}, 0)

        # ไม่ ok → จดความพยายามลง ledger ก่อน (relay-report จะได้เห็นครั้งที่พัง/ชนโควต้าด้วย) แล้วค่อยตัดสินว่าสลับหรือหยุด
        attempt_row(tool, st, rotated_from)
        if st in ("quota", "crash", "timeout"):
            # timeout = ค้าง/หมดเวลา · นับเป็นพังเข้า cooldown + สลับตัวถัดไปเหมือน crash
            record_fail(cwd, tool, cd_cfg, time.time())  # พังซ้ำครบเกณฑ์ = พักตัวนี้ชั่วคราว
            rotated_from = tool
            continue   # สลับตัวถัดไปในสาย
        if st == "auth":
            # สมอง (brain) ตัวหน้าล็อกอินหลุด + ยังมีสมองสำรองในสาย → สลับไปตัวถัดไป
            if is_brain and tool != chain[-1]:
                rotated_from = tool
                continue
            auth_hint = (
                "ให้แอดมินตรวจ AI Portal token ใน ~/.hermes/.env"
                if Path(str((adapters[tool].get("cmd") or [""])[0])).name == "relay-portal"
                else f"ล็อกอินใหม่: {tool} login"
            )
            relay_now("clear")
            emit({"status":"auth","tool":tool,"reason_human":REASON["auth"],
                "hint":auth_hint,"tried":tried,"ledger_written":True}, 20)
        if st == "not_found":
            rotated_from = tool
            continue   # ไม่มีตัวนี้ → ลองตัวถัดไป

    relay_now("clear")
    status, reason, exit_code = summarize_final_failure(tried)
    emit({"status":status,"reason_human":reason,"tried":tried}, exit_code)

if __name__ == "__main__":
    main()
