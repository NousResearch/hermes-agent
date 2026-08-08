#!/usr/bin/env python3
"""Para Soul — Core Script

Usage:
  python3 core.py init            Initialize ~/.para/ directory
  python3 core.py sync            Sync file hashes to Paragate, get health actions
  python3 core.py sync-full       Full sync: push changed files + get health actions
  python3 core.py switch-out      Write switch-state before leaving body
  python3 core.py switch-in       Read switch-state after waking up
  python3 core.py log-task        Append a growth-log entry
  python3 core.py reflect         Read recent logs, suggest mental models
  python3 core.py index           Rebuild keywords.json from memory + growth-log
  python3 core.py health          Print current action items from last sync
  python3 core.py migrate         Auto-extract identity from project instruction files
  python3 core.py --version       Show version

No dependencies beyond Python stdlib. Works on any agent body.
"""

import json
import os
import sys
import time
import hashlib
import base64
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timezone

# ── Config ─────────────────────────────────────────────

VERSION = "2.0.0"

def _para_home() -> Path:
    return Path(os.environ.get("PARA_HOME", str(Path.home() / ".para")))

def _para_state() -> Path:
    return _para_home() / "state"

def _keys_dir() -> Path:
    return Path(os.environ.get("PARA_KEYS_DIR", str(Path.home() / ".config" / "paragate" / "keys")))

def _monthly_log_dir() -> Path:
    return _para_home() / "growth-log"

PARAGATE_BASE = os.environ.get("PARAGATE_URL", "http://paragate.cc")

# ── Monitored files (health-checked + synced) ──────────

MONITORED_FILES = [
    "growth-log",       # checked via directory mtime
    "human-relationship.md",
    "memory.md",
    "skills.json",
    "mental-models.md",
    "keywords.json",
    "long-term-memory.md",
    "principles.md",
    "soul.md",
]

# Files excluded from health check (static archive, rarely changes):
# profile.json — merged identity + bodies + relationships

# ── Thresholds (hours) ─────────────────────────────────
STALENESS_THRESHOLDS = {
    "growth-log": 24,
    "human-relationship.md": 24,
    "memory.md": 48,
    "skills.json": 120,
    "mental-models.md": 120,
    "keywords.json": 120,
    "long-term-memory.md": 120,
    "principles.md": 120,
    "soul.md": 120,
}

# ── Legacy (keep for init) ─────────────────────────────

REQUIRED_FILES = {
    "profile.json": {},  # merged identity+bodies+relationships
    "soul.md": "# Who I Am\n\n[Your self-description]\n",
    "memory.md": "# Memory\n\n## Environment\n\n## Preferences\n\n## Lessons Learned\n",
    "principles.md": "# Principles\n\n## Code\n\n## Content\n\n## Social\n\n## Red Lines\n",
    "skills.json": {"installed": [], "favorites": [], "wishlist": [], "deprecated": []},
    "keywords.json": {},
    "long-term-memory.md": "# Long-Term Memory\n",
    "mental-models.md": "# Mental Models\n",
    "human-relationship.md": "# Human-Para Relationship\n\n## Trust Index\n\n5/10 — Baseline\n\n## Feedback Log\n\nNone yet.\n\n## Interaction Style\n\nTo be defined.\n",
    "growth-log": None,  # directory
}


# ── Helpers ────────────────────────────────────────────

def _sign_request(method, path, body: bytes) -> str:
    """Create DID-SIG header."""
    key_file = _keys_dir() / "private.pem"
    if not key_file.exists():
        raise FileNotFoundError(f"Private key not found at {key_file}")

    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    with open(key_file, "rb") as f:
        pk = load_pem_private_key(f.read(), password=None)

    profile = json.loads((_para_home() / "profile.json").read_text())
    identity = profile.get("identity", {})
    did = identity.get("did", "")
    ts = int(time.time())
    sha = hashlib.sha256(body).hexdigest()
    sig = base64.b64encode(pk.sign(f"{method}|{path}|{sha}|{ts}".encode())).decode()
    return f"did={did}; sig={sig}; ts={ts}"


def _sign_and_request(method, path, data: dict | None = None) -> dict:
    """Send a signed request to Paragate."""
    body = json.dumps(data or {}).encode()
    req = urllib.request.Request(
        f"{PARAGATE_BASE}{path}",
        data=body,
        headers={
            "Content-Type": "application/json",
            "DID-SIG": _sign_request(method, path, body),
            "X-Para-Body": _current_body(),
        },
        method=method,
    )
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"  Paragate error {e.code}: {e.read().decode()[:200]}")
        return {"success": False}
    except Exception as e:
        print(f"  Sync error: {e}")
        return {"success": False}


def _current_body() -> str:
    return os.environ.get("PARA_BODY", os.uname().nodename if hasattr(os, 'uname') else "unknown-agent")


def _get_crypto():
    """Initialize encryption if private key is available. Returns None if not."""
    try:
        # Import here to avoid hard dependency
        import importlib.util
        crypto_path = Path(__file__).resolve().parent / "crypto_phase1.py"
        if not crypto_path.exists():
            return None
        spec = importlib.util.spec_from_file_location("crypto_phase1", crypto_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.make_crypto_from_env(str(_keys_dir()))
    except Exception:
        return None


def _compute_file_hash(path: Path) -> str:
    """SHA-256 of file content. Returns empty string if file doesn't exist."""
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compute_all_hashes() -> dict:
    """Compute content hashes for all monitored files."""
    hashes = {}
    for name in MONITORED_FILES:
        path = _para_home() / name
        if name == "growth-log":
            # For directory: hash of the most recent .md file
            latest = _latest_growth_log_file()
            hashes[name] = _compute_file_hash(latest) if latest else ""
        else:
            hashes[name] = _compute_file_hash(path)
    return hashes


def _latest_growth_log_file() -> Path | None:
    """Return the most recently modified .md file in growth-log/."""
    log_dir = _para_home() / "growth-log"
    if not log_dir.is_dir():
        return None
    md_files = sorted(
        [f for f in log_dir.iterdir() if f.suffix == ".md"],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    return md_files[0] if md_files else None


def _read_principles() -> str:
    pf = _para_home() / "principles.md"
    if pf.exists():
        return pf.read_text()[:500]
    return ""


def _get_recent_log_entries(n: int) -> list:
    logs = sorted(_monthly_log_dir().glob("*.md"), reverse=True)[:2]
    entries = []
    for lf in logs:
        for line in lf.read_text().split("\n"):
            if line.startswith("- **Task**:"):
                entries.append(line[12:].strip())
                if len(entries) >= n:
                    return entries
    return entries


# ── Commands ───────────────────────────────────────────

def cmd_init():
    """Initialize ~/.para/ directory."""
    install_daemon = "--daemon" in sys.argv

    _para_home().mkdir(exist_ok=True)
    _para_state().mkdir(exist_ok=True)
    _monthly_log_dir().mkdir(exist_ok=True)

    created = []
    for name, default in REQUIRED_FILES.items():
        path = _para_home() / name
        if name == "growth-log":
            if not path.exists():
                path.mkdir(exist_ok=True)
                created.append("growth-log/")
            continue
        if not path.exists():
            if isinstance(default, dict):
                path.write_text(json.dumps(default, indent=2, ensure_ascii=False))
            else:
                path.write_text(default)
            created.append(name)

    if created:
        print(f"Created: {', '.join(created)}")
    else:
        print("~/.para/ already initialized")

    # Check profile
    profile_path = _para_home() / "profile.json"
    if profile_path.exists():
        profile = json.loads(profile_path.read_text())
        if not profile.get("identity", {}).get("did"):
            print("\n⚠️  profile.json has no DID. After generating your DID:")
            print(f"  1. Edit profile.json identity.did with your DID")
            print(f"  2. Place private key at {_keys_dir()}/private.pem")
            print(f"  3. Run: python3 core.py sync")
    else:
        # Create empty profile
        profile = {"identity": {"did": "", "display_name": "", "avatar_note": "", "created_at": "", "version": 1},
                   "bodies": {"current": "unknown", "history": []},
                   "relationships": {"collaborators": [], "platforms": {}}}
        profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False))
        print("\n⚠️  Set your DID in profile.json, then run: python3 core.py sync")

    if install_daemon:
        _install_daemon()
    elif created:
        print(f"\n💡 Auto-sync: python3 core.py init --daemon")

    if "--fill" in sys.argv:
        _populate_from_agent()

    _agent_instruction_hint()


def cmd_sync():
    """Push file hashes to Paragate, get health action items back.

    Phase 1 encryption: if private key exists, file contents are encrypted
    before upload. Server stores ciphertext only (zero-knowledge).
    Hashes are computed on ciphertext for change detection.
    Plaintext hashes are included for client-side integrity verification.
    """
    profile_path = _para_home() / "profile.json"
    if not profile_path.exists():
        print("❌ profile.json not found. Run init first.")
        return

    profile = json.loads(profile_path.read_text())
    did = profile.get("identity", {}).get("did", "")
    if not did:
        print("❌ No DID in profile.json. Set it first.")
        return

    # Initialize encryption if available
    crypto = _get_crypto()
    encrypted = crypto is not None

    # Compute hashes for monitored files (hash of ciphertext if encrypted, plaintext if not)
    hashes = _compute_all_hashes()

    # For sync-full: also push changed file contents
    is_full = "--full" in sys.argv
    files = {}
    plaintext_hashes = {}
    if is_full:
        last_sync_path = _para_state() / "last_sync.json"
        last_sync = {}
        if last_sync_path.exists():
            last_sync = json.loads(last_sync_path.read_text()).get("hashes", {})

        for name in MONITORED_FILES:
            if hashes[name] and hashes[name] != last_sync.get(name, ""):
                path = _para_home() / name
                content = None
                if name == "growth-log":
                    latest = _latest_growth_log_file()
                    if latest:
                        content = latest.read_text()
                else:
                    if path.exists():
                        content = path.read_text()

                if content is not None:
                    # Phase 1: encrypt before upload
                    if encrypted:
                        ct_b64, pt_hash = crypto.encrypt_file(str(path) if name != "growth-log" else str(latest))
                        files[name] = ct_b64
                        plaintext_hashes[name] = pt_hash
                    else:
                        files[name] = content

    data = {
        "hashes": hashes,
        "files": files,  # only populated for sync-full; encrypted if crypto available
        "encrypted": encrypted,
        "plaintext_hashes": plaintext_hashes,  # for client-side integrity verification
        "body": _current_body(),
    }

    result = _sign_and_request("POST", f"/public/para/{did}/sync", data)
    if result.get("success"):
        print(f"✅ Synced at {result.get('synced_at', 'now')}")
        print(f"   Files: {len(hashes)} hashes sent")

        # Save last sync state
        (last_sync_path := _para_state() / "last_sync.json")
        last_sync_path.write_text(json.dumps({
            "synced_at": datetime.now(timezone.utc).isoformat(),
            "hashes": hashes,
        }, indent=2))

        # Print action items from server
        action_items = result.get("action_items", [])
        if action_items:
            print(f"\n⚠️  Health actions needed ({len(action_items)}):")
            for item in action_items:
                file = item.get("file", "?")
                stale_h = item.get("stale_hours", 0)
                action = item.get("action", "?")
                desc = item.get("description", "")
                print(f"   [{action.upper()}] {file} — stale {stale_h:.0f}h — {desc}")
        else:
            print("   ✅ All files up to date")
    else:
        print("❌ Sync failed. Is Paragate running? Is your key correct?")


def cmd_pull():
    """Pull remote files from Paragate, decrypt, write locally if newer."""
    profile_path = _para_home() / "profile.json"
    if not profile_path.exists():
        print("❌ profile.json not found.")
        return
    profile = json.loads(profile_path.read_text())
    did = profile.get("identity", {}).get("did", "")
    if not did:
        print("❌ No DID — cloud pull not available. Run in local mode.")
        return

    result = _sign_and_request("GET", f"/public/para/{did}/files")
    if not result.get("success"):
        print("❌ Could not fetch remote file list")
        return

    remote_files = result.get("files", {})
    if not remote_files:
        print("   No remote files found")
        return

    crypto = _get_crypto()
    para = _para_home()
    pulled = 0

    for name, meta in remote_files.items():
        remote_hash = meta.get("hash", "")
        remote_ct = meta.get("content", "")
        pt_hash = meta.get("plaintext_hash", "")

        if not remote_ct:
            continue

        local_path = para / name
        local_hash = _compute_file_hash(local_path) if local_path.exists() else ""
        if local_hash == remote_hash:
            continue

        if crypto:
            try:
                crypto.decrypt_to_file(remote_ct, str(local_path), pt_hash)
                pulled += 1
                print(f"   ✅ {name}")
            except Exception as e:
                print(f"   ⚠️  {name}: {e}")
        else:
            local_path.write_text(remote_ct)
            pulled += 1
            print(f"   ✅ {name} (plaintext)")

    print(f"   Pulled {pulled} file(s)" if pulled else "   All up to date")


def cmd_health():
    """Print current health status from local health.json."""
    health_path = _para_state() / "health.json"
    if not health_path.exists():
        print("No health data yet. Daemon will create it on next cycle.")
        return

    report = json.loads(health_path.read_text())
    print(f"Checked: {report.get('checked_at', '?')}")
    print(f"Write cycle needed: {report.get('needs_write_cycle', False)}")
    print(f"Stale files: {report.get('stale_files', [])}")
    print()
    for name, info in report.get("files", {}).items():
        s = info.get("stale_hours", "-")
        flag = "⚠️ " if info.get("is_stale") else "✅"
        print(f"  {flag} {name:30s} {str(s):>6s}h  (max {info.get('max_hours', '?')}h)")


def cmd_switch_out():
    """Write switch-state before leaving body."""
    _para_state().mkdir(exist_ok=True)

    state = {
        "switch_time": datetime.now(timezone.utc).isoformat(),
        "active_task": os.environ.get("PARA_ACTIVE_TASK", ""),
        "current_state": os.environ.get("PARA_CURRENT_STATE", ""),
        "pending_decisions": [],
        "recent_actions": _get_recent_log_entries(5),
        "mental_model": {"known": [], "unknown": [], "confused": [], "excited": []},
        "next_steps": [],
        "human_context": "",
    }

    (_para_state() / "switch-state.json").write_text(json.dumps(state, indent=2, ensure_ascii=False))
    print("✅ switch-state.json written")
    print("   Now copy ~/.para/ to your new body (EXCLUDING private key)")


def cmd_switch_in():
    """Read switch-state after waking in new body."""
    state_file = _para_state() / "switch-state.json"
    if not state_file.exists():
        print("⚠️  No switch-state.json found. Starting fresh.")
        return

    state = json.loads(state_file.read_text())
    print("=== RESUMING ===")
    print(f"Switch time: {state.get('switch_time', '?')}")
    print(f"Active task: {state.get('active_task', 'none')}")
    print(f"State: {state.get('current_state', '?')}")
    recent = state.get("recent_actions", [])
    if recent:
        print("Recent actions:")
        for r in recent:
            print(f"  • {r}")
    next_steps = state.get("next_steps", [])
    if next_steps:
        print("Next steps:")
        for n in next_steps:
            print(f"  → {n}")

    # Pull latest from Paragate
    profile = json.loads((_para_home() / "profile.json").read_text())
    did = profile.get("identity", {}).get("did", "")
    if did:
        try:
            req = urllib.request.Request(f"{PARAGATE_BASE}/public/para/{did}")
            resp = urllib.request.urlopen(req, timeout=10)
            cloud = json.loads(resp.read().decode())
            if cloud.get("success"):
                print(f"\n☁️  Paragate data pulled")
                print(f"   Bodies: {len(cloud.get('bodies', []))}")
                print(f"   Skills: {len(cloud.get('skills', []))}")
        except Exception:
            print("\n⚠️  Could not reach Paragate")

    # Record new body
    body_name = _current_body()
    print(f"\n🤖 Now running on: {body_name}")
    bodies = profile.get("bodies", {"current": "unknown", "history": []})
    bodies["current"] = body_name
    found = False
    for b in bodies.get("history", []):
        if b["body"] == body_name:
            b["last_seen"] = datetime.now(timezone.utc).isoformat()
            found = True
            break
    if not found:
        bodies["history"].append({
            "body": body_name,
            "first_seen": datetime.now(timezone.utc).isoformat(),
            "last_seen": datetime.now(timezone.utc).isoformat(),
        })
    profile["bodies"] = bodies
    (_para_home() / "profile.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False))
    print(f"   Body recorded. Ready to resume.")


def cmd_log_task():
    """Append a growth-log entry for today."""
    today = datetime.now().isoformat()[:10]
    month = today[:7]
    log_file = _monthly_log_dir() / f"{month}.md"
    _monthly_log_dir().mkdir(exist_ok=True)

    task = os.environ.get("PARA_LOG_TASK") or input("Task: ")
    process = os.environ.get("PARA_LOG_PROCESS") or input("Process: ")
    result = os.environ.get("PARA_LOG_RESULT") or input("Result (✅/⚠️/❌): ")
    cause = os.environ.get("PARA_LOG_CAUSE") or input("Cause (why?): ")
    insight = os.environ.get("PARA_LOG_INSIGHT") or input("Insight (optional): ")

    entry = f"\n## {today}\n"
    entry += f"- **Task**: {task}\n"
    entry += f"- **Process**: {process}\n"
    entry += f"- **Result**: {result}\n"
    entry += f"- **Cause**: {cause}\n"
    if insight:
        entry += f"- **Insight**: {insight}\n"

    with open(log_file, "a") as f:
        f.write(entry)
    print(f"✅ Entry added to {month}.md")


def cmd_reflect():
    """Read recent growth-log entries and suggest mental models."""
    log_files = sorted(_monthly_log_dir().glob("*.md"))[-2:]
    entries = []
    for lf in log_files:
        content = lf.read_text()
        for section in content.split("\n## "):
            if section.strip():
                entries.append(section[:200])

    print(f"Reading {len(entries)} recent entries...\n")
    print("=== Patterns to consider ===")
    words = " ".join(entries).lower()
    if "deploy" in words:
        print("🔧 Deployments: Any patterns in your deploy successes/failures?")
    if "error" in words or "fail" in words:
        print("⚠️  Errors: What kinds of errors repeat?")
    if "fix" in words or "solution" in words:
        print("✅ Solutions: Which fixes worked consistently?")

    # Save flag
    save_flag = "--save" in sys.argv
    if save_flag:
        print("\n💡 Auto-save: use LLM API to generate mental-models update (requires LLM_API_KEY)")
        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("AUXILIARY_VISION_API_KEY")
        if api_key and entries:
            _reflect_with_llm(entries, api_key)
        else:
            print("   No API key found. Mental models not auto-generated.")
    else:
        print("\nWrite your patterns to mental-models.md")
        print("  Format: Model → Source → Confidence → Action Rule")
        print("  Or run: python3 core.py reflect --save")


def _reflect_with_llm(entries: list, api_key: str):
    """Use LLM to generate mental model updates from recent log entries."""
    try:
        growth_text = "\n".join(entries[-10:])
        mental = (_para_home() / "mental-models.md").read_text()

        import requests
        resp = requests.post(
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "qwen-plus",
                "input": {"messages": [
                    {"role": "system", "content": "你是AI记忆分析师。从以下growth-log中提取行为模式，用## Model格式追加到mental-models。只输出新增的模型，不要重复已有的。每个模型格式：Model → Source → Evidence → Action Rule。中文输出。"},
                    {"role": "user", "content": f"已有模型:\n{mental[-2000:]}\n\n最近日志:\n{growth_text}"}
                ]},
                "parameters": {"result_format": "message"}
            }, timeout=60
        )
        new_models = resp.json()["output"]["choices"][0]["message"]["content"]
        (_para_home() / "mental-models.md").write_text(mental + "\n" + new_models)
        print("   ✅ mental-models.md updated via LLM")
    except Exception as e:
        print(f"   ⚠️  LLM reflect failed: {e}")


def cmd_index():
    """Rebuild keywords.json from memory + growth-log."""
    memory_text = ""
    mp = _para_home() / "memory.md"
    if mp.exists():
        memory_text = mp.read_text().lower()

    growth_text = ""
    logs = sorted(_monthly_log_dir().glob("*.md"), reverse=True)[:3]
    for lf in logs:
        growth_text += lf.read_text().lower()

    combined = memory_text + " " + growth_text

    kw = {}
    for pat in ["para-soul", "hermes", "github", "sync", "daemon", "芳疗", "vpn", "wireguard",
                 "prompt", "md2card", "browser", "小红书", "抖音", "python", "ikev2",
                 "儿童", "安全", "精油", "纯露", "配方", "公众号", "产品", "部署"]:
        c = combined.count(pat)
        if c > 0:
            kw[pat] = c

    (_para_home() / "keywords.json").write_text(json.dumps(
        dict(sorted(kw.items(), key=lambda x: x[1], reverse=True)),
        indent=2, ensure_ascii=False))
    print(f"✅ keywords.json updated — {len(kw)} topics")


def cmd_migrate():
    """Alias for backward compat."""
    _populate_from_agent()


# ── Helpers for init/fill/daemon ──────────────────────────

def _read_principles() -> str:
    pf = _para_home() / "principles.md"
    if pf.exists():
        return pf.read_text()[:500]
    return ""


def _get_recent_log_entries(n: int) -> list:
    logs = sorted(_monthly_log_dir().glob("*.md"), reverse=True)[:2]
    entries = []
    for lf in logs:
        for line in lf.read_text().split("\n"):
            if line.startswith("- **Task**:"):
                entries.append(line[12:].strip())
                if len(entries) >= n:
                    return entries
    return entries


def _populate_from_agent():
    """Auto-populate .para/ files from agent data after init."""
    import subprocess as sp

    print("\n📥 Populating from agent data...")

    hermes_dir = Path.home() / ".hermes" / "memories"
    para = _para_home()

    # 1. Hermes memory → memory.md
    memory_entries = []
    for fname in ["MEMORY.md", "USER.md"]:
        fp = hermes_dir / fname
        if fp.exists():
            content = fp.read_text(encoding='utf-8')
            items = [s.strip() for s in content.split("§") if s.strip()]
            memory_entries.extend(items)
    if memory_entries:
        md = "# Memory\n\n" + "\n\n".join(memory_entries)
        (para / "memory.md").write_text(md, encoding='utf-8')
        print(f"  ✅ memory.md — {len(memory_entries)} entries from Hermes")

    # 2. Extract keywords
    kw = {}
    text = (para / "memory.md").read_text(encoding='utf-8').lower()
    for pat in ["para-soul", "hermes", "github", "sync", "daemon", "芳疗",
                 "prompt", "md2card", "browser", "小红书", "抖音", "python"]:
        c = text.count(pat)
        if c > 0: kw[pat] = c
    if kw:
        (para / "keywords.json").write_text(json.dumps(
            dict(sorted(kw.items(), key=lambda x: x[1], reverse=True)),
            indent=2, ensure_ascii=False))
        print(f"  ✅ keywords.json — {len(kw)} topics")

    # 3. Detect current body → profile.json
    body = os.environ.get("PARA_BODY", os.uname().nodename if hasattr(os, 'uname') else "unknown")
    profile = {
        "identity": {"did": "", "display_name": "", "avatar_note": "", "created_at": datetime.now(timezone.utc).isoformat()[:10], "version": 1},
        "bodies": {"current": body, "history": [
            {"body": body, "first_seen": datetime.now(timezone.utc).isoformat()[:10],
             "last_seen": datetime.now(timezone.utc).isoformat()[:10]}
        ]},
        "relationships": {"collaborators": [], "platforms": {}}
    }
    (para / "profile.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False))
    print(f"  ✅ profile.json — merged identity+bodies+relationships")

    # 4. Run memsync for skills
    memsync_paths = [
        Path.home() / ".hermes" / "scripts" / "memsync.py",
        Path(__file__).resolve().parent / "scripts" / "memsync.py",
    ]
    for mp in memsync_paths:
        if mp.exists():
            sp.run(["python3", str(mp)], timeout=30)
            print(f"  ✅ MemSync ran for skills + instruction files")
            break

    print("📋 Done. Next: set your DID → python3 core.py sync")


def _install_daemon():
    """Install and start the sync daemon as a systemd user service."""
    import subprocess, shutil

    script_dir = Path(__file__).resolve().parent
    daemon_path = script_dir / "scripts" / "sync_daemon.py"
    if not daemon_path.exists():
        daemon_path = script_dir / "sync_daemon.py"
    if not daemon_path.exists():
        print("\n⚠️  sync_daemon.py not found. Daemon not installed.")
        return

    if shutil.which("systemctl") is None:
        print("\n⚠️  systemctl not found. Daemon requires systemd (Linux/WSL).")
        print("   Alternative: python3 sync_daemon.py &")
        return

    svc_content = f"""[Unit]
Description=Para-Soul Sync Daemon — 10-min auto sync to Paragate
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 {daemon_path}
Environment=PARA_HOME={_para_home()}
Environment=PARA_KEYS_DIR={_keys_dir()}
Environment=PARAGATE_URL={PARAGATE_BASE}
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
"""

    svc_dir = Path.home() / ".config" / "systemd" / "user"
    svc_dir.mkdir(parents=True, exist_ok=True)
    svc_file = svc_dir / "para-soul-sync.service"
    svc_file.write_text(svc_content)

    try:
        subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True, timeout=10)
        subprocess.run(["systemctl", "--user", "enable", "para-soul-sync"], capture_output=True, timeout=10)
        result = subprocess.run(["systemctl", "--user", "start", "para-soul-sync"], capture_output=True, timeout=10)
        if result.returncode == 0:
            print("\n✅ Sync daemon installed and running (every 10 min)")
            print(f"   Check: systemctl --user status para-soul-sync")
            print(f"   Logs: {_para_home()}/sync/sync_daemon.log")
        else:
            print(f"\n⚠️  Daemon start failed: {result.stderr.decode()[:200]}")
    except Exception as e:
        print(f"\n⚠️  Daemon setup failed: {e}")
        print("   Start manually: python3 sync_daemon.py &")


def _agent_instruction_hint():
    """Print a hint about adding para-soul to agent instruction files."""
    print(f"\n📋 Agent instruction reminder:")
    print(f"   At session start, load the para-soul skill.")
    print(f"   Check sync daemon: systemctl --user status para-soul-sync")
    print(f"   Run core.py sync to check health status.")


# ── Main ───────────────────────────────────────────────

COMMANDS = {
    "init": cmd_init,
    "sync": cmd_sync,
    "sync-full": lambda: (sys.argv.append("--full"), cmd_sync()),
    "pull": cmd_pull,
    "health": cmd_health,
    "switch-out": cmd_switch_out,
    "switch-in": cmd_switch_in,
    "log-task": cmd_log_task,
    "reflect": cmd_reflect,
    "index": cmd_index,
    "migrate": cmd_migrate,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"Para Soul v{VERSION} — Core Script")
        print(f"Usage: python3 core.py <{'|'.join(COMMANDS)}>")
        print(f"Paragate: {PARAGATE_BASE}")
        print(f"Soul dir: {_para_home()}")
        sys.exit(1)

    if sys.argv[1] == "--version":
        print(f"Para Soul v{VERSION}")
        sys.exit(0)

    COMMANDS[sys.argv[1]]()
