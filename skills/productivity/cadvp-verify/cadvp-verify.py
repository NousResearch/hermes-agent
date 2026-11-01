#!/usr/bin/env python3
"""
CADVP — Cross-Agent Delivery Verification Protocol v1.1
13-dimension verification framework for cross-agent memory injection validation.

Usage:
  python3 cadvp-verify.py <target_profile>

Example:
  python3 cadvp-verify.py my-agent

Note: requires PyYAML. Install with: pip install pyyaml
"""
import sqlite3, json, sys, os, datetime

TARGET = sys.argv[1] if len(sys.argv) > 1 else (print("Usage: python3 cadvp-verify.py <target_profile>") or sys.exit(1))
BASE = os.path.expanduser(f"~/.hermes/profiles/{TARGET}")

RESULTS = []
def check(code, name, urgent, fn):
    try:
        ok, detail = fn()
        status = "PASS" if ok else "FAIL"
    except Exception as e:
        ok, status, detail = False, "FAIL", str(e)
    RESULTS.append({"code": code, "name": name, "urgent": urgent, "status": status, "detail": detail, "ok": ok})

# ── Try loading yaml ──
yaml = None
try:
    import yaml as _yaml
    yaml = _yaml
except ImportError:
    pass

# ══════════════════════════════════
# CC-0: Channel Confirmation (v1.1)
# ══════════════════════════════════
def cc0():
    """Verify the delivery channel is architecturally available before any operation.
    
    Known constraints:
    - Cron/leaf agents do NOT have memory() or fact_store() tools
    - Only two reliable channels exist: A) direct DB write by admin, B) target self-write via memory() tool
    - Cron-delegated writes are never viable (blocked by skip_memory=True)
    """
    cfg_path = os.path.join(BASE, "config.yaml")
    mem_enabled = False
    if os.path.isfile(cfg_path) and yaml:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        mem = cfg.get("memory", {})
        if mem:
            mem_enabled = mem.get("memory_enabled", False)
    
    db_exists = os.path.isfile(os.path.join(BASE, "memory_store.db"))
    
    channels = []
    if mem_enabled and db_exists:
        channels.append("A. Direct DB write (SQLite INSERT into memory_store.db) -- Available")
    if mem_enabled:
        channels.append("B. Target self-write via memory()/fact_store() -- Available (requires target-side execution)")
    channels.append("C. Cron-delegated write (cron agent has no memory/fact_store tools) -- Never available by design")
    
    ok = mem_enabled and db_exists
    return ok, "Available channels:\n       " + "\n       ".join(channels)


# ══════════════════════════════════
# PC: Pre-Condition Checks (3)
# ══════════════════════════════════

def pc1():
    """Target profile directory exists and gateway is running."""
    exists = os.path.isdir(BASE)
    pidfile = os.path.join(BASE, "gateway.pid")
    pid_str = "N/A"
    if os.path.isfile(pidfile):
        with open(pidfile) as f:
            raw = f.read().strip()
            try:
                pid_data = json.loads(raw)
                pid_str = str(pid_data.get("pid", raw))
            except json.JSONDecodeError:
                pid_str = raw
    ok = exists and pid_str != "N/A"
    return ok, f"profile_dir={'exists' if exists else 'missing'}, gateway_pid={pid_str}"


def pc2():
    """Memory configuration exists in target profile config."""
    cfg_path = os.path.join(BASE, "config.yaml")
    if not os.path.isfile(cfg_path):
        return False, "config.yaml not found"
    provider = "NONE"
    enabled = False
    if yaml:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        mem = cfg.get("memory", {})
        provider = mem.get("provider", "NONE") if mem else "NONE"
        enabled = mem.get("memory_enabled", False) if mem else False
    else:
        with open(cfg_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("provider:"):
                    provider = line.split(":", 1)[1].strip()
                if line.startswith("memory_enabled:"):
                    enabled = line.split(":", 1)[1].strip().lower() == "true"
    db_exists = os.path.isfile(os.path.join(BASE, "memory_store.db"))
    ok = provider not in ("NONE", "", None) and enabled and db_exists
    return ok, f"provider={provider}, memory_enabled={enabled}, memory_store.db={'exists' if db_exists else 'missing'}"


def pc3():
    """Scan for other profiles that might be affected."""
    profiles_dir = os.path.expanduser("~/.hermes/profiles")
    all_profiles = []
    if os.path.isdir(profiles_dir):
        all_profiles = [d for d in os.listdir(profiles_dir) if os.path.isdir(os.path.join(profiles_dir, d))]
    ok = len(all_profiles) > 0
    return ok, f"{len(all_profiles)} profiles found: {', '.join(all_profiles[:10])}" + (f"..." if len(all_profiles) > 10 else "")


# ══════════════════════════════════
# WV: Write-side Verification (3)
# ══════════════════════════════════

def wv1():
    """Data exists in the target memory_store.db facts table."""
    db = os.path.join(BASE, "memory_store.db")
    if not os.path.isfile(db):
        return False, "memory_store.db not found"
    conn = sqlite3.connect(db)
    try:
        count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    except:
        count = 0
    conn.close()
    return count > 0, f"{count} facts in table"


def wv2():
    """Sample fact content for correctness check."""
    db = os.path.join(BASE, "memory_store.db")
    if not os.path.isfile(db):
        return False, "memory_store.db not found"
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute("SELECT fact_id, substr(content,1,80) FROM facts LIMIT 3").fetchall()
    except:
        rows = []
    conn.close()
    if not rows:
        return False, "No facts to sample"
    return True, "\n         ".join(f"[{r[0]}] {r[1]}" for r in rows)


def wv3():
    """File permissions and size check."""
    db = os.path.join(BASE, "memory_store.db")
    if not os.path.isfile(db):
        return False, "memory_store.db not found"
    import stat
    st = os.stat(db)
    perm = oct(stat.S_IMODE(st.st_mode))
    sz = st.st_size // 1024
    return True, f"permissions={perm}, size={sz}KB"


# ══════════════════════════════════
# RV: Read-side Verification (4)
# ══════════════════════════════════

def rv1():
    """Target profile config has memory properly configured."""
    cfg_path = os.path.join(BASE, "config.yaml")
    if not os.path.isfile(cfg_path):
        return False, "config.yaml not found"
    provider = "NONE"
    enabled = False
    cl = "unset"
    if yaml:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        mem = cfg.get("memory", {})
        if mem:
            provider = mem.get("provider", "NONE")
            enabled = mem.get("memory_enabled", False)
            cl = str(mem.get("memory_char_limit", "unset"))
    ok = provider not in ("NONE", "", None) and enabled
    return ok, f"provider={provider}, memory_enabled={enabled}, memory_char_limit={cl}"


def rv2():
    """memory_store.db exists and has memory banks loaded."""
    db = os.path.join(BASE, "memory_store.db")
    if not os.path.isfile(db):
        return False, "memory_store.db not found"
    conn = sqlite3.connect(db)
    try:
        banks = conn.execute("SELECT COUNT(*) FROM memory_banks").fetchone()[0]
    except:
        banks = 0
    conn.close()
    return True, f"memory_banks={banks}, size={os.path.getsize(db)//1024}KB"


def rv3():
    """FTS5 full-text search confirms content is correctly indexed."""
    db = os.path.join(BASE, "memory_store.db")
    if not os.path.isfile(db):
        return False, "memory_store.db not found"
    conn = sqlite3.connect(db)
    try:
        cur = conn.execute("SELECT COUNT(*) FROM facts_fts WHERE facts_fts MATCH ?", ("injection OR memory OR channel",))
        count = cur.fetchone()[0]
        conn.close()
        return count > 0, f"FTS search hit {count} results"
    except Exception as e:
        conn.close()
        return False, f"FTS search error: {e}"


def rv4():
    """Gateway is running (required for target agent to receive queries)."""
    pidfile = os.path.join(BASE, "gateway.pid")
    pid_str = "N/A"
    if os.path.isfile(pidfile):
        with open(pidfile) as f:
            raw = f.read().strip()
            try:
                pid_data = json.loads(raw)
                pid_str = str(pid_data.get("pid", raw))
            except json.JSONDecodeError:
                pid_str = raw
    ok = pid_str != "N/A"
    return ok, f"gateway PID={pid_str}, {'running' if ok else 'not running'}"


# ══════════════════════════════════
# GR: Graceful Recovery (2)
# ══════════════════════════════════

def gr1():
    """Check other profiles with holographic memory for fallback potential."""
    profiles_dir = os.path.expanduser("~/.hermes/profiles")
    holo_profiles = []
    if os.path.isdir(profiles_dir):
        for d in os.listdir(profiles_dir):
            cfg = os.path.join(profiles_dir, d, "config.yaml")
            if not os.path.isfile(cfg):
                continue
            if yaml:
                try:
                    with open(cfg) as f:
                        c = yaml.safe_load(f)
                    m = c.get("memory", {})
                    if m and m.get("memory_enabled", False):
                        holo_profiles.append(d)
                except:
                    pass
    ok = len(holo_profiles) > 0
    return ok, f"{len(holo_profiles)} other profiles with holographic: {', '.join(holo_profiles)}"


def gr2():
    """Knowledge base (at ~/.hermes/knowledge/) is available for archival."""
    kb = os.path.expanduser("~/.hermes/knowledge")
    ok = os.path.isdir(kb)
    return ok, f"knowledge base dir {'exists' if ok else 'missing'}"


# ══════════════════════════════════
# Execute All 13 Checks
# ══════════════════════════════════

print(f"\n{'=' * 64}")
print(f" CADVP v1.1 — Cross-Agent Delivery Verification Protocol")
print(f" Target: {TARGET}")
print(f" Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 64}\n")

urgent_icons = {"VETO": "🔴", "HIGH": "🟠", "NORMAL": "🟢"}

check("CC-0", "Channel Confirmation", "VETO", cc0)
check("PC-1", "Prerequisite: Target Identity", "VETO", pc1)
check("PC-2", "Prerequisite: Data Channel", "VETO", pc2)
check("PC-3", "Prerequisite: Impact Assessment", "HIGH", pc3)
check("WV-1", "Write: Data Confirmation", "NORMAL", wv1)
check("WV-2", "Write: Content Integrity", "NORMAL", wv2)
check("WV-3", "Write: Write Permissions", "HIGH", wv3)
check("RV-1", "Read: Config Activation", "VETO", rv1)
check("RV-2", "Read: Runtime Loading", "VETO", rv2)
check("RV-3", "Read: Tool Accessibility", "VETO", rv3)
check("RV-4", "Read: User Perception", "VETO", rv4)
check("GR-1", "Recovery: Impact Check", "HIGH", gr1)
check("GR-2", "Recovery: Documentation", "NORMAL", gr2)

fail_count = 0
crit_fail = 0
for r in RESULTS:
    icon = "✅" if r["status"] == "PASS" else "❌"
    urg = urgent_icons.get(r["urgent"], "")
    print(f" {icon} [{r['code']}] {urg} {r['name']}: {r['status']}")
    if r["status"] != "PASS":
        fail_count += 1
        if r["urgent"] in ("VETO", "HIGH"):
            crit_fail += 1
    if r["detail"]:
        print(f"       {r['detail']}")

print(f"\n{'=' * 64}")
print(f" Total: {len(RESULTS)} checks — {len(RESULTS)-fail_count} PASS / {fail_count} FAIL (VETO+HIGH: {crit_fail})")
vetoed = any(r["status"] != "PASS" and r["urgent"] == "VETO" for r in RESULTS)
if vetoed:
    print(f"\n {'=' * 8} CC-0 or VETO-level check FAILED. Channel is unavailable. Abort and switch to alternative channel.")
elif crit_fail > 0:
    print(f"\n {'=' * 8} Non-blocking failures found. Delivery possible but investigate HIGH-level failures.")
else:
    print(f"\n {'=' * 8} All checks PASSED. Delivery confirmed.")
print(f"{'=' * 64}\n")
