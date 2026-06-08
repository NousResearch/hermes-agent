"""
Denní Telegram report o stavu AI firem v Paperclip.
Spouští se jako Hermes cron job každý den v 8:00.
"""

import json
import os
import sys
import subprocess
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta

PC_API    = os.environ.get("PAPERCLIP_API_URL", "http://localhost:3100")
KEYS_FILE = os.path.join(os.path.dirname(__file__), "agent_keys.json")
MEM_API   = "http://localhost:8767"
MEM_SECRET = os.environ.get("PAPERCLIP_DAEMON_SECRET", "")

# Načíst libovolný agent klíč pro čtení (board read-only přes agent klíč)
# Potřebujeme board session — použijeme cookie soubor
COOKIE_FILE = "/tmp/pc_report.txt"
# Control plane base URL z env. Veřejná doména paperclip.frigeble.com:443 padla pod
# Plane PM proxy (žádný vhost/cert → TLS alert), control plane teď běží na localhost:3100.
PC_PUBLIC = os.environ.get("PAPERCLIP_API_URL", "http://localhost:3100")
ORIGIN = PC_PUBLIC

COMPANY_NAMES = {
    "a02a5207-0585-4386-b462-29329e9bf835": "Hermes Platform",
    "9e500522-663d-4ce9-ace4-ce59ee52e1b7": "FVE Optimizer",
    "33e1cb09-93f3-45e7-8e7d-6a2396d7aef4": "YT Czech Dubbing",
    "e7fb58f4-f077-4ed4-85a2-f6cfd6860786": "Infrastructure",
}

STATUS_ICONS = {
    "todo":        "📋",
    "backlog":     "🗂",
    "in_progress": "⚙️",
    "done":        "✅",
    "cancelled":   "🚫",
}


def _curl(method, path, data=None, use_board=True):
    cmd = ["curl", "-s"]
    if use_board:
        cmd += ["-b", COOKIE_FILE, "-H", f"Origin: {PC_PUBLIC}"]
    cmd += ["-H", "Content-Type: application/json"]
    if method in ("POST", "PATCH"):
        cmd += ["-X", method, "-d", json.dumps(data or {})]
    cmd.append(f"{PC_PUBLIC}{path}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    if not r.stdout.strip():
        return None
    try:
        return json.loads(r.stdout)
    except Exception:
        return None


def _mem(path, data=None):
    cmd = ["curl", "-s", "-H", f"X-Paperclip-Secret: {MEM_SECRET}",
           "-H", "Content-Type: application/json"]
    if data:
        cmd += ["-X", "POST", "-d", json.dumps(data)]
    cmd.append(f"{MEM_API}{path}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    try:
        return json.loads(r.stdout)
    except Exception:
        return {}


def login():
    """Přihlásit se do Paperclip pro board přístup."""
    # Credentials výhradně z prostředí (.env) — žádný hardcoded fallback.
    pw = os.environ.get("PAPERCLIP_BOARD_PASSWORD")
    email = os.environ.get("PAPERCLIP_BOARD_EMAIL")
    if not pw or not email:
        print("ERROR: chybí PAPERCLIP_BOARD_EMAIL / PAPERCLIP_BOARD_PASSWORD v prostředí (.env)",
              file=sys.stderr)
        sys.exit(2)
    cmd = ["curl", "-s", "-X", "POST",
           "-H", "Content-Type: application/json",
           "-c", COOKIE_FILE,
           "-d", json.dumps({"email": email, "password": pw}),
           f"{PC_PUBLIC}/api/auth/sign-in/email"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    d = json.loads(r.stdout) if r.stdout.strip() else {}
    return bool(d.get("token"))


def get_company_summary(company_id: str, company_name: str) -> dict:
    """Načte souhrn firmy: issues, agenti, costs."""
    # Issues
    issues = _curl("GET", f"/api/companies/{company_id}/issues") or []
    
    # Filtrovat na relevantní (ne starší než 7 dní)
    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    
    by_status = {}
    recent_done = []
    in_progress = []
    blockers = []
    
    for issue in (issues if isinstance(issues, list) else []):
        status = issue.get("status", "?")
        by_status[status] = by_status.get(status, 0) + 1
        
        if status == "in_progress":
            in_progress.append(issue.get("title", "?"))
        
        # Hotové za posledních 24h
        updated = issue.get("updatedAt", "")
        if status == "done" and updated > (datetime.now(timezone.utc) - timedelta(days=1)).isoformat():
            recent_done.append(issue.get("title", "?"))
        
        # Blokátory — issues in_progress déle než 3 dny
        if status == "in_progress":
            created = issue.get("createdAt", "")
            if created < (datetime.now(timezone.utc) - timedelta(days=3)).isoformat():
                blockers.append(issue.get("title", "?"))

    # Agenti
    agents = _curl("GET", f"/api/companies/{company_id}/agents") or []
    agent_statuses = {}
    for a in (agents if isinstance(agents, list) else []):
        s = a.get("status", "?")
        agent_statuses[s] = agent_statuses.get(s, 0) + 1

    # Costs — aktuální měsíc
    costs = _curl("GET", f"/api/companies/{company_id}/costs/summary") or {}
    spent_cents = costs.get("spentMonthlyCents", 0) if isinstance(costs, dict) else 0
    budget_cents = costs.get("budgetMonthlyCents", 0) if isinstance(costs, dict) else 0

    # Memory
    mem = _mem("/count", None)
    mem_url = f"{MEM_API}/count?scope=project:{company_name.lower().replace(' ','-')}"
    mem_count_r = subprocess.run(
        ["curl", "-s", "-H", f"X-Paperclip-Secret: {MEM_SECRET}",
         f"{MEM_API}/count?scope=project:{company_name.lower().replace(' ', '-')}"],
        capture_output=True, text=True, timeout=5
    )
    try:
        mem_count = json.loads(mem_count_r.stdout).get("count", 0)
    except Exception:
        mem_count = 0

    return {
        "name": company_name,
        "id": company_id,
        "by_status": by_status,
        "in_progress": in_progress,
        "recent_done": recent_done,
        "blockers": blockers,
        "agents": agent_statuses,
        "spent_usd": spent_cents / 100,
        "budget_usd": budget_cents / 100,
        "mem_count": mem_count,
    }


def format_report(summaries: list[dict]) -> str:
    now = datetime.now(timezone.utc).strftime("%d.%m.%Y")
    lines = [f"📊 Denní přehled AI firem — {now}\n"]
    
    total_spent = 0.0
    total_budget = 0.0
    total_mems = 0

    for s in summaries:
        name = s["name"]
        by_status = s["by_status"]
        
        # Ikona stavu firmy
        if s["blockers"]:
            icon = "🔴"
        elif s["in_progress"]:
            icon = "🟢"
        elif by_status.get("todo", 0) > 0:
            icon = "🟡"
        else:
            icon = "⚪"
        
        lines.append(f"{icon} {name}")
        
        # Hotové včera
        if s["recent_done"]:
            for t in s["recent_done"][:3]:
                lines.append(f"   ✅ {t[:55]}")
        
        # In progress
        for t in s["in_progress"][:3]:
            lines.append(f"   ⚙️  {t[:55]}")
        
        # Blokátory
        for t in s["blockers"][:2]:
            lines.append(f"   ⚠️  Blokátor: {t[:45]}")
        
        # Statistika issues
        todo = by_status.get("todo", 0)
        wip  = by_status.get("in_progress", 0)
        done = by_status.get("done", 0)
        if todo or wip or done:
            lines.append(f"   📋 todo:{todo} | ⚙️:{wip} | ✅:{done}")
        
        # Náklady
        if s["budget_usd"] > 0:
            pct = int(100 * s["spent_usd"] / s["budget_usd"]) if s["budget_usd"] else 0
            lines.append(f"   💰 ${s['spent_usd']:.2f} / ${s['budget_usd']:.0f} ({pct}%)")
        
        # Paměť
        if s["mem_count"] > 0:
            lines.append(f"   🧠 {s['mem_count']} vzpomínek")
        
        lines.append("")
        total_spent  += s["spent_usd"]
        total_budget += s["budget_usd"]
        total_mems   += s["mem_count"]

    # Souhrn
    lines.append("─────────────────────────")
    lines.append(f"💰 Celkem: ${total_spent:.2f}")
    if total_budget > 0:
        lines.append(f"   Budget: ${total_budget:.0f}/měsíc")
    lines.append(f"🧠 Memory: {total_mems} vzpomínek celkem")
    lines.append(f"\n🔗 {PC_PUBLIC}")

    return "\n".join(lines)


def main():
    print("Přihlašuji se do Paperclip...")
    if not login():
        print("ERROR: Login selhal")
        sys.exit(1)
    
    print("Načítám data firem...")
    summaries = []
    for cid, cname in COMPANY_NAMES.items():
        try:
            s = get_company_summary(cid, cname)
            summaries.append(s)
            print(f"  {cname}: {s['by_status']}")
        except Exception as e:
            print(f"  CHYBA {cname}: {e}")
    
    report = format_report(summaries)
    print("\n" + "="*50)
    print(report)
    print("="*50)
    
    return report


if __name__ == "__main__":
    main()
