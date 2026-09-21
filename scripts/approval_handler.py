#!/usr/bin/env python3
"""
Discord Approval Handler — polls #research-digest for "Approve" replies
and auto-files kanban tasks. Token loaded from /home/kensei/.hermes/.env.
"""
import json, os, re, sys, time, urllib.request, urllib.error
from pathlib import Path
from datetime import datetime, timezone, timedelta

TZ = timezone(timedelta(hours=1))
KENSEI_ID = "1506024421104812274"
CHANNEL_ID = "1506021736536215813"

# P13 isolation: paths derive from HERMES_HOME / WIKI_DIR (env-overridable)
# so local disposable runs never touch /home/kensei/.hermes or the
# production wiki repos checkout.
_HERMES_HOME = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
WIKI_REPOS = Path(os.environ.get("WIKI_DIR", os.path.expanduser("~/docs/wiki"))) / "repos"
STATE = _HERMES_HOME / "state" / "approval-handler-state.json"

# P13 isolation: when --dry-run is passed, every write path (Discord API
# POST, wiki page rewrite, kanban task creation, state file save) is
# suppressed. Read paths (load_state, discord_api GET, find_wiki) run
# unchanged so the approval detection is still exercised.
_DRY_RUN = False

def get_token():
    """Read token from dotenv."""
    dotenv = _HERMES_HOME / ".env"
    if dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("DISCORD_BOT_TOKEN="):
                key = line[len("DISCORD_BOT_TOKEN="):].strip().strip('"').strip("'")
                return key
    return os.environ.get("DISCORD_BOT_TOKEN", "")

def discord_api(endpoint, data=None):
    if _DRY_RUN and data is not None:
        return None  # dry-run: suppress write (POST) calls; reads (GET) run
    token = get_token()
    if not token:
        print("err: no token")
        return None
    url = f"https://discord.com/api/v10{endpoint}"
    hdrs = {"Authorization": f"Bot {token}", "Content-Type": "application/json"}
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=hdrs, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            return {"_skip": f"rate: {e.headers.get('Retry-After',5)}"}
        return {"_err": f"{e.code}"}
    except Exception as e:
        return {"_err": str(type(e).__name__)}

def load_state():
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, Exception):
            pass
    return {"processed": [], "last": 0}

def save_state(st):
    if _DRY_RUN:
        return
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st), encoding="utf-8")

def extract_repo(text):
    m = re.search(r'[Aa]pprove\s+["\']?([a-zA-Z0-9][-a-zA-Z0-9._/]+)["\']?', text)
    return m.group(1).strip().rstrip('.') if m else None

def find_wiki(repo_name):
    repo_name = repo_name.lower().replace(" ", "-")
    for f in WIKI_REPOS.glob("*.md"):
        if f.stem == repo_name or f.stem == repo_name.split("/")[-1]:
            return f
    for f in WIKI_REPOS.glob("*.md"):
        if repo_name in f.stem or f.stem in repo_name:
            return f
    return None

def update_wiki(page):
    c = page.read_text()
    if 'adoption_status: implemented' in c or 'adoption_status: approved' in c:
        return False
    c = re.sub(r'(adoption_status:\s*)\S+', r'\1approved', c)
    if 'tags:' in c and 'approved' not in c:
        c = re.sub(r'(tags:\s*\[.*?)\]', r'\1, approved]', c)
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    if 'updated:' in c:
        c = re.sub(r'updated:\s*\S+', f'updated: {today}', c)
    else:
        c = c.replace("---\n", f"---\nupdated: {today}\n", 1)
    if _DRY_RUN:
        return True
    page.write_text(c)
    return True

def kanban_task(repo_name):
    import subprocess
    if _DRY_RUN:
        return None
    title = f"GitRadar Approval: {repo_name}"
    key = f"approval-{repo_name.lower().replace('/','-')}-{datetime.now(TZ).strftime('%Y-%m-%d')}"
    body = f"Approved via Discord reply. Auto-created.\n\nRepo: {repo_name}\n\nInclude `repo:{repo_name}` in completion result for auto-tagging."
    cmd = ["hermes", "kanban", "create", title, "--assignee", "octacon",
           "--priority", "3", "--triage", "--body", body,
           "--idempotency-key", key, "--created-by", "discord-approval-handler", "--json"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        out = r.stdout.strip()
        if r.returncode == 0:
            try:
                return json.loads(out).get("id", out)
            except json.JSONDecodeError:
                return out
    except Exception:
        pass
    return None

def main():
    global _DRY_RUN
    if "--dry-run" in sys.argv:
        _DRY_RUN = True
    h = datetime.now(TZ).hour
    if h < 7 or h >= 22:
        return
    
    st = load_state()
    processed = set(st.get("processed", []))
    
    msgs = discord_api(f"/channels/{CHANNEL_ID}/messages?limit=20")
    if not msgs or isinstance(msgs, dict) and "_err" in msgs:
        return
    
    approved = []
    
    for msg in msgs:
        mid = msg.get("id")
        if mid in processed:
            continue
        ref = msg.get("message_reference")
        if not ref:
            continue
        content = msg.get("content", "")
        repo = extract_repo(content)
        if not repo:
            continue
        ref_mid = ref.get("message_id")
        ref_msg = discord_api(f"/channels/{CHANNEL_ID}/messages/{ref_mid}") if ref_mid else None
        if not ref_msg or ref_msg.get("_err") or ref_msg.get("author",{}).get("id") != KENSEI_ID:
            continue
        
        processed.add(mid)
        page = find_wiki(repo)
        if page and update_wiki(page):
            print(f"wiki: {page}")
        tid = kanban_task(repo)
        if tid:
            approved.append((repo, tid))
    
    st["processed"] = list(processed)
    st["last"] = int(time.time())
    save_state(st)
    if approved:
        print(f"approved: {len(approved)}")
        for r, t in approved:
            print(f"  {r} → {t}")
        # P2-10 Feedback: record approval signals
        try:
            from hermes_cli.feedback_loop import FeedbackLoop, classify_message
            loop = FeedbackLoop()
            for repo, _ in approved:
                signal = classify_message(
                    f"Approve {repo}",
                    linked_task_id=None,
                )
                if signal.signal_type.value != "unknown":
                    loop.record(signal)
            print(f"  feedback: {len(approved)} signals recorded")
        except ImportError:
            pass

if __name__ == "__main__":
    main()