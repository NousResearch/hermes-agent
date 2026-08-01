#!/usr/bin/env python3
"""
Proposal Approval Handler - polls #research-ops for !approve/!reject replies
to mashup review posts. Auto-files kanban triage tasks for approved proposals.
Token loaded from $HERMES_HOME/.env.
"""
import json, os, re, time, urllib.request, urllib.error, subprocess
from pathlib import Path
from datetime import datetime, timezone, timedelta

HERMES_HOME = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
TZ = timezone(timedelta(hours=1))
KENSEI_ID = "1506024421104812274"
CHANNEL_ID = "1507448577784283367"
PROPOSALS_DIR = HERMES_HOME / "runbooks" / "proposals"
STATE = HERMES_HOME / "state" / "proposal-approval-state.json"

# Idea-card index: proposals/<date>/idea-index.jsonl (one JSON object per line).
# Phase 2 three-gate flow:
#   GATE 2 (pitch): idea flagged pitch:true -> pitch.md generated
#   GATE 3 (prototype): explicit approval -> prototype.md + pitch.md attached
IDEA_INDEX_NAME = "idea-index.jsonl"

DRY_RUN = False


def get_token():
    dotenv = HERMES_HOME / ".env"
    if dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("DISCORD_BOT_TOKEN="):
                key = line[len("DISCORD_BOT_TOKEN="):].strip().strip(chr(34)).strip(chr(39))
                return key
    return os.environ.get("DISCORD_BOT_TOKEN", "")


def discord_api(endpoint, data=None):
    token = get_token()
    if not token:
        return {"_err": "no token"}
    url = f"https://discord.com/api/v10{endpoint}"
    hdrs = {"Authorization": f"Bot {token}", "Content-Type": "application/json"}
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=hdrs, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            return {"_skip": f"rate: {e.headers.get(chr(82)+chr(101)+chr(116)+chr(114)+chr(121)+chr(45)+chr(65)+chr(102)+chr(116)+chr(101)+chr(114), 5)}"}
        return {"_err": f"HTTP {e.code}"}
    except Exception as e:
        return {"_err": str(type(e).__name__)}


def load_state():
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, Exception):
            pass
    return {"approved": [], "rejected": [], "last_msg_id": None}


def save_state(st):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=2), encoding="utf-8")


def parse_command(content):
    approve = re.search(r"!approve\s+([a-zA-Z0-9][-a-zA-Z0-9._]+)", content)
    if approve:
        return ("approve", approve.group(1).strip().rstrip("."))
    reject = re.search(r"!reject\s+([a-zA-Z0-9][-a-zA-Z0-9._]+)", content)
    if reject:
        return ("reject", reject.group(1).strip().rstrip("."))
    pitch = re.search(r"!pitch\s+([a-zA-Z0-9][-a-zA-Z0-9._]+)", content)
    if pitch:
        return ("pitch", pitch.group(1).strip().rstrip("."))
    return (None, None)


def find_proposal_html(slug):
    if not PROPOSALS_DIR.exists():
        return None, None
    for f in sorted(PROPOSALS_DIR.glob("mashup-*.html"), reverse=True):
        text = f.read_text(encoding="utf-8", errors="replace")
        if f"id={slug!r}" in text or (chr(34)+slug+chr(34)) in text:
            return f, text
    return None, None


def extract_proposal(text, slug):
    pattern = rf"(<section\s+id={chr(34)}{slug}{chr(34)}[^>]*>.*?</section>)"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)
    pattern = rf"(<section\s+id={chr(34)}{slug}-[^\"]*\"[^>]*>.*?</section>)"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else None


def extract_title(html_section):
    m = re.search(r"<h2[^>]*>(.*?)</h2>", html_section, re.DOTALL)
    return m.group(1).strip() if m else "Proposal"


def extract_effort(html_section):
    m = re.search(r"Effort:\s*(S|M|L|Small|Medium|Large)", html_section, re.IGNORECASE)
    if m:
        e = m.group(1)[0].upper()
        return {"S": "3", "M": "2", "L": "1"}.get(e, "2")
    return "2"


def find_idea_card(slug):
    """Look up the idea card for a slug across idea-index.jsonl files (newest first).

    Returns (card_dict, index_path) or (None, None).
    """
    if not PROPOSALS_DIR.exists():
        return None, None
    for idx in sorted(PROPOSALS_DIR.rglob(IDEA_INDEX_NAME), reverse=True):
        for line in idx.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                card = json.loads(line)
            except json.JSONDecodeError:
                continue
            if card.get("slug") == slug:
                return card, idx
    return None, None


def artifact_paths(idx_path, slug):
    """Resolve pitch.md / prototype.md for a slug inside its proposals/<date> dir."""
    base = idx_path.parent if idx_path is not None else PROPOSALS_DIR
    return base / f"{slug}-pitch.md", base / f"{slug}-prototype.md"


def create_kanban_triage(slug, title, html_section):
    key = f"proposal-{slug}-{datetime.now(TZ).strftime(chr(37)+chr(89)+chr(45)+chr(37)+chr(109)+chr(45)+chr(37)+chr(100))}"
    # Phase 2: attach GATE 2 (pitch) + GATE 3 (prototype) artifacts when present
    card, idx = find_idea_card(slug)
    attachments = ""
    if idx is not None:
        pitch_path, proto_path = artifact_paths(idx, slug)
        for label, p in (("pitch", pitch_path), ("prototype", proto_path)):
            if p.exists():
                attachments += f"\n---\n\n## {label.capitalize()} ({p.name})\n\n{p.read_text(encoding='utf-8', errors='replace')}\n"
    body = (
        f"Approved proposal from research mashup review.\n\n"
        f"---\n\n"
        f"{html_section}\n\n"
        f"{attachments}"
        f"---\n\n"
        f"Slug: `{slug}`\n"
        f"Approved via Discord by Sahil.\n"
        f"Route through Kensei Intake then Orchestrator then Octacon for build.\n"
    )
    priority = extract_effort(html_section)
    if card and card.get("recommendation", {}).get("priority"):
        try:
            raw = card["recommendation"]["priority"]
            priority = str(int(int(raw) // 3 + 1))
        except (ValueError, TypeError, ZeroDivisionError):
            pass
    cmd = [
        "hermes", "kanban", "create", title,
        "--triage", "--priority", priority,
        "--body", body,
        "--idempotency-key", key,
        "--created-by", "proposal-approval-handler",
        "--tag", "proposal-approved",
        "--tag", "paper-mashup",
        "--json",
    ]
    if DRY_RUN:
        print(f"[dry-run] would create kanban triage: {title} (slug={slug}, priority={priority})")
        return "dry-run-skipped"
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        out = r.stdout.strip()
        if r.returncode == 0:
            try:
                return json.loads(out).get("id", out)
            except json.JSONDecodeError:
                return out
        else:
            return f"err: {r.stderr.strip()[:200]}"
    except Exception as e:
        return f"err: {e}"


def main():
    if DRY_RUN:
        print("[dry-run] would poll #research-ops and process !approve/!reject commands")
        return

    h = datetime.now(TZ).hour
    if h < 7 or h >= 22:
        return

    st = load_state()
    approved_ids = set(st.get("approved", []))
    rejected_ids = set(st.get("rejected", []))
    last_msg_id = st.get("last_msg_id")

    msgs = discord_api(f"/channels/{CHANNEL_ID}/messages?limit=30")
    if not msgs or not isinstance(msgs, list):
        return

    new_last = last_msg_id
    actions = []

    for msg in msgs:
        mid = msg.get("id")
        if last_msg_id and int(mid) <= int(last_msg_id):
            continue
        ref = msg.get("message_reference")
        if not ref:
            continue
        ref_mid = ref.get("message_id")
        if msg.get("author", {}).get("bot", False):
            continue
        content = msg.get("content", "")
        action, slug = parse_command(content)
        if not action or not slug:
            continue
        ref_msg = discord_api(f"/channels/{CHANNEL_ID}/messages/{ref_mid}")
        if not ref_msg or ref_msg.get("_err"):
            continue
        if ref_msg.get("author", {}).get("id") != KENSEI_ID:
            continue
        slug = slug.lower()
        if slug in approved_ids or slug in rejected_ids:
            continue
        prop_file, prop_text = find_proposal_html(slug)
        if not prop_file:
            print(f"warn: slug {slug!r} not found in any proposal file")
            continue
        section = extract_proposal(prop_text, slug)
        if not section:
            print(f"warn: could not extract section for slug {slug!r}")
            continue
        title = extract_title(section)
        if action == "approve":
            tid = create_kanban_triage(slug, title, section)
            approved_ids.add(slug)
            actions.append(f"approved: {slug} -> kanban {tid}")
        elif action == "reject":
            rejected_ids.add(slug)
            actions.append(f"rejected: {slug}")
        elif action == "pitch":
            # GATE 2: mark idea as pitch-worthy; NO kanban task (prototype gate)
            card, idx = find_idea_card(slug)
            if card is not None and idx is not None:
                card["pitch"] = True
                card["pitch_requested_at"] = datetime.now(TZ).isoformat()
                # rewrite index line in place
                lines = idx.read_text(encoding="utf-8").splitlines()
                for i, line in enumerate(lines):
                    try:
                        c = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if c.get("slug") == slug:
                        lines[i] = json.dumps(card)
                        break
                idx.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                actions.append(f"pitched: {slug} (pitch.md generation queued)")
            else:
                print(f"warn: !pitch {slug} but no idea card found")
        if new_last is None or int(mid) > int(new_last):
            new_last = mid

    if actions:
        st["approved"] = list(approved_ids)
        st["rejected"] = list(rejected_ids)
        st["last_msg_id"] = new_last
        save_state(st)
        for a in actions:
            print(a)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Proposal approval handler")
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen, do not poll Discord or create tasks")
    args = parser.parse_args()
    DRY_RUN = args.dry_run
    main()
