#!/usr/bin/env python3
"""Post Hermes Kanban task events to Discord.

Watches ~/.hermes/kanban.db task_events and posts lifecycle updates. By default,
connected Kanban task graphs are grouped into project-scoped Discord threads
instead of a noisy flat #kanban-log stream.

Managed local deployment:
    install -m 0755 scripts/kanban_discord_log.py ~/.hermes/scripts/kanban_discord_log.py

The checked-in script is the runtime source of truth for the local watcher; the
copy under ~/.hermes/scripts/ is just the operator-managed deployment target.
"""
import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

HERMES_AGENT_ROOT = HOME = Path.home() / ".hermes"
AGENT_REPO = HERMES_AGENT_ROOT / "hermes-agent"
if str(AGENT_REPO) not in sys.path:
    sys.path.insert(0, str(AGENT_REPO))

from hermes_cli import kanban_project_model as kpm  # type: ignore
from hermes_cli import kanban_discord_approvals as kap  # type: ignore

ENV_PATH = HOME / ".env"


def resolve_db_path() -> Path:
    raw = (os.getenv("KANBAN_DB_PATH") or "").strip()
    if not raw or raw.lower() == "default":
        return HOME / "kanban.db"
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = HOME / path
    return path


def resolve_state_path() -> Path:
    raw = (os.getenv("KANBAN_DISCORD_LOG_STATE") or "").strip()
    if not raw:
        return HOME / "state" / "kanban_discord_log_state.json"
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = HOME / path
    return path


DB_PATH = resolve_db_path()
STATE_PATH = resolve_state_path()
CHANNEL_NAME = os.getenv("DISCORD_KANBAN_LOG_CHANNEL_NAME", "kanban-log")
PROJECT_CHANNEL_NAME = os.getenv("DISCORD_KANBAN_PROJECT_LOG_CHANNEL_NAME", "kanban-project-logs")
RED_CHANNEL_NAME = os.getenv("DISCORD_RED_KANBAN_LOG_CHANNEL_NAME", "red-kanban-log")
RED_ASSIGNEES = {
    "red-antonetta",
    "red-scribe",
    "red-recon",
    "red-labops",
    "red-reporter",
    "red-reviewer",
    "red-exploitdev",
    "red-toolsmith",
}
SYNTHETIC_CREATED_BY = {"weak-handoff-replay", "sparse-handoff-robustness", "threshold-supplement"}
SYNTHETIC_ROOT_PREFIXES = ("root ",)
POLL_SECONDS = int(os.getenv("KANBAN_LOG_POLL_SECONDS", "5"))
THREAD_MODE = os.getenv("KANBAN_DISCORD_THREAD_MODE", "1").lower() not in {"0", "false", "no"}
THREAD_DEBUG = os.getenv("KANBAN_DISCORD_THREAD_DEBUG", "").lower() in {"1", "true", "yes"}
NOISE_KINDS = {"claimed", "spawned", "heartbeat"}
DANGER = {"blocked", "failed", "crashed", "timed_out", "spawn_failed", "gave_up"}
TERMINAL_STATUSES = {"done", "archived"}
PROJECT_WORDS = ("project", "orchestrat", "umbrella", "fan-in", "fanout", "fan-out", "milestone", "phase")


def load_env(path: Path):
    if not path.exists():
        return
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def discord_api(method, endpoint, body=None):
    token = os.environ["DISCORD_BOT_TOKEN"]
    cmd = [
        "curl", "-sS", "-X", method,
        "-H", f"Authorization: Bot {token}",
        "-H", "Content-Type: application/json",
        "-H", "User-Agent: DiscordBot (https://github.com/NousResearch/hermes-agent, 1.0)",
    ]
    if body is not None:
        cmd += ["--data", json.dumps(body)]
    cmd.append(f"https://discord.com/api/v10{endpoint}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(f"Discord API curl failed for {method} {endpoint}: {proc.stderr.strip()}")
    raw = proc.stdout.strip()
    if not raw:
        return {}
    data = json.loads(raw)
    if isinstance(data, dict) and "code" in data and "message" in data and str(data.get("code")) not in ("0",):
        raise RuntimeError(f"Discord API {method} {endpoint} returned error code {data.get('code')}: {data.get('message')}")
    return data


def post(channel_id, content, dry_run=False, components=None):
    body = content if isinstance(content, dict) else {"content": content}
    if components is not None:
        body["components"] = components
    text = str(body.get("content") or "")
    if len(text) > 1900:
        body["content"] = text[:1850] + "\n…[truncated]"
    if dry_run:
        print(f"DRY-RUN post channel={channel_id}:\n{body.get('content', '')}\ncomponents={json.dumps(body.get('components', []), ensure_ascii=False)}\n---")
        return {"id": f"dry-msg-{int(time.time() * 1000)}", "channel_id": channel_id}
    return discord_api("POST", f"/channels/{channel_id}/messages", body)


_CHANNEL_CACHE = {}


def get_channel(channel_id, dry_run=False):
    if dry_run:
        return {"id": channel_id, "type": 0, "name": f"dry-{channel_id}"}
    if channel_id not in _CHANNEL_CACHE:
        _CHANNEL_CACHE[channel_id] = discord_api("GET", f"/channels/{channel_id}")
    return _CHANNEL_CACHE[channel_id]


def create_thread(parent_channel_id, name, message, dry_run=False, applied_tags=None):
    parent = get_channel(parent_channel_id, dry_run=dry_run)
    parent_type = int(parent.get("type", 0) or 0)
    body = {"name": name[:90], "auto_archive_duration": 10080, "message": {"content": message}}
    # Text channels need an explicit public-thread type. Forum/media channels
    # create posts via the same endpoint but reject/ignore the text-thread type;
    # the created post's thread id is still returned as `id` and can be reused
    # by the project_threads state map.
    if parent_type not in {15, 16}:  # GUILD_FORUM / GUILD_MEDIA
        body["type"] = 11
    elif applied_tags:
        body["applied_tags"] = list(applied_tags)
    if dry_run:
        thread_id = f"dry-thread-{parent_channel_id}-{abs(hash(name)) % 100000}"
        surface = "forum-post" if parent_type in {15, 16} else "thread"
        print(f"DRY-RUN create_{surface} parent={parent_channel_id} name={name}:\n{message}\n---")
        return {"id": thread_id, "parent_id": parent_channel_id, "name": name}
    return discord_api("POST", f"/channels/{parent_channel_id}/threads", body)


def ensure_channel():
    return ensure_named_channel(
        configured=os.getenv("DISCORD_KANBAN_LOG_CHANNEL") or os.getenv("DISCORD_KANBAN_LOG_CHANNEL_ID"),
        name=CHANNEL_NAME,
        topic="Hermes Kanban lifecycle log: project thread milestones plus compact one-off events.",
        required_label="DISCORD_KANBAN_LOG_CHANNEL",
    )


def ensure_red_channel(default_channel_id):
    configured = os.getenv("DISCORD_RED_KANBAN_LOG_CHANNEL") or os.getenv("DISCORD_RED_KANBAN_LOG_CHANNEL_ID")
    if not configured:
        return default_channel_id
    return ensure_named_channel(
        configured=configured,
        name=RED_CHANNEL_NAME,
        topic="Hermes Red-lane Kanban lifecycle log. Scope-sensitive red project milestones only.",
        required_label="DISCORD_RED_KANBAN_LOG_CHANNEL",
    )


def ensure_project_channel(default_channel_id):
    configured = os.getenv("DISCORD_KANBAN_PROJECT_LOG_CHANNEL") or os.getenv("DISCORD_KANBAN_PROJECT_LOG_CHANNEL_ID")
    if configured:
        return configured
    return ensure_named_channel(
        configured=None,
        name=PROJECT_CHANNEL_NAME,
        topic="Project-scoped Hermes Kanban logs. One thread per umbrella project; completion pings happen inside the project thread.",
        required_label="DISCORD_KANBAN_PROJECT_LOG_CHANNEL",
    )


def ensure_named_channel(configured=None, name=None, topic=None, required_label="DISCORD_KANBAN_LOG_CHANNEL"):
    if configured:
        return configured
    home_channel = os.environ.get("DISCORD_HOME_CHANNEL") or os.environ.get("DISCORD_SYSTEM_CHANNEL")
    if not home_channel:
        raise RuntimeError(f"Need DISCORD_HOME_CHANNEL or {required_label} in ~/.hermes/.env")
    home = discord_api("GET", f"/channels/{home_channel}")
    guild_id = home.get("guild_id")
    if not guild_id:
        raise RuntimeError("Configured Discord home channel is not a guild channel; cannot infer guild")
    channels = discord_api("GET", f"/guilds/{guild_id}/channels")
    for ch in channels:
        if ch.get("name") == name and ch.get("type") in {0, 15, 16}:
            return ch["id"]
    created = discord_api("POST", f"/guilds/{guild_id}/channels", {"name": name, "type": 0, "topic": topic})
    return created["id"]


def load_state():
    if STATE_PATH.exists():
        try:
            state = json.loads(STATE_PATH.read_text())
            state.setdefault("components", {})
            state.setdefault("task_aliases", {})
            state.setdefault("red_thread_posts", {})
            return state
        except Exception:
            pass
    return {"last_event_id": 0, "components": {}, "task_aliases": {}, "red_thread_posts": {}}


def save_state(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(STATE_PATH)


def parse_payload(raw):
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {"raw": raw}


def parse_task_metadata(payload):
    metadata = payload.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}
    return metadata if isinstance(metadata, dict) else {}


def extract_thread_id(task, payload):
    metadata = parse_task_metadata(payload)
    for key in ("discord_thread_id", "thread_id"):
        value = metadata.get(key)
        if value:
            return str(value)
    body = task.get("body") or ""
    import re
    for pattern in (
        r"Discord thread id:\s*([0-9]{6,})",
        r"Discord thread:\s*`?([0-9]{6,})`?",
    ):
        m = re.search(pattern, body, flags=re.I)
        if m:
            return m.group(1)
    return None


def summarize_findings(metadata):
    findings = metadata.get("findings") or metadata.get("decisions") or metadata.get("outputs") or []
    if isinstance(findings, str):
        findings = [findings]
    if not isinstance(findings, list):
        return []
    out = []
    for item in findings:
        text = str(item).strip()
        if text:
            out.append(text.rstrip('.'))
        if len(out) >= 3:
            break
    return out


def suggest_next_action(task, payload):
    metadata = parse_task_metadata(payload)
    for key in ("recommended_next_skill", "next_wave_priority", "follow_on_branch_executed"):
        value = metadata.get(key)
        if value:
            return str(value).strip()
    summary = payload.get("summary") or payload.get("result") or ""
    if "queued successor orchestrator" in summary.lower():
        return "successor orchestrator queued to fan out the next wave"
    return f"review downstream handoff from `{task.get('id')}` and use it for the next justified branch"


def format_red_thread_update(task, ev, payload):
    metadata = parse_task_metadata(payload)
    title = (task.get("title") or task.get("id") or "task").strip()
    assignee = (task.get("assignee") or payload.get("assignee") or "unknown").strip()
    kind = ev["kind"]
    lines = [f"🧠 **Worker update:** {title}", f"👤 **Owner:** `{assignee}`"]
    if kind == "created":
        parents = payload.get("parents") or []
        if isinstance(parents, str):
            parents = [parents]
        lines.append("🚦 **Stage status:** queued")
        if parents:
            lines.append(f"🔗 **Depends on:** {', '.join(str(p) for p in parents[:4])}")
        lines.append("📌 **Why it exists:** new branch/stage created and waiting to start")
    elif kind == "claimed":
        lines.append("▶️ **Stage status:** started")
        run_id = payload.get("run_id")
        if run_id:
            lines.append(f"🆔 **Run id:** `{run_id}`")
        lines.append("⚙️ **What is happening:** worker claimed the branch and execution is underway")
    elif kind == "completed":
        summary = (payload.get("summary") or payload.get("result") or "completed").strip()
        lines.append("✅ **Stage status:** finished")
        lines.append(f"📝 **Summary:** {summary}")
        findings = summarize_findings(metadata)
        if findings:
            lines.append("🔍 **Findings:**")
            for item in findings:
                lines.append(f"- {item}")
        lines.append(f"➡️ **Suggested next action:** {suggest_next_action(task, payload)}")
    elif kind == "blocked":
        reason = payload.get("reason") or payload.get("summary") or payload.get("error") or "blocked"
        lines.append("⛔ **Stage status:** blocked")
        lines.append(f"🚧 **Blocked:** {reason}")
        lines.append("➡️ **Suggested next action:** resolve the blocker or route the missing reviewer/fan-in step, then resume the workflow")
    else:
        return None
    return "\n".join(lines)[:1800]


def inherited_component_thread_id(con, task_id):
    for linked_task in component_tasks(con, task_id):
        thread_id = extract_thread_id(linked_task, {})
        if thread_id:
            return thread_id
    return None


def maybe_post_red_thread_update(con, state, task, ev, payload, dry_run=False):
    if ev["kind"] not in {"created", "claimed", "completed", "blocked"}:
        return
    if not is_red_task(task):
        return
    thread_id = extract_thread_id(task, payload) or inherited_component_thread_id(con, ev["task_id"])
    if not thread_id:
        return
    posted = state.setdefault("red_thread_posts", {})
    event_key = str(ev["id"])
    if posted.get(event_key):
        return
    body = format_red_thread_update(task, ev, payload)
    if not body:
        return
    post(thread_id, body, dry_run=dry_run)
    posted[event_key] = {"thread_id": thread_id, "task_id": ev["task_id"], "kind": ev["kind"]}


def red_thread_target(con, task, payload, task_id):
    if not is_red_task(task):
        return None
    return extract_thread_id(task, payload) or inherited_component_thread_id(con, task_id)


def task_lookup(con, task_id):
    con.row_factory = sqlite3.Row
    row = con.execute("select id,title,body,assignee,status,priority,created_at,started_at,completed_at,current_run_id,created_by from tasks where id=?", (task_id,)).fetchone()
    return dict(row) if row else {"id": task_id, "title": "unknown", "body": "", "assignee": "unknown", "status": "unknown", "created_at": 0, "priority": 0, "created_by": ""}


def is_synthetic_test_task(task):
    created_by = (task.get("created_by") or "").strip()
    title = (task.get("title") or "").strip().lower()
    if created_by in SYNTHETIC_CREATED_BY:
        return True
    return any(title.startswith(prefix) for prefix in SYNTHETIC_ROOT_PREFIXES)


def is_red_task(task):
    assignee = (task.get("assignee") or "").strip()
    return assignee in RED_ASSIGNEES or assignee.startswith("red-")


def component_task_ids(con, task_id):
    seen = {task_id}
    q = deque([task_id])
    while q:
        cur = q.popleft()
        rows = con.execute("select parent_id, child_id from task_links where parent_id=? or child_id=?", (cur, cur)).fetchall()
        for row in rows:
            parent_id = row["parent_id"] if isinstance(row, sqlite3.Row) else row[0]
            child_id = row["child_id"] if isinstance(row, sqlite3.Row) else row[1]
            for nxt in (parent_id, child_id):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
    return seen


def component_tasks(con, task_id):
    ids = sorted(component_task_ids(con, task_id))
    return [task_lookup(con, tid) for tid in ids]


def component_root(con, tasks):
    ids = {t["id"] for t in tasks}
    child_rows = con.execute("select child_id from task_links where child_id in (%s)" % ",".join("?" for _ in ids), tuple(ids)).fetchall() if ids else []
    children = {r["child_id"] if isinstance(r, sqlite3.Row) else r[0] for r in child_rows}
    roots = [t for t in tasks if t["id"] not in children] or tasks
    roots.sort(key=lambda t: (-(t.get("priority") or 0), t.get("created_at") or 0, t.get("id") or ""))
    return roots[0]


def project_trigger(task, payload, component_size):
    if kpm.extract_task_project_metadata(task, payload):
        return True
    if component_size > 1:
        return True
    metadata = payload.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}
    if payload.get("project_completion") or metadata.get("project_completion"):
        return True
    haystack = f"{task.get('title') or ''}\n{task.get('body') or ''}".lower()
    return any(word in haystack for word in PROJECT_WORDS)


def thread_title(root, tasks):
    # New project-thread model: Discord thread name matches the Project Hub
    # title exactly when project metadata is present. Legacy untagged project
    # graphs still get a `kanban:` prefix so they remain obviously mechanical.
    meta = kpm.extract_task_project_metadata(root)
    title = (meta.get("project_title") or root.get("title") or root.get("id") or "Kanban project").strip()
    if meta.get("project_hub_slug"):
        return title[:90]
    return f"kanban: {title}"[:90]


def choose_component_channel(tasks, general_channel_id, red_channel_id):
    return red_channel_id if any(is_red_task(t) for t in tasks) else general_channel_id


def refresh_component_state(con, state, root, tasks, channel_id):
    root_id = root["id"]
    comp = state.setdefault("components", {}).setdefault(root_id, {})
    comp.setdefault("thread_id", None)
    comp["channel_id"] = channel_id
    comp["title"] = thread_title(root, tasks)
    comp["task_ids"] = sorted(t["id"] for t in tasks)
    comp.setdefault("status", "open")
    comp.setdefault("completed_ping_sent", False)
    for tid in comp["task_ids"]:
        state.setdefault("task_aliases", {})[tid] = root_id
    return comp


def project_terminal_state(tasks):
    active = [t for t in tasks if t.get("status") != "archived"]
    if active and all(t.get("status") == "done" for t in active):
        return "completed"
    if any(t.get("status") == "blocked" for t in active):
        return "blocked"
    if any(t.get("status") in {"failed", "crashed", "timed_out", "spawn_failed"} for t in active):
        return "failed"
    return None


def explicit_project_completion(payload):
    metadata = payload.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}
    return bool(payload.get("project_completion") or metadata.get("project_completion"))


def allowed_user_mentions():
    users = [u.strip() for u in os.environ.get("DISCORD_ALLOWED_USERS", "").replace(",", " ").split() if u.strip()]
    return " ".join(f"<@{u}>" for u in users)


def maybe_post_human_approval(task, ev, payload, target_channel_id, project_ctx=None, dry_run=False):
    if not kap.is_human_approval_gate(task, ev["kind"], payload):
        return False
    req = kap.build_approval_request(task, {**payload, "run_id": ev.get("run_id")}, project_ctx)
    body = kap.build_message_payload(req, allowed_user_mentions())
    post(target_channel_id, body, dry_run=dry_run)
    return True


def recent_task_failures(con, task_id):
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        select outcome, status, error, summary, profile, started_at, ended_at
        from task_runs
        where task_id=?
        order by started_at desc
        limit 5
        """,
        (task_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def recovery_suggestions(task, kind, payload, failures):
    title = (task.get("title") or "").lower()
    assignee = task.get("assignee") or "unknown"
    outcomes = [f.get("outcome") or f.get("status") for f in failures if f.get("outcome") or f.get("status")]
    repeated = len([o for o in outcomes if o in {"crashed", "timed_out", "spawn_failed", "failed", "blocked"}]) >= 2
    gave_up = kind == "gave_up" or payload.get("trigger_outcome") or payload.get("failures")
    if not (repeated or gave_up or kind in {"blocked", "spawn_failed"}):
        return []
    suggestions = []
    if kind == "blocked":
        suggestions.extend([
            "answer the blocker in-thread, then unblock/requeue the same task",
            "create a smaller recovery task with the blocked task id in context",
            "if the blocker is a missing decision/credential, resolve that explicitly before retrying",
        ])
    elif "spawn_failed" in outcomes or kind == "spawn_failed":
        suggestions.extend([
            f"smoke-test profile `{assignee}` directly (`hermes --profile {assignee} chat -Q -q ...`)",
            "check profile auth/config/toolsets and missing skill names",
            "after profile is healthy, reset/requeue the task instead of cloning blindly",
        ])
    elif "timed_out" in outcomes:
        suggestions.extend([
            "split the work into a smaller recovery task with tighter acceptance criteria",
            "raise max_runtime_seconds only if progress/heartbeat evidence shows it was genuinely still working",
            "ask the worker to write partial artifacts early so retries can resume instead of restart",
        ])
    elif "crashed" in outcomes or kind == "crashed" or kind == "gave_up":
        suggestions.extend([
            f"smoke-test profile `{assignee}` outside Kanban to separate profile crash from task logic",
            "inspect the latest run/gateway logs for the first traceback or missing dependency",
            "create a bounded recovery task carrying this task id, prior errors, and any parent handoff",
        ])
    else:
        suggestions.extend([
            "inspect task comments/runs before retrying",
            "make a smaller recovery task if the same failure repeats",
            "route to a more appropriate specialist if the assignee is mismatched",
        ])
    if any(word in title for word in ("discord", "gateway", "kanban-log", "watcher")):
        suggestions.append("for Discord/gateway work, verify env channel ids and send a test message before requeueing")
    out = []
    for s in suggestions:
        if s not in out:
            out.append(s)
    return out[:4]


def format_event(con, ev, compact=False):
    task = task_lookup(con, ev["task_id"])
    payload = parse_payload(ev.get("payload"))
    kind = ev["kind"]
    title = task.get("title") or "untitled"
    assignee = task.get("assignee") or "unassigned"
    status = task.get("status") or "unknown"
    failures = recent_task_failures(con, ev["task_id"]) if (kind in DANGER or payload.get("outcome") in DANGER) else []
    suggestions = recovery_suggestions(task, kind, payload, failures)
    prefix = "Kanban" if not compact else "•"
    lines = [
        f"{prefix} {kind}: {title}",
        f"task: `{ev['task_id']}` | assignee: `{assignee}` | status: `{status}` | run: `{ev.get('run_id') or '-'}`",
    ]
    if kind == "created":
        parents = payload.get("parents") or []
        lines.append(f"created for `{payload.get('assignee', assignee)}`; parents: {', '.join(parents) if parents else 'none'}")
    elif kind == "claimed":
        lines.append(f"claimed by `{payload.get('lock', 'unknown')}`")
    elif kind == "spawned":
        lines.append(f"spawned pid `{payload.get('pid', 'unknown')}`")
    elif kind == "completed":
        summary = payload.get("summary") or payload.get("result") or "completed"
        lines.append(f"summary: {summary}")
    elif kind == "commented":
        body = payload.get("body") or payload.get("comment") or payload.get("summary") or "comment added"
        author = payload.get("author") or "unknown"
        lines.append(f"comment by `{author}`: {str(body)[:700]}")
    elif kind == "blocked":
        reason = payload.get("reason") or payload.get("summary") or payload.get("error") or "blocked"
        lines.append(f"action needed: {reason}")
    elif kind in DANGER:
        lines.append(f"issue: {payload.get('error') or payload.get('reason') or payload.get('outcome') or json.dumps(payload)[:400]}")
    else:
        small = {k: v for k, v in payload.items() if k in ("status", "outcome", "summary", "error", "reason")}
        if small:
            lines.append("details: " + json.dumps(small, ensure_ascii=False)[:700])
    if suggestions:
        lines.append("suggested next moves:")
        for idx, suggestion in enumerate(suggestions, 1):
            lines.append(f"{idx}. {suggestion}")
    return "\n".join(lines)


def should_skip_thread_event(kind):
    return kind in NOISE_KINDS and not THREAD_DEBUG


def route_event(con, state, ev, channel_id, project_channel_id, red_channel_id, dry_run=False):
    task = task_lookup(con, ev["task_id"])
    if is_synthetic_test_task(task):
        return "skipped-synthetic-test-task"
    payload = parse_payload(ev.get("payload"))
    if ev["kind"] == "archived" or payload.get("discord_silent"):
        return "skipped-silent-archive"
    project_ctx = kpm.resolve_project_context(con, ev["task_id"], payload)
    if project_ctx:
        thread_key = kpm.project_thread_key(project_ctx)
        thread_id = project_ctx.get("discord_thread_id") or state.setdefault("project_threads", {}).get(thread_key)
        if not thread_id:
            # Discord requires a starter message for public-thread creation.
            # Keep it minimal; all real progress stays in the thread.
            thread = create_thread(
                project_channel_id,
                str(project_ctx.get("project_title") or task.get("title") or project_ctx["project_hub_slug"]),
                kpm.format_project_thread_starter(project_ctx),
                dry_run=dry_run,
            )
            thread_id = thread["id"]
            state.setdefault("project_threads", {})[thread_key] = thread_id
        if kpm.should_post_project_thread_event(ev["kind"], payload):
            body = kpm.format_project_thread_update(task, ev["kind"], payload, project_ctx)
            post(thread_id, body, dry_run=dry_run)
        if maybe_post_human_approval(task, ev, payload, thread_id, project_ctx, dry_run=dry_run):
            return "posted-human-approval"
        return "posted-project-thread" if kpm.should_post_project_thread_event(ev["kind"], payload) else "skipped-project-noise"

    if maybe_post_human_approval(task, ev, payload, channel_id, None, dry_run=dry_run):
        return "posted-human-approval"

    maybe_post_red_thread_update(con, state, task, ev, payload, dry_run=dry_run)
    if red_thread_target(con, task, payload, ev["task_id"]):
        return "posted-red-thread"
    tasks = component_tasks(con, ev["task_id"])
    root = component_root(con, tasks)
    target_channel_id = choose_component_channel(tasks, channel_id, red_channel_id)
    is_project_event = THREAD_MODE and project_trigger(root if root else task, payload, len(tasks))
    if not is_project_event:
        if ev["kind"] in NOISE_KINDS and not THREAD_DEBUG:
            return "skipped-flat-noise"
        post(target_channel_id, format_event(con, ev), dry_run=dry_run)
        return "posted-flat"

    if target_channel_id == channel_id:
        target_channel_id = project_channel_id

    comp = refresh_component_state(con, state, root, tasks, target_channel_id)
    if not comp.get("thread_id"):
        intro = f"Kanban project opened: {comp['title']}\nroot: `{root['id']}` | tasks: {len(comp['task_ids'])} | lane: {'red' if target_channel_id == red_channel_id and red_channel_id != channel_id else 'general'}"
        thread = create_thread(target_channel_id, comp["title"], intro, dry_run=dry_run)
        comp["thread_id"] = thread["id"]
        # No parent-channel announcements. The parent channel is not a progress
        # feed; the thread itself is the live room.

    if not should_skip_thread_event(ev["kind"]):
        post(comp["thread_id"], format_event(con, ev, compact=True), dry_run=dry_run)

    terminal = None
    if ev["kind"] == "completed":
        terminal = "completed" if explicit_project_completion(payload) else project_terminal_state(tasks)
    elif ev["kind"] == "blocked":
        terminal = "blocked"
    elif ev["kind"] in DANGER:
        terminal = "failed"
    if terminal and comp.get("status") != terminal:
        comp["status"] = terminal
        parent_msg = f"Kanban project {terminal}: {comp['title']} | root `{root['id']}` | tasks: {len(comp['task_ids'])}"
        if terminal in {"blocked", "failed"}:
            post(comp["thread_id"], parent_msg, dry_run=dry_run)
        if terminal == "completed" and not comp.get("completed_ping_sent"):
            thread_msg = f"✅ Final: {comp['title']}"
            post(comp["thread_id"], thread_msg, dry_run=dry_run)
            comp["completed_ping_sent"] = True
    return "posted-thread"


def run_once(state, channel_id, project_channel_id, red_channel_id, dry_run=False, limit=50):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "select id, task_id, run_id, kind, payload, created_at from task_events where id > ? order by id asc limit ?",
        (int(state.get("last_event_id", 0)), limit),
    ).fetchall()
    for row in rows:
        ev = dict(row)
        try:
            route_event(con, state, ev, channel_id, project_channel_id, red_channel_id, dry_run=dry_run)
        except Exception as e:
            print(f"post failed for event {ev['id']}: {e}", file=sys.stderr, flush=True)
        state["last_event_id"] = ev["id"]
        if not dry_run:
            save_state(state)
    return len(rows)


def initialize_state(state, channel_id, dry_run=False):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    if not state.get("last_event_id"):
        max_id = con.execute("select coalesce(max(id),0) from task_events").fetchone()[0]
        state["last_event_id"] = max_id
        if not dry_run:
            save_state(state)
        post(channel_id, f"Kanban log online. Watching `{DB_PATH}` from event id `{max_id}`. Project graphs will use Discord threads; one-off tasks remain compact.", dry_run=dry_run)


def build_smoke_db(path):
    con = sqlite3.connect(path)
    con.executescript(
        """
        create table tasks (id text primary key, title text not null, body text, assignee text, status text not null, priority integer default 0, created_by text, created_at integer not null, started_at integer, completed_at integer, workspace_kind text default 'scratch', workspace_path text, claim_lock text, claim_expires integer, tenant text, result text, idempotency_key text, spawn_failures integer default 0, worker_pid integer, last_spawn_error text, max_runtime_seconds integer, last_heartbeat_at integer, current_run_id integer, workflow_template_id text, current_step_key text, skills text, consecutive_failures integer default 0, last_failure_error text, max_retries integer);
        create table task_links (parent_id text not null, child_id text not null, primary key (parent_id, child_id));
        create table task_events (id integer primary key, task_id text not null, run_id integer, kind text not null, payload text, created_at integer not null);
        create table task_runs (id integer primary key, task_id text not null, profile text, step_key text, status text not null, claim_lock text, claim_expires integer, worker_pid integer, max_runtime_seconds integer, last_heartbeat_at integer, started_at integer not null, ended_at integer, outcome text, summary text, metadata text, error text);
        """
    )
    rows = [
        ("p1", "Demo umbrella project", "orchestration project", "antonetta", "done", 10, 1),
        ("c1", "Demo worker", "", "forge", "done", 0, 2),
        ("c2", "Demo red worker", "Discord thread id: 777777", "red-recon", "done", 0, 3),
        ("r1", "Standalone red thread task", "Discord thread id: 888888", "red-antonetta", "blocked", 0, 4),
        ("one", "Small one-off", "", "forge", "done", 0, 5),
        ("rev", "Autonomous review gate", "", "forge", "blocked", 0, 6),
        ("hum", "Human approval gate", "", "forge", "blocked", 0, 7),
    ]
    con.executemany("insert into tasks (id,title,body,assignee,status,priority,created_at) values (?,?,?,?,?,?,?)", rows)
    con.executemany("insert into task_links (parent_id, child_id) values (?,?)", [("p1", "c1"), ("p1", "c2")])
    events = [
        (1, "p1", None, "created", json.dumps({"assignee": "antonetta", "parents": []}), 1),
        (2, "c1", 7, "spawned", json.dumps({"pid": 123}), 2),
        (3, "c1", 7, "completed", json.dumps({"summary": "worker done"}), 3),
        (4, "c2", 8, "created", json.dumps({"assignee": "red-recon", "parents": ["p1"]}), 4),
        (5, "c2", 8, "claimed", json.dumps({"run_id": 8}), 5),
        (6, "c2", 8, "completed", json.dumps({"summary": "red worker done", "metadata": {"findings": ["validated Discord lifecycle formatting"]}}), 6),
        (7, "r1", 9, "created", json.dumps({"assignee": "red-antonetta", "parents": []}), 7),
        (8, "r1", 9, "claimed", json.dumps({"run_id": 9}), 8),
        (9, "r1", 9, "blocked", json.dumps({"reason": "need reviewer signoff"}), 9),
        (10, "p1", 10, "completed", json.dumps({"summary": "project done", "metadata": {"project_completion": True}}), 10),
        (11, "one", 11, "completed", json.dumps({"summary": "one-off done"}), 11),
        (12, "rev", 12, "blocked", json.dumps({"reason": "review-required: implementation handoff", "metadata": {"review_required": True}}), 12),
        (13, "hum", 13, "blocked", json.dumps({"reason": "human-gate: approve live decision", "metadata": {"human_approval_required": True, "what_is_approved": "continuing past the human gate", "if_approved": "the task unblocks", "risk_rollback": "deny keeps it blocked"}}), 13),
    ]
    con.executemany("insert into task_events (id, task_id, run_id, kind, payload, created_at) values (?,?,?,?,?,?)", events)
    con.commit()
    con.close()


def smoke_test():
    global DB_PATH, STATE_PATH
    old_db, old_state = DB_PATH, STATE_PATH
    with tempfile.TemporaryDirectory() as td:
        DB_PATH = Path(td) / "kanban.db"
        STATE_PATH = Path(td) / "state.json"
        build_smoke_db(DB_PATH)
        os.environ["DISCORD_ALLOWED_USERS"] = "12345"
        state = {"last_event_id": 0, "components": {}, "task_aliases": {}}
        count = run_once(state, "general-channel", "project-channel", "red-channel", dry_run=True, limit=20)
        comp = state["components"].get("p1")
        assert count == 13, count
        assert comp and comp["thread_id"], state
        assert comp["channel_id"] == "red-channel", comp
        assert comp["completed_ping_sent"] is True, comp
        assert state["task_aliases"]["c1"] == "p1", state
        assert "one" not in state["task_aliases"], state
        assert "r1" not in state["task_aliases"], state
        red_posts = sorted(item["kind"] for item in state["red_thread_posts"].values())
        assert red_posts == ["blocked", "claimed", "claimed", "completed", "created", "created"], red_posts
    DB_PATH, STATE_PATH = old_db, old_state
    print("smoke-test ok: grouping, red routing, terminal detection, and mention policy passed")


def main():
    parser = argparse.ArgumentParser(description="Mirror Hermes Kanban events to Discord with project-thread grouping.")
    parser.add_argument("--dry-run", action="store_true", help="print intended Discord writes instead of sending them")
    parser.add_argument("--once", action="store_true", help="process available events once and exit")
    parser.add_argument("--smoke-test", action="store_true", help="run deterministic local smoke test without Discord")
    args = parser.parse_args()
    if args.smoke_test:
        smoke_test()
        return

    load_env(ENV_PATH)
    if not args.dry_run and not os.environ.get("DISCORD_BOT_TOKEN"):
        raise RuntimeError("Missing DISCORD_BOT_TOKEN")
    channel_id = os.getenv("DISCORD_KANBAN_LOG_CHANNEL") or os.getenv("DISCORD_KANBAN_LOG_CHANNEL_ID") or ("dry-general" if args.dry_run else ensure_channel())
    project_channel_id = os.getenv("DISCORD_KANBAN_PROJECT_LOG_CHANNEL") or os.getenv("DISCORD_KANBAN_PROJECT_LOG_CHANNEL_ID") or ("dry-project" if args.dry_run else ensure_project_channel(channel_id))
    red_channel_id = os.getenv("DISCORD_RED_KANBAN_LOG_CHANNEL") or os.getenv("DISCORD_RED_KANBAN_LOG_CHANNEL_ID") or channel_id
    if not args.dry_run:
        channel_id = ensure_channel()
        red_channel_id = ensure_red_channel(channel_id)
    state = load_state()
    initialize_state(state, channel_id, dry_run=args.dry_run)

    if args.once:
        run_once(state, channel_id, project_channel_id, red_channel_id, dry_run=args.dry_run)
        return

    while True:
        run_once(state, channel_id, project_channel_id, red_channel_id, dry_run=args.dry_run)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
