#!/usr/bin/env python3
"""KENSEI Triage Processor - no_agent version, v3.0

Scans all kanban boards for triage tasks, classifies them, and updates status.
Writes NEEDS-HUMAN tasks to a pending-investigation file for the investigator cron.
SILENT — does NOT notify directly. All notification handled by kensei-triage-investigator.

Uses correct CLI: promote (triage→todo), block (triage→blocked).
"""
import subprocess
import json
import os
import sys
import time
import sqlite3
from datetime import datetime, timedelta

# Cross-process write lock — prevents WAL checkpoint races
# Cron workdir is the canonical runtime checkout.  Prefer an explicit
# relocation override, then cwd, rather than the mutable ~/.hermes/scripts
# directory that contains this compatibility entrypoint.
_RUNTIME_ROOT = os.path.abspath(os.environ.get("HERMES_AGENT_ROOT", os.getcwd()))
sys.path.insert(0, _RUNTIME_ROOT)
sys.path.insert(0, os.path.join(_RUNTIME_ROOT, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from kanban_write_lock import write_lock
except ImportError:
    write_lock = None

# W1-G (Batch 1): board DB identities resolved via _board_compat so retired
# slugs (default->core, ops->security-ops, content-lead->content) map to the
# current canonical DB path. Semantic board labels preserved: BOARDS keeps
# the legacy slugs for CLI routing; _get_board_db resolves to the live DB.
import _board_compat
BOARDS = ['ops', 'research', 'apps', 'content-lead', 'default']
_HERMES_HOME = os.environ.get('HERMES_HOME', os.path.expanduser('~/.hermes'))
STATE_FILE = os.path.join(_HERMES_HOME, 'data', 'triage-state.json')
PENDING_FILE = os.path.join(_HERMES_HOME, 'data', 'pending-investigation.json')

# P13 isolation: when --dry-run is passed, every write path (task UPDATE,
# block_task, promote_task, save_json, _set_task_tier, _set_task_assignee,
# _promote_to_pipeline) is suppressed. Read paths run unchanged so the
# classification and routing decisions are still computed and printed.
_DRY_RUN = False

# WS-2 board routing: keyword → board + assignee mapping derived
# from governance/routing/goal-subgoal-routing-map.md
ROUTING_MAP = {
    # code/backend → apps board, octacon
    'apps': {
        'keywords': ['code', 'implement', 'build', 'refactor', 'migrate', 'debug',
                     'api', 'backend', 'frontend', 'component', 'database', 'schema',
                     'test', 'fix bug', 'pr ', 'pull request', 'deploy'],
        'default_assignee': 'octacon',
    },
    # content → content-lead board, ceecee
    'content-lead': {
        'keywords': ['content', 'post', 'draft', 'article', 'brand', 'voice',
                     'social', 'copy', 'tweet', 'linkedin', 'blog'],
        'default_assignee': 'ceecee',
    },
    # ops/infra → ops board, wesker
    'ops': {
        'keywords': ['ops', 'infra', 'docker', 'deploy', 'vps', 'nginx', 'backup',
                     'security', 'cron', 'gateway', 'restart', 'health', 'disk',
                     'memory', 'service', 'systemctl'],
        'default_assignee': 'wesker',
    },
    # research → research board, remii
    'research': {
        'keywords': ['research', 'analysis', 'report', 'market', 'scan',
                     'digest', 'signal', 'trend', 'survey', 'benchmark',
                     'evaluate', 'investigate'],
        'default_assignee': 'remii',
    },
}


def _route_task(title, body):
    """Route a task to the right board and assignee based on keyword matching.
    Returns (board, assignee) or (None, None) if no match."""
    combined = ((title or '') + ' ' + (body or '')).lower()
    for board, cfg in ROUTING_MAP.items():
        for kw in cfg['keywords']:
            if kw in combined:
                return board, cfg['default_assignee']
    return None, None

def _get_board_db(board):
    """Get the db_path for a board slug via _board_compat (reversible legacy fallback)."""
    return _board_compat.resolve_board_db_str(board)


def _set_task_tier(board, task_id, tier):
    """Persist the tier classification on a task."""
    db_path = _get_board_db(board)
    if not os.path.exists(db_path):
        return
    try:
        conn = sqlite3.connect(db_path)
        if write_lock is not None:
            with write_lock(conn):
                conn.execute("UPDATE tasks SET tier = ? WHERE id = ?", (tier, task_id))
                conn.commit()
        else:
            conn.execute("UPDATE tasks SET tier = ? WHERE id = ?", (tier, task_id))
            conn.commit()
        conn.close()
    except Exception:
        pass


def run_hermes_list(board, status):
    """Run hermes kanban list --status <status> --json for a board."""
    if _DRY_RUN:
        return []
    cmd = ['hermes', 'kanban', '--board', board, 'list', '--status', status, '--json']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return None

def promote_task(task_id, board):
    """Promote a task from triage to todo."""
    cmd = ['hermes', 'kanban', '--board', board, 'promote', task_id]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return result.returncode == 0

def block_task(task_id, board, reason):
    """Block a task with a decision-needed reason."""
    cmd = ['hermes', 'kanban', '--board', board, 'block', task_id]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    blocked_ok = result.returncode == 0

    # Add comment with the reason
    comment_cmd = ['hermes', 'kanban', '--board', board, 'comment', task_id, '--body', reason]
    subprocess.run(comment_cmd, capture_output=True, text=True, timeout=30)

    return blocked_ok

def load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path, data):
    if _DRY_RUN:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

def _set_task_assignee(board, task_id, assignee):
    """Set the assignee on a task via direct SQL."""
    db_path = _get_board_db(board)
    if not os.path.exists(db_path):
        return False
    try:
        conn = sqlite3.connect(db_path)
        if write_lock is not None:
            with write_lock(conn):
                conn.execute("UPDATE tasks SET assignee = ? WHERE id = ? AND (assignee IS NULL OR assignee = '')",
                             (assignee, task_id))
                conn.commit()
        else:
            conn.execute("UPDATE tasks SET assignee = ? WHERE id = ? AND (assignee IS NULL OR assignee = '')",
                         (assignee, task_id))
            conn.commit()
        conn.close()
        return True
    except Exception:
        return False

def _promote_to_pipeline(task_id, board, stage):
    """Promote a task to a pipeline stage (research/prd/spec/council).

    Sets status=stage and pipeline_stage=stage via direct SQL.
    Returns True on success, False on failure.
    """
    db_path = _get_board_db(board)
    if not os.path.exists(db_path):
        return False
    try:
        conn = sqlite3.connect(db_path)
        if write_lock is not None:
            with write_lock(conn):
                conn.execute(
                    "UPDATE tasks SET status = ?, pipeline_stage = ? WHERE id = ?",
                    (stage, stage, task_id)
                )
                conn.commit()
        else:
            conn.execute(
                "UPDATE tasks SET status = ?, pipeline_stage = ? WHERE id = ?",
                (stage, stage, task_id)
            )
            conn.commit()
        conn.close()
        return True
    except Exception:
        return False

def _null_ctx():
    import contextlib
    return contextlib.nullcontext()


def _route_to_backlog(board, task_id):
    """S3 gate (Sahil 30/08/26): move a triage task to backlog instead of
    promoting it. Used for proposal-prefixed tasks and tasks whose parent
    proposal is archived/backlog — they await explicit Sahil approval."""
    db_path = _get_board_db(board)
    if not db_path or not os.path.exists(db_path):
        return False
    try:
        conn = sqlite3.connect(db_path)
        try:
            if write_lock is not None:
                ctx = write_lock(conn)
            else:
                ctx = None
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE tasks SET status='backlog', status_reason=? "
                "WHERE id=? AND status='triage'",
                ("Backlog (S3 gate 30/08/26): proposal/idea task — awaiting explicit Sahil build approval.", task_id),
            )
            moved = conn.execute("SELECT changes()").fetchone()[0]
            if moved:
                conn.execute(
                    "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?,?,?,?)",
                    (task_id, "commented",
                     json.dumps({"by": "triage-processor", "action": "triage -> backlog (S3 proposal/parent gate)"}),
                     int(time.time())))
            conn.commit()
        finally:
            conn.close()
        return True
    except Exception:
        return False


def _parent_not_active(board, task_id):
    """True when the task traces to a parent proposal that is archived or
    backlog — its family must not auto-promote without Sahil's approval."""
    db_path = _get_board_db(board)
    if not db_path or not os.path.exists(db_path):
        return False
    statuses = []
    try:
        conn = sqlite3.connect(db_path)
        # via 'from_decompose_of' creation events
        for row in conn.execute(
                "SELECT t2.status FROM task_events e "
                "JOIN tasks t ON t.id = e.task_id "
                "JOIN tasks t2 ON t2.id = json_extract(e.payload, '$.from_decompose_of') "
                "WHERE e.task_id = ? AND e.kind = 'created'", (task_id,)):
            statuses.append(row[0])
        # via task_links (children of the proposal root)
        for row in conn.execute(
                "SELECT t2.status FROM task_links l JOIN tasks t2 ON t2.id = l.parent_id "
                "WHERE l.child_id = ?", (task_id,)):
            statuses.append(row[0])
        conn.close()
    except Exception:
        pass
    return any(s in ("archived", "backlog") for s in statuses)


def classify_task(title, body, *, pipeline_mode=None):
    """Classify triage task: AUTO-PROMOTE vs NEEDS HUMAN, and tier (fast/full).

    FIX 2026-08-13 (Spectator Mode incident t_9df6f54b): tasks created via
    ``hermes feature create`` carry ``pipeline_mode`` ('full'/'express'). They
    are pipeline-owned work with intake-gated bodies — keyword classification
    must NEVER re-route them (NEEDS-HUMAN blocked them out of the pipeline,
    the gateway auto-decomposer then hijacked them, research+council never
    ran). Pipeline-mode tasks always AUTO-PROMOTE; the main() caller handles
    the pipeline_stage promotion path.
    """
    # Pipeline-owned tasks bypass the keyword classifier entirely.
    if pipeline_mode:
        return 'AUTO-PROMOTE', 'full'
    title_lower = title.lower() if title else ''
    body_lower = (body or '').lower()

    # Tier classification: fast vs full
    # Full-tier keywords: product, feature, research, multi-step, build, implement
    full_keywords = [
        'product', 'feature', 'research', 'multi-step', 'build',
        'implement', 'refactor', 'migrate', 'redesign', 'architecture',
        'pipeline', 'workflow', 'end-to-end', 'deploy app',
    ]
    # Fast-tier keywords: ops, infra, config, cron, quick fix
    fast_keywords = [
        'config', 'cron', 'restart', 'health check', 'fix typo',
        'update doc', 'add column', 'bump version', 'cleanup',
        'orphan', 'stale', 'disk', 'lock', 'skill activation',
    ]
    # Short keywords like 'lock' match substrings (e.g. inside
    # 'blocked'), causing false tier=fast for NEEDS HUMAN tasks.
    SHORT_KWS = {'lock', 'disk'}
    _combined = title_lower + ' ' + body_lower
    tier = 'full'
    for kw in fast_keywords:
        if kw in SHORT_KWS:
            # Use word-boundary match for short keywords
            import re
            if re.search(r'\b' + re.escape(kw) + r'\b', _combined):
                tier = 'fast'
                break
        elif kw in _combined:
            tier = 'fast'
            break

    needs_human_prefixes = ['fork/product:', 'adopt:', 'extract:', 'plugin/skill:']
    for prefix in needs_human_prefixes:
        if title_lower.startswith(prefix):
            return 'NEEDS HUMAN', tier

    auto_promote_keywords = ['orphan process', 'disk alert', 'lock file', 'stale config',
                             'uncommitted changes', 'missing-but-trivial files',
                             'skills tracking', 'usage data']
    for keyword in auto_promote_keywords:
        if keyword in title_lower:
            return 'AUTO-PROMOTE', tier

    return 'NEEDS HUMAN', tier

def _read_task_pipeline_mode(board, task_id):
    """Read pipeline_mode (and tier) straight from the board DB.

    ``hermes kanban list --json`` does not surface pipeline_mode, so the
    classifier cannot see it via the CLI path. Feature-created tasks are
    identifiable ONLY by their pipeline_mode column — read it here.
    Returns the mode string ('full'/'express'/None).
    """
    db_path = _get_board_db(board)
    if not db_path or not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT pipeline_mode FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        conn.close()
        if row:
            return row[0]
    except Exception:
        pass
    return None


def main():
    global _DRY_RUN
    if '--dry-run' in sys.argv:
        _DRY_RUN = True
    all_triage = []
    for board in BOARDS:
        tasks = run_hermes_list(board, 'triage')
        if tasks is None or not isinstance(tasks, list):
            continue
        for task in tasks:
            tid = task.get('id')
            title = task.get('title', '')
            body = task.get('body', '')
            assignee = task.get('assignee', '')
            if tid:
                all_triage.append((tid, board, title, body, assignee))

    if not all_triage:
        # No triage tasks — clear pending file if empty
        state = load_json(STATE_FILE, {'notified_ids': [], 'notified_at': 0, 'notified_count': 0})
        if set(state.get('notified_ids', [])) != set():
            state['notified_ids'] = []
            state['notified_at'] = int(time.time())
            state['notified_count'] = 0
            save_json(STATE_FILE, state)
        # Clear pending file
        save_json(PENDING_FILE, {'tasks': [], 'timestamp': int(time.time())})
        return  # SILENT

    auto_promoted = []
    pending_tasks = []
    errors = []

    for task_id, board, title, body, assignee in all_triage:
        # FIX 2026-08-13: pipeline-owned tasks (feature create) must be
        # identified by pipeline_mode in the DB — the CLI list JSON does
        # not expose it. These ALWAYS auto-promote into the feature
        # pipeline; keyword classification must not touch them.

        # S3 GATE (Sahil 30/08/26): proposal-prefixed tasks and tasks whose
        # parent proposal is archived/backlog must NOT auto-promote — they
        # wait in backlog until Sahil explicitly approves them as work.
        task_pipeline_mode = _read_task_pipeline_mode(board, task_id)
        if not task_pipeline_mode and (
                title.lower().startswith('proposal:') or _parent_not_active(board, task_id)):
            if _route_to_backlog(board, task_id):
                continue  # routed to backlog; skip classification entirely
        classification, tier = classify_task(title, body, pipeline_mode=task_pipeline_mode)

        # WS-2 board routing: auto-assign unassigned tasks based on keywords
        # EXCEPT pipeline-owned tasks: their assignee follows the pipeline's
        # own stage-owner routing. Keyword-routing a feature task to octacon/
        # wesker/etc. would set the wrong initial assignee before the
        # dispatcher's stage-owner reassignment kicks in.
        routed_board, routed_assignee = None, None
        if not assignee or not assignee.strip():
            if not task_pipeline_mode:
                routed_board, routed_assignee = _route_task(title, body)

        if classification == 'AUTO-PROMOTE':
            if tier == 'full':
                # Feature pipeline: tier=full tasks enter at 'research'
                # stage. Intake gate must pass (body has problem + success
                # criteria) before promotion.
                from hermes_cli.feature_pipeline import validate_intake_brief
                gate_fail = validate_intake_brief(body)
                if gate_fail:
                    reason = f"Intake gate failed: {gate_fail}"
                    if block_task(task_id, board, reason):
                        _set_task_tier(board, task_id, tier)
                        pending_tasks.append({
                            'id': task_id, 'board': board, 'title': title,
                            'body': body, 'assignee': assignee,
                            'blocked_at': int(time.time()), 'investigated': False
                        })
                    else:
                        errors.append({'id': task_id, 'board': board, 'title': title, 'error': 'block failed'})
                    continue
                # Gate passed — promote to 'research' with pipeline_stage set
                if _promote_to_pipeline(task_id, board, 'research'):
                    _set_task_tier(board, task_id, tier)
                    if routed_assignee:
                        _set_task_assignee(board, task_id, routed_assignee)
                    auto_promoted.append({'id': task_id, 'board': board, 'title': title, 'tier': tier, 'stage': 'research'})
                else:
                    errors.append({'id': task_id, 'board': board, 'title': title, 'error': 'promote to research failed'})
            else:
                # Fast tier: promote to 'todo' via SQL (hermes kanban promote
                # only accepts 'todo'/'blocked', not 'triage')
                db_path = _get_board_db(board)
                promoted_ok = False
                if db_path and os.path.exists(db_path):
                    try:
                        conn = sqlite3.connect(db_path)
                        if write_lock is not None:
                            with write_lock(conn):
                                conn.execute(
                                    "UPDATE tasks SET status='todo', updated_at=strftime('%s','now') WHERE id=? AND status='triage'",
                                    (task_id,))
                                conn.commit()
                        else:
                            conn.execute(
                                "UPDATE tasks SET status='todo', updated_at=strftime('%s','now') WHERE id=? AND status='triage'",
                                (task_id,))
                            conn.commit()
                        conn.close()
                        promoted_ok = True
                    except Exception:
                        pass
                if promoted_ok:
                    _set_task_tier(board, task_id, tier)
                    if routed_assignee:
                        _set_task_assignee(board, task_id, routed_assignee)
                    auto_promoted.append({'id': task_id, 'board': board, 'title': title, 'tier': tier})
                else:
                    errors.append({'id': task_id, 'board': board, 'title': title, 'error': 'promote failed'})
        else:
            reason = f"Needs Sahil's decision: {title[:120]}"
            # block_task CLI only accepts 'running'/'ready' — triage
            # tasks must be transitioned directly via SQL, then
            # commented via CLI (which works on any status)
            db_path = _get_board_db(board)
            if db_path and os.path.exists(db_path):
                try:
                    conn = sqlite3.connect(db_path)
                    if write_lock is not None:
                        with write_lock(conn):
                            conn.execute(
                                "UPDATE tasks SET status='blocked', updated_at=strftime('%s','now') WHERE id=? AND status='triage'",
                                (task_id,))
                            conn.commit()
                    else:
                        conn.execute(
                            "UPDATE tasks SET status='blocked', updated_at=strftime('%s','now') WHERE id=? AND status='triage'",
                            (task_id,))
                        conn.commit()
                    conn.close()
                    blocked_ok = True
                except Exception:
                    blocked_ok = False
            else:
                blocked_ok = False
            # Add comment via CLI (works on any status)
            if not _DRY_RUN:
                comment_cmd = ['hermes', 'kanban', '--board', board, 'comment', task_id, reason]
                subprocess.run(comment_cmd, capture_output=True, text=True, timeout=30)
            if blocked_ok:
                _set_task_tier(board, task_id, tier)
                pending_tasks.append({
                    'id': task_id,
                    'board': board,
                    'title': title,
                    'body': body,
                    'assignee': assignee,
                    'blocked_at': int(time.time()),
                    'investigated': False
                })
            else:
                errors.append({'id': task_id, 'board': board, 'title': title, 'error': 'block failed'})
                print(f"[ERROR] block failed for {task_id} ({board}): {title[:80]}", file=sys.stderr)

    # Pipeline bypass check
    bypass_found = []
    for board in BOARDS:
        for status in ['todo', 'ready']:
            tasks = run_hermes_list(board, status)
            if tasks is None or not isinstance(tasks, list):
                continue
            for task in tasks:
                tid = task.get('id')
                title = task.get('title', '').lower()
                if any(title.startswith(p) for p in ['fork/product:', 'adopt:', 'extract:', 'plugin/skill:']):
                    comment = "Pipeline bypass — created at todo/ready without human approval. See codebase-consolidation skill PITFALLS."
                    if block_task(tid, board, comment):
                        bypass_found.append({'id': tid, 'board': board, 'title': task.get('title', '')})

    # Write pending investigation file for the investigator cron
    pending_data = {
        'tasks': pending_tasks,
        'bypass_found': bypass_found,
        'auto_promoted_count': len(auto_promoted),
        'error_count': len(errors),
        'timestamp': int(time.time())
    }
    save_json(PENDING_FILE, pending_data)

    # Update triage state for debounce tracking
    current_ids = [t['id'] for t in pending_tasks]
    state = load_json(STATE_FILE, {'notified_ids': [], 'notified_at': 0, 'notified_count': 0})
    state['notified_ids'] = current_ids
    state['notified_at'] = int(time.time())
    state['notified_count'] = len(current_ids)
    save_json(STATE_FILE, state)

    # SILENT — notification handled by kensei-triage-investigator cron
    # Only print if there's a critical error (bypass found with no pending tasks)
    if bypass_found and not pending_tasks:
        lines = ['⚠️ Pipeline bypass blocked (no triage tasks):']
        for b in bypass_found:
            lines.append(f"• `{b['id']}` ({b['board']}): {b['title']}")
        print('\n'.join(lines))

    # FIX 2026-08-13 (Spectator Mode incident t_9df6f54b): errors used to be
    # silently folded into pending-investigation.json's error_count — nobody
    # ever saw them. A triage block/promote failure (write-lock collision,
    # WAL race) meant a task silently stayed in triage where the gateway
    # auto-decomposer could hijack it. Errors must be VISIBLE: emit a
    # compact alert line so the cron's stdout delivery surfaces it.
    if errors:
        lines = [f'⚠️ triage processor: {len(errors)} action(s) failed:']
        for e in errors:
            lines.append(
                f"• `{e.get('id')}` ({e.get('board')}): {e.get('error')}"
            )
        print('\n'.join(lines), file=sys.stderr)

if __name__ == '__main__':
    main()
