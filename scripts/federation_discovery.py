"""Bounded departmental discovery using existing Hermes profiles and Kanban claims."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import tempfile
import sqlite3
import sys
from collections import defaultdict


def plan_discovery(spec, tasks, day, links=()):
    children = defaultdict(set)
    for parent, child in links:
        children[parent].add(child)
    owned = defaultdict(set)
    for task in tasks:
        creator = str(task.get('created_by', ''))
        if not creator.startswith('federation-discovery-') or not task.get('id'):
            continue
        pending = [task['id']]
        while pending:
            task_id = pending.pop()
            if task_id in owned[creator]:
                continue
            owned[creator].add(task_id)
            pending.extend(children[task_id])
    federation_ids = set().union(*owned.values()) if owned else set()
    active = [task for task in tasks if task.get('status') not in {'done', 'archived'}]
    federation_active = [task for task in active if task.get('id') in federation_ids or str(task.get('created_by', '')).startswith('federation-discovery-')]
    capacity = max(0, spec['max_active'] - len(federation_active))
    planned = []
    for department in spec['departments']:
        creator = 'federation-discovery-' + department['id']
        key = creator + '-' + day
        title = '[Federation] ' + department['title'] + ' ' + day
        already_created = any(task.get('created_by') == creator and task.get('title') == title for task in tasks)
        active_children = [task['id'] for task in active if task.get('id') in owned[creator] and task.get('created_by') != creator]
        if len(active_children) >= 2 or already_created or any(task.get('created_by') == creator or task.get('assignee') == department['assignee'] for task in active):
            continue
        if len(planned) >= min(capacity, spec['max_dispatches']):
            break
        planned.append(dict(department, creator=creator, key=key, task_title=title, active_children=active_children))
    return planned


def run(hermes, *args):
    result = subprocess.run(
        [hermes, *args], check=True, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=60,
    )
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hermes', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    spec = json.loads((root / 'configs/federation/discovery.json').read_text(encoding="utf-8"))
    tasks = run(args.hermes, 'kanban', '--board', spec['board'], 'list', '--json')
    day = datetime.now(timezone.utc).date().isoformat()
    sys.path.insert(0, str(root))
    from hermes_cli.kanban_db import kanban_db_path
    connection = sqlite3.connect(kanban_db_path(spec['board']).as_uri() + '?mode=ro', uri=True)
    try:
        links = connection.execute('SELECT parent_id,child_id FROM task_links').fetchall()
    finally:
        connection.close()
    plan = plan_discovery(spec, tasks, day, links)
    if not args.apply:
        print(json.dumps({'status': 'planned', 'departments': plan}))
        return
    receipts = []
    for department in plan:
        body = (f'Authoritative discovery source: {root}. Read {root}/AGENTS.md and {root}/configs/federation/roles.json. '
                'LunaBot source: /Users/mikedemott/LunaBot; its library spec is '
                '/Users/mikedemott/LunaBot/docs/superpowers/specs/2026-08-26-governed-library-librarian-design.md. '
                'Active catalogue implementation: /Users/mikedemott/.codex/lunabot-support/worktrees/codex-library-vault-catalog-20260905. '
                f'Lunar City sources are at {root}/apps/desktop/src/app/lunar-city and '
                f'{root}/apps/desktop/public/lunar-city. This is a read-only discovery workspace; '
                'put findings in the task result/comment and assign implementation to an isolated child.\n\n'
                + spec['instructions'] + '\nExisting active children: ' + ', '.join(department['active_children'])
                + '\n\nDepartment assignment: ' + department['brief'])
        workspace = tempfile.mkdtemp(prefix='hermes-discovery-' + department['id'] + '-')
        result = run(args.hermes, 'kanban', '--board', spec['board'], 'create',
                     department['task_title'],
                     '--body', body, '--assignee', department['assignee'],
                     '--project', department['project'], '--workspace', 'dir:' + workspace,
                     '--idempotency-key', department['key'], '--max-runtime', '15m',
                     '--max-retries', '2', '--created-by', department['creator'], '--json')
        receipts.append({'department': department['id'], 'task': result})
    if receipts:
        print(json.dumps({'status': 'dispatched', 'receipts': receipts}))


if __name__ == '__main__':
    main()
