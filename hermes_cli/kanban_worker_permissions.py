"""Owned-run ceilings: approval survives neither missing board proof nor scope drift."""
import json
import os
import sqlite3
from pathlib import Path


def owned_worker_skips_memory(config):
    """Suppress persistent input before construction, not project/SOUL rules."""
    from agent.delegation_context import owned_kanban_task
    from hermes_cli.kanban_db_policy import PolicyViolation
    required = config.get('agent', {}).get('require_execution_receipt', False)
    if type(required) is not bool:
        raise PolicyViolation('Execution policy: malformed profile receipt requirement')
    return required and bool(owned_kanban_task())


def pin_owned_worker_tools(agent, config):
    from agent.tool_permissions import ToolPermissionPolicy, agent_tool_policy, apply_agent_tool_policy
    from hermes_cli.kanban_db_policy import _load_policy, policy_provider, PolicyViolation
    required = config.get('agent', {}).get('require_execution_receipt', False)
    if type(required) is not bool:
        raise PolicyViolation('Execution policy: malformed profile receipt requirement')
    task_id = os.environ.get('HERMES_KANBAN_TASK')
    database = os.environ.get('HERMES_KANBAN_DB')
    if not task_id or not database:
        if required:
            raise PolicyViolation('Execution policy: this profile requires an approved owned Kanban run')
        return
    path = Path(database).resolve()
    if not path.is_file() and not required:
        return
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as conn:
        conn.row_factory = sqlite3.Row
        if _load_policy(conn, required=required) is None:
            return
        provider = policy_provider(conn)
        if provider is None:
            raise PolicyViolation('Execution policy: worker provider unavailable')
        run_id = int(os.environ.get('HERMES_KANBAN_RUN_ID', '0'))
        claim = os.environ.get('HERMES_KANBAN_CLAIM_LOCK')
        profile = os.environ.get('HERMES_PROFILE')
        row = conn.execute('''SELECT p.snapshot FROM native_policy_receipts p
            JOIN tasks t ON t.id=p.task_id JOIN task_runs r ON r.id=p.consumed_run_id
            WHERE p.task_id=? AND p.consumed_run_id=? AND p.profile=?
            AND t.current_run_id=r.id AND t.status='running' AND t.assignee=p.profile
            AND r.profile=p.profile AND r.ended_at IS NULL AND r.status='running'
            AND t.claim_lock=? AND r.claim_lock=?
            AND p.expires_at>CAST(strftime('%s','now') AS INTEGER)''',
            (task_id, run_id, profile, claim, claim)).fetchone()
        if row is None:
            raise PolicyViolation('Execution policy: missing owned-run permission snapshot')
        saved = json.loads(row[0])
        effective = provider.runtime_snapshot(conn, task_id)
        verifier = getattr(provider, 'verify_worker_snapshot', None)
        if not callable(verifier) or verifier(conn, task_id, saved, effective) is not True:
            raise PolicyViolation('Execution policy: worker scope or effective runtime changed')
        if agent.model != effective['model'] or agent.provider != effective['provider']:
            raise PolicyViolation('Execution policy: worker model/provider differs from approved runtime')
    names = frozenset(effective['tools'])
    existing = agent_tool_policy(agent).allowed_names
    agent._tool_policy = ToolPermissionPolicy(names if existing is None else names & existing)
    apply_agent_tool_policy(agent)
    if agent.valid_tool_names != names:
        raise PolicyViolation('Execution policy: actual worker tools differ from approved snapshot')
    agent._kanban_permission_snapshot = saved
