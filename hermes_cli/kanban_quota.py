"""Orcamento finito de cota, separado de falhas do trabalho.

Uma tentativa inicial e uma repeticao automatica. Depois, desbloqueio explicito
permite uma nova sonda; relogio e cooldown sozinhos nao reabrem o circuito.
"""
from __future__ import annotations

import sqlite3

MAX_CONSECUTIVE_QUOTA_ATTEMPTS = 2
QUOTA_EXHAUSTED_REASON = (
    "Automatic quota retry budget exhausted after two consecutive rate-limited runs. "
    "Check provider quota or select an eligible route before explicitly unblocking. "
    "Workspaces, attachments and run history are preserved."
)


def quota_attempts_exhausted(conn: sqlite3.Connection, task_id: str) -> bool:
    # unblock_task limpa last_failure_error: uma sonda manual fica autorizada.
    row = conn.execute(
        "SELECT last_failure_error FROM tasks WHERE id=?", (task_id,),
    ).fetchone()
    if row is None or not row['last_failure_error']:
        return False
    runs = conn.execute(
        "SELECT outcome FROM task_runs WHERE task_id=? AND ended_at IS NOT NULL "
        "ORDER BY id DESC LIMIT ?", (task_id, MAX_CONSECUTIVE_QUOTA_ATTEMPTS),
    ).fetchall()
    return len(runs) == MAX_CONSECUTIVE_QUOTA_ATTEMPTS and all(
        r['outcome'] == 'rate_limited' for r in runs
    )


def park_exhausted_quota(conn: sqlite3.Connection, task_id: str, lane: str) -> bool:
    # Importacao tardia evita ciclo com o dispatcher. Nenhum run sintetico:
    # o estacionamento nao e uma tentativa nem uma falha de implementacao.
    from hermes_cli import kanban_db as kb

    if lane not in ('ready', 'review'):
        return False
    with kb.write_txn(conn):
        if not quota_attempts_exhausted(conn, task_id):
            return False
        changed = conn.execute(
            "UPDATE tasks SET status='blocked', block_kind='capability', "
            "last_failure_error=? WHERE id=? AND status=? "
            "AND current_run_id IS NULL AND worker_pid IS NULL AND claim_lock IS NULL",
            (QUOTA_EXHAUSTED_REASON, task_id, lane),
        ).rowcount
        if changed != 1:
            return False
        kb._append_event(conn, task_id, 'blocked', {
            'reason': QUOTA_EXHAUSTED_REASON, 'kind': 'capability',
            'source_status': lane, 'resume_status': lane,
            'quota_attempt_limit': MAX_CONSECUTIVE_QUOTA_ATTEMPTS,
        })
        task = kb.get_task(conn, task_id)
    kb._fire_task_hook('kanban_task_blocked', task, task_id, None, reason=QUOTA_EXHAUSTED_REASON)
    return True
