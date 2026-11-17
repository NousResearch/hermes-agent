"""Existing Todo and compression consumers of persistent work facts."""
from __future__ import annotations

import json
import sqlite3

from hermes_cli.task_delivery.store import WorkLedger


def ledger_for_agent(agent):
    from hermes_state import AsyncSessionDB, SessionDB
    db = getattr(agent, '_session_db', None)
    if isinstance(db, AsyncSessionDB):
        db = db._db
    if not isinstance(db, SessionDB) or not getattr(agent, 'session_id', None):
        return None
    return WorkLedger(db)


def recovery_context(agent):
    ledger = ledger_for_agent(agent)
    return ledger.recovery_context(agent.session_id) if ledger else ''


def execute_todo(agent, args, ctx):
    from tools.todo_tool import TodoStore, todo_tool
    ledger = ledger_for_agent(agent)
    previous = agent._todo_store.snapshot()
    try:
        if ledger and isinstance(args.get('todos'), list):
            proposed = TodoStore()
            proposed.restore(**agent._todo_store.snapshot())
            proposed.write(args['todos'], args.get('merge', False))
            ledger.check_todos(agent.session_id, proposed.read())
        result = todo_tool(todos=args.get('todos'), merge=args.get('merge', False), store=agent._todo_store)
        payload = json.loads(result)
        if ledger and 'todos' in payload:
            if args.get('todos') is not None:
                ledger.sync_todos(agent.session_id, payload['todos'])
            delivery = ledger.projection(agent.session_id)
            if delivery:
                payload['delivery'] = delivery
                return json.dumps(payload, ensure_ascii=False)
        return result
    except (ValueError, OSError, sqlite3.Error) as exc:
        agent._todo_store.restore(**previous)
        return json.dumps({'error': str(exc), 'todos': agent._todo_store.read()}, ensure_ascii=False)


def new_todo_store(agent):
    from tools.todo_tool import TodoStore
    return TodoStore(delivery_context=lambda: recovery_context(agent))
