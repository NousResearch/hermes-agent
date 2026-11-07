"""Worker constructor facts survive an occupied model_config on reserved rows.

Regression for #149379: ``worker_create`` merged constructor facts
(``max_iterations`` / ``max_tokens`` / ``reasoning_config`` / ``yolo_mode``) only
beside a ``_delegate_from`` marker, so a kanban/cron/local/oneshot row whose
``model_config`` already carried a sidecar blob (``_usage_anchor``) silently lost
every fact the worker's ``session.create`` receipt carried.
"""
import json

import pytest

from hermes_state import SessionDB
from hermes_state_runtime import (
    begin_runtime_epoch, mutate_worker_execution, register_worker_execution,
)

CONSTRUCTOR_KEYS = {'max_iterations': 500, 'max_tokens': None,
                    'reasoning_config': {'enabled': True, 'effort': 'medium'}}
USAGE_ANCHOR = {'prompt_tokens': 1000, 'completion_tokens': 50, 'base_count': 3,
                'base_last_role': 'user', 'base_last_fp': 'fp', 'base_prefix_fp': 'pfp'}


@pytest.fixture
def reserved_board(tmp_path):
    db = SessionDB(tmp_path / 'state.db')
    epoch = begin_runtime_epoch(db, instance_id='fixture')

    def reserve(sid, source, model_config=None):
        db.create_session(sid, source, model_config=model_config)
        scope = dict(epoch=epoch, execution_id='exec-' + sid, session_id=sid, generation=0)
        register_worker_execution(db, **scope, kind='compute', adoption_secret='s')
        return scope

    yield db, reserve
    db.close()


def test_constructor_facts_merge_beside_usage_anchor(reserved_board):
    """Census shape: the anchor receipt lands before ``session.create``; the facts must
    merge beside it, never drop the anchor."""
    db, reserve = reserved_board
    scope = reserve('kanban-row', 'kanban')
    mutate_worker_execution(db, **scope, sequence=1, operation='session.sidecars',
                            payload={'patch': {'_usage_anchor': USAGE_ANCHOR}})
    mutate_worker_execution(db, **scope, sequence=2, operation='session.create',
                            payload={'source': 'kanban', 'model_config': dict(CONSTRUCTOR_KEYS)})
    config = json.loads(db.get_session('kanban-row')['model_config'])
    assert config['reasoning_config'] == {'enabled': True, 'effort': 'medium'}
    assert config['max_iterations'] == 500
    assert config['_usage_anchor'] == USAGE_ANCHOR  # guard: existing keys survive the merge


def test_delegate_child_merge_keeps_lineage_marker(reserved_board):
    """The delegate-child contract predates the fix and must not regress."""
    db, reserve = reserved_board
    scope = reserve('child-row', 'subagent', model_config={'_delegate_from': 'parent'})
    mutate_worker_execution(db, **scope, sequence=1, operation='session.create',
                            payload={'source': 'subagent', 'model_config': dict(CONSTRUCTOR_KEYS)})
    config = json.loads(db.get_session('child-row')['model_config'])
    assert config['_delegate_from'] == 'parent'
    assert config['reasoning_config'] == {'enabled': True, 'effort': 'medium'}
