"""A refused Kanban claim must report WHY it was refused.

``build_kanban_policy`` raises ``RuntimeStoreError`` for each refusal it diagnoses
(``stale_kanban_claim``, ``profile_mismatch``). ``RuntimeStoreError`` subclasses ``ValueError``,
and the surrounding ``except (ValueError, OSError, sqlite3.Error)`` used to rewrite every one of
them into a generic ``invalid_kanban_claim`` — hiding a live dispatch failure behind a code that
names neither the claim nor the profile. These tests pin the honest codes and the carried cause.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

# The claim-admission module pulls the dispatcher, which reads install metadata from the real
# home at import time; that read is the module's, not this suite's test data.
pytestmark = pytest.mark.allow_real_home_io


def _claimed_card(tmp_path, assignee='default'):
    """A board + running claim, and the owner connection that may admit it."""
    from contextlib import closing
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from gateway.session_contract import Principal

    workspace = tmp_path / 'workspace'
    workspace.mkdir(exist_ok=True)
    kb.create_board('fleet')
    with closing(connect(board='fleet')) as conn:
        task_id = kb.create_task(conn, title='Refusal codes', body='Goal criteria',
                                 assignee=assignee, workspace_kind='dir',
                                 workspace_path=str(workspace))
        kb.recompute_ready(conn)
        task = kb.claim_task(conn, task_id)
    actor = Principal('owner', str(tmp_path), frozenset({'session:create'}), 'transport')
    connection = SimpleNamespace(authority=SimpleNamespace(profile_id=str(tmp_path)),
                                 actor=actor, native_owner=True)
    params = dict(board='fleet', task_id=task.id, run_id=task.current_run_id,
                  claim_lock=task.claim_lock)
    config = {'model': {'default': 'loop-model', 'provider': 'custom'},
              'platform_toolsets': {'cli': ['terminal', 'file']}}
    return connection, params, config, task


@pytest.fixture
def fleet_home(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.delenv('HERMES_DELEGATED_CHILD', raising=False)
    monkeypatch.setattr('hermes_cli.profiles.resolve_profile_env',
                        lambda name: str(tmp_path) if name == 'default' else str(tmp_path / name))
    return tmp_path


def test_a_stale_claim_reports_stale_not_a_generic_refusal(fleet_home):
    from gateway.session_kanban import build_kanban_policy
    from hermes_state_runtime import RuntimeStoreError

    connection, params, config, _task = _claimed_card(fleet_home)
    with pytest.raises(RuntimeStoreError, match='stale_kanban_claim'):
        build_kanban_policy(connection, params | {'claim_lock': 'forged'}, config)


def test_assignee_of_another_profile_reports_profile_mismatch(fleet_home):
    """A claim whose assignee is not this owner's profile is refused by name, not generically."""
    from gateway.session_kanban import build_kanban_policy
    from hermes_state_runtime import RuntimeStoreError

    connection, params, config, _task = _claimed_card(fleet_home, assignee='other-profile')
    with pytest.raises(RuntimeStoreError, match='profile_mismatch'):
        build_kanban_policy(connection, params, config)


def test_an_unreadable_board_carries_its_cause_in_the_message(fleet_home):
    """The generic arm still reports invalid_kanban_claim, but now says what went wrong."""
    from gateway.session_kanban import build_kanban_policy
    from hermes_state_runtime import RuntimeStoreError

    connection, params, config, _task = _claimed_card(fleet_home)
    missing = Path(fleet_home) / 'no-such-board.db'
    with pytest.raises(RuntimeStoreError) as caught:
        build_kanban_policy(connection, params | {'db': str(missing)}, config)
    assert str(caught.value).startswith('invalid_kanban_claim')
    assert 'no-such-board.db' in str(caught.value)
    assert isinstance(caught.value.__cause__, OSError)
