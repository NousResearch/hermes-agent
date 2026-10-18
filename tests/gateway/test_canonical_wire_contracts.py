"""Canonical RPC validation must not reuse the independently shipped serve wire."""
from types import SimpleNamespace

import pytest

from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_controls import AuthorityConnection
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch
from tui_gateway.contracts.registry import CANONICAL_METHODS, METHODS, ContractViolation, validate_params


@pytest.mark.asyncio
async def test_canonical_resume_rejects_unknown_keys_and_validates_real_snapshot(tmp_path):
    db = SessionDB(db_path=tmp_path / 'state.db')
    viewer = None
    try:
        db.create_session('s', source='test')
        epoch = begin_runtime_epoch(db, instance_id='test')
        authority = SessionAuthority(SimpleNamespace(), profile_id='test', instance_id='test', db=db, epoch=epoch)
        authority.sessions['s'] = LiveSession(None, 'route')
        viewer = AuthorityConnection(authority, object(), {'user_id': 'human'})
        from gateway.session_group_controls import GROUP_METHODS
        assert set(viewer.handlers()) | set(GROUP_METHODS) | {'profiles.list'} <= set(METHODS) | set(CANONICAL_METHODS)
        refused = await viewer.dispatch({'id': 1, 'method': 'session.resume', 'params': {
            'session_id': 's', 'execution_generaton': 100}})
        assert refused['error']['data']['reason'] == 'invalid_params'
        malformed = await viewer.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': ['s']}})
        assert malformed['error']['data']['reason'] == 'invalid_params'
        assert viewer.subscriptions == {}
        result = await viewer.dispatch({'id': 2, 'method': 'session.resume', 'params': {'session_id': 's'}})
        snapshot = CANONICAL_METHODS['session.resume'].result.model_validate(result['result'])
        assert snapshot.subscription_id == viewer.subscriptions['s']
        assert snapshot.revision == 0 and snapshot.pending == []
        assert snapshot.info.desktop_protocol == 'hermes-gateway-v1'
    finally:
        if viewer is not None:
            await viewer.close()
        db.close()


def test_canonical_admission_and_creation_models_leave_legacy_wire_unchanged():
    canonical = CANONICAL_METHODS['prompt.submit']
    params = {'session_id': 's', 'input_id': 'input', 'text': 'hello', 'finite': True}
    assert validate_params(canonical, params)[1] is None
    assert validate_params(METHODS['prompt.submit'], params)[1] is not None
    legacy_edit = {'session_id': 's', 'text': 'edit', 'truncate_before_row_id': 3, 'confirm_truncate': True}
    assert validate_params(METHODS['prompt.submit'], legacy_edit)[1] is None
    assert validate_params(canonical, legacy_edit)[1] is not None
    assert 'execution_generation' in CANONICAL_METHODS['approval.respond'].params.model_fields
    assert 'execution_generation' not in METHODS['approval.respond'].params.model_fields
    assert CANONICAL_METHODS['session.create'].params.model_validate({'ignore_rules': True}).ignore_rules


def test_canonical_session_create_accepts_the_kanban_dispatcher_source():
    # Regression (v0.22.2, #134889 post-merge): the canonical session.create gate closed
    # ``source`` to a 6-value Literal while the codebase treats it as an open vocabulary
    # (legacy METHODS wire: ``str | None``; hermes_state, session_lifecycle, and
    # session_context all carry open sets). The dispatcher's injected
    # HERMES_SESSION_SOURCE=kanban (hermes_cli/kanban_db_dispatch.py), the Desktop app's
    # ``source: 'desktop'`` (apps/desktop session-create-request.ts), and cron's ``cron``
    # all died at dispatch with a bare invalid_params — the CLI's worker create sends
    # session_source_for()'s value verbatim.
    from tui_gateway.contracts.registry import canonical_param_problems
    contract = CANONICAL_METHODS['session.create']
    legacy = METHODS['session.create']
    # The closed set must not be narrower than the legacy wire it re-declares.
    assert legacy.params.model_fields['source'].annotation is str or 'str' in str(
        legacy.params.model_fields['source'].annotation)
    for source in ('cli', 'tui', 'gui', 'acp', 'tool', 'oneshot', 'kanban', 'desktop', 'cron'):
        assert canonical_param_problems(contract, {'source': source}) == [], source
    # The value stays free-form (the row store owns what a source label means); the closed-
    # KEY part of the gate still refuses an unknown *key*, not an unknown source string.
    assert contract.params.model_validate({'source': 'kanban'}).model_dump()['source'] == 'kanban'
    assert canonical_param_problems(contract, {}) == []
    assert canonical_param_problems(contract, {'source_typo': 'cli'}) == ['source_typo']


@pytest.mark.asyncio
async def test_canonical_result_validation_catches_a_handler_contract_violation(tmp_path, monkeypatch):
    from tui_gateway.contracts import registry
    monkeypatch.setattr(registry, 'STRICT', True)
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        authority = SessionAuthority(SimpleNamespace(), profile_id='test', instance_id='test', db=db,
                                     epoch=begin_runtime_epoch(db, instance_id='test'))
        viewer = AuthorityConnection(authority, object(), {'user_id': 'human'})

        async def broken_ping(ref, params):
            return {'not_pong': True}

        monkeypatch.setattr(viewer, 'ping', broken_ping)
        with pytest.raises(ContractViolation, match='ping'):
            await viewer.dispatch({'id': 1, 'method': 'ping', 'params': {}})
    finally:
        db.close()
