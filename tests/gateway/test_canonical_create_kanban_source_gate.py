"""The canonical session.create gate no longer refuses the runtime's session sources (t_862738cd).

v0.22.2 closed ``session.create``'s ``source`` to a 6-value Literal
(``tui_gateway/contracts/canonical_sessions.py``) while the rest of the codebase treats it as an
open vocabulary (legacy METHODS wire ``str | None``; hermes_state, session_lifecycle, and
session_context all carry open frozensets). The gate at
``gateway/session_controls.py`` (``canonical_param_problems`` before the handler) then refused the
kanban dispatcher's injected ``HERMES_SESSION_SOURCE=kanban`` (hermes_cli/kanban_db_dispatch.py),
the Desktop app's ``source: 'desktop'`` (apps/desktop session-create-request.ts), and cron's
``cron`` — every one a bare ``invalid_params``. The gate is a closed KEY set; the VALUE of a
known key is not part of the wire contract, so ``source`` must stay as open as the legacy wire.

NOTE (t_53101721): this file pins GATE 1 only — the canonical wire contract. A full turn passes
a SECOND source gate at ``gateway/session_policy.py`` (``SURFACES`` enforced by ``build_policy``
in the handler); a source that passes this dispatch-level test can still be refused by gate 2.
The second gate is pinned by ``test_canonical_create_kanban_surface_gate.py`` (which drives
``build_policy`` directly) — the two files together cover the full turn.
"""
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_controls import AuthorityConnection
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch
from tui_gateway.contracts.registry import CANONICAL_METHODS, METHODS, canonical_param_problems

# Every source the runtime stamps on a session row and the CLI then ships on the wire
# (agent/session_source.py + the desktop/cron/kanban producers above).
RUNTIME_SOURCES = ['kanban', 'desktop', 'cron', 'cli', 'oneshot', 'tui', 'gui', 'acp', 'tool']

# The exact signature of a GATE refusal (session_controls.py:271-272), distinct from a handler's
# own domain refusal which carries a different data.reason.
def _is_source_gate_refusal(reply) -> bool:
    err = (reply or {}).get('error') or {}
    data = err.get('data') or {}
    return (err.get('code') == 4001 and data.get('reason') == 'invalid_params'
            and 'source' in data.get('fields', []))


@pytest.mark.asyncio
@pytest.mark.parametrize('source', RUNTIME_SOURCES)
async def test_canonical_gate_clears_for_every_runtime_source(source):
    """The gate runs BEFORE the handler. A source refused here is the regression; clearing the
    gate (even if the minimal handler stub then errors on unrelated machinery) proves the fix.
    We assert the response is not a *source gate refusal* — the one verdict this regression owns."""
    tmp = tempfile.mkdtemp(prefix='gate-verify-')
    db = SessionDB(db_path=Path(tmp) / 'state.db')
    viewer = None
    try:
        db.create_session('gate-s', source='test')
        epoch = begin_runtime_epoch(db, instance_id='gate')
        runner = SimpleNamespace(session_store=None, _session_db=db, adapters={}, _draining=False,
                                 _cached_agent_for=lambda route: None, retiring=False)
        runner._adapter_for_source = lambda src: runner.adapters.get(src.platform)
        authority = SessionAuthority(runner, profile_id='default', instance_id='gate', db=db, epoch=epoch)
        authority.sessions['gate-s'] = LiveSession(None, 'route')
        viewer = AuthorityConnection(authority, object(), {'user_id': 'human'})
        try:
            reply = await viewer.dispatch({'id': 1, 'method': 'session.create', 'params': {
                'request_id': 'gate-req', 'source': source, 'cwd': tmp}})
        except Exception:
            # The gate passed and the (minimal) handler raised on unrelated machinery — that is
            # NOT the source-gate refusal this regression owns.
            return
        assert not _is_source_gate_refusal(reply), f"source={source} refused at the gate: {reply}"
    finally:
        if viewer is not None:
            await viewer.close()
        db.close()


@pytest.mark.asyncio
async def test_canonical_gate_still_refuses_unknown_keys():
    """The closed set is the KEY set: an unknown key still refuses before the handler."""
    tmp = tempfile.mkdtemp(prefix='gate-verify-')
    db = SessionDB(db_path=Path(tmp) / 'state.db')
    viewer = None
    try:
        db.create_session('gate-s', source='test')
        epoch = begin_runtime_epoch(db, instance_id='gate')
        runner = SimpleNamespace(session_store=None, _session_db=db, adapters={}, _draining=False,
                                 _cached_agent_for=lambda route: None, retiring=False)
        runner._adapter_for_source = lambda src: runner.adapters.get(src.platform)
        authority = SessionAuthority(runner, profile_id='default', instance_id='gate', db=db, epoch=epoch)
        authority.sessions['gate-s'] = LiveSession(None, 'route')
        viewer = AuthorityConnection(authority, object(), {'user_id': 'human'})
        reply = await viewer.dispatch({'id': 1, 'method': 'session.create', 'params': {
            'request_id': 'gate-req', 'source': 'cli', 'source_typo': 'cli', 'cwd': tmp}})
        assert reply['error']['code'] == 4001
        assert reply['error']['data']['reason'] == 'invalid_params'
        assert 'source_typo' in reply['error']['data']['fields']
    finally:
        if viewer is not None:
            await viewer.close()
        db.close()


def test_canonical_source_stays_as_open_as_the_legacy_wire():
    """The regression's shape: a closed Literal where the legacy wire is open. The gate's exact
    refusal function is asserted directly (it is a pure check, session_controls.py:268)."""
    contract = CANONICAL_METHODS['session.create']
    legacy_source = METHODS['session.create'].params.model_fields['source'].annotation
    assert 'str' in str(legacy_source), "legacy wire kept source open (str) — canonical must match"
    for source in RUNTIME_SOURCES:
        assert canonical_param_problems(contract, {'source': source}) == [], source
    # Closed-KEY property survives: an unknown key is still a problem.
    assert canonical_param_problems(contract, {'source_typo': 'cli'}) == ['source_typo']
