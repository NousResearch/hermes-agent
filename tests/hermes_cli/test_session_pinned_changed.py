import asyncio
import importlib
from unittest.mock import AsyncMock, patch

import pytest

from hermes_cli.web_server_sessions import _mutate_session_request
from tui_gateway.contracts.registry import EVENTS, check_payload
# Ensure the events module is imported so @event registrations run.
import tui_gateway.contracts.events


def test_pinned_event_contract_is_registered_and_closed():
    payload = {'session_id': 's1', 'profile': 'default', 'pinned': True, 'revision': 8}
    assert 'session.pinned_changed' in EVENTS
    check_payload('session.pinned_changed', payload)
    with pytest.raises(Exception):
        EVENTS['session.pinned_changed'].payload.model_validate({**payload, 'unexpected': True})


@pytest.mark.parametrize('payload', [
    {'pinned': True},
    {'pinned': False},
])
def test_successful_pin_mutation_broadcasts_authoritative_receipt(payload):
    events = []
    with patch('hermes_cli.web_server_sessions._session_mutation_context', return_value=(None, object())), \
         patch('hermes_cli.web_server_sessions._mutate_via_gateway_owner', new=AsyncMock(return_value={
             'session_id': 's1', 'revision': 12, 'operation': 'sidebar'})), \
         patch('tui_gateway.server._broadcast_global_event', side_effect=lambda kind, body: events.append((kind, body))):
        result = asyncio.run(_mutate_session_request(
            object(), 'default', 's1', request_id='r1', expected_revision=11,
            operation='sidebar', payload=payload))

    assert result['revision'] == 12
    assert events == [('session.pinned_changed', {
        'session_id': 's1', 'profile': 'default', 'pinned': payload['pinned'], 'revision': 12})]


def test_archive_driven_unpin_does_not_double_broadcast():
    with patch('hermes_cli.web_server_sessions._session_mutation_context', return_value=(None, object())), \
         patch('hermes_cli.web_server_sessions._mutate_via_gateway_owner', new=AsyncMock(return_value={
             'session_id': 's1', 'revision': 12, 'operation': 'sidebar'})), \
         patch('tui_gateway.server._broadcast_global_event') as broadcast:
        asyncio.run(_mutate_session_request(
            object(), 'default', 's1', request_id='r1', expected_revision=11,
            operation='sidebar', payload={'archived': True, 'pinned': False}))
    broadcast.assert_not_called()


def test_failed_mutation_does_not_broadcast():
    from persistence.core.errors import AuthorityRefusal
    with patch('hermes_cli.web_server_sessions._session_mutation_context', return_value=(None, object())), \
         patch('hermes_cli.web_server_sessions._mutate_via_gateway_owner',
               new=AsyncMock(side_effect=AuthorityRefusal('revision_conflict'))), \
         patch('tui_gateway.server._broadcast_global_event') as broadcast:
        with pytest.raises(Exception) as error:
            asyncio.run(_mutate_session_request(
                object(), 'default', 's1', request_id='r1', expected_revision=11,
                operation='sidebar', payload={'pinned': True}))
        assert error.value.status_code == 409
    broadcast.assert_not_called()
