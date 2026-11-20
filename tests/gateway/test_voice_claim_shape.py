"""Voice submission shape validation does not confer transcript ownership."""

import pytest

from gateway import session_surface
from persistence.core.errors import AuthorityRefusal


@pytest.fixture
def claim():
    return {'call_id': 'call:13', 'delegation_id': 'request:7', 'event_ids': ['evt-2', 'evt-1']}


def test_voice_coverage_preserves_provider_ids_and_order(claim):
    assert session_surface.admit_voice_claim(
        {'surface': 'voice-live', 'voice_claim': claim}, request_id='voice-live:["call:13","request:7"]',
    ) == {'voice_claim_v1': claim}
    assert session_surface.admit_voice_claim({'surface': 'voice-live'}, request_id='legacy-request') == {}


@pytest.mark.parametrize('bad', [
    {'surface': 'hud'},
    {'voice_claim': {'call_id': 'call:13', 'delegation_id': 'request:7', 'event_ids': ['evt', 'evt']}},
    {'voice_claim': {'call_id': 'call:13', 'delegation_id': 'request:7', 'event_ids': []}},
    {'voice_claim': {'call_id': 'call:13', 'delegation_id': 'request:7', 'event_ids': ['']}},
    {'voice_claim': {'call_id': 'call:13', 'delegation_id': 'request:7', 'event_ids': ['evt'], 'row_id': 23}},
])
def test_voice_coverage_rejects_other_surfaces_invalid_ids_and_row_coordinates(claim, bad):
    with pytest.raises(AuthorityRefusal, match='invalid_params'):
        session_surface.admit_voice_claim(
            {'surface': 'voice-live', 'voice_claim': claim, **bad}, request_id='voice-live:["call:13","request:7"]',
        )


def test_same_delegation_cannot_start_with_a_fresh_request_id(claim):
    with pytest.raises(AuthorityRefusal, match='invalid_params'):
        session_surface.admit_voice_claim({'surface': 'voice-live', 'voice_claim': claim}, request_id='fresh-key')
