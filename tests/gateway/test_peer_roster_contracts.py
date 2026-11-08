"""Peer selectors are closed wire values, never caller-owned transport coordinates."""
import pytest
from pydantic import ValidationError

from tui_gateway.contracts.canonical_peers import (
    GroupsCreateMixedParams, PeerBot, PeerCatalog, PeersDMParams, PeersDMResult,
)


@pytest.mark.parametrize('field', ['url', 'api_key', 'home_install_id', 'authority_gateway_id'])
def test_peer_dm_rejects_client_transport_and_authority_fields(field):
    with pytest.raises(ValidationError):
        PeersDMParams.model_validate(dict(route_id='opaque', message='hello', request_id='one',
                                         **{field: 'caller-controlled'}))


def test_mixed_member_rejects_an_advertised_endpoint():
    members = [dict(kind='local', profile='default', member_id='a', handle='a'),
               dict(kind='peer', route_id='opaque', member_id='b', handle='b',
                    endpoint='https://caller.example')]
    with pytest.raises(ValidationError):
        GroupsCreateMixedParams.model_validate(dict(room_id='r', name='Room', members=members))


@pytest.mark.parametrize('model', [PeerBot, PeerCatalog, PeersDMResult])
def test_peer_result_shapes_reject_undeclared_transport_fields(model):
    assert model.model_json_schema()['additionalProperties'] is False
