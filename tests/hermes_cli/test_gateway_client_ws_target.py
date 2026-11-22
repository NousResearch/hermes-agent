"""Every owner dial (viewer, worker RPC, A2A forward, bot delivery, Ink bootstrap) builds the same target."""
from types import SimpleNamespace

from hermes_cli.gateway_client import GATEWAY_WS_PROTOCOL, gateway_ws_target


def test_gateway_ws_target_maps_scheme_and_carries_one_ticket_protocol():
    assert gateway_ws_target(SimpleNamespace(api_origin='http://127.0.0.1:9'), 't1') == (
        'ws://127.0.0.1:9/api/ws', [GATEWAY_WS_PROTOCOL, 'hermes-gateway-ticket.t1'])
    url, protocols = gateway_ws_target(SimpleNamespace(api_origin='https://owner.local'), 't2')
    assert url == 'wss://owner.local/api/ws' and protocols[0] == 'hermes-gateway-v1'
