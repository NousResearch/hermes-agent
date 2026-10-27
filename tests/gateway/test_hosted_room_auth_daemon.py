"""Hosted rooms over real OIDC-authenticated ordinary daemon sockets: one owner, many tabs."""
from tests.gateway.fixtures.hosted_auth_probe import probe


def test_every_signed_in_tab_shares_the_owners_hosted_rooms(tmp_path):
    receipt = probe(tmp_path)
    # The second account's token exchange happened, then its sign-in was refused.
    assert receipt['actors'] == ['alice', 'mallory', 'alice']
    assert receipt['rooms'] == ['tab1-room', 'tab2-room']
    assert receipt['shared_views'] == 4
    assert receipt['owner_stops'] == 2
