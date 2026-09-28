"""Hosted-room isolation over real OIDC-authenticated ordinary daemon sockets."""
from tests.gateway.fixtures.hosted_auth_probe import probe


def test_two_authenticated_actors_own_separate_hosted_rooms(tmp_path):
    receipt = probe(tmp_path)
    assert receipt['actors'] == ['alice', 'bob']
    assert receipt['rooms'] == ['alice-room', 'bob-room']
    assert receipt['cross_actor_denials'] == 10
    assert receipt['owner_stops'] == 2
