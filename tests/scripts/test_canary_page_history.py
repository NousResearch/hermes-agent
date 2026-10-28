"""Already-published canary pages remain orderable after the claim-tag change."""
import pytest

from tests.scripts.test_release_r2 import r2_server  # noqa: F401
from tests.scripts.test_render_builds_publication import rbt


@pytest.mark.parametrize("stamp,advances", [
    ("20261019T064100Z", False),
    ("20261020T064100Z", True),
    ("20261021T064100Z", True),
])
def test_new_canary_orders_against_published_legacy_page(request, stamp, advances):
    server = request.getfixturevalue("r2_server")
    base = f"http://127.0.0.1:{server.server_port}/hermes-releases"
    key = "releases/canary/index.html"
    previous = rbt._page("Canary downloads", "v1.2.3+canary.20261020T064100Z", []).encode()
    server.store[key] = (previous, '"old"')
    tag = f"canary.{stamp}-v1.2.3"
    result = rbt.write_channel_page(tag, {}, base)
    assert (result is not None) is advances
    if advances:
        assert rbt.recorded_build(server.store[key][0].decode()) == tag
    else:
        assert server.store[key][0] == previous


def test_stable_cannot_replace_published_legacy_canary_page():
    previous = rbt._page("Canary downloads", "v1.2.3+canary.20261020T064100Z", [])
    assert not rbt.supersedes(previous, "v9.9.9")
