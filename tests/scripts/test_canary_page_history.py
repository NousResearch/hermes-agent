"""A published canary page only advances to a newer canary claim."""
import pytest

from tests.scripts.test_release_r2 import r2_server
from tests.scripts.test_render_builds_publication import rbt

PUBLISHED = "canary.20261020T064100Z-v1.2.3"


@pytest.mark.parametrize("stamp,advances", [
    ("20261019T064100Z", False),
    ("20261020T064100Z", True),
    ("20261021T064100Z", True),
])
def test_new_canary_orders_against_published_page(request, stamp, advances):
    server = request.getfixturevalue("r2_server")
    base = f"http://127.0.0.1:{server.server_port}/hermes-releases"
    key = "releases/canary/index.html"
    previous = rbt._page("Canary downloads", PUBLISHED, []).encode()
    server.store[key] = (previous, '"old"')
    tag = f"canary.{stamp}-v1.2.3"
    result = rbt.write_channel_page(tag, {}, base)
    assert (result is not None) is advances
    if advances:
        assert rbt.recorded_build(server.store[key][0].decode()) == tag
    else:
        assert server.store[key][0] == previous


def test_stable_cannot_replace_published_canary_page():
    previous = rbt._page("Canary downloads", PUBLISHED, [])
    assert not rbt.supersedes(previous, "v9.9.9")
