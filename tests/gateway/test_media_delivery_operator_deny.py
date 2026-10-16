"""Operator deny roots override every automatic local-media allow path."""

from gateway import media_policy
from gateway.platforms import base


def test_operator_deny_overrides_cache_allowlist(tmp_path, monkeypatch):
    cache = tmp_path / "cache" / "scratch"
    cache.mkdir(parents=True)
    blocked = cache / "unverified-banner.png"
    blocked.write_bytes(b"\x89PNG\r\n\x1a\n")

    monkeypatch.setattr(base, "MEDIA_DELIVERY_SAFE_ROOTS", (cache,))
    monkeypatch.setattr(media_policy, "_routed_gateway_cfg", lambda: None)
    monkeypatch.setattr(
        media_policy,
        "_load_gateway_cfg",
        lambda config=None: {"media_delivery_deny_dirs": [str(cache)]},
    )

    assert base.validate_media_delivery_path(str(blocked)) is None


def test_operator_deny_does_not_block_neighboring_cache_root(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    denied = cache / "scratch"
    allowed = cache / "images"
    denied.mkdir(parents=True)
    allowed.mkdir(parents=True)
    deliverable = allowed / "approved.png"
    deliverable.write_bytes(b"\x89PNG\r\n\x1a\n")

    monkeypatch.setattr(base, "MEDIA_DELIVERY_SAFE_ROOTS", (cache,))
    monkeypatch.setattr(media_policy, "_routed_gateway_cfg", lambda: None)
    monkeypatch.setattr(
        media_policy,
        "_load_gateway_cfg",
        lambda config=None: {"media_delivery_deny_dirs": [str(denied)]},
    )

    assert base.validate_media_delivery_path(str(deliverable)) == str(deliverable.resolve())
