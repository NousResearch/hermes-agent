"""Regression: the WhatsApp inbound-media path validator follows the active
profile override.

The bridge writes inbound media into the active profile's cache (resolved via
get_image_cache_dir() etc.). _is_allowed_bridge_path() validates those paths.
It previously checked against IMAGE_CACHE_DIR / AUDIO_CACHE_DIR module
constants value-imported at import time, so under a profile override the
validator rejected media the bridge had legitimately written into the override
profile's cache. The validator must resolve the cache roots per-call.
"""
from pathlib import Path

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _make_profile(root: Path) -> Path:
    for sub in ("cache/images", "cache/audio", "cache/videos", "cache/documents"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root


def test_validator_follows_override_switch(tmp_path):
    """A path under profile A is rejected while the override is profile B."""
    from plugins.platforms.whatsapp.adapter import _is_allowed_bridge_path

    prof_a = _make_profile(tmp_path / "profA")
    prof_b = _make_profile(tmp_path / "profB")
    a_media = prof_a / "cache" / "images" / "img_a.jpg"
    a_media.write_bytes(b"\xff\xd8\xff\x00")

    # Under B, B's own media validates...
    b_media = prof_b / "cache" / "images" / "img_b.jpg"
    b_media.write_bytes(b"\xff\xd8\xff\x00")
    token = set_hermes_home_override(str(prof_b))
    try:
        assert _is_allowed_bridge_path(str(b_media)) is True
        # ...and a path from a *different* profile is not in B's cache roots.
        assert _is_allowed_bridge_path(str(a_media)) is False
    finally:
        reset_hermes_home_override(token)


