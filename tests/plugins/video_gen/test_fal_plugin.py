"""Tests for the FAL video gen plugin — family routing, payload shape."""

from __future__ import annotations

import pytest

from agent import video_gen_registry


@pytest.fixture(autouse=True)
def _reset_registry():
    video_gen_registry._reset_for_tests()
    yield
    video_gen_registry._reset_for_tests()


def test_fal_provider_registers():
    from plugins.video_gen.fal import FALVideoGenProvider, DEFAULT_MODEL

    provider = FALVideoGenProvider()
    video_gen_registry.register_provider(provider)

    assert video_gen_registry.get_provider("fal") is provider
    assert provider.display_name == "FAL"
    # DEFAULT_MODEL is the cheap-tier default
    assert provider.default_model() == DEFAULT_MODEL
    assert DEFAULT_MODEL in {"pixverse-v6", "ltx-2.3"}


def test_kling_4k_uses_start_image_url():
    """Kling v3 4K's image-to-video endpoint expects start_image_url,
    not image_url. The family must declare image_param_key='start_image_url'."""
    from plugins.video_gen.fal import FAL_FAMILIES, _build_payload

    meta = FAL_FAMILIES["kling-v3-4k"]
    assert meta.get("image_param_key") == "start_image_url"
    payload = _build_payload(
        meta,
        prompt="x",
        image_url="https://example.com/i.png",
        duration=5,
        aspect_ratio="16:9",
        resolution="720p",
        negative_prompt=None,
        audio=None,
        seed=None,
    )
    assert payload.get("start_image_url") == "https://example.com/i.png"
    assert "image_url" not in payload


def test_minimax_h3_int_duration_and_resolution_alias():
    """MiniMax H3 requires duration as a JSON integer and uses the
    768P/2K/4K resolution enum — the tool's 720p/1080p values must map."""
    from plugins.video_gen.fal import FAL_FAMILIES, _build_payload

    meta = FAL_FAMILIES["minimax-h3"]
    payload = _build_payload(
        meta,
        prompt="x",
        image_url=None,
        duration=7,
        aspect_ratio="16:9",
        resolution="720p",
        negative_prompt=None,
        audio=True,
        seed=None,
    )
    assert payload["duration"] == 7 and isinstance(payload["duration"], int)
    assert payload["resolution"] == "768P"
    assert payload["aspect_ratio"] == "16:9"
    # H3 has no generate_audio key (audio is native/always-on)
    assert "generate_audio" not in payload

    hi = _build_payload(
        meta, prompt="x", image_url=None, duration=5, aspect_ratio="16:9",
        resolution="1080p", negative_prompt=None, audio=None, seed=None,
    )
    assert hi["resolution"] == "2K"


def test_image_drop_keys_strips_aspect_ratio_on_i2v():
    """Seedance 2.5 / MiniMax H3 / Grok 1.5 i2v endpoints derive the
    aspect ratio from the input image; sending the key is rejected."""
    from plugins.video_gen.fal import FAL_FAMILIES, _build_payload

    for fid in ("seedance-2.5", "minimax-h3", "grok-imagine-1.5"):
        meta = FAL_FAMILIES[fid]
        i2v = _build_payload(
            meta, prompt="x", image_url="https://example.com/i.png",
            duration=5, aspect_ratio="16:9", resolution="480p",
            negative_prompt=None, audio=None, seed=None,
        )
        assert "aspect_ratio" not in i2v, fid
        # ...but text-to-video keeps it
        t2v = _build_payload(
            meta, prompt="x", image_url=None, duration=5,
            aspect_ratio="16:9", resolution="480p",
            negative_prompt=None, audio=None, seed=None,
        )
        assert t2v.get("aspect_ratio") == "16:9", fid


def test_seedance_25_string_duration_up_to_30():
    """Seedance 2.5 keeps the stringified duration convention and supports
    the full 4-30s range."""
    from plugins.video_gen.fal import FAL_FAMILIES, _build_payload

    meta = FAL_FAMILIES["seedance-2.5"]
    payload = _build_payload(
        meta, prompt="x", image_url=None, duration=30, aspect_ratio="1:1",
        resolution="480p", negative_prompt=None, audio=True, seed=None,
    )
    assert payload["duration"] == "30"
    assert payload["generate_audio"] is True


def test_gemini_omni_flash_is_image_only():
    """Gemini Omni Flash has no t2v endpoint on FAL — text jobs must
    error cleanly instead of submitting to a None endpoint."""
    from plugins.video_gen.fal import FAL_FAMILIES

    meta = FAL_FAMILIES["gemini-omni-flash"]
    assert meta.get("text_endpoint") is None
    assert meta.get("image_endpoint")


def test_every_family_has_required_metadata():
    """Invariant: every family entry carries the picker-facing metadata and
    at least one endpoint."""
    from plugins.video_gen.fal import FAL_FAMILIES

    for fid, meta in FAL_FAMILIES.items():
        assert meta.get("display"), fid
        assert meta.get("tier") in {"cheap", "premium"}, fid
        assert meta.get("text_endpoint") or meta.get("image_endpoint"), fid


class TestFamilyRouting:
    """The headline behavior: image_url presence picks the endpoint."""

    @pytest.fixture
    def with_fake_fal(self, monkeypatch):
        """Stub fal_client.submit to capture which endpoint we hit."""
        import sys
        import types

        captured = {"endpoint": None, "arguments": None}

        class FakeHandle:
            def get(self):
                return {"video": {"url": "https://fake/out.mp4"}}

        fake = types.ModuleType("fal_client")
        def _submit(endpoint, arguments=None, headers=None):
            captured["endpoint"] = endpoint
            captured["arguments"] = arguments
            return FakeHandle()
        fake.submit = _submit  # type: ignore
        monkeypatch.setitem(sys.modules, "fal_client", fake)

        # Reset the lazy global so it picks up our stub
        from plugins.video_gen import fal as fal_plugin
        fal_plugin._fal_client = None
        # Also reset the managed client cache
        fal_plugin._managed_fal_video_client = None
        fal_plugin._managed_fal_video_client_config = None

        monkeypatch.setenv("FAL_KEY", "test")
        # Force direct mode — no managed gateway
        monkeypatch.setattr(fal_plugin, "_resolve_managed_fal_video_gateway", lambda: None)
        return captured

    def test_text_to_video_routes_to_text_endpoint(self, with_fake_fal):
        from plugins.video_gen.fal import FALVideoGenProvider

        result = FALVideoGenProvider().generate(
            "a dog running",
            model="pixverse-v6",
        )
        assert result["success"] is True
        assert with_fake_fal["endpoint"] == "fal-ai/pixverse/v6/text-to-video"
        assert result["modality"] == "text"
        assert with_fake_fal["arguments"]["prompt"] == "a dog running"
        assert "image_url" not in with_fake_fal["arguments"]

    def test_image_to_video_routes_to_image_endpoint(self, with_fake_fal):
        from plugins.video_gen.fal import FALVideoGenProvider

        result = FALVideoGenProvider().generate(
            "animate this dog",
            model="pixverse-v6",
            image_url="https://example.com/dog.png",
        )
        assert result["success"] is True
        assert with_fake_fal["endpoint"] == "fal-ai/pixverse/v6/image-to-video"
        assert result["modality"] == "image"
        assert with_fake_fal["arguments"]["image_url"] == "https://example.com/dog.png"

    def test_default_family_text_routing(self, with_fake_fal):
        """No model arg → DEFAULT_MODEL → text-to-video endpoint."""
        from plugins.video_gen.fal import FALVideoGenProvider, FAL_FAMILIES, DEFAULT_MODEL

        result = FALVideoGenProvider().generate("a dog")
        assert result["success"] is True
        expected_endpoint = FAL_FAMILIES[DEFAULT_MODEL]["text_endpoint"]
        assert with_fake_fal["endpoint"] == expected_endpoint


    def test_unknown_family_falls_back_to_default(self, with_fake_fal):
        from plugins.video_gen.fal import FALVideoGenProvider, FAL_FAMILIES, DEFAULT_MODEL

        result = FALVideoGenProvider().generate(
            "x",
            model="not-a-real-family",
        )
        assert result["success"] is True
        expected_endpoint = FAL_FAMILIES[DEFAULT_MODEL]["text_endpoint"]
        assert with_fake_fal["endpoint"] == expected_endpoint

    def test_premium_seedance_routing(self, with_fake_fal):
        """Sanity check the premium-tier seedance routes correctly."""
        from plugins.video_gen.fal import FALVideoGenProvider

        result = FALVideoGenProvider().generate(
            "a dog",
            model="seedance-2.0",
            image_url="https://example.com/dog.png",
        )
        assert result["success"] is True
        assert with_fake_fal["endpoint"] == "bytedance/seedance-2.0/image-to-video"
        # Seedance uses regular image_url (not start_image_url)
        assert with_fake_fal["arguments"]["image_url"] == "https://example.com/dog.png"


class TestPayloadBuilder:
    def test_drops_unsupported_keys(self):
        """Veo enum-clamps duration, supports aspect+resolution+audio+neg."""
        from plugins.video_gen.fal import FAL_FAMILIES, _build_payload

        meta = FAL_FAMILIES["veo3.1"]
        p = _build_payload(
            meta,
            prompt="x",
            image_url=None,
            duration=12,           # not in enum (4,6,8) — snap to 8
            aspect_ratio="16:9",
            resolution="720p",
            negative_prompt="ugly",
            audio=True,
            seed=42,
        )
        assert p["prompt"] == "x"
        assert p["duration"] == "8s"  # veo3.1 uses "Ns" format per FAL API
        assert p["aspect_ratio"] == "16:9"
        assert p["resolution"] == "720p"
        assert p["generate_audio"] is True
        assert p["negative_prompt"] == "ugly"
        assert p["seed"] == 42

    def test_pixverse_range_clamps_correctly(self):
        from plugins.video_gen.fal import FAL_FAMILIES, _build_payload

        meta = FAL_FAMILIES["pixverse-v6"]
        p = _build_payload(
            meta,
            prompt="x",
            image_url="https://i.png",
            duration=99,        # over max → 15
            aspect_ratio="16:9",
            resolution="540p",
            negative_prompt=None,
            audio=None,
            seed=None,
        )
        assert p["duration"] == "15"


    def test_ltx_omits_duration_aspect_resolution(self):
        """LTX 2.3 doesn't declare duration/aspect/resolution enums —
        the payload should NOT include those keys (let FAL default)."""
        from plugins.video_gen.fal import FAL_FAMILIES, _build_payload

        meta = FAL_FAMILIES["ltx-2.3"]
        p = _build_payload(
            meta,
            prompt="x",
            image_url=None,
            duration=8,
            aspect_ratio="16:9",
            resolution="720p",
            negative_prompt="ugly",
            audio=True,
            seed=None,
        )
        assert "duration" not in p
        assert "aspect_ratio" not in p
        assert "resolution" not in p
        # But audio + negative are advertised
        assert p["generate_audio"] is True
        assert p["negative_prompt"] == "ugly"

    def test_range_families_omit_duration_when_unspecified(self):
        """Range-based families must omit `duration` when the caller doesn't
        specify one so FAL applies its endpoint default, not the minimum."""
        from plugins.video_gen.fal import FAL_FAMILIES, _build_payload

        for family_id in ("pixverse-v6", "seedance-2.0", "kling-v3-4k"):
            meta = FAL_FAMILIES[family_id]
            p = _build_payload(
                meta,
                prompt="x",
                image_url=None,
                duration=None,
                aspect_ratio="16:9",
                resolution="720p",
                negative_prompt=None,
                audio=None,
                seed=None,
            )
            assert "duration" not in p, (
                f"{family_id}: duration=None should omit the field, "
                f"got {p.get('duration')!r}"
            )

    def test_happy_horse_minimal_payload(self):
        """Happy Horse has sparse docs — payload should be minimal."""
        from plugins.video_gen.fal import FAL_FAMILIES, _build_payload

        meta = FAL_FAMILIES["happy-horse"]
        p = _build_payload(
            meta,
            prompt="a horse galloping",
            image_url=None,
            duration=8,
            aspect_ratio="16:9",
            resolution="720p",
            negative_prompt="watermark",
            audio=True,
            seed=None,
        )
        # Only prompt — no payload bloat for fields we can't verify
        assert p == {"prompt": "a horse galloping"}


class TestUpscalePass:
    """Opt-in SeedVR2 upscale chain after generation."""

    @pytest.fixture
    def with_fake_fal(self, monkeypatch):
        """Stub fal_client.submit, capturing every endpoint hit in order."""
        import sys
        import types

        captured = {"calls": []}

        class FakeHandle:
            def __init__(self, endpoint):
                self._endpoint = endpoint

            def get(self):
                if self._endpoint.endswith("upscale/video"):
                    return {"video": {"url": "https://fake/upscaled.mp4"}}
                return {"video": {"url": "https://fake/native.mp4"}}

        fake = types.ModuleType("fal_client")
        def _submit(endpoint, arguments=None, headers=None):
            captured["calls"].append((endpoint, arguments))
            return FakeHandle(endpoint)
        fake.submit = _submit  # type: ignore
        monkeypatch.setitem(sys.modules, "fal_client", fake)

        from plugins.video_gen import fal as fal_plugin
        fal_plugin._fal_client = None
        fal_plugin._managed_fal_video_client = None
        fal_plugin._managed_fal_video_client_config = None

        monkeypatch.setenv("FAL_KEY", "test")
        monkeypatch.setattr(fal_plugin, "_resolve_managed_fal_video_gateway", lambda: None)
        return captured

    def test_upscale_chains_seedvr(self, with_fake_fal):
        from plugins.video_gen.fal import FALVideoGenProvider, UPSCALER_ENDPOINT

        result = FALVideoGenProvider().generate(
            "a dog", model="pixverse-v6", upscale=True,
        )
        assert result["success"] is True
        assert result["video"] == "https://fake/upscaled.mp4"
        assert result["upscaled"] is True
        assert result["upscale_factor"] == 2
        endpoints = [c[0] for c in with_fake_fal["calls"]]
        assert endpoints == ["fal-ai/pixverse/v6/text-to-video", UPSCALER_ENDPOINT]
        # Upscale request carries the native URL + factor mode.
        upscale_args = with_fake_fal["calls"][1][1]
        assert upscale_args["video_url"] == "https://fake/native.mp4"
        assert upscale_args["upscale_mode"] == "factor"

    def test_no_upscale_by_default(self, with_fake_fal):
        from plugins.video_gen.fal import FALVideoGenProvider

        result = FALVideoGenProvider().generate("a dog", model="pixverse-v6")
        assert result["success"] is True
        assert result["video"] == "https://fake/native.mp4"
        assert result["upscaled"] is False
        assert len(with_fake_fal["calls"]) == 1

    def test_upscale_failure_falls_back_to_native(self, with_fake_fal, monkeypatch):
        from plugins.video_gen import fal as fal_plugin
        from plugins.video_gen.fal import FALVideoGenProvider

        monkeypatch.setattr(fal_plugin, "_upscale_video", lambda url: None)
        result = FALVideoGenProvider().generate(
            "a dog", model="pixverse-v6", upscale=True,
        )
        assert result["success"] is True
        assert result["video"] == "https://fake/native.mp4"
        assert result["upscaled"] is False
