"""Tests for the FAL video gen plugin — family routing, payload shape."""

from __future__ import annotations

from unittest.mock import Mock

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

    def test_flux3_image_to_video_unchanged(self, with_fake_fal):
        from plugins.video_gen.fal import FALVideoGenProvider

        result = FALVideoGenProvider().generate(
            "animate this",
            model="flux-3",
            image_url="https://example.com/start.png",
            duration=20,
        )
        assert result["success"] is True
        assert with_fake_fal["endpoint"] == "blackforestlabs/flux-3/image-to-video"
        assert with_fake_fal["arguments"]["image_url"] == "https://example.com/start.png"
        assert "keyframes" not in with_fake_fal["arguments"]

    def test_flux3_draft_image_to_video(self, with_fake_fal):
        from plugins.video_gen.fal import FALVideoGenProvider

        result = FALVideoGenProvider().generate(
            "animate this",
            model="flux-3",
            image_url="https://example.com/start.png",
            duration=20,
            draft=True,
        )
        assert result["success"] is True
        assert with_fake_fal["endpoint"] == "blackforestlabs/flux-3/image-to-video/draft"
        assert result.get("draft") is True
        assert "resolution" not in with_fake_fal["arguments"]

    def test_flux3_keyframes_to_video(self, with_fake_fal):
        from plugins.video_gen.fal import FALVideoGenProvider

        result = FALVideoGenProvider().generate(
            "walk from the tray to the bite",
            model="flux-3",
            duration=20,
            keyframes=[
                {"frame_index": 0, "image_url": "https://example.com/a.png"},
                {"frame_index": 480, "image_url": "https://example.com/b.png"},
            ],
        )
        assert result["success"] is True
        assert with_fake_fal["endpoint"] == "blackforestlabs/flux-3/keyframes-to-video"
        assert result["modality"] == "keyframes"
        assert with_fake_fal["arguments"]["keyframes"] == [
            {"frame_index": 0, "image_url": "https://example.com/a.png"},
            {"frame_index": 480, "image_url": "https://example.com/b.png"},
        ]
        assert "image_url" not in with_fake_fal["arguments"]

    def test_flux3_keyframes_draft(self, with_fake_fal):
        from plugins.video_gen.fal import FALVideoGenProvider

        result = FALVideoGenProvider().generate(
            "walk from the tray to the bite",
            model="flux-3",
            duration=10,
            draft=True,
            keyframes=[
                {"frame_index": 0, "image_url": "https://example.com/a.png"},
                {"frame_index": 120, "image_url": "https://example.com/b.png"},
            ],
        )
        assert result["success"] is True
        assert with_fake_fal["endpoint"] == "blackforestlabs/flux-3/keyframes-to-video/draft"

    def test_flux3_first_last_frame(self, with_fake_fal):
        from plugins.video_gen.fal import FALVideoGenProvider

        result = FALVideoGenProvider().generate(
            "bite the sandwich",
            model="flux-3",
            image_url="https://example.com/start.png",
            end_image_url="https://example.com/end.png",
            duration=8,
        )
        assert result["success"] is True
        assert with_fake_fal["endpoint"] == "blackforestlabs/flux-3/first-last-frame-to-video"
        args = with_fake_fal["arguments"]
        assert args["start_image_url"] == "https://example.com/start.png"
        assert args["end_image_url"] == "https://example.com/end.png"
        assert "image_url" not in args

    def test_flux3_draft_enhance(self, with_fake_fal):
        from plugins.video_gen.fal import FALVideoGenProvider

        result = FALVideoGenProvider().generate(
            "",
            model="flux-3",
            draft_cache_url="https://example.com/cache.bin",
        )
        assert result["success"] is True
        assert with_fake_fal["endpoint"] == "blackforestlabs/flux-3/draft-enhance"
        assert with_fake_fal["arguments"] == {
            "draft_cache_url": "https://example.com/cache.bin",
        }

    def test_flux3_keyframes_need_duration(self, with_fake_fal):
        from plugins.video_gen.fal import FALVideoGenProvider

        result = FALVideoGenProvider().generate(
            "x",
            model="flux-3",
            keyframes=[{"frame_index": 0, "image_url": "https://example.com/a.png"}],
        )
        assert result["success"] is False
        assert result["error_type"] == "invalid_request"

    def test_other_family_rejects_keyframes(self, with_fake_fal):
        from plugins.video_gen.fal import FALVideoGenProvider

        result = FALVideoGenProvider().generate(
            "x",
            model="pixverse-v6",
            duration=5,
            keyframes=[{"frame_index": 0, "image_url": "https://example.com/a.png"}],
        )
        assert result["success"] is False
        assert result["error_type"] == "modality_unsupported"


def test_flux3_keyframe_draft_and_enhance_use_managed_gateway(monkeypatch):
    """The Hermes-credit workflow must stay on managed fal-queue end to end."""
    from plugins.video_gen import fal as fal_plugin

    calls = []

    class FakeHandle:
        request_id = "managed-request-1"

        def __init__(self, result):
            self._result = result

        def get(self):
            return self._result

    class ManagedClient:
        def submit(self, endpoint, arguments=None, headers=None):
            calls.append({
                "endpoint": endpoint,
                "arguments": arguments,
                "headers": headers,
            })
            result = {"video": {"url": "https://managed.example/out.mp4"}}
            if endpoint.endswith("/draft"):
                result["draft_cache"] = {
                    "url": "https://managed.example/draft-cache.bin",
                }
            return FakeHandle(result)

    direct_client = Mock()
    direct_client.submit.side_effect = AssertionError(
        "Nous selection must not fall back to direct FAL billing"
    )
    managed_gateway = object()
    managed_client = ManagedClient()

    monkeypatch.setattr(fal_plugin, "_check_fal_video_available", lambda: True)
    monkeypatch.setattr(fal_plugin, "_load_fal_client", lambda: None)
    monkeypatch.setattr(
        fal_plugin,
        "_resolve_managed_fal_video_gateway",
        lambda: managed_gateway,
    )
    monkeypatch.setattr(
        fal_plugin,
        "_get_managed_fal_video_client",
        lambda gateway: managed_client if gateway is managed_gateway else None,
    )
    monkeypatch.setattr(fal_plugin, "_fal_client", direct_client)

    provider = fal_plugin.FALVideoGenProvider()
    draft = provider.generate(
        "move between frames",
        model="flux-3",
        duration=10,
        draft=True,
        keyframes=[
            {"frame_index": 0, "image_url": "https://example.com/a.png"},
            {"frame_index": 240, "image_url": "https://example.com/b.png"},
        ],
    )
    enhanced = provider.generate(
        "",
        model="flux-3",
        draft_cache_url=draft["draft_cache_url"],
    )

    assert draft["success"] is True
    assert enhanced["success"] is True
    assert [call["endpoint"] for call in calls] == [
        "blackforestlabs/flux-3/keyframes-to-video/draft",
        "blackforestlabs/flux-3/draft-enhance",
    ]
    assert calls[0]["arguments"]["keyframes"][1]["frame_index"] == 240
    assert "resolution" not in calls[0]["arguments"]
    assert calls[1]["arguments"] == {
        "draft_cache_url": "https://managed.example/draft-cache.bin",
    }
    direct_client.submit.assert_not_called()


class TestFamilyKeyNormalization:
    def test_full_endpoint_paths_resolve_to_their_own_family(self):
        """A configured endpoint path must resolve to the family that declares
        it. The segment scan alone reads the "seedance-2.0" in
        ".../seedance-2.0/mini/..." and bills the full-price family."""
        from plugins.video_gen.fal import FAL_FAMILIES, _family_endpoints, _normalize_family_key

        for fid, meta in FAL_FAMILIES.items():
            for endpoint in _family_endpoints(meta):
                assert _normalize_family_key(endpoint) == fid, endpoint

    def test_bare_and_prefixed_ids_still_resolve(self):
        from plugins.video_gen.fal import _normalize_family_key

        assert _normalize_family_key("seedance-2.5") == "seedance-2.5"
        assert _normalize_family_key("bytedance/seedance-2.5") == "seedance-2.5"
        assert _normalize_family_key("  pixverse-v6  ") == "pixverse-v6"
        assert _normalize_family_key("nonsense/thing") is None

    def test_truncated_endpoint_stems_resolve(self):
        """Config often stores the FAL app path without the modality leaf."""
        from plugins.video_gen.fal import _normalize_family_key

        assert _normalize_family_key("bytedance/seedance-2.0/mini") == "seedance-2.0-mini"
        assert _normalize_family_key("bytedance/seedance-2.0") == "seedance-2.0"
        assert _normalize_family_key("minimax/h3") == "minimax-h3"
        assert _normalize_family_key("xai/grok-imagine-video/v1.5") == "grok-imagine-1.5"
        assert _normalize_family_key("google/gemini-omni-flash") == "gemini-omni-flash"
        assert _normalize_family_key("blackforestlabs/flux-3") == "flux-3"

    def test_capabilities_span_longest_family_duration(self):
        """capabilities() is active-MODEL-aware (#95681 diet): it reports
        the resolved family's real window, so the schema doesn't overstate
        short families or understate Seedance 2.5. The union fallback
        (resolution failure) must still span the 30s ceiling."""
        from unittest.mock import patch as _patch

        import plugins.video_gen.fal as _fp
        from plugins.video_gen.fal import FAL_FAMILIES, FALVideoGenProvider

        # Active model resolved → that family's actual window.
        meta = FAL_FAMILIES["seedance-2.5"]
        with _patch.object(_fp, "_resolve_family",
                           return_value=("seedance-2.5", meta)):
            caps = FALVideoGenProvider().capabilities()
        assert caps["max_duration"] >= 30
        # A short family must NOT be inflated to the union ceiling.
        short = FAL_FAMILIES["pixverse-v6"]
        durs = short.get("durations")
        hi = durs[1] if isinstance(durs, tuple) else max(durs)
        with _patch.object(_fp, "_resolve_family",
                           return_value=("pixverse-v6", short)):
            caps = FALVideoGenProvider().capabilities()
        assert caps["max_duration"] == hi

        # Resolution failure → union fallback still spans the ceiling.
        with _patch.object(_fp, "_resolve_family",
                           side_effect=RuntimeError("no config")):
            caps = FALVideoGenProvider().capabilities()
        assert caps["max_duration"] >= 30
        assert caps["min_duration"] <= 1


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

    @pytest.mark.parametrize(
        "family_id",
        [
            "seedance-2.0",
            "seedance-2.0-mini",
            "seedance-2.5",
            "minimax-h3",
            "flux-3",
            "grok-imagine-1.5",
            "gemini-omni-flash",
        ],
    )
    def test_seed_dropped_for_families_without_seed_support(self, family_id):
        """These FAL endpoints declare no `seed`; the gateway forwards whatever
        we send, so an unknown key would reach the vendor."""
        from plugins.video_gen.fal import FAL_FAMILIES, _build_payload

        p = _build_payload(
            FAL_FAMILIES[family_id],
            prompt="x",
            image_url="https://i.png",
            duration=None,
            aspect_ratio="16:9",
            resolution="720p",
            negative_prompt=None,
            audio=None,
            seed=42,
        )
        assert "seed" not in p

    def test_minimax_h3_uses_uppercase_resolution_enum(self):
        """FAL spells MiniMax H3 resolutions "768P"/"2K"/"4K"; tool-style
        values like "720p" are aliased via resolution_aliases."""
        from plugins.video_gen.fal import FAL_FAMILIES, _build_payload

        meta = FAL_FAMILIES["minimax-h3"]
        accepted = _build_payload(
            meta, prompt="x", image_url=None, duration=7, aspect_ratio="16:9",
            resolution="2K", negative_prompt=None, audio=None, seed=None,
        )
        assert accepted["resolution"] == "2K"
        assert accepted["duration"] == 7

        aliased = _build_payload(
            meta, prompt="x", image_url=None, duration=7, aspect_ratio="16:9",
            resolution="720p", negative_prompt=None, audio=None, seed=None,
        )
        assert aliased["resolution"] == "768P"

    def test_audio_only_sent_for_families_that_declare_it(self):
        """minimax-h3 and the i2v-only families have no generate_audio field."""
        from plugins.video_gen.fal import FAL_FAMILIES, _build_payload

        for family_id in ("minimax-h3", "grok-imagine-1.5", "gemini-omni-flash"):
            p = _build_payload(
                FAL_FAMILIES[family_id],
                prompt="x", image_url="https://i.png", duration=None,
                aspect_ratio="16:9", resolution="720p", negative_prompt="ugly",
                audio=True, seed=None,
            )
            assert "generate_audio" not in p, family_id
            assert "negative_prompt" not in p, family_id

    @pytest.mark.parametrize(
        "family_id,expected",
        [
            ("minimax-h3", 7),          # FAL types duration as an integer
            ("flux-3", 7),              # mixed ["auto", 5, 6, ...] literal enum
            ("grok-imagine-1.5", 7),
            ("gemini-omni-flash", 7),
            ("seedance-2.5", "7"),      # FAL enum is strings: "auto","4",...
            ("seedance-2.0-mini", "7"),
            ("pixverse-v6", "7"),       # unchanged legacy string form
            ("veo3.1", "6s"),           # unchanged suffix form (7 snaps to 6)
        ],
    )
    def test_duration_is_emitted_in_the_form_fal_declares(self, family_id, expected):
        from plugins.video_gen.fal import FAL_FAMILIES, _build_payload

        p = _build_payload(
            FAL_FAMILIES[family_id],
            prompt="x", image_url=None, duration=7, aspect_ratio="16:9",
            resolution="720p", negative_prompt=None, audio=None, seed=None,
        )
        assert p["duration"] == expected
        assert type(p["duration"]) is type(expected)

    def test_i2v_only_families_declare_no_text_endpoint(self):
        """Catalog invariant: Gemini Omni Flash animates an existing image only."""
        from plugins.video_gen.fal import FAL_FAMILIES

        meta = FAL_FAMILIES["gemini-omni-flash"]
        assert meta.get("text_endpoint") is None
        assert meta["image_endpoint"]

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

        monkeypatch.setattr(
            fal_plugin,
            "_upscale_video",
            lambda url, source_request_id=None: None,
        )
        result = FALVideoGenProvider().generate(
            "a dog", model="pixverse-v6", upscale=True,
        )
        assert result["success"] is True
        assert result["video"] == "https://fake/native.mp4"
        assert result["upscaled"] is False

    def test_managed_upscale_binds_the_source_request(self, monkeypatch):
        from plugins.video_gen import fal as fal_plugin

        captured = {}

        class FakeHandle:
            def get(self):
                return {"video": {"url": "https://fake/upscaled.mp4"}}

        monkeypatch.setattr(
            fal_plugin,
            "_resolve_managed_fal_video_gateway",
            lambda: object(),
        )
        monkeypatch.setattr(
            fal_plugin,
            "_submit_fal_video_request",
            lambda endpoint, arguments: (
                captured.update(endpoint=endpoint, arguments=arguments)
                or FakeHandle()
            ),
        )

        assert (
            fal_plugin._upscale_video(
                "https://fake/native.mp4",
                "source-request-1",
            )
            == "https://fake/upscaled.mp4"
        )
        assert captured["arguments"]["source_request_id"] == "source-request-1"

    def test_managed_upscale_without_source_request_falls_back(self, monkeypatch):
        from plugins.video_gen import fal as fal_plugin

        submit = Mock()
        monkeypatch.setattr(
            fal_plugin,
            "_resolve_managed_fal_video_gateway",
            lambda: object(),
        )
        monkeypatch.setattr(fal_plugin, "_submit_fal_video_request", submit)

        assert fal_plugin._upscale_video("https://fake/native.mp4") is None
        submit.assert_not_called()
