"""Native-detail OpenAI image patch preflight regression tests."""

from __future__ import annotations

import base64
import io

from PIL import Image

from agent.image_patch_preflight import (
    OPENAI_IMAGE_MAX_PATCHES,
    OPENAI_IMAGE_TARGET_PATCHES,
    fit_dimensions_to_patch_budget,
    image_patch_count,
    image_patch_preflight_settings,
    is_gpt56_image_patch_limited,
    is_image_patch_preflight_excluded,
    is_openai_image_patch_limited,
    is_openai_image_patch_preflight_enabled,
    is_patch_overage_error,
    request_with_image_patch_preflight,
    resize_image_parts_to_patch_budget,
)


def _jpeg_data_url(size: tuple[int, int]) -> str:
    output = io.BytesIO()
    Image.new("RGB", size, color=(24, 72, 120)).save(output, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def _data_url_dimensions(data_url: str) -> tuple[int, int]:
    payload = base64.b64decode(data_url.partition(",")[2])
    with Image.open(io.BytesIO(payload)) as image:
        return image.size


def test_incident_dimensions_match_provider_patch_count():
    assert image_patch_count(7008, 4672) == 31_974
    fitted = fit_dimensions_to_patch_budget(7008, 4672)
    assert fitted[0] < 7008
    assert fitted[1] < 4672
    assert image_patch_count(*fitted) <= 27_000


def test_model_family_match_is_narrow():
    assert is_gpt56_image_patch_limited("gpt-5.6-sol")
    assert is_gpt56_image_patch_limited("openai/gpt-5.6-terra")
    assert not is_gpt56_image_patch_limited("gpt-5.5")
    assert not is_gpt56_image_patch_limited("my-gpt-5.6-copy")


def test_native_detail_model_match_covers_current_model_and_snapshots():
    assert is_openai_image_patch_limited("gpt-6-astra")
    assert is_openai_image_patch_limited("openai/gpt-6-astra-2026-09-01")
    assert is_openai_image_patch_limited("gpt-5.6-sol")
    assert not is_openai_image_patch_limited("my-gpt-6-astra-copy")
    assert not is_openai_image_patch_limited("gpt-6-astral")
    assert not is_openai_image_patch_limited("gpt-5.5")


def test_several_images_each_receive_their_own_budget():
    original = _jpeg_data_url((65, 65))
    parts = [{"type": "input_image", "image_url": original} for _ in range(12)]
    messages = [{"role": "user", "content": parts}]
    changed = resize_image_parts_to_patch_budget(
        messages, max_patches=8, target_patches=4,
    )
    assert changed == 12
    assert all(part["image_url"] == original for part in parts)
    assert all(
        image_patch_count(*_data_url_dimensions(part["image_url"])) <= 4
        for part in messages[0]["content"]
    )


def test_profile_setting_is_opt_in_and_fail_closed(monkeypatch):
    import hermes_cli.config

    monkeypatch.setattr(
        hermes_cli.config,
        "load_config",
        lambda: {"agent": {"openai_image_patch_preflight": True}},
    )
    assert is_openai_image_patch_preflight_enabled() is True

    monkeypatch.setattr(hermes_cli.config, "load_config", lambda: {"agent": {}})
    assert is_openai_image_patch_preflight_enabled() is False

    def _unreadable_config():
        raise OSError("unreadable")

    monkeypatch.setattr(hermes_cli.config, "load_config", _unreadable_config)
    assert is_openai_image_patch_preflight_enabled() is False


def test_settings_support_thresholds_safety_ratio_and_model_exclusions(monkeypatch):
    import hermes_cli.config

    monkeypatch.setattr(
        hermes_cli.config,
        "load_config",
        lambda: {"agent": {
            "openai_image_patch_preflight": True,
            "openai_image_patch_max_patches": 20_000,
            "openai_image_patch_target_ratio": 0.8,
            "openai_image_patch_excluded_models": ["local/*", "exact-model"],
        }},
    )

    settings = image_patch_preflight_settings()
    assert settings.enabled is True
    assert settings.max_patches == 20_000
    assert settings.target_patches == 16_000
    assert is_image_patch_preflight_excluded("local/vision-model", settings.excluded_models)
    assert is_image_patch_preflight_excluded("exact-model", settings.excluded_models)
    assert not is_image_patch_preflight_excluded("other-vision-model", settings.excluded_models)


def test_config_schema_exposes_opt_in_patch_preflight_defaults():
    from hermes_cli.config import DEFAULT_CONFIG

    agent = DEFAULT_CONFIG["agent"]
    assert agent["openai_image_patch_preflight"] is False
    assert agent["openai_image_patch_max_patches"] == 30_000
    assert agent["openai_image_patch_target_patches"] == 27_000
    assert agent["openai_image_patch_target_ratio"] == 0.9
    assert agent["openai_image_patch_excluded_models"] == []


def test_settings_defaults_keep_feature_disabled_with_30k_limit_and_90_percent_target(monkeypatch):
    import hermes_cli.config

    monkeypatch.setattr(hermes_cli.config, "load_config", lambda: {"agent": {}})
    settings = image_patch_preflight_settings()
    assert settings.enabled is False
    assert settings.max_patches == OPENAI_IMAGE_MAX_PATCHES == 30_000
    assert settings.target_patches == OPENAI_IMAGE_TARGET_PATCHES == 27_000


def test_max_override_without_target_override_keeps_90_percent_margin(monkeypatch):
    import hermes_cli.config

    monkeypatch.setattr(
        hermes_cli.config,
        "load_config",
        lambda: {"agent": {
            "openai_image_patch_preflight": True,
            "openai_image_patch_max_patches": 20_000,
            "openai_image_patch_target_patches": OPENAI_IMAGE_TARGET_PATCHES,
            "openai_image_patch_target_ratio": 0.9,
        }},
    )

    settings = image_patch_preflight_settings()
    assert settings.max_patches == 20_000
    assert settings.target_patches == 18_000


def test_explicit_target_wins_over_ratio_even_when_equal_to_default(monkeypatch):
    import hermes_cli.config

    monkeypatch.setattr(
        hermes_cli.config,
        "load_config",
        lambda: {"agent": {
            "openai_image_patch_preflight": True,
            "openai_image_patch_max_patches": 40_000,
            "openai_image_patch_target_patches": OPENAI_IMAGE_TARGET_PATCHES,
            "openai_image_patch_target_ratio": 0.5,
        }},
    )
    monkeypatch.setattr(
        hermes_cli.config,
        "read_user_config_raw",
        lambda: {"agent": {"openai_image_patch_target_patches": OPENAI_IMAGE_TARGET_PATCHES}},
    )

    settings = image_patch_preflight_settings()
    assert settings.target_patches == OPENAI_IMAGE_TARGET_PATCHES


def test_merged_default_target_does_not_masquerade_as_explicit(monkeypatch):
    import hermes_cli.config

    monkeypatch.setattr(
        hermes_cli.config,
        "load_config",
        lambda: {"agent": {
            "openai_image_patch_max_patches": 40_000,
            "openai_image_patch_target_patches": OPENAI_IMAGE_TARGET_PATCHES,
            "openai_image_patch_target_ratio": 0.9,
        }},
    )
    monkeypatch.setattr(
        hermes_cli.config,
        "read_user_config_raw",
        lambda: {"agent": {"openai_image_patch_max_patches": 40_000}},
    )

    settings = image_patch_preflight_settings()
    assert settings.target_patches == 36_000


def test_all_models_are_eligible_unless_configured_excluded(monkeypatch):
    import hermes_cli.config

    monkeypatch.setattr(
        hermes_cli.config,
        "load_config",
        lambda: {"agent": {"openai_image_patch_preflight": True}},
    )
    original = _jpeg_data_url((65, 65))
    request = {"model": "claude-vision-future", "messages": [{
        "role": "user", "content": [{"type": "input_image", "image_url": original}],
    }]}
    seen = []

    request_with_image_patch_preflight(
        request,
        lambda kwargs: seen.append(kwargs) or "ok",
        max_patches=8,
        target_patches=4,
    )

    assert _data_url_dimensions(seen[0]["messages"][0]["content"][0]["image_url"])[0] <= 64
    assert request["messages"][0]["content"][0]["image_url"] == original


def test_patch_overage_http_400_resizes_and_retries_exactly_once(monkeypatch):
    import hermes_cli.config

    monkeypatch.setattr(
        hermes_cli.config,
        "load_config",
        lambda: {"agent": {"openai_image_patch_preflight": True}},
    )
    original = _jpeg_data_url((65, 65))
    request = {"model": "any-vision-model", "messages": [{
        "role": "user", "content": [{"type": "input_image", "image_url": original}],
    }]}
    calls = []

    class PatchOverage(Exception):
        status_code = 400

    def send(kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise PatchOverage("Image is 9 patches; maximum is 8 patches")
        return "retried"

    result = request_with_image_patch_preflight(
        request,
        send,
        max_patches=100,
        target_patches=4,
        retry_max_patches=8,
    )

    assert result == "retried"
    assert len(calls) == 2
    assert calls[0]["messages"][0]["content"][0]["image_url"] == original
    assert image_patch_count(*_data_url_dimensions(
        calls[1]["messages"][0]["content"][0]["image_url"],
    )) < 4
    assert request["messages"][0]["content"][0]["image_url"] == original


def test_patch_overage_does_not_retry_when_no_smaller_positive_budget_exists(monkeypatch):
    import hermes_cli.config

    monkeypatch.setattr(
        hermes_cli.config,
        "load_config",
        lambda: {"agent": {"openai_image_patch_preflight": True}},
    )
    monkeypatch.setattr(hermes_cli.config, "read_user_config_raw", lambda: {"agent": {}})
    calls = []

    class PatchOverage(Exception):
        status_code = 400

    def send(kwargs):
        calls.append(kwargs)
        raise PatchOverage("Image requires 2 patches; maximum is 1 patch")

    try:
        request_with_image_patch_preflight(
            {"model": "vision", "messages": []},
            send,
            max_patches=1,
            target_patches=1,
        )
    except PatchOverage as exc:
        assert "image patch preflight enabled=true" in str(exc).lower()
    else:
        raise AssertionError("expected patch overage")
    assert len(calls) == 1


def test_patch_overage_recovery_is_stricter_than_successful_preflight(monkeypatch):
    import hermes_cli.config

    monkeypatch.setattr(
        hermes_cli.config,
        "load_config",
        lambda: {"agent": {"openai_image_patch_preflight": True}},
    )
    monkeypatch.setattr(hermes_cli.config, "read_user_config_raw", lambda: {"agent": {}})
    original = _jpeg_data_url((321, 321))  # 11x11 = 121 patches
    request = {"model": "vision", "messages": [{
        "role": "user", "content": [{"type": "input_image", "image_url": original}],
    }], "stream": False}
    calls = []

    class PatchOverage(Exception):
        status_code = 400

    def send(kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise PatchOverage("Image requires 80 patches; maximum is 60 patches")
        return "retried"

    result = request_with_image_patch_preflight(
        request, send, max_patches=120, target_patches=80,
    )

    assert result == "retried"
    first_url = calls[0]["messages"][0]["content"][0]["image_url"]
    retry_url = calls[1]["messages"][0]["content"][0]["image_url"]
    first_patches = image_patch_count(*_data_url_dimensions(first_url))
    retry_patches = image_patch_count(*_data_url_dimensions(retry_url))
    assert first_patches <= 80
    assert retry_patches < first_patches
    assert retry_patches <= 60


def test_chat_completions_lazy_patch_error_retries_before_first_event(monkeypatch):
    import hermes_cli.config

    monkeypatch.setattr(hermes_cli.config, "load_config", lambda: {"agent": {"openai_image_patch_preflight": True}})
    monkeypatch.setattr(hermes_cli.config, "read_user_config_raw", lambda: {"agent": {}})
    original = _jpeg_data_url((97, 97))
    request = {"model": "vision", "stream": True, "messages": [{
        "role": "user", "content": [{"type": "input_image", "image_url": original}],
    }]}
    calls = []

    class PatchOverage(Exception):
        status_code = 400

    def send(kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            def failed_stream():
                raise PatchOverage("Image requires 8 patches; maximum is 6 patches")
                yield
            return failed_stream()
        return iter(["event"])

    stream = request_with_image_patch_preflight(request, send, max_patches=16, target_patches=8)
    assert list(stream) == ["event"]
    assert len(calls) == 2


def test_responses_lazy_patch_error_after_event_is_annotated_without_replay(monkeypatch):
    import hermes_cli.config

    monkeypatch.setattr(hermes_cli.config, "load_config", lambda: {"agent": {"openai_image_patch_preflight": True}})
    monkeypatch.setattr(hermes_cli.config, "read_user_config_raw", lambda: {"agent": {}})
    original = _jpeg_data_url((97, 97))
    request = {"model": "vision", "stream": True, "input": [{
        "role": "user", "content": [{"type": "input_image", "image_url": original}],
    }]}
    calls = []

    class PatchOverage(Exception):
        status_code = 400

    def send(kwargs):
        calls.append(kwargs)

        def partial_stream():
            yield "already-emitted"
            raise PatchOverage("maximum image patches exceeded")

        return partial_stream()

    stream = request_with_image_patch_preflight(request, send, max_patches=16, target_patches=8)
    iterator = iter(stream)
    assert next(iterator) == "already-emitted"
    try:
        next(iterator)
    except PatchOverage as exc:
        assert "image patch preflight enabled=true" in str(exc).lower()
    else:
        raise AssertionError("expected lazy patch overage")
    assert len(calls) == 1


def test_patch_overage_error_reports_preflight_state_and_does_not_retry_when_disabled(monkeypatch):
    import hermes_cli.config

    monkeypatch.setattr(hermes_cli.config, "load_config", lambda: {"agent": {}})
    calls = []

    class PatchOverage(Exception):
        status_code = 400

    def send(kwargs):
        calls.append(kwargs)
        raise PatchOverage("maximum image patches exceeded")

    try:
        request_with_image_patch_preflight({"model": "vision", "messages": []}, send)
    except PatchOverage as exc:
        assert "image patch preflight enabled=false" in str(exc).lower()
    else:
        raise AssertionError("expected patch overage")
    assert len(calls) == 1


def test_patch_overage_detection_requires_http_400_and_patch_language():
    class Error(Exception):
        def __init__(self, message, status_code):
            super().__init__(message)
            self.status_code = status_code

    assert is_patch_overage_error(Error("maximum image patches exceeded", 400))
    assert not is_patch_overage_error(Error("maximum image patches exceeded", 500))
    assert not is_patch_overage_error(Error("invalid request", 400))


def test_main_chat_provider_dispatch_applies_preflight(monkeypatch):
    import hermes_cli.config
    from types import SimpleNamespace
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request

    monkeypatch.setattr(
        hermes_cli.config,
        "load_config",
        lambda: {"agent": {
            "openai_image_patch_preflight": True,
            "openai_image_patch_max_patches": 8,
            "openai_image_patch_target_patches": 4,
        }},
    )
    original = _jpeg_data_url((65, 65))
    seen = []

    class Completions:
        def create(self, **kwargs):
            seen.append(kwargs)
            return "ok"

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    agent = SimpleNamespace(api_mode="chat_completions", provider="openai", client=client)
    request = {"model": "future-vision", "messages": [{
        "role": "user", "content": [{"type": "input_image", "image_url": original}],
    }]}

    result = _dispatch_nonstreaming_api_request(agent, request, make_client=lambda *a, **k: client)

    assert result == "ok"
    assert image_patch_count(*_data_url_dimensions(
        seen[0]["messages"][0]["content"][0]["image_url"],
    )) <= 4
    assert request["messages"][0]["content"][0]["image_url"] == original


def test_direct_user_image_is_resized_without_mutating_aliased_history():
    original = _jpeg_data_url((65, 65))  # 3x3 = 9 patches
    history_part = {"type": "image_url", "image_url": {"url": original}}
    history_message = {"role": "user", "content": [history_part]}
    api_message = history_message.copy()
    api_messages = [api_message]

    changed = resize_image_parts_to_patch_budget(
        api_messages,
        max_patches=8,
        target_patches=4,
    )

    assert changed == 1
    resized_url = api_message["content"][0]["image_url"]["url"]
    assert resized_url != original
    assert image_patch_count(*_data_url_dimensions(resized_url)) <= 4
    assert history_message["content"][0] is history_part
    assert history_part["image_url"]["url"] == original


def test_responses_function_output_image_is_resized():
    original = _jpeg_data_url((65, 65))
    messages = [{
        "type": "function_call_output",
        "call_id": "call_vision",
        "output": [
            {"type": "input_text", "text": "Image loaded."},
            {"type": "input_image", "image_url": original},
        ],
    }]

    changed = resize_image_parts_to_patch_budget(
        messages,
        max_patches=8,
        target_patches=4,
    )

    assert changed == 1
    resized_url = messages[0]["output"][1]["image_url"]
    assert image_patch_count(*_data_url_dimensions(resized_url)) <= 4


def test_in_budget_image_is_left_byte_identical():
    original = _jpeg_data_url((64, 64))  # 2x2 = 4 patches
    messages = [{
        "role": "user",
        "content": [{"type": "input_image", "image_url": original}],
    }]

    changed = resize_image_parts_to_patch_budget(
        messages,
        max_patches=8,
        target_patches=4,
    )

    assert changed == 0
    assert messages[0]["content"][0]["image_url"] == original
