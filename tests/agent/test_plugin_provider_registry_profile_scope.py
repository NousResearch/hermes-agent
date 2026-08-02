"""Profile isolation contracts shared by plugin provider registries."""

from __future__ import annotations

import os
import threading
from dataclasses import FrozenInstanceError

import pytest

from agent import (
    browser_registry,
    image_gen_registry,
    transcription_registry,
    tts_registry,
    video_gen_registry,
    web_search_registry,
)
from agent.browser_provider import BrowserProvider
from agent.image_gen_provider import ImageGenProvider
from agent.plugin_profile_scope import (
    ProfileKey,
    bind_profile_key,
    bound_to_profile,
    current_profile_key,
    normalize_profile_key,
    provider_registration_transaction,
)
from agent.transcription_provider import TranscriptionProvider
from agent.tts_provider import TTSProvider
from agent.video_gen_provider import VideoGenProvider
from agent.web_search_provider import WebSearchProvider
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


class _ImageProvider(ImageGenProvider):
    def __init__(self, name: str, marker: str, available: bool = True):
        self._name = name
        self.marker = marker
        self.available = available

    @property
    def name(self):
        return self._name

    def is_available(self):
        return self.available

    def generate(self, prompt, aspect_ratio="landscape", **kwargs):
        return {"success": True}


class _VideoProvider(VideoGenProvider):
    def __init__(self, name: str, marker: str, available: bool = True):
        self._name = name
        self.marker = marker
        self.available = available

    @property
    def name(self):
        return self._name

    def is_available(self):
        return self.available

    def generate(self, prompt, **kwargs):
        return {"success": True}


class _WebProvider(WebSearchProvider):
    def __init__(self, name: str, marker: str, available: bool = True):
        self._name = name
        self.marker = marker
        self.available = available

    @property
    def name(self):
        return self._name

    def is_available(self):
        return self.available

    def search(self, query, limit=5):
        return {"success": True, "data": {"web": []}}

    def supports_extract(self):
        return True

    def extract(self, urls, **kwargs):
        return []


class _BrowserProvider(BrowserProvider):
    def __init__(self, name: str, marker: str, available: bool = True):
        self._name = name
        self.marker = marker
        self.available = available

    @property
    def name(self):
        return self._name

    def is_available(self):
        return self.available

    def create_session(self, task_id):
        return {}

    def close_session(self, session_id):
        return True

    def emergency_cleanup(self, session_id):
        return None


class _TTSProvider(TTSProvider):
    def __init__(self, name: str, marker: str, available: bool = True):
        self._name = name
        self.marker = marker
        self.available = available

    @property
    def name(self):
        return self._name

    def is_available(self):
        return self.available

    def synthesize(self, text, output_path, **kwargs):
        return output_path


class _TranscriptionProvider(TranscriptionProvider):
    def __init__(self, name: str, marker: str, available: bool = True):
        self._name = name
        self.marker = marker
        self.available = available

    @property
    def name(self):
        return self._name

    def is_available(self):
        return self.available

    def transcribe(self, file_path, **kwargs):
        return {"success": True, "transcript": "", "provider": self.name}


REGISTRIES = (
    (image_gen_registry, _ImageProvider),
    (video_gen_registry, _VideoProvider),
    (web_search_registry, _WebProvider),
    (browser_registry, _BrowserProvider),
    (tts_registry, _TTSProvider),
    (transcription_registry, _TranscriptionProvider),
)


@pytest.fixture(autouse=True)
def _reset_registries():
    for module, _factory in REGISTRIES:
        module._reset_for_tests()
    yield
    for module, _factory in REGISTRIES:
        module._reset_for_tests()


def test_profile_key_is_normalized_and_immutable():
    key = normalize_profile_key("  ALPHA  ")
    assert key == ProfileKey("alpha")
    assert normalize_profile_key(key) is key
    with pytest.raises(FrozenInstanceError):
        key.value = "beta"  # type: ignore[misc]


def test_custom_profile_paths_preserve_case_sensitive_identity(tmp_path):
    upper_path = tmp_path / "ProfileA"
    lower_path = tmp_path / "profilea"
    upper = normalize_profile_key(upper_path)
    lower = normalize_profile_key(lower_path)
    paths_are_distinct = os.path.normcase(str(upper_path)) != os.path.normcase(
        str(lower_path)
    )
    assert (upper != lower) is paths_are_distinct


def test_context_binding_resets_and_delayed_callback_freezes_identity():
    original = current_profile_key()
    with bind_profile_key("Alpha"):
        assert current_profile_key() == ProfileKey("alpha")
        callback = bound_to_profile(lambda: current_profile_key())
        with bind_profile_key("beta"):
            assert callback() == ProfileKey("alpha")
            assert current_profile_key() == ProfileKey("beta")
    assert current_profile_key() == original


@pytest.mark.parametrize("registry,factory", REGISTRIES)
def test_registration_lookup_and_last_writer_are_profile_local(registry, factory):
    alpha_first = factory("shared", "alpha-first")
    alpha_last = factory("shared", "alpha-last")
    beta = factory("shared", "beta")

    registry.register_provider(alpha_first, profile_key="alpha")
    registry.register_provider(beta, profile_key="beta")
    registry.register_provider(alpha_last, profile_key="alpha")

    assert registry.get_provider("shared", profile_key="alpha") is alpha_last
    assert registry.get_provider("shared", profile_key="beta") is beta
    assert registry.get_provider("shared", profile_key="gamma") is None
    assert registry.list_providers(profile_key="gamma") == []


def test_registration_transaction_rolls_back_all_touched_registries():
    baseline = _ImageProvider("stable", "before")
    image_gen_registry.register_provider(baseline, profile_key="alpha")

    with pytest.raises(RuntimeError, match="plugin load failed"):
        with provider_registration_transaction("alpha"):
            image_gen_registry.register_provider(
                _ImageProvider("stable", "overwritten"), profile_key="alpha"
            )
            video_gen_registry.register_provider(
                _VideoProvider("new", "temporary"), profile_key="alpha"
            )
            raise RuntimeError("plugin load failed")

    assert image_gen_registry.get_provider("stable", profile_key="alpha") is baseline
    assert video_gen_registry.get_provider("new", profile_key="alpha") is None


def test_transaction_rejects_cross_profile_mutation_and_rolls_back():
    with pytest.raises(RuntimeError, match="different profile"):
        with provider_registration_transaction("alpha"):
            image_gen_registry.register_provider(
                _ImageProvider("alpha-only", "temporary"), profile_key="alpha"
            )
            image_gen_registry.register_provider(
                _ImageProvider("beta-only", "forbidden"), profile_key="beta"
            )

    assert image_gen_registry.list_providers(profile_key="alpha") == []
    assert image_gen_registry.list_providers(profile_key="beta") == []


def test_caught_nested_transaction_failure_rolls_back_outer_transaction():
    baseline = _ImageProvider("shared", "baseline")
    image_gen_registry.register_provider(baseline, profile_key="alpha")

    with pytest.raises(RuntimeError, match="failed closed"):
        with provider_registration_transaction("alpha"):
            try:
                with provider_registration_transaction("alpha"):
                    image_gen_registry.register_provider(
                        _ImageProvider("shared", "temporary"), profile_key="alpha"
                    )
                    raise ValueError("nested plugin load failed")
            except ValueError:
                pass

    assert image_gen_registry.get_provider("shared", profile_key="alpha") is baseline


def test_transaction_rollback_does_not_clobber_concurrent_last_writer():
    baseline = _ImageProvider("shared", "baseline")
    temporary = _ImageProvider("shared", "temporary")
    concurrent = _ImageProvider("shared", "concurrent")
    image_gen_registry.register_provider(baseline, profile_key="alpha")

    with pytest.raises(RuntimeError, match="plugin load failed"):
        with provider_registration_transaction("alpha"):
            image_gen_registry.register_provider(temporary, profile_key="alpha")
            worker = threading.Thread(
                target=lambda: image_gen_registry.register_provider(
                    concurrent, profile_key="alpha"
                )
            )
            worker.start()
            worker.join(timeout=5)
            assert not worker.is_alive()
            raise RuntimeError("plugin load failed")

    assert image_gen_registry.get_provider("shared", profile_key="alpha") is concurrent


def test_transaction_rollback_preserves_concurrent_same_object_writer():
    baseline = _ImageProvider("shared", "baseline")
    shared = _ImageProvider("shared", "shared")
    image_gen_registry.register_provider(baseline, profile_key="alpha")

    with pytest.raises(RuntimeError, match="plugin load failed"):
        with provider_registration_transaction("alpha"):
            image_gen_registry.register_provider(shared, profile_key="alpha")
            worker = threading.Thread(
                target=lambda: image_gen_registry.register_provider(
                    shared, profile_key="alpha"
                )
            )
            worker.start()
            worker.join(timeout=5)
            assert not worker.is_alive()
            raise RuntimeError("plugin load failed")

    assert image_gen_registry.get_provider("shared", profile_key="alpha") is shared


def test_active_availability_does_not_leak_between_profiles(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    alpha = _ImageProvider("only", "alpha", available=False)
    beta = _ImageProvider("only", "beta", available=True)
    image_gen_registry.register_provider(alpha, profile_key="alpha")
    image_gen_registry.register_provider(beta, profile_key="beta")

    assert image_gen_registry.get_active_provider(profile_key="alpha") is None
    assert image_gen_registry.get_active_provider(profile_key="beta") is beta


def test_active_availability_callbacks_run_under_selected_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    image = _ImageProvider("only", "image")
    video = _VideoProvider("only", "video")
    web = _WebProvider("ddgs", "web")
    browser = _BrowserProvider("browser-use", "browser")
    providers = (image, video, web, browser)
    for provider in providers:
        provider.is_available = lambda: current_profile_key() == ProfileKey("alpha")

    image_gen_registry.register_provider(image, profile_key="alpha")
    video_gen_registry.register_provider(video, profile_key="alpha")
    web_search_registry.register_provider(web, profile_key="alpha")
    browser_registry.register_provider(browser, profile_key="alpha")

    with bind_profile_key("beta"):
        assert image_gen_registry.get_active_provider(profile_key="alpha") is image
        assert video_gen_registry.get_active_provider(profile_key="alpha") is video
        assert web_search_registry.get_active_search_provider(profile_key="alpha") is web
        assert browser_registry._resolve(None, profile_key="alpha") is browser
        assert current_profile_key() == ProfileKey("beta")


@pytest.mark.parametrize(
    "registry,factory,config_section,config_key,resolve",
    (
        (
            image_gen_registry,
            _ImageProvider,
            "image_gen",
            "provider",
            lambda: image_gen_registry.get_active_provider(profile_key="alpha"),
        ),
        (
            video_gen_registry,
            _VideoProvider,
            "video_gen",
            "provider",
            lambda: video_gen_registry.get_active_provider(profile_key="alpha"),
        ),
        (
            web_search_registry,
            _WebProvider,
            "web",
            "search_backend",
            lambda: web_search_registry.get_active_search_provider(
                profile_key="alpha"
            ),
        ),
        (
            web_search_registry,
            _WebProvider,
            "web",
            "extract_backend",
            lambda: web_search_registry.get_active_extract_provider(
                profile_key="alpha"
            ),
        ),
    ),
)
def test_active_config_reads_run_under_selected_profile(
    monkeypatch, registry, factory, config_section, config_key, resolve
):
    from hermes_cli import config as hermes_config

    alpha_choice = factory("alpha-choice", "alpha")
    beta_choice = factory("beta-choice", "beta")
    registry.register_provider(alpha_choice, profile_key="alpha")
    registry.register_provider(beta_choice, profile_key="alpha")

    def _profile_config():
        key = current_profile_key().value
        return {config_section: {config_key: f"{key}-choice"}}

    monkeypatch.setattr(hermes_config, "load_config_readonly", _profile_config)

    with bind_profile_key("beta"):
        assert resolve() is alpha_choice
        assert current_profile_key() == ProfileKey("beta")


def test_explicit_profile_resolution_reads_real_selected_profile_config(
    tmp_path, monkeypatch
):
    hermes_root = tmp_path / "hermes"
    alpha_home = hermes_root / "profiles" / "alpha"
    beta_home = hermes_root / "profiles" / "beta"
    alpha_home.mkdir(parents=True)
    beta_home.mkdir(parents=True)
    alpha_config = """\
image_gen:
  provider: alpha-choice
video_gen:
  provider: alpha-choice
web:
  search_backend: alpha-choice
  extract_backend: alpha-choice
"""
    beta_config = alpha_config.replace("alpha-choice", "beta-choice")
    (alpha_home / "config.yaml").write_text(alpha_config, encoding="utf-8")
    (beta_home / "config.yaml").write_text(beta_config, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_root))

    alpha_image = _ImageProvider("alpha-choice", "alpha")
    beta_image = _ImageProvider("beta-choice", "beta")
    alpha_video = _VideoProvider("alpha-choice", "alpha")
    beta_video = _VideoProvider("beta-choice", "beta")
    alpha_web = _WebProvider("alpha-choice", "alpha")
    beta_web = _WebProvider("beta-choice", "beta")
    for provider in (alpha_image, beta_image):
        image_gen_registry.register_provider(provider, profile_key="alpha")
    for provider in (alpha_video, beta_video):
        video_gen_registry.register_provider(provider, profile_key="alpha")
    for provider in (alpha_web, beta_web):
        web_search_registry.register_provider(provider, profile_key="alpha")

    home_token = set_hermes_home_override(beta_home)
    try:
        with bind_profile_key("beta"):
            resolved = (
                image_gen_registry.get_active_provider(profile_key="alpha"),
                video_gen_registry.get_active_provider(profile_key="alpha"),
                web_search_registry.get_active_search_provider(profile_key="alpha"),
                web_search_registry.get_active_extract_provider(profile_key="alpha"),
            )
            assert resolved == (
                alpha_image,
                alpha_video,
                alpha_web,
                alpha_web,
            ), "explicit alpha resolution read beta config"
            assert current_profile_key() == ProfileKey("beta")
    finally:
        reset_hermes_home_override(home_token)


def test_reserved_builtin_rejection_is_profile_local(caplog):
    tts_registry.register_provider(_TTSProvider("edge", "alpha"), profile_key="alpha")
    tts_registry.register_provider(_TTSProvider("custom", "beta"), profile_key="beta")

    assert tts_registry.get_provider("edge", profile_key="alpha") is None
    assert tts_registry.list_providers(profile_key="alpha") == []
    assert [p.name for p in tts_registry.list_providers(profile_key="beta")] == ["custom"]
