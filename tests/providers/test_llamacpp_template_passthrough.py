"""llamacpp profile: per-model chat_template_kwargs passthrough.

Hermes config can declare arbitrary chat_template_kwargs per model
via the per-model metadata dict inside a provider entry's ``models``
mapping::

    providers:
      llamacpp:
        api: http://rig:8080/v1
        models:
          model-a:
            chat_template_kwargs: {custom_flag: true}
          model-b: {}

The profile reads the entry matching its base_url and merges the model's
kwargs BENEATH the reasoning-key mapping: reasoning_effort and
enable_thinking stay governed by the reasoning mapping, so a reasoning key
smuggled into the passthrough never reaches the wire. Precedence chain,
lowest first: passthrough < reasoning keys < entry extra_body (the
transport's existing last-write escape hatch, which replaces top-level
extra_body keys wholesale - pinned here as documented behavior).
"""

from __future__ import annotations

import shutil

import pytest
import yaml

from tests.providers.test_llamacpp_profile import (
    _fresh_hermes_home,
    _installed_plugin_dir,
)
from tests.providers.test_llamacpp_reasoning_kwargs import (
    _fake_probe,
    _probe_pkg,
    _qwen38_caps,
)
from tests.providers.test_plugin_discovery import _clear_provider_caches

pytestmark = pytest.mark.skipif(
    _installed_plugin_dir() is None,
    reason="real llamacpp user plugin not installed on this machine",
)

BASE_URL = "http://rig:8080/v1"


@pytest.fixture()
def rig(tmp_path, monkeypatch):
    """Fresh HERMES_HOME with the real plugin; yields (hermes_home, profile)."""
    hermes_home = _fresh_hermes_home(tmp_path, monkeypatch)
    plugin_dir = hermes_home / "plugins" / "model-providers" / "llamacpp"
    plugin_dir.parent.mkdir(parents=True)
    shutil.copytree(
        _installed_plugin_dir(),
        plugin_dir,
        ignore=shutil.ignore_patterns(".git", "__pycache__"),
    )
    _clear_provider_caches()
    from providers import get_provider_profile

    profile = get_provider_profile("llamacpp")
    assert profile is not None and profile.name == "llamacpp"
    yield hermes_home, profile
    _clear_provider_caches()


def _write_models_config(hermes_home, models, base_url=BASE_URL):
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(
            {"providers": {"llamacpp": {"api": base_url, "models": models}}}
        )
    )


def _kwargs(profile, model, reasoning_config=None):
    extra, top = profile.build_api_kwargs_extras(
        reasoning_config=reasoning_config,
        model=model,
        base_url=BASE_URL,
    )
    assert top == {}
    return extra.get("chat_template_kwargs")


# ── config read: which model gets which kwargs ──────────────────────────


def test_passthrough_on_configured_model_only(rig):
    hermes_home, profile = rig
    _write_models_config(
        hermes_home,
        {
            "model-a": {
                "chat_template_kwargs": {"custom_flag": True, "pad_tokens": 7}
            },
            "model-b": {},
        },
    )
    assert _kwargs(profile, "model-a") == {"custom_flag": True, "pad_tokens": 7}
    assert _kwargs(profile, "model-b") is None
    assert _kwargs(profile, "model-c") is None  # not in the entry catalog


def test_other_base_url_entries_do_not_leak(rig):
    hermes_home, profile = rig
    _write_models_config(
        hermes_home,
        {"model-a": {"chat_template_kwargs": {"x": 1}}},
        base_url="http://other:9090/v1",
    )
    assert _kwargs(profile, "model-a") is None


def test_legacy_list_form_models(rig):
    """custom_providers list entries with models: [{id: ..., ...}] rows."""
    hermes_home, profile = rig
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "custom_providers": [
                    {
                        "name": "llamacpp",
                        "base_url": BASE_URL,
                        "models": [
                            {
                                "id": "model-a",
                                "chat_template_kwargs": {"custom_flag": True},
                            },
                            "model-b",
                        ],
                    }
                ]
            }
        )
    )
    assert _kwargs(profile, "model-a") == {"custom_flag": True}
    assert _kwargs(profile, "model-b") is None


# ── merge precedence: passthrough beneath the reasoning-key mapping ─────


def test_passthrough_merges_beneath_reasoning_effort(rig, monkeypatch):
    hermes_home, profile = rig
    _write_models_config(
        hermes_home, {"model-a": {"chat_template_kwargs": {"custom_flag": True}}}
    )
    _fake_probe(monkeypatch, profile, None)  # unknown caps -> verbatim effort
    got = _kwargs(profile, "model-a", {"enabled": True, "effort": "medium"})
    assert got == {"custom_flag": True, "reasoning_effort": "medium"}


def test_reasoning_key_in_passthrough_cannot_override_mapping(rig, monkeypatch):
    hermes_home, profile = rig
    _write_models_config(
        hermes_home,
        {
            "model-a": {
                "chat_template_kwargs": {
                    "reasoning_effort": "ultra",
                    "enable_thinking": True,
                    "custom_flag": True,
                }
            }
        },
    )
    _fake_probe(monkeypatch, profile, _qwen38_caps(_probe_pkg(profile)))
    got = _kwargs(profile, "model-a", {"enabled": True, "effort": "medium"})
    assert got == {"custom_flag": True, "reasoning_effort": "medium"}


def test_reasoning_key_in_passthrough_dropped_without_reasoning_config(rig):
    hermes_home, profile = rig
    _write_models_config(
        hermes_home,
        {
            "model-a": {
                "chat_template_kwargs": {
                    "reasoning_effort": "ultra",
                    "enable_thinking": True,
                    "custom_flag": True,
                }
            }
        },
    )
    assert _kwargs(profile, "model-a") == {"custom_flag": True}


def test_thinking_off_wins_over_passthrough_toggle(rig, monkeypatch):
    hermes_home, profile = rig
    _write_models_config(
        hermes_home,
        {
            "model-a": {
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "custom_flag": True,
                }
            }
        },
    )
    _fake_probe(monkeypatch, profile, _qwen38_caps(_probe_pkg(profile)))
    got = _kwargs(profile, "model-a", {"enabled": False})
    assert got == {"custom_flag": True, "enable_thinking": False}
    assert got["enable_thinking"] is False  # JSON boolean, never a string


def test_per_model_reasoning_overrides_unchanged(rig, monkeypatch):
    """Override behavior is identical with the passthrough present."""
    from hermes_constants import resolve_reasoning_config

    hermes_home, profile = rig
    _write_models_config(
        hermes_home,
        {
            "model-a": {"chat_template_kwargs": {"custom_flag": True}},
            "model-b": {},
        },
    )
    _fake_probe(monkeypatch, profile, None)
    cfg = {
        "agent": {
            "reasoning_effort": "medium",
            "reasoning_overrides": {"model-a": "low", "model-b": "xhigh"},
        }
    }
    got_a = _kwargs(profile, "model-a", resolve_reasoning_config(cfg, "model-a"))
    got_b = _kwargs(profile, "model-b", resolve_reasoning_config(cfg, "model-b"))
    assert got_a == {"custom_flag": True, "reasoning_effort": "low"}
    assert got_b == {"reasoning_effort": "xhigh"}


# ── transport level: entry extra_body keeps the last word ───────────────


def _transport_kwargs(profile, model, **params):
    from agent.transports.chat_completions import ChatCompletionsTransport

    return ChatCompletionsTransport().build_kwargs(
        model=model,
        messages=[{"role": "user", "content": "hello"}],
        tools=None,
        provider_profile=profile,
        base_url=BASE_URL,
        **params,
    )


def test_transport_merges_passthrough_and_effort(rig, monkeypatch):
    hermes_home, profile = rig
    _write_models_config(
        hermes_home, {"model-a": {"chat_template_kwargs": {"custom_flag": True}}}
    )
    _fake_probe(monkeypatch, profile, None)
    kw = _transport_kwargs(
        profile, "model-a", reasoning_config={"enabled": True, "effort": "medium"}
    )
    assert kw["extra_body"]["chat_template_kwargs"] == {
        "custom_flag": True,
        "reasoning_effort": "medium",
    }


def test_entry_extra_body_stays_the_last_word(rig, monkeypatch):
    """Entry extra_body rides request_overrides and replaces top-level
    extra_body keys wholesale - the existing, deliberate escape hatch.
    Pinned, not changed: per-model kwargs belong in the models mapping."""
    hermes_home, profile = rig
    _write_models_config(
        hermes_home, {"model-a": {"chat_template_kwargs": {"custom_flag": True}}}
    )
    _fake_probe(monkeypatch, profile, None)
    kw = _transport_kwargs(
        profile,
        "model-a",
        reasoning_config={"enabled": True, "effort": "medium"},
        request_overrides={"extra_body": {"chat_template_kwargs": {"forced": 1}}},
    )
    assert kw["extra_body"]["chat_template_kwargs"] == {"forced": 1}


def test_entry_extra_body_other_keys_coexist(rig, monkeypatch):
    hermes_home, profile = rig
    _write_models_config(
        hermes_home, {"model-a": {"chat_template_kwargs": {"custom_flag": True}}}
    )
    _fake_probe(monkeypatch, profile, None)
    kw = _transport_kwargs(
        profile,
        "model-a",
        reasoning_config={"enabled": True, "effort": "medium"},
        request_overrides={"extra_body": {"service_note": "x"}},
    )
    assert kw["extra_body"]["service_note"] == "x"
    assert kw["extra_body"]["chat_template_kwargs"] == {
        "custom_flag": True,
        "reasoning_effort": "medium",
    }
