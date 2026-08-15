"""Tests for the llamacpp opt-in prompt warmer.

``prompt_warmer_enabled`` gates the warm request on config: default off,
enabled only by boolean True (per-model metadata wins over the session-wide
``agent.prompt_warmer``). ``warm_prompt_cache`` sends one minimal
completion carrying the session preamble plus the same profile extras a
real request would, and never raises.
"""

from __future__ import annotations

import io
import json
import shutil
import urllib.request

import pytest

from tests.providers.test_llamacpp_profile import (
    _fresh_hermes_home,
    _installed_plugin_dir,
)
from tests.providers.test_llamacpp_reasoning_budget import BASE_URL, _write_config
from tests.providers.test_plugin_discovery import _clear_provider_caches

pytestmark = pytest.mark.skipif(
    _installed_plugin_dir() is None,
    reason="real llamacpp user plugin not installed on this machine",
)


@pytest.fixture()
def rig(tmp_path, monkeypatch):
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


class _FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_warmer_off_by_default(rig):
    hermes_home, profile = rig
    _write_config(hermes_home, models={"m1": {}})

    assert profile.prompt_warmer_enabled(base_url=BASE_URL, model="m1") is False


def test_warmer_on_via_agent_config(rig):
    hermes_home, profile = rig
    _write_config(hermes_home, agent={"prompt_warmer": True}, models={"m1": {}})

    assert profile.prompt_warmer_enabled(base_url=BASE_URL, model="m1") is True


def test_per_model_flag_wins_and_is_strict(rig):
    hermes_home, profile = rig
    _write_config(
        hermes_home,
        agent={"prompt_warmer": True},
        models={"m1": {"prompt_warmer": False}, "m2": {"prompt_warmer": "yes"}},
    )

    # Per-model False beats the session-wide True; a non-boolean stays off.
    assert profile.prompt_warmer_enabled(base_url=BASE_URL, model="m1") is False
    assert profile.prompt_warmer_enabled(base_url=BASE_URL, model="m2") is False


def test_warm_request_carries_preamble_and_profile_extras(rig, monkeypatch):
    hermes_home, profile = rig
    _write_config(
        hermes_home,
        agent={"prompt_warmer": True},
        models={"m1": {"chat_template_kwargs": {"my_var": True}}},
    )

    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data)
        seen["timeout"] = timeout
        return _FakeResponse(
            {"timings": {"prompt_n": 512, "predicted_n": 1}}
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    tools = [
        {
            "type": "function",
            "function": {"name": "read_file", "parameters": {"type": "object"}},
        }
    ]
    result = profile.warm_prompt_cache(
        base_url=BASE_URL,
        model="m1",
        system_prompt="SESSION PREAMBLE",
        tools=tools,
    )

    assert result == 512
    assert seen["url"] == BASE_URL.rstrip("/") + "/chat/completions"
    body = seen["body"]
    assert body["model"] == "m1"
    assert body["messages"] == [{"role": "system", "content": "SESSION PREAMBLE"}]
    assert body["max_tokens"] == 1
    assert body["stream"] is False
    # Tool definitions render into the early prompt region server-side, so
    # the warm must carry the same list the first turn will send.
    assert body["tools"] == tools
    # The same profile extras a real request would carry.
    assert body["chat_template_kwargs"] == {"my_var": True}


def test_no_request_without_preamble(rig, monkeypatch):
    hermes_home, profile = rig
    _write_config(hermes_home, agent={"prompt_warmer": True}, models={"m1": {}})

    def fail_urlopen(req, timeout=None):
        raise AssertionError("warm request must not be sent")

    monkeypatch.setattr(urllib.request, "urlopen", fail_urlopen)

    assert profile.warm_prompt_cache(base_url=BASE_URL, model="m1") is None
    assert (
        profile.warm_prompt_cache(base_url=BASE_URL, model="", system_prompt="x")
        is None
    )


def test_warm_failure_returns_none_never_raises(rig, monkeypatch):
    hermes_home, profile = rig
    _write_config(hermes_home, agent={"prompt_warmer": True}, models={"m1": {}})

    def boom(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", boom)

    assert (
        profile.warm_prompt_cache(
            base_url=BASE_URL, model="m1", system_prompt="PREAMBLE"
        )
        is None
    )
