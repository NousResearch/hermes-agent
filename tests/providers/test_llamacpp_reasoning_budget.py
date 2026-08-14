"""llamacpp profile: per-request reasoning_budget_tokens emission.

A configured reasoning token budget reaches the server as the
top-level request field ``reasoning_budget_tokens`` (llama-server derives
the thinking start/end tags from the chat template server-side, see
tools/server/server-common.cpp). Config surface:

- session-wide: ``agent.reasoning_budget_tokens``
- per-model: ``reasoning_budget_tokens`` in the entry's per-model
  metadata dict (the same ``models`` mapping surface); the
  per-model value wins, and -1 passes verbatim (server semantics:
  explicitly disabled, overriding any launch-flag default).

Emission is gated on the server build: the field landed in llama.cpp
build 8287 (#20297, acb7c7906). The profile parses /props build_info via
the probe and OMITS the field when the build is older, unknown, or no
props are reachable (cold llama-swap model) - never a request an old
server could reject.
"""

from __future__ import annotations

import shutil

import pytest
import yaml

from tests.providers.test_llamacpp_profile import (
    _fresh_hermes_home,
    _installed_plugin_dir,
)
from tests.providers.test_llamacpp_reasoning_kwargs import _probe_pkg
from tests.providers.test_plugin_discovery import _clear_provider_caches

pytestmark = pytest.mark.skipif(
    _installed_plugin_dir() is None,
    reason="real llamacpp user plugin not installed on this machine",
)

BASE_URL = "http://rig:8080/v1"
# a current build whose template accepts the mid efforts (so the caps the
# probe parses from it keep the clamp emitting 'medium' verbatim)
NEW_BUILD = {
    "build_info": "b10433-9b05354ec",
    "chat_template": "{% if reasoning_effort in ('low', 'medium', 'high') %}{% endif %}",
}


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


def _write_config(hermes_home, *, agent=None, models=None):
    cfg = {"providers": {"llamacpp": {"api": BASE_URL, "models": models or {}}}}
    if agent:
        cfg["agent"] = agent
    (hermes_home / "config.yaml").write_text(yaml.safe_dump(cfg))


def _fake_probe_props(monkeypatch, profile, props):
    """probe_model returns the given /props payload (caps parsed from it)."""
    probe_pkg = _probe_pkg(profile)
    caps = None
    if isinstance(props, dict):
        caps = probe_pkg.parse_template_caps(props.get("chat_template", ""))
    result = probe_pkg.ProbeResult(
        server=probe_pkg.ServerInfo(kind="llama-swap", running=("m",)),
        props=props,
        caps=caps,
    )
    monkeypatch.setattr(probe_pkg, "probe_model", lambda *a, **k: result)


def _extras(profile, model, reasoning_config=None):
    extra, top = profile.build_api_kwargs_extras(
        reasoning_config=reasoning_config,
        model=model,
        base_url=BASE_URL,
    )
    assert top == {}
    return extra


# ── config resolution ───────────────────────────────────────────────────


def test_session_budget_emitted(rig, monkeypatch):
    hermes_home, profile = rig
    _write_config(
        hermes_home, agent={"reasoning_budget_tokens": 256}, models={"model-a": {}}
    )
    _fake_probe_props(monkeypatch, profile, NEW_BUILD)
    extra = _extras(profile, "model-a")
    assert extra == {"reasoning_budget_tokens": 256}
    # session-wide applies to any model on this endpoint, catalog or not
    assert _extras(profile, "model-c") == {"reasoning_budget_tokens": 256}


def test_per_model_budget_overrides_session(rig, monkeypatch):
    hermes_home, profile = rig
    _write_config(
        hermes_home,
        agent={"reasoning_budget_tokens": 256},
        models={
            "model-a": {"reasoning_budget_tokens": 128},
            "model-b": {},
            "model-c": {"reasoning_budget_tokens": -1},
        },
    )
    _fake_probe_props(monkeypatch, profile, NEW_BUILD)
    assert _extras(profile, "model-a") == {"reasoning_budget_tokens": 128}
    assert _extras(profile, "model-b") == {"reasoning_budget_tokens": 256}
    # -1 passes verbatim: explicitly disabled, beats the launch-flag default
    assert _extras(profile, "model-c") == {"reasoning_budget_tokens": -1}


def test_no_budget_config_emits_nothing(rig, monkeypatch):
    hermes_home, profile = rig
    _write_config(hermes_home, models={"model-a": {}})
    _fake_probe_props(monkeypatch, profile, NEW_BUILD)
    assert _extras(profile, "model-a") == {}


@pytest.mark.parametrize("bad", [True, "many", -2, 1.5])
def test_invalid_budget_values_omitted(rig, monkeypatch, bad):
    hermes_home, profile = rig
    _write_config(
        hermes_home,
        models={"model-a": {"reasoning_budget_tokens": bad}},
    )
    _fake_probe_props(monkeypatch, profile, NEW_BUILD)
    assert _extras(profile, "model-a") == {}


# ── build gating ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "build_info,expected",
    [
        ("b8286-old", False),  # one before the field landed
        ("b8287-abc", True),  # the introducing build (#20297)
        ("b10433-9b05354ec", True),  # the rig's served build
        ("garbage", False),
        ("", False),
    ],
)
def test_build_gate(rig, monkeypatch, build_info, expected):
    hermes_home, profile = rig
    _write_config(hermes_home, agent={"reasoning_budget_tokens": 256})
    _fake_probe_props(monkeypatch, profile, {"build_info": build_info})
    extra = _extras(profile, "model-a")
    if expected:
        assert extra == {"reasoning_budget_tokens": 256}
    else:
        assert "reasoning_budget_tokens" not in extra


def test_no_props_omits(rig, monkeypatch):
    """Cold llama-swap model / unreachable props: never emit unverified."""
    hermes_home, profile = rig
    _write_config(hermes_home, agent={"reasoning_budget_tokens": 256})
    _fake_probe_props(monkeypatch, profile, None)
    assert "reasoning_budget_tokens" not in _extras(profile, "model-a")


def test_no_base_url_omits(rig, monkeypatch):
    hermes_home, profile = rig
    _write_config(hermes_home, agent={"reasoning_budget_tokens": 256})
    extra, top = profile.build_api_kwargs_extras(
        reasoning_config=None, model="model-a"
    )
    assert extra == {} and top == {}


# ── coexistence with the reasoning-key mapping and the passthrough ──────


def test_budget_coexists_with_effort_and_passthrough(rig, monkeypatch):
    hermes_home, profile = rig
    _write_config(
        hermes_home,
        agent={"reasoning_budget_tokens": 256},
        models={"model-a": {"chat_template_kwargs": {"custom_flag": True}}},
    )
    _fake_probe_props(monkeypatch, profile, NEW_BUILD)
    extra = _extras(profile, "model-a", {"enabled": True, "effort": "medium"})
    assert extra == {
        "chat_template_kwargs": {
            "custom_flag": True,
            "reasoning_effort": "medium",
        },
        "reasoning_budget_tokens": 256,
    }


def test_transport_merges_budget_into_extra_body(rig, monkeypatch):
    from agent.transports.chat_completions import ChatCompletionsTransport

    hermes_home, profile = rig
    _write_config(hermes_home, agent={"reasoning_budget_tokens": 256})
    _fake_probe_props(monkeypatch, profile, NEW_BUILD)
    kw = ChatCompletionsTransport().build_kwargs(
        model="model-a",
        messages=[{"role": "user", "content": "hello"}],
        tools=None,
        provider_profile=profile,
        base_url=BASE_URL,
    )
    assert kw["extra_body"]["reasoning_budget_tokens"] == 256
    assert "reasoning_budget_tokens" not in kw  # never a top-level OpenAI param
