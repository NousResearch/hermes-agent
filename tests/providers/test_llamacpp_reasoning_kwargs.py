"""llamacpp profile: chat_template_kwargs.reasoning_effort emission.

Configured reasoning effort reaches the server as
chat_template_kwargs.reasoning_effort (llama-server's Jinja variable
channel; the OpenAI SDK merges extra_body into the JSON body top level).
Per-model agent.reasoning_overrides win over the global effort -
resolution happens in hermes_constants.resolve_reasoning_config, so these
tests chain the real resolver into the hook.
"""

from __future__ import annotations

import shutil

import pytest

from tests.providers.test_llamacpp_profile import (
    _fresh_hermes_home,
    _installed_plugin_dir,
)
from tests.providers.test_plugin_discovery import _clear_provider_caches

pytestmark = pytest.mark.skipif(
    _installed_plugin_dir() is None,
    reason="real llamacpp user plugin not installed on this machine",
)


@pytest.fixture()
def llamacpp_profile(tmp_path, monkeypatch):
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
    yield profile
    _clear_provider_caches()


def test_effort_reaches_chat_template_kwargs(llamacpp_profile):
    extra, top = llamacpp_profile.build_api_kwargs_extras(
        reasoning_config={"enabled": True, "effort": "medium"}
    )
    assert extra == {"chat_template_kwargs": {"reasoning_effort": "medium"}}
    assert top == {}


@pytest.mark.parametrize("effort", ["low", "high", "xhigh", "minimal"])
def test_hook_passes_levels_verbatim(llamacpp_profile, effort):
    extra, _ = llamacpp_profile.build_api_kwargs_extras(
        reasoning_config={"enabled": True, "effort": effort}
    )
    assert extra["chat_template_kwargs"]["reasoning_effort"] == effort


def test_no_reasoning_config_emits_nothing(llamacpp_profile):
    assert llamacpp_profile.build_api_kwargs_extras(reasoning_config=None) == (
        {},
        {},
    )


def test_empty_effort_emits_nothing(llamacpp_profile):
    extra, top = llamacpp_profile.build_api_kwargs_extras(
        reasoning_config={"enabled": True, "effort": ""}
    )
    assert extra == {} and top == {}


def test_disabled_emits_no_reasoning_effort(llamacpp_profile):
    """Thinking-off wiring is tested separately; here we only guarantee no effort leaks."""
    extra, top = llamacpp_profile.build_api_kwargs_extras(
        reasoning_config={"enabled": False}
    )
    assert "reasoning_effort" not in str(extra)
    assert top == {}


def test_per_model_overrides_feed_the_hook(llamacpp_profile):
    from hermes_constants import resolve_reasoning_config

    cfg = {
        "agent": {
            "reasoning_effort": "medium",
            "reasoning_overrides": {"model-a": "low", "model-b": "xhigh"},
        }
    }
    for model, expected in (
        ("model-a", "low"),
        ("model-b", "xhigh"),
        ("model-c", "medium"),  # no override -> global
    ):
        rc = resolve_reasoning_config(cfg, model)
        extra, _ = llamacpp_profile.build_api_kwargs_extras(reasoning_config=rc)
        assert extra["chat_template_kwargs"]["reasoning_effort"] == expected, model


def test_transport_merges_into_extra_body(llamacpp_profile):
    from agent.transports.chat_completions import ChatCompletionsTransport

    kw = ChatCompletionsTransport().build_kwargs(
        model="qwen38-27b-mtp-q8",
        messages=[{"role": "user", "content": "hello"}],
        tools=None,
        provider_profile=llamacpp_profile,
        reasoning_config={"enabled": True, "effort": "medium"},
        base_url="http://rig:8080/v1",
    )
    assert kw["extra_body"]["chat_template_kwargs"] == {
        "reasoning_effort": "medium"
    }
    assert "reasoning_effort" not in kw  # never a top-level OpenAI param


# ── template-aware clamping ───────────────────────────────────────


def _probe_pkg(profile):
    """The plugin package's probe submodule (what `from . import probe`
    resolves to inside the plugin)."""
    import importlib

    return importlib.import_module(type(profile).__module__ + ".probe")


def _fake_probe(monkeypatch, profile, caps):
    probe_pkg = _probe_pkg(profile)
    result = probe_pkg.ProbeResult(
        server=probe_pkg.ServerInfo(kind="llama-swap", running=("m",)),
        props={} if caps is not None else None,
        caps=caps,
    )
    monkeypatch.setattr(probe_pkg, "probe_model", lambda *a, **k: result)
    return probe_pkg


def _qwen38_caps(probe_pkg):
    return probe_pkg.TemplateCaps(
        has_reasoning_effort=True,
        accepted_efforts=("xhigh", "medium", "low"),
        remapped_efforts={"high": "xhigh"},
        default_effort="xhigh",
        supports_thinking_toggle=True,
        tolerated_efforts=("xhigh", "medium", "low", "high"),
    )


def _emitted_effort(profile, effort):
    extra, _ = profile.build_api_kwargs_extras(
        reasoning_config={"enabled": True, "effort": effort},
        base_url="http://rig:8080/v1",
        model="m",
    )
    kwargs = extra.get("chat_template_kwargs")
    return kwargs["reasoning_effort"] if kwargs else None


@pytest.mark.parametrize(
    "effort,expected",
    [
        ("low", "low"),  # accepted verbatim
        ("medium", "medium"),
        ("xhigh", "xhigh"),
        ("high", "xhigh"),  # template's own remap, applied client-side
        ("minimal", "low"),  # nearest accepted below
        ("max", "xhigh"),  # nearest accepted above
        ("ultra", "xhigh"),
    ],
)
def test_clamp_against_qwen38_template(
    llamacpp_profile, monkeypatch, effort, expected
):
    probe_pkg = _probe_pkg(llamacpp_profile)
    _fake_probe(monkeypatch, llamacpp_profile, _qwen38_caps(probe_pkg))
    assert _emitted_effort(llamacpp_profile, effort) == expected


def test_no_effort_template_omits(llamacpp_profile, monkeypatch):
    probe_pkg = _probe_pkg(llamacpp_profile)
    caps = probe_pkg.TemplateCaps(
        has_reasoning_effort=False,
        accepted_efforts=(),
        remapped_efforts={},
        default_effort=None,
        supports_thinking_toggle=False,
        tolerated_efforts=(),
    )
    _fake_probe(monkeypatch, llamacpp_profile, caps)
    assert _emitted_effort(llamacpp_profile, "medium") is None


def test_unknown_caps_pass_verbatim(llamacpp_profile, monkeypatch):
    """Cold/unknown model (probe returns no caps): keep the verbatim passthrough."""
    _fake_probe(monkeypatch, llamacpp_profile, None)
    assert _emitted_effort(llamacpp_profile, "medium") == "medium"
    assert _emitted_effort(llamacpp_profile, "high") == "high"


def test_no_base_url_passes_verbatim(llamacpp_profile):
    extra, _ = llamacpp_profile.build_api_kwargs_extras(
        reasoning_config={"enabled": True, "effort": "high"}
    )
    assert extra["chat_template_kwargs"]["reasoning_effort"] == "high"
