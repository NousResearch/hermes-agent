"""``ctx.models``: the public model catalog plugins use instead of private ``hermes_cli.models`` helpers."""

from __future__ import annotations

import pytest

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from hermes_constants import get_hermes_home


def _ctx(home) -> PluginContext:
    return PluginContext(PluginManifest(name="picker", version="1.0.0"), PluginManager(scope_key=str(home)))


def test_ready_answers_in_the_plugin_profile_not_the_launch_profile(tmp_path, monkeypatch):
    """A secondary profile's plugin must not see the launch profile's key, even though that key is in
    the process env and an unscoped credential read would find it."""
    launch = get_hermes_home()
    (launch / ".env").write_text("DEEPSEEK_API_KEY=sk-launch-0123456789\n", encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-launch-0123456789")
    secondary = tmp_path / "profiles" / "b"
    secondary.mkdir(parents=True)
    (secondary / ".env").write_text("", encoding="utf-8")

    assert _ctx(launch).models.ready("deepseek") is True
    assert _ctx(secondary).models.ready("deepseek") is False


def test_parse_canonicalizes_and_refuses_bare_or_unknown_specs():
    models = _ctx(get_hermes_home()).models

    assert models.parse("deep-seek:deepseek-flash") == ("deepseek", "deepseek-flash")
    assert "deepseek" in models.providers()
    for spec in ("deepseek-flash", "nosuchprovider:model-x", "deepseek:"):
        with pytest.raises(ValueError):
            models.parse(spec)
