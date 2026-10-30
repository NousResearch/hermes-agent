"""``ctx.agent_loop`` is announced only in the process that owns the agent turn."""

from __future__ import annotations

from typing import Any, cast

from hermes_cli.plugin_host_child import RemotePluginContext
from hermes_cli.plugins import PluginContext
from hermes_cli.plugins_manifest import PluginManifest


def test_in_process_context_exposes_the_pool():
    ctx = PluginContext(PluginManifest(name="demo"), cast(Any, object()))

    loop = ctx.agent_loop
    assert loop.plugin_id == "demo"
    assert loop.capability_version == "agent_loop_v1"


def test_a_hosted_plugin_sees_no_agent_loop():
    remote = RemotePluginContext(cast(Any, None), "demo",
                                 {"ctx_methods": ["agent_loop", "register_tool"]})

    assert getattr(remote, "agent_loop", None) is None, (
        "a callable placeholder would lie to getattr() feature detection")


def test_the_audit_flags_direct_agent_loop_use(tmp_path):
    from hermes_cli.plugin_isolation_audit import audit_plugin_dir

    plugin = tmp_path / "direct"
    plugin.mkdir()
    (plugin / "__init__.py").write_text(
        "def register(ctx):\n    ctx.agent_loop.run('hi')\n", encoding="utf-8")

    report = audit_plugin_dir(plugin, {"name": "direct"})

    assert report.host_ready is False
    assert any("agent_loop" in reason for reason in report.reasons)


def test_the_audit_accepts_a_feature_detected_agent_loop(tmp_path):
    from hermes_cli.plugin_isolation_audit import audit_plugin_dir

    plugin = tmp_path / "guarded"
    plugin.mkdir()
    (plugin / "__init__.py").write_text(
        "def register(ctx):\n"
        "    loop = getattr(ctx, 'agent_loop', None)\n"
        "    if loop:\n"
        "        loop.run('hi')\n", encoding="utf-8")

    assert audit_plugin_dir(plugin, {"name": "guarded"}).host_ready is True
