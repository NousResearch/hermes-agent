from __future__ import annotations

import argparse
from pathlib import Path

from hermes_cli.subcommands.plugins import build_plugins_parser


def _parse_plugins_args(*argv: str):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_plugins_parser(subparsers, cmd_plugins=lambda args: None)
    return parser.parse_args(["plugins", *argv])


def test_plugins_parser_exposes_doctor() -> None:
    doctor = _parse_plugins_args("doctor", "sample", "--ci")

    assert (doctor.plugins_action, doctor.target, doctor.ci) == (
        "doctor",
        "sample",
        True,
    )


def test_doctor_uses_registration_to_reject_bad_hook_and_callback_signature(
    tmp_path: Path,
) -> None:
    from hermes_cli.plugin_dev import doctor_plugin

    plugin = tmp_path / "bad-plugin"
    plugin.mkdir()
    (plugin / "plugin.yaml").write_text(
        "\n".join(
            [
                "name: bad-plugin",
                "version: 0.1.0",
                "description: broken contract",
                "provides_hooks:",
                "  - typo_hook",
                "  - pre_tool_call",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (plugin / "__init__.py").write_text(
        "def callback(tool_name):\n"
        "    return None\n\n"
        "def register(ctx):\n"
        "    ctx.register_hook('typo_hook', callback)\n"
        "    ctx.register_hook('pre_tool_call', callback)\n",
        encoding="utf-8",
    )

    report = doctor_plugin(plugin)
    messages = "\n".join(f.message for f in report.findings)
    assert report.ok is False
    assert "unknown hook 'typo_hook'" in messages
    assert "must accept **kwargs" in messages


def test_doctor_accepts_manifest_defaults_from_runtime_parser(tmp_path: Path) -> None:
    from hermes_cli.plugin_dev import doctor_plugin

    plugin = tmp_path / "minimal"
    plugin.mkdir()
    (plugin / "plugin.yaml").write_text("name: minimal\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(
        "def register(ctx):\n    pass\n", encoding="utf-8"
    )

    report = doctor_plugin(plugin)
    assert report.ok, report.format_text()
    assert report.manifest is not None
    assert report.manifest.kind == "standalone"


def test_doctor_restores_global_tool_policy_and_module_state(tmp_path: Path) -> None:
    import sys

    from hermes_cli.plugin_dev import doctor_plugin
    from tools.registry import registry

    target = tmp_path / "cleanup-plugin"
    target.mkdir()
    (target / "plugin.yaml").write_text(
        "name: cleanup-plugin\nprovides_tools: [cleanup_plugin_ping]\n",
        encoding="utf-8",
    )
    (target / "__init__.py").write_text(
        "import json\n\n"
        "def ping(args, **kwargs):\n    return json.dumps({'ok': True})\n\n"
        "def register(ctx):\n"
        "    ctx.register_tool(name='cleanup_plugin_ping', toolset='cleanup', "
        "schema={'name': 'cleanup_plugin_ping', 'description': 'test', "
        "'parameters': {'type': 'object'}}, handler=ping)\n",
        encoding="utf-8",
    )
    before_policy = dict(registry._plugin_override_policy)
    before_modules = {
        name
        for name in sys.modules
        if name == "hermes_plugins" or name.startswith("hermes_plugins.")
    }

    report = doctor_plugin(target)

    assert report.ok, report.format_text()
    assert report.registered_tools == ("cleanup_plugin_ping",)
    assert registry.get_entry("cleanup_plugin_ping") is None
    assert registry._plugin_override_policy == before_policy
    after_modules = {
        name
        for name in sys.modules
        if name == "hermes_plugins" or name.startswith("hermes_plugins.")
    }
    assert after_modules == before_modules


def test_doctor_blocks_live_network(tmp_path: Path) -> None:
    from hermes_cli.plugin_dev import doctor_plugin

    plugin = tmp_path / "network-plugin"
    plugin.mkdir()
    (plugin / "plugin.yaml").write_text("name: network-plugin\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(
        "import socket\n\n"
        "def register(ctx):\n"
        "    socket.create_connection(('example.com', 443))\n",
        encoding="utf-8",
    )

    report = doctor_plugin(plugin)
    assert report.ok is False
    assert "network access is disabled while Plugin Doctor runs" in report.format_text()


_PROVIDER_PLUGIN = (
    "from providers import register_provider\n"
    "from providers.base import ProviderProfile\n\n"
    "register_provider(\n"
    "    ProviderProfile(\n"
    "        name='{name}',\n"
    "        display_name='Demo Provider',\n"
    "        env_vars=('DEMO_API_KEY',),\n"
    "        base_url='https://api.demo.example.com/v1',\n"
    "        auth_type='api_key',\n"
    "    )\n"
    ")\n"
)


def _write_provider_plugin(root: Path, name: str, *, manifest_extra: str = "", body: str | None = None) -> Path:
    plugin = root / name
    plugin.mkdir()
    (plugin / "plugin.yaml").write_text(
        f"name: {name}\nkind: model-provider\nversion: 0.1.0\n{manifest_extra}",
        encoding="utf-8",
    )
    (plugin / "__init__.py").write_text(
        _PROVIDER_PLUGIN.format(name=name) if body is None else body,
        encoding="utf-8",
    )
    return plugin


def test_doctor_accepts_model_provider_plugin(tmp_path: Path) -> None:
    """A model-provider plugin registers at import and has no register(ctx).

    Validating one through the standalone path fails it on a function the
    contract never asks for, which is what happens to every bundled provider.
    """
    from hermes_cli.plugin_dev import doctor_plugin

    plugin = _write_provider_plugin(tmp_path, "demo-provider")

    report = doctor_plugin(plugin)
    assert report.ok, report.format_text()
    assert report.manifest is not None
    assert report.manifest.kind == "model-provider"
    assert report.registered_providers == ("demo-provider",)
    assert "1 provider(s)" in report.format_text()


def test_doctor_rejects_model_provider_that_registers_nothing(tmp_path: Path) -> None:
    from hermes_cli.plugin_dev import doctor_plugin

    plugin = _write_provider_plugin(
        tmp_path, "empty-provider", body="# registers nothing\n"
    )

    report = doctor_plugin(plugin)
    assert report.ok is False
    messages = "\n".join(f.message for f in report.findings)
    assert "registered no ProviderProfile" in messages


def test_doctor_warns_when_model_provider_declares_tools(tmp_path: Path) -> None:
    from hermes_cli.plugin_dev import doctor_plugin

    plugin = _write_provider_plugin(
        tmp_path, "declares-tools", manifest_extra="provides_tools:\n  - something\n"
    )

    report = doctor_plugin(plugin)
    assert report.ok, report.format_text()
    warnings = [f.message for f in report.findings if f.level == "warning"]
    assert any("provides_tools" in message for message in warnings)


def test_doctor_restores_provider_registry(tmp_path: Path) -> None:
    import providers

    from hermes_cli.plugin_dev import doctor_plugin

    plugin = _write_provider_plugin(tmp_path, "restore-provider")

    report = doctor_plugin(plugin)
    assert report.ok, report.format_text()
    assert "restore-provider" not in providers._REGISTRY
