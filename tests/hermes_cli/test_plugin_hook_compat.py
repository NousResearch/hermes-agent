"""Compatibility tests for legacy plugin hook signatures."""

import logging

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def test_legacy_transform_terminal_output_hook_warns_once_and_runs(caplog):
    calls = []

    def legacy_transform(output, context):
        calls.append((output, dict(context)))
        return f"{context['command']}|{context['returncode']}|{output}"

    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="legacy-plugin", source="user"), manager)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        ctx.register_hook("transform_terminal_output", legacy_transform)
        first = manager.invoke_hook(
            "transform_terminal_output",
            command="echo hello",
            output="abcdef",
            returncode=7,
            task_id="task-1",
            env_type="local",
        )
        second = manager.invoke_hook(
            "transform_terminal_output",
            command="printf ok",
            output="xyz",
            returncode=0,
            task_id="task-2",
            env_type="local",
        )

    assert first == ["echo hello|7|abcdef"]
    assert second == ["printf ok|0|xyz"]
    assert calls[0][1]["task_id"] == "task-1"
    assert calls[1][1]["env_type"] == "local"

    warnings = [
        record.message
        for record in caplog.records
        if "legacy (output, context) signature" in record.message
    ]
    assert len(warnings) == 1
