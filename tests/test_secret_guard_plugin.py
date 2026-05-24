from __future__ import annotations

import asyncio
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import yaml

import hermes_cli.plugins as plugins_mod


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIR = ROOT / "plugins" / "secret-guard"


def _load_plugin_module():
    spec = importlib.util.spec_from_file_location(
        "secret_guard_test_plugin", PLUGIN_DIR / "__init__.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _short_secret() -> str:
    return "gh" + "p_" + ("A" * 30)


def _fine_grained_secret() -> str:
    return "github" + "_pat_" + ("B" * 50)


def _scan_notice() -> str:
    return (
        "Security scan "
        + "— "
        + "Git"
        + "Hub PAT detected: a credential matching a known "
        + "provider pattern was found."
    )


def test_secret_patterns_detect_and_redact_without_leaking_to_logs(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    plugin = _load_plugin_module()

    text = f"alpha {_short_secret()} beta {_fine_grained_secret()} gamma {_scan_notice()}"

    assert plugin._has_secret(text)
    assert plugin._has_security_scan_notice(text)
    assert plugin._should_block_sensitive(text)

    redacted = plugin._redact(text)
    assert plugin.REDACTION in redacted
    assert plugin.SCAN_NOTICE_REDACTION in redacted
    assert _short_secret() not in redacted
    assert _fine_grained_secret() not in redacted

    plugin._log_detection("unit", tool_name="example", scan_notice=True)
    log_text = (tmp_path / "hermes_home" / "logs" / "secret_guard.log").read_text(
        encoding="utf-8"
    )
    assert _short_secret() not in log_text
    assert _fine_grained_secret() not in log_text
    entry = json.loads(log_text.splitlines()[-1])
    assert entry["where"] == "unit"
    assert entry["scan_notice"] is True


def test_pre_tool_call_blocks_sensitive_args_and_allows_normal_args(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    plugin = _load_plugin_module()

    assert plugin.block_tool_call("terminal", {"command": "echo safe"}) is None

    blocked = plugin.block_tool_call(
        "terminal", {"command": "echo " + _scan_notice()}, task_id="task-1", future="ok"
    )
    assert blocked["action"] == "block"
    assert "sensitive" not in blocked["message"].lower()
    assert "GitHub PAT-like credential" in blocked["message"]

    log_text = (tmp_path / "hermes_home" / "logs" / "secret_guard.log").read_text(
        encoding="utf-8"
    )
    entry = json.loads(log_text.splitlines()[-1])
    assert entry["where"] == "pre_tool_call"
    assert entry["tool_name"] == "terminal"
    assert entry["task_id"] == "task-1"
    assert entry["scan_notice"] is True


def test_transform_hooks_redact_with_tolerant_signatures(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    plugin = _load_plugin_module()

    tool_result = plugin.redact_tool_result(
        "demo", "result " + _short_secret(), args={"x": 1}, unexpected="field"
    )
    terminal_result = plugin.redact_terminal_output(
        result="terminal " + _fine_grained_secret(), unknown="field"
    )
    llm_result = plugin.redact_llm_output(
        "assistant " + _scan_notice(), session_id="s", model="m", platform="telegram"
    )

    assert tool_result == "result " + plugin.REDACTION
    assert terminal_result == "terminal " + plugin.REDACTION
    assert plugin.SCAN_NOTICE_REDACTION in llm_result
    assert "Security scan" not in llm_result
    assert "provider pattern" not in llm_result


def test_pre_gateway_dispatch_skips_and_schedules_warning(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    plugin = _load_plugin_module()

    sends: list[tuple[str, str, dict | None]] = []

    class Adapter:
        async def send(self, chat_id, text, metadata=None):
            sends.append((chat_id, text, metadata))

    async def _run():
        source = SimpleNamespace(
            platform="telegram",
            chat_id="chat-1",
            user_id="user-1",
            thread_id="thread-1",
            chat_type="dm",
            message_id="source-message",
        )
        event = SimpleNamespace(
            text="please inspect " + _short_secret(),
            source=source,
            message_id="event-message",
        )
        gateway = SimpleNamespace(adapters={"telegram": Adapter()})

        result = plugin.pre_gateway_dispatch(event=event, gateway=gateway, extra="ignored")
        await asyncio.sleep(0)
        return result

    result = asyncio.run(_run())

    assert result == {
        "action": "skip",
        "reason": "secret-guard: sensitive-credential-detected",
    }
    assert sends
    chat_id, warning, metadata = sends[0]
    assert chat_id == "chat-1"
    assert "revoke/rotate" in warning
    assert metadata["thread_id"] == "thread-1"
    assert metadata["direct_messages_topic_id"] == "thread-1"
    assert metadata["telegram_reply_to_message_id"] == "event-message"


def test_bundled_plugin_registers_expected_hooks(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes_home"
    bundled = tmp_path / "bundled_plugins"
    shutil.copytree(PLUGIN_DIR, bundled / "secret-guard")
    (hermes_home / "plugins").mkdir(parents=True)
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["secret-guard"]}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))

    plugins_mod._plugin_manager = plugins_mod.PluginManager()
    plugins_mod.discover_plugins()

    loaded = plugins_mod._plugin_manager._plugins["secret-guard"]
    assert loaded.enabled is True
    assert set(loaded.hooks_registered) == {
        "pre_gateway_dispatch",
        "pre_tool_call",
        "transform_tool_result",
        "transform_terminal_output",
        "transform_llm_output",
    }
