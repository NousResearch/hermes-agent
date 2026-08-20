"""Tests for the bundled observability/live_glass plugin."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "observability" / "live_glass"


def _fresh_plugin():
    mod_name = "plugins.observability.live_glass"
    sys.modules.pop(mod_name, None)
    return importlib.import_module(mod_name)


class TestManifest:
    def test_plugin_directory_exists(self):
        assert PLUGIN_DIR.is_dir()
        assert (PLUGIN_DIR / "plugin.yaml").exists()
        assert (PLUGIN_DIR / "__init__.py").exists()

    def test_manifest_fields(self):
        data = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text(encoding="utf-8"))
        assert data["name"] == "live-glass"
        assert data["version"]
        assert set(data["hooks"]) == {
            "post_tool_call",
            "pre_approval_request",
            "post_approval_response",
        }


class TestEventBus:
    def test_publish_validates_event_type_and_replays_to_subscribers(self):
        live_glass = _fresh_plugin()
        live_glass.reset_event_bus_for_tests()

        seen = []
        unsubscribe = live_glass.subscribe(seen.append)

        event = live_glass.publish("log", {"message": "hello"}, session_id="s1")
        assert event["type"] == "log"
        assert event["payload"] == {"message": "hello"}
        assert event["session_id"] == "s1"
        assert event["sequence"] == 1
        assert seen == [event]

        unsubscribe()
        live_glass.publish("log", {"message": "after-unsubscribe"})
        assert len(seen) == 1

        replayed = []
        live_glass.subscribe(replayed.append, replay=True, event_types={"log"})
        assert [item["payload"]["message"] for item in replayed] == [
            "hello",
            "after-unsubscribe",
        ]

        try:
            live_glass.publish("unknown", {})
        except ValueError as exc:
            assert "unknown live-glass event type" in str(exc)
        else:  # pragma: no cover - assertion clarity
            raise AssertionError("publish() accepted an unknown event type")

    def test_get_events_filters_by_type_and_sequence(self):
        live_glass = _fresh_plugin()
        live_glass.reset_event_bus_for_tests()

        live_glass.publish("log", {"n": 1})
        live_glass.publish("frame", {"n": 2})
        live_glass.publish("log", {"n": 3})

        assert [event["payload"]["n"] for event in live_glass.get_events(event_type="log")] == [1, 3]
        assert [event["payload"]["n"] for event in live_glass.get_events(since_sequence=1)] == [2, 3]
        assert [event["payload"]["n"] for event in live_glass.get_events(limit=1)] == [3]

    def test_extracts_computer_use_frame_from_multimodal_tool_result(self):
        live_glass = _fresh_plugin()
        live_glass.reset_event_bus_for_tests()

        seen = []
        live_glass.subscribe(seen.append)
        live_glass.on_post_tool_call(
            tool_name="computer_use",
            args={"action": "capture", "mode": "som"},
            result={
                "_multimodal": True,
                "content": [
                    {"type": "text", "text": "capture mode=som 10x20"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
                ],
                "meta": {"mode": "som", "width": 10, "height": 20},
                "text_summary": "capture mode=som 10x20",
            },
            session_id="sess",
            tool_call_id="call",
            status="ok",
        )

        assert [event["type"] for event in seen] == ["log", "frame"]
        frame = seen[1]
        assert frame["session_id"] == "sess"
        assert frame["tool_call_id"] == "call"
        assert frame["payload"] == {
            "image_url": "data:image/png;base64,abc123",
            "mime_type": "image/png",
            "mode": "som",
            "width": 10,
            "height": 20,
            "summary": "capture mode=som 10x20",
            "source": "computer_use",
        }

    def test_approval_request_hook_emits_event_without_authority(self):
        live_glass = _fresh_plugin()
        live_glass.reset_event_bus_for_tests()

        seen = []
        live_glass.subscribe(seen.append)
        live_glass.on_pre_approval_request(
            command="rm -rf /tmp/example",
            description="dangerous command",
            pattern_key="rm_rf",
            pattern_keys=["rm_rf"],
            session_key="gateway:abc",
            surface="gateway",
            turn_id="turn",
            tool_call_id="tool",
        )

        assert len(seen) == 1
        event = seen[0]
        assert event["type"] == "approval_request"
        assert event["session_id"] == "gateway:abc"
        assert event["turn_id"] == "turn"
        assert event["tool_call_id"] == "tool"
        assert event["payload"] == {
            "command": "rm -rf /tmp/example",
            "description": "dangerous command",
            "pattern_key": "rm_rf",
            "pattern_keys": ["rm_rf"],
            "surface": "gateway",
            "source": "approval_hook",
        }


class TestPluginRegistration:
    def test_register_hooks(self):
        live_glass = _fresh_plugin()
        calls = []

        class Ctx:
            def register_hook(self, name, fn):
                calls.append((name, fn.__name__))

        live_glass.register(Ctx())
        assert {("post_tool_call", "on_post_tool_call")}.issubset(set(calls))
        assert ("pre_approval_request", "on_pre_approval_request") in calls
        assert ("post_approval_response", "on_post_approval_response") in calls

    def test_plugin_manager_loads_when_enabled(self, tmp_path, monkeypatch):
        from hermes_cli import plugins as plugins_mod

        home = tmp_path / ".hermes"
        home.mkdir()
        (home / "config.yaml").write_text(
            yaml.safe_dump({"plugins": {"enabled": ["observability/live_glass"]}}),
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        manager = plugins_mod.PluginManager()
        manager.discover_and_load()
        loaded = manager._plugins["observability/live_glass"]
        assert loaded.enabled is True
        assert set(loaded.hooks_registered) == {
            "post_tool_call",
            "pre_approval_request",
            "post_approval_response",
        }
