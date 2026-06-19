"""Tests for the bundled observability/timeline plugin."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "observability" / "timeline"


def _fresh_plugin(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    sys.modules.pop("plugins.observability.timeline", None)
    return importlib.import_module("plugins.observability.timeline")


class TestManifest:
    def test_manifest_fields(self):
        data = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text())
        assert data["name"] == "timeline"
        assert data["version"]
        assert "pre_tool_call" in data["hooks"]
        assert "post_tool_call" in data["hooks"]
        assert "pre_api_request" in data["hooks"]
        assert "pre_gateway_dispatch" in data["hooks"]
        assert "post_gateway_delivery" in data["hooks"]


class TestTimelineStorage:
    def test_records_turn_llm_tool_events_in_sqlite(self, tmp_path, monkeypatch):
        mod = _fresh_plugin(monkeypatch, tmp_path)

        mod.on_pre_llm_call(
            session_id="session-1",
            turn_id="turn-1",
            platform="slack",
            chat_id="C123",
            thread_id="178.456",
            gateway_session_key="agent:main:slack:channel:C123:178.456",
            model="test-model",
            user_message="hello sk-abcdefghijklmnopqrstuvwxyz123456",
        )
        mod.on_pre_api_request(
            session_id="session-1",
            turn_id="turn-1",
            api_request_id="api-1",
            provider="test-provider",
            model="test-model",
            api_call_count=1,
        )
        mod.on_post_api_request(
            session_id="session-1",
            turn_id="turn-1",
            api_request_id="api-1",
            finish_reason="tool_calls",
            assistant_tool_call_count=1,
            usage={"input_tokens": 3, "output_tokens": 2},
        )
        mod.on_pre_tool_call(
            session_id="session-1",
            turn_id="turn-1",
            tool_call_id="tool-1",
            tool_name="read_file",
            args={"path": "/tmp/example", "api_key": "secret-value"},
        )
        mod.on_post_tool_call(
            session_id="session-1",
            turn_id="turn-1",
            tool_call_id="tool-1",
            tool_name="read_file",
            status="ok",
            result="line 1",
        )
        mod.on_post_llm_turn(
            session_id="session-1",
            turn_id="turn-1",
            assistant_response="done",
        )

        runs = mod.list_runs(limit=5)
        assert len(runs) == 1
        assert runs[0]["run_id"] == "turn:turn-1"
        assert runs[0]["status"] == "completed"
        assert runs[0]["source"] == "agent:main:slack:channel:C123:178.456"
        assert mod.list_runs(source="agent:main:slack:channel:C123")[0]["run_id"] == "turn:turn-1"

        run, events = mod.get_run("turn:turn-1")
        assert run is not None
        assert [e["event_type"] for e in events] == [
            "turn",
            "llm.request",
            "llm.response",
            "tool.call",
            "tool.result",
            "turn",
        ]
        tool_payload = json.loads(events[3]["payload_json"])
        assert tool_payload["args"]["api_key"] == "[REDACTED]"
        assert "secret-value" not in events[3]["payload_json"]
        assert "sk-tes...wxyz" not in events[0]["summary"]

    def test_records_gateway_delivery_and_thread_cli_query(self, tmp_path, monkeypatch):
        mod = _fresh_plugin(monkeypatch, tmp_path)
        mod.on_post_gateway_delivery(
            platform="slack",
            chat_id="C123",
            chat_type="group",
            thread_id="178.456",
            source="slack:C123:178.456",
            operation="send",
            status="ok",
            success=True,
            message_id="m-1",
            content_preview="reply text",
            content_chars=10,
            metadata={"thread_id": "178.456"},
            transport="edit",
        )

        rows = mod.list_thread_runs(platform="slack", chat_id="C123", thread_id="178.456")
        assert len(rows) == 1
        assert rows[0]["run_id"].startswith("process:")
        assert rows[0]["source"] == "slack:C123:178.456"

        _run, events = mod.get_run(rows[0]["run_id"])
        assert events[0]["event_type"] == "gateway.delivery"
        assert events[0]["name"] == "delivery_send"
        assert events[0]["status"] == "ok"
        payload = json.loads(events[0]["payload_json"])
        assert payload["message_id"] == "m-1"
        assert payload["content_preview"] == "reply text"

    def test_writes_self_contained_dashboard_html(self, tmp_path, monkeypatch):
        mod = _fresh_plugin(monkeypatch, tmp_path)
        mod.on_post_gateway_delivery(
            platform="slack",
            chat_id="C123",
            thread_id="178.456",
            source="slack:C123:178.456",
            operation="send",
            status="ok",
            success=True,
            message_id="m-1",
            content_preview="dashboard reply",
        )

        from plugins.observability.timeline.dashboard import write_dashboard

        out = write_dashboard(
            tmp_path / "timeline.html",
            list_runs=mod.list_runs,
            list_thread_runs=mod.list_thread_runs,
            get_run=mod.get_run,
            iso=mod._iso,
            platform="slack",
            chat_id="C123",
            thread_id="178.456",
        )
        html = out.read_text()
        assert "Hermes Timeline" in html
        assert "timeline-data" in html
        assert "dashboard reply" in html
        assert "gateway.delivery" in html
        assert "class=\"overview\"" in html
        assert "class=\"lane\"" in html
        assert "Delivery only" in html
        assert "laneOpenState" in html
        assert "data-lane-action" in html

    def test_timeline_server_serves_html_and_json(self, tmp_path, monkeypatch):
        import threading
        import urllib.request

        mod = _fresh_plugin(monkeypatch, tmp_path)
        mod.on_post_gateway_delivery(
            platform="slack",
            chat_id="C123",
            thread_id="178.456",
            source="slack:C123:178.456",
            operation="send",
            status="ok",
            success=True,
            message_id="m-1",
            content_preview="server reply",
        )

        from plugins.observability.timeline.server import TimelineRequestHandler, TimelineServer

        server = TimelineServer(
            ("127.0.0.1", 0),
            TimelineRequestHandler,
            deps={
                "list_runs": mod.list_runs,
                "list_thread_runs": mod.list_thread_runs,
                "get_run": mod.get_run,
                "iso": mod._iso,
            },
            default_filters={"limit": 10, "platform": "slack", "source": "", "chat_id": "C123", "thread_id": "178.456"},
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            html = urllib.request.urlopen(base + "/", timeout=5).read().decode()
            assert "Hermes Timeline" in html
            assert "/api/timeline" in html
            data = json.loads(urllib.request.urlopen(base + "/api/timeline", timeout=5).read().decode())
            assert data["runs"]
            assert data["runs"][0]["events"][0]["event_type"] == "gateway.delivery"
            assert data["runs"][0]["events"][0]["payload"]["content_preview"] == "server reply"
        finally:
            server.shutdown()
            server.server_close()

    def test_get_run_allows_unique_prefix(self, tmp_path, monkeypatch):
        mod = _fresh_plugin(monkeypatch, tmp_path)
        mod.on_pre_llm_call(session_id="s", turn_id="abcdef123456", user_message="hi")
        run, _events = mod.get_run("turn:abc")
        assert run is not None
        assert run["run_id"] == "turn:abcdef123456"

    def test_saves_and_uses_current_thread_preset(self, tmp_path, monkeypatch, capsys):
        import argparse

        mod = _fresh_plugin(monkeypatch, tmp_path)
        mod.on_post_gateway_delivery(
            platform="slack",
            chat_id="C123",
            thread_id="178.456",
            source="slack:C123:178.456",
            operation="send",
            status="ok",
            success=True,
            message_id="m-1",
            content_preview="preset reply",
        )
        preset = mod.save_preset("current", platform="slack", chat_id="C123", thread_id="178.456", limit=25)
        assert preset["name"] == "current"
        assert mod.get_preset("current")["chat_id"] == "C123"

        mod._print_current(argparse.Namespace(preset="current", limit=10, json=False))
        out = capsys.readouterr().out
        assert "slack:C123:178.456" in out

        dashboard_args = argparse.Namespace(
            preset="current",
            output=str(tmp_path / "preset.html"),
            limit=10,
            platform="",
            source="",
            chat_id="",
            thread_id="",
        )
        mod._print_dashboard(dashboard_args)
        assert "preset reply" in (tmp_path / "preset.html").read_text()


    def test_stats_prune_and_vacuum_retention_helpers(self, tmp_path, monkeypatch):
        mod = _fresh_plugin(monkeypatch, tmp_path)
        mod.on_pre_llm_call(session_id="s-old", turn_id="old", platform="slack", user_message="old")
        mod.on_pre_llm_call(session_id="s-new", turn_id="new", platform="cli", user_message="new")

        old_ts = mod._now() - (40 * 86400)
        conn = mod._ensure_db()
        try:
            conn.execute("UPDATE timeline_runs SET started_at = ? WHERE run_id = ?", (old_ts, "turn:old"))
            conn.execute("UPDATE timeline_events SET ts = ? WHERE run_id = ?", (old_ts, "turn:old"))
            conn.commit()
        finally:
            conn.close()

        stats = mod.timeline_stats()
        assert stats["runs"] == 2
        assert stats["events"] == 2
        assert stats["platforms"]["slack"] == 1
        assert stats["platforms"]["cli"] == 1

        dry = mod.prune_timeline(days=30, dry_run=True)
        assert dry["runs_matched"] == 1
        assert dry["events_matched"] == 1
        assert dry["runs_deleted"] == 0
        assert mod.get_run("turn:old")[0] is not None

        pruned = mod.prune_timeline(days=30, vacuum=True)
        assert pruned["runs_deleted"] == 1
        assert pruned["events_deleted"] == 1
        assert pruned["vacuumed"] is True
        assert mod.get_run("turn:old") == (None, [])
        assert mod.get_run("turn:new")[0] is not None

        vacuumed = mod.vacuum_timeline()
        assert vacuumed["bytes_before"] >= 0
        assert vacuumed["bytes_after"] >= 0


    def test_session_end_finishes_session_run_even_when_turn_id_present(self, tmp_path, monkeypatch):
        mod = _fresh_plugin(monkeypatch, tmp_path)
        mod.on_session_start(session_id="session-2", platform="cli", model="m")
        mod.on_session_end(session_id="session-2", turn_id="turn-2", platform="cli", completed=True)

        session_run, _events = mod.get_run("session:session-2")
        turn_run, turn_events = mod.get_run("turn:turn-2")
        assert session_run is not None
        assert session_run["status"] == "completed"
        assert turn_run is not None
        assert turn_run["status"] == "completed"
        assert turn_events[-1]["name"] == "session_end"


class TestCliRegistration:
    def test_register_adds_hooks_and_timeline_cli(self, tmp_path, monkeypatch):
        mod = _fresh_plugin(monkeypatch, tmp_path)
        registered_hooks = []
        cli_commands = []

        class Ctx:
            def register_hook(self, name, callback):
                registered_hooks.append((name, callback))

            def register_cli_command(self, name, **kwargs):
                cli_commands.append((name, kwargs))

        mod.register(Ctx())
        hook_names = {name for name, _ in registered_hooks}
        assert "pre_tool_call" in hook_names
        assert "post_tool_call" in hook_names
        assert "pre_api_request" in hook_names
        assert "post_api_request" in hook_names
        assert "pre_gateway_dispatch" in hook_names
        assert "post_gateway_delivery" in hook_names
        assert cli_commands and cli_commands[0][0] == "timeline"
        parser = __import__("argparse").ArgumentParser()
        cli_commands[0][1]["setup_fn"](parser)
        help_text = parser.format_help()
        assert "serve" in help_text
        assert "current" in help_text
        assert "preset" in help_text
        assert "stats" in help_text
        assert "prune" in help_text
        assert "vacuum" in help_text
