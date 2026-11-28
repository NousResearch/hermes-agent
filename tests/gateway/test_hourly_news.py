import http.client
import io
import itertools
import json
import os
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

from integrations import sonos_hourly_news


class HourlyNewsTests(unittest.TestCase):
    def test_restore_resumes_interrupted_source_at_saved_position_when_news_is_current(self):
        class FakeClient:
            def __init__(self):
                self.calls = []
                self.uri = "http://speaker/news.mp3"

            def media_info(self):
                return {"current_uri": self.uri}

            def transport_state(self):
                return "PLAYING"

            def set_uri(self, uri, metadata=""):
                self.calls.append(("set_uri", uri, metadata))
                self.uri = uri

            def seek_rel_time(self, position):
                self.calls.append(("seek", position))

            def play(self):
                self.calls.append(("play",))

        client = FakeClient()
        snapshot = {
            "transport": "PLAYING", "source_uri": "x-sonos-http:source", "metadata": "<DIDL>track</DIDL>",
            "position": "00:12:34", "duration": "00:50:00",
        }
        with patch.object(sonos_hourly_news, "_playback_blocked", return_value=False), patch.object(
            sonos_hourly_news, "_speaker_operation_lock", return_value=nullcontext()
        ), patch.object(sonos_hourly_news, "_log_event"), patch.object(sonos_hourly_news, "_read_stream_state", return_value={}):
            restored = sonos_hourly_news._restore_interrupted_source(client, "http://speaker/news.mp3", snapshot)
        self.assertTrue(restored)
        self.assertEqual(client.calls, [
            ("set_uri", "x-sonos-http:source", "<DIDL>track</DIDL>"),
            ("seek", "00:12:34"), ("play",),
        ])

    def test_snapshot_preserves_didl_and_finite_track_position_but_not_its_own_proxy(self):
        class FakeClient:
            def __init__(self, uri="x-sonos-http:radio"):
                self.uri = uri

            def media_info(self):
                return {"current_uri": self.uri}

            def position_info(self):
                return {
                    "track_uri": self.uri, "track_metadata": "<DIDL>radio metadata</DIDL>",
                    "rel_time": "00:02:10", "track_duration": "00:20:00",
                }

        with patch.object(sonos_hourly_news, "_read_stream_state", return_value={"port": 1234}):
            snapshot = sonos_hourly_news._capture_resume_snapshot(FakeClient(), "PLAYING")
            active_news = sonos_hourly_news._capture_resume_snapshot(
                FakeClient("http://127.0.0.1:1234/news.mp3"), "PLAYING"
            )
        self.assertEqual(snapshot["metadata"], "<DIDL>radio metadata</DIDL>")
        self.assertEqual(snapshot["duration"], "00:20:00")
        self.assertEqual(snapshot["position"], "00:02:10")
        self.assertEqual(active_news, {})

        queue_uri = "x-rincon-queue:RINCON_123#0"

        class QueuedClient(FakeClient):
            def __init__(self):
                self.uri = queue_uri

            def position_info(self):
                return {
                    "track_uri": "x-sonos-http:current-track", "track_metadata": "<DIDL>current</DIDL>",
                    "rel_time": "00:01:00", "track_duration": "00:04:00",
                }

        with patch.object(sonos_hourly_news, "_read_stream_state", return_value={}):
            queued = sonos_hourly_news._capture_resume_snapshot(QueuedClient(), "PLAYING")
        self.assertEqual(queued["source_uri"], queue_uri)

    def test_restore_does_not_seek_non_seekable_live_source(self):
        class FakeClient:
            def __init__(self):
                self.uri = "http://speaker/news.mp3"
                self.calls = []

            def media_info(self): return {"current_uri": self.uri}
            def transport_state(self): return "PLAYING"
            def set_uri(self, uri, metadata=""):
                self.calls.append(("set_uri", uri, metadata))
                self.uri = uri
            def seek_rel_time(self, position): self.calls.append(("seek", position))
            def play(self): self.calls.append(("play",))

        client = FakeClient()
        snapshot = {"transport": "PLAYING", "source_uri": "http://radio/live", "position": "00:02:10", "duration": "0:00:00"}
        with patch.object(sonos_hourly_news, "_playback_blocked", return_value=False), patch.object(
            sonos_hourly_news, "_speaker_operation_lock", return_value=nullcontext()
        ), patch.object(sonos_hourly_news, "_log_event"), patch.object(sonos_hourly_news, "_read_stream_state", return_value={}):
            self.assertTrue(sonos_hourly_news._restore_interrupted_source(client, "http://speaker/news.mp3", snapshot))
        self.assertEqual([call[0] for call in client.calls], ["set_uri", "play"])

    def test_restore_does_not_override_manual_transport_or_uri_changes_or_policy(self):
        snapshot = {"transport": "PLAYING", "source_uri": "http://radio/source", "position": "00:12:34"}

        class FakeClient:
            def __init__(self, transport="PLAYING", uri="http://speaker/news.mp3"):
                self.state, self.uri, self.calls = transport, uri, []

            def media_info(self): return {"current_uri": self.uri}
            def transport_state(self): return self.state
            def set_uri(self, uri, metadata=""): self.calls.append(("set_uri", uri, metadata))
            def seek_rel_time(self, position): self.calls.append(("seek", position))
            def play(self): self.calls.append(("play",))

        for client, blocked in (
            (FakeClient(transport="PAUSED_PLAYBACK"), False),
            (FakeClient(transport="STOPPED"), False),
            (FakeClient(uri="http://speaker/other.mp3"), False),
            (FakeClient(), True),
        ):
            with self.subTest(transport=client.state, uri=client.uri, blocked=blocked), patch.object(
                sonos_hourly_news, "_playback_blocked", return_value=blocked
            ), patch.object(sonos_hourly_news, "_speaker_operation_lock", return_value=nullcontext()):
                self.assertFalse(sonos_hourly_news._restore_interrupted_source(client, "http://speaker/news.mp3", snapshot))
                self.assertEqual(client.calls, [])

    def test_default_settings_interrupt_with_france24(self):
        settings = sonos_hourly_news.default_settings()
        self.assertTrue(settings["enabled"])
        self.assertEqual(settings["station"], "france24")
        self.assertEqual(settings["mode"], "interrupt")
        self.assertEqual(settings["duration_minutes"], 15)
        self.assertEqual(settings["after_news_action"], "radio")
        self.assertEqual(settings["after_news_station_id"], "")
        for duration in (5, 10, 15, 30):
            self.assertEqual(
                sonos_hourly_news.update_settings(settings, duration_minutes=duration)["duration_minutes"],
                duration,
            )

    def test_after_news_settings_validate_action_and_station_id(self):
        settings = sonos_hourly_news.default_settings()
        updated = sonos_hourly_news.update_settings(
            settings, after_news_action="resume", after_news_station_id="managed-123"
        )
        self.assertEqual(updated["after_news_action"], "resume")
        self.assertEqual(updated["after_news_station_id"], "managed-123")
        for action in ("radio", "resume", "idle"):
            self.assertEqual(
                sonos_hourly_news.update_settings(settings, after_news_action=action)["after_news_action"],
                action,
            )
        with self.assertRaises(ValueError):
            sonos_hourly_news.update_settings(settings, after_news_action="play")

    def test_ffmpeg_command_pins_system_ca_bundle_and_keeps_tls_verification(self):
        with patch.object(sonos_hourly_news, "_ffmpeg_ca_file", return_value="/etc/ssl/cert.pem", create=True):
            command = sonos_hourly_news._ffmpeg_command("https://live.example/stream")
        ca_index = command.index("-ca_file")
        input_index = command.index("-i")
        self.assertEqual(command[ca_index + 1], "/etc/ssl/cert.pem")
        self.assertLess(ca_index, input_index)
        self.assertEqual(command[command.index("-tls_verify") + 1], "1")

    def test_ffmpeg_environment_propagates_ca_to_nested_hls_requests(self):
        with patch.object(sonos_hourly_news, "_ffmpeg_ca_file", return_value="/fixture/ca.pem"):
            env = sonos_hourly_news._ffmpeg_environment()
        self.assertEqual(env["SSL_CERT_FILE"], "/fixture/ca.pem")
        self.assertEqual(env.get("PATH"), os.environ.get("PATH"))

    def test_audio_preflight_is_bounded_and_rejects_failed_or_empty_audio(self):
        for code, audio in ((0, b"a" * 32644), (251, b""), (0, b"")):
            with self.subTest(code=code, audio_bytes=len(audio)), patch.object(
                sonos_hourly_news.subprocess, "run",
                return_value=subprocess.CompletedProcess([], code, audio, b"failed https://example.test/?sig=secret"),
            ) as run, patch.object(sonos_hourly_news, "_log_event"):
                if code == 0 and audio:
                    sonos_hourly_news._preflight_audio("https://media.example/live")
                else:
                    with self.assertRaises(RuntimeError) as error:
                        sonos_hourly_news._preflight_audio("https://media.example/live")
                    self.assertNotIn("sig=secret", str(error.exception))
                self.assertEqual(run.call_args.kwargs["timeout"], 20)
                self.assertEqual(run.call_args.args[0][-3:], ["-t", "2", "pipe:1"])
                self.assertEqual(run.call_args.args[0][run.call_args.args[0].index("-tls_verify") + 1], "1")

    def test_audio_preflight_timeout_does_not_expose_signed_url(self):
        with patch.object(sonos_hourly_news.subprocess, "run", side_effect=subprocess.TimeoutExpired(
            ["ffmpeg", "https://media.example/live?sig=secret"], 20,
        )), patch.object(sonos_hourly_news, "_log_event"):
            with self.assertRaisesRegex(RuntimeError, "timed out") as error:
                sonos_hourly_news._preflight_audio("https://media.example/live?sig=secret")
        self.assertNotIn("sig=secret", str(error.exception))

    def test_ffmpeg_error_excerpt_redacts_signed_upstream_url(self):
        stderr = io.BytesIO(b"HTTP Error 403: https://media.example/live.mp3?expire=123&sig=secret-token")
        excerpt = sonos_hourly_news._ffmpeg_error_excerpt(stderr)
        self.assertIn("HTTP Error 403", excerpt)
        self.assertIn("<url>", excerpt)
        self.assertNotIn("secret-token", excerpt)

    def test_proxy_logs_sanitized_ffmpeg_failure_details(self):
        events = []

        class FailedFfmpeg:
            stdout = io.BytesIO()

            def poll(self):
                return 7

            def terminate(self):
                return None

            def wait(self, timeout):
                return 7

            def kill(self):
                return None

        def fake_popen(*_args, **kwargs):
            kwargs["stderr"].write(b"HTTP Error 403: https://media.example/live.mp3?sig=secret-token")
            kwargs["stderr"].flush()
            return FailedFfmpeg()

        with patch.object(sonos_hourly_news, "_write_stream_state"), patch.object(
            sonos_hourly_news, "_log_event", side_effect=lambda event, **fields: events.append((event, fields))
        ), patch.object(sonos_hourly_news.subprocess, "Popen", side_effect=fake_popen):
            server = sonos_hourly_news._NewsStreamServer(("127.0.0.1", 0), "https://media.example/live", 10)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
                connection.request("GET", "/news.mp3")
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(response.read(), b"")
                connection.close()
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()

        failure = next(fields for event, fields in events if event == "proxy_stream_failed")
        self.assertEqual(failure["ffmpeg_exit_code"], 7)
        self.assertIn("HTTP Error 403", failure["error"])
        self.assertNotIn("secret-token", failure["error"])

    def test_renderer_disconnect_does_not_mark_upstream_failed_or_shutdown_proxy(self):
        class StreamingFfmpeg:
            stdout = io.BytesIO(b"audio")

            def poll(self):
                return None

            def terminate(self):
                return None

            def wait(self, timeout):
                return 0

        server = sonos_hourly_news._NewsStreamServer(("127.0.0.1", 0), "https://media.example/live", 10)
        handler = object.__new__(sonos_hourly_news._NewsStreamHandler)
        handler.path = "/news.mp3"
        handler.server = server
        handler.wfile = Mock()
        handler.wfile.write.side_effect = BrokenPipeError()
        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()
        try:
            with patch.object(sonos_hourly_news.subprocess, "Popen", return_value=StreamingFfmpeg()), patch.object(
                sonos_hourly_news, "_write_stream_state"
            ), patch.object(sonos_hourly_news, "_log_event"):
                handler.do_GET()
            self.assertFalse(server.stream_failed)
            self.assertEqual(server.request_count, 1)
            self.assertEqual(server.active_requests, 0)
        finally:
            server.server_close()

    def test_hourly_news_cli_imports_when_run_outside_project_directory(self):
        with tempfile.TemporaryDirectory() as state_dir:
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            env["HOME"] = state_dir
            result = subprocess.run(
                [sys.executable, str(Path(sonos_hourly_news.__file__).resolve()), "status"],
                cwd=state_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)["success"])

    def test_news_disable_persists_off_and_reports_scheduler_pause_failure(self):
        with tempfile.TemporaryDirectory() as state_dir, patch.object(
            sonos_hourly_news, "CONFIG_PATH", Path(state_dir) / "hourly-news.json"
        ), patch.object(sonos_hourly_news, "STATE_DIR", Path(state_dir)), patch.object(
            sonos_hourly_news, "_cron_call", return_value={"success": False}
        ) as cron_call:
            sonos_hourly_news.save_settings({
                **sonos_hourly_news.default_settings(), "enabled": True, "job_id": "news-job-42",
            })
            result = sonos_hourly_news.settings_operation("set_enabled", enabled=False)
            status = sonos_hourly_news.settings_operation("status")
        self.assertFalse(result["success"])
        self.assertFalse(status["settings"]["enabled"])
        self.assertIn("scheduler", result)
        cron_call.assert_called_once_with(
            "pause", job_id="news-job-42", reason="Hourly news disabled from Sonos settings"
        )

    def test_kill_automations_disables_news_and_post_news_action_and_pauses_job(self):
        with tempfile.TemporaryDirectory() as state_dir, patch.object(
            sonos_hourly_news, "CONFIG_PATH", Path(state_dir) / "hourly-news.json"
        ), patch.object(sonos_hourly_news, "STATE_DIR", Path(state_dir)), patch.object(
            sonos_hourly_news, "_cron_call", return_value={"success": True}
        ) as cron_call:
            sonos_hourly_news.save_settings({
                **sonos_hourly_news.default_settings(), "enabled": True, "job_id": "news-job-42",
                "after_news_action": "radio",
            })
            result = sonos_hourly_news.settings_operation("kill_automations")
            status = sonos_hourly_news.settings_operation("status")
        self.assertTrue(result["success"])
        self.assertFalse(status["settings"]["enabled"])
        self.assertEqual(status["settings"]["after_news_action"], "idle")
        cron_call.assert_called_once_with(
            "pause", job_id="news-job-42", reason="Hourly news and post-news radio paused from Sonos settings"
        )

    def test_settings_operation_persists_after_news_action_and_station_id(self):
        with tempfile.TemporaryDirectory() as state_dir, patch.object(
            sonos_hourly_news, "CONFIG_PATH", Path(state_dir) / "hourly-news.json"
        ), patch.object(sonos_hourly_news, "STATE_DIR", Path(state_dir)):
            action = sonos_hourly_news.settings_operation("set_after_news_action", after_news_action="resume")
            station = sonos_hourly_news.settings_operation("set_after_news_station", station_id="managed-123")
            status = sonos_hourly_news.settings_operation("status")
        self.assertEqual(action["settings"]["after_news_action"], "resume")
        self.assertEqual(station["settings"]["after_news_station_id"], "managed-123")
        self.assertEqual(status["settings"]["after_news_action"], "resume")
        self.assertEqual(status["settings"]["after_news_station_id"], "managed-123")
        self.assertTrue(all(set(item) == {"id", "name"} for item in status["managed_radio_stations"]))

    def test_status_exposes_safe_managed_station_records_without_urls(self):
        from integrations import sonos_radio_stations

        station = {"id": "managed-123", "name": "Radio", "url": "https://radio.example/secret"}
        with tempfile.TemporaryDirectory() as state_dir, patch.object(
            sonos_hourly_news, "CONFIG_PATH", Path(state_dir) / "hourly-news.json"
        ), patch.object(sonos_hourly_news, "STATE_DIR", Path(state_dir)), patch.object(
            sonos_radio_stations, "list_stations", return_value=[station]
        ):
            status = sonos_hourly_news.settings_operation("status")
        self.assertEqual(status["managed_radio_stations"], [{"id": "managed-123", "name": "Radio"}])
        self.assertNotIn("https://radio.example/secret", json.dumps(status))

    def test_post_news_expiry_plays_managed_radio_only_for_current_playing_news(self):
        class FakeClient:
            def __init__(self, state="PLAYING", uri="http://speaker/news.mp3"):
                self.state, self.uri, self.calls = state, uri, []
            def media_info(self): return {"current_uri": self.uri}
            def transport_state(self): return self.state
            def set_uri(self, uri): self.calls.append(("set_uri", uri)); self.uri = uri
            def play(self): self.calls.append(("play",))

        station = {"id": "managed-123", "name": "Radio", "url": "https://radio.example/stream"}
        for state, uri, blocked, expected in (
            ("PLAYING", "http://speaker/news.mp3", False, True),
            ("STOPPED", "http://speaker/news.mp3", False, False),
            ("PLAYING", "http://speaker/other.mp3", False, False),
            ("PLAYING", "http://speaker/news.mp3", True, False),
        ):
            client = FakeClient(state, uri)
            with self.subTest(state=state, uri=uri, blocked=blocked), patch.object(
                sonos_hourly_news, "_playback_blocked", return_value=blocked
            ), patch.object(sonos_hourly_news, "_speaker_operation_lock", return_value=nullcontext()), patch.object(
                sonos_hourly_news, "_get_managed_radio_station", return_value=station
            ), patch.object(sonos_hourly_news, "_log_event") as log:
                result = sonos_hourly_news._handle_post_news_expiry(
                    client, "http://speaker/news.mp3", {}, "radio", "managed-123"
                )
            self.assertEqual(result, expected)
            self.assertEqual(bool(client.calls), expected)
            self.assertNotIn(station["url"], repr(log.call_args_list))

    def test_post_news_expiry_resume_and_idle_have_distinct_behaviors(self):
        snapshot = {"transport": "PLAYING", "source_uri": "x-sonos-http:source"}
        class FakeClient:
            def __init__(self):
                self.calls = []
            def media_info(self): return {"current_uri": "http://speaker/news.mp3"}
            def transport_state(self): return "PLAYING"
            def stop(self): self.calls.append(("stop",))
            def clear_media(self): self.calls.append(("clear_media",))

        client = FakeClient()
        with patch.object(sonos_hourly_news, "_restore_interrupted_source", return_value=True) as restore, patch.object(
            sonos_hourly_news, "_playback_blocked", return_value=False
        ), patch.object(sonos_hourly_news, "_speaker_operation_lock", return_value=nullcontext()), patch.object(
            sonos_hourly_news, "_log_event"
        ):
            self.assertTrue(sonos_hourly_news._handle_post_news_expiry(client, "http://speaker/news.mp3", snapshot, "resume", ""))
            self.assertFalse(sonos_hourly_news._handle_post_news_expiry(client, "http://speaker/news.mp3", snapshot, "idle", ""))
        restore.assert_called_once_with(client, "http://speaker/news.mp3", snapshot)
        self.assertEqual(client.calls, [("stop",), ("clear_media",)])

    def test_post_news_expiry_invalid_station_leaves_idle_without_logging_url(self):
        client = type("Client", (), {"media_info": lambda self: {"current_uri": "http://speaker/news.mp3"}, "transport_state": lambda self: "PLAYING"})()
        with patch.object(sonos_hourly_news, "_playback_blocked", return_value=False), patch.object(
            sonos_hourly_news, "_speaker_operation_lock", return_value=nullcontext()
        ), patch.object(sonos_hourly_news, "_get_managed_radio_station", return_value=None), patch.object(
            sonos_hourly_news, "_log_event"
        ) as log:
            self.assertFalse(sonos_hourly_news._handle_post_news_expiry(client, "http://speaker/news.mp3", {}, "radio", "missing"))
        log.assert_called()

    def test_update_settings_only_accepts_allowlisted_values(self):
        settings = sonos_hourly_news.default_settings()
        self.assertEqual(
            sonos_hourly_news.update_settings(settings, station="dw", mode="idle", duration_minutes=15)["station"],
            "dw",
        )
        with self.assertRaises(ValueError):
            sonos_hourly_news.update_settings(settings, station="random-channel")
        with self.assertRaises(ValueError):
            sonos_hourly_news.update_settings(settings, duration_minutes=7)

    def test_rendered_stream_headers_are_sonos_compatible(self):
        self.assertEqual(sonos_hourly_news.stream_content_type(Path("news.mp3")), "audio/mpeg")
        self.assertEqual(sonos_hourly_news.stream_content_type(Path("news.wav")), "audio/wav")

    def test_cron_wrapper_uses_current_python_and_hourly_trigger(self):
        wrapper = sonos_hourly_news.wrapper_text()
        self.assertIn("hourly-news", wrapper)
        self.assertIn("sys.executable", wrapper)
        self.assertIn("trigger", wrapper)
        self.assertIn("raise SystemExit(result.returncode)", wrapper)

    def test_diagnostic_errors_redact_stream_urls(self):
        error = sonos_hourly_news._safe_error(
            RuntimeError("GET https://example.test/live?token=secret failed")
        )
        self.assertNotIn("token=secret", error)
        self.assertIn("<url>", error)

    def test_trigger_skips_before_resolving_source_when_policy_is_blocked(self):
        with tempfile.TemporaryDirectory() as state_dir, patch.object(
            sonos_hourly_news, "STATE_DIR", Path(state_dir)
        ), patch.object(
            sonos_hourly_news, "LOG_PATH", Path(state_dir) / "hourly-news.jsonl"
        ), patch.object(sonos_hourly_news, "load_settings", return_value=sonos_hourly_news.default_settings()), patch.object(
            sonos_hourly_news, "_playback_blocked", return_value=True
        ), patch.object(sonos_hourly_news, "_sonos_client") as client_factory, patch.object(
            sonos_hourly_news, "_resolve_live_url"
        ) as resolve:
            result = sonos_hourly_news.trigger_news()
        self.assertEqual(result["skipped"], "policy_blocked")
        client_factory.assert_not_called()
        resolve.assert_not_called()

    def test_playback_rechecks_policy_inside_operation_lock(self):
        class FakeClient:
            def __init__(self):
                self.set_uri_calls = 0
                self.play_calls = 0

            def set_uri(self, _uri):
                self.set_uri_calls += 1

            def play(self):
                self.play_calls += 1

        fake_sonos_brain = types.ModuleType("sonos_brain")
        fake_sonos_brain.__dict__.update({
            "SPEAKER_OPERATION_LOCK": "test-lock",
            "_exclusive_operation_lock": lambda _lock: nullcontext(),
        })
        client = FakeClient()
        with patch.dict(sys.modules, {"sonos_brain": fake_sonos_brain}), patch.object(
            sonos_hourly_news, "_playback_blocked", side_effect=[False, True]
        ):
            started = sonos_hourly_news.start_sonos_playback(client, "http://example.test/news.mp3", interrupt=False)
        self.assertFalse(started)
        self.assertEqual(client.set_uri_calls, 0)
        self.assertEqual(client.play_calls, 0)


class HourlyNewsFailureTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        for name in ("STATE_DIR", "STREAM_STATE_PATH", "RESUME_STATE_PATH", "PID_PATH", "LOG_PATH"):
            value = Path(directory) if name == "STATE_DIR" else Path(directory) / name
            self.enterContext(patch.object(sonos_hourly_news, name, value))
        self.client = Mock()
        self.client.uri = "x-sonos-http:prior-source"
        self.client.state = "PLAYING"
        self.client.media_info.side_effect = lambda: {"current_uri": self.client.uri}
        self.client.transport_state.side_effect = lambda: self.client.state
        self.client.position_info.return_value = {
            "track_metadata": "<DIDL>prior</DIDL>", "rel_time": "00:02:00", "track_duration": "00:10:00",
        }
        self.client.set_uri.side_effect = lambda uri, *args: setattr(self.client, "uri", uri)
        self.client.play.side_effect = lambda: setattr(self.client, "state", "PLAYING")
        self.client.stop.side_effect = lambda: setattr(self.client, "state", "STOPPED")
        brain = types.ModuleType("sonos_brain")
        brain._stop_fm_program = Mock()
        brain.SPEAKER_OPERATION_LOCK = Path(directory) / "speaker.lock"
        brain._exclusive_operation_lock = lambda _path: nullcontext()
        self.enterContext(patch.dict(sys.modules, {"sonos_brain": brain}))
        self.enterContext(patch.object(sonos_hourly_news, "load_settings", return_value=sonos_hourly_news.default_settings()))
        self.enterContext(patch.object(sonos_hourly_news, "_sonos_client", return_value=self.client))
        self.policy = self.enterContext(patch.object(sonos_hourly_news, "_playback_blocked", return_value=False))
        self.enterContext(patch.object(sonos_hourly_news, "_speaker_operation_lock", side_effect=nullcontext))
        self.resolve = self.enterContext(patch.object(sonos_hourly_news, "_resolve_live_url", return_value="https://media.example/live"))
        self.enterContext(patch.object(sonos_hourly_news, "_resolve_live_metadata", return_value={}))
        self.preflight = self.enterContext(patch.object(sonos_hourly_news, "_preflight_audio", create=True))
        self.enterContext(patch.object(sonos_hourly_news, "_stop_existing_news"))
        self.enterContext(patch.object(sonos_hourly_news, "_local_lan_ip", return_value="127.0.0.1"))
        self.enterContext(patch.object(sonos_hourly_news, "_log_event"))
        self.enterContext(patch.object(sonos_hourly_news.time, "monotonic", side_effect=itertools.count()))
        self.enterContext(patch.object(sonos_hourly_news.time, "sleep"))
        self.process = Mock(stdout=io.BytesIO(b"READY 1234\n"), stderr=io.BytesIO())
        self.process.poll.return_value = None
        def spawn_proxy(*_args, **_kwargs):
            self.process.stdout = io.BytesIO(b"READY 1234\n")
            return self.process
        self.spawn = self.enterContext(patch.object(sonos_hourly_news.subprocess, "Popen", side_effect=spawn_proxy))
        self.enterContext(patch.object(sonos_hourly_news.select, "select", return_value=([self.process.stdout], [], [])))
        self.start_playback = sonos_hourly_news.start_sonos_playback
        self.handover = self.enterContext(patch.object(sonos_hourly_news, "start_sonos_playback"))

    def assert_no_speaker_changes(self):
        for name in ("stop", "clear_media", "set_uri", "play", "seek_rel_time"):
            getattr(self.client, name).assert_not_called()

    def test_manual_play_now_skips_any_non_idle_transport_even_in_interrupt_mode(self):
        result = sonos_hourly_news.trigger_news(only_if_idle=True)
        self.assertEqual(result["skipped"], "busy")
        self.resolve.assert_not_called()
        self.preflight.assert_not_called()
        self.assert_no_speaker_changes()

    def test_upstream_resolution_failure_leaves_source_untouched(self):
        self.resolve.side_effect = RuntimeError("upstream unavailable")
        with self.assertRaisesRegex(RuntimeError, "upstream unavailable"):
            sonos_hourly_news.trigger_news()
        self.spawn.assert_not_called()
        self.assert_no_speaker_changes()

    def test_preflight_failure_does_not_start_proxy_or_handover(self):
        self.preflight.side_effect = RuntimeError("certificate verify failed")
        with self.assertRaisesRegex(RuntimeError, "certificate verify failed"):
            sonos_hourly_news.trigger_news()
        self.spawn.assert_not_called()
        self.handover.assert_not_called()
        self.resolve.assert_called_once()
        self.assert_no_speaker_changes()

    def test_preflight_does_not_override_a_new_source_manual_stop_or_policy(self):
        for change in ("source", "stop", "policy"):
            with self.subTest(change=change):
                self.client.uri, self.client.state = "x-sonos-http:prior-source", "PLAYING"
                self.policy.return_value = False
                def change_playback(_url):
                    if change == "source":
                        self.client.uri = "x-sonos-vli:new-airplay"
                    elif change == "stop":
                        self.client.state = "STOPPED"
                    else:
                        self.policy.return_value = True
                self.preflight.side_effect = change_playback
                result = sonos_hourly_news.trigger_news()
                self.assertFalse(result["started"])
                self.spawn.assert_not_called()
                self.handover.assert_not_called()
                self.assert_no_speaker_changes()

    def test_failed_handover_restores_owned_source_without_clearing_media(self):
        def fail_handover(_client, uri, **_kwargs):
            self.client.uri = uri
            raise RuntimeError("handover failed")
        self.handover.side_effect = fail_handover
        with self.assertRaisesRegex(RuntimeError, "handover failed"):
            sonos_hourly_news.trigger_news()
        self.client.set_uri.assert_called_once_with("x-sonos-http:prior-source", "<DIDL>prior</DIDL>")
        self.client.play.assert_called_once()
        self.client.stop.assert_not_called()
        self.client.clear_media.assert_not_called()
        self.process.terminate.assert_called_once()

    def test_failed_handover_leaves_new_source_manual_stop_and_policy_untouched(self):
        for change in ("source", "stop", "policy"):
            with self.subTest(change=change):
                self.client.uri, self.client.state = "x-sonos-http:prior-source", "PLAYING"
                self.policy.return_value = False
                def fail_handover(_client, uri, **_kwargs):
                    self.client.uri = "x-sonos-vli:new-airplay" if change == "source" else uri
                    self.client.state = "STOPPED" if change == "stop" else "PLAYING"
                    self.policy.return_value = change == "policy"
                    raise RuntimeError("handover failed")
                self.handover.side_effect = fail_handover
                with self.assertRaisesRegex(RuntimeError, "handover failed"):
                    sonos_hourly_news.trigger_news()
                self.assert_no_speaker_changes()

    def test_handover_rechecks_source_under_lock_and_never_stops_for_retries(self):
        self.client.uri = "x-sonos-vli:new-airplay"
        self.assertFalse(self.start_playback(
            self.client, "http://speaker/news.mp3", interrupt=True,
            expected_source=("PLAYING", "x-sonos-http:prior-source"),
        ))
        self.assert_no_speaker_changes()
        def failed_play():
            self.client.uri = "x-sonos-vli:new-airplay"
            raise RuntimeError("play failed")
        self.client.play.side_effect = failed_play
        with self.assertRaisesRegex(RuntimeError, "play failed"):
            self.start_playback(
                self.client, "http://speaker/news.mp3", interrupt=True,
                expected_source=("PLAYING", "x-sonos-vli:new-airplay"),
            )
        self.client.set_uri.assert_called_once_with("http://speaker/news.mp3")
        self.client.stop.assert_not_called()

    def test_handover_aborts_if_source_changes_while_fm_worker_stops(self):
        with patch.object(sys.modules["sonos_brain"], "_stop_fm_program", side_effect=lambda: setattr(
            self.client, "uri", "x-sonos-vli:new-airplay"
        )):
            self.assertFalse(self.start_playback(
                self.client, "http://speaker/news.mp3", interrupt=True,
                expected_source=("PLAYING", "x-sonos-http:prior-source"),
            ))
        self.assert_no_speaker_changes()

    def test_recovery_does_not_seek_or_play_a_source_selected_during_restore(self):
        self.client.uri = "http://speaker/news.mp3"
        self.client.set_uri.side_effect = lambda *_args: setattr(self.client, "uri", "x-sonos-vli:new-airplay")
        self.assertFalse(sonos_hourly_news._restore_interrupted_source(
            self.client, "http://speaker/news.mp3", {
                "transport": "PLAYING", "source_uri": "x-sonos-http:prior-source",
                "position": "00:02:00", "duration": "00:10:00",
            },
        ))
        self.client.seek_rel_time.assert_not_called()
        self.client.play.assert_not_called()

    def test_verification_does_not_mistake_new_user_playback_for_news(self):
        def handover(_client, uri, **_kwargs):
            self.client.uri = uri
            return True
        self.handover.side_effect = handover
        sizes = itertools.count(32768, 32768)
        def stream_state():
            if self.client.uri.startswith("http://127.0.0.1:1234/"):
                self.client.uri = "x-sonos-vli:new-airplay"
            return {"request_count": 1, "bytes_sent": next(sizes), "active_requests": 1}
        with patch.object(sonos_hourly_news, "_read_stream_state", side_effect=stream_state):
            with self.assertRaisesRegex(RuntimeError, "source changed"):
                sonos_hourly_news.trigger_news()
        self.assert_no_speaker_changes()

    def test_proxy_failure_recovers_snapshot_instead_of_starting_post_news_radio(self):
        resolved = sonos_hourly_news.STATE_DIR / "resolved.url"
        resolved.write_text("https://media.example/live")
        server = Mock(server_address=("127.0.0.1", 1234), stream_failed=True,
                      ffmpeg_processes=set(), ffmpeg_lock=nullcontext(), request_count=1,
                      bytes_sent=32768, active_requests=0)
        server.expired = threading.Event()
        snapshot = {"transport": "PLAYING", "source_uri": "x-sonos-http:prior-source"}
        sonos_hourly_news._write_resume_state({"news_uri": "http://127.0.0.1:1234/news.mp3", "snapshot": snapshot})
        with patch.object(sonos_hourly_news, "_NewsStreamServer", return_value=server), patch.object(
            sonos_hourly_news.threading, "Timer"
        ), patch.object(sonos_hourly_news, "_restore_interrupted_source", return_value=True) as restore, patch.object(
            sonos_hourly_news, "_handle_post_news_expiry"
        ) as expiry:
            self.assertEqual(sonos_hourly_news._serve(resolved, 15), 0)
        restore.assert_called_once_with(self.client, "http://127.0.0.1:1234/news.mp3", snapshot)
        expiry.assert_not_called()


if __name__ == "__main__":
    unittest.main()
