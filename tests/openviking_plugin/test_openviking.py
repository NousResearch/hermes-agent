"""Tests for plugins/memory/openviking/__init__.py — URI normalization, payload handling, and auto-start."""

import json
import os
import subprocess
import threading
from unittest.mock import MagicMock, patch

from plugins.memory.openviking import OpenVikingMemoryProvider


class FakeVikingClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get(self, path, params=None, **kwargs):
        self.calls.append((path, params or {}))
        response = self.responses[(path, tuple(sorted((params or {}).items())))]
        if isinstance(response, Exception):
            raise response
        return response


class TestOpenVikingSummaryUriNormalization:
    def test_normalize_summary_uri_maps_pseudo_files_to_parent_directory(self):
        assert OpenVikingMemoryProvider._normalize_summary_uri("viking://user/hermes/.overview.md") == "viking://user/hermes"
        assert OpenVikingMemoryProvider._normalize_summary_uri("viking://resources/.abstract.md") == "viking://resources"
        assert OpenVikingMemoryProvider._normalize_summary_uri("viking://") == "viking://"
        assert OpenVikingMemoryProvider._normalize_summary_uri("viking://user/hermes/memories/profile.md") == "viking://user/hermes/memories/profile.md"


class TestOpenVikingRead:
    def test_overview_read_normalizes_uri_and_unwraps_result(self):
        provider = OpenVikingMemoryProvider()
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/content/overview",
                    (("uri", "viking://user/hermes"),),
                ): {"result": {"content": "overview text"}},
            }
        )

        result = json.loads(provider._tool_read({"uri": "viking://user/hermes/.overview.md", "level": "overview"}))

        assert result["uri"] == "viking://user/hermes/.overview.md"
        assert result["resolved_uri"] == "viking://user/hermes"
        assert result["level"] == "overview"
        assert result["content"] == "overview text"
        assert provider._client.calls == [(
            "/api/v1/content/overview",
            {"uri": "viking://user/hermes"},
        )]

    def test_full_read_keeps_original_uri(self):
        provider = OpenVikingMemoryProvider()
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/content/read",
                    (("uri", "viking://user/hermes/memories/profile.md"),),
                ): {"result": "full text"},
            }
        )

        result = json.loads(provider._tool_read({"uri": "viking://user/hermes/memories/profile.md", "level": "full"}))

        assert result["uri"] == "viking://user/hermes/memories/profile.md"
        assert result["resolved_uri"] == "viking://user/hermes/memories/profile.md"
        assert result["level"] == "full"
        assert result["content"] == "full text"
        assert provider._client.calls == [(
            "/api/v1/content/read",
            {"uri": "viking://user/hermes/memories/profile.md"},
        )]

    def test_overview_file_uri_routes_straight_to_content_read_via_stat_probe(self):
        """Pre-check via fs/stat: file URIs skip the directory-only endpoint entirely."""
        provider = OpenVikingMemoryProvider()
        file_uri = "viking://user/hermes/memories/entities/mem_abc.md"
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/fs/stat",
                    (("uri", file_uri),),
                ): {"result": {"isDir": False}},
                (
                    "/api/v1/content/read",
                    (("uri", file_uri),),
                ): {"result": {"content": "full content"}},
            }
        )

        result = json.loads(provider._tool_read({"uri": file_uri, "level": "overview"}))

        assert result["uri"] == file_uri
        assert result["resolved_uri"] == file_uri
        assert result["level"] == "overview"
        assert result["fallback"] == "content/read"
        assert result["content"] == "full content"
        assert provider._client.calls == [
            ("/api/v1/fs/stat", {"uri": file_uri}),
            ("/api/v1/content/read", {"uri": file_uri}),
        ]

    def test_overview_dir_uri_skips_stat_when_pseudo_summary(self):
        """Pseudo-URI path already resolves to dir, so no stat probe needed."""
        provider = OpenVikingMemoryProvider()
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/content/overview",
                    (("uri", "viking://user/hermes"),),
                ): {"result": "overview"},
            }
        )

        result = json.loads(provider._tool_read({"uri": "viking://user/hermes/.overview.md", "level": "overview"}))

        assert result["content"] == "overview"
        # No fs/stat call — normalization already determined it's a directory.
        assert provider._client.calls == [
            ("/api/v1/content/overview", {"uri": "viking://user/hermes"}),
        ]

    def test_overview_directory_uri_uses_stat_probe_then_overview(self):
        """Non-pseudo directory URI: stat → isDir=True → summary endpoint."""
        provider = OpenVikingMemoryProvider()
        dir_uri = "viking://user/hermes/memories"
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/fs/stat",
                    (("uri", dir_uri),),
                ): {"result": {"isDir": True}},
                (
                    "/api/v1/content/overview",
                    (("uri", dir_uri),),
                ): {"result": "dir overview"},
            }
        )

        result = json.loads(provider._tool_read({"uri": dir_uri, "level": "overview"}))

        assert result["content"] == "dir overview"
        assert "fallback" not in result
        assert provider._client.calls == [
            ("/api/v1/fs/stat", {"uri": dir_uri}),
            ("/api/v1/content/overview", {"uri": dir_uri}),
        ]

    def test_overview_file_uri_falls_back_via_exception_when_stat_indeterminate(self):
        """If fs/stat raises or returns unknown shape, legacy exception fallback still kicks in."""
        provider = OpenVikingMemoryProvider()
        file_uri = "viking://user/hermes/memories/entities/mem_abc.md"
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/fs/stat",
                    (("uri", file_uri),),
                ): RuntimeError("stat unavailable"),
                (
                    "/api/v1/content/overview",
                    (("uri", file_uri),),
                ): RuntimeError("500 Internal Server Error"),
                (
                    "/api/v1/content/read",
                    (("uri", file_uri),),
                ): {"result": {"content": "fallback full content"}},
            }
        )

        result = json.loads(provider._tool_read({"uri": file_uri, "level": "overview"}))

        assert result["uri"] == file_uri
        assert result["level"] == "overview"
        assert result["fallback"] == "content/read"
        assert result["content"] == "fallback full content"
        assert provider._client.calls == [
            ("/api/v1/fs/stat", {"uri": file_uri}),
            ("/api/v1/content/overview", {"uri": file_uri}),
            ("/api/v1/content/read", {"uri": file_uri}),
        ]

    def test_summary_uri_error_does_not_fallback_and_raises(self):
        provider = OpenVikingMemoryProvider()
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/content/overview",
                    (("uri", "viking://user/hermes"),),
                ): RuntimeError("500 Internal Server Error"),
            }
        )

        try:
            provider._tool_read({"uri": "viking://user/hermes/.overview.md", "level": "overview"})
            assert False, "Expected summary endpoint error to be raised"
        except RuntimeError:
            pass

        assert provider._client.calls == [
            ("/api/v1/content/overview", {"uri": "viking://user/hermes"}),
        ]


class TestOpenVikingBrowse:
    def test_list_browse_unwraps_and_normalizes_entry_shapes(self):
        provider = OpenVikingMemoryProvider()
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/fs/ls",
                    (("uri", "viking://user/hermes"),),
                ): {
                    "result": {
                        "entries": [
                            {"name": "memories", "uri": "viking://user/hermes/memories", "type": "dir"},
                            {"rel_path": "profile.md", "uri": "viking://user/hermes/memories/profile.md", "isDir": False, "abstract": "Profile"},
                        ]
                    }
                },
            }
        )

        result = json.loads(provider._tool_browse({"action": "list", "path": "viking://user/hermes"}))

        assert result["path"] == "viking://user/hermes"
        assert result["entries"] == [
            {"name": "memories", "uri": "viking://user/hermes/memories", "type": "dir", "abstract": ""},
            {"name": "profile.md", "uri": "viking://user/hermes/memories/profile.md", "type": "file", "abstract": "Profile"},
        ]
        assert provider._client.calls == [(
            "/api/v1/fs/ls",
            {"uri": "viking://user/hermes"},
        )]


class TestOpenVikingMemoryUriBuilder:
    """Regression tests for _build_memory_uri — fixes #36969.

    Before the fix the URI omitted /agent/{agent}/, causing all agents
    under the same user to share the same memory namespace.
    """

    def _make_provider(self, user="alice", agent="coder"):
        p = OpenVikingMemoryProvider.__new__(OpenVikingMemoryProvider)
        p._user = user
        p._agent = agent
        return p

    def test_uri_layout_includes_agent_segment(self):
        """URI must contain /agent/{agent}/ between user and memories."""
        p = self._make_provider(user="alice", agent="coder")
        uri = p._build_memory_uri("preferences")
        assert uri.startswith("viking://user/alice/agent/coder/memories/preferences/mem_")
        assert uri.endswith(".md")

    def test_uri_uses_configured_agent_not_default(self):
        """_agent value must be interpolated — not hardcoded to 'hermes'."""
        p = self._make_provider(user="alice", agent="research-bot")
        uri = p._build_memory_uri("entities")
        assert "/agent/research-bot/" in uri
        assert "/agent/hermes/" not in uri

    def test_uri_slug_is_twelve_hex_chars_and_unique(self):
        """Slug must be 12 hex chars and differ between calls."""
        import re
        p = self._make_provider()
        uri1 = p._build_memory_uri("preferences")
        uri2 = p._build_memory_uri("preferences")
        slug1 = uri1.split("/mem_")[1].replace(".md", "")
        slug2 = uri2.split("/mem_")[1].replace(".md", "")
        assert re.fullmatch(r"[0-9a-f]{12}", slug1)
        assert re.fullmatch(r"[0-9a-f]{12}", slug2)
        assert slug1 != slug2

    def test_uri_subdir_placed_correctly_for_all_categories(self):
        """All five category subdirs must appear between memories/ and slug."""
        p = self._make_provider(user="u", agent="a")
        subdirs = ["preferences", "entities", "events", "cases", "patterns"]
        for subdir in subdirs:
            uri = p._build_memory_uri(subdir)
            assert f"/memories/{subdir}/mem_" in uri, (
                f"subdir '{subdir}' not placed correctly in URI: {uri}"
            )


class TestOpenVikingAutoStart:
    """Tests for the auto-start server logic in initialize()."""

    @staticmethod
    def _make_fake_monotonic(step=1.0, max_calls=None):
        """Fake time.monotonic: increments by *step*, jumps to 100.0 after *max_calls*."""
        state = {"t": 0.0, "calls": 0}

        def _fake():
            state["calls"] += 1
            if max_calls is not None and state["calls"] > max_calls:
                return 100.0  # past any reasonable deadline
            current = state["t"]
            state["t"] += step
            return current

        return _fake

    @patch.object(OpenVikingMemoryProvider, "_find_server_binary", return_value="/usr/local/bin/openviking-server")
    @patch("plugins.memory.openviking.subprocess.Popen")
    @patch("plugins.memory.openviking._VikingClient")
    def test_auto_start_launches_server_on_health_failure(self, MockClient, MockPopen, mock_find):
        """health() fails → Popen should be called."""
        mock_client_instance = MagicMock()
        mock_client_instance.health.return_value = False
        MockClient.return_value = mock_client_instance

        # Popen returns a mock process that stays alive
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345
        MockPopen.return_value = mock_proc

        # poll_client reports healthy on 3rd call
        mock_poll_client = MagicMock()
        health_call_count = {"n": 0}

        def health_succeeds_on_3rd():
            health_call_count["n"] += 1
            return health_call_count["n"] >= 3

        mock_poll_client.health.side_effect = health_succeeds_on_3rd
        MockClient.return_value = mock_poll_client

        with patch("plugins.memory.openviking.time.monotonic", side_effect=self._make_fake_monotonic()), \
             patch("plugins.memory.openviking.time.sleep"):
            provider = OpenVikingMemoryProvider()
            provider.initialize("test-session")

            # Wait for daemon thread
            for t in threading.enumerate():
                if t.name == "openviking-server-start":
                    t.join(timeout=5.0)

        MockPopen.assert_called_once()
        cmd = MockPopen.call_args[0][0]
        assert cmd[0] == "/usr/local/bin/openviking-server"
        assert "--config" in cmd
        assert "--host" in cmd
        assert "--port" in cmd
        assert MockPopen.call_args[1]["start_new_session"] is True
        assert health_call_count["n"] >= 3

    @patch.object(OpenVikingMemoryProvider, "_find_server_binary", return_value=None)
    @patch("plugins.memory.openviking._VikingClient")
    def test_auto_start_skipped_when_server_not_found(self, MockClient, mock_find):
        """Binary not found → no Popen, client stays None."""
        mock_client_instance = MagicMock()
        mock_client_instance.health.return_value = False
        MockClient.return_value = mock_client_instance

        with patch("plugins.memory.openviking.subprocess.Popen") as MockPopen:
            provider = OpenVikingMemoryProvider()
            provider.initialize("test-session")

            # Wait for daemon thread
            for t in threading.enumerate():
                if t.name == "openviking-server-start":
                    t.join(timeout=5.0)

            # No process was started
            MockPopen.assert_not_called()

        assert provider._client is None

    @patch("plugins.memory.openviking._VikingClient")
    def test_no_auto_start_when_server_already_healthy(self, MockClient):
        """health() succeeds → no auto-start."""
        mock_client_instance = MagicMock()
        mock_client_instance.health.return_value = True
        MockClient.return_value = mock_client_instance

        with patch("plugins.memory.openviking.subprocess.Popen") as MockPopen:
            provider = OpenVikingMemoryProvider()
            provider.initialize("test-session")

            # No server process should be spawned
            MockPopen.assert_not_called()

        assert provider._client is not None

    @patch.object(OpenVikingMemoryProvider, "_find_server_binary", return_value="/usr/local/bin/openviking-server")
    @patch("plugins.memory.openviking.subprocess.Popen")
    @patch("plugins.memory.openviking._VikingClient")
    def test_auto_start_graceful_degradation_on_timeout(self, MockClient, MockPopen, mock_find):
        """Server never healthy → client stays None."""
        mock_poll_client = MagicMock()
        mock_poll_client.health.return_value = False
        MockClient.return_value = mock_poll_client

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345
        MockPopen.return_value = mock_proc

        # Force timeout after 5 polling iterations
        with patch("plugins.memory.openviking.time.monotonic",
                    side_effect=self._make_fake_monotonic(max_calls=5)), \
             patch("plugins.memory.openviking.time.sleep"):
            provider = OpenVikingMemoryProvider()
            provider.initialize("test-session")

            for t in threading.enumerate():
                if t.name == "openviking-server-start":
                    t.join(timeout=5.0)

        assert provider._client is None

    @patch.object(OpenVikingMemoryProvider, "_find_server_binary", return_value="/usr/local/bin/openviking-server")
    @patch("plugins.memory.openviking.subprocess.Popen")
    @patch("plugins.memory.openviking._VikingClient")
    def test_auto_start_handles_premature_exit(self, MockClient, MockPopen, mock_find):
        """Server exits early → client stays None."""
        mock_poll_client = MagicMock()
        mock_poll_client.health.return_value = False
        MockClient.return_value = mock_poll_client

        mock_proc = MagicMock()
        poll_call_count = {"n": 0}

        def poll_exits():
            poll_call_count["n"] += 1
            if poll_call_count["n"] >= 2:
                return 1
            return None

        mock_proc.poll.side_effect = poll_exits
        mock_proc.returncode = 1
        mock_proc.pid = 12345
        MockPopen.return_value = mock_proc

        with patch("plugins.memory.openviking.time.monotonic", side_effect=self._make_fake_monotonic()), \
             patch("plugins.memory.openviking.time.sleep"):
            provider = OpenVikingMemoryProvider()
            provider.initialize("test-session")

            for t in threading.enumerate():
                if t.name == "openviking-server-start":
                    t.join(timeout=5.0)

        assert provider._client is None
        assert provider._server_process is None

    def test_no_duplicate_start_when_server_already_starting(self):
        """Sentinel blocks duplicate _start_server_thread() calls."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345

        mock_poll_client = MagicMock()
        mock_poll_client.health.return_value = False

        provider = OpenVikingMemoryProvider()
        provider._endpoint = "http://127.0.0.1:1933"
        provider._api_key = ""
        provider._account = "default"
        provider._user = "default"
        provider._agent = "hermes"

        with patch.object(OpenVikingMemoryProvider, "_find_server_binary", return_value="/usr/local/bin/openviking-server"), \
             patch("plugins.memory.openviking.subprocess.Popen", return_value=mock_proc) as MockPopen, \
             patch("plugins.memory.openviking._VikingClient", return_value=mock_poll_client), \
             patch("plugins.memory.openviking.time.sleep"):
            # Stall monotonic at 0 so the polling loop never exits.
            with patch("plugins.memory.openviking.time.monotonic", return_value=0.0):
                provider._start_server_thread()

                import time as _time
                _time.sleep(0.1)

                assert provider._server_process is not None

                # Second call: sentinel active → returns early.
                provider._start_server_thread()

            for t in threading.enumerate():
                if t.name == "openviking-server-start":
                    t.join(timeout=5.0)

        assert MockPopen.call_count == 1

    @patch.object(OpenVikingMemoryProvider, "_find_server_binary", return_value="/usr/local/bin/openviking-server")
    @patch("plugins.memory.openviking.subprocess.Popen")
    @patch("plugins.memory.openviking._VikingClient")
    def test_auto_start_uses_openviking_config_env(self, MockClient, MockPopen, mock_find):
        """OPENVIKING_CONFIG env var → --config flag."""
        mock_poll_client = MagicMock()
        mock_poll_client.health.return_value = False
        MockClient.return_value = mock_poll_client

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345
        MockPopen.return_value = mock_proc

        with patch.dict(os.environ, {"OPENVIKING_CONFIG": "/custom/ov.conf"}), \
             patch("plugins.memory.openviking.time.monotonic",
                    side_effect=self._make_fake_monotonic(max_calls=3)), \
             patch("plugins.memory.openviking.time.sleep"):
            provider = OpenVikingMemoryProvider()
            provider.initialize("test-session")

            for t in threading.enumerate():
                if t.name == "openviking-server-start":
                    t.join(timeout=5.0)

        cmd = MockPopen.call_args[0][0]
        config_idx = cmd.index("--config")
        assert cmd[config_idx + 1] == "/custom/ov.conf"

    def test_shutdown_terminates_auto_started_server(self):
        """shutdown() terminates a process we spawned."""
        provider = OpenVikingMemoryProvider()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 99999
        provider._server_process = mock_proc

        provider.shutdown()

        mock_proc.terminate.assert_called_once()

    def test_shutdown_does_not_terminate_external_server(self):
        """No server_process → shutdown() is a no-op."""
        provider = OpenVikingMemoryProvider()
        provider._server_process = None
        provider.shutdown()

    def test_shutdown_clears_sentinel_without_terminating(self):
        """_server_process=True → cleared to None, no terminate/kill."""
        provider = OpenVikingMemoryProvider()
        provider._server_process = True
        provider.shutdown()
        assert provider._server_process is None

    @patch.object(OpenVikingMemoryProvider, "_find_server_binary", return_value=None)
    @patch("plugins.memory.openviking._VikingClient")
    def test_shutdown_waits_for_start_thread(self, MockClient, mock_find):
        """shutdown() joins the start thread before clearing _server_process."""
        mock_client_instance = MagicMock()
        mock_client_instance.health.return_value = False
        MockClient.return_value = mock_client_instance

        provider = OpenVikingMemoryProvider()
        provider.initialize("test-session")

        # Thread exits quickly (binary not found). shutdown must join it.
        provider.shutdown()

        assert provider._server_process is None

    @patch.object(OpenVikingMemoryProvider, "_find_server_binary", return_value="/usr/local/bin/openviking-server")
    @patch("plugins.memory.openviking.subprocess.Popen")
    @patch("plugins.memory.openviking._VikingClient")
    def test_restarts_server_after_process_exit(self, MockClient, MockPopen, mock_find):
        """Dead process reference → second _start_server_thread() restarts."""
        mock_poll_client = MagicMock()
        mock_poll_client.health.return_value = False
        MockClient.return_value = mock_poll_client

        # The new process that Popen will return (single call)
        live_proc = MagicMock()
        live_proc.poll.return_value = None
        live_proc.pid = 22222
        MockPopen.return_value = live_proc

        # Simulate a previously-started process that has since exited
        dead_proc = MagicMock()
        dead_proc.poll.return_value = 1
        dead_proc.returncode = 1
        dead_proc.pid = 11111

        provider = OpenVikingMemoryProvider()
        provider._endpoint = "http://127.0.0.1:1933"
        provider._api_key = ""
        provider._account = "default"
        provider._user = "default"
        provider._agent = "hermes"
        provider._server_process = dead_proc

        with patch("plugins.memory.openviking.time.monotonic",
                    side_effect=self._make_fake_monotonic(max_calls=5)), \
             patch("plugins.memory.openviking.time.sleep"):
            provider._start_server_thread()

            for t in threading.enumerate():
                if t.name == "openviking-server-start":
                    t.join(timeout=5.0)

        assert MockPopen.call_count == 1


class TestFindServerBinary:
    """Tests for _find_server_binary(): PATH → venv bin fallback."""

    def test_finds_binary_in_path(self):
        """shutil.which finds it → return immediately."""
        provider = OpenVikingMemoryProvider()
        with patch("plugins.memory.openviking.shutil.which", return_value="/usr/local/bin/openviking-server"):
            assert provider._find_server_binary() == "/usr/local/bin/openviking-server"

    def test_falls_back_to_venv_bin(self):
        """shutil.which=None → check venv bin directory."""
        provider = OpenVikingMemoryProvider()
        # Create a fake executable
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake executable
            fake_bin = Path(tmpdir) / "openviking-server"
            fake_bin.write_text("#!/bin/sh\necho ok\n")
            fake_bin.chmod(0o755)

            with patch("plugins.memory.openviking.shutil.which", return_value=None), \
                 patch("plugins.memory.openviking.sys") as mock_sys:
                mock_sys.executable = str(Path(tmpdir) / "python")
                result = provider._find_server_binary()

            assert result == str(fake_bin)

    def test_returns_none_when_not_found_anywhere(self):
        """Neither PATH nor venv bin has it → None."""
        provider = OpenVikingMemoryProvider()
        with patch("plugins.memory.openviking.shutil.which", return_value=None), \
             patch("plugins.memory.openviking.Path") as mock_path_cls:
            mock_venv_bin = MagicMock()
            mock_venv_bin.is_file.return_value = False
            mock_parent = MagicMock()
            mock_parent.__truediv__ = lambda self, other: mock_venv_bin
            mock_path_cls.return_value.parent = mock_parent

            assert provider._find_server_binary() is None

    @patch.object(OpenVikingMemoryProvider, "_find_server_binary", return_value="/venv/bin/openviking-server")
    @patch("plugins.memory.openviking.subprocess.Popen")
    @patch("plugins.memory.openviking._VikingClient")
    def test_auto_start_uses_venv_binary_when_not_in_path(self, MockClient, MockPopen, mock_find):
        """Integration: venv fallback binary → auto-start works."""
        mock_client_instance = MagicMock()
        mock_client_instance.health.return_value = False
        MockClient.return_value = mock_client_instance

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 54321
        MockPopen.return_value = mock_proc

        mock_poll_client = MagicMock()
        health_count = {"n": 0}
        mock_poll_client.health.side_effect = lambda: (health_count.update(n=health_count["n"] + 1) or health_count["n"]) >= 2
        MockClient.return_value = mock_poll_client

        with patch("plugins.memory.openviking.time.monotonic",
                    side_effect=TestOpenVikingAutoStart._make_fake_monotonic()), \
             patch("plugins.memory.openviking.time.sleep"):
            provider = OpenVikingMemoryProvider()
            provider.initialize("test-session")

            for t in threading.enumerate():
                if t.name == "openviking-server-start":
                    t.join(timeout=5.0)

        MockPopen.assert_called_once()
        cmd = MockPopen.call_args[0][0]
        assert cmd[0] == "/venv/bin/openviking-server"


class TestServerLogRedirection:
    """Tests for _open_server_log() and Popen stdout/stderr wiring."""

    def test_open_server_log_returns_file_handle(self):
        """_open_server_log() returns open file handle on success."""
        import tempfile
        from pathlib import Path

        # Reset class-level cache
        OpenVikingMemoryProvider._server_log_fh = None

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_home = Path(tmpdir)
            with patch.dict("sys.modules", {"hermes_constants": MagicMock(get_hermes_home=lambda: mock_home)}):
                fh = OpenVikingMemoryProvider._open_server_log()

            try:
                assert fh is not None
                assert fh.mode == "a"
                assert (mock_home / "logs").is_dir()
            finally:
                if fh:
                    fh.close()
                OpenVikingMemoryProvider._server_log_fh = None

    def test_open_server_log_returns_none_on_failure(self):
        """Log dir not writable → returns None."""
        # Reset class-level cache
        OpenVikingMemoryProvider._server_log_fh = None

        with patch.dict("sys.modules", {"hermes_constants": MagicMock()}) as mods:
            mods["hermes_constants"].get_hermes_home.side_effect = OSError("no home")
            result = OpenVikingMemoryProvider._open_server_log()

        assert result is None
        OpenVikingMemoryProvider._server_log_fh = None

    def test_open_server_log_caches_handle(self):
        """Repeated calls return the same handle."""
        import tempfile
        from pathlib import Path

        OpenVikingMemoryProvider._server_log_fh = None

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_home = Path(tmpdir)
            with patch.dict("sys.modules", {"hermes_constants": MagicMock(get_hermes_home=lambda: mock_home)}):
                fh1 = OpenVikingMemoryProvider._open_server_log()
                fh2 = OpenVikingMemoryProvider._open_server_log()

            try:
                assert fh1 is fh2
            finally:
                if fh1:
                    fh1.close()
                OpenVikingMemoryProvider._server_log_fh = None

    @patch.object(OpenVikingMemoryProvider, "_find_server_binary", return_value="/usr/local/bin/openviking-server")
    @patch("plugins.memory.openviking.subprocess.Popen")
    @patch("plugins.memory.openviking._VikingClient")
    def test_popen_receives_log_handle_not_devnull(self, MockClient, MockPopen, mock_find):
        """_open_server_log succeeds → Popen gets real file handle, not DEVNULL."""
        OpenVikingMemoryProvider._server_log_fh = None

        mock_client_instance = MagicMock()
        mock_client_instance.health.return_value = False
        MockClient.return_value = mock_client_instance

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345
        MockPopen.return_value = mock_proc

        mock_poll_client = MagicMock()
        mock_poll_client.health.return_value = False
        MockClient.return_value = mock_poll_client

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_home = Path(tmpdir)
            with patch.dict("sys.modules", {"hermes_constants": MagicMock(get_hermes_home=lambda: mock_home)}), \
                 patch("plugins.memory.openviking.time.monotonic",
                        side_effect=TestOpenVikingAutoStart._make_fake_monotonic(max_calls=3)), \
                 patch("plugins.memory.openviking.time.sleep"):
                provider = OpenVikingMemoryProvider()
                provider.initialize("test-session")

                for t in threading.enumerate():
                    if t.name == "openviking-server-start":
                        t.join(timeout=5.0)

            # Popen stdout/stderr should NOT be DEVNULL
            popen_kwargs = MockPopen.call_args[1]
            assert popen_kwargs["stdout"] is not subprocess.DEVNULL
            assert popen_kwargs["stderr"] is not subprocess.DEVNULL

        if OpenVikingMemoryProvider._server_log_fh:
            OpenVikingMemoryProvider._server_log_fh.close()
        OpenVikingMemoryProvider._server_log_fh = None

    @patch.object(OpenVikingMemoryProvider, "_find_server_binary", return_value="/usr/local/bin/openviking-server")
    @patch("plugins.memory.openviking.subprocess.Popen")
    @patch("plugins.memory.openviking._VikingClient")
    def test_popen_falls_back_to_devnull_when_log_unavailable(self, MockClient, MockPopen, mock_find):
        """_open_server_log fails → Popen gets DEVNULL."""
        OpenVikingMemoryProvider._server_log_fh = None

        mock_client_instance = MagicMock()
        mock_client_instance.health.return_value = False
        MockClient.return_value = mock_client_instance

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345
        MockPopen.return_value = mock_proc

        mock_poll_client = MagicMock()
        mock_poll_client.health.return_value = False
        MockClient.return_value = mock_poll_client

        with patch.dict("sys.modules", {"hermes_constants": MagicMock()}) as mods:
            mods["hermes_constants"].get_hermes_home.side_effect = OSError("no home")
            with patch("plugins.memory.openviking.time.monotonic",
                        side_effect=TestOpenVikingAutoStart._make_fake_monotonic(max_calls=3)), \
                 patch("plugins.memory.openviking.time.sleep"):
                provider = OpenVikingMemoryProvider()
                provider.initialize("test-session")

                for t in threading.enumerate():
                    if t.name == "openviking-server-start":
                        t.join(timeout=5.0)

        # Popen stdout/stderr should be DEVNULL (fallback)
        popen_kwargs = MockPopen.call_args[1]
        assert popen_kwargs["stdout"] is subprocess.DEVNULL
        assert popen_kwargs["stderr"] is subprocess.DEVNULL

        OpenVikingMemoryProvider._server_log_fh = None


class TestAutoStartWhitelist:
    """Loopback whitelist: only local endpoints trigger auto-start."""

    def test_auto_start_port_for_loopback(self):
        provider = OpenVikingMemoryProvider()
        for endpoint, expected in [
            ("http://127.0.0.1:1933", 1933),
            ("http://localhost:1933", 1933),
            ("http://[::1]:1933", 1933),
        ]:
            provider._endpoint = endpoint
            assert provider._auto_start_port() == expected

        # Non-loopback / HTTPS → None
        provider._endpoint = "http://openviking.example.com:1933"
        assert provider._auto_start_port() is None
        provider._endpoint = "https://127.0.0.1:1933"
        assert provider._auto_start_port() is None

    @patch("plugins.memory.openviking._VikingClient")
    def test_no_auto_start_for_remote_endpoint(self, MockClient):
        """Remote endpoint → no Popen, client stays None."""
        mock_client_instance = MagicMock()
        mock_client_instance.health.return_value = False
        MockClient.return_value = mock_client_instance

        with patch.dict(os.environ, {"OPENVIKING_ENDPOINT": "http://openviking.example.com:1933"}), \
             patch("plugins.memory.openviking.subprocess.Popen") as MockPopen:
            provider = OpenVikingMemoryProvider()
            provider.initialize("test-session")
            MockPopen.assert_not_called()

        assert provider._client is None
        assert provider._server_process is None

    @patch.object(OpenVikingMemoryProvider, "_find_server_binary", return_value="/usr/local/bin/openviking-server")
    @patch("plugins.memory.openviking.subprocess.Popen")
    @patch("plugins.memory.openviking._VikingClient")
    def test_auto_start_proceeds_for_loopback_endpoint(self, MockClient, MockPopen, mock_find):
        """Loopback endpoint → Popen called normally."""
        mock_poll_client = MagicMock()
        health_count = {"n": 0}
        mock_poll_client.health.side_effect = lambda: (health_count.update(n=health_count["n"] + 1) or health_count["n"]) >= 2
        MockClient.return_value = mock_poll_client

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345
        MockPopen.return_value = mock_proc

        with patch("plugins.memory.openviking.time.monotonic",
                    side_effect=TestOpenVikingAutoStart._make_fake_monotonic()), \
             patch("plugins.memory.openviking.time.sleep"):
            provider = OpenVikingMemoryProvider()
            provider.initialize("test-session")

            for t in threading.enumerate():
                if t.name == "openviking-server-start":
                    t.join(timeout=5.0)

        MockPopen.assert_called_once()
