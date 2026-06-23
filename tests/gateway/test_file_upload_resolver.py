"""Tests for _resolve_max_request_bytes priority chain."""
import os


class TestResolveMaxRequestBytes:
    """Tests for _resolve_max_request_bytes priority chain."""

    def test_default_when_no_config_no_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("API_SERVER_MAX_REQUEST_BYTES", raising=False)
        from gateway.platforms.api_server import _resolve_max_request_bytes

        assert _resolve_max_request_bytes() == 10_000_000

    def test_env_wins_over_default(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("API_SERVER_MAX_REQUEST_BYTES", "52428800")
        from gateway.platforms.api_server import _resolve_max_request_bytes

        assert _resolve_max_request_bytes() == 52_428_800

    def test_config_wins_over_env(self, monkeypatch, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "gateway:\n  file_upload:\n    max_request_bytes: 100000000\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("API_SERVER_MAX_REQUEST_BYTES", "52428800")
        from gateway.platforms.api_server import _resolve_max_request_bytes

        assert _resolve_max_request_bytes() == 100_000_000

    def test_quoted_yaml_string_falls_through(self, monkeypatch, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "gateway:\n  file_upload:\n    max_request_bytes: \"not-a-number\"\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("API_SERVER_MAX_REQUEST_BYTES", "77777777")
        from gateway.platforms.api_server import _resolve_max_request_bytes

        assert _resolve_max_request_bytes() == 77_777_777
