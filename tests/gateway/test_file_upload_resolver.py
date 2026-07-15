"""Tests for _resolve_max_request_bytes priority chain."""


class TestResolveMaxRequestBytes:
    """Tests for _resolve_max_request_bytes priority chain."""

    def test_default_when_no_config(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from gateway.platforms.api_server import _resolve_max_request_bytes

        assert _resolve_max_request_bytes() == 10_000_000

    def test_config_value(self, monkeypatch, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "gateway:\n  file_upload:\n    max_request_bytes: 100000000\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from gateway.platforms.api_server import _resolve_max_request_bytes

        assert _resolve_max_request_bytes() == 100_000_000

    def test_unlimited_zero(self, monkeypatch, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "gateway:\n  file_upload:\n    max_request_bytes: 0\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from gateway.platforms.api_server import _resolve_max_request_bytes

        assert _resolve_max_request_bytes() == 0

    def test_quoted_yaml_string_falls_through(self, monkeypatch, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "gateway:\n  file_upload:\n    max_request_bytes: \"not-a-number\"\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from gateway.platforms.api_server import _resolve_max_request_bytes

        assert _resolve_max_request_bytes() == 10_000_000
