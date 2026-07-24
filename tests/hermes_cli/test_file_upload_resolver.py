"""Tests for _resolve_fs_data_url_max_bytes and _resolve_dashboard_ws_max_size_bytes."""


class TestResolveFsDataUrlMaxBytes:
    """Tests for _resolve_fs_data_url_max_bytes priority chain."""

    def test_default_when_no_config(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import hermes_cli.web_server as ws

        assert ws._resolve_fs_data_url_max_bytes() == 16_777_216

    def test_config_value(self, monkeypatch, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "gateway:\n  file_upload:\n    fs_data_url_max_bytes: 100000000\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import hermes_cli.web_server as ws

        assert ws._resolve_fs_data_url_max_bytes() == 100_000_000

    def test_unlimited_zero(self, monkeypatch, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "gateway:\n  file_upload:\n    fs_data_url_max_bytes: 0\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import hermes_cli.web_server as ws

        assert ws._resolve_fs_data_url_max_bytes() == 0

    def test_quoted_yaml_string_falls_through(self, monkeypatch, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "gateway:\n  file_upload:\n    fs_data_url_max_bytes: \"not-a-number\"\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import hermes_cli.web_server as ws

        assert ws._resolve_fs_data_url_max_bytes() == 16_777_216


class TestResolveDashboardWsMaxSizeBytes:
    """Tests for _resolve_dashboard_ws_max_size_bytes priority chain."""

    def test_default_when_no_config(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import hermes_cli.web_server as ws

        assert ws._resolve_dashboard_ws_max_size_bytes() == 16_777_216

    def test_config_value(self, monkeypatch, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "gateway:\n  file_upload:\n    dashboard_ws_max_size_bytes: 100000000\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import hermes_cli.web_server as ws

        assert ws._resolve_dashboard_ws_max_size_bytes() == 100_000_000

    def test_unlimited_zero(self, monkeypatch, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "gateway:\n  file_upload:\n    dashboard_ws_max_size_bytes: 0\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import hermes_cli.web_server as ws

        assert ws._resolve_dashboard_ws_max_size_bytes() == 0

    def test_quoted_yaml_string_falls_through(self, monkeypatch, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "gateway:\n  file_upload:\n    dashboard_ws_max_size_bytes: \"not-a-number\"\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import hermes_cli.web_server as ws

        assert ws._resolve_dashboard_ws_max_size_bytes() == 16_777_216
