"""Tests for _resolve_fs_data_url_max_bytes and _resolve_dashboard_ws_max_size_bytes."""


class TestResolveFsDataUrlMaxBytes:
    """Tests for _resolve_fs_data_url_max_bytes priority chain."""

    def test_default_when_no_config_no_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("HERMES_FS_DATA_URL_MAX_BYTES", raising=False)
        import hermes_cli.web_server as ws

        assert ws._resolve_fs_data_url_max_bytes() == 16_777_216

    def test_env_wins_over_default(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_FS_DATA_URL_MAX_BYTES", "52428800")
        import hermes_cli.web_server as ws

        assert ws._resolve_fs_data_url_max_bytes() == 52_428_800

    def test_config_wins_over_env(self, monkeypatch, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "gateway:\n  file_upload:\n    fs_data_url_max_bytes: 100000000\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_FS_DATA_URL_MAX_BYTES", "52428800")
        import hermes_cli.web_server as ws

        assert ws._resolve_fs_data_url_max_bytes() == 100_000_000

    def test_quoted_yaml_string_falls_through(self, monkeypatch, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "gateway:\n  file_upload:\n    fs_data_url_max_bytes: \"not-a-number\"\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_FS_DATA_URL_MAX_BYTES", "77777777")
        import hermes_cli.web_server as ws

        assert ws._resolve_fs_data_url_max_bytes() == 77_777_777


class TestResolveDashboardWsMaxSizeBytes:
    """Tests for _resolve_dashboard_ws_max_size_bytes priority chain."""

    def test_default_when_no_config_no_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("HERMES_DASHBOARD_WS_MAX_SIZE_BYTES", raising=False)
        import hermes_cli.web_server as ws

        assert ws._resolve_dashboard_ws_max_size_bytes() == 16_777_216

    def test_env_wins_over_default(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_DASHBOARD_WS_MAX_SIZE_BYTES", "52428800")
        import hermes_cli.web_server as ws

        assert ws._resolve_dashboard_ws_max_size_bytes() == 52_428_800

    def test_config_wins_over_env(self, monkeypatch, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "gateway:\n  file_upload:\n    dashboard_ws_max_size_bytes: 100000000\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_DASHBOARD_WS_MAX_SIZE_BYTES", "52428800")
        import hermes_cli.web_server as ws

        assert ws._resolve_dashboard_ws_max_size_bytes() == 100_000_000

    def test_quoted_yaml_string_falls_through(self, monkeypatch, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "gateway:\n  file_upload:\n    dashboard_ws_max_size_bytes: \"not-a-number\"\n"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_DASHBOARD_WS_MAX_SIZE_BYTES", "77777777")
        import hermes_cli.web_server as ws

        assert ws._resolve_dashboard_ws_max_size_bytes() == 77_777_777
