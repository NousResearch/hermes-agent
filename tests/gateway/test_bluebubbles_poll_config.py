"""BlueBubbles polling settings survive YAML loading and environment overrides."""

import pytest

from gateway.config import Platform, load_gateway_config


@pytest.fixture
def bluebubbles_config_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234")
    monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "test-password")
    monkeypatch.delenv("BLUEBUBBLES_POLL_INTERVAL", raising=False)
    return tmp_path


def _write_poll_interval(home, interval):
    (home / "config.yaml").write_text(
        "platforms:\n"
        "  bluebubbles:\n"
        "    enabled: true\n"
        "    extra:\n"
        f"      poll_interval: {interval}\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize("interval", [0, 30])
def test_yaml_poll_interval_survives_env_credentials(bluebubbles_config_home, interval):
    _write_poll_interval(bluebubbles_config_home, interval)

    config = load_gateway_config()

    assert config.platforms[Platform.BLUEBUBBLES].extra["poll_interval"] == interval


@pytest.mark.parametrize("raw, expected", [("30", 30), ("0", 0), ("0.5", 0.5)])
def test_env_poll_interval_overrides_yaml(bluebubbles_config_home, monkeypatch, raw, expected):
    _write_poll_interval(bluebubbles_config_home, 60)
    monkeypatch.setenv("BLUEBUBBLES_POLL_INTERVAL", raw)

    config = load_gateway_config()

    assert config.platforms[Platform.BLUEBUBBLES].extra["poll_interval"] == expected
