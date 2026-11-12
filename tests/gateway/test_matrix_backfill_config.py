"""Matrix thread backfill limit configuration."""
import pytest

from gateway.config import Platform
from tests.gateway.matrix_helpers import _make_adapter


class TestMatrixConfigLoading:
    def test_thread_backfill_limit_reaches_adapter_from_config_yaml(self, tmp_path, monkeypatch):
        from gateway.config import load_gateway_config
        from plugins.platforms.matrix.adapter import MatrixAdapter

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "matrix:\n  thread_backfill_limit: 5\n  room_backfill_limit: 7\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("MATRIX_HOMESERVER", "https://matrix.example.org")
        monkeypatch.setenv("MATRIX_USER_ID", "@bot:example.org")
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_test_token")

        config = load_gateway_config().platforms[Platform.MATRIX]

        adapter = MatrixAdapter(config)
        assert (
            config.extra["thread_backfill_limit"],
            adapter._thread_backfill_limit,
            config.extra["room_backfill_limit"],
            adapter._room_backfill_limit,
        ) == (5, 5, 7, 7)

    @pytest.mark.parametrize("key", ["thread_backfill_limit", "room_backfill_limit"])
    def test_dashboard_offers_backfill_limit_at_the_adapter_default(self, key):
        from starlette.testclient import TestClient
        from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

        client = TestClient(app, headers={_SESSION_HEADER_NAME: _SESSION_TOKEN})
        field = (
            client.get("/api/config/schema").json()["fields"].get(f"matrix.{key}", {})
        )
        shown = client.get("/api/config").json()["matrix"].get(key)

        assert (field.get("type"), shown) == (
            "number",
            getattr(_make_adapter(), f"_{key}"),
        )
