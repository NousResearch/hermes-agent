"""Custom-endpoint save semantics served by ``/api/providers/custom-endpoints``.

Split out of ``test_web_server.py`` (over the file-lines cap) so endpoint-save
regressions have a topical home. See #150116.
"""

import pytest


class TestCustomEndpointSave:
    """Desktop's Custom Endpoints form → ``providers.<name>`` config writes."""

    @pytest.fixture(autouse=True)
    def _setup_test_client(self, monkeypatch, _isolate_hermes_home):
        """Create a TestClient and isolate the state DB under the test HERMES_HOME."""
        try:
            from starlette.testclient import TestClient
        except ImportError:
            pytest.skip("fastapi/starlette not installed")

        import hermes_state
        from hermes_constants import get_hermes_home
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")

        self.client = TestClient(app)
        self.client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    def test_save_drops_stale_legacy_api_url_fields(self):
        """A pre-v12 entry's legacy ``api``/``url`` must not outrank the saved ``base_url`` (#150116).

        The runtime resolves the endpoint as ``api`` or ``url`` or ``base_url``
        (``agent.agent_init``), so a stale legacy ``api:`` kept routing every
        request to the old host while the panel displayed — and the user kept
        editing — ``base_url``. Saving must drop both legacy spellings.
        """
        from hermes_cli.config import get_config_path, load_config

        get_config_path().write_text(
            "providers:\n"
            "  strata:\n"
            "    name: strata\n"
            "    api: http://192.168.3.89:18080/v1\n"
            "    base_url: http://192.168.3.89:18075/v1\n"
            "    model: qwen3.8-flash-next-iq3_s\n",
            encoding="utf-8",
        )

        saved = self.client.post(
            "/api/providers/custom-endpoints",
            json={"id": "strata", "name": "strata",
                  "base_url": "http://192.168.3.89:18075/v1", "model": "qwen3.8-flash-next-iq3_s"},
        )
        assert saved.status_code == 200, saved.text

        entry = load_config()["providers"]["strata"]
        assert entry["base_url"] == "http://192.168.3.89:18075/v1"
        assert "api" not in entry, f"stale legacy api survived the save: {entry}"
        assert "url" not in entry, f"stale legacy url survived the save: {entry}"
