"""Proactive code-skew surface on /api/status.

The reactive half shipped first: ``/api/model/options`` refuses with 503
``Restart required`` (``_dashboard_code_skew_guard``).  That only fires when the
user happens to open the Models page, so a Desktop user who never does runs the
pre-update checkout for days with no signal at all (#118998).

The Desktop app already polls ``/api/status`` every 60s (status-snapshot flow),
so the backend publishes its skew state there: same ``detect_code_skew()``
truth, a surface the owning app already reads.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from gateway import code_skew
from hermes_cli import web_server


class TestStatusSkewSurface:
    """``/api/status`` reports this process's code skew to its owning app."""

    @pytest.fixture(autouse=True)
    def _setup_test_client(self, monkeypatch):
        monkeypatch.setattr(code_skew, "_boot_fingerprint", None)
        try:
            self.client = TestClient(web_server.app)
        except ImportError:  # pragma: no cover - fastapi/starlette not installed
            pytest.skip("fastapi/starlette not installed")

    def test_skew_is_reported_with_both_revisions(self, monkeypatch):
        monkeypatch.setattr(
            code_skew, "detect_code_skew", lambda: ("abc1234567", "def4567890")
        )

        body = self.client.get("/api/status").json()

        assert body["code_skew"] == {"boot_rev": "abc1234567", "disk_rev": "def4567890"}

    def test_no_skew_means_the_field_is_absent_not_null(self, monkeypatch):
        """Fresh checkout (or non-git install) keeps the exact pre-fix shape."""
        monkeypatch.setattr(code_skew, "detect_code_skew", lambda: None)

        body = self.client.get("/api/status").json()

        assert "code_skew" not in body

    def test_a_raising_detector_never_breaks_the_status_probe(self, monkeypatch):
        def boom():
            raise RuntimeError("fingerprint read failed")

        monkeypatch.setattr(code_skew, "detect_code_skew", boom)

        response = self.client.get("/api/status")

        assert response.status_code == 200
        assert "code_skew" not in response.json()

    def test_model_options_still_refuses_while_skew_is_published(self, monkeypatch):
        """The proactive surface does not weaken the reactive guard."""
        monkeypatch.setattr(
            code_skew, "detect_code_skew", lambda: ("abc1234567", "def4567890")
        )

        assert (
            self.client.get("/api/status").json()["code_skew"]["disk_rev"]
            == "def4567890"
        )

        with pytest.raises(HTTPException) as excinfo:
            _get_model_options()

        assert excinfo.value.status_code == 503


def _get_model_options():
    import asyncio

    import hermes_cli.web_routers.models as _rt_models

    return asyncio.run(_rt_models.get_model_options())
