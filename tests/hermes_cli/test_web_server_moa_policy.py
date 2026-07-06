"""Web-server policy tests for restricting MoA as a persisted default model."""

import pytest
from fastapi import HTTPException


def test_apply_main_model_assignment_rejects_moa_persistence(monkeypatch):
    from hermes_cli import web_server

    monkeypatch.setattr(web_server, "load_config", lambda: {})
    monkeypatch.setattr(web_server, "save_config", lambda cfg: None)

    with pytest.raises(HTTPException) as exc:
        web_server._apply_model_assignment_sync(
            scope="main",
            provider="moa",
            model="default",
            task="",
            base_url="",
            api_key="",
        )

    assert exc.value.status_code == 400
    assert "cannot be saved as the default model/provider" in str(exc.value.detail)
