import pytest

from hermes_cli import models as models_mod
from hermes_cli.models_validate import _Request, _validate_custom


@pytest.mark.parametrize("api_mode", ["codex_responses", "codex_app_server"])
def test_codex_modes_accept_without_listing(monkeypatch, api_mode):
    monkeypatch.setattr(models_mod, "probe_api_models",
                        lambda *a, **k: {"models": None, "probed_url": "https://example.invalid/v1/models"})
    req = _Request(requested="my-model", lookup="my-model", provider="custom", normalized="custom",
                   api_key="x", base_url="https://example.invalid/v1", api_mode=api_mode, headers=None)
    verdict = _validate_custom(req)
    assert verdict["accepted"] is True
    assert "without verification" in verdict["message"]
