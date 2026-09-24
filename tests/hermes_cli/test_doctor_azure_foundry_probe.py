"""``hermes doctor`` connectivity probe for Azure Foundry Anthropic-style endpoints (#66756).

The ``/anthropic`` route on Foundry has no ``GET /models``; a working deployment answered the generic
Bearer ``/models`` probe with HTTP 404. The probe must follow the runtime protocol instead:
``POST <base>/v1/messages`` with Bearer auth and the ``api-version`` query the Anthropic adapter sends.
"""

from __future__ import annotations

import httpx

from hermes_cli import doctor_connectivity as dc

_AZURE_BASE = "https://res.services.ai.azure.com/anthropic"


def _run_probe(monkeypatch, status: int, base_url_in_env: bool):
    calls: list = []

    def _post(url, headers=None, params=None, json=None, timeout=None):
        calls.append(("POST", url, headers, params, json))
        return httpx.Response(status, json={"type": "error", "error": {"type": "invalid_request_error", "message": "x"}}
                              if status == 400 else {"id": "msg_1"})

    def _get(url, headers=None, timeout=None):
        calls.append(("GET", url, headers, None, None))
        return httpx.Response(404)

    monkeypatch.setattr(httpx, "post", _post)
    monkeypatch.setattr(httpx, "get", _get)
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "k")
    if base_url_in_env:
        monkeypatch.setenv("AZURE_FOUNDRY_BASE_URL", _AZURE_BASE)
    else:
        monkeypatch.delenv("AZURE_FOUNDRY_BASE_URL", raising=False)
    monkeypatch.setattr(dc, "_model_cfg", lambda: {"provider": "azure-foundry", "base_url": _AZURE_BASE, "default": "claude-sonnet-5"})
    res = dc._probe_apikey_provider("Azure Foundry", ("AZURE_FOUNDRY_API_KEY", "AZURE_FOUNDRY_BASE_URL"), None,
                                    "AZURE_FOUNDRY_BASE_URL", True)
    return res, calls


def test_anthropic_style_foundry_probe_posts_messages_like_the_runtime(monkeypatch):
    """Healthy row, one POST /v1/messages with Bearer auth + api-version + max_tokens=1 — never GET /models."""
    for status in (200, 400):
        res, calls = _run_probe(monkeypatch, status, base_url_in_env=True)
        assert res.issues == [] and "\u2713" in res.lines[0][0], (status, res)
        assert [c[0] for c in calls] == ["POST"]
        _, url, headers, params, body = calls[0]
        assert url == _AZURE_BASE + "/v1/messages"
        assert headers["Authorization"] == "Bearer k" and "x-api-key" not in headers
        assert headers["anthropic-version"] == "2023-06-01"
        assert params == {"api-version": "2025-04-15"}
        assert body["max_tokens"] == 1 and body["model"] == "claude-sonnet-5"


def test_foundry_base_url_falls_back_to_config_and_404_stays_visible(monkeypatch):
    """Base URL only in ``model.base_url`` is still probed; a real 404 on /v1/messages is still reported."""
    res, calls = _run_probe(monkeypatch, 200, base_url_in_env=False)
    assert calls[0][1] == _AZURE_BASE + "/v1/messages" and res.issues == []
    res, _ = _run_probe(monkeypatch, 404, base_url_in_env=True)
    assert "HTTP 404" in res.lines[0][2]
