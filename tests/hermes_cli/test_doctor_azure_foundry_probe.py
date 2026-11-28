"""``hermes doctor`` connectivity probe for Azure Foundry Anthropic-style endpoints (#66756).

The ``/anthropic`` route on Foundry has no ``GET /models``; a working deployment answered the generic
Bearer ``/models`` probe with HTTP 404. The probe must follow the runtime protocol instead:
``POST <base>/v1/messages`` with Bearer auth and the ``api-version`` query the Anthropic adapter sends.
"""

from __future__ import annotations

import json

import httpx
import pytest

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


@pytest.mark.parametrize(
    "model_cfg",
    [{}, {"default": None}, {"default": ""}, {"default": "  "}],
    ids=["absent", "null", "empty", "blank"],
)
def test_messages_probe_without_a_configured_model_never_sends_a_retired_one(
    monkeypatch, tmp_path, model_cfg
):
    """An unset deployment must not select Sonnet 4.5, which retires on 2026-11-30."""
    bodies: list = []

    def _post(url, headers=None, params=None, json=None, timeout=None):
        bodies.append(json)
        assert isinstance(json, dict)
        if json["model"] in {"claude-sonnet-4-5", "claude-sonnet-4-5-20250929"}:
            return httpx.Response(404)
        return httpx.Response(200, json={"content": [{"type": "thinking"}]})

    monkeypatch.setattr(httpx, "post", _post)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "k")
    monkeypatch.delenv("AZURE_FOUNDRY_BASE_URL", raising=False)
    (tmp_path / "config.yaml").write_text(
        json.dumps(
            {
                "model": {
                    "provider": "azure-foundry",
                    "base_url": _AZURE_BASE,
                    **model_cfg,
                }
            }
        ),
        encoding="utf-8",
    )
    res = dc._probe_apikey_provider(
        "Azure Foundry",
        ("AZURE_FOUNDRY_API_KEY",),
        None,
        "AZURE_FOUNDRY_BASE_URL",
        True,
    )
    assert "\u2713" in res.lines[0][0] and res.issues == []
    model = bodies[0]["model"]
    assert model.startswith("claude-") and not model.startswith("claude-sonnet-4-5")
    assert bodies == [
        {
            "model": model,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "ping"}],
        }
    ]


@pytest.mark.parametrize(
    "deployment",
    [
        "my-sonnet-deployment",
        "  my-sonnet-deployment  ",
        "claude-sonnet-4-5",
        "claude-sonnet-4-5-20250929",
    ],
)
def test_messages_probe_preserves_configured_deployment_and_its_failure(
    monkeypatch, tmp_path, deployment
):
    """Deployment names are user-defined; a missing one must remain visible to doctor."""
    bodies: list = []

    def _post(url, headers=None, params=None, json=None, timeout=None):
        bodies.append(json)
        return httpx.Response(404)

    monkeypatch.setattr(httpx, "post", _post)
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "k")
    monkeypatch.delenv("AZURE_FOUNDRY_BASE_URL", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        json.dumps(
            {
                "model": {
                    "provider": "azure-foundry",
                    "base_url": _AZURE_BASE,
                    "default": deployment,
                }
            }
        ),
        encoding="utf-8",
    )
    res = dc._probe_apikey_provider(
        "Azure Foundry",
        ("AZURE_FOUNDRY_API_KEY",),
        None,
        "AZURE_FOUNDRY_BASE_URL",
        True,
    )
    assert [body["model"] for body in bodies] == [deployment.strip()]
    assert "HTTP 404" in res.lines[0][2]
