"""Behaviour of the terraform-enterprise helper against a fake TFE API (no network)."""

from __future__ import annotations

import importlib.util
import io
import json
import urllib.error
from pathlib import Path

import pytest

SCRIPT_PATH = (Path(__file__).resolve().parents[2] / "optional-skills" / "devops"
               / "terraform-enterprise" / "scripts" / "tfe_api.py")


@pytest.fixture
def tfe(monkeypatch):
    spec = importlib.util.spec_from_file_location("tfe_api_skill", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("TFE_TOKEN", "TFE_ADDRESS", "TFE_HOSTNAME", "TFE_ORGANIZATION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TFE_TOKEN", "test-token")
    return mod


class FakeApi:
    """Records requests and answers from a {(method, path-without-query): payload} table."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __call__(self, req, timeout=None):
        path, _, query = req.full_url.partition("?")
        self.calls.append({"method": req.get_method(), "url": req.full_url, "query": query,
                           "auth": req.get_header("Authorization"),
                           "body": json.loads(req.data) if req.data else None})
        payload = self.routes[(req.get_method(), path)]
        if isinstance(payload, list):
            payload = payload.pop(0)
        if isinstance(payload, urllib.error.HTTPError):
            raise payload
        return io.BytesIO(json.dumps(payload).encode())


def _install(monkeypatch, tfe, routes):
    api = FakeApi(routes)
    monkeypatch.setattr(tfe.urllib.request, "urlopen", api)
    return api


@pytest.mark.parametrize("env,expected", [
    ({}, "https://app.terraform.io/api/v2"),
    ({"TFE_ADDRESS": "https://tfe.example.com/"}, "https://tfe.example.com/api/v2"),
    ({"TFE_ADDRESS": "tfe.example.com"}, "https://tfe.example.com/api/v2"),
    ({"TFE_HOSTNAME": "tfe.internal"}, "https://tfe.internal/api/v2"),
])
def test_base_url_honours_standard_tfe_variables(tfe, env, expected):
    assert tfe.base_url(env) == expected


def test_missing_token_fails_with_a_named_variable(tfe, monkeypatch, capsys):
    monkeypatch.delenv("TFE_TOKEN")
    assert tfe.main(["whoami"]) == 1
    assert "TFE_TOKEN" in capsys.readouterr().err


def test_workspace_commands_need_an_organization(tfe, capsys):
    assert tfe.main(["workspaces"]) == 1
    assert "TFE_ORGANIZATION" in capsys.readouterr().err


def test_workspaces_paginate_and_send_bearer_token(tfe, monkeypatch, capsys):
    monkeypatch.setenv("TFE_ADDRESS", "https://tfe.example.com")
    monkeypatch.setenv("TFE_ORGANIZATION", "acme")
    page = lambda ids, nxt: {"data": [{"id": i, "attributes": {"name": i}} for i in ids],
                             "meta": {"pagination": {"next-page": nxt}}}
    api = _install(monkeypatch, tfe, {
        ("GET", "https://tfe.example.com/api/v2/organizations/acme/workspaces"):
            [page(["ws-1", "ws-2"], 2), page(["ws-3"], None)]})

    assert tfe.main(["workspaces", "--search", "prod"]) == 0

    names = [w["name"] for w in json.loads(capsys.readouterr().out)]
    assert names == ["ws-1", "ws-2", "ws-3"]
    assert all(c["auth"] == "Bearer test-token" for c in api.calls)
    assert "search%5Bname%5D=prod" in api.calls[0]["query"]
    assert "page%5Bnumber%5D=2" in api.calls[1]["query"]


def test_start_run_resolves_workspace_name_and_posts_run(tfe, monkeypatch, capsys):
    base = "https://app.terraform.io/api/v2"
    api = _install(monkeypatch, tfe, {
        ("GET", f"{base}/organizations/acme/workspaces/network"): {"data": {"id": "ws-abc"}},
        ("POST", f"{base}/runs"): {"data": {"id": "run-1", "attributes": {"status": "pending"}}}})

    assert tfe.main(["start-run", "network", "--org", "acme", "--plan-only"]) == 0

    assert json.loads(capsys.readouterr().out)["id"] == "run-1"
    body = api.calls[-1]["body"]["data"]
    assert body["relationships"]["workspace"]["data"]["id"] == "ws-abc"
    assert body["attributes"]["plan-only"] is True


def test_api_errors_surface_the_jsonapi_detail(tfe, monkeypatch, capsys):
    url = "https://app.terraform.io/api/v2/runs/run-1/actions/apply"
    err = urllib.error.HTTPError(url, 409, "Conflict", {}, io.BytesIO(json.dumps(
        {"errors": [{"status": "409", "title": "conflict",
                     "detail": "transition not allowed"}]}).encode()))
    _install(monkeypatch, tfe, {("POST", url): err})

    assert tfe.main(["apply", "run-1"]) == 1
    assert "HTTP 409 transition not allowed" in capsys.readouterr().err
