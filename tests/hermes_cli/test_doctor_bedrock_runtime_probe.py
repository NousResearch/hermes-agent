"""``hermes doctor`` must check the Bedrock endpoint inference actually uses (#87195).

The "AWS Bedrock" row calls ``bedrock:ListFoundationModels`` — the **control** plane. The reporter
of #87195 read that green tick as healthy while every request to the configured endpoint came back
401, because the Bedrock API Key route (``bedrock-mantle.<region>.api.aws``) authenticates with a
bearer token the control-plane call never touches.

These tests lock both halves: that the control-plane row now says which plane it checked, and that
the new row follows the config to the runtime endpoint and reports its refusal.
"""

from __future__ import annotations

import json
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer

import hermes_yaml as yaml
import pytest

from hermes_cli import doctor_connectivity as dc

# Measured verbatim from https://bedrock-mantle.us-east-1.api.aws/v1/models on 2026-09-25 — no auth
# header (x-amzn-RequestId req_dbpbb4byxg2q3rof2fwgbiz4b7df3raj7lyftrgojjkos2prgrlq) and a bad bearer
# token (req_q6kzumhrytxyn77vapjvsrulqch5humhe5fcfch72fdz7kun6qfq). us-east-2 answers identically.
_MISSING_HEADER = {"error": {"code": "invalid_api_key", "message": "Missing 'authorization' or 'x-api-key' header",
                             "param": None, "type": "permission_denied_error"}}
_INVALID_TOKEN = {"error": {"code": "invalid_api_key", "message": "Invalid bearer token",
                            "param": None, "type": "permission_denied_error"}}
_MODELS_OK = {"object": "list", "data": [{"id": "openai.gpt-5.6-terra", "object": "model"}]}

TOKEN = "test-bedrock-bearer-token"


@pytest.fixture
def mantle_stand_in(monkeypatch):
    """A local stand-in for the Bedrock API Key endpoint, with the measured bodies."""
    seen: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        status = 401

        def do_GET(self):  # noqa: N802 - http.server API
            auth = self.headers.get("Authorization") or ""
            seen.append({"path": self.path, "authorization": auth})
            status = self.status
            if status == 200:
                body = json.dumps(_MODELS_OK).encode()
            else:
                body = json.dumps(_INVALID_TOKEN if auth else _MISSING_HEADER).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002 - http.server API; quiet
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    # The production host test cannot match a loopback stand-in, which is why the host shape is a
    # module constant rather than inlined in _is_bedrock_runtime_url.
    monkeypatch.setattr(dc, "_BEDROCK_RUNTIME_HOST_PARTS", ("127.0.0.1", ""))
    yield Handler, seen, f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def _write_config(monkeypatch, tmp_path, cfg: dict, *, name: str = "home"):
    home = tmp_path / name
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _mantle_config(base_url: str) -> dict:
    """The shape ``hermes_cli/model_setup_flows_bedrock.py`` writes for the API-key route."""
    return {
        "model": {"provider": "custom:bedrock-mantle", "name": "openai.gpt-5.6-terra"},
        "providers": {"bedrock-mantle": {"base_url": base_url, "key_env": "AWS_BEARER_TOKEN_BEDROCK"}},
    }


def _runtime_row(monkeypatch, *, label: str = "AWS Bedrock runtime"):
    """Drive the production entry point: the doctor's probe table, run by ``run_probes``."""
    monkeypatch.setattr(dc, "_APIKEY_PROVIDERS_CACHE", [])
    probes = [(lbl, fn) for lbl, fn in dc.build_probes() if lbl == label]
    assert probes, f"{label!r} probe missing from the doctor's connectivity table"
    (result,) = dc.run_probes(probes)
    return result


def test_rejected_token_is_reported_and_the_control_plane_row_is_named(monkeypatch, tmp_path, mantle_stand_in):
    """The defect: a 401 on the configured endpoint must not be invisible behind a green control-plane row."""
    handler, seen, base = mantle_stand_in
    handler.status = 401
    _write_config(monkeypatch, tmp_path, _mantle_config(base))
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", TOKEN)

    result = _runtime_row(monkeypatch)
    ((glyph, _label, detail),) = result.lines
    assert "✗" in glyph
    assert "AWS_BEARER_TOKEN_BEDROCK" in detail and base in detail and "401" in detail
    # The issue line has to tell the user why the other Bedrock row disagrees.
    assert result.issues and "control plane" in result.issues[0]
    assert "AWS_BEARER_TOKEN_BEDROCK" in result.issues[0] and base in result.issues[0]
    # The real token went to the endpoint (not a synthetic verdict) and never leaks into the row text.
    assert seen == [{"path": "/v1/models", "authorization": f"Bearer {TOKEN}"}]
    assert TOKEN not in detail and TOKEN not in result.issues[0]


def test_working_endpoint_is_ok(monkeypatch, tmp_path, mantle_stand_in):
    handler, seen, base = mantle_stand_in
    handler.status = 200
    _write_config(monkeypatch, tmp_path, _mantle_config(base))
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", TOKEN)

    result = _runtime_row(monkeypatch)
    ((glyph, _label, detail),) = result.lines
    assert "✓" in glyph and "AWS_BEARER_TOKEN_BEDROCK" in detail and not result.issues
    assert len(seen) == 1


def test_configured_route_without_a_token_warns_and_makes_no_request(monkeypatch, tmp_path, mantle_stand_in):
    _handler, seen, base = mantle_stand_in
    _write_config(monkeypatch, tmp_path, _mantle_config(base))
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)

    result = _runtime_row(monkeypatch)
    ((glyph, _label, detail),) = result.lines
    assert "⚠" in glyph and "AWS_BEARER_TOKEN_BEDROCK" in detail
    assert result.issues and base in result.issues[0]
    assert seen == []  # no credential to send: no request


def test_a_non_bedrock_route_produces_no_row_and_no_request(monkeypatch, tmp_path, mantle_stand_in):
    """SigV4/Converse and every other provider must not see a new row — that route has no free list call."""
    _handler, seen, _base = mantle_stand_in
    _write_config(monkeypatch, tmp_path, {"model": {"provider": "anthropic", "name": "claude-opus-5"}})
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", TOKEN)

    result = _runtime_row(monkeypatch)
    assert result.lines == [] and result.issues == [] and seen == []


def test_an_unreachable_endpoint_warns_rather_than_raising(monkeypatch, tmp_path):
    """A connection error is not an auth verdict; the doctor must keep running."""
    monkeypatch.setattr(dc, "_BEDROCK_RUNTIME_HOST_PARTS", ("127.0.0.1", ""))
    # Port 1 is reserved and nothing listens on it, so httpx raises ConnectError.
    _write_config(monkeypatch, tmp_path, _mantle_config("http://127.0.0.1:1/v1"))
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", TOKEN)

    result = _runtime_row(monkeypatch)
    ((glyph, _label, _detail),) = result.lines
    assert "⚠" in glyph and result.issues == ["Check network connectivity"]


def test_the_wizards_own_config_is_recognized_as_the_runtime_route(monkeypatch):
    """Run the real Bedrock API-key setup flow and check the probe follows what it wrote.

    Hand-written config in the tests above proves the probe logic; this proves it matches the shape
    the product actually saves, including the ``bedrock-mantle.<region>.api.aws`` host and
    ``key_env``. No request is made — only the route resolution is under test.
    """
    import hermes_cli.auth as auth_mod
    from hermes_cli.model_setup_flows_bedrock import _model_flow_bedrock_api_key

    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", TOKEN)
    monkeypatch.setattr(auth_mod, "_prompt_model_selection", lambda *a, **k: "openai.gpt-5.6-terra")
    monkeypatch.setattr(auth_mod, "_save_model_choice", lambda *a, **k: None)
    monkeypatch.setattr(auth_mod, "deactivate_provider", lambda *a, **k: None)
    _model_flow_bedrock_api_key({}, "us-east-1")

    base, key_var = dc._configured_bedrock_runtime()
    assert base.startswith("https://bedrock-mantle.us-east-1.api.aws")
    assert key_var == "AWS_BEARER_TOKEN_BEDROCK"
    assert dc._is_bedrock_runtime_url(base)


@pytest.mark.parametrize("base_url", [
    "https://bedrock-mantle.us-east-1.api.aws/v1",
    "https://bedrock-mantle.ap-northeast-1.api.aws/v1",
])
def test_every_region_of_the_endpoint_is_recognized(base_url):
    assert dc._is_bedrock_runtime_url(base_url)


@pytest.mark.parametrize("base_url", [
    "", None, "not a url",
    "https://api.openai.com/v1",
    "https://bedrock-runtime.us-east-1.amazonaws.com",          # the SigV4 data plane, not this route
    "https://bedrock-mantle.us-east-1.api.aws.evil.example/v1",  # host suffix must match, not a prefix
    "https://evil.example/bedrock-mantle.us-east-1.api.aws",     # path, not host
])
def test_other_hosts_are_not_mistaken_for_the_endpoint(base_url):
    assert not dc._is_bedrock_runtime_url(base_url)


def test_a_bare_model_base_url_at_the_endpoint_is_also_followed(monkeypatch, tmp_path, mantle_stand_in):
    """``model.base_url`` with no ``providers:`` entry behind it still names the flow's key var."""
    handler, seen, base = mantle_stand_in
    handler.status = 401
    _write_config(monkeypatch, tmp_path, {"model": {"provider": "custom", "base_url": base}})
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", TOKEN)

    result = _runtime_row(monkeypatch)
    ((glyph, _label, detail),) = result.lines
    assert "✗" in glyph and "AWS_BEARER_TOKEN_BEDROCK" in detail
    assert seen and seen[0]["authorization"] == f"Bearer {TOKEN}"


def test_the_probe_table_covers_the_configured_endpoint(monkeypatch, tmp_path):
    """The gap itself, stated in terms that exist on ``main``: with the API-key route configured, the
    connectivity table used to hold exactly one Bedrock row, and it queried the control plane."""
    _write_config(monkeypatch, tmp_path, _mantle_config("https://bedrock-mantle.us-east-1.api.aws/v1"))
    monkeypatch.setattr(dc, "_APIKEY_PROVIDERS_CACHE", [])
    labels = [label for label, _fn in dc.build_probes()]
    assert [label for label in labels if label.startswith("AWS Bedrock")] == ["AWS Bedrock", "AWS Bedrock runtime"], labels


def test_the_control_plane_row_says_which_plane_it_checked(monkeypatch):
    """The other half of the fix: the existing row must stop reading as "Bedrock works"."""
    import agent.bedrock_adapter as ba

    monkeypatch.setattr(ba, "has_aws_credentials", lambda *a, **k: True)
    monkeypatch.setattr(ba, "resolve_aws_auth_env_var", lambda *a, **k: "AWS_ACCESS_KEY_ID")
    monkeypatch.setattr(ba, "resolve_bedrock_region", lambda *a, **k: "us-east-1")

    class _Client:
        def list_foundation_models(self):
            return {"modelSummaries": [{"modelId": "anthropic.claude-opus-5"}]}

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda *a, **k: _Client()
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    result = dc._probe_bedrock()
    ((glyph, _label, detail),) = result.lines
    assert "✓" in glyph and "control plane" in detail, detail
    assert "1 models" in detail and "AWS_ACCESS_KEY_ID" in detail
