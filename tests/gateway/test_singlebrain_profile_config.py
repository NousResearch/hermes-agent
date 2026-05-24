"""Local Single Brain beta profile wiring checks.

These tests validate only local config/source behavior. They do not start or
restart the gateway, post to Slack, read vault data, or perform external writes.
"""

from pathlib import Path

import pytest
import yaml

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


REPO_ROOT = Path(__file__).resolve().parents[2]
SINGLEBRAIN_PROFILE = REPO_ROOT.parent / "profiles" / "single-brain"
CONFIG_PATH = SINGLEBRAIN_PROFILE / "config.yaml"

EXPECTED_ALLOWED = {
    "GET /health",
    "GET /health/detailed",
    "GET /v1/models",
    "GET /v1/capabilities",
}
EXPECTED_DENIED = {
    "POST /v1/chat/completions",
    "POST /v1/responses",
    "POST /v1/runs",
    "POST /v1/runs/*/approval",
    "POST /v1/runs/*/stop",
    "DELETE /v1/responses/*",
    "POST /api/jobs",
    "PATCH /api/jobs/*",
    "DELETE /api/jobs/*",
    "POST /api/jobs/*/pause",
    "POST /api/jobs/*/resume",
    "POST /api/jobs/*/run",
}


def _singlebrain_api_server_block() -> dict:
    if not CONFIG_PATH.exists():
        pytest.skip(f"Single Brain profile config not present in this checkout: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return ((cfg.get("platforms") or {}).get("api_server") or {})


def test_singlebrain_profile_wires_api_server_readonly_report_only_policy():
    api_server = _singlebrain_api_server_block()
    policy = ((api_server.get("extra") or {}).get("access_policy") or {})

    assert api_server.get("enabled") is True
    assert (api_server.get("extra") or {}).get("host") == "127.0.0.1"
    assert (api_server.get("extra") or {}).get("port") == 8642
    assert policy["name"] == "singlebrain-ambient-sg-beta"
    assert policy["mode"] == "read_only_report_only"
    assert set(policy["allowed_endpoints"]) == EXPECTED_ALLOWED
    assert set(policy["denied_endpoints"]) == EXPECTED_DENIED
    assert policy["report_path"] == "artifacts/singlebrain-gateway-api-access-report.jsonl"
    assert policy["external_writes_enabled"] is False
    assert policy["live_slack_posting_enabled"] is False
    assert policy["gateway_restart_enabled"] is False


def test_singlebrain_profile_policy_normalizes_and_blocks_write_endpoints():
    api_server = _singlebrain_api_server_block()
    adapter = APIServerAdapter(
        PlatformConfig(
            enabled=api_server.get("enabled", False),
            extra=api_server.get("extra") or {},
        )
    )

    advertised = adapter._access_policy_for_capabilities()
    assert advertised is not None
    assert advertised["mode"] == "read_only_report_only"
    assert "POST /v1/runs" in advertised["denied_endpoints"]

    denied = adapter._read_only_policy_denial("POST", "/v1/runs")
    assert denied is not None
    assert denied.status == 403

    wildcard_denied = adapter._read_only_policy_denial("POST", "/v1/runs/run_123/approval")
    assert wildcard_denied is not None
    assert wildcard_denied.status == 403

    allowed_probe = adapter._read_only_policy_denial("GET", "/v1/capabilities")
    assert allowed_probe is None
