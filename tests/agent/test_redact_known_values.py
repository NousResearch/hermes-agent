"""The profile's own configured secret values are masked by exact value, per profile."""

import json
import logging
from urllib.parse import quote

import pytest

from agent.redact import RedactingFormatter, redact_sensitive_text
from agent.redact_known_values import MARKER, clear_known_secret_values
from hermes_cli.debug_redaction import redact_debug_support_text
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

# Opaque on purpose: no vendor prefix, so only the exact-value pass can know them.
ENV_VAL = "q7Wz2pLx9Rv4Kd8Nf3Ty"
AUTH_VAL = "Zm9vYmFyYmF6cXV4NjQ0Mg"
CFG_VAL = "plainConfigKey88231x"
OAUTH_VAL = "mcpOauthAccess5521zz"
HDR_VAL = "hdrBearerOpaque77zqp"
QUERY_VAL = "urlQueryKey/94+wx="
MCP_ENV_VAL = "mcpEnvOpaque!\"q\\31"
B_ENV_VAL = "profileBOnly9931kkd"


@pytest.fixture(autouse=True)
def _fresh_registry():
    clear_known_secret_values()
    yield
    clear_known_secret_values()


def _profile(root, name, env_lines):
    home = root / name
    (home / "mcp-tokens").mkdir(parents=True)
    (home / ".env").write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    return home


@pytest.fixture
def homes(tmp_path):
    a = _profile(tmp_path, "a", [f"ACME_SERVICE_TOKEN={ENV_VAL}", "SHORT_TOKEN=abc123", "LOG_LEVEL=debuginfotrace"])
    (a / "auth.json").write_text(json.dumps({"providers": {"x": {
        "access_token": AUTH_VAL, "agent_key_expires_at": "2026-10-25T00:00:00Z"}}}), encoding="utf-8")
    (a / "config.yaml").write_text(
        "model:\n"
        f"  api_key: {CFG_VAL}\n"
        "mcp_servers:\n"
        "  srv:\n"
        f"    url: https://mcp.example.test/mcp?apikey={quote(QUERY_VAL, safe='')}\n"
        f"    headers: {{Authorization: 'Bearer {HDR_VAL}'}}\n"
        f"    env: {{SRV_SECRET: {json.dumps(MCP_ENV_VAL)}, NODE_ENV: productionmode}}\n",
        encoding="utf-8")
    (a / "mcp-tokens" / "srv.json").write_text(json.dumps({"access_token": OAUTH_VAL}), encoding="utf-8")
    b = _profile(tmp_path, "b", [f"OTHER_API_KEY={B_ENV_VAL}"])
    return a, b


def _surfaces(text):
    record = logging.LogRecord("t", logging.INFO, __file__, 1, text, None, None)
    return {
        "redactor": redact_sensitive_text(text, force=True),
        "log": RedactingFormatter("%(message)s").format(record),
        "debug share": redact_debug_support_text(text),
    }


def test_configured_values_masked_in_every_form_and_scoped_to_their_profile(homes):
    a, b = homes
    secrets = [ENV_VAL, AUTH_VAL, CFG_VAL, OAUTH_VAL, HDR_VAL, QUERY_VAL, MCP_ENV_VAL]
    leaked = "\n".join(
        [f"provider said: {v}" for v in secrets]
        + [f'{{"echo": {json.dumps(MCP_ENV_VAL)}}}', f"GET /cb?k={quote(QUERY_VAL, safe='')}"]
        + [f"note {B_ENV_VAL}", "short abc123 expires 2026-10-25T00:00:00Z debuginfotrace productionmode"])
    token = set_hermes_home_override(a)
    try:
        for surface, out in _surfaces(leaked).items():
            for value in secrets:
                assert value not in out, (surface, value)
            assert json.dumps(MCP_ENV_VAL)[1:-1] not in out, surface
            assert quote(QUERY_VAL, safe="") not in out, surface
            assert MARKER in out
            # Short values, timestamps and non-credential names are not registered.
            assert "short abc123 expires 2026-10-25T00:00:00Z debuginfotrace productionmode" in out
            # Another profile's value is neither masked nor confirmed here.
            assert f"note {B_ENV_VAL}" in out
    finally:
        reset_hermes_home_override(token)
    token = set_hermes_home_override(b)
    try:
        out = redact_sensitive_text(leaked, force=True)
        assert B_ENV_VAL not in out
        assert f"provider said: {ENV_VAL}" in out and f"provider said: {CFG_VAL}" in out
    finally:
        reset_hermes_home_override(token)
