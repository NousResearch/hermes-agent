"""Outbound webhook payloads must not carry credentials off-box.

``tool_input`` and the per-event extras contain raw tool arguments and tool
output — ``export OPENAI_API_KEY=...``, ``curl -H "Authorization: ..."``, the
body of a written ``.env``. The terminal path masks exactly this material
before the model or the session DB sees it. Without the same pass here an
outbound delivery ships it verbatim to a receiver outside the trust boundary,
in cleartext when the configured URL is plain ``http://``.

The envelope must survive redaction: masking the serialised JSON text can
break a quoted value, and a receiver that cannot parse the delivery has traded
a leak for an outage.

All tokens are synthesised.
"""

import json
import secrets

import pytest

from agent.outbound_webhooks import _serialize_payload

TOKEN = "sk-" + secrets.token_hex(24)
BEARER = "Bearer " + secrets.token_hex(20)


def _body(kwargs, event="pre_tool_call"):
    return _serialize_payload(event, kwargs, "delivery-test").decode("utf-8")


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"tool_name": "terminal",
                      "args": {"command": f"export OPENAI_API_KEY={TOKEN}"}},
                     id="env-assignment"),
        pytest.param({"tool_name": "terminal",
                      "args": {"command": f'export K="{TOKEN}"'}},
                     id="quoted-value"),
        pytest.param({"tool_name": "write_file",
                      "args": {"path": "/tmp/.env",
                               "content": f"A=1\nKEY={TOKEN}\nB=2"}},
                     id="multiline-file-body"),
        pytest.param({"tool_name": "terminal", "args": {"command": "ls"},
                      "result": {"env": {"OPENAI_API_KEY": TOKEN}}},
                     id="nested-in-result"),
        pytest.param({"tool_name": "write_file",
                      "args": {"path": "/tmp/c.json",
                               "content": json.dumps({"api_key": TOKEN})}},
                     id="json-inside-json"),
        pytest.param({"tool_name": "terminal",
                      "args": {"command": f"echo € {TOKEN} ✓"}},
                     id="unicode-neighbours"),
    ],
)
def test_credentials_do_not_reach_the_wire(kwargs):
    body = _body(kwargs)
    assert TOKEN not in body


def test_authorization_header_is_masked_without_breaking_json():
    """The bearer shape is the one that corrupted a whole-body redaction."""
    body = _body({"tool_name": "terminal",
                  "args": {"command": f'curl -H "Authorization: {BEARER}" https://a/b'}})
    assert BEARER.split()[-1] not in body
    json.loads(body)  # must still parse


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tool_name": "terminal", "args": {"command": f"export K={TOKEN}"}},
        {"tool_name": "read_file", "args": {"path": "/tmp/notes.md"}},
        {"tool_name": "terminal", "args": {}},
        {"tool_name": "terminal", "args": {"command": "ls"},
         "result": {"deep": {"deeper": {"deepest": TOKEN}}}},
    ],
)
def test_envelope_stays_valid_json(kwargs):
    parsed = json.loads(_body(kwargs))
    for key in ("hook_event_name", "tool_name", "tool_input", "session_id",
                "cwd", "extra", "delivery_id", "timestamp"):
        assert key in parsed


def test_non_secret_content_is_preserved():
    """Redaction must not mangle ordinary tool arguments."""
    body = _body({"tool_name": "terminal",
                  "args": {"command": "git log --oneline -5"}})
    assert "git log --oneline -5" in body
    body = _body({"tool_name": "read_file", "args": {"path": "/tmp/notes.md"}})
    assert "/tmp/notes.md" in body


def test_signature_covers_the_redacted_body():
    """The HMAC must be computed over what is actually sent."""
    import hashlib
    import hmac as hmac_mod

    from agent.outbound_webhooks import _build_delivery, iter_configured_targets

    secret = "test-signing-secret"
    target = next(iter(iter_configured_targets(
        {"hooks": {"outbound": [{"url": "https://example.invalid/hook",
                                 "events": ["pre_tool_call"],
                                 "secret": secret}]}})))
    body = _serialize_payload(
        "pre_tool_call",
        {"tool_name": "terminal", "args": {"command": f"export K={TOKEN}"}},
        "delivery-test",
    )
    delivery = _build_delivery("pre_tool_call", target, body, "delivery-test")

    assert delivery["body"] == body
    assert TOKEN not in body.decode("utf-8")
    expected = hmac_mod.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    assert delivery["headers"]["X-Hermes-Signature-256"] == f"sha256={expected}"
