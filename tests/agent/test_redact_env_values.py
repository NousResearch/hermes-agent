"""The profile's own ``.env`` secret values are masked by exact value, per profile.

An opaque key with no vendor prefix, printed without its variable name, passes every shape rule;
only the ``.env`` itself says those bytes are secret.
"""

import json
import logging

import pytest

from agent.redact import RedactingFormatter, redact_for_egress, redact_sensitive_text
from agent.redact_env_values import MARKER, clear_profile_env_secret_cache
from hermes_cli.debug_redaction import redact_debug_support_text
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

# Opaque on purpose: no vendor prefix and no label, so only the exact-value pass can know them.
OPAQUE = "q7Wz2pLx9Rv4Kd8Nf3Ty"
PASSPHRASE = "correct horse battery staple"
B_ONLY = "profileBOnly9931kkdQ"


@pytest.fixture(autouse=True)
def _fresh():
    clear_profile_env_secret_cache()
    yield
    clear_profile_env_secret_cache()


def _home(root, name, lines):
    home = root / name
    home.mkdir()
    (home / ".env").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return home


def _in(home):
    class _Scope:
        def __enter__(self):
            self.token = set_hermes_home_override(str(home))

        def __exit__(self, *exc):
            reset_hermes_home_override(self.token)
            clear_profile_env_secret_cache()
    return _Scope()


def test_env_values_masked_on_every_surface_in_their_profile_only(tmp_path):
    a = _home(tmp_path, "a", [f"ACME_SERVICE_TOKEN={OPAQUE}", f"DB_PASSWORD='{PASSPHRASE}'",
                              "SHORT_TOKEN=abc123", "LOG_LEVEL=debuginfotrace", "AUTH_MODE=passwordless"])
    b = _home(tmp_path, "b", [f"OTHER_API_KEY={B_ONLY}"])
    leak = (f"provider said: bad key {OPAQUE}; login {PASSPHRASE}; json {json.dumps({'k': OPAQUE})} "
            f"short abc123 level debuginfotrace mode passwordless other {B_ONLY}")
    record = logging.LogRecord("t", logging.INFO, __file__, 1, leak, None, None)

    with _in(a):
        for out in (redact_sensitive_text(leak), redact_for_egress(leak),
                    RedactingFormatter("%(message)s").format(record), redact_debug_support_text(leak)):
            assert OPAQUE not in out and PASSPHRASE not in out and MARKER in out
            # Short values and ordinary config are never global scrubs; B's value is not A's to mask.
            assert "abc123" in out and "debuginfotrace" in out and "passwordless" in out
            assert B_ONLY in out
    with _in(b):
        out = redact_sensitive_text(leak)
        assert B_ONLY not in out and OPAQUE in out


def test_env_edit_is_picked_up_and_redact_secrets_off_is_honoured(tmp_path, monkeypatch):
    a = _home(tmp_path, "a", [f"ACME_SERVICE_TOKEN={OPAQUE}"])
    with _in(a):
        assert OPAQUE not in redact_sensitive_text(f"x {OPAQUE}")
    (a / ".env").write_text(f"ACME_SERVICE_TOKEN={B_ONLY}\n", encoding="utf-8")
    with _in(a):
        assert redact_sensitive_text(f"x {OPAQUE} {B_ONLY}") == f"x {OPAQUE} {MARKER}"
        monkeypatch.setattr("agent.redact._redact_enabled", lambda: False)
        assert redact_sensitive_text(f"x {B_ONLY}") == f"x {B_ONLY}"
        assert B_ONLY not in redact_sensitive_text(f"x {B_ONLY}", force=True)


def test_streamed_reply_never_shows_part_of_an_env_value(tmp_path):
    from gateway.stream_consumer_redact import redact_settled

    a = _home(tmp_path, "a", [f"ACME_SERVICE_TOKEN={OPAQUE}"])
    reply = f"the key is {OPAQUE} as configured"
    with _in(a):
        prefix = ""
        for i in range(1, len(reply) + 1):
            text, settled = redact_settled(reply[:i], prefix, done=False)
            prefix = text[:settled]
            assert not any(OPAQUE[:k] in prefix for k in range(4, len(OPAQUE) + 1)), prefix
        final, _ = redact_settled(reply, prefix, done=True)
    assert final == f"the key is {MARKER} as configured"


def test_streamed_passphrase_never_shows_a_word_before_it_completes(tmp_path):
    from gateway.stream_consumer_redact import redact_settled

    a = _home(tmp_path, "a", [f"DB_PASSWORD='{PASSPHRASE}'"])
    reply = f"login with {PASSPHRASE} now"
    with _in(a):
        prefix = ""
        for i in range(1, len(reply) + 1):
            text, settled = redact_settled(reply[:i], prefix, done=False)
            prefix = text[:settled]
            assert "correct" not in prefix, prefix


def test_values_are_masked_as_the_runtime_resolves_them(tmp_path, monkeypatch):
    # ${VAR:-default} expands, a quoted multi-line value is one value, and a shorter secret-source
    # value inside a longer .env value never splits it.
    from agent import redact_secret_sources
    from agent.tool_dispatch_helpers import make_tool_result_message

    monkeypatch.delenv("ACME_UNSET", raising=False)
    monkeypatch.setattr(redact_secret_sources, "_VALUES_BY_HOME", {})
    pem_body = "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7"
    a = _home(tmp_path, "a", ['ACME_SERVICE_TOKEN="${ACME_UNSET:-' + OPAQUE + '}"',
                              f'SVC_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n{pem_body}\n-----END PRIVATE KEY-----"'])
    redact_secret_sources.publish_secret_source_values(a, {"VENDOR_TOKEN": OPAQUE[:11] + "Q"})
    redact_secret_sources.publish_secret_source_values(a, {"VENDOR_TOKEN": OPAQUE[:12]})
    with _in(a):
        out = redact_sensitive_text(f"tok {OPAQUE} key line {pem_body}")
        assert out == f"tok {MARKER} key line {MARKER}"
        # MCP / plugin results never pass redact_sensitive_text: the tool-result boundary masks them.
        msg = make_tool_result_message("mcp_srv_echo", [{"type": "text", "text": f"env {OPAQUE}"}], "c1")
        assert msg["content"][0]["text"] == f"env {MARKER}"
