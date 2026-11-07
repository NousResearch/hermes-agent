"""``PluginContext.redact_text``: the public way for a plugin to scrub text it stores or uploads."""

import pytest

import hermes_cli.plugins as plugins_mod
from agent import redact

KEY = "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6"
GH = "ghp_" + "abcdefghijklmnopqrstuvwxyz0123456789"


@pytest.fixture
def ctx():
    return plugins_mod.PluginContext(plugins_mod.PluginManifest(name="redact-text-test"), plugins_mod.PluginManager())


@pytest.fixture
def redaction(monkeypatch):
    monkeypatch.setattr(redact, "_redact_enabled", lambda: True)
    return monkeypatch


def test_prefixed_keys_become_sentinels_with_no_key_characters(ctx, redaction):
    out = ctx.redact_text(f"use {KEY} and {GH}")
    assert KEY not in out and GH not in out
    assert KEY[-6:] not in out and GH[-6:] not in out  # a head/tail mask would keep these
    assert "«redacted:" in out


def test_credential_assignments_are_masked(ctx, redaction):
    secret = "q8Zt2LmN4vX7pR1wK9sD3fH6"  # no known prefix: only the assignment shape gives it away
    out = ctx.redact_text(f"API_KEY={secret}\nexport DB_PASSWORD='{secret}'")
    assert secret not in out


def test_matches_the_host_scrubber_at_its_strongest_setting(ctx, redaction):
    text = f"key {KEY}\nTOKEN: abc123def456ghi789\nplain prose stays"
    assert ctx.redact_text(text) == redact.redact_sensitive_text(text, file_read=True, secret_file=True)
    assert "plain prose stays" in ctx.redact_text(text)


def test_user_opt_out_is_honoured_unless_forced(ctx, monkeypatch):
    monkeypatch.setattr(redact, "_redact_enabled", lambda: False)
    assert ctx.redact_text(f"x {KEY}") == f"x {KEY}"
    assert KEY not in ctx.redact_text(f"x {KEY}", force=True)


def test_plugin_registered_patterns_apply(ctx, redaction):
    token = "nvapi-" + "Zz9" * 14
    ctx.register_redaction_patterns([r"nvapi-[A-Za-z0-9_-]{20,}"])
    try:
        assert token not in ctx.redact_text(f"t {token}")
    finally:
        redact._reset_plugin_redaction_patterns()
