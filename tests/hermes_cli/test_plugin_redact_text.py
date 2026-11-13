"""``PluginContext.redact_text``: the public way for a plugin to scrub text it stores or uploads."""

import pytest

import hermes_agent.hermes_cli.plugins as plugins_mod
from hermes_agent.agent import redact

KEY = "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6"
GH = "ghp_" + "abcdefghijklmnopqrstuvwxyz0123456789"


@pytest.fixture
def ctx():
    return plugins_mod.PluginContext(plugins_mod.PluginManifest(name="redact-text-test"), plugins_mod.PluginManager())


def test_keys_assignments_and_plugin_patterns_leave_no_secret_characters(ctx, monkeypatch):
    monkeypatch.setattr(redact, "_redact_enabled", lambda: True)
    bare = "q8Zt2LmN4vX7pR1wK9sD3fH6"  # no known prefix: only the assignment shape gives it away
    token = "nvapi-" + "Zz9" * 14
    link_secret = "REL" + "x7Kq2Wm9Pz4Tn8Vb"  # opaque: only the credential-named query param gives it away
    ctx.register_redaction_patterns([r"nvapi-[A-Za-z0-9_-]{20,}"])
    try:
        out = ctx.redact_text(f"use {KEY} and {GH}\nAPI_KEY={bare}\nt {token}\nplain prose stays\n"
                              f"link https://example.com/resume?token={link_secret}&view=public")
    finally:
        redact._reset_plugin_redaction_patterns()
    for secret in (KEY, GH, bare, token, link_secret):
        assert secret not in out
    assert KEY[-6:] not in out and GH[-6:] not in out  # a head/tail mask would keep these
    assert "«redacted:" in out and "plain prose stays" in out


def test_user_opt_out_is_honoured_unless_forced(ctx, monkeypatch):
    monkeypatch.setattr(redact, "_redact_enabled", lambda: False)
    assert ctx.redact_text(f"x {KEY}") == f"x {KEY}"
    assert KEY not in ctx.redact_text(f"x {KEY}", force=True)
