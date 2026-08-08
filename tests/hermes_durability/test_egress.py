"""Egress guardrail boundary: redaction, obfuscation bypasses, middleware."""

import pytest

from hermes_durability.egress import (EgressBlocked, guard_outbound_text,
                                      normalize_for_detection)

GHP = "ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"


def test_plain_text_passes_through():
    assert guard_outbound_text("hello world", platform="telegram") == "hello world"


def test_secret_redacted_at_egress():
    out = guard_outbound_text(f"your token is {GHP}", platform="telegram")
    assert GHP not in out
    assert "your token is" in out


def test_ansi_glue_bypass_defeated():
    # secret body split by ANSI styling sequences hides it from a raw regex
    glued = f"ghp_AbCdEfGh\x1b[31mIjKlMnOpQrStUvWxYz\x1b[0m0123456789"
    out = guard_outbound_text(f"token {glued}", platform="telegram")
    assert "IjKlMnOpQrStUvWxYz" not in out


def test_zero_width_split_defeated():
    glued = "ghp_AbCdEfGh​IjKlMnOpQrStUvWxYz‍0123456789"
    out = guard_outbound_text(f"token {glued}", platform="telegram")
    assert "IjKlMnOpQrStUvWxYz" not in out


def test_fullwidth_forms_defeated():
    # full-width latin NFKC-folds back to ascii before detection
    fullwidth = "ｇｈｐ＿" + "ＡｂＣｄＥｆＧｈＩｊＫｌＭｎＯｐＱｒＳｔ" + "０１２３４５６７８９"
    out = guard_outbound_text(f"token {fullwidth}", platform="telegram")
    assert "ＡｂＣｄＥｆＧｈ" not in out


def test_normalize_for_detection():
    assert normalize_for_detection("a\x1b[1mb​c") == "abc"
    assert normalize_for_detection("ＡＢＣ") == "ABC"


def test_disabled_via_env(monkeypatch):
    monkeypatch.setenv("HERMES_EGRESS_GUARDRAIL", "false")
    text = f"token {GHP}"
    assert guard_outbound_text(text, platform="telegram") == text


def test_fail_closed_when_redactor_raises(monkeypatch):
    import agent.redact as redact_mod

    def boom(*a, **k):
        raise RuntimeError("redactor broke")

    monkeypatch.setattr(redact_mod, "redact_sensitive_text", boom)
    with pytest.raises(EgressBlocked):
        guard_outbound_text("anything", platform="telegram")


def _install_callbacks(monkeypatch, callbacks):
    import hermes_cli.middleware as mw
    import hermes_cli.plugins as plugins_mod

    monkeypatch.setattr(plugins_mod, "has_middleware",
                        lambda kind: kind == "outbound_message")
    monkeypatch.setattr(mw, "_get_middleware_callbacks",
                        lambda kind: list(callbacks))


def test_middleware_can_rewrite(monkeypatch):
    _install_callbacks(monkeypatch,
                       [lambda **kw: {"text": kw["text"] + " [checked]",
                                      "source": "test-plugin"}])
    out = guard_outbound_text("hello", platform="telegram")
    assert out == "hello [checked]"


def test_middleware_can_block(monkeypatch):
    _install_callbacks(monkeypatch,
                       [lambda **kw: {"action": "block",
                                      "reason": "policy says no"}])
    with pytest.raises(EgressBlocked, match="policy says no"):
        guard_outbound_text("hello", platform="telegram")


def test_middleware_chains_sequentially(monkeypatch):
    # Each callback must see the PREVIOUS callback's rewrite — last-writer-
    # wins over the original body would silently drop a security rewrite.
    _install_callbacks(monkeypatch, [
        lambda **kw: {"text": kw["text"].replace("PII", "[scrubbed]")},
        lambda **kw: {"text": kw["text"] + " -- footer"},
    ])
    out = guard_outbound_text("has PII inside", platform="telegram")
    assert out == "has [scrubbed] inside -- footer"


def test_middleware_raising_callback_skipped_fail_open(monkeypatch):
    def boom(**kw):
        raise RuntimeError("plugin bug")

    _install_callbacks(monkeypatch, [
        boom,
        lambda **kw: {"text": kw["text"] + " [ok]"},
    ])
    assert guard_outbound_text("hello", platform="telegram") == "hello [ok]"


def test_apply_middleware_false_skips_plugins(monkeypatch):
    _install_callbacks(monkeypatch,
                       [lambda **kw: {"text": kw["text"] + " [footer]"}])
    out = guard_outbound_text("hello", platform="telegram",
                              apply_middleware=False)
    assert out == "hello"


def test_outbound_message_is_valid_middleware_kind():
    from hermes_cli.middleware import (OUTBOUND_MESSAGE_MIDDLEWARE,
                                       VALID_MIDDLEWARE)

    assert OUTBOUND_MESSAGE_MIDDLEWARE in VALID_MIDDLEWARE


def test_apply_outbound_message_middleware_no_callbacks(monkeypatch):
    import hermes_cli.plugins as plugins_mod

    monkeypatch.setattr(plugins_mod, "has_middleware", lambda kind: False)
    from hermes_cli.middleware import apply_outbound_message_middleware

    result = apply_outbound_message_middleware("body", platform="slack")
    assert result.text == "body"
    assert not result.changed and not result.blocked
