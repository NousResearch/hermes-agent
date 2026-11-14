"""Regression for #81073: egress Bearer residue must use ASCII token edges."""

from agent.redact import redact_for_egress


def test_egress_redacts_latin_letter_glued_bearer_residue():
    """A Latin letter outside [A-Za-z0-9_] is still ``\\w``, so ``\\b`` misses it."""
    glue = "\u00e9"
    token = "g" * 24
    secret = f"Bearer {token}"
    raw = f"before{glue}{secret} after"

    out = redact_for_egress(raw)

    assert token not in out
    assert secret not in out
    assert out.startswith(f"before{glue}")
    assert out.endswith(" after")
    assert "Bearer [redacted]" in out


def test_egress_preserves_ascii_embedded_bearer_shape():
    """``useBearer`` stays one ASCII word and must not widen into a residue match."""
    token = "g" * 24
    text = f"useBearer {token} suffix"

    assert redact_for_egress(text) == text


def test_egress_redacts_unicode_glued_prefix_tokens():
    """The prefix half of #81073: a CJK / fullwidth-digit neighbour must not hide ``sk-…``."""
    token = "sk-" + "a" * 20
    for glue in ("xx\u4e2d", "xx\uff11", "\u8ba4\u8bc1"):
        assert token not in redact_for_egress(glue + token)


def test_egress_redacts_cjk_glued_bearer_residue():
    token = "g" * 24
    out = redact_for_egress(f"\u8ba4\u8bc1Bearer {token}")
    assert token not in out
