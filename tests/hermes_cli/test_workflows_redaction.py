from hermes_cli.workflows_redaction import redact_sensitive


def test_redact_sensitive_recurses_through_dicts_and_lists() -> None:
    value = {
        "api_key": "abc",
        "nested": [{"Authorization": "Bearer token", "safe": "visible"}],
        "password_hint": "secret",
    }

    assert redact_sensitive(value) == {
        "api_key": "[REDACTED]",
        "nested": [{"Authorization": "[REDACTED]", "safe": "visible"}],
        "password_hint": "[REDACTED]",
    }


def test_redact_sensitive_leaves_scalars_and_safe_keys() -> None:
    assert redact_sensitive({"topic": "ui dogfood", "score": 0.95}) == {
        "topic": "ui dogfood",
        "score": 0.95,
    }


def test_redact_sensitive_keeps_safe_diagnostic_text_readable() -> None:
    assert redact_sensitive("token count exceeded while parsing") == "token count exceeded while parsing"
