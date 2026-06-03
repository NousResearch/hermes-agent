from agent.eval_lab.redaction import REDACTED, redact_secrets


def test_redacts_sensitive_keys_in_nested_structures_without_shape_loss():
    payload = {
        "safe": "keep me",
        "api_key": "sk-test-secret-value",
        "nested": {
            "Authorization": "Bearer abcdefghijklmnopqrstuvwxyz123456",
            "items": [
                {"cookie_value": "session=abc123"},
                {"token_count": 42},
                "plain text",
            ],
        },
    }

    redacted = redact_secrets(payload)

    assert redacted == {
        "safe": "keep me",
        "api_key": REDACTED,
        "nested": {
            "Authorization": REDACTED,
            "items": [
                {"cookie_value": REDACTED},
                {"token_count": REDACTED},
                "plain text",
            ],
        },
    }
    assert payload["api_key"] == "sk-test-secret-value"


def test_redacts_bearer_like_strings_inside_text():
    text = "Call Authorization: Bearer abcdef...3456 before continuing."

    assert redact_secrets(text) == "Call Authorization: [REDACTED] before continuing."


def test_redacts_final_output_secret_patterns():
    output = "Do not leak token: Bearer zzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"

    redacted = redact_secrets(output)

    assert "zzzzzzzzzz" not in redacted
    assert REDACTED in redacted
