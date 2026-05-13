"""Regression tests for Gemini API-key probing in hermes doctor."""


def test_gemini_models_probe_uses_key_query_param_not_bearer_header():
    """Regression for #25108: Google Generative Language /models uses ?key=."""
    from hermes_cli.doctor import _build_models_probe_request

    url, headers = _build_models_probe_request(
        "https://generativelanguage.googleapis.com/v1beta/models",
        "AIza-test-key",
    )

    assert url == "https://generativelanguage.googleapis.com/v1beta/models?key=AIza-test-key"
    assert "Authorization" not in headers
    assert headers["User-Agent"]


def test_non_gemini_models_probe_keeps_bearer_header():
    from hermes_cli.doctor import _build_models_probe_request

    url, headers = _build_models_probe_request(
        "https://api.deepseek.com/v1/models",
        "sk-test",
    )

    assert url == "https://api.deepseek.com/v1/models"
    assert headers["Authorization"] == "Bearer sk-test"



def test_gemini_models_probe_preserves_existing_query_params_and_encodes_key():
    from hermes_cli.doctor import _build_models_probe_request

    url, headers = _build_models_probe_request(
        "https://generativelanguage.googleapis.com/v1beta/models?alt=json",
        "AIza-test+key&bad=value",
    )

    assert url == "https://generativelanguage.googleapis.com/v1beta/models?alt=json&key=AIza-test%2Bkey%26bad%3Dvalue"
    assert "Authorization" not in headers
