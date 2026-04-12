from agent.openai_compat import create_async_openai_client, create_openai_client

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


def test_gemini_aistudio_key_uses_x_goog_api_key_header_only():
    client = create_openai_client(api_key="AIzaSy-test-key", base_url=GEMINI_BASE_URL)

    assert client.default_headers["x-goog-api-key"] == "AIzaSy-test-key"
    assert "Authorization" not in client.default_headers
    assert client.api_key == ""


def test_gemini_bearer_token_keeps_authorization_header():
    client = create_async_openai_client(api_key="AQ.Ab8-test-token", base_url=GEMINI_BASE_URL)

    assert client.default_headers["Authorization"] == "Bearer AQ.Ab8-test-token"
    assert "x-goog-api-key" not in client.default_headers
