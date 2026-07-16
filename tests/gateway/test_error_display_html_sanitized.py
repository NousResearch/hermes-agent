"""Regression: gateway error display must never forward raw HTML to chat.

The agent layer summarizes API errors before they reach the result dict's
``error`` field (``_summarize_api_error``), but the gateway historically
forwarded whatever landed there straight into the platform message.  These
tests pin the defense-in-depth contract on
``_normalize_empty_agent_response``: even if an agent-layer path misses
summarization, a provider/proxy HTML error page (e.g. a Cloudflare
challenge) is collapsed to a one-liner before it is sent to Discord,
Telegram, etc.
"""

from gateway.run import _normalize_empty_agent_response, _sanitize_error_text


_HTML_ERROR_PAGE = (
    "<!DOCTYPE html><html><head>"
    "<title>chatgpt.com | 502: Bad gateway</title></head>"
    "<body><div id='cf-error'>"
    + ("<p>" + "x" * 2000 + "</p>")
    + "Cloudflare</div></body></html>"
)

_HTML_NO_TITLE = (
    "<!DOCTYPE html>\n<html>\n<body><script>window._cf_chl_opt={};</script>"
    "</body>\n</html>\n"
)


class TestSanitizeErrorText:
    def test_html_page_collapses_to_title(self):
        text = _sanitize_error_text(_HTML_ERROR_PAGE)
        assert "<html" not in text.lower()
        assert text == "chatgpt.com | 502: Bad gateway"

    def test_html_page_without_title_gets_placeholder(self):
        text = _sanitize_error_text(_HTML_NO_TITLE)
        assert "<html" not in text.lower()
        assert text == "HTML error page from provider"

    def test_plain_error_passes_through(self):
        assert _sanitize_error_text("500 Internal Server Error") == (
            "500 Internal Server Error"
        )


class TestGatewayErrorDisplayHtmlSanitized:
    def test_failed_result_with_html_error_is_sanitized(self):
        agent_result = {
            "final_response": None,
            "api_calls": 1,
            "failed": True,
            "error": _HTML_ERROR_PAGE,
        }
        response = _normalize_empty_agent_response(agent_result, "", history_len=5)

        assert "The request failed:" in response
        assert "<html" not in response.lower()
        assert "<!doctype" not in response.lower()
        assert "502" in response  # the informative part survives

    def test_partial_result_with_html_error_is_sanitized(self):
        agent_result = {
            "final_response": None,
            "api_calls": 3,
            "failed": False,
            "interrupted": False,
            "partial": True,
            "error": _HTML_ERROR_PAGE,
        }
        response = _normalize_empty_agent_response(agent_result, "", history_len=5)

        assert "Processing stopped:" in response
        assert "<html" not in response.lower()
        assert "<!doctype" not in response.lower()
