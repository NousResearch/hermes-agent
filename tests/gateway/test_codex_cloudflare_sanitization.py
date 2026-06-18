from gateway.run import _sanitize_gateway_final_response


def test_whatsapp_codex_cloudflare_html_is_suppressed():
    html = """
    <!doctype html>
    <html>
      <body>
        <noscript>Enable JavaScript and cookies to continue</noscript>
        <script>
          window._cf_chl_opt = {
            cUPMDTk: "/backend-api/codex/chat/completions?__cf_chl_tk=abc"
          };
        </script>
      </body>
    </html>
    """

    result = _sanitize_gateway_final_response("whatsapp", html)

    assert "Cloudflare challenge" in result
    assert "_cf_chl_opt" not in result
    assert "/backend-api/codex/chat/completions" not in result
    assert len(result) < 300


def test_non_cloudflare_whatsapp_response_is_unchanged():
    text = "<p>Normal assistant response mentioning HTML.</p>"

    assert _sanitize_gateway_final_response("whatsapp", text) == text
