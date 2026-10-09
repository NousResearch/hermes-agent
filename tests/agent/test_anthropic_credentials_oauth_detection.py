import agent.anthropic_credentials as anth_cred


def test_is_oauth_token_excludes_console_key_formats():
    """Regular Console API keys (sk-ant-api*, sk-ant-usr*, sk-ant-svc*) are not OAuth tokens."""
    assert anth_cred._is_oauth_token("sk-ant-api03-abc") is False
    assert anth_cred._is_oauth_token("sk-ant-usr-abc") is False
    assert anth_cred._is_oauth_token("sk-ant-svc-abc") is False


def test_is_oauth_token_detects_real_oauth_formats():
    """Genuine OAuth/setup tokens are still correctly detected."""
    assert anth_cred._is_oauth_token("sk-ant-oat01-abc") is True
    assert anth_cred._is_oauth_token("eyJhbGciOi...") is True
    assert anth_cred._is_oauth_token("cc-abc123") is True
