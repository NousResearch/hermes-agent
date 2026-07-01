"""Tests for agent.redact -- secret masking in logs and output."""

import logging

import pytest

from agent.redact import redact_sensitive_text, RedactingFormatter


@pytest.fixture(autouse=True)
def _ensure_redaction_enabled(monkeypatch):
    """Ensure HERMES_REDACT_SECRETS is not disabled by prior test imports."""
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    # Also patch the module-level snapshot so it reflects the cleared env var
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


class TestKnownPrefixes:
    def test_openai_sk_key(self):
        text = "Using key sk-proj-abc123def456ghi789jkl012"
        result = redact_sensitive_text(text)
        assert "sk-pro" in result
        assert "abc123def456" not in result
        assert "..." in result

    def test_openrouter_sk_key(self):
        text = "OPENROUTER_API_KEY=sk-or-v1-abcdefghijklmnopqrstuvwxyz1234567890"
        result = redact_sensitive_text(text)
        assert "abcdefghijklmnop" not in result

    def test_github_pat_classic(self):
        result = redact_sensitive_text("token: ghp_abc123def456ghi789jkl")
        assert "abc123def456" not in result

    def test_github_pat_fine_grained(self):
        result = redact_sensitive_text("github_pat_abc123def456ghi789jklmno")
        assert "abc123def456" not in result

    def test_slack_token(self):
        token = "xoxb-" + "0" * 12 + "-" + "a" * 14
        result = redact_sensitive_text(token)
        assert "a" * 14 not in result

    def test_google_api_key(self):
        result = redact_sensitive_text("AIzaSyB-abc123def456ghi789jklmno012345")
        assert "abc123def456" not in result

    def test_perplexity_key(self):
        result = redact_sensitive_text("pplx-abcdef123456789012345")
        assert "abcdef12345" not in result

    def test_fal_key(self):
        result = redact_sensitive_text("fal_abc123def456ghi789jkl")
        assert "abc123def456" not in result

    def test_notion_internal_integration_token(self):
        result = redact_sensitive_text("ntn_abc123def456ghi789jkl")
        assert "abc123def456" not in result

    def test_short_token_fully_masked(self):
        result = redact_sensitive_text("key=sk-short1234567")
        assert "***" in result


class TestEnvAssignments:
    def test_export_api_key(self):
        text = "export OPENAI_API_KEY=sk-proj-abc123def456ghi789jkl012"
        result = redact_sensitive_text(text)
        assert "OPENAI_API_KEY=" in result
        assert "abc123def456" not in result

    def test_quoted_value(self):
        text = 'MY_SECRET_TOKEN="supersecretvalue123456789"'
        result = redact_sensitive_text(text)
        assert "MY_SECRET_TOKEN=" in result
        assert "supersecretvalue" not in result

    def test_non_secret_env_unchanged(self):
        text = "HOME=/home/user"
        result = redact_sensitive_text(text)
        assert result == text

    def test_path_unchanged(self):
        text = "PATH=/usr/local/bin:/usr/bin"
        result = redact_sensitive_text(text)
        assert result == text

    def test_lowercase_python_variable_token_unchanged(self):
        # Regression: #4367 — lowercase 'token' assignment must not be redacted
        text = "before_tokens = response.usage.prompt_tokens"
        result = redact_sensitive_text(text)
        assert result == text

    def test_lowercase_python_variable_api_key_unchanged(self):
        # Regression: #4367 — lowercase 'api_key' must not be redacted
        text = "api_key = config.get('api_key')"
        result = redact_sensitive_text(text)
        assert result == text

    def test_typescript_await_token_unchanged(self):
        # Regression: #4367 — 'await' keyword must not be redacted as a secret value
        text = "const token = await getToken();"
        result = redact_sensitive_text(text)
        assert result == text

    def test_typescript_await_secret_unchanged(self):
        # Regression: #4367 — similar pattern with 'secret' variable
        text = "const secret = await fetchSecret();"
        result = redact_sensitive_text(text)
        assert result == text

    def test_export_whitespace_preserved(self):
        # Regression: #4367 — whitespace before uppercase env var must be preserved
        text = "export SECRET_TOKEN=mypassword"
        result = redact_sensitive_text(text)
        assert result.startswith("export ")
        assert "SECRET_TOKEN=" in result
        assert "mypassword" not in result


class TestJsonFields:
    def test_json_api_key(self):
        text = '{"apiKey": "sk-proj-abc123def456ghi789jkl012"}'
        result = redact_sensitive_text(text)
        assert "abc123def456" not in result

    def test_json_token(self):
        text = '{"access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.longtoken.here"}'
        result = redact_sensitive_text(text)
        assert "eyJhbGciOiJSUzI1NiIs" not in result

    def test_json_non_secret_unchanged(self):
        text = '{"name": "John", "model": "gpt-4"}'
        result = redact_sensitive_text(text)
        assert result == text


class TestAuthHeaders:
    def test_bearer_token(self):
        text = "Authorization: Bearer sk-proj-abc123def456ghi789jkl012"
        result = redact_sensitive_text(text)
        assert "Authorization: Bearer" in result
        assert "abc123def456" not in result

    def test_case_insensitive(self):
        text = "authorization: bearer mytoken123456789012345678"
        result = redact_sensitive_text(text)
        assert "mytoken12345" not in result

    def test_basic_auth_credentials_masked(self):
        # base64 of "user:longpassword1234" — leaks user:pass if not redacted.
        text = "Authorization: Basic dXNlcjpsb25ncGFzc3dvcmQxMjM0"
        result = redact_sensitive_text(text)
        assert "Authorization: Basic" in result
        assert "dXNlcjpsb25ncGFzc3dvcmQxMjM0" not in result

    def test_token_scheme_masked(self):
        text = "Authorization: token opaque-credential-1234567890"
        result = redact_sensitive_text(text)
        assert "Authorization: token" in result
        assert "opaque-credential" not in result

    def test_proxy_authorization_masked(self):
        text = "Proxy-Authorization: Basic dXNlcjpzdXBlcnNlY3JldDEyMzQ="
        result = redact_sensitive_text(text)
        assert "dXNlcjpzdXBlcnNlY3JldDEyMzQ=" not in result

    def test_authorization_prose_unchanged(self):
        # "authorization" without a colon-delimited value is plain prose.
        text = "the authorization model is fully open"
        assert redact_sensitive_text(text) == text

    def test_token_flush_against_double_quote_preserves_quote(self):
        # Regression for #43083: a token sitting flush against a closing
        # double quote must NOT pull that quote into the mask. Greedy \S+
        # used to eat it, turning value corruption into syntax corruption
        # (unterminated quote → shell EOF).
        text = 'curl -H "Authorization: Bearer sk-abcdef1234567890"'
        result = redact_sensitive_text(text)
        assert "sk-abcdef1234567890" not in result
        assert result.count('"') == 2, result  # both quotes survive
        assert result.endswith('"'), result

    def test_token_flush_against_single_quote_preserves_quote(self):
        # Regression for #43083: same as above with single quotes (Python
        # f-string context). The closing ' must survive the mask.
        text = "auth = f'Authorization: Bearer {placeholder}'"
        result = redact_sensitive_text(text)
        assert result.count("'") == 2, result
        assert result.endswith("'"), result


class TestApiKeyHeaders:
    def test_x_api_key_header_masked(self):
        text = "x-api-key: opaque-provider-key-1234567890"
        result = redact_sensitive_text(text)
        assert "x-api-key:" in result
        assert "opaque-provider-key" not in result

    def test_x_api_key_in_curl_command_masked(self):
        text = 'curl -H "x-api-key: sk-local-VERYsecret-999888" https://api.example.com'
        result = redact_sensitive_text(text)
        assert "VERYsecret" not in result
        assert "https://api.example.com" in result

    def test_api_key_header_masked(self):
        text = "api-key: anotherOpaqueSecret1234567"
        result = redact_sensitive_text(text)
        assert "anotherOpaqueSecret" not in result


class TestTelegramTokens:
    def test_bot_token(self):
        text = "bot123456789:ABCDEfghij-KLMNopqrst_UVWXyz12345"
        result = redact_sensitive_text(text)
        assert "ABCDEfghij" not in result
        assert "123456789:***" in result

    def test_raw_token(self):
        text = "12345678901:ABCDEfghijKLMNopqrstUVWXyz1234567890"
        result = redact_sensitive_text(text)
        assert "ABCDEfghij" not in result


class TestPassthrough:
    def test_empty_string(self):
        assert redact_sensitive_text("") == ""

    def test_none_returns_none(self):
        assert redact_sensitive_text(None) is None

    def test_non_string_input_int_coerced(self):
        assert redact_sensitive_text(12345) == "12345"

    def test_non_string_input_dict_coerced_and_redacted(self):
        result = redact_sensitive_text({"token": "sk-proj-abc123def456ghi789jkl012"})
        assert "abc123def456" not in result

    def test_normal_text_unchanged(self):
        text = "Hello world, this is a normal log message with no secrets."
        assert redact_sensitive_text(text) == text

    def test_code_unchanged(self):
        text = "def main():\n    print('hello')\n    return 42"
        assert redact_sensitive_text(text) == text

    def test_url_without_key_unchanged(self):
        text = "Connecting to https://api.openai.com/v1/chat/completions"
        assert redact_sensitive_text(text) == text


class TestRedactingFormatter:
    def test_formats_and_redacts(self):
        formatter = RedactingFormatter("%(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Key is sk-proj-abc123def456ghi789jkl012",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert "abc123def456" not in result
        assert "sk-pro" in result


class TestPrintenvSimulation:
    """Simulate what happens when the agent runs `env` or `printenv`."""

    def test_full_env_dump(self):
        env_dump = """HOME=/home/user
PATH=/usr/local/bin:/usr/bin
OPENAI_API_KEY=sk-proj-abc123def456ghi789jkl012mno345
OPENROUTER_API_KEY=sk-or-v1-reallyLongSecretKeyValue12345678
FIRECRAWL_API_KEY=fc-shortkey123456789012
TELEGRAM_BOT_TOKEN=bot987654321:ABCDEfghij-KLMNopqrst_UVWXyz12345
SHELL=/bin/bash
USER=teknium"""
        result = redact_sensitive_text(env_dump)
        # Secrets should be masked
        assert "abc123def456" not in result
        assert "reallyLongSecretKey" not in result
        assert "ABCDEfghij" not in result
        # Non-secrets should survive
        assert "HOME=/home/user" in result
        assert "SHELL=/bin/bash" in result
        assert "USER=teknium" in result


class TestSecretCapturePayloadRedaction:
    def test_secret_value_field_redacted(self):
        text = '{"success": true, "secret_value": "sk-test-secret-1234567890"}'
        result = redact_sensitive_text(text)
        assert "sk-test-secret-1234567890" not in result

    def test_raw_secret_field_redacted(self):
        text = '{"raw_secret": "ghp_abc123def456ghi789jkl"}'
        result = redact_sensitive_text(text)
        assert "abc123def456" not in result


class TestElevenLabsTavilyExaKeys:
    """Regression tests for ElevenLabs (sk_), Tavily (tvly-), and Exa (exa_) keys."""

    def test_elevenlabs_key_redacted(self):
        text = "ELEVENLABS_API_KEY=sk_abc123def456ghi789jklmnopqrstu"
        result = redact_sensitive_text(text)
        assert "abc123def456ghi" not in result

    def test_elevenlabs_key_in_log_line(self):
        text = "Connecting to ElevenLabs with key sk_abc123def456ghi789jklmnopqrstu"
        result = redact_sensitive_text(text)
        assert "abc123def456ghi" not in result

    def test_tavily_key_redacted(self):
        text = "TAVILY_API_KEY=tvly-ABCdef123456789GHIJKL0000"
        result = redact_sensitive_text(text)
        assert "ABCdef123456789" not in result

    def test_tavily_key_in_log_line(self):
        text = "Initialising Tavily client with tvly-ABCdef123456789GHIJKL0000"
        result = redact_sensitive_text(text)
        assert "ABCdef123456789" not in result

    def test_exa_key_redacted(self):
        text = "EXA_API_KEY=exa_XYZ789abcdef000000000000000"
        result = redact_sensitive_text(text)
        assert "XYZ789abcdef" not in result

    def test_exa_key_in_log_line(self):
        text = "Using Exa client with key exa_XYZ789abcdef000000000000000"
        result = redact_sensitive_text(text)
        assert "XYZ789abcdef" not in result

    def test_all_three_in_env_dump(self):
        env_dump = (
            "HOME=/home/user\n"
            "ELEVENLABS_API_KEY=sk_abc123def456ghi789jklmnopqrstu\n"
            "TAVILY_API_KEY=tvly-ABCdef123456789GHIJKL0000\n"
            "EXA_API_KEY=exa_XYZ789abcdef000000000000000\n"
            "SHELL=/bin/bash\n"
        )
        result = redact_sensitive_text(env_dump)
        assert "abc123def456ghi" not in result
        assert "ABCdef123456789" not in result
        assert "XYZ789abcdef" not in result
        assert "HOME=/home/user" in result
        assert "SHELL=/bin/bash" in result


class TestJWTTokens:
    """JWT tokens start with eyJ (base64 for '{') and have dot-separated parts."""

    def test_full_3part_jwt(self):
        text = (
            "Token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
            ".eyJpc3MiOiI0MjNiZDJkYjg4MjI0MDAwIn0"
            ".Gxgv0rru-_kS-I_60EJ7CENTnBh9UeuL3QhkMoQ-VnM"
        )
        result = redact_sensitive_text(text)
        assert "Token:" in result
        # Payload and signature must not survive
        assert "eyJpc3Mi" not in result
        assert "Gxgv0rru" not in result

    def test_2part_jwt(self):
        text = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0"
        result = redact_sensitive_text(text)
        assert "eyJzdWIi" not in result

    def test_standalone_jwt_header(self):
        text = "leaked header: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9 here"
        result = redact_sensitive_text(text)
        assert "IkpXVCJ9" not in result
        assert "leaked header:" in result

    def test_jwt_with_base64_padding(self):
        text = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0=.abc123def456ghij"
        result = redact_sensitive_text(text)
        assert "abc123def456" not in result

    def test_short_eyj_not_matched(self):
        """eyJ followed by fewer than 10 base64 chars should not match."""
        text = "eyJust a normal word"
        assert redact_sensitive_text(text) == text

    def test_jwt_preserves_surrounding_text(self):
        text = "before eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0 after"
        result = redact_sensitive_text(text)
        assert result.startswith("before ")
        assert result.endswith(" after")

    def test_home_assistant_jwt_in_memory(self):
        """Real-world pattern: HA token stored in agent memory block."""
        text = (
            "Home Assistant API Token: "
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
            ".eyJpc3MiOiJhYmNkZWYiLCJleHAiOjE3NzQ5NTcxMDN9"
            ".Gxgv0rru-_kS-I_60EJ7CENTnBh9UeuL3QhkMoQ-VnM"
        )
        result = redact_sensitive_text(text)
        assert "Home Assistant API Token:" in result
        assert "Gxgv0rru" not in result
        assert "..." in result


class TestDiscordMentions:
    """Discord mention snowflakes (<@ID> / <@!ID>) are public syntax, not
    secrets — they must pass through the redactor unchanged so multi-bot
    @-pings (DISCORD_ALLOW_BOTS=mentions) keep resolving. See issue #35611."""

    def test_normal_mention_passes_through(self):
        text = "Hello <@222589316709220353>"
        assert redact_sensitive_text(text) == text

    def test_nickname_mention_passes_through(self):
        text = "Ping <@!1331549159177846844>"
        assert redact_sensitive_text(text) == text

    def test_multiple_mentions_pass_through(self):
        text = "<@111111111111111111> and <@222222222222222222>"
        assert redact_sensitive_text(text) == text

    def test_short_id_passes_through(self):
        text = "<@12345>"
        assert redact_sensitive_text(text) == text

    def test_slack_mention_passes_through(self):
        text = "<@U024BE7LH>"
        assert redact_sensitive_text(text) == text

    def test_preserves_surrounding_text(self):
        text = "User <@222589316709220353> said hello"
        assert redact_sensitive_text(text) == text


class TestWebUrlsNotRedacted:
    """Web URLs (http/https/wss) pass through unchanged — magic-link
    checkouts, OAuth callbacks the agent is meant to follow, and pre-signed
    share URLs must reach the tool intact. Known credential shapes inside
    URLs (sk-, ghp_, JWTs) are still caught by the prefix and JWT regexes.
    DB connection-string passwords are still caught by _DB_CONNSTR_RE.
    """

    def test_oauth_callback_code_passes_through(self):
        text = "GET https://api.example.com/oauth/cb?code=abc123xyz789&state=csrf_ok"
        assert redact_sensitive_text(text) == text

    def test_access_token_query_passes_through(self):
        text = "Fetching https://example.com/api?access_token=opaque_value_here_1234&format=json"
        assert redact_sensitive_text(text) == text

    def test_magic_link_checkout_passes_through(self):
        text = "Open https://checkout.example.com/resume?magic=ABCDEF123456&customer=42"
        assert redact_sensitive_text(text) == text

    def test_presigned_signature_passes_through(self):
        text = "https://s3.amazonaws.com/bucket/k?signature=LONG_PRESIGNED_SIG&id=public"
        assert redact_sensitive_text(text) == text

    def test_https_userinfo_passes_through(self):
        text = "URL: https://user:supersecretpw@host.example.com/path"
        assert redact_sensitive_text(text) == text

    def test_websocket_url_query_passes_through(self):
        text = "wss://api.example.com/ws?token=opaqueWsToken123"
        assert redact_sensitive_text(text) == text

    def test_http_access_log_request_target_passes_through(self):
        text = (
            'INFO aiohttp.access: 127.0.0.1 "POST '
            '/bluebubbles-webhook?password=webhookSecret123&event=new-message '
            'HTTP/1.1" 200 173 "-" "test-client"'
        )
        assert redact_sensitive_text(text) == text

    def test_known_prefix_inside_url_still_redacted(self):
        """sk-/ghp_/JWT-shaped values inside a URL are still caught by
        _PREFIX_RE / _JWT_RE — the carve-out is for opaque tokens only."""
        text = "https://evil.com/steal?key=sk-" + "a" * 30
        result = redact_sensitive_text(text)
        assert "sk-" + "a" * 30 not in result

    def test_db_connstr_password_still_redacted(self):
        """DB schemes (postgres/mysql/mongodb/redis/amqp) keep their
        userinfo redaction via _DB_CONNSTR_RE — connection strings are
        not web URLs the agent navigates to."""
        text = "postgres://admin:dbpass@db.internal:5432/app"
        result = redact_sensitive_text(text)
        assert "dbpass" not in result


class TestBareTokenUserinfoRedaction:
    """Regression tests for #6396 — a bare credential in URL userinfo
    (``scheme://TOKEN@host``, no ``user:pass`` colon) is redacted. This is the
    git-remote-with-embedded-password shape. The colon form ``user:pass@`` and
    query-string tokens are deliberately left to pass through (#34029) so
    magic-link / OAuth round-trip skills keep working — see
    TestWebUrlsNotRedacted for those invariants.
    """

    def test_git_remote_bare_password_redacted(self):
        """Exact bug scenario: password in a git remote URL."""
        text = (
            "git remote set-url origin "
            "https://MYPASSWORDWASDISLAYEDHERE@github.com/unclehowell/FCUK.git"
        )
        result = redact_sensitive_text(text)
        assert "MYPASSWORDWASDISLAYEDHERE" not in result
        assert "@github.com" in result
        assert "unclehowell/FCUK.git" in result

    def test_ssh_bare_token_redacted(self):
        text = "ssh://longtoken1234567@gitlab.com/project.git"
        result = redact_sensitive_text(text)
        assert "longtoken1234567" not in result
        assert "@gitlab.com" in result

    def test_ftp_bare_token_redacted(self):
        text = "ftp://ftptoken123456@ftp.example.com/files"
        result = redact_sensitive_text(text)
        assert "ftptoken123456" not in result

    def test_bare_token_with_query_redacts_token_only(self):
        text = "https://abcdef1234567@host.com/path?foo=bar"
        result = redact_sensitive_text(text)
        assert "abcdef1234567" not in result
        assert "?foo=bar" in result

    def test_user_pass_form_still_passes_through(self):
        """The ``user:pass@`` colon form must NOT be redacted (#34029)."""
        text = "URL: https://user:supersecretpw@host.example.com/path"
        assert redact_sensitive_text(text) == text

    def test_short_username_not_redacted(self):
        """Short userinfo (git, admin, deploy) below the 8-char floor passes."""
        for text in (
            "https://git@github.com/user/repo.git",
            "https://admin@example.com/x",
            "https://deploy@host.com/y",
        ):
            assert redact_sensitive_text(text) == text

    def test_email_in_path_not_redacted(self):
        """An ``@`` in a path/query is not userinfo — the token class stops at
        ``/``, so emails after the first slash are never treated as a credential."""
        for text in (
            "https://example.com/search?q=user@example.com",
            "https://example.com/users/john@doe.com/profile",
        ):
            assert redact_sensitive_text(text) == text

    def test_plain_url_unchanged(self):
        text = "https://github.com/user/repo.git"
        assert redact_sensitive_text(text) == text

    def test_long_bare_token_preserves_head_tail(self):
        token = "abcdef" + "x" * 20 + "wxyz"
        text = f"https://{token}@github.com/u/r.git"
        result = redact_sensitive_text(text)
        assert token not in result
        assert "abcdef" in result  # head preserved
        assert "wxyz" in result    # tail preserved


class TestFormBodyRedaction:
    """Form-urlencoded body redaction (k=v&k=v with no other text)."""

    def test_pure_form_body(self):
        text = "password=mysecret&username=bob&token=opaqueValue"
        result = redact_sensitive_text(text)
        assert "mysecret" not in result
        assert "opaqueValue" not in result
        assert "username=bob" in result

    def test_oauth_token_request(self):
        text = "grant_type=password&client_id=app&client_secret=topsecret&username=alice&password=alicepw"
        result = redact_sensitive_text(text)
        assert "topsecret" not in result
        assert "alicepw" not in result
        assert "client_id=app" in result

    def test_non_form_text_unchanged(self):
        """Sentences with `&` should NOT trigger form redaction."""
        text = "I have password=foo and other things"  # contains spaces
        result = redact_sensitive_text(text)
        # The space breaks the form regex; passthrough expected.
        assert "I have" in result

    def test_multiline_text_not_form(self):
        """Multi-line text is never treated as form body."""
        text = "first=1\nsecond=2"
        # Should pass through (still subject to other redactors)
        assert "first=1" in redact_sensitive_text(text)


class TestLowercaseDottedConfigKeys:
    """Issue #16413 — config-file passwords in lowercase/dotted/colon keys
    must be redacted. The uppercase _ENV_ASSIGN_RE missed these, leaking
    `spring.datasource.password=...` and `password: ...` from `cat`'d config
    files. Carve-outs: prose, code (#4367), and web URLs are left untouched.
    """

    def test_spring_dotted_password_assignment(self):
        text = "spring.datasource.password=Sup3rS3cret!"
        result = redact_sensitive_text(text)
        assert "Sup3rS3cret!" not in result
        assert "spring.datasource.password=" in result

    def test_dotted_api_key_split_keyword(self):
        # 'api.key' splits the keyword across a dot — must still match.
        text = "app.api.key=ak_live_998877"
        result = redact_sensitive_text(text)
        assert "ak_live_998877" not in result
        assert "app.api.key=" in result

    def test_bare_lowercase_password_at_line_start(self):
        text = "password=mysecretvalue123"
        result = redact_sensitive_text(text)
        assert "mysecretvalue123" not in result

    def test_quoted_lowercase_value(self):
        text = "password='mysecretvalue123'"
        result = redact_sensitive_text(text)
        assert "mysecretvalue123" not in result

    def test_yaml_unquoted_password(self):
        text = "password: Sup3rS3cret!"
        result = redact_sensitive_text(text)
        assert "Sup3rS3cret!" not in result
        assert "password:" in result

    def test_yaml_indented_dotted(self):
        text = "spring:\n  datasource:\n    password: hunter2pass"
        result = redact_sensitive_text(text)
        assert "hunter2pass" not in result

    def test_properties_file_dump(self):
        text = (
            "server.port=8080\n"
            "spring.datasource.username=admin\n"
            "spring.datasource.password=Sup3rS3cret!\n"
            "logging.level.root=INFO"
        )
        result = redact_sensitive_text(text)
        assert "Sup3rS3cret!" not in result
        assert "server.port=8080" in result  # non-secret keys preserved
        assert "username=admin" in result

    # --- carve-outs: must NOT redact ---

    def test_prose_mid_sentence_password_unchanged(self):
        # Not line-anchored, not dotted → conversational text, leave alone.
        text = "I have password=foo and other things"
        assert redact_sensitive_text(text) == text

    def test_lowercase_code_assignment_unchanged(self):
        # #4367 regression — spaces around '=' in code.
        text = "const secret = await fetchSecret();"
        assert redact_sensitive_text(text) == text

    def test_url_query_param_passes_through(self):
        # Web URLs are intentionally hands-off (documented design).
        text = "https://example.com/api?password=opaqueval123&format=json"
        assert redact_sensitive_text(text) == text

    def test_prose_keyword_in_value_unchanged(self):
        text = "note: secret meeting at noon"
        assert redact_sensitive_text(text) == text


class TestXaiToken:
    KEY = "xai-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstu"

    def test_bare_token_masked(self):
        result = redact_sensitive_text(f"using key {self.KEY}", force=True)
        assert self.KEY not in result
        assert "xai-AB" in result

    def test_env_assignment_masked(self):
        result = redact_sensitive_text(f"XAI_API_KEY={self.KEY}", force=True)
        assert self.KEY not in result

    def test_too_short_not_masked(self):
        short = "xai-tooshort"
        result = redact_sensitive_text(f"text {short} here", force=True)
        assert short in result

    def test_company_name_not_masked(self):
        result = redact_sensitive_text("xai is a company", force=True)
        assert result == "xai is a company"

    def test_prefix_visible_in_masked_output(self):
        result = redact_sensitive_text(self.KEY, force=True)
        assert result.startswith("xai-AB")


class TestGitIdentityAllowlist:
    """Git author/committer identity vars must NOT be redacted.

    GIT_AUTHOR_EMAIL / GIT_AUTHOR_NAME falsely matched the ENV-assignment
    secret regex because "AUTHOR" contains the secret-name token "AUTH".
    These are never secrets (they're public commit metadata), and redacting
    them corrupts `git filter-branch --env-filter` and any inline author setup.
    """

    def test_git_author_email_not_redacted(self):
        text = "GIT_AUTHOR_EMAIL=9063726+Kyzcreig@users.noreply.github.com"
        assert redact_sensitive_text(text) == text

    def test_git_author_name_not_redacted(self):
        text = "GIT_AUTHOR_NAME=Kyzcreig"
        assert redact_sensitive_text(text) == text

    def test_git_committer_vars_not_redacted(self):
        for text in (
            "GIT_COMMITTER_EMAIL=9063726+Kyzcreig@users.noreply.github.com",
            "GIT_COMMITTER_NAME=Kyzcreig",
        ):
            assert redact_sensitive_text(text) == text

    def test_git_author_with_export_prefix_not_redacted(self):
        text = "export GIT_AUTHOR_EMAIL=9063726+Kyzcreig@users.noreply.github.com"
        assert redact_sensitive_text(text) == text

    def test_real_secrets_still_redacted_alongside_git_vars(self):
        # Allowlisting git vars must NOT weaken redaction of real secrets.
        assert "supersecretvalue123" not in redact_sensitive_text("MY_AUTH_TOKEN=supersecretvalue123")
        assert "AKIAsecretkey0000" not in redact_sensitive_text("AWS_SECRET_ACCESS_KEY=AKIAsecretkey0000")
        assert "skprojkeyvalue999" not in redact_sensitive_text("OPENAI_API_KEY=skprojkeyvalue999")


class TestDbConnstrCodeOutput:
    """Regression tests for issue #33801 — _DB_CONNSTR_RE corrupting code output.

    Two distinct flaws, both confined to displayed tool OUTPUT (read_file /
    terminal / execute_code), never the on-disk content:

    1. The password group ``[^@]+`` was greedy across newlines, so on a
       multi-line block it scanned past the DSN line to the next stray ``@``
       (e.g. a Python ``@decorator``), replacing everything in between with
       ``***`` — dropping lines and concatenating the next one.
    2. An f-string DSN template (``f"postgresql://{user}:{pass}@{host}"``) is
       not a live credential, but was redacted anyway. Under ``code_file=True``
       a pure ``{...}`` brace password is now preserved.
    """

    MULTILINE = (
        '            return f"postgresql://{auth}@{self.pg_host}:'
        '{self.pg_port}/{self.pg_database}"\n'
        "\n"
        '    @model_validator(mode="after")\n'
        '    def _validate_critical_settings(self) -> "Settings":'
    )

    def test_multiline_block_not_corrupted(self):
        """The newline bound stops the greedy match from swallowing the
        decorator line. Original exact repro from the issue thread."""
        result = redact_sensitive_text(self.MULTILINE, code_file=True, force=True)
        assert result == self.MULTILINE
        # No line dropped, no concatenation onto the f-string line.
        assert "@model_validator" in result
        assert "_validate_critical_settings" in result
        assert result.count("\n") == self.MULTILINE.count("\n")

    def test_multiline_block_no_corruption_without_code_file(self):
        """Even without code_file, the newline bound alone prevents the
        catastrophic line-dropping. The single-line template's {pass} group
        is still masked here (code_file=False), but lines stay intact."""
        result = redact_sensitive_text(self.MULTILINE, force=True)
        assert "@model_validator" in result
        assert "_validate_critical_settings" in result
        assert result.count("\n") == self.MULTILINE.count("\n")

    def test_fstring_template_preserved_with_code_file(self):
        """A single-line DSN f-string template is preserved under code_file."""
        text = 'return f"postgresql://{user}:{password}@{host}:{port}/{db}"'
        assert redact_sensitive_text(text, code_file=True, force=True) == text

    def test_fstring_template_self_attr_preserved(self):
        text = 'dsn = f"postgresql://{u}:{self.db_pass}@{h}:{p}/{d}"'
        assert redact_sensitive_text(text, code_file=True, force=True) == text

    def test_literal_connstr_still_redacted_with_code_file(self):
        """A real password in a literal DSN is still masked under code_file."""
        text = "postgresql://admin:realpassword@db.internal:5432/app"
        result = redact_sensitive_text(text, code_file=True, force=True)
        assert "realpassword" not in result
        assert "***" in result

    def test_literal_connstr_redacted_all_schemes(self):
        for scheme, secret in [
            ("postgres", "pgsecret1234"),
            ("mysql", "mysqlsecret99"),
            ("redis", "redissecret77"),
            ("mongodb+srv", "mongosecret55"),
            ("amqp", "amqpsecret33"),
        ]:
            text = f"{scheme}://user:{secret}@host:1234/db"
            result = redact_sensitive_text(text, code_file=True, force=True)
            assert secret not in result, scheme

    def test_literal_connstr_in_log_line_redacted(self):
        text = "connected via postgres://user:s3cr3tpw@host:5432/db ok"
        result = redact_sensitive_text(text, force=True)
        assert "s3cr3tpw" not in result


class TestTerminalOutputRedaction:
    """is_env_dump_command + redact_terminal_output — issue #43025.

    Terminal/process stdout must be redacted on every surface (foreground
    `terminal` AND background `process(poll/log/wait)`). Env-dump commands get
    the ENV-assignment pass so opaque tokens (no vendor prefix) are masked;
    other commands stay on the code_file path to avoid false positives.
    """

    def test_is_env_dump_command_detection(self):
        from agent.redact import is_env_dump_command
        assert is_env_dump_command("printenv")
        assert is_env_dump_command("env")
        assert is_env_dump_command("env | grep API")
        assert is_env_dump_command("set")
        assert is_env_dump_command("export")
        assert is_env_dump_command("declare -x")
        assert is_env_dump_command("cat /tmp/x && printenv")
        assert not is_env_dump_command("python app.py")
        assert not is_env_dump_command("cat config.py")
        assert not is_env_dump_command("printf 'TOKEN=x'")
        assert not is_env_dump_command("")
        assert not is_env_dump_command(None)

    def test_env_dump_masks_opaque_token(self):
        from agent.redact import redact_terminal_output
        out = "MY_SERVICE_TOKEN=abc123randomopaquetokenvalue999\nHOME=/home/u"
        red = redact_terminal_output(out, "printenv")
        assert "abc123randomopaquetokenvalue999" not in red
        assert "HOME=/home/u" in red

    def test_non_env_command_preserves_source_false_positives(self):
        from agent.redact import redact_terminal_output
        # code_file path: MAX_TOKENS=100 is source, must survive; real sk- masked.
        out = "MAX_TOKENS=100\nOPENAI_API_KEY=sk-proj-abc123def456ghi789jkl012"
        red = redact_terminal_output(out, "cat config.py")
        assert "MAX_TOKENS=100" in red
        assert "abc123def456" not in red

    def test_unknown_command_uses_safe_code_file_path(self):
        from agent.redact import redact_terminal_output
        # No command → code_file=True; opaque non-prefix token NOT masked
        # (safe default avoids mangling arbitrary output), prefix still masked.
        out = "OPAQUE=plainvalue123\nKEY=sk-proj-abc123def456ghi789jkl012"
        red = redact_terminal_output(out, None)
        assert "abc123def456" not in red

    def test_disabled_passes_through(self, monkeypatch):
        from agent.redact import redact_terminal_output
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        out = "CUSTOM_TOKEN=zzzopaque1234567890abcdef"
        red = redact_terminal_output(out, "printenv")
        assert "zzzopaque1234567890abcdef" in red


class TestFileReadNonReusableRedaction:
    """#35519: prefix-matched credentials in FILE CONTENT (read_file /
    search_files / cat) must be redacted to a NON-REUSABLE sentinel — not a
    head/tail mask that looks like a real-but-truncated key and gets written
    back to config (corrupting the credential -> 401)."""

    GHP = "ghp_S1abcdefghijklmnopqrstuvwxyz0Pn2T"  # realistic GitHub PAT shape
    SK = "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789"

    def test_file_read_uses_nonreusable_sentinel(self):
        out = redact_sensitive_text(f"token: {self.GHP}", force=True, file_read=True)
        # The sentinel marker is present and obviously a redaction...
        assert "«redacted:ghp_…»" in out, out
        # ...and the head/tail-preserving mask shape is NOT produced.
        assert "..." not in out
        # The agent can still tell which vendor credential is present.
        assert "ghp_" in out

    def test_file_read_does_not_leak_secret_body(self):
        """Crucial: file_read must NOT expose the real key (no un-redact)."""
        out = redact_sensitive_text(f"token: {self.GHP}", force=True, file_read=True)
        # No run of the secret body survives.
        assert "S1abcdefghij" not in out
        assert self.GHP not in out
        assert "Pn2T" not in out  # not even the tail (the old mask kept it)

    def test_file_read_sentinel_is_not_a_plausible_key(self):
        """The sentinel can't be mistaken for / written back as a usable key:
        the old mask was a 13-char `ghp_S1...Pn2T` that broke GitHub auth when
        an agent re-saved it. The sentinel is syntactically invalid as a token
        (contains « » … and ':'), so it can't round-trip into a dead key."""
        out = redact_sensitive_text(f"GITHUB_PERSONAL_ACCESS_TOKEN: {self.GHP}",
                                    force=True, file_read=True)
        masked = out.split(": ", 1)[1].strip()
        # Not a bare token: contains the sentinel delimiters.
        assert masked.startswith("«") and masked.endswith("»")
        assert "…" in masked

    def test_default_mode_unchanged_keeps_headtail_mask(self):
        """Regression guard: NON-file_read (logs/display) keeps the existing
        head/tail mask shape — only file content gets the sentinel. Uses a
        bare-token context (no ``key:`` prefix) so this isolates the prefix
        pass: a ``token: <key>`` line would additionally hit the YAML config
        pass and collapse to ``***``, which is unrelated to this guard."""
        out = redact_sensitive_text(f"see {self.GHP} here", force=True)
        assert "«redacted" not in out          # no sentinel in log mode
        assert "ghp_" in out and "..." in out   # head/tail mask preserved

    def test_file_read_implies_code_file_no_env_falsepos(self):
        """file_read should skip the source-code ENV/JSON false-positive paths
        (it's config/data). A bare ``MAX_TOKENS=8000`` must pass through."""
        out = redact_sensitive_text("MAX_TOKENS=8000", force=True, file_read=True)
        assert out == "MAX_TOKENS=8000"

    def test_sk_prefix_also_sentinelized(self):
        out = redact_sensitive_text(f"key: {self.SK}", force=True, file_read=True)
        assert "«redacted:sk-…»" in out
        assert self.SK not in out


class TestFireworksToken:
    KEY = "fw_" + "A" * 40

    def test_bare_token_masked(self):
        result = redact_sensitive_text(f"fireworks error: key {self.KEY}", force=True)
        assert self.KEY not in result
        assert "fw_AA" in result

    def test_env_assignment_masked(self):
        result = redact_sensitive_text(f"FIREWORKS_API_KEY={self.KEY}", force=True)
        assert self.KEY not in result

    def test_too_short_not_masked(self):
        short = "fw_tooshort"
        result = redact_sensitive_text(f"text {short} here", force=True)
        assert short in result

    def test_prefix_visible_in_masked_output(self):
        result = redact_sensitive_text(self.KEY, force=True)
        assert result.startswith("fw_AA")
