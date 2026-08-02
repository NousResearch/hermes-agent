"""Tests for agent.redact -- secret masking in logs and output."""

import logging

import pytest

from agent import redact as redact_module
from agent.redact import (
    redact_cdp_url,
    redact_sensitive_text,
    RedactingFormatter,
    sanitize_terminal_secret_url,
    split_incomplete_sensitive_suffix,
    StreamingSecretSanitizer,
)


@pytest.fixture(autouse=True)
def _ensure_redaction_enabled(monkeypatch):
    """Ensure HERMES_REDACT_SECRETS is not disabled by prior test imports."""
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    # Also patch the module-level snapshot so it reflects the cleared env var
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


class TestKnownPrefixes:




    def test_gitlab_token_prefixes(self):
        """GitLab token families redact via their literal prefixes.

        Ported from openclaw/openclaw#112954; follow-up invited in #4541.
        """
        tokens = [
            # NOTE: every token is prefix + suffix CONCATENATION so no
            # contiguous token literal exists in this file — GitHub push
            # protection blocks realistic GitLab-token-shaped literals.
            "glpat-" + "Zx9AbCdEfGhIjKlMnOpQ",       # personal access token
            "gloas-" + "a" * 64,                     # OAuth application secret
            "gldt-" + "AbCdEfGhIjKlMnOpQrSt",        # deploy token
            "glrt-" + "t1_AbCdEfGhIjKlMnOpQrSt",     # runner auth token
            "glrt-" + "A" * 27 + ".01." + "a" * 9,   # routable (dotted) runner token
            "glrtr-" + "B" * 27 + ".01." + "b" * 9,  # routable runner registration
            "glcbt-" + "a1B2_AbCdEfGhIjKlMnOpQ",     # CI/CD job token
            "glptt-" + "c" * 40,                     # pipeline trigger token
            "glft-" + "AbCdEfGhIjKlMnOp",            # feed token
            "glimt-" + "AbCdEfGhIjKlMnOpQrStUvWxY",  # incoming mail token
            "glagent-" + "d" * 50,                   # agent (KAS) token
            "glsoat-" + "AbCdEfGhIjKlMnOpQrSt",      # service-account token
            "glffct-" + "AbCdEfGhIjKlMnOpQrSt",      # feature-flags client token
            "glwt-" + "AbCdEfGhIjKlMnOpQrSt",        # workspace token
            "GR1348941" + "E" * 20,                  # legacy runner registration
        ]
        for token in tokens:
            result = redact_sensitive_text(f"leaked {token} in output")
            secret_body = token.split("-", 1)[-1] if "-" in token else token[9:]
            assert secret_body not in result, f"{token!r} survived redaction: {result!r}"

    def test_gitlab_prefix_requires_word_boundary_and_length(self):
        """Prose and embedded identifiers must not false-positive."""
        for benign in [
            "the glossary explains gitlab tokens",   # no prefix at all
            "glpat-short",                            # suffix under 10 chars
            "myglpat-AbCdEfGhIjKlMnOpQrSt",           # embedded — lookbehind blocks
        ]:
            assert redact_sensitive_text(benign) == benign

    def test_slack_token(self):
        token = "xoxb-" + "0" * 12 + "-" + "a" * 14
        result = redact_sensitive_text(token)
        assert "a" * 14 not in result





    def test_fireworks_keys(self):
        samples = [
            "fw-" + "A" * 40,
            "fw_" + "B" * 40,
            "fpk_" + "C" * 40,
        ]

        for token in samples:
            result = redact_sensitive_text(f"provider error {token}")
            assert token not in result
            assert "..." in result

    def test_short_fireworks_like_words_unchanged(self):
        text = "fw-tooshort fw_tooshort fpk_tooshort"
        assert redact_sensitive_text(text) == text




class TestStreamingPrefixCandidates:
    def test_every_known_prefix_pattern_has_streaming_spec(self):
        assert len(redact_module._STREAMING_PREFIX_SPECS) == len(
            redact_module._PREFIX_PATTERNS
        )

    def test_incomplete_known_prefix_is_held(self):
        text = "visible\nsk-abcdefghi"

        visible, held = split_incomplete_sensitive_suffix(text)

        assert visible == "visible\n"
        assert held == "sk-abcdefghi"

    def test_diverged_prefix_is_released(self):
        text = "visible\nsk-abc is documentation"

        assert split_incomplete_sensitive_suffix(text) == (text, "")

    def test_progress_completed_json_quote_does_not_retain_raw_value(self):
        secret = "opaqueCompletedJsonCredential123"
        sanitizer = StreamingSecretSanitizer(token_candidates_only=True)

        visible = sanitizer.feed(f'{{"token": "{secret}"')

        assert secret not in visible
        assert "..." in visible
        assert sanitizer.pending == ""
        assert sanitizer.flush() == ""

    def test_progress_partial_json_still_retains_until_completion(self):
        secret = "opaqueSplitJsonCredential123"
        sanitizer = StreamingSecretSanitizer(token_candidates_only=True)

        assert sanitizer.feed('{"token": "') == "{"
        assert sanitizer.pending == '"token": "'
        visible = sanitizer.feed(f'{secret}"}}')

        assert secret not in visible
        assert "..." in visible
        assert sanitizer.pending == ""

    @pytest.mark.parametrize("token_candidates_only", [False, True])
    @pytest.mark.parametrize(
        "payload",
        [
            '{"token"' + " " * 200 + ': "opaqueChunkedJsonCredential123"}',
            "OPENAI_API_KEY=opaqueChunkedEnvCredential123\n",
        ],
        ids=["json-unbounded-whitespace", "env-assignment"],
    )
    def test_one_byte_feeds_preserve_streaming_redaction(
        self,
        token_candidates_only,
        payload,
    ):
        sanitizer = StreamingSecretSanitizer(
            token_candidates_only=token_candidates_only,
        )

        visible = "".join(sanitizer.feed(char) for char in payload)
        visible += sanitizer.flush()

        assert "opaqueChunked" not in visible
        assert "***" in visible or "..." in visible
        assert sanitizer.pending == ""

    @pytest.mark.parametrize(
        ("text", "held"),
        [
            (
                'visible\n{"token": "opaque',
                '{"token": "opaque',
            ),
            (
                "visible\npostgresql://operator:opaque",
                "postgresql://operator:opaque",
            ),
            (
                "visible\n-----BEGIN PRIVATE KEY-----\nopaque",
                "-----BEGIN PRIVATE KEY-----\nopaque",
            ),
        ],
    )
    def test_unterminated_structured_secret_is_held(self, text, held):
        visible, actual_held = split_incomplete_sensitive_suffix(text)

        if held.startswith("{"):
            assert visible == "visible\n{"
            assert actual_held == held[1:]
        else:
            assert visible == "visible\n"
            assert actual_held == held

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ('{"token": "opaque', '{"token": "***"'),
            ("postgresql://operator:opaque", "postgresql://operator:***"),
            (
                "-----BEGIN PRIVATE KEY-----\nopaque",
                "[REDACTED PRIVATE KEY]",
            ),
        ],
    )
    def test_terminal_unterminated_structured_secret_is_masked(self, text, expected):
        assert split_incomplete_sensitive_suffix(text, final=True) == (expected, "")

    def test_complete_structured_secret_is_released_for_static_redaction(self):
        text = '{"token": "opaque"}'

        assert split_incomplete_sensitive_suffix(text) == (text, "")

    def test_complete_private_key_is_released_only_as_a_complete_unit(self):
        text = (
            "-----BEGIN PRIVATE KEY-----\n"
            "SYNTHETICINERTPRIVATEKEYBODY1234567890\n"
            "-----END PRIVATE KEY-----"
        )

        assert split_incomplete_sensitive_suffix(text) == (text, "")
        assert redact_sensitive_text(text, force=True) == "[REDACTED PRIVATE KEY]"

    def test_new_private_key_prefix_after_complete_block_remains_held(self):
        complete = (
            "-----BEGIN PRIVATE KEY-----\n"
            "SYNTHETICINERTPRIVATEKEYBODY1234567890\n"
            "-----END PRIVATE KEY-----"
        )
        next_prefix = "\n-----BEG"

        visible, held = split_incomplete_sensitive_suffix(complete + next_prefix)

        assert visible == complete + "\n"
        assert held == "-----BEG"

    @pytest.mark.parametrize(
        "candidate",
        [
            'prefix"',
            "prefixpost",
            "prefixe",
        ],
        ids=["embedded-json", "embedded-db", "embedded-jwt"],
    )
    def test_embedded_canonical_opener_prefix_is_held(self, candidate):
        assert split_incomplete_sensitive_suffix(candidate) == ("", candidate)

    def test_first_sensitive_form_pair_is_held_with_its_key(self):
        candidate = "token=opaqueFirstFormCredential"

        assert split_incomplete_sensitive_suffix(candidate) == ("", candidate)

    def test_empty_quoted_env_concatenation_is_held_with_assignment(self):
        candidate = 'OPENAI_API_KEY=""'

        assert split_incomplete_sensitive_suffix(candidate) == ("", candidate)

    def test_database_opener_with_long_username_is_retained(self):
        opener = "postgresql://" + "u" * 600
        text = "visible\n" + opener

        assert split_incomplete_sensitive_suffix(text) == ("visible\n", opener)

    def test_every_enabled_assignment_and_header_has_partial_streaming_state(self):
        candidates = [
            *(f"SERVICE_{word[:-1]}" for word in redact_module._SECRET_ENV_WORDS),
            *(name[:-1] for name in redact_module._AUTH_HEADER_NAMES),
            *(name[:-1] for name in redact_module._SECRET_HEADER_NAME_VALUES),
        ]

        for candidate in candidates:
            assert split_incomplete_sensitive_suffix(candidate) == ("", candidate)

    def test_every_enabled_form_key_has_partial_streaming_state(self):
        for key in redact_module._SENSITIVE_QUERY_PARAMS:
            candidate = f"mode=x&{key[:-1]}"

            assert split_incomplete_sensitive_suffix(candidate) == ("", candidate)

    @pytest.mark.parametrize(
        "candidate",
        [
            "g",
            '"',
            "p",
            "-",
            "O",
            "A",
            "x",
            "mode=x&",
            "12345678:",
            "ey",
            "+",
            "OPENAI_API_KEY   ",
        ],
        ids=[
            "known-prefix",
            "json",
            "db",
            "pem",
            "env",
            "authorization",
            "api-key-header",
            "empty-form-key",
            "telegram-empty-body",
            "jwt-before-j",
            "phone-plus-only",
            "env-pre-equals-whitespace",
        ],
    )
    def test_streaming_state_is_prefix_closed(self, candidate):
        assert split_incomplete_sensitive_suffix(candidate) == ("", candidate)

    @pytest.mark.parametrize(
        "candidate",
        [
            "A" * 50 + "TOKEN" + "B" * 50 + " " * 200 + "=",
            '"token"' + " " * 200 + ":",
            "Authorization:" + " " * 200,
        ],
        ids=["env-max-name-whitespace", "json-whitespace", "auth-whitespace"],
    )
    def test_streaming_openers_match_canonical_whitespace_and_lengths(self, candidate):
        assert split_incomplete_sensitive_suffix(candidate) == ("", candidate)

    @pytest.mark.parametrize("progress_only", [False, True])
    def test_quote_rich_candidate_scan_uses_finite_literals(
        self,
        monkeypatch,
        progress_only,
    ):
        candidate = '"ordinary" ' * 2_000 + '"token"' + " " * 2_000 + ":"
        original = redact_module._could_be_json_opener_at
        calls = 0

        def _counted(text, start):
            nonlocal calls
            calls += 1
            return original(text, start)

        monkeypatch.setattr(redact_module, "_could_be_json_opener_at", _counted)

        if progress_only:
            sanitizer = StreamingSecretSanitizer(token_candidates_only=True)
            sanitizer.feed(candidate)
        else:
            split_incomplete_sensitive_suffix(candidate)

        assert calls <= len(redact_module._STREAMING_JSON_KEYS)

    @pytest.mark.parametrize("progress_only", [False, True])
    def test_chunked_json_whitespace_advances_incrementally(
        self,
        monkeypatch,
        progress_only,
    ):
        secret = "opaqueIncrementalJsonCredential123"
        payload = (
            '"token"'
            + " " * 10_000
            + ":"
            + " " * 10_000
            + f'"{secret}"'
        )
        original = redact_module._could_be_json_opener_at
        calls = 0

        def _counted(text, start):
            nonlocal calls
            calls += 1
            return original(text, start)

        monkeypatch.setattr(redact_module, "_could_be_json_opener_at", _counted)
        sanitizer = StreamingSecretSanitizer(
            token_candidates_only=progress_only,
        )

        visible = "".join(sanitizer.feed(char) for char in payload)
        visible += sanitizer.flush()

        assert secret not in visible
        assert calls <= 8 * len(redact_module._STREAMING_JSON_KEYS)

    @pytest.mark.parametrize("progress_only", [False, True])
    @pytest.mark.parametrize(
        "payload",
        [
            '"token"' + " " * 2_000 + "is ordinary prose",
            '"token"' + " " * 2_000 + ":" + " " * 2_000 + "not-a-string",
        ],
        ids=["invalid-colon", "invalid-value-opener"],
    )
    def test_chunked_json_whitespace_releases_on_divergence(
        self,
        progress_only,
        payload,
    ):
        sanitizer = StreamingSecretSanitizer(
            token_candidates_only=progress_only,
        )

        visible = "".join(sanitizer.feed(char) for char in payload)
        visible += sanitizer.flush()

        assert visible == payload
        assert sanitizer.pending == ""

    @pytest.mark.parametrize("progress_only", [False, True])
    def test_environment_candidate_scan_is_bounded(
        self,
        monkeypatch,
        progress_only,
    ):
        candidate = "!" * 10_000 + "OPENAI_API_KEY"
        original = redact_module._could_be_env_opener_at
        calls = 0

        def _counted(text, start):
            nonlocal calls
            calls += 1
            return original(text, start)

        monkeypatch.setattr(redact_module, "_could_be_env_opener_at", _counted)

        if progress_only:
            sanitizer = StreamingSecretSanitizer(token_candidates_only=True)
            sanitizer.feed(candidate)
        else:
            split_incomplete_sensitive_suffix(candidate)

        assert calls <= redact_module._MAX_STREAMING_ENV_NAME

    @pytest.mark.parametrize("progress_only", [False, True])
    @pytest.mark.parametrize("quote", ["", '"', "'"], ids=["unquoted", "double", "single"])
    def test_chunked_environment_value_advances_incrementally(
        self,
        monkeypatch,
        progress_only,
        quote,
    ):
        secret = "x" * 20_000
        payload = f"OPENAI_API_KEY={quote}{secret}"
        terminator = f"{quote} tail" if quote else " tail"
        original = redact_module._active_env_assignment_start
        calls = 0

        def _counted(text):
            nonlocal calls
            calls += 1
            return original(text)

        monkeypatch.setattr(
            redact_module,
            "_active_env_assignment_start",
            _counted,
        )
        sanitizer = StreamingSecretSanitizer(
            token_candidates_only=progress_only,
        )

        visible = "".join(sanitizer.feed(char) for char in payload)
        visible += "".join(sanitizer.feed(char) for char in terminator)
        visible += sanitizer.flush()

        assert visible == f"OPENAI_API_KEY={quote}***{quote} tail"
        assert secret not in visible
        assert sanitizer.pending == ""
        assert calls <= 32


class TestEnvAssignments:
    def test_export_api_key(self):
        text = "export OPENAI_API_KEY=sk-proj-abc123def456ghi789jkl012"
        result = redact_sensitive_text(text)
        assert "OPENAI_API_KEY=" in result
        assert "abc123def456" not in result


    def test_non_secret_env_unchanged(self):
        text = "HOME=/home/user"
        result = redact_sensitive_text(text)
        assert result == text






    def test_export_whitespace_preserved(self):
        # Regression: #4367 — whitespace before uppercase env var must be preserved
        text = "export SECRET_TOKEN=mypassword"
        result = redact_sensitive_text(text)
        assert result.startswith("export ")
        assert "SECRET_TOKEN=" in result
        assert "mypassword" not in result


class TestEnvLookupPreserved:
    """Programmatic env var lookups must not be corrupted (issue #2852)."""

    def test_os_getenv_single_quote_uppercase_key(self):
        text = "MY_API_KEY=os.getenv('OPENAI_API_KEY')"
        assert redact_sensitive_text(text, force=True) == text






    def test_real_env_value_still_redacted(self):
        text = "HOMEASSISTANT_TOKEN=eyJhbGciOiJIUzI1NiJ9.abc123.xyz"
        result = redact_sensitive_text(text, force=True)
        assert "eyJhbGciOiJIUzI1NiJ9" not in result


    def test_multiline_prose_with_code_snippet(self):
        text = """Set it up like this:
    HA_TOKEN=os.getenv('HOMEASSISTANT_TOKEN')
    if not HA_TOKEN:
        raise ValueError('Missing credentials')"""
        result = redact_sensitive_text(text, force=True)
        assert "os.getenv('HOMEASSISTANT_TOKEN')" in result







class TestJsonFields:
    def test_json_api_key(self):
        text = '{"apiKey": "sk-proj-abc123def456ghi789jkl012"}'
        result = redact_sensitive_text(text)
        assert "abc123def456" not in result


    def test_json_non_secret_unchanged(self):
        text = '{"name": "John", "model": "gpt-4"}'
        result = redact_sensitive_text(text)
        assert result == text


class TestAuthHeaders:





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



    def test_non_string_input_dict_coerced_and_redacted(self):
        result = redact_sensitive_text({"token": "sk-proj-abc123def456ghi789jkl012"})
        assert "abc123def456" not in result





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



class TestElevenLabsTavilyExaKeys:
    """Regression tests for ElevenLabs (sk_), Tavily (tvly-), and Exa (exa_) keys."""

    def test_elevenlabs_key_redacted(self):
        text = "ELEVENLABS_API_KEY=sk_abc123def456ghi789jklmnopqrstu"
        result = redact_sensitive_text(text)
        assert "abc123def456ghi" not in result






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


    def test_2part_jwt(self):
        text = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0"
        result = redact_sensitive_text(text)
        assert "eyJzdWIi" not in result




    def test_jwt_preserves_surrounding_text(self):
        text = "before eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0 after"
        result = redact_sensitive_text(text)
        assert result.startswith("before ")
        assert result.endswith(" after")



class TestDiscordMentions:
    """Discord mention snowflakes (<@ID> / <@!ID>) are public syntax, not
    secrets — they must pass through the redactor unchanged so multi-bot
    @-pings (DISCORD_ALLOW_BOTS=mentions) keep resolving. See issue #35611."""

    def test_normal_mention_passes_through(self):
        text = "Hello <@222589316709220353>"
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


class TestStrictUrlCredentialRedaction:
    @pytest.mark.parametrize(
        ("text", "secret", "expected"),
        [
            (
                "https://x.test/#access_token=FRAG_SECRET&view=public",
                "FRAG_SECRET",
                "https://x.test/#access_token=***&view=public",
            ),
            (
                "/resume?token=REL_SECRET&view=public",
                "REL_SECRET",
                "/resume?token=***&view=public",
            ),
            (
                "https://x.test/cb?client%5Fsecret=ENC_SECRET&view=public",
                "ENC_SECRET",
                "https://x.test/cb?client%5Fsecret=***&view=public",
            ),
            (
                "https://x.test/cb?client%255Fsecret=DOUBLE_SECRET&view=public",
                "DOUBLE_SECRET",
                "https://x.test/cb?client%255Fsecret=***&view=public",
            ),
            (
                "/resume?token=SEMICOLON_SECRET;view=public",
                "SEMICOLON_SECRET",
                "/resume?token=***;view=public",
            ),
            (
                "//user:NET_SECRET@x.test/path",
                "NET_SECRET",
                "//user:***@x.test/path",
            ),
        ],
    )
    def test_masks_all_url_reference_forms_only_when_opted_in(
        self, text, secret, expected
    ):
        assert redact_sensitive_text(text) == text

        result = redact_sensitive_text(text, redact_url_credentials=True)

        assert secret not in result
        assert result == expected

    def test_similarly_named_public_params_remain_unchanged(self):
        text = "/metrics?token_count=17&session_id=public"
        assert redact_sensitive_text(text, redact_url_credentials=True) == text


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




class TestFormBodyRedaction:
    """Form-urlencoded body redaction (k=v&k=v with no other text)."""

    def test_pure_form_body(self):
        text = "password=mysecret&username=bob&token=opaqueValue"
        result = redact_sensitive_text(text)
        assert "mysecret" not in result
        assert "opaqueValue" not in result
        assert "username=bob" in result


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





class TestConfigKeyRedosResistance:
    """The dotted-key patterns must not backtrack exponentially (ReDoS).

    Before the possessive-quantifier rewrite, a non-matching run of ~40
    dotted segments took ~30ms and doubled every ~4 segments; 100 segments
    would effectively hang the redactor (it runs on every log line).
    """

    def test_long_dotted_run_completes_fast(self):
        import time

        # 100 dotted segments with no '=' — worst case for the old pattern.
        text = ".".join(["segment"] * 100) + " end"
        t0 = time.perf_counter()
        assert redact_sensitive_text(text) == text
        assert time.perf_counter() - t0 < 2.0

    def test_long_dotted_run_with_keyword_completes_fast(self):
        """Exercise _CFG_DOTTED_RE directly (bypasses the keyword pre-gate).

        The pre-gate skips the regex when no secret keyword is present, so
        test_long_dotted_run_completes_fast only guards the pre-gate.  This
        test includes a keyword but no '=' so the regex runs and must still
        complete quickly thanks to the possessive quantifiers.
        """
        import time

        text = ".".join(["segment"] * 100) + ".token end"
        t0 = time.perf_counter()
        assert redact_sensitive_text(text) == text
        assert time.perf_counter() - t0 < 2.0

    def test_long_dotted_secret_still_redacted(self):
        # Possessive quantifiers must not change matching behavior.
        text = ".".join(["seg"] * 50) + ".password=Sup3rS3cret!"
        result = redact_sensitive_text(text)
        assert "Sup3rS3cret!" not in result
        assert ".password=" in result

    def test_yaml_assign_redos_resistance(self):
        """_YAML_ASSIGN_RE must not backtrack excessively on long inputs."""
        import time

        # 100 lines of a long dotted key with a secret keyword but no
        # matching colon-value form — stresses the regex without matching.
        line = "a." * 50 + "token not_an_assignment"
        text = "\n".join([line] * 100)
        t0 = time.perf_counter()
        redact_sensitive_text(text)
        assert time.perf_counter() - t0 < 2.0

    def test_yaml_assign_secret_still_redacted(self):
        # Possessive quantifiers must not change YAML matching behavior.
        text = "spring.datasource.password: hunter2"
        result = redact_sensitive_text(text)
        assert "hunter2" not in result
        assert "password:" in result


class TestXaiToken:
    KEY = "xai-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstu"

    def test_bare_token_masked(self):
        result = redact_sensitive_text(f"using key {self.KEY}", force=True)
        assert self.KEY not in result
        assert "xai-AB" in result


    def test_too_short_not_masked(self):
        short = "xai-tooshort"
        result = redact_sensitive_text(f"text {short} here", force=True)
        assert short in result




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





class TestFireworksToken:
    KEY = "fw_" + "A" * 40

    def test_bare_token_masked(self):
        result = redact_sensitive_text(f"fireworks error: key {self.KEY}", force=True)
        assert self.KEY not in result
        assert "fw_AA" in result


    def test_too_short_not_masked(self):
        short = "fw_tooshort"
        result = redact_sensitive_text(f"text {short} here", force=True)
        assert short in result



class TestRedactCdpUrl:
    """redact_cdp_url() is the single chokepoint for CDP endpoint log redaction.

    Unlike the global pass (which deliberately lets web-URL query params and
    userinfo through for OAuth/magic-link workflows), CDP endpoint credentials
    are pure secrets and must always be masked. Both the browser tool's
    session/discovery logs and the supervisor's attach-timeout error route
    through this helper.
    """


    def test_masks_multiple_query_credentials(self):
        url = "wss://provider.example/session?token=aaa-secret&apikey=bbb-secret"
        out = redact_cdp_url(url)
        assert "aaa-secret" not in out
        assert "bbb-secret" not in out




    def test_none_returns_empty(self):
        assert redact_cdp_url(None) == ""


class TestKeywordWordBoundary:
    """Ported from nearai/ironclaw#6129 — a secret keyword embedded inside a
    larger prose word (``Secretary`` ⊃ ``secret``, ``tokenizer`` ⊃ ``token``,
    ``authored`` ⊃ ``auth``) must NOT trigger the lowercase/dotted/YAML config
    passes. Real key shapes (separators, camelCase, acronyms, plurals, common
    concatenated compounds, all-caps env style) must keep redacting.
    """

    # ── prose words embedding a keyword are preserved ──────────────────

    def test_secretary_yaml_value_preserved(self):
        text = "Secretary: JanetYellen1234567890"
        assert redact_sensitive_text(text) == text








    # ── real key shapes still redact ────────────────────────────────────

    def test_separator_keys_still_redacted(self):
        for text in (
            "client_secret: abc123def456ghi789jkl",
            "auth_token: xyz789xyz789xyz789xyz",
            "my_secret: topvalue123456789012345",
            "db.password=hunter2verylongpassword",
        ):
            result = redact_sensitive_text(text)
            assert result != text, text

    def test_camelcase_keys_still_redacted(self):
        for text in (
            "clientSecret: abc123def456ghi789jkl",
            "secretKey: abc123def456ghi789jklmno",
            "APIToken: abc123def456ghi789jklmn",
        ):
            result = redact_sensitive_text(text)
            assert result != text, text


    def test_plural_keys_still_redacted(self):
        text = "secrets: hunter2hunter2hunter2hh"
        result = redact_sensitive_text(text)
        assert "hunter2hunter2hunter2hh" not in result


    def test_all_caps_embedded_keyword_still_redacted(self):
        # All-caps keys keep legacy embedded matching (MYTOKEN=…).
        text = "MYTOKEN=abcdefgh1234567890123456"
        result = redact_sensitive_text(text)
        assert "abcdefgh1234567890123456" not in result


class TestSanitizeTerminalSecretUrl:
    @pytest.mark.parametrize(
        "key",
        ["to%6ben", "%74oken", "X-Amz-Signature"],
    )
    def test_masks_canonical_sensitive_query_name(self, key):
        secret = "opaqueTerminalUrlCredential123"
        url = f"https://example.com/image.png?{key}={secret}&width=1024"

        result = sanitize_terminal_secret_url(url)

        assert secret not in result
        assert f"{key}=***" in result
        assert "width=1024" in result

    def test_preserves_ordinary_query_parameters_byte_for_byte(self):
        url = (
            "https://example.com/image.png?token_count=7&state=public&X-Amz-Expires=300"
        )

        assert sanitize_terminal_secret_url(url) == url
