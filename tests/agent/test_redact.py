"""Tests for agent.redact -- secret masking in logs and output."""

import logging

import pytest

from agent.redact import mask_secret, redact_cdp_url, redact_sensitive_text, RedactingFormatter


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

    @pytest.mark.parametrize(
        "text",
        [
            'IDENTITY_TOKEN="bailu"',
            "--override-tensor per_layer_token_embd.weight=CPU",
            'runtime.token="local"',
            '{"token": "CPU"}',
            "token: CPU",
        ],
    )
    def test_ambiguous_key_preserves_obviously_noncredential_value(self, text):
        assert redact_sensitive_text(text, force=True) == text

    @pytest.mark.parametrize(
        "text, cleartext",
        [
            ("PASSWORD=hunter2", "hunter2"),
            ("SECRET_TOKEN=bailu", "bailu"),
            ("id_token=local", "local"),
            ("CUSTOM_TOKEN=opaqueValue123456789", "opaqueValue123456789"),
            ('{"token": "opaqueValue123456789"}', "opaqueValue123456789"),
            ('{"key_material": "CPU"}', "CPU"),
            ('{"bearer": "local"}', "local"),
            ("TOKEN=" + "sk-" + "a" * 30, "a" * 20),
        ],
    )
    def test_strong_key_or_credential_shaped_value_still_redacts(
        self, text, cleartext
    ):
        assert cleartext not in redact_sensitive_text(text, force=True)






    def test_export_whitespace_preserved(self):
        # Regression: #4367 — whitespace before uppercase env var must be preserved
        text = "export SECRET_TOKEN=mypassword"
        result = redact_sensitive_text(text)
        assert result.startswith("export ")
        assert "SECRET_TOKEN=" in result
        assert "mypassword" not in result


class TestBareSecretEnvSuffixes:
    """Bare *_KEY / *_PASS / *_PW env suffixes mask, incl. lowercase — #77484."""

    def test_upper_suffix_keys_mask(self):
        for text in ("FAL_KEY=sk-abc123def456", "OPENAI_KEY=sk-abc123def456",
                     "MYSQL_PASS=ghi789", "DB_PW=jkl012"):
            result = redact_sensitive_text(text, force=True)
            assert "=" in result and result.split("=", 1)[1] != text.split("=", 1)[1]

    def test_lowercase_env_name_masks(self):
        # Opaque value with NO known prefix: only the env-name regex can
        # catch it — a sk-/ghp_ value would be masked by _PREFIX_RE on old
        # code too, making this test false-pass (review finding).
        result = redact_sensitive_text("openai_key=xyzzyplugh1234567890abcd", force=True)
        assert "xyzzyplugh1234567890abcd" not in result

    def test_prose_words_with_keyword_unchanged(self):
        # KEYBOARD / PASSAGE embed the bare keyword but are prose, not creds
        for text in ("KEYBOARD=notsecret", "PASSAGE=notsecret"):
            result = redact_sensitive_text(text, force=True)
            assert result == text

    def test_form_body_not_swallowed(self):
        # A bare `password=`/`token=` in a form body must not be eaten greedily
        text = "password=mysecret&username=bob&token=opaqueValue"
        result = redact_sensitive_text(text, force=True)
        assert "mysecret" not in result
        assert "opaqueValue" not in result
        assert "username=bob" in result


class TestControlCharSplitTokens:
    """Tokens split by control/zero-width chars must still mask — #77484."""

    def _assert_split_masked(self, text, tok):
        # Bare token (no KEY= context). Assert the LONGEST FRAGMENT is gone
        # from the result: the token is split in the input, so `tok` (whole,
        # contiguous) is absent from ANY output — even one that leaks every
        # fragment verbatim (review finding: whole-token assertion is a false
        # negative; the tail fragment appears contiguously in the leak).
        result = redact_sensitive_text(text, force=True)
        longest = max(tok[10:], tok[:10], key=len)
        assert longest not in result

    def test_newline_split_token_masks(self):
        tok = "ghp_abcdef1234567890ABCDEF1234567890abcdef"
        self._assert_split_masked(f"{tok[:10]}\n{tok[10:]}", tok)

    def test_esc_split_token_masks(self):
        tok = "ghp_abcdef1234567890ABCDEF1234567890abcdef"
        self._assert_split_masked(f"{tok[:10]}\x1b{tok[10:]}", tok)

    def test_zero_width_split_token_masks(self):
        tok = "ghp_abcdef1234567890ABCDEF1234567890abcdef"
        self._assert_split_masked(f"{tok[:10]}\u200b{tok[10:]}", tok)

    def test_complete_token_does_not_swallow_next_line(self):
        # A COMPLETE token at end-of-line followed by ordinary text must not
        # be joined across the newline — the ordinary prefix pass masks the
        # token; joining would swallow the adjacent line (browser
        # accessibility annotations regressed this way: "button [ref=e3]"
        # disappeared into the mask).
        tok = "ghp_" + "F" * 29
        text = f"text: Token: {tok}\nbutton [ref=e3]: Copy\n"
        result = redact_sensitive_text(text, force=True)
        assert "F" * 20 not in result
        assert "button" in result
        assert "ref=e3" in result

    def test_selfmatching_head_esc_split_tail_masked(self):
        # A split where the HEAD fragment alone already matches _PREFIX_RE
        # (>= 10 body chars) but the tail doesn't: the join must still run
        # for non-newline controls, or the tail leaks in cleartext. Only
        # LINE-crossing spans skip the join (see the annotation test).
        head = "sk-" + "a" * 15
        tail = "b" * 25
        result = redact_sensitive_text(head + "\x1b" + tail, force=True)
        assert tail not in result
        assert "a" * 12 not in result

    def test_env_dump_lines_not_joined(self):
        # Control-stripping must not join unrelated env lines into one match
        env_dump = (
            "HOME=/home/user\n"
            "ELEVENLABS_API_KEY=sk_abc123def456ghi789jkl\n"
            "EXA_API_KEY=exa_XY789abcdef01234\n"
            "SHELL=/bin/bash\n"
        )
        result = redact_sensitive_text(env_dump, force=True)
        assert "SHELL=/bin/bash" in result
        assert "HOME=/home/user" in result


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


class TestInterruptDebugLogRedaction:
    """#75461: both interrupt_debug.log write sites in cli.py route their
    message values through redact_sensitive_text(force=True) before the
    existing 60-char truncation. force=True is mandatory because the file
    is persistent and must never hold a raw credential even when the user
    has disabled global logging redaction (security.redact_secrets: false)."""

    # Telegram bot token shape that triggered the original leak.
    TOKEN = "1234567890:" + "A" * 35

    def test_force_masks_token(self):
        """The contract both CLI write sites rely on: force=True masks a
        Telegram-token-shaped payload."""
        result = redact_sensitive_text(self.TOKEN, force=True)
        assert "A" * 35 not in result
        assert "1234567890:***" in result

    def test_force_masks_even_when_redaction_disabled(self, monkeypatch):
        """Crucial: the persistent debug file must not leak a credential when
        the user has turned off global redaction. Without force=True the
        function returns the input unchanged (agent/redact.py:687-688)."""
        monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
        result = redact_sensitive_text(self.TOKEN, force=True)
        assert "A" * 35 not in result
        assert "1234567890:***" in result

    def test_truncation_happens_after_redaction(self):
        """Both CLI sites apply [:60] after redaction. A long payload whose
        secret sits near the start must still be masked — the secret cannot
        survive merely because it falls within the first 60 chars."""
        payload = self.TOKEN + " trailing context " + "x" * 80
        truncated = redact_sensitive_text(payload, force=True)[:60]
        assert "A" * 35 not in truncated

    def test_non_string_payload_coerced_and_masked(self):
        """Both CLI sites call str() on the payload before redacting; the
        dict payload path (interrupt queue) must not leak either."""
        payload = {"msg": self.TOKEN}
        result = redact_sensitive_text(str(payload), force=True)
        assert "A" * 35 not in result

    def test_url_userinfo_credentials_masked(self):
        """P2 follow-up: URL userinfo credentials (``user:pass@``) survive
        force=True alone. The write sites now pass redact_url_credentials=True,
        which masks the password portion of the userinfo."""
        text = "https://alice:CorrectHorseBatteryStaple@example.com/path"
        result = redact_sensitive_text(text, force=True, redact_url_credentials=True)
        assert "CorrectHorseBatteryStaple" not in result
        assert "alice:***@example.com" in result

    def test_url_credential_named_query_param_masked(self):
        """Credential-named query parameters (OAuth ``code=``, ``token=``, ...)
        are masked under the same call shape used at the write sites."""
        text = "https://example.com/oauth/cb?code=abc123xyz789&state=ok"
        result = redact_sensitive_text(text, force=True, redact_url_credentials=True)
        assert "abc123xyz789" not in result
        assert "code=***&state=ok" in result

    def test_url_redaction_truncation_still_works(self):
        """[:60] is still applied after redaction: a long URL whose credential
        sits near the start must not survive merely because it falls within the
        first 60 chars."""
        payload = "https://alice:secret@example.com/" + "x" * 80
        truncated = redact_sensitive_text(payload, force=True, redact_url_credentials=True)[:60]
        assert "secret" not in truncated


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

    def test_long_opaque_assignment_run_completes_fast(self):
        """Lowercase env scanning stays linear on compaction payload blobs."""
        import time

        # A serialized tool payload can contain a long opaque alphanumeric
        # value followed by '=' without containing a secret-key suffix.  The
        # old unanchored lowercase-env pattern retried its greedy prefix from
        # every byte, making this quadratic while holding the GIL.
        text = "a" * 20_000 + "=value"
        t0 = time.perf_counter()
        assert redact_sensitive_text(text, force=True) == text
        assert time.perf_counter() - t0 < 2.0

    def test_dotted_cfg_scan_stays_linear_with_keyword_elsewhere(self):
        """_CFG_DOTTED_RE must stay linear once the pre-gate passes.

        The ``_CFG_SECRET_WORD_RE`` pre-gate only skips secret-FREE text, so a
        payload that contains a real secret assignment AND a long opaque
        dotted run still reaches the backtrackable ``*`` prefix. Without the
        run-start lookbehind the sub retries that prefix from every byte of
        the run (quadratic while holding the GIL).
        """
        import time

        text = "password=hunter2\n" + "a." * 15_000 + "=value"
        t0 = time.perf_counter()
        result = redact_sensitive_text(text, force=True)
        assert "hunter2" not in result
        assert time.perf_counter() - t0 < 2.0

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
    `terminal` AND background `process(poll/log/wait)`). Env-dump commands
    and commands that read ``.env`` files get the ENV-assignment pass so
    opaque tokens (no vendor prefix) are masked; other commands stay on
    the code_file path to avoid false positives.
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

    # ── .env file detection (issue #61352 v2) ──

    def test_command_reads_env_file_detection(self):
        from agent.redact import _command_reads_env_file
        # Basic detection
        assert _command_reads_env_file("cat .env")
        assert _command_reads_env_file("cat .env.local")
        assert _command_reads_env_file("cat .env.production")
        assert _command_reads_env_file("cat .envrc")
        assert _command_reads_env_file("head .env")
        assert _command_reads_env_file("tail .env")
        assert _command_reads_env_file("type .env")
        assert _command_reads_env_file("nl .env")
        assert _command_reads_env_file("bat .env")
        # With flags
        assert _command_reads_env_file("cat -n .env")
        assert _command_reads_env_file("cat -A .env")
        # With paths
        assert _command_reads_env_file("cat ~/.hermes/.env")
        assert _command_reads_env_file("cat /home/user/project/.env")
        assert _command_reads_env_file("cat ./config/.env.local")
        # In a pipeline / sequence
        assert _command_reads_env_file("cat .env | grep KEY")
        assert _command_reads_env_file("echo '---' && cat .env")
        # Windows-style backslash paths
        assert _command_reads_env_file("cat C:\\Users\\test\\.env")
        # Quoted paths (plain split leaves the quotes attached)
        assert _command_reads_env_file('cat ".env"')
        assert _command_reads_env_file("cat '.env'")
        # Case-insensitive basename (macOS/Windows filesystems)
        assert _command_reads_env_file("cat .ENV")

    def test_command_reads_env_file_excludes_templates(self):
        from agent.redact import _command_reads_env_file
        # Templates/examples should NOT trigger
        assert not _command_reads_env_file("cat .env.example")
        assert not _command_reads_env_file("cat .env.sample")
        assert not _command_reads_env_file("cat .env.template")
        assert not _command_reads_env_file("cat .env.dist")

    def test_command_reads_env_file_rejects_non_env_files(self):
        from agent.redact import _command_reads_env_file
        assert not _command_reads_env_file("cat config.py")
        assert not _command_reads_env_file("cat README.md")
        assert not _command_reads_env_file("cat .envrc.bak")  # .bak not in list
        assert not _command_reads_env_file("python app.py")
        assert not _command_reads_env_file("echo .env")  # echo is not a file-read cmd
        assert not _command_reads_env_file("")
        assert not _command_reads_env_file(None)

    def test_cat_env_file_masks_opaque_token(self):
        """cat .env → code_file=False → generic ENV pass redacts opaque keys."""
        from agent.redact import redact_terminal_output
        out = (
            "MISTRAL_API_KEY=abc123opaqueSecretValue\n"
            "NOUS_API_KEY=xyz789opaqueKey\n"
            "DEBUG=true\n"
        )
        red = redact_terminal_output(out, "cat .env")
        assert "abc123opaqueSecretValue" not in red
        assert "xyz789opaqueKey" not in red
        assert "DEBUG=true" in red  # non-secret key preserved

    def test_cat_env_file_with_flags_masks_opaque_token(self):
        """cat -n .env → still detected as .env read."""
        from agent.redact import redact_terminal_output
        out = "     1\tMISTRAL_API_KEY=abc123opaqueSecretValue\n"
        red = redact_terminal_output(out, "cat -n .env")
        assert "abc123opaqueSecretValue" not in red

    def test_cat_env_file_in_pipeline_masks_opaque_token(self):
        """cat .env | grep KEY → still detected as .env read."""
        from agent.redact import redact_terminal_output
        out = "MISTRAL_API_KEY=abc123opaqueSecretValue"
        red = redact_terminal_output(out, "cat .env | grep MISTRAL")
        assert "abc123opaqueSecretValue" not in red

    def test_cat_env_example_not_redacted_as_env(self):
        """cat .env.example → NOT treated as .env read (template file)."""
        from agent.redact import redact_terminal_output
        out = "MISTRAL_API_KEY=placeholder_value_here"
        red = redact_terminal_output(out, "cat .env.example")
        # Should NOT be redacted by the ENV-assignment pass (code_file=True).
        # The placeholder value should survive since it has no vendor prefix.
        assert "placeholder_value_here" in red

    def test_cat_env_local_masks_opaque_token(self):
        """cat .env.local → detected as .env read."""
        from agent.redact import redact_terminal_output
        out = "CUSTOM_API_KEY=opaquecustomkey123456"
        red = redact_terminal_output(out, "cat .env.local")
        assert "opaquecustomkey123456" not in red

    def test_cat_env_inline_comment_preserved(self):
        """Inline comments after env values are preserved (issue #61352 review)."""
        from agent.redact import redact_terminal_output
        out = "MISTRAL_API_KEY=abc123secret # used for tests"
        red = redact_terminal_output(out, "cat .env")
        assert "abc123secret" not in red
        assert "MISTRAL_API_KEY=*** # used for tests" in red

    def test_cat_env_export_inline_comment_preserved(self):
        """export KEY=VALUE # comment — comment preserved."""
        from agent.redact import redact_terminal_output
        out = "export MISTRAL_API_KEY=abc123secret # prod key"
        red = redact_terminal_output(out, "cat .env")
        assert "abc123secret" not in red
        assert "export MISTRAL_API_KEY=*** # prod key" in red



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


class TestMaskSecretControlStripping:
    """Issue #55319/#55321: mask_secret() must not emit control bytes
    (newline, NUL, DEL, C1) in the visible head/tail of a masked secret —
    they corrupt config/status/dump display output."""

    def test_newline_stripped_from_mask(self):
        # The #55319 probe: a newline inside the preserved head.
        assert mask_secret("ab\ncd0123456789zzzz") == "abcd...zzzz"

    def test_c1_control_stripped_from_mask(self):
        assert mask_secret("abcd0123456789zz\x85q") == "abcd...9zzq"

    def test_printable_mask_unchanged(self):
        assert mask_secret("abcdef0123456789zzzz") == "abcd...zzzz"

    def test_all_control_value_returns_empty_fallback(self):
        assert mask_secret("\n\x85\u200b") == ""
        assert mask_secret("\n\x85\u200b", empty="(not set)") == "(not set)"


class TestValueAwareGatingCorpus:
    """Issue #96607: corpus-level before/after for value-aware gating.

    Redaction must mask a keyword-named assignment ONLY when the value has
    credential shape (vendor prefix, hex/base64/high-entropy, or a strong
    credential-specific key name). Bare technical vocabulary — ``token``,
    ``key``, ``cpu`` — in ordinary technical prose/config must pass through
    byte-for-byte, on every assignment family (ENV, dotted config, JSON,
    YAML).
    """

    # Realistic technical prose. On pre-fix main every line was corrupted
    # to ``***`` despite containing no secret.
    TECHNICAL_CORPUS = [
        'IDENTITY_TOKEN="bailu"',
        "--override-tensor per_layer_token_embd.weight=CPU",
        "MAX_TOKENS=4096",
        "runtime.token=local",
        "The tokenizer splits on whitespace; set max_new_tokens=256.",
        "num_key_value_heads=8",
        "token: CPU",
        "llm_load_tensors: per_layer_token_embd.weight=CPU buffer",
    ]

    # Obviously-fake but shape-realistic secrets: every one of these must
    # STAY masked after the gating change (fail-closed on credential shape
    # or strong key names).
    FAKE_SECRET_CORPUS = [
        ("API_KEY=sk-fakefakefakefakefake1234567890abcd", "fakefake"),
        ("GITHUB_TOKEN=ghp_FAKEfakeFAKEfake1234567890fake", "FAKEfake"),
        ("MY_SERVICE_TOKEN=A9f3kZq7Lm2Xw8Rt4Yv6", "A9f3kZq7"),
        ("TOKEN=6f1d2a9c8b3e4f5a6d7c8b9a0e1f2d3c", "6f1d2a9c"),
        ("password=hunter2", "hunter2"),
        ("db_password: hunter2", "hunter2"),
        ("auth_token: 9f8e7d6c5b4a39281706f5e4d3c2b1a0", "9f8e7d6c"),
        ('"token": "Zx9Qw8Er7Ty6Ui5Op4As3"', "Zx9Qw8Er"),
        ("SESSION_TOKEN=shrt", "shrt"),
        ("client_secret=abc", "abc"),
        ("spring.datasource.password=fakePass123", "fakePass123"),
    ]

    def test_technical_prose_survives_intact(self):
        for line in self.TECHNICAL_CORPUS:
            assert redact_sensitive_text(line, force=True) == line, line

    def test_technical_corpus_as_one_block_survives_intact(self):
        # The multi-line shape a model actually reads from tool output.
        block = "\n".join(self.TECHNICAL_CORPUS)
        assert redact_sensitive_text(block, force=True) == block

    def test_shape_realistic_fake_secrets_still_masked(self):
        for line, cleartext in self.FAKE_SECRET_CORPUS:
            result = redact_sensitive_text(line, force=True)
            assert result != line, line
            assert cleartext not in result, line

    def test_mixed_block_masks_only_the_secret_lines(self):
        # Precondition guard: both halves must actually exercise the gate.
        secret_line = "MY_SERVICE_TOKEN=A9f3kZq7Lm2Xw8Rt4Yv6"
        prose_line = 'IDENTITY_TOKEN="bailu"'
        block = f"{prose_line}\n{secret_line}"
        result = redact_sensitive_text(block, force=True)
        assert prose_line in result
        assert "A9f3kZq7Lm2Xw8Rt4Yv6" not in result
