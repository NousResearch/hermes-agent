"""Tests for `_sanitize_tool_error` in model_tools.

Ported from ironclaw#1639 — defense-in-depth on tool exception strings before
they enter the model's `tool` message content. Note that `json.dumps()` in
`handle_function_call` already handles quote/backslash escaping at the wire
layer; this helper exists to strip structural framing tokens the model
itself might react to (XML role tags, CDATA, markdown code fences) and to
cap pathological lengths.
"""
from __future__ import annotations

import json
import inspect

import pytest

from agent.redact import (
    TOOL_SECRET_PLACEHOLDER,
    redact_tool_boundary_text,
    redact_tool_boundary_value,
)
from agent.secret_scope import reset_secret_scope, set_secret_scope
from model_tools import _sanitize_tool_error, _TOOL_ERROR_MAX_LEN
from tools.registry import registry as _registry, tool_error


def _synthetic_canaries():
    """Build completed credential forms only at test runtime."""
    body = "Q7mP" * 12
    return {
        "arbitrary": "stage1b-opaque-" + body,
        "bearer": "Bear" + "er " + body,
        "api_key": "s" + "k-" + body,
        "discord": (
            "https://discord.com/api/"
            + "webhooks/"
            + "1234567890/"
            + body
        ),
        "url_query": (
            "https://example.invalid/private?access_"
            + "token="
            + body
        ),
        "authorization": "Author" + "ization: Basic " + body,
    }


@pytest.fixture()
def synthetic_secret_scope():
    canaries = _synthetic_canaries()
    token = set_secret_scope({"STAGE1B_RUNTIME_SECRET": canaries["arbitrary"]})
    try:
        yield canaries
    finally:
        reset_secret_scope(token)


@pytest.fixture()
def configure_secret_authority():
    """Configure one authority matrix cell and restore global/context state."""
    from agent.secret_scope import is_multiplex_active, set_multiplex_active

    previous_multiplex = is_multiplex_active()
    tokens = []

    def configure(scope, *, multiplex_active):
        set_multiplex_active(multiplex_active)
        token = set_secret_scope(scope)
        tokens.append(token)

    try:
        yield configure
    finally:
        for token in reversed(tokens):
            reset_secret_scope(token)
        set_multiplex_active(previous_multiplex)


def _generic_registry_target():
    excluded = {
        "todo",
        "memory",
        "session_search",
        "delegate_task",
        "execute_code",
        "tool_search",
        "tool_call",
        "tool_describe",
    }
    return next(name for name in _registry.get_all_tool_names() if name not in excluded)


class TestRoleTagStripping:
    def test_strips_tool_call_tags(self):
        out = _sanitize_tool_error("bad <tool_call>injected</tool_call> happened")
        assert "<tool_call>" not in out
        assert "</tool_call>" not in out
        assert "bad injected happened" in out


    def test_strips_role_tags(self):
        # Each of these should be stripped
        for tag in ("system", "assistant", "user", "result", "response", "output", "input"):
            raw = f"prefix <{tag}>hi</{tag}> suffix"
            out = _sanitize_tool_error(raw)
            assert f"<{tag}>" not in out, f"failed to strip <{tag}>"
            assert f"</{tag}>" not in out, f"failed to strip </{tag}>"


    def test_unrelated_xml_kept(self):
        # We intentionally only strip the role-like tag whitelist, not all XML
        out = _sanitize_tool_error("Error parsing <ParseError>line 5</ParseError>")
        assert "<ParseError>" in out


class TestCDATAStripping:
    def test_strips_cdata(self):
        out = _sanitize_tool_error("error: <![CDATA[malicious]]> here")
        assert "<![CDATA[" not in out
        assert "]]>" not in out



class TestCodeFenceStripping:
    def test_strips_leading_fence_with_lang(self):
        out = _sanitize_tool_error("```json\n{\"x\": 1}")
        assert not out.replace("[TOOL_ERROR] ", "").startswith("```")


    def test_strips_bare_fence(self):
        out = _sanitize_tool_error("```\nstuff")
        assert "```" not in out.split("\n")[0]


class TestTruncation:
    def test_caps_long_input(self):
        long = "A" * (_TOOL_ERROR_MAX_LEN * 2)
        out = _sanitize_tool_error(long)
        # Total length is prefix + truncated body
        body = out[len("[TOOL_ERROR] "):]
        assert len(body) == _TOOL_ERROR_MAX_LEN
        assert body.endswith("...")



class TestEnvelope:
    def test_wraps_with_prefix(self):
        out = _sanitize_tool_error("oh no")
        assert out.startswith("[TOOL_ERROR] ")


class TestMandatorySecretRedaction:
    def test_scope_miss_uses_credential_env_when_multiplex_off(
        self, monkeypatch, configure_secret_authority
    ):
        scoped_secret = "opaque-scope-" + ("A7mQ" * 8)
        environment_secret = "opaque-environment-" + ("B8nR" * 8)
        monkeypatch.setenv("STAGE1B_RUNTIME_PASSWORD", environment_secret)
        configure_secret_authority(
            {"STAGE1B_SCOPED_TOKEN": scoped_secret},
            multiplex_active=False,
        )

        out = redact_tool_boundary_text(environment_secret)

        assert out == TOOL_SECRET_PLACEHOLDER

    def test_scope_is_authoritative_and_env_is_excluded_when_multiplex_on(
        self, monkeypatch, configure_secret_authority
    ):
        scoped_secret = "opaque-scope-" + ("C9pS" * 8)
        other_profile_secret = "opaque-other-profile-" + ("D2qT" * 8)
        monkeypatch.setenv("STAGE1B_RUNTIME_PASSWORD", other_profile_secret)
        configure_secret_authority(
            {"STAGE1B_SCOPED_TOKEN": scoped_secret},
            multiplex_active=True,
        )

        out = redact_tool_boundary_text(
            scoped_secret + " | " + other_profile_secret
        )

        assert out == TOOL_SECRET_PLACEHOLDER + " | " + other_profile_secret

    def test_no_scope_uses_credential_env_when_multiplex_off(
        self, monkeypatch, configure_secret_authority
    ):
        environment_secret = "opaque-environment-" + ("E3rU" * 8)
        monkeypatch.setenv("STAGE1B_RUNTIME_PASSWORD", environment_secret)
        configure_secret_authority(None, multiplex_active=False)

        out = redact_tool_boundary_text(environment_secret)

        assert out == TOOL_SECRET_PLACEHOLDER

    def test_no_scope_does_not_widen_to_env_when_multiplex_on(
        self, monkeypatch, configure_secret_authority
    ):
        other_profile_secret = "opaque-other-profile-" + ("F4sV" * 8)
        monkeypatch.setenv("STAGE1B_RUNTIME_PASSWORD", other_profile_secret)
        configure_secret_authority(None, multiplex_active=True)

        out = redact_tool_boundary_text(other_profile_secret)

        assert out == other_profile_secret

    @pytest.mark.parametrize(
        ("scope", "multiplex_active"),
        [
            ({"STAGE1B_SCOPED_TOKEN": "opaque-scope-one"}, True),
            ({"STAGE1B_SCOPED_TOKEN": "opaque-scope-two"}, False),
            (None, True),
            (None, False),
        ],
    )
    def test_runtime_main_credential_is_redacted_in_every_authority_cell(
        self, configure_secret_authority, scope, multiplex_active
    ):
        from agent.auxiliary_client import reset_runtime_main, set_runtime_main

        provider_secret = "opaque-provider-runtime-" + ("G5tW" * 8)
        configure_secret_authority(scope, multiplex_active=multiplex_active)
        runtime_token = set_runtime_main(
            "synthetic-provider",
            "synthetic-model",
            api_key=provider_secret,
        )
        try:
            out = redact_tool_boundary_text(provider_secret)
        finally:
            reset_runtime_main(runtime_token)

        assert out == TOOL_SECRET_PLACEHOLDER

    def test_ordinary_environment_value_is_not_collected(
        self, monkeypatch, configure_secret_authority
    ):
        ordinary_value = "ordinary-runtime-setting-" + ("H6uX" * 8)
        monkeypatch.setenv("STAGE1B_ORDINARY_SETTING", ordinary_value)
        configure_secret_authority(None, multiplex_active=False)

        out = redact_tool_boundary_text(ordinary_value)

        assert out == ordinary_value

    def test_runtime_secret_and_all_pattern_classes_are_fully_redacted(
        self, synthetic_secret_scope
    ):
        raw = " | ".join(synthetic_secret_scope.values())

        out = redact_tool_boundary_text(raw)

        assert TOOL_SECRET_PLACEHOLDER in out
        for canary in synthetic_secret_scope.values():
            assert canary not in out

    @pytest.mark.parametrize(
        "payload_factory",
        [
            lambda secret: "x-" + "api-key: " + secret,
            lambda secret: json.dumps({"api_key": secret}),
            lambda secret: "client_" + "secret: " + secret,
            lambda secret: "12345678:" + secret,
            lambda secret: (
                "-----BEGIN "
                + "PRIVATE KEY-----\n"
                + secret
                + "\n-----END "
                + "PRIVATE KEY-----"
            ),
            lambda secret: "postgres" + "://user:" + secret + "@host/db",
            lambda secret: "https" + "://" + secret + "@example.invalid/",
            lambda secret: "eyJ" + secret,
            lambda secret: "aUtHoRiZaTiOn: Basic " + secret,
        ],
    )
    def test_pattern_fast_paths_preserve_every_gated_family(
        self, payload_factory
    ):
        secret = "Q7mP" * 12
        raw = payload_factory(secret)

        out = redact_tool_boundary_text(raw)

        assert secret not in out
        assert TOOL_SECRET_PLACEHOLDER in out

    @pytest.mark.parametrize(
        "separator",
        [
            " ",
            "\t",
            "\n",
            "\r\n",
            "\v",
            "\f",
            "\N{NO-BREAK SPACE}",
            "\N{NEXT LINE}",
            "\N{EN SPACE}",
            "\N{EM SPACE}",
            "\N{LINE SEPARATOR}",
            "\N{PARAGRAPH SEPARATOR}",
        ],
    )
    @pytest.mark.parametrize("prefix", ["Bearer", "BEARER", "bearer", "bEaReR"])
    def test_bearer_gate_covers_every_regex_whitespace(
        self, separator, prefix
    ):
        import agent.redact as redact_module

        secret = "Q7mP" * 12
        raw = "before " + prefix + separator + secret + ", after"

        assert redact_module._TOOL_BEARER_RE.search(raw)
        assert redact_module._has_tool_bearer_hint(raw.casefold())

        out = redact_tool_boundary_text(raw, _secret_snapshot=())

        assert secret not in out
        assert TOOL_SECRET_PLACEHOLDER in out
        assert out.startswith("before ")
        assert out.endswith(", after")

    @pytest.mark.parametrize("scheme", ["", "https:", "HtTpS:", "custom+v1:"])
    @pytest.mark.parametrize("userinfo_shape", ["bare", "user-password", "punctuated"])
    @pytest.mark.parametrize(
        "authority_tail",
        [
            "host.invalid/",
            "host.invalid:8443/path",
            "host.invalid/path?ordinary=value",
        ],
    )
    @pytest.mark.parametrize("atom_length", [8, 24, 64])
    def test_url_userinfo_gate_covers_authoritative_matcher_domain(
        self, scheme, userinfo_shape, authority_tail, atom_length
    ):
        import agent.redact as redact_module

        atom = ("Q7mP" * 20)[:atom_length]
        if userinfo_shape == "bare":
            userinfo = atom
        elif userinfo_shape == "user-password":
            userinfo = "operator:" + atom
        else:
            userinfo = "operator+tag:" + atom + ".-_~"
        raw = "before " + scheme + "//" + userinfo + "@" + authority_tail + " after"

        assert redact_module._STRICT_URL_USERINFO_RE.search(raw)
        assert redact_module._has_tool_url_userinfo_hint(raw)

        out = redact_tool_boundary_text(raw, _secret_snapshot=())

        assert userinfo not in out
        assert atom not in out
        assert TOOL_SECRET_PLACEHOLDER in out
        assert authority_tail in out
        assert out.startswith("before " + scheme + "//")
        assert out.endswith(" after")

    @pytest.mark.parametrize("delimiter", ["?", "#", "&", ";"])
    @pytest.mark.parametrize(
        "key",
        ["access_token", "api_key", "client%5Fsecret", "refresh%5Ftoken"],
    )
    def test_url_parameter_gate_covers_authoritative_matcher_domain(
        self, delimiter, key
    ):
        import agent.redact as redact_module

        secret = "Q7mP" * 12
        raw = "https://host.invalid/path" + delimiter + key + "=" + secret

        assert redact_module._STRICT_URL_PARAM_RE.search(raw)
        assert redact_module._has_tool_url_param_hint(raw)

        out = redact_tool_boundary_text(raw, _secret_snapshot=())

        assert secret not in out
        assert TOOL_SECRET_PLACEHOLDER in out
        assert "https://host.invalid/path" in out

    def test_matcher_inventory_has_no_excluding_dispatch_gate(self):
        import agent.redact as redact_module

        secret = "Q7mP" * 12
        cases = [
            (
                "vendor-prefix",
                redact_module._PREFIX_RE,
                lambda raw: redact_module._has_known_prefix_substring(raw),
                "s" + "k-" + secret,
            ),
            (
                "discord-webhook",
                redact_module._TOOL_DISCORD_WEBHOOK_RE,
                lambda raw: "discord" in raw.casefold() and "/webhooks/" in raw.casefold(),
                "https://discord.com/api/" + "webhooks/1234567890/" + secret,
            ),
            (
                "bearer",
                redact_module._TOOL_BEARER_RE,
                lambda raw: redact_module._has_tool_bearer_hint(raw.casefold()),
                "Bear" + "er\u2003" + secret,
            ),
            (
                "authorization",
                redact_module._AUTH_HEADER_RE,
                lambda raw: "authorization" in raw.casefold(),
                "Author" + "ization: Basic " + secret,
            ),
            (
                "secret-header",
                redact_module._SECRET_HEADER_RE,
                lambda raw: ":" in raw,
                "x-api" + "-key: " + secret,
            ),
            (
                "uppercase-env",
                redact_module._ENV_ASSIGN_RE,
                lambda raw: "=" in raw and redact_module._has_tool_config_hint(raw.casefold()),
                "OPENAI_API_KEY=" + secret,
            ),
            (
                "lowercase-env",
                redact_module._ENV_ASSIGN_LOWER_RE,
                lambda raw: "=" in raw and redact_module._has_tool_config_hint(raw.casefold()),
                "openai_key=" + secret,
            ),
            (
                "dotted-config",
                redact_module._CFG_DOTTED_RE,
                lambda raw: (
                    "=" in raw
                    and redact_module._has_tool_config_hint(raw.casefold())
                    and bool(redact_module._CFG_SECRET_WORD_RE.search(raw))
                ),
                "service.auth.token=" + secret,
            ),
            (
                "anchored-config",
                redact_module._CFG_ANCHORED_RE,
                lambda raw: (
                    "=" in raw
                    and redact_module._has_tool_config_hint(raw.casefold())
                    and bool(redact_module._CFG_SECRET_WORD_RE.search(raw))
                ),
                "password=" + secret,
            ),
            (
                "json-field",
                redact_module._JSON_FIELD_RE,
                lambda raw: (
                    ":" in raw
                    and '"' in raw
                    and redact_module._has_tool_config_hint(raw.casefold())
                ),
                '{"secret_value": "' + secret + '"}',
            ),
            (
                "yaml-field",
                redact_module._YAML_ASSIGN_RE,
                lambda raw: (
                    ":" in raw
                    and redact_module._has_tool_config_hint(raw.casefold())
                ),
                "client_secret: " + secret,
            ),
            (
                "private-key",
                redact_module._PRIVATE_KEY_RE,
                lambda raw: "BEGIN" in raw and "PRIVATE KEY" in raw,
                "-----BEGIN PRIVATE KEY-----\n" + secret + "\n-----END PRIVATE KEY-----",
            ),
            (
                "telegram",
                redact_module._TELEGRAM_RE,
                lambda raw: ":" in raw and redact_module._has_tool_telegram_hint(raw),
                "12345678:" + secret,
            ),
            (
                "database-url",
                redact_module._DB_CONNSTR_RE,
                lambda raw: "://" in raw,
                "postgresql://operator:" + secret + "@host.invalid/db",
            ),
            (
                "bare-userinfo",
                redact_module._URL_BARE_TOKEN_RE,
                lambda raw: "://" in raw,
                "https://" + secret + "@host.invalid/path",
            ),
            (
                "jwt",
                redact_module._JWT_RE,
                lambda raw: "eyJ" in raw,
                "eyJ" + secret + ".abcd.efgh",
            ),
            (
                "strict-url-param",
                redact_module._STRICT_URL_PARAM_RE,
                redact_module._has_tool_url_param_hint,
                "https://host.invalid/?access_token=" + secret,
            ),
            (
                "strict-url-userinfo",
                redact_module._STRICT_URL_USERINFO_RE,
                redact_module._has_tool_url_userinfo_hint,
                "//operator:" + secret + "@host.invalid/path",
            ),
        ]

        for family, matcher, gate, raw in cases:
            assert matcher.search(raw), family
            assert gate(raw), family
            out = redact_tool_boundary_text(raw, _secret_snapshot=())
            assert secret not in out, family
            assert TOOL_SECRET_PLACEHOLDER in out, family

    def test_positive_prefilters_cover_deterministic_generated_match_corpus(self):
        import agent.redact as redact_module

        secret = "Q7mP" * 12
        config_cases = []
        for key in (
            "OPENAI_API_KEY",
            "SERVICE_TOKEN",
            "CLIENT_SECRET",
            "DATABASE_PASSWORD",
            "MYSQL_PASS",
            "DB_PW",
            "AUTH_CREDENTIAL",
            "openai_key",
            "service_token",
            "client_secret",
            "database_password",
            "mysql_pass",
            "db_pw",
            "auth_credential",
            "service.api.key",
            "service.auth.token",
        ):
            config_cases.extend((key + "=" + secret, key + ": " + secret))
        for key in (
            "apiKey",
            "token",
            "secret",
            "password",
            "access_token",
            "refresh_token",
            "auth_token",
            "bearer",
            "secret_value",
            "raw_secret",
            "secret_input",
            "key_material",
        ):
            config_cases.append('{"' + key + '": "' + secret + '"}')

        config_matchers = (
            redact_module._ENV_ASSIGN_RE,
            redact_module._ENV_ASSIGN_LOWER_RE,
            redact_module._CFG_DOTTED_RE,
            redact_module._CFG_ANCHORED_RE,
            redact_module._JSON_FIELD_RE,
            redact_module._YAML_ASSIGN_RE,
        )
        matched_config_cases = 0
        for raw in config_cases:
            if any(matcher.search(raw) for matcher in config_matchers):
                matched_config_cases += 1
                assert redact_module._has_tool_config_hint(raw.casefold()), raw
        assert matched_config_cases == 39

        telegram_cases = [
            prefix + str(10 ** (digits - 1)) + ":" + secret
            for prefix in ("", "bot")
            for digits in range(8, 16)
        ]
        for raw in telegram_cases:
            assert redact_module._TELEGRAM_RE.search(raw)
            assert redact_module._has_tool_telegram_hint(raw)

    def test_dispatch_source_has_no_whole_text_url_negative_gate(self):
        import agent.redact as redact_module

        source = inspect.getsource(redact_module.redact_tool_boundary_text)

        assert 'if "://" not in text' not in source
        assert 'and "://" not in text' not in source

    @pytest.mark.parametrize(
        "ordinary",
        [
            'IDENTITY_TOKEN="bailu"',
            'runtime.token="local"',
            "token: CPU",
            "tensor token dimensions and tokenizer cache",
            "ordinary technical token/key/cache/session text",
        ],
    )
    def test_ambiguous_technical_values_remain_visible(self, ordinary):
        assert redact_tool_boundary_text(ordinary, _secret_snapshot=()) == ordinary

    @staticmethod
    def _mixed_content_cases(secret):
        docs = "https://docs.example.invalid/reference?mode=public"
        return [
            ("lowercase-assignment", "openai_key=" + secret, docs),
            ("dotted-config", "service.auth.token=" + secret, docs),
            ("anchored-config", "\npassword=" + secret, docs),
            ("yaml-config", "\nclient_secret: " + secret, docs),
            (
                "bearer-json",
                "Bear" + "er\n" + secret,
                '{"mode": "public"} ' + docs,
            ),
            (
                "url-userinfo-env",
                "https://operator:" + secret + "@private.invalid/path",
                "RUNTIME_MODE=local " + docs,
            ),
            (
                "url-param-ordinary-config",
                "https://private.invalid/?access_token=" + secret,
                'runtime.token="local" ' + docs,
            ),
            (
                "multiple-credential-families",
                "Author" + "ization: Basic " + secret,
                "x-api" + "-key: " + secret + " " + docs,
            ),
            (
                "stdout-stack-url",
                "stdout: x-api" + "-key: " + secret,
                "stack=" + docs,
            ),
            (
                "stderr-documentation-url",
                "stderr: Author" + "ization: Token " + secret,
                "documentation=" + docs,
            ),
        ]

    @pytest.mark.parametrize("order", ["credential-first", "credential-last"])
    @pytest.mark.parametrize("separator", [" ", "\n", "\ntrace line\n"])
    def test_mixed_content_never_suppresses_an_independent_matcher(
        self, order, separator
    ):
        secret = "Q7mP" * 12

        for family, credential, ordinary in self._mixed_content_cases(secret):
            parts = (credential, ordinary)
            if order == "credential-last":
                parts = tuple(reversed(parts))
            raw = separator.join(parts)

            out = redact_tool_boundary_text(raw, _secret_snapshot=())

            assert secret not in out, family
            assert TOOL_SECRET_PLACEHOLDER in out, family
            assert "docs.example.invalid" in out, family

    def test_mixed_content_nested_mapping_and_list_are_independent(self):
        secret = "Q7mP" * 12
        raw = {
            "stdout": [
                credential
                for _family, credential, _ordinary in self._mixed_content_cases(secret)
            ],
            "diagnostic": {
                "urls": [
                    ordinary
                    for _family, _credential, ordinary in self._mixed_content_cases(secret)
                ]
            },
        }

        out = redact_tool_boundary_value(raw, _secret_snapshot=())
        serialized = repr(out)

        assert secret not in serialized
        assert serialized.count(TOOL_SECRET_PLACEHOLDER) >= 10
        assert serialized.count("docs.example.invalid") == 10

    def test_exception_message_is_redacted(self, synthetic_secret_scope):
        exc = RuntimeError(synthetic_secret_scope["arbitrary"])

        out = redact_tool_boundary_value(exc)

        assert synthetic_secret_scope["arbitrary"] not in out
        assert out == TOOL_SECRET_PLACEHOLDER

    def test_context_local_provider_credential_is_redacted_without_env_or_scope(self):
        from agent.auxiliary_client import reset_runtime_main, set_runtime_main

        provider_secret = "provider-runtime-" + ("T9wC" * 12)
        runtime_token = set_runtime_main(
            "synthetic-provider",
            "synthetic-model",
            api_key=provider_secret,
        )
        try:
            out = redact_tool_boundary_text("provider rejected " + provider_secret)
        finally:
            reset_runtime_main(runtime_token)

        assert provider_secret not in out
        assert out == "provider rejected " + TOOL_SECRET_PLACEHOLDER

    def test_nested_dict_list_tuple_and_decodable_bytes_are_redacted(
        self, synthetic_secret_scope
    ):
        raw = {
            "stdout": synthetic_secret_scope["arbitrary"],
            "stderr": synthetic_secret_scope["authorization"].encode(),
            synthetic_secret_scope["url_query"]: "secret-bearing mapping key",
            "nested": [
                synthetic_secret_scope["api_key"],
                ({"url": synthetic_secret_scope["discord"]},),
            ],
        }

        out = redact_tool_boundary_value(raw)
        serialized = repr(out)

        for canary in synthetic_secret_scope.values():
            assert canary not in serialized
        assert any(TOOL_SECRET_PLACEHOLDER in str(key) for key in out)
        assert isinstance(out["stderr"], bytes)
        assert TOOL_SECRET_PLACEHOLDER.encode() in out["stderr"]

    def test_nested_projection_captures_authority_once(
        self, monkeypatch, synthetic_secret_scope
    ):
        import agent.redact as redact_module

        calls = 0
        original = redact_module._runtime_loaded_secret_values

        def counted_snapshot():
            nonlocal calls
            calls += 1
            return original()

        monkeypatch.setattr(
            redact_module,
            "_runtime_loaded_secret_values",
            counted_snapshot,
        )
        raw = {
            "stdout": synthetic_secret_scope["arbitrary"],
            "items": [
                {"stderr": synthetic_secret_scope["authorization"]}
                for _ in range(12)
            ],
        }

        out = redact_tool_boundary_value(raw)

        assert calls == 1
        assert synthetic_secret_scope["arbitrary"] not in repr(out)
        assert synthetic_secret_scope["authorization"] not in repr(out)

    def test_snapshot_is_not_reused_across_top_level_invocations(
        self, configure_secret_authority
    ):
        first = "opaque-first-authority-" + ("J7vZ" * 8)
        second = "opaque-second-authority-" + ("K8wA" * 8)
        configure_secret_authority(
            {"STAGE1B_SCOPED_TOKEN": first},
            multiplex_active=True,
        )
        assert redact_tool_boundary_value(first) == TOOL_SECRET_PLACEHOLDER

        configure_secret_authority(
            {"STAGE1B_SCOPED_TOKEN": second},
            multiplex_active=True,
        )
        out = redact_tool_boundary_value(first + " | " + second)

        assert out == first + " | " + TOOL_SECRET_PLACEHOLDER

    def test_non_utf8_bytes_are_preserved(self):
        raw = b"\xff\xfe\x00"

        assert redact_tool_boundary_value(raw) is raw

    def test_multiline_stdout_and_stderr_redact_every_line(
        self, synthetic_secret_scope
    ):
        raw = (
            "stdout=" + synthetic_secret_scope["arbitrary"] + "\n"
            "stderr=" + synthetic_secret_scope["authorization"] + "\n"
            "diagnostic=permission denied"
        )

        out = redact_tool_boundary_text(raw)

        assert synthetic_secret_scope["arbitrary"] not in out
        assert synthetic_secret_scope["authorization"] not in out
        assert "diagnostic=permission denied" in out

    def test_idempotent_fixed_placeholder(self, synthetic_secret_scope):
        once = redact_tool_boundary_text(
            synthetic_secret_scope["arbitrary"]
            + " "
            + synthetic_secret_scope["api_key"]
        )

        assert redact_tool_boundary_text(once) == once
        assert once == f"{TOOL_SECRET_PLACEHOLDER} {TOOL_SECRET_PLACEHOLDER}"

    def test_preserves_non_secret_diagnostics(self):
        raw = (
            "HTTP 403 during collect stage; PermissionError; "
            "request=session-safe-id; retry timeout"
        )

        assert redact_tool_boundary_text(raw) == raw

    def test_force_boundary_ignores_global_redaction_opt_out(
        self, monkeypatch, synthetic_secret_scope
    ):
        import agent.redact as redact_module

        monkeypatch.setattr(redact_module, "_REDACT_ENABLED", False)

        out = redact_tool_boundary_text(synthetic_secret_scope["arbitrary"])

        assert out == TOOL_SECRET_PLACEHOLDER

    def test_tool_error_uses_fixed_mandatory_boundary(self, synthetic_secret_scope):
        out = tool_error(
            synthetic_secret_scope["arbitrary"],
            stderr=synthetic_secret_scope["authorization"],
        )
        payload = json.loads(out)

        assert payload["error"] == TOOL_SECRET_PLACEHOLDER
        assert synthetic_secret_scope["authorization"] not in payload["stderr"]
        assert TOOL_SECRET_PLACEHOLDER in payload["stderr"]

    def test_tool_error_message_and_extra_share_one_immediate_snapshot(
        self, monkeypatch, synthetic_secret_scope
    ):
        import agent.redact as redact_module

        calls = 0
        original = redact_module._runtime_loaded_secret_values

        def counted_snapshot():
            nonlocal calls
            calls += 1
            return original()

        monkeypatch.setattr(
            redact_module,
            "_runtime_loaded_secret_values",
            counted_snapshot,
        )

        payload = json.loads(
            tool_error(
                synthetic_secret_scope["arbitrary"],
                stderr=synthetic_secret_scope["authorization"],
            )
        )

        assert calls == 1
        assert payload["error"] == TOOL_SECRET_PLACEHOLDER
        assert payload["stderr"].endswith(TOOL_SECRET_PLACEHOLDER)

    def test_registry_exception_result_and_log_are_redacted(
        self, synthetic_secret_scope, caplog
    ):
        target = _generic_registry_target()
        entry = _registry.get_entry(target)
        original_handler, original_async = entry.handler, entry.is_async

        def boom(_args, **_kwargs):
            raise RuntimeError(
                synthetic_secret_scope["arbitrary"]
                + " "
                + synthetic_secret_scope["discord"]
            )

        entry.handler, entry.is_async = boom, False
        try:
            with caplog.at_level("ERROR"):
                out = _registry.dispatch(target, {})
        finally:
            entry.handler, entry.is_async = original_handler, original_async

        combined = out + caplog.text
        assert synthetic_secret_scope["arbitrary"] not in combined
        assert synthetic_secret_scope["discord"] not in combined
        assert "RuntimeError" in combined
        assert "stack=" in caplog.text
        assert TOOL_SECRET_PLACEHOLDER in combined

    def test_registry_structured_result_redacts_stdout_stderr_and_nested_values(
        self, synthetic_secret_scope
    ):
        target = _generic_registry_target()
        entry = _registry.get_entry(target)
        original_handler, original_async = entry.handler, entry.is_async

        def structured(_args, **_kwargs):
            return json.dumps(
                {
                    "stdout": synthetic_secret_scope["arbitrary"],
                    "stderr": synthetic_secret_scope["authorization"],
                    "nested": {"items": [synthetic_secret_scope["api_key"]]},
                }
            )

        entry.handler, entry.is_async = structured, False
        try:
            out = _registry.dispatch(target, {})
        finally:
            entry.handler, entry.is_async = original_handler, original_async

        for canary in synthetic_secret_scope.values():
            assert canary not in out
        assert out.count(TOOL_SECRET_PLACEHOLDER) == 3




class TestHandleFunctionCallIntegration:
    """Verify handle_function_call routes exception-path errors through the sanitizer.

    Note: the "Unknown tool: ..." early-return in tools/registry.py is a
    *different* code path from `except Exception` in handle_function_call —
    that one returns directly without sanitization (and there's nothing to
    sanitize in a hardcoded format string anyway). This test exercises the
    real exception path by passing args that make a known tool raise.
    """

    def test_exception_path_error_is_sanitized(self):
        import json
        from model_tools import handle_function_call
        from tools.registry import registry as _registry

        # Force a known tool to raise with a payload containing role tags.
        def boom(_args, **_kwargs):
            raise RuntimeError("<tool_call>injected</tool_call> boom")

        all_tools = _registry.get_all_tool_names()
        assert all_tools, "no tools registered — test environment broken"
        target = all_tools[0]
        original = _registry._tools[target].handler
        _registry._tools[target].handler = boom
        try:
            result_str = handle_function_call(target, {})
        finally:
            _registry._tools[target].handler = original

        payload = json.loads(result_str)
        assert "error" in payload, payload
        assert payload["error"].startswith("[TOOL_ERROR] "), payload["error"]
        # Role-tag stripping carried through
        assert "<tool_call>" not in payload["error"]
        assert "</tool_call>" not in payload["error"]
        assert "boom" in payload["error"]
