"""Provider-egress exact-secret regression gates for issue #77487."""

from __future__ import annotations

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from agent.agent_runtime_helpers import sanitize_api_messages


ALTERNATE_JSON_SECRET = "S/Key"
ALTERNATE_JSON_SPELLING = r"\u0053\/K\u0065y"


def _nest_json_string_spelling(spelling: str, layers: int) -> str:
    for _ in range(layers - 1):
        spelling = json.dumps(spelling)[1:-1]
    return spelling


def _nested_alternate_json_spelling(layers: int) -> str:
    return _nest_json_string_spelling(ALTERNATE_JSON_SPELLING, layers)


ALTERNATE_JSON_SPELLINGS = tuple(
    pytest.param(
        _nested_alternate_json_spelling(layers),
        id=f"json-escape-layers-{layers}",
    )
    for layers in (1, 2, 5, 10, 13)
)
MALFORMED_JSON_FRAGMENT_KINDS = (
    "unterminated",
    "invalid_escape",
    "quote_parity",
    "unquoted",
    "single_quoted",
)


def _json_with_alternate_secret(
    field: str, spelling: str = ALTERNATE_JSON_SPELLING
) -> str:
    return f'{{"{field}":"{spelling}"}}'


def _malformed_json_with_alternate_secret(field: str, kind: str) -> str:
    if kind == "unterminated":
        return f'{{"{field}":"{ALTERNATE_JSON_SPELLING}'
    if kind == "invalid_escape":
        return f'{{"{field}":"\\q{ALTERNATE_JSON_SPELLING}"}}'
    if kind == "quote_parity":
        return f'{{"broken":"prefix "{field}":"{ALTERNATE_JSON_SPELLING}"}}'
    if kind == "unquoted":
        return f"{{{field}:{ALTERNATE_JSON_SPELLING}}}"
    if kind == "single_quoted":
        return f"{{'{field}':'{ALTERNATE_JSON_SPELLING}'}}"
    raise AssertionError(f"unknown malformed JSON fragment kind: {kind}")


@pytest.fixture(autouse=True)
def _enable_redaction(monkeypatch):
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


@pytest.fixture
def applied_secret_home(tmp_path, monkeypatch):
    from hermes_cli import env_loader
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    secret = "opaque-fixture-secret-77487"
    home_key = str(tmp_path.resolve())
    monkeypatch.setitem(
        env_loader._SECRET_SOURCE_VALUES_BY_HOME,
        home_key,
        {"ARBITRARY_SOURCE_NAME": secret},
    )
    token = set_hermes_home_override(tmp_path)
    try:
        yield tmp_path, secret
    finally:
        reset_hermes_home_override(token)


@pytest.fixture
def json_special_secret_home(tmp_path, monkeypatch):
    from hermes_cli import env_loader
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    secret = 'Q7"\\Z!'
    home_key = str(tmp_path.resolve())
    monkeypatch.setitem(
        env_loader._SECRET_SOURCE_VALUES_BY_HOME,
        home_key,
        {"JSON_SPECIAL_SECRET": secret},
    )
    token = set_hermes_home_override(tmp_path)
    try:
        yield tmp_path, secret
    finally:
        reset_hermes_home_override(token)


def _paired_messages(secret: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_77487",
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "arguments": json.dumps(
                            {"command": f"printf '{secret} {secret}'"}
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "terminal",
            "tool_call_id": "call_77487",
            "content": [
                {"type": "text", "text": f"first={secret}; second={secret}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"https://example.invalid/{secret}/{secret}/image.png"
                    },
                },
            ],
        },
    ]


def test_provider_copy_masks_content_occurrences_and_preserves_replay_history(
    applied_secret_home,
):
    """Text and non-text metadata are sanitized without mutating history."""
    _home, secret = applied_secret_home
    messages = _paired_messages(secret)
    original = copy.deepcopy(messages)

    provider_bound = sanitize_api_messages(messages)

    arguments = provider_bound[0]["tool_calls"][0]["function"]["arguments"]
    text = provider_bound[1]["content"][0]["text"]
    image_url = provider_bound[1]["content"][1]["image_url"]["url"]
    # #43083: replayed executable arguments cannot be display-masked without
    # making the model copy the placeholder into a later tool call. This field
    # requires authenticated vault/token references, not raw substitution.
    assert secret in arguments
    assert secret not in text
    assert secret not in image_url
    assert text.count("***") == 2
    assert image_url.count("***") == 2

    # #43083: provider sanitization must not poison the replayable source.
    assert messages == original
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == original[0][
        "tool_calls"
    ][0]["function"]["arguments"]


def test_provider_copy_masks_json_encoded_tool_result_without_mutating_source(
    json_special_secret_home,
):
    """JSON string escaping cannot hide an exact secret from provider egress."""
    _home, secret = json_special_secret_home
    arguments = json.dumps({"command": f"printf {secret}"})
    tool_result = json.dumps(
        {"result": {"credential": secret, "items": ["safe", secret]}},
        separators=(",", ":"),
    )
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-json-77487",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": arguments},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-json-77487",
            "content": tool_result,
        },
    ]
    original = copy.deepcopy(messages)

    provider_bound = sanitize_api_messages(messages)

    encoded_secret = json.dumps(secret)[1:-1]
    provider_result = provider_bound[1]["content"]
    assert encoded_secret not in provider_result
    assert json.loads(provider_result) == {
        "result": {"credential": "***", "items": ["safe", "***"]}
    }
    # Executable provider arguments and caller-owned history stay byte-exact.
    assert provider_bound[0]["tool_calls"][0]["function"]["arguments"] == arguments
    assert messages == original


@pytest.mark.parametrize("alternate_spelling", ALTERNATE_JSON_SPELLINGS)
def test_provider_copy_masks_equivalent_json_escape_spellings_without_mutation(
    monkeypatch, alternate_spelling
):
    """Escaped solidus and Unicode spellings decode only to masked values."""
    from hermes_cli import env_loader
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    monkeypatch.setitem(
        env_loader._SECRET_SOURCE_VALUES_BY_HOME,
        str(home.resolve()),
        {"ALTERNATE_JSON_SECRET": ALTERNATE_JSON_SECRET},
    )
    content = _json_with_alternate_secret("credential", alternate_spelling)
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-alternate-provider-77487",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": content},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-alternate-provider-77487",
            "content": content,
        },
    ]
    original = copy.deepcopy(messages)

    provider_bound = sanitize_api_messages(messages)

    provider_content = provider_bound[1]["content"]
    assert json.loads(provider_content) == {"credential": "***"}
    assert ALTERNATE_JSON_SECRET not in provider_content
    assert ALTERNATE_JSON_SPELLING not in provider_content
    assert alternate_spelling not in provider_content
    assert provider_bound[0]["tool_calls"][0]["function"]["arguments"] == content
    assert messages == original


@pytest.mark.parametrize("fragment_kind", MALFORMED_JSON_FRAGMENT_KINDS)
def test_provider_copy_masks_malformed_alternate_json_fragments_without_mutation(
    monkeypatch, fragment_kind
):
    """Incomplete or invalid quoting cannot bypass disposable provider masking."""
    from hermes_cli import env_loader
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    monkeypatch.setitem(
        env_loader._SECRET_SOURCE_VALUES_BY_HOME,
        str(home.resolve()),
        {"MALFORMED_JSON_SECRET": ALTERNATE_JSON_SECRET},
    )
    content = _malformed_json_with_alternate_secret("credential", fragment_kind)
    messages = [{"role": "tool", "content": content}]
    original = copy.deepcopy(messages)

    provider_bound = sanitize_api_messages(messages)

    assert "***" in provider_bound[0]["content"]
    assert ALTERNATE_JSON_SECRET not in provider_bound[0]["content"]
    assert ALTERNATE_JSON_SPELLING not in provider_bound[0]["content"]
    assert messages == original


def test_valid_json_masks_only_string_values_and_preserves_structural_tokens(
    tmp_path, monkeypatch
):
    """Credential-shaped punctuation cannot corrupt a valid JSON document."""
    from agent.redact import redact_provider_message_values
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    secrets = {
        "OPEN_TOKEN": "{",
        "QUOTE_TOKEN": '"',
        "COLON_TOKEN": ":",
        "COMMA_TOKEN": ",",
        "TRUE_TOKEN": "true",
        "NULL_TOKEN": "null",
        "NUMBER_TOKEN": "1",
        "PAYLOAD_TOKEN": "actual-secret",
    }
    content = (
        '{"true":true,"null":null,"number":1,'
        '"items":["actual-secret","true","null","1","{","\\\"",":",","]}'
    )
    source = [{"role": "tool", "content": content}]
    original = copy.deepcopy(source)

    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    token = set_secret_scope(secrets)
    try:
        provider_bound = redact_provider_message_values(source, home=tmp_path)
    finally:
        reset_secret_scope(token)

    assert json.loads(provider_bound[0]["content"]) == {
        "true": True,
        "null": None,
        "number": 1,
        "items": ["***"] * 8,
    }
    assert source == original


def test_valid_json_masks_authoritative_object_key(tmp_path, monkeypatch):
    """A credential used as a JSON key is still removed before provider egress."""
    from agent.redact import redact_provider_message_values
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    secret = "credential-key-77487-super-long-value"
    second_secret = "another-long-credential-key-77487"
    numeric_secret = "839201774875551234567890123456"
    source = [
        {
            "role": "tool",
            "content": json.dumps(
                {
                    "***": "ordinary",
                    secret: "tool-result",
                    second_secret: "second-result",
                    numeric_secret: "numeric-result",
                }
            ),
        }
    ]
    original = copy.deepcopy(source)

    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    token = set_secret_scope(
        {
            "API_TOKEN": secret,
            "SECOND_API_TOKEN": second_secret,
            "NUMERIC_API_TOKEN": numeric_secret,
        }
    )
    try:
        provider_bound = redact_provider_message_values(source, home=tmp_path)
    finally:
        reset_secret_scope(token)

    redacted_object = json.loads(provider_bound[0]["content"])
    assert secret not in provider_bound[0]["content"]
    assert second_secret not in provider_bound[0]["content"]
    assert numeric_secret not in provider_bound[0]["content"]
    assert sorted(redacted_object.values()) == [
        "numeric-result",
        "ordinary",
        "second-result",
        "tool-result",
    ]
    assert len(redacted_object) == 4
    assert source == original


def test_valid_json_key_marker_avoids_later_existing_key(tmp_path, monkeypatch):
    """Key-marker selection is independent of object-key ordering."""
    from agent.redact import redact_provider_message_values
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    secret = "credential-key-77487-reverse-order"
    source = [
        {
            "role": "tool",
            "content": json.dumps({secret: "sensitive", "***": "ordinary"}),
        }
    ]
    original = copy.deepcopy(source)

    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    token = set_secret_scope({"API_TOKEN": secret})
    try:
        provider_bound = redact_provider_message_values(source, home=tmp_path)
    finally:
        reset_secret_scope(token)

    redacted_object = json.loads(provider_bound[0]["content"])
    assert secret not in provider_bound[0]["content"]
    assert sorted(redacted_object.values()) == ["ordinary", "sensitive"]
    assert len(redacted_object) == 2
    assert source == original


def test_valid_json_key_marker_avoids_partial_existing_key(tmp_path, monkeypatch):
    """A marker cannot compose an existing key at a replacement boundary."""
    from agent.redact import redact_provider_message_values
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    secret = "credential-key-77487"
    source = [
        {
            "role": "tool",
            "content": json.dumps(
                {
                    f"prefix{secret}": "sensitive",
                    "prefix***": "ordinary",
                }
            ),
        }
    ]
    original = copy.deepcopy(source)

    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    token = set_secret_scope({"API_TOKEN": secret})
    try:
        provider_bound = redact_provider_message_values(source, home=tmp_path)
    finally:
        reset_secret_scope(token)

    redacted_object = json.loads(provider_bound[0]["content"])
    assert secret not in provider_bound[0]["content"]
    assert sorted(redacted_object.values()) == ["ordinary", "sensitive"]
    assert len(redacted_object) == 2
    assert source == original


def test_invalid_json_keeps_aggressive_structural_exact_masking(
    tmp_path, monkeypatch
):
    """Malformed/non-JSON text retains the prior fail-closed raw scanner."""
    from agent.redact import redact_known_secret_values
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    source = '{"flag":true'
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    token = set_secret_scope({"TRUE_TOKEN": "true"})
    try:
        redacted = redact_known_secret_values(source, home=tmp_path)
    finally:
        reset_secret_scope(token)

    assert redacted == '{"flag":***'
    assert source == '{"flag":true'


def test_authoritative_short_secret_is_masked_even_when_concatenated(
    tmp_path, monkeypatch
):
    from agent.redact import redact_known_secret_values
    from hermes_cli import env_loader

    monkeypatch.setitem(
        env_loader._SECRET_SOURCE_VALUES_BY_HOME,
        str(tmp_path.resolve()),
        {"ARBITRARY_EXTERNAL_NAME": "p4ss"},
    )

    assert redact_known_secret_values(
        "p4ss field, p4ss123 and myp4ss all contain the credential",
        home=tmp_path,
    ) == "*** field, ***123 and my*** all contain the credential"


def test_literal_only_exact_mask_skips_escape_scanner(monkeypatch):
    """Plain text keeps exact masking without paying for escape decoding."""
    from agent.redact import _compile_exact_secret_pattern, _redact_exact_with_pattern

    pattern = _compile_exact_secret_pattern(("p4ss",))
    monkeypatch.setattr(
        "agent.redact._scan_json_string_fragment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("escape scanner should not run")
        ),
    )

    assert _redact_exact_with_pattern("plain p4ss text", pattern) == "plain *** text"
    assert _redact_exact_with_pattern("plain safe text", pattern) == "plain safe text"


def test_nested_escape_work_budget_exhaustion_fails_closed(monkeypatch):
    """An uninspected derived layer is removed instead of forwarded."""
    from agent.redact import _compile_exact_secret_pattern, _redact_exact_with_pattern

    pattern = _compile_exact_secret_pattern((ALTERNATE_JSON_SECRET,))
    source = _json_with_alternate_secret(
        "credential", _nested_alternate_json_spelling(13)
    )
    original = source
    monkeypatch.setattr(
        "agent.redact._json_escape_decode_work_budget",
        lambda text: len(text),
    )

    redacted = _redact_exact_with_pattern(source, pattern)

    assert json.loads(redacted) == {"credential": "***"}
    assert ALTERNATE_JSON_SECRET not in redacted
    assert ALTERNATE_JSON_SPELLING not in redacted
    assert source == original


def test_exact_mask_chooses_marker_that_cannot_collide_with_active_secrets(
    tmp_path, monkeypatch
):
    """Even marker-shaped and one-character credentials remain absent."""
    from agent.redact import (
        _compile_exact_secret_pattern,
        redact_provider_message_values,
    )
    from hermes_cli import env_loader

    secrets = ("***", "*", "[REDACTED]", "<redacted-secret>")
    monkeypatch.setitem(
        env_loader._SECRET_SOURCE_VALUES_BY_HOME,
        str(tmp_path.resolve()),
        {f"COLLISION_SECRET_{index}": secret for index, secret in enumerate(secrets)},
    )
    matcher = _compile_exact_secret_pattern(secrets)
    assert matcher is not None
    assert matcher.replacement
    assert all(
        secret not in matcher.replacement and matcher.replacement not in secret
        for secret in secrets
    )

    messages = [{"role": "tool", "content": " | ".join(secrets)}]
    original = copy.deepcopy(messages)
    provider_bound = redact_provider_message_values(messages, home=tmp_path)

    assert all(secret not in provider_bound[0]["content"] for secret in secrets)
    assert provider_bound[0]["content"] != original[0]["content"]
    assert messages == original


def test_provider_marker_repetition_cannot_synthesize_active_secret(
    tmp_path, monkeypatch
):
    """Adjacent replacements remain safe under arbitrary marker composition."""
    from hermes_cli import env_loader
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    active_secrets = ("***", "REDACTED][REDACTED")
    monkeypatch.setitem(
        env_loader._SECRET_SOURCE_VALUES_BY_HOME,
        str(tmp_path.resolve()),
        {f"MARKER_SECRET_{index}": secret for index, secret in enumerate(active_secrets)},
    )
    source = [{"role": "tool", "content": "******"}]
    original = copy.deepcopy(source)
    token = set_hermes_home_override(tmp_path)
    try:
        provider_bound = sanitize_api_messages(source)
    finally:
        reset_hermes_home_override(token)

    content = provider_bound[0]["content"]
    assert all(secret not in content for secret in active_secrets)
    assert content != "[REDACTED][REDACTED]"
    assert source == original


def test_provider_marker_cannot_compose_with_unchanged_source_context(
    tmp_path, monkeypatch
):
    """A replacement boundary cannot synthesize another active secret."""
    from agent.redact import _compile_exact_secret_pattern
    from hermes_cli import env_loader
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    active_secrets = ("x", "ab*")
    matcher = _compile_exact_secret_pattern(active_secrets)
    assert matcher is not None
    assert matcher.replacement == "[REDACTED]"

    monkeypatch.setitem(
        env_loader._SECRET_SOURCE_VALUES_BY_HOME,
        str(tmp_path.resolve()),
        {f"BOUNDARY_SECRET_{index}": secret for index, secret in enumerate(active_secrets)},
    )
    source = [{"role": "tool", "content": "abx"}]
    original = copy.deepcopy(source)
    token = set_hermes_home_override(tmp_path)
    try:
        provider_bound = sanitize_api_messages(source)
    finally:
        reset_hermes_home_override(token)

    assert all(secret not in provider_bound[0]["content"] for secret in active_secrets)
    assert provider_bound[0]["content"] == f"ab{matcher.replacement}"
    assert source == original


@pytest.mark.parametrize("layers", (1, 2, 5, 13))
def test_provider_nested_unicode_json_escape_replaces_complete_valid_span(
    tmp_path, monkeypatch, layers
):
    """A canonical suffix cannot corrupt its nested JSON escape container."""
    from hermes_cli import env_loader
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    secret = "é"
    spelling = _nest_json_string_spelling(json.dumps(secret)[1:-1], layers)
    content = f'{{"x":"{spelling}"}}'
    source = [{"role": "tool", "content": content}]
    original = copy.deepcopy(source)
    monkeypatch.setitem(
        env_loader._SECRET_SOURCE_VALUES_BY_HOME,
        str(tmp_path.resolve()),
        {"UNICODE_JSON_SECRET": secret},
    )
    token = set_hermes_home_override(tmp_path)
    try:
        provider_bound = sanitize_api_messages(source)
    finally:
        reset_hermes_home_override(token)

    provider_content = provider_bound[0]["content"]
    assert json.loads(provider_content) == {"x": "***"}
    assert secret not in provider_content
    assert spelling not in provider_content
    assert source == original


@pytest.mark.parametrize("layers", (1, 2, 5, 13))
def test_provider_simple_slash_json_escape_replaces_complete_valid_span(
    tmp_path, monkeypatch, layers
):
    r"""A one-byte secret encoded as ``\/`` cannot leave invalid JSON."""
    from hermes_cli import env_loader
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    secret = "/"
    spelling = _nest_json_string_spelling(r"\/", layers)
    content = f'{{"x":"{spelling}"}}'
    source = [{"role": "tool", "content": content}]
    original = copy.deepcopy(source)
    monkeypatch.setitem(
        env_loader._SECRET_SOURCE_VALUES_BY_HOME,
        str(tmp_path.resolve()),
        {"SLASH_JSON_SECRET": secret},
    )
    token = set_hermes_home_override(tmp_path)
    try:
        provider_bound = sanitize_api_messages(source)
    finally:
        reset_hermes_home_override(token)

    provider_content = provider_bound[0]["content"]
    assert json.loads(provider_content) == {"x": "***"}
    assert secret not in provider_content
    assert spelling not in provider_content
    assert source == original


def test_request_local_pattern_scope_rebinds_nested_same_home_profile(
    tmp_path, monkeypatch
):
    """Operation-local reuse never turns into a stale home-only profile cache."""
    from agent.redact import _exact_secret_pattern_scope, redact_known_secret_values
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    first = "first-profile-secret-77487"
    second = "second-profile-secret-77487"
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    first_token = set_secret_scope({"ACTIVE_PROFILE_TOKEN": first})
    try:
        with _exact_secret_pattern_scope(home=tmp_path):
            assert redact_known_secret_values(first, home=tmp_path) == "***"
            second_token = set_secret_scope({"ACTIVE_PROFILE_TOKEN": second})
            try:
                with _exact_secret_pattern_scope(home=tmp_path):
                    assert (
                        redact_known_secret_values(
                            f"{first} | {second}", home=tmp_path
                        )
                        == f"{first} | ***"
                    )
            finally:
                reset_secret_scope(second_token)
            assert redact_known_secret_values(first, home=tmp_path) == "***"
    finally:
        reset_secret_scope(first_token)


def test_provider_copy_reuses_operation_local_exact_pattern(tmp_path, monkeypatch):
    """Provider-copy redaction shares the request's immutable secret snapshot."""
    import agent.redact as redact
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    secret = "p4ss"
    source = [{"role": "tool", "content": secret}]
    original = copy.deepcopy(source)
    collections = 0

    def collect_exact_secret_values(home):
        nonlocal collections
        collections += 1
        return (secret,)

    monkeypatch.setattr(
        redact, "_collect_exact_secret_values", collect_exact_secret_values
    )
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    token = set_secret_scope({"ACTIVE_TOKEN": secret})
    try:
        with redact._exact_secret_pattern_scope(home=tmp_path):
            first = redact.redact_provider_message_values(source, home=tmp_path)
            second = redact.redact_provider_message_values(source, home=tmp_path)
    finally:
        reset_secret_scope(token)

    assert collections == 1
    assert first[0]["content"] == "***"
    assert second[0]["content"] == "***"
    assert source == original


def test_operation_pattern_scope_retains_scope_identity(tmp_path, monkeypatch):
    """The cache owns its scope object so numeric identity cannot be reused."""
    import agent.redact as redact
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    secret = "r8$!"
    scope = {"ACTIVE_TOKEN": secret}
    monkeypatch.setattr(
        redact, "_collect_exact_secret_values", lambda home: (secret,)
    )
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    token = set_secret_scope(scope)
    try:
        with redact._exact_secret_pattern_scope(home=tmp_path):
            cached = redact._EXACT_SECRET_PATTERN_SCOPE.get()

            assert cached is not None
            assert cached[0] == (str(tmp_path.resolve()), True)
            assert cached[1] is scope
            assert cached[2] is not None
    finally:
        reset_secret_scope(token)


@pytest.mark.parametrize("source_kind", ("process", "scope", "multiplex"))
def test_config_style_credential_suffixes_do_not_mask_provider_text(
    tmp_path, monkeypatch, source_kind
):
    """Flag-shaped ambient settings do not become transcript-wide secrets."""
    import agent.redact as redact
    from agent.secret_scope import reset_secret_scope, set_secret_scope
    from hermes_cli import env_loader

    settings = {
        "SKIP_AUTH": "false",
        "DEBUG_AUTH": "true",
        "REQUIRE_AUTH": "1",
        "USE_TOKEN": "true",
        "SHOW_PASSWORD": "false",
        "AUTH_USE_AUTH": "false",
    }
    credentials = {
        "PASSWORD": "p4ss",
        "REFRESH_TOKEN": "r8$!",
    }
    environment = {**settings, **credentials}
    content = " | ".join(f"{name}={value}" for name, value in environment.items())
    source = [{"role": "tool", "content": content}]
    original = copy.deepcopy(source)

    monkeypatch.setattr(env_loader, "get_secret_source_values", lambda home: {})
    monkeypatch.setattr(
        redact.os, "environ", environment if source_kind == "process" else {}
    )
    monkeypatch.setattr(
        "agent.secret_scope._MULTIPLEX_ACTIVE", source_kind == "multiplex"
    )
    token = set_secret_scope(environment if source_kind != "process" else None)
    try:
        provider_bound = redact.redact_provider_message_values(source, home=tmp_path)
    finally:
        reset_secret_scope(token)

    provider_content = provider_bound[0]["content"]
    assert all(
        f"{name}={value}" in provider_content for name, value in settings.items()
    )
    assert "PASSWORD=p4ss" not in provider_content
    assert "REFRESH_TOKEN=r8$!" not in provider_content
    assert provider_content.endswith("PASSWORD=*** | REFRESH_TOKEN=***")
    assert source == original


@pytest.mark.parametrize("source_kind", ("process", "scope", "multiplex"))
def test_config_style_names_with_arbitrary_short_values_remain_secrets(
    tmp_path, monkeypatch, source_kind
):
    """Name-shape exclusions do not create a blanket short-secret bypass."""
    import agent.redact as redact
    from agent.secret_scope import reset_secret_scope, set_secret_scope
    from hermes_cli import env_loader

    credentials = {
        "SKIP_AUTH": "p4ss",
        "USE_TOKEN": "r8$!",
    }
    source = [
        {
            "role": "tool",
            "content": "SKIP_AUTH=p4ss | USE_TOKEN=r8$!",
        }
    ]
    original = copy.deepcopy(source)

    monkeypatch.setattr(env_loader, "get_secret_source_values", lambda home: {})
    monkeypatch.setattr(
        redact.os, "environ", credentials if source_kind == "process" else {}
    )
    monkeypatch.setattr(
        "agent.secret_scope._MULTIPLEX_ACTIVE", source_kind == "multiplex"
    )
    token = set_secret_scope(credentials if source_kind != "process" else None)
    try:
        provider_bound = redact.redact_provider_message_values(source, home=tmp_path)
    finally:
        reset_secret_scope(token)

    assert provider_bound[0]["content"] == "SKIP_AUTH=*** | USE_TOKEN=***"
    assert source == original


def test_authoritative_scoped_credential_aliases_are_literal_matched(tmp_path):
    from agent.redact import redact_known_secret_values
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    secrets = {
        "MYSQL_PASS": "scope-pass-77487",
        "DB_PASSWD": "scope-passwd-77487",
        "DB_PW": "scope-pw-77487",
        "PGPASSWORD": "scope-pgpassword-77487",
        "MYSQL_PWD": "scope-mysql-pwd-77487",
        "PASSWORD": "scope-password-77487",
        "PASSWD": "scope-bare-passwd-77487",
        "PASS": "scope-bare-pass-77487",
        "PW": "scope-bare-pw-77487",
        "REDISCLI_AUTH": "scope-redis-auth-77487",
        "VAULT_CREDENTIAL": "scope-vault-credential-77487",
        "SERVICE_CREDENTIALS": "scope-service-credentials-77487",
        "PROXY_AUTH": "scope-proxy-auth-77487",
        # A terminal substring is not enough: source-name classification must
        # remain narrow so ordinary process configuration is not collected.
        "COMPASS": "scope-noncredential-77487",
        "AUTH_USE_AUTH": "true",
    }
    token = set_secret_scope(secrets)
    try:
        text = " | ".join(secrets.values())
        redacted = redact_known_secret_values(text, home=tmp_path)

        for name, secret in secrets.items():
            if name not in {"COMPASS", "AUTH_USE_AUTH"}:
                assert secret not in redacted
        assert redacted.endswith("scope-noncredential-77487 | true")
    finally:
        reset_secret_scope(token)


def test_ordinary_scoped_key_names_do_not_corrupt_provider_content(tmp_path):
    """Generic ``*_KEY`` configuration values are not exact-mask inputs."""
    from agent.redact import redact_provider_message_values
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    ordinary = {
        "PARTITION_KEY": "default",
        "SORT_KEY": "id",
        "CACHE_KEY": "cache-name",
        "IDEMPOTENCY_KEY": "request-id",
        "AUTH_USE_AUTH": "true",
    }
    credentials = {
        "SERVICE_API_KEY": "scoped-api-key-77487",
        "SERVICE_TOKEN": "scoped-token-77487",
        "SERVICE_SECRET": "scoped-secret-77487",
        "SERVICE_PASSWORD": "scoped-password-77487",
        "FAL_KEY": "scoped-fal-key-77487",
        "API_SERVER_KEY": "scoped-server-key-77487",
        "AZURE_ANTHROPIC_KEY": "scoped-azure-key-77487",
        "VOICE_TOOLS_OPENAI_KEY": "scoped-voice-key-77487",
        "PORCUPINE_ACCESS_KEY": "scoped-access-key-77487",
        "AWS_SECRET_ACCESS_KEY": "scoped-aws-key-77487",
    }
    source = [
        {
            "role": "tool",
            "content": " | ".join((*ordinary.values(), *credentials.values())),
        }
    ]
    token = set_secret_scope({**ordinary, **credentials})
    try:
        provider_bound = redact_provider_message_values(source, home=tmp_path)

        content = provider_bound[0]["content"]
        assert all(value in content for value in ordinary.values())
        assert all(value not in content for value in credentials.values())
        assert source[0]["content"] == " | ".join(
            (*ordinary.values(), *credentials.values())
        )
    finally:
        reset_secret_scope(token)


def test_auth_use_auth_arbitrary_scope_value_is_masked(tmp_path):
    """Honcho's exact flag name is exempt only for documented flag values."""
    from agent.redact import redact_provider_message_values
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    source = [{"role": "tool", "content": "AUTH_USE_AUTH=p4ss"}]
    original = copy.deepcopy(source)
    token = set_secret_scope({"AUTH_USE_AUTH": "p4ss"})
    try:
        provider_bound = redact_provider_message_values(source, home=tmp_path)
    finally:
        reset_secret_scope(token)

    assert provider_bound[0]["content"] == "AUTH_USE_AUTH=***"
    assert source == original


def test_multiplex_scoped_credential_suffixes_mask_immutable_provider_copy(
    tmp_path, monkeypatch
):
    """Profile-only repository aliases are removed at provider egress."""
    from agent.redact import redact_provider_message_values
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    credentials = {
        "VAULT_CREDENTIAL": "multiplex-vault-credential-77487",
        "SERVICE_CREDENTIALS": "multiplex-service-credentials-77487",
        "PROXY_AUTH": "multiplex-proxy-auth-77487",
    }
    source = [{"role": "tool", "content": " | ".join(credentials.values())}]
    original = copy.deepcopy(source)
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    token = set_secret_scope(credentials)
    try:
        provider_bound = redact_provider_message_values(source, home=tmp_path)
    finally:
        reset_secret_scope(token)

    assert provider_bound[0]["content"] == "*** | *** | ***"
    assert source == original


@pytest.mark.parametrize("source_kind", ("process", "scope", "multiplex"))
def test_repository_bearer_aliases_mask_immutable_provider_copy(
    tmp_path, monkeypatch, source_kind
):
    """Exact Bedrock and A2A aliases are credentials without broadening bearer."""
    import agent.redact as redact
    from agent.secret_scope import reset_secret_scope, set_secret_scope
    from hermes_cli import env_loader

    credentials = {
        "AWS_BEARER_TOKEN_BEDROCK": "bedrock-bearer-77487",
        "A2A_BEARER_TOKEN": "a2a-bearer-77487",
    }
    source = [{"role": "tool", "content": " | ".join(credentials.values())}]
    original = copy.deepcopy(source)

    monkeypatch.setattr(env_loader, "get_secret_source_values", lambda home: {})
    monkeypatch.setattr(
        redact.os, "environ", credentials if source_kind == "process" else {}
    )
    monkeypatch.setattr(
        "agent.secret_scope._MULTIPLEX_ACTIVE", source_kind == "multiplex"
    )
    token = set_secret_scope(credentials if source_kind != "process" else None)
    try:
        provider_bound = redact.redact_provider_message_values(source, home=tmp_path)
    finally:
        reset_secret_scope(token)

    assert provider_bound[0]["content"] == "*** | ***"
    assert source == original


def test_provider_copy_preserves_controls_and_binary_data_but_masks_visible_values(
    tmp_path, monkeypatch
):
    """Short-secret collisions cannot corrupt valid multimodal request fields."""
    from agent.redact import redact_provider_message_values
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    controls = {
        "DATA_TOKEN": "A",
        "TTL_TOKEN": "1h",
        "DETAIL_TOKEN": "high",
        "MEDIA_TOKEN": "image/png",
        "TYPE_TOKEN": "text",
        "ROLE_TOKEN": "user",
        "ID_TOKEN": "block-1",
        "NAME_TOKEN": "fixture-name",
        "TOOL_CALL_TOKEN": "tool-call-1",
        "CALL_TOKEN": "call-1",
        "CACHE_TOKEN": "ephemeral",
    }
    visible = " | ".join(controls.values())
    source = [
        {
            "role": "user",
            "id": "block-1",
            "name": "fixture-name",
            "tool_call_id": "tool-call-1",
            "tool_use_id": "tool-call-1",
            "call_id": "call-1",
            "content": [
                {
                    "type": "text",
                    "id": "block-1",
                    "name": "fixture-name",
                    "tool_call_id": "tool-call-1",
                    "tool_use_id": "tool-call-1",
                    "call_id": "call-1",
                    "text": visible,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.test/A?ttl=1h",
                        "detail": "high",
                    },
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "A",
                    },
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                },
            ],
        }
    ]
    original = copy.deepcopy(source)

    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    token = set_secret_scope(controls)
    try:
        provider_bound = redact_provider_message_values(source, home=tmp_path)
    finally:
        reset_secret_scope(token)

    message = provider_bound[0]
    text_block, image_url_block, image_block = message["content"]
    assert message["role"] == "user"
    assert message["id"] == "block-1"
    assert message["name"] == "fixture-name"
    assert message["tool_call_id"] == "tool-call-1"
    assert message["tool_use_id"] == "tool-call-1"
    assert message["call_id"] == "call-1"
    assert text_block["type"] == "text"
    assert text_block["id"] == "block-1"
    assert text_block["name"] == "fixture-name"
    assert text_block["tool_call_id"] == "tool-call-1"
    assert text_block["tool_use_id"] == "tool-call-1"
    assert text_block["call_id"] == "call-1"
    assert all(secret not in text_block["text"] for secret in controls.values())
    assert image_url_block["image_url"]["detail"] == "high"
    assert "A" not in image_url_block["image_url"]["url"]
    assert "1h" not in image_url_block["image_url"]["url"]
    assert image_block["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": "A",
    }
    assert image_block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert source == original


def test_provider_copy_preserves_valid_base64_data_uri_payload(tmp_path, monkeypatch):
    """Opaque data-URI bytes stay valid when a short secret collides."""
    from agent.redact import redact_provider_message_values
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    source = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                }
            ],
        }
    ]
    original = copy.deepcopy(source)

    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    token = set_secret_scope({"PASSWORD": "A", "MEDIA_TOKEN": "image/png"})
    try:
        provider_bound = redact_provider_message_values(source, home=tmp_path)
    finally:
        reset_secret_scope(token)

    assert provider_bound[0]["content"][0]["image_url"]["url"] == (
        "data:image/png;base64,AAAA"
    )
    assert source == original


def test_final_native_image_url_preserves_base64_data_uri(tmp_path, monkeypatch):
    """Codex-native input_image URLs keep opaque base64 payloads valid."""
    from agent.redact import redact_provider_api_kwargs
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    source = {
        "input": [
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}
        ]
    }
    original = copy.deepcopy(source)

    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    token = set_secret_scope({"PASSWORD": "A"})
    try:
        provider_bound = redact_provider_api_kwargs(source, home=tmp_path)
    finally:
        reset_secret_scope(token)

    assert provider_bound["input"][0]["image_url"] == (
        "data:image/png;base64,AAAA"
    )
    assert source == original


def test_final_provider_kwargs_gate_preserves_native_executable_replay(
    tmp_path, monkeypatch
):
    """Provider-native arguments stay exact while adjacent text is masked."""
    from agent.redact import redact_provider_api_kwargs
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    secret = "p4ss"
    source = {
        "model": "fixture-model",
        "system": [{"text": f"system sees {secret}"}],
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"visible {secret}"},
                    {
                        "type": "tool_use",
                        "id": "call-anthropic",
                        "name": "terminal",
                        "input": {"command": f"printf {secret}"},
                    },
                    {
                        "toolUse": {
                            "toolUseId": "call-bedrock",
                            "name": "terminal",
                            "input": {"command": f"printf {secret}"},
                        }
                    },
                ],
            }
        ],
    }
    original = copy.deepcopy(source)
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    token = set_secret_scope({"REPLAY_PASSWORD": secret})
    try:
        dispatched = redact_provider_api_kwargs(source, home=tmp_path)
    finally:
        reset_secret_scope(token)

    content = dispatched["messages"][0]["content"]
    assert dispatched["system"] == [{"text": "system sees ***"}]
    assert content[0]["text"] == "visible ***"
    assert content[1]["input"] == {"command": f"printf {secret}"}
    assert content[2]["toolUse"]["input"] == {"command": f"printf {secret}"}
    assert source == original


def test_final_dispatch_gate_masks_secret_composed_by_thinking_merge(
    tmp_path, monkeypatch
):
    """Post-sanitize role repair cannot create a provider-visible secret."""
    from agent.chat_completion_helpers import build_api_kwargs
    from agent.secret_scope import reset_secret_scope, set_secret_scope
    from run_agent import AIAgent

    secret = "known-prefix\n\nknown-suffix"
    source = [
        {"role": "user", "content": "known-prefix"},
        {"role": "assistant", "content": "", "reasoning": "hidden"},
        {"role": "user", "content": "known-suffix"},
    ]
    original = copy.deepcopy(source)

    class CapturingTransport:
        def build_kwargs(self, **kwargs):
            return kwargs

    class DispatchAgent:
        tools = []
        api_mode = "bedrock_converse"
        model = "fixture-bedrock"
        max_tokens = 100
        _bedrock_region = "us-east-1"
        _bedrock_guardrail_config = None

        def _get_transport(self):
            return CapturingTransport()

    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    token = set_secret_scope({"MERGED_API_KEY": secret})
    try:
        # The first production gate sees two harmless fragments. The real
        # thinking-only repair then removes the assistant and joins them with
        # the exact newlines that complete the configured secret.
        first_gate = sanitize_api_messages(source)
        merged = AIAgent._drop_thinking_only_and_merge_users(first_gate)
        assert merged[0]["content"] == secret

        dispatched = build_api_kwargs(DispatchAgent(), merged)
    finally:
        reset_secret_scope(token)

    assert secret not in dispatched["messages"][0]["content"]
    assert dispatched["messages"][0]["content"] == "***"
    assert source == original


def test_force_ascii_normalization_precedes_final_dispatch_gate(
    tmp_path, monkeypatch
):
    """ASCII fallback cannot compose a secret after provider redaction."""
    from agent.chat_completion_helpers import build_api_kwargs
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    class CapturingTransport:
        def build_kwargs(self, **kwargs):
            return kwargs

    class DispatchAgent:
        tools = []
        api_mode = "bedrock_converse"
        model = "fixture-bedrock"
        max_tokens = 100
        _bedrock_region = "us-east-1"
        _bedrock_guardrail_config = None
        _force_ascii_payload = True

        def _get_transport(self):
            return CapturingTransport()

    source = [{"role": "user", "content": "abécd"}]
    original = copy.deepcopy(source)
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    token = set_secret_scope({"ACTIVE_TOKEN": "abcd"})
    try:
        dispatched = build_api_kwargs(DispatchAgent(), source)
    finally:
        reset_secret_scope(token)

    assert dispatched["messages"][0]["content"] == "***"
    assert source == original


def test_codex_harmony_preflight_cannot_compose_final_provider_secret(
    tmp_path, monkeypatch
):
    """The real Codex preflight is followed by a provider-native payload gate."""
    from agent.redact import (
        redact_provider_api_kwargs,
        redact_provider_message_values,
    )
    from agent.secret_scope import reset_secret_scope, set_secret_scope
    from agent.transports.codex import ResponsesApiTransport

    secret = "<｜start｜>"
    source = {
        "model": "gpt-5-codex",
        "instructions": "guard <|start|>",
        "input": [{"role": "user", "content": "payload <|start|>"}],
        "store": False,
    }
    original = copy.deepcopy(source)
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    token = set_secret_scope({"HARMONY_TOKEN": secret})
    try:
        pre_gate = dict(source)
        pre_gate["input"] = redact_provider_message_values(
            source["input"],
            home=tmp_path,
        )
        preflight = ResponsesApiTransport().preflight_kwargs(
            pre_gate,
            allow_stream=False,
            sanitize_harmony_tokens=True,
        )
        assert secret in preflight["instructions"]
        assert secret in preflight["input"][0]["content"]
        dispatched = redact_provider_api_kwargs(preflight, home=tmp_path)
    finally:
        reset_secret_scope(token)

    assert dispatched["instructions"] == "guard ***"
    assert dispatched["input"][0]["content"] == "payload ***"
    assert source == original


def test_creds_and_bearer_suffixes_require_authoritative_source(
    tmp_path, monkeypatch
):
    """Undeclared ambient aliases stay visible; source-backed values do not."""
    from agent.redact import redact_known_secret_values
    from agent.secret_scope import reset_secret_scope, set_secret_scope
    from hermes_cli import env_loader

    aliases = {
        "LEGACY_CREDS": "source-backed-creds-77487",
        "EDGE_BEARER": "source-backed-bearer-77487",
    }
    text = " | ".join(aliases.values())
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    token = set_secret_scope(aliases)
    try:
        assert redact_known_secret_values(text, home=tmp_path) == text
        monkeypatch.setitem(
            env_loader._SECRET_SOURCE_VALUES_BY_HOME,
            str(tmp_path.resolve()),
            aliases,
        )
        assert redact_known_secret_values(text, home=tmp_path) == "*** | ***"
    finally:
        reset_secret_scope(token)


def test_repository_declared_plugin_credentials_are_value_sensitively_masked(
    tmp_path,
):
    """Password metadata covers exact plugin names without masking SA paths."""
    from agent.redact import redact_known_secret_values
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    chat_inline_service_account = json.dumps(
        {
            "type": "service_account",
            "private_key": "inline-google-private-key-77487",
        }
    )
    application_inline_service_account = json.dumps(
        {
            "type": "service_account",
            "private_key": "application-default-private-key-77487",
        }
    )
    credentials = {
        "BUZZ_PRIVATE_KEY": "buzz-private-key-77487",
        "A2A_PEER_TOKENS": "alice:a2a-peer-token-77487",
        "GOOGLE_CHAT_SERVICE_ACCOUNT_JSON": chat_inline_service_account,
        "GOOGLE_APPLICATION_CREDENTIALS": application_inline_service_account,
        "GOOGLE_PRIVATE_KEY_TOKEN": "inline-google-private-key-77487",
    }
    token = set_secret_scope(credentials)
    try:
        redacted = redact_known_secret_values(
            " | ".join(credentials.values()), home=tmp_path
        )
        assert redacted == "*** | *** | *** | *** | ***"
    finally:
        reset_secret_scope(token)


def test_a2a_peer_token_container_collects_each_parsed_token(tmp_path):
    """A peer echo exposes a component, not the full configured container."""
    from agent.redact import redact_known_secret_values
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    token = set_secret_scope(
        {
            "A2A_PEER_TOKENS": (
                "alice:a2a-alice-token-77487, malformed, "
                "bob: a2a-bob-token-77487, empty:"
            )
        }
    )
    try:
        assert redact_known_secret_values(
            "alice=a2a-alice-token-77487 bob=a2a-bob-token-77487 "
            "ordinary=malformed",
            home=tmp_path,
        ) == "alice=*** bob=*** ordinary=malformed"
    finally:
        reset_secret_scope(token)


def test_inline_google_credentials_collect_private_scalar_not_public_metadata(
    tmp_path,
):
    """Reformatted inline JSON still masks private key scalar echoes only."""
    from agent.redact import redact_known_secret_values
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    private_key = "google-private-key-scalar-77487"
    project_id = "public-project-id-77487"
    inline = json.dumps(
        {
            "project_id": project_id,
            "private_key": private_key,
            "type": "service_account",
        },
        indent=2,
        sort_keys=True,
    )
    token = set_secret_scope({"GOOGLE_APPLICATION_CREDENTIALS": inline})
    try:
        assert redact_known_secret_values(
            f"private={private_key} project={project_id}", home=tmp_path
        ) == f"private=*** project={project_id}"
    finally:
        reset_secret_scope(token)

    token = set_secret_scope(
        {"GOOGLE_APPLICATION_CREDENTIALS": "/run/secrets/google-credentials.json"}
    )
    try:
        path = "/run/secrets/google-credentials.json"
        assert redact_known_secret_values(path, home=tmp_path) == path
    finally:
        reset_secret_scope(token)

    chat_service_account_path = "/run/secrets/google-chat-service-account.json"
    application_credentials_path = "/run/secrets/application-default.json"
    token = set_secret_scope({
        "GOOGLE_CHAT_SERVICE_ACCOUNT_JSON": chat_service_account_path,
        "GOOGLE_APPLICATION_CREDENTIALS": application_credentials_path,
    })
    try:
        assert (
            redact_known_secret_values(
                f"{chat_service_account_path} | {application_credentials_path}",
                home=tmp_path,
            )
            == f"{chat_service_account_path} | {application_credentials_path}"
        )
    finally:
        reset_secret_scope(token)


def test_ordinary_process_key_names_do_not_corrupt_provider_content(
    tmp_path, monkeypatch
):
    """Single-profile process fallback applies the same narrow classifier."""
    from agent.redact import redact_provider_message_values
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    ordinary = {
        "PARTITION_KEY": "default",
        "SORT_KEY": "id",
        "CACHE_KEY": "cache-name",
        "IDEMPOTENCY_KEY": "request-id",
    }
    credentials = {
        "SERVICE_API_KEY": "process-api-key-77487",
        "FAL_KEY": "process-fal-key-77487",
    }
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", False)
    for name, value in {**ordinary, **credentials}.items():
        monkeypatch.setenv(name, value)
    token = set_secret_scope(None)
    try:
        source = [
            {
                "role": "tool",
                "content": " | ".join((*ordinary.values(), *credentials.values())),
            }
        ]
        provider_bound = redact_provider_message_values(source, home=tmp_path)

        content = provider_bound[0]["content"]
        assert all(value in content for value in ordinary.values())
        assert all(value not in content for value in credentials.values())
    finally:
        reset_secret_scope(token)


def test_single_profile_scope_overlays_process_credentials_by_name(
    tmp_path, monkeypatch
):
    """A nonempty scope keeps process-only credentials, with scope winning."""
    from agent.redact import redact_known_secret_values
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", False)
    monkeypatch.setenv("PROCESS_ONLY_TOKEN", "process-only-token-77487")
    monkeypatch.setenv("SHARED_TOKEN", "shadowed-process-token-77487")
    token = set_secret_scope(
        {
            "SCOPE_ONLY_API_KEY": "scope-only-key-77487",
            "SHARED_TOKEN": "scope-wins-token-77487",
        }
    )
    try:
        redacted = redact_known_secret_values(
            " | ".join(
                (
                    "process-only-token-77487",
                    "scope-only-key-77487",
                    "scope-wins-token-77487",
                    "shadowed-process-token-77487",
                )
            ),
            home=tmp_path,
        )
        assert redacted == "*** | *** | *** | shadowed-process-token-77487"
    finally:
        reset_secret_scope(token)


def test_multiplex_scope_does_not_collect_process_only_credentials(
    tmp_path, monkeypatch
):
    """Active multiplexing keeps process-global sibling credentials opaque."""
    from agent.redact import redact_known_secret_values
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    monkeypatch.setenv("PROCESS_ONLY_TOKEN", "other-profile-token-77487")
    monkeypatch.setenv(
        "PROCESS_ONLY_CREDENTIAL", "other-profile-credential-77487"
    )
    token = set_secret_scope(
        {
            "SCOPE_ONLY_TOKEN": "active-profile-token-77487",
            "SCOPE_ONLY_AUTH": "active-profile-auth-77487",
        }
    )
    try:
        assert redact_known_secret_values(
            "other-profile-token-77487 | other-profile-credential-77487 | "
            "active-profile-token-77487 | active-profile-auth-77487",
            home=tmp_path,
        ) == "other-profile-token-77487 | other-profile-credential-77487 | *** | ***"
    finally:
        reset_secret_scope(token)


@pytest.mark.parametrize(
    ("name", "secret"),
    [
        ("MYSQL_PASS", "process-pass-77487"),
        ("DB_PASSWD", "process-passwd-77487"),
        ("DB_PW", "process-pw-77487"),
        ("PGPASSWORD", "process-pgpassword-77487"),
        ("MYSQL_PWD", "process-mysql-pwd-77487"),
        ("PASSWORD", "process-password-77487"),
        ("PASSWD", "process-bare-passwd-77487"),
        ("PASS", "process-bare-pass-77487"),
        ("PW", "process-bare-pw-77487"),
        ("REDISCLI_AUTH", "process-redis-auth-77487"),
        ("VAULT_CREDENTIAL", "process-vault-credential-77487"),
        ("SERVICE_CREDENTIALS", "process-service-credentials-77487"),
        ("PROXY_AUTH", "process-proxy-auth-77487"),
    ],
)
def test_single_profile_process_credential_alias_is_literal_matched(
    tmp_path, monkeypatch, name, secret
):
    from agent.redact import redact_known_secret_values
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", False)
    monkeypatch.setenv(name, secret)
    token = set_secret_scope(None)
    try:
        assert redact_known_secret_values(
            f"bare value: {secret}", home=tmp_path
        ) == "bare value: ***"
    finally:
        reset_secret_scope(token)


def test_historical_argument_fields_remain_exact_for_replay(applied_secret_home):
    """#43083 applies to modern and legacy replay argument shapes."""
    from agent.redact import redact_provider_message_values

    _home, secret = applied_secret_home
    arguments = json.dumps({"command": f"echo {secret}"})
    messages = [
        {
            "role": "assistant",
            "content": f"provider-visible prose {secret}",
            "tool_calls": [
                {
                    "id": "call_77487",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": arguments},
                }
            ],
            "function_call": {"name": "terminal", "arguments": arguments},
        }
    ]
    original = copy.deepcopy(messages)

    provider_bound = redact_provider_message_values(messages)

    assert secret not in provider_bound[0]["content"]
    assert provider_bound[0]["tool_calls"][0]["function"]["arguments"] == arguments
    assert provider_bound[0]["function_call"]["arguments"] == arguments
    assert messages == original


def test_provider_exact_value_mask_respects_redaction_opt_out(
    applied_secret_home, monkeypatch
):
    _home, secret = applied_secret_home
    messages = _paired_messages(secret)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)

    provider_bound = sanitize_api_messages(messages)

    assert provider_bound[1]["content"][0]["text"] == (
        f"first={secret}; second={secret}"
    )
    assert provider_bound[1]["content"][1]["image_url"]["url"] == (
        f"https://example.invalid/{secret}/{secret}/image.png"
    )


def _response(content: str) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=[])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="fixture-model")


def _auxiliary_replay_messages(secret: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": f"visible auxiliary content {secret}",
            "tool_calls": [
                {
                    "id": "call-auxiliary-77487",
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "arguments": json.dumps({"command": f"printf {secret}"}),
                    },
                }
            ],
            "anthropic_content_blocks": [
                {
                    "type": "thinking",
                    "thinking": f"signed thought {secret}",
                    "signature": "signed-auxiliary-77487",
                },
                {"type": "text", "text": f"signed sequence text {secret}"},
            ],
            "reasoning_details": [
                {"type": "reasoning.text", "text": f"signed detail {secret}"}
            ],
            "codex_reasoning_items": [
                {"type": "reasoning", "encrypted_content": f"sealed-{secret}"}
            ],
        }
    ]


def test_sync_auxiliary_boundary_masks_disposable_provider_messages(
    applied_secret_home, monkeypatch
):
    """Direct auxiliary callers cannot bypass the provider-copy gate."""
    import agent.auxiliary_client as auxiliary_client

    _home, secret = applied_secret_home
    source = _auxiliary_replay_messages(secret)
    original = copy.deepcopy(source)
    captured = {}
    sentinel = _response("safe")

    def fake_impl(**kwargs):
        captured.update(copy.deepcopy(kwargs))
        return sentinel

    monkeypatch.setattr(auxiliary_client, "_call_llm_impl", fake_impl)

    assert auxiliary_client.call_llm(messages=source) is sentinel
    outbound = captured["messages"][0]
    assert secret not in outbound["content"]
    assert outbound["tool_calls"] == original[0]["tool_calls"]
    assert outbound["anthropic_content_blocks"] == original[0][
        "anthropic_content_blocks"
    ]
    assert outbound["reasoning_details"] == original[0]["reasoning_details"]
    assert outbound["codex_reasoning_items"] == original[0]["codex_reasoning_items"]
    assert source == original


def test_async_auxiliary_boundary_masks_disposable_provider_messages(
    applied_secret_home, monkeypatch
):
    """The async central boundary enforces the same immutable provider copy."""
    import agent.auxiliary_client as auxiliary_client

    _home, secret = applied_secret_home
    source = _auxiliary_replay_messages(secret)
    original = copy.deepcopy(source)
    captured = {}
    sentinel = _response("safe")

    async def fake_impl(**kwargs):
        captured.update(copy.deepcopy(kwargs))
        return sentinel

    monkeypatch.setattr(auxiliary_client, "_async_call_llm_impl", fake_impl)

    assert asyncio.run(auxiliary_client.async_call_llm(messages=source)) is sentinel
    assert secret not in captured["messages"][0]["content"]
    assert captured["messages"][0]["tool_calls"] == original[0]["tool_calls"]
    assert source == original


def test_async_auxiliary_redaction_offloads_with_context_scope(
    tmp_path, monkeypatch
):
    """Thread offload keeps the active multiplex profile's ContextVar secrets."""
    import agent.auxiliary_client as auxiliary_client
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    secret = "async-context-secret-77487"
    source = [{"role": "tool", "content": f"thread result {secret}"}]
    captured = {}
    offloads = []
    sentinel = _response("safe")
    real_to_thread = asyncio.to_thread

    async def spy_to_thread(func, /, *args, **kwargs):
        offloads.append(func)
        return await real_to_thread(func, *args, **kwargs)

    async def fake_impl(**kwargs):
        captured.update(copy.deepcopy(kwargs))
        return sentinel

    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    monkeypatch.setattr(asyncio, "to_thread", spy_to_thread)
    monkeypatch.setattr(auxiliary_client, "_async_call_llm_impl", fake_impl)
    token = set_secret_scope({"ACTIVE_PROFILE_TOKEN": secret})
    try:
        assert asyncio.run(auxiliary_client.async_call_llm(messages=source)) is sentinel
    finally:
        reset_secret_scope(token)

    assert offloads
    assert secret not in captured["messages"][0]["content"]
    assert source == [{"role": "tool", "content": f"thread result {secret}"}]


def test_auxiliary_boundary_respects_explicit_redaction_opt_out(
    applied_secret_home, monkeypatch
):
    import agent.auxiliary_client as auxiliary_client

    _home, secret = applied_secret_home
    source = _auxiliary_replay_messages(secret)
    captured = {}

    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
    monkeypatch.setattr(
        auxiliary_client,
        "_call_llm_impl",
        lambda **kwargs: captured.update(kwargs) or _response("safe"),
    )

    auxiliary_client.call_llm(messages=source)

    assert captured["messages"] is source
    assert secret in captured["messages"][0]["content"]


def test_title_output_is_masked_before_persistence(applied_secret_home, monkeypatch):
    from agent.title_generator import _persist_session_title, generate_title

    _home, secret = applied_secret_home
    monkeypatch.setattr(
        "agent.title_generator.call_llm", lambda **_kwargs: _response(secret)
    )
    stored = {}

    class SessionStore:
        def set_auto_title_if_empty(self, session_id, title):
            stored[session_id] = title
            return True

    title = generate_title("request", "response")
    persisted = _persist_session_title(
        SessionStore(), "session-77487", title, source="llm"
    )

    assert title == "***"
    assert persisted == "***"
    assert stored == {"session-77487": "***"}


def test_title_input_masks_before_maximum_length_cap(tmp_path, monkeypatch):
    import agent.auxiliary_client as auxiliary_client
    import agent.title_generator as title_generator
    from agent.title_generator import generate_title
    from hermes_cli import env_loader
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    secret = "cycle16-background-title-secret-" + "ABCDEFGHIJ" * 7
    home_key = str(tmp_path.resolve())
    monkeypatch.setitem(
        env_loader._SECRET_SOURCE_VALUES_BY_HOME,
        home_key,
        {"PROFILE_TITLE_API_TOKEN": secret},
    )
    source = "x" * (title_generator.MAX_TITLE_INPUT_CHARS - 30) + secret
    captured = {}

    def fake_impl(**kwargs):
        captured["messages"] = copy.deepcopy(kwargs["messages"])
        return _response('{"title":"Safe title"}')

    monkeypatch.setattr(title_generator, "_auto_title_enabled", lambda: True)
    monkeypatch.setattr(title_generator, "_title_language", lambda: "")
    monkeypatch.setattr(auxiliary_client, "_call_llm_impl", fake_impl)
    home_token = set_hermes_home_override(tmp_path)
    try:
        title = generate_title(source)
    finally:
        reset_hermes_home_override(home_token)

    provider_text = json.dumps(captured["messages"])
    assert title == "Safe title"
    assert secret not in provider_text
    assert secret[:30] not in provider_text
    assert source == "x" * (title_generator.MAX_TITLE_INPUT_CHARS - 30) + secret


def test_instant_title_masks_long_secret_before_truncation(tmp_path, monkeypatch):
    from agent.title_generator import apply_instant_title
    from hermes_cli import env_loader
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    secret = "cycle16-instant-title-secret-material-" + "ABCDEFGHIJ" * 4
    home_key = str(tmp_path.resolve())
    monkeypatch.setitem(
        env_loader._SECRET_SOURCE_VALUES_BY_HOME,
        home_key,
        {"PROFILE_TITLE_API_TOKEN": secret},
    )
    source = f"Investigate {secret} and details"
    stored = []
    callbacks = []

    class SessionStore:
        def set_auto_title_if_empty(self, session_id, title):
            stored.append((session_id, title))
            return True

    home_token = set_hermes_home_override(tmp_path)
    try:
        result = apply_instant_title(
            SessionStore(),
            "session-instant-title-77487",
            source,
            lambda title, title_source: callbacks.append((title, title_source)),
        )
    finally:
        reset_hermes_home_override(home_token)

    assert result == "Investigate *** and details"
    assert callbacks == [(result, "derived")]
    assert stored == [("session-instant-title-77487", result)]
    assert secret not in result
    assert secret[:48] not in result
    assert source == f"Investigate {secret} and details"


def test_maybe_auto_title_thread_preserves_profile_redaction_context(
    tmp_path, monkeypatch
):
    """The real background worker keeps profile-only input/output masking."""
    import threading

    import agent.auxiliary_client as auxiliary_client
    import agent.title_generator as title_generator
    from agent.secret_scope import reset_secret_scope, set_secret_scope
    from hermes_cli import env_loader
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    secret = "profile-title-thread-secret-77487"
    home_key = str(tmp_path.resolve())
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    monkeypatch.setitem(
        env_loader._SECRET_SOURCE_VALUES_BY_HOME,
        home_key,
        {"PROFILE_ONLY_TITLE_SECRET": secret},
    )
    captured = {}
    persisted = threading.Event()
    worker_persisted = threading.Event()
    source_user = f"request contains {secret}"
    persist_calls = []

    class SessionStore:
        def get_session_title(self, _session_id):
            return None

        def get_conversation_root(self, session_id):
            return session_id

        def set_auto_title_if_empty(self, session_id, title):
            captured["persisted"] = (session_id, title)
            persist_calls.append((session_id, title))
            persisted.set()
            if len(persist_calls) >= 2:
                worker_persisted.set()
            return True

    def fake_impl(**kwargs):
        captured["messages"] = copy.deepcopy(kwargs["messages"])
        return _response(secret)

    monkeypatch.setattr(title_generator, "_auto_title_enabled", lambda: True)
    monkeypatch.setattr(title_generator, "_title_language", lambda: "")
    monkeypatch.setattr(auxiliary_client, "_call_llm_impl", fake_impl)
    home_token = set_hermes_home_override(tmp_path)
    secret_token = set_secret_scope({"PROFILE_ONLY_TITLE_TOKEN": secret})
    try:
        title_generator.maybe_auto_title(
            SessionStore(),
            "session-title-context-77487",
            source_user,
            [{"role": "user", "content": source_user}],
        )
        assert persisted.wait(timeout=10), "auto-title worker did not persist"
        assert worker_persisted.wait(timeout=10), "auto-title upgrade did not persist"
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)

    assert secret not in json.dumps(captured["messages"])
    assert "***" in captured["messages"][1]["content"]
    assert captured["persisted"] == ("session-title-context-77487", "***")
    assert source_user == f"request contains {secret}"


def test_oneshot_output_masks_exact_secret(applied_secret_home, monkeypatch):
    from agent.oneshot import run_oneshot

    _home, secret = applied_secret_home
    monkeypatch.setattr("agent.oneshot.call_llm", lambda **_kwargs: _response(secret))

    assert run_oneshot(instructions="answer", user_input="request") == "***"


def test_plugin_sync_and_async_outputs_mask_exact_secret(
    applied_secret_home,
):
    from agent.plugin_llm import _TrustPolicy, make_plugin_llm_for_test

    _home, secret = applied_secret_home

    def sync_caller(**_kwargs):
        return "fixture", "model", _response(secret)

    async def async_caller(**_kwargs):
        return "fixture", "model", _response(secret)

    llm = make_plugin_llm_for_test(
        plugin_id="redaction-fixture",
        policy=_TrustPolicy(plugin_id="redaction-fixture"),
        sync_caller=sync_caller,
        async_caller=async_caller,
    )

    assert llm.complete([{"role": "user", "content": "request"}]).text == "***"
    assert (
        asyncio.run(llm.acomplete([{"role": "user", "content": "request"}])).text
        == "***"
    )


def test_plugin_structured_output_preserves_json_structure_while_masking_values(
    tmp_path, monkeypatch
):
    """Plugin JSON output remains parseable when secrets resemble JSON syntax."""
    from agent.plugin_llm import _TrustPolicy, make_plugin_llm_for_test
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    raw_text = (
        '{"ok":true,"missing":null,'
        '"secret":"plugin-secret","literal":"true","punctuation":"{"}'
    )
    response = _response(raw_text)

    def sync_caller(**_kwargs):
        return "fixture", "model", response

    llm = make_plugin_llm_for_test(
        plugin_id="structured-redaction-fixture",
        policy=_TrustPolicy(plugin_id="structured-redaction-fixture"),
        sync_caller=sync_caller,
    )
    monkeypatch.setattr("agent.secret_scope._MULTIPLEX_ACTIVE", True)
    token = set_secret_scope(
        {
            "PLUGIN_TOKEN": "plugin-secret",
            "TRUE_TOKEN": "true",
            "NULL_TOKEN": "null",
            "OPEN_TOKEN": "{",
        }
    )
    try:
        result = llm.complete_structured(
            instructions="return JSON",
            input=[{"type": "text", "text": "request"}],
            json_mode=True,
        )
    finally:
        reset_secret_scope(token)

    assert result.content_type == "json"
    assert result.parsed == {
        "ok": True,
        "missing": None,
        "secret": "***",
        "literal": "***",
        "punctuation": "***",
    }
    assert json.loads(result.text) == result.parsed
    assert response.choices[0].message.content == raw_text


def test_auxiliary_text_outputs_respect_redaction_opt_out(
    applied_secret_home, monkeypatch
):
    from agent.oneshot import run_oneshot

    _home, secret = applied_secret_home
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
    monkeypatch.setattr("agent.oneshot.call_llm", lambda **_kwargs: _response(secret))

    assert run_oneshot(instructions="answer", user_input="request") == secret


def test_moa_reference_and_synthesis_requests_mask_exact_values(
    applied_secret_home, monkeypatch
):
    """MoA copies are clean while replayable modern/legacy arguments stay exact."""
    from agent.moa_loop import aggregate_moa_context

    _home, secret = applied_secret_home
    arguments = json.dumps({"command": f"printf {secret}"})
    api_messages = [
        {"role": "user", "content": "inspect the latest result"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_moa_77487",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": arguments},
                }
            ],
            "function_call": {"name": "terminal", "arguments": arguments},
        },
        {
            "role": "tool",
            "tool_call_id": "call_moa_77487",
            "content": f"external tool returned {secret}",
        },
    ]
    original = copy.deepcopy(api_messages)
    calls: list[dict] = []

    def fake_call_llm(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        if kwargs["task"] == "moa_reference":
            return _response(f"advisor repeated {secret}")
        return _response("synthesized guidance")

    monkeypatch.setattr(
        "agent.moa_loop._slot_runtime",
        lambda slot: {
            "provider": slot.get("provider"),
            "model": slot.get("model"),
            "base_url": "https://example.invalid/v1",
            "api_key": "fixture-runtime-key",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)

    aggregate_moa_context(
        user_prompt=f"separate synthesis prompt contains {secret}",
        api_messages=api_messages,
        reference_models=[{"provider": "fixture", "model": "advisor"}],
        aggregator={"provider": "fixture", "model": "aggregator"},
    )

    reference_call = next(call for call in calls if call["task"] == "moa_reference")
    synthesis_call = next(call for call in calls if call["task"] == "moa_aggregator")
    assert secret not in json.dumps(reference_call["messages"])
    assert secret not in json.dumps(synthesis_call["messages"])
    assert api_messages == original
    assert api_messages[1]["tool_calls"][0]["function"]["arguments"] == arguments
    assert api_messages[1]["function_call"]["arguments"] == arguments


def test_moa_reference_masks_json_encoded_arguments_without_mutating_source(
    json_special_secret_home,
):
    """Rendered advisory arguments mask decoded values after JSON escaping."""
    from agent.moa_loop import _redacted_reference_messages

    _home, secret = json_special_secret_home
    arguments = json.dumps({"command": f"printf {secret}"})
    messages = [
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-moa-json-77487",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": arguments},
                }
            ],
            "function_call": {"name": "terminal", "arguments": arguments},
        },
    ]
    original = copy.deepcopy(messages)

    advisory = _redacted_reference_messages(messages)

    rendered = "\n".join(str(message.get("content", "")) for message in advisory)
    assert json.dumps(secret)[1:-1] not in rendered
    assert secret not in rendered
    assert messages == original
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == arguments
    assert messages[1]["function_call"]["arguments"] == arguments


@pytest.mark.parametrize("alternate_spelling", ALTERNATE_JSON_SPELLINGS)
def test_moa_reference_masks_alternate_json_escapes_without_mutating_source(
    monkeypatch, alternate_spelling
):
    """Disposable advisory rendering decodes equivalent JSON spellings."""
    from agent.moa_loop import _redacted_reference_messages
    from hermes_cli import env_loader
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    monkeypatch.setitem(
        env_loader._SECRET_SOURCE_VALUES_BY_HOME,
        str(home.resolve()),
        {"ALTERNATE_JSON_SECRET": ALTERNATE_JSON_SECRET},
    )
    arguments = _json_with_alternate_secret("command", alternate_spelling)
    messages = [
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-moa-alternate-json-77487",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": arguments},
                }
            ],
            "function_call": {"name": "terminal", "arguments": arguments},
        },
    ]
    original = copy.deepcopy(messages)

    advisory = _redacted_reference_messages(messages)

    rendered = "\n".join(str(message.get("content", "")) for message in advisory)
    assert ALTERNATE_JSON_SECRET not in rendered
    assert ALTERNATE_JSON_SPELLING not in rendered
    assert alternate_spelling not in rendered
    assert messages == original


@pytest.mark.parametrize("fragment_kind", MALFORMED_JSON_FRAGMENT_KINDS)
def test_moa_reference_masks_malformed_alternate_json_fragments_without_mutation(
    monkeypatch, fragment_kind
):
    """Disposable MoA rendering masks incomplete and invalid quoted fragments."""
    from agent.moa_loop import _redacted_reference_messages
    from hermes_cli import env_loader
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    monkeypatch.setitem(
        env_loader._SECRET_SOURCE_VALUES_BY_HOME,
        str(home.resolve()),
        {"MALFORMED_MOA_JSON_SECRET": ALTERNATE_JSON_SECRET},
    )
    arguments = _malformed_json_with_alternate_secret("command", fragment_kind)
    messages = [
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-moa-malformed-json-77487",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": arguments},
                }
            ],
        },
    ]
    original = copy.deepcopy(messages)

    advisory = _redacted_reference_messages(messages)

    rendered = "\n".join(str(message.get("content", "")) for message in advisory)
    assert "***" in rendered
    assert ALTERNATE_JSON_SECRET not in rendered
    assert ALTERNATE_JSON_SPELLING not in rendered
    assert messages == original


def test_anthropic_signed_ordered_replay_is_preserved_as_one_opaque_sequence(
    applied_secret_home,
):
    """Signed replay keeps every ordered block byte-exact and in place."""
    from agent.anthropic_adapter import convert_messages_to_anthropic

    _home, secret = applied_secret_home
    arguments = json.dumps({"command": f"printf {secret}"})
    ordered = [
        {
            "type": "thinking",
            "thinking": f"first signed reasoning stays byte exact {secret}",
            "signature": "sig-first-77487",
        },
        {"type": "text", "text": f"signature-bound provider text {secret}"},
        {
            "type": "thinking",
            "thinking": f"second signed reasoning stays byte exact {secret}",
            "signature": "sig-second-77487",
        },
        {
            "type": "tool_use",
            "id": "toolu_77487",
            "name": "terminal",
            "input": {"command": f"printf {secret}"},
        },
    ]
    messages = [
        {"role": "user", "content": "continue"},
        {
            "role": "assistant",
            "content": f"fallback content {secret}",
            "reasoning_details": [ordered[0]],
            "tool_calls": [
                {
                    "id": "toolu_77487",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": arguments},
                }
            ],
            "anthropic_content_blocks": ordered,
        },
        {
            "role": "tool",
            "tool_call_id": "toolu_77487",
            "content": "done",
        },
    ]
    original = copy.deepcopy(messages)

    provider_bound = sanitize_api_messages(messages)
    _system, anthropic_messages = convert_messages_to_anthropic(
        provider_bound,
        base_url=None,
        model="claude-opus-4-8",
    )

    assistant = next(m for m in anthropic_messages if m["role"] == "assistant")
    blocks = assistant["content"]
    assert [block["type"] for block in blocks] == [
        "thinking",
        "text",
        "thinking",
        "tool_use",
    ]
    assert blocks == ordered
    assert blocks[3]["input"] == {"command": f"printf {secret}"}
    assert messages == original


def test_anthropic_unsigned_ordered_text_is_still_redacted(applied_secret_home):
    """Text-only ordered lists have no signature contract and stay maskable."""
    from agent.redact import redact_provider_message_values

    _home, secret = applied_secret_home
    ordered = [
        {"type": "text", "text": f"first visible text {secret}"},
        {"type": "text", "text": f"second visible text {secret}"},
    ]
    messages = [
        {
            "role": "assistant",
            "content": f"fallback content {secret}",
            "anthropic_content_blocks": ordered,
        }
    ]
    original = copy.deepcopy(messages)

    provider_bound = redact_provider_message_values(messages)

    assert all(secret not in block["text"] for block in provider_bound[0]["anthropic_content_blocks"])
    assert messages == original


def test_anthropic_redacted_thinking_keeps_complete_ordered_sequence(
    applied_secret_home,
):
    """Opaque redacted-thinking data also binds its surrounding block list."""
    from agent.redact import redact_provider_message_values

    _home, secret = applied_secret_home
    ordered = [
        {"type": "text", "text": f"prefix text {secret}"},
        {"type": "redacted_thinking", "data": f"opaque-data-{secret}"},
        {"type": "text", "text": f"trailing text {secret}"},
    ]
    messages = [
        {
            "role": "assistant",
            "content": f"fallback content {secret}",
            "anthropic_content_blocks": ordered,
        }
    ]
    original = copy.deepcopy(messages)

    provider_bound = redact_provider_message_values(messages)

    assert secret not in provider_bound[0]["content"]
    assert provider_bound[0]["anthropic_content_blocks"] == ordered
    assert messages == original


def test_openrouter_reasoning_details_are_opaque_replay_bytes(applied_secret_home):
    """OpenRouter requires its complete reasoning-details sequence unchanged."""
    from agent.redact import redact_provider_message_values

    _home, secret = applied_secret_home
    details = [
        {
            "type": "reasoning.summary",
            "summary": f"provider summary {secret}",
            "id": "summary-77487",
        },
        {
            "type": "reasoning.text",
            "text": f"provider reasoning text {secret}",
            "format": "openrouter-v1",
        },
        {
            "type": "reasoning.signature",
            "signature": f"signed-{secret}",
            "data": f"opaque-{secret}",
        },
        {
            "type": "future.unknown",
            "opaque": {"nested": [secret, {"value": secret}]},
        },
    ]
    messages = [
        {
            "role": "assistant",
            "content": f"safe visible field {secret}",
            "reasoning_details": details,
        }
    ]
    original = copy.deepcopy(messages)

    provider_bound = redact_provider_message_values(messages)

    assert secret not in provider_bound[0]["content"]
    assert provider_bound[0]["reasoning_details"] == details
    assert messages == original


def test_provider_copy_preserves_opaque_codex_replay_fields(applied_secret_home):
    """Encrypted state and stable item identity remain exact; visible text does not."""
    from agent.redact import redact_provider_message_values

    _home, secret = applied_secret_home
    messages = [
        {
            "role": "assistant",
            "content": f"visible fallback {secret}",
            "codex_reasoning_items": [
                {
                    "type": "reasoning",
                    "id": f"rs_{secret}",
                    "encrypted_content": secret,
                }
            ],
            "codex_message_items": [
                {
                    "type": "message",
                    "role": "assistant",
                    "id": f"msg_{secret}",
                    "phase": "commentary",
                    "content": [
                        {"type": "output_text", "text": f"visible item {secret}"}
                    ],
                }
            ],
        }
    ]
    original = copy.deepcopy(messages)

    provider_bound = redact_provider_message_values(messages)

    assert provider_bound[0]["codex_reasoning_items"] == original[0][
        "codex_reasoning_items"
    ]
    item = provider_bound[0]["codex_message_items"][0]
    assert item["id"] == f"msg_{secret}"
    assert secret not in item["content"][0]["text"]
    assert messages == original
