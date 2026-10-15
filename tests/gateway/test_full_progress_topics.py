"""Full-mode progress and redaction behavior, split from topic-routing tests."""

import asyncio
import json
import threading
from tests.waiting import wait_until
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.session import SessionSource
from tests.gateway.test_run_progress_topics import (
    CodeBlockProgressAdapter, FakeAgent, ProgressCaptureAdapter, SmallLimitProgressAdapter,
    _ACTIVE_ADAPTER, _make_runner, _run_with_agent,
)

class SmallLimitCodeBlockProgressAdapter(SmallLimitProgressAdapter):
    """Tiny-limit adapter that also renders terminal commands as fences."""

    supports_code_blocks = True


class Utf16SmallLimitProgressAdapter(SmallLimitProgressAdapter):
    """Counts UTF-16 code units, as Telegram-compatible adapters do."""

    @property
    def message_len_fn(self):
        return lambda text: len(text.encode("utf-16-le")) // 2

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        if self.message_len_fn(content) > self.MAX_MESSAGE_LENGTH:
            self.oversized_sends.append(content)
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id=self._mint_id())


class FailFirstEditProgressAdapter(SmallLimitProgressAdapter):
    """Permanently rejects the first accumulated progress edit."""

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(platform=platform)
        self.failed_edit_ids = []

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        if not self.failed_edit_ids:
            self.failed_edit_ids.append(message_id)
            self.edits.append(
                {
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "content": content,
                }
            )
            return SendResult(success=False, error="message cannot be edited")
        return await super().edit_message(chat_id, message_id, content)


class TrackingSmallLimitProgressAdapter(SmallLimitProgressAdapter):
    """Records IDs so a continuation chunk cannot be edited later."""

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(platform=platform)
        self.sent_message_ids = []
        self.edited_message_ids = []

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        result = await super().send(
            chat_id, content, reply_to=reply_to, metadata=metadata
        )
        self.sent_message_ids.append(result.message_id)
        return result

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.edited_message_ids.append(message_id)
        return await super().edit_message(chat_id, message_id, content)


class FullTerminalArgsAgent:
    """Emits every terminal option so full mode cannot collapse to command-only."""

    ARGS = {
        "command": "python -c \"print('full terminal payload')\"",
        "workdir": "/tmp/full-mode-project",
        "timeout": 321,
        "background": True,
        "pty": False,
        "notify_on_complete": True,
        "watch_patterns": ["ready", "finished"],
    }
    RESULT_MARKER = "FULL-MODE-RESULT-MUST-NOT-RENDER"
    THINKING_MARKER = "FULL-MODE-THINKING-MUST-NOT-RENDER"

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        assert cb is not None
        cb("tool.started", "terminal", self.ARGS["command"], dict(self.ARGS))
        cb(
            "tool.completed",
            "terminal",
            None,
            None,
            result={"output": self.RESULT_MARKER},
        )
        cb("_thinking", self.THINKING_MARKER)
        wait_until(lambda: _ACTIVE_ADAPTER["adapter"].sent, timeout=3, what="tool progress send")
        return {"final_response": "done", "messages": [], "api_calls": 1}


class FullMediaArgsAgent:
    """Emits ordinary non-terminal media/path/question arguments."""

    ARGS = {
        "media": ["/tmp/frame-α.png", "/tmp/frame-β.jpg"],
        "path": "/tmp/screenshots",
        "question": "Compare every frame without dropping metadata",
    }

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        assert cb is not None
        cb(
            "tool.started", "vision_analyze", self.ARGS["question"], dict(self.ARGS)
        )
        wait_until(lambda: _ACTIVE_ADAPTER["adapter"].sent, timeout=3, what="tool progress send")
        return {"final_response": "done", "messages": [], "api_calls": 1}


class FullSecretArgsAgent:
    """Emits a credential while global redaction is disabled by the test."""

    SECRET = "opaque-credential-604"
    ARGS = {
        "token": SECRET,
        "path": "/tmp/non-secret-path",
        "nested": {"question": "keep this structure"},
    }

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        assert cb is not None
        cb(
            "tool.started", "custom_upload", "upload", dict(self.ARGS)
        )
        wait_until(lambda: _ACTIVE_ADAPTER["adapter"].sent, timeout=3, what="tool progress send")
        return {"final_response": "done", "messages": [], "api_calls": 1}


class OversizedSingleFullEntryAgent:
    """Emits one progress entry larger than the adapter's message limit."""

    PAYLOAD = "lossless-" + "0123456789abcdef" * 40 + "-end"
    ARGS = {"question": PAYLOAD}

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        assert cb is not None
        cb(
            "tool.started", "full_payload_tool", "oversized-preview", dict(self.ARGS)
        )
        wait_until(lambda: _ACTIVE_ADAPTER["adapter"].sent, timeout=3, what="tool progress send")
        return {"final_response": "done", "messages": [], "api_calls": 1}


class FullStrictRedactionAgent:
    """Exercises full-mode's chat-strict structured redaction boundary."""

    QUERY_SECRET = "opaque-query-value-917263"
    QUERY_VISIBLE_VALUE = "ordinary-query-value-381"
    URL_USER = "opaque-url-user-624"
    URL_PASSWORD = "opaque-url-password-735"
    URL_FRAGMENT = "opaque-url-fragment-846"
    AUTHORIZATION = "Custom opaque-header-authorization-159"
    COOKIE = "session=opaque-cookie-value-260"
    QUOTED_SECRET = 'opaque-before-quote"opaque-after-quote-371'
    TOKEN_PREFIX_VALUE = "opaque-token-prefix-key-value-593"
    REDACTED_KEY_ONE = "https://example.test/key?q=opaque-one"
    REDACTED_KEY_TWO = "https://example.test/key?q=opaque-two"
    ARGS = {
        "request_url": (
            f"prefix https://{URL_USER}:{URL_PASSWORD}@api.example.test/v1/items"
            f"?access_token={QUERY_SECRET}&visible={QUERY_VISIBLE_VALUE}"
            f"#{URL_FRAGMENT} suffix"
        ),
        "headers": {
            "Authorization": AUTHORIZATION,
            "Cookie": COOKIE,
        },
        "secret_value": QUOTED_SECRET,
        "tokenPrefixData": TOKEN_PREFIX_VALUE,
        "safe": {"path": "/tmp/visible", "items": ["alpha", ("beta",)]},
        REDACTED_KEY_ONE: "first-collision-value",
        REDACTED_KEY_TWO: "second-collision-value",
    }

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        self.tool_progress_callback(
            "tool.started", "strict_redaction_tool", "strict", self.ARGS
        )
        wait_until(lambda: _ACTIVE_ADAPTER["adapter"].sent, timeout=3, what="tool progress send")
        return {"final_response": "done", "messages": [], "api_calls": 1}


class FullCliCredentialAgent:
    """Emits opaque CLI and structured credentials for the enqueue boundary."""

    SECRETS = {
        "curl_short": "opaque-curl-short-481",
        "curl_long": "opaque-curl-long-592",
        "password": "opaque-password-quote-'-603",
        "token": "opaque-cli-token-714",
        "bearer": "opaque-body-bearer-825",
        "jwt": "opaque-body-jwt-936",
        "env_password": "opaque env password phrase 147",
        "recovery_key": "opaque recovery key material 258",
    }
    RECOVERY_ENV_NAME = "MATRIX_RECOVERY_" + "KEY"
    COMMAND = (
        f"export MY_DATABASE_PASSWORD='{SECRETS['env_password']}' "
        f"export {RECOVERY_ENV_NAME}='{SECRETS['recovery_key']}' "
        f"curl -u alice:{SECRETS['curl_short']} "
        f"--user=alice:{SECRETS['curl_long']} "
        f"--password \"{SECRETS['password']}\" "
        f"--token='{SECRETS['token']}' https://example.test/health"
    )
    ARGS = {
        "command": COMMAND,
        "body": {"bearer": SECRETS["bearer"], "jwt": SECRETS["jwt"]},
        "safe": {"path": "/tmp/visible", "token_count": 4},
    }

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        self.tool_progress_callback(
            "tool.started", "credential_cli", "strict", self.ARGS
        )
        wait_until(lambda: _ACTIVE_ADAPTER["adapter"].sent, timeout=3, what="tool progress send")
        return {"final_response": "done", "messages": [], "api_calls": 1}


class OversizedVerboseTerminalAgent:
    """Emits a verbose fenced entry that exceeds a tiny adapter limit."""

    COMMAND = "printf 'verbose fence'\n" + "x" * 500

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        self.tool_progress_callback(
            "tool.started", "terminal", self.COMMAND, {"command": self.COMMAND}
        )
        wait_until(lambda: _ACTIVE_ADAPTER["adapter"].sent, timeout=3, what="tool progress send")
        return {"final_response": "done", "messages": [], "api_calls": 1}


class EditFailureThenOversizedFullAgent:
    """Disables editing before emitting an oversized marked full entry."""

    BIG_ARGS = {"question": "after-edit-failure-" + "abcdef0123456789" * 40}

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        cb("tool.started", "first_full_tool", "first", {"value": "first"})
        wait_until(lambda: _ACTIVE_ADAPTER["adapter"].sent, timeout=3, what="first tool send")
        cb("tool.started", "second_full_tool", "second", {"value": "second"})
        wait_until(lambda: _ACTIVE_ADAPTER["adapter"].failed_edit_ids,
                   timeout=3, what="first edit refusal")
        cb("tool.started", "full_payload_tool", "big", self.BIG_ARGS)
        wait_until(lambda: self.BIG_ARGS["question"][-12:] in
                   "".join(row["content"] for row in _ACTIVE_ADAPTER["adapter"].sent),
                   timeout=3, what="oversized final chunk")
        return {"final_response": "done", "messages": [], "api_calls": 1}


class OversizedThenSmallFullAgent:
    """Emits a continuation-split full entry followed by another tool."""

    BIG_ARGS = {"question": "split-state-" + "0123456789abcdef" * 40}
    SMALL_ARGS = {"question": "fresh bubble after split"}

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        cb = self.tool_progress_callback
        cb("tool.started", "full_payload_tool", "big", self.BIG_ARGS)
        wait_until(lambda: self.BIG_ARGS["question"][-12:] in
                   "".join(row["content"] for row in _ACTIVE_ADAPTER["adapter"].sent),
                   timeout=3, what="first split complete")
        cb("tool.started", "post_split_tool", "small", self.SMALL_ARGS)
        wait_until(lambda: any("post_split_tool" in row["content"]
                               for row in _ACTIVE_ADAPTER["adapter"].sent),
                   timeout=3, what="later progress send")
        return {"final_response": "done", "messages": [], "api_calls": 1}


class NativeCardsProgressAdapter(ProgressCaptureAdapter):
    def native_task_cards_enabled(self):
        return True

    async def send_native_task_card_progress(self, **kwargs):
        raise AssertionError("full arguments must not become compact native cards")


def _first_progress_json(adapter):
    """Return the JSON dict from a one-entry full-mode progress message."""
    assert adapter.sent, "expected a full-mode progress bubble"
    content = adapter.sent[0]["content"]
    _, json_text = content.split("\n", 1)
    return content, json.loads(json_text)


def _expected_full_message(tool_name, args):
    from agent.display import get_tool_emoji
    from gateway.full_progress_redaction import _redact_full_progress_args

    safe_args = _redact_full_progress_args(args)
    return (
        f"{get_tool_emoji(tool_name, default='⚙️')} {tool_name}\n"
        f"{json.dumps(safe_args, ensure_ascii=False, default=str)}"
    )


def test_full_mode_structured_redactor_is_recursive_and_non_mutating():
    from gateway.full_progress_redaction import _redact_full_progress_args

    original_tuple = ("visible", {"password": "opaque-password-482"})
    original_list = [original_tuple]
    original = {
        "safe": original_list,
        "apiKey": "opaque-api-key-604",
        "x-api-key": "opaque-x-api-key-609",
        "private_key": {"nested": "opaque-private-key-715"},
        "key_material": ["opaque-key-material-826"],
    }

    redacted = _redact_full_progress_args(original)

    assert redacted is not original
    assert redacted["safe"] is not original_list
    assert redacted["safe"][0] is not original_tuple
    assert isinstance(redacted["safe"][0], tuple)
    assert original["safe"] == [
        ("visible", {"password": "opaque-password-482"})
    ]
    assert redacted["safe"][0][1]["password"] == "***"
    assert redacted["apiKey"] == "***"
    assert redacted["x-api-key"] == "***"
    assert redacted["private_key"] == "***"
    assert redacted["key_material"] == "***"


@pytest.mark.parametrize(
    "value, expected",
    [
        (
            "前https://user:password@example.test/x?plain=opaque#fragment",
            "前https://***@example.test/x?plain=***#***",
        ),
        (
            "prefix（https://name:pass@example.test/path?q=hidden#anchor",
            "prefix（https://***@example.test/path?q=***#***",
        ),
    ],
)
def test_full_mode_redacts_urls_adjacent_to_cjk_and_punctuation(value, expected):
    from gateway.full_progress_redaction import _redact_full_progress_string

    rendered = _redact_full_progress_string(value)

    assert rendered == expected
    secrets = (
        "user",
        "password",
        "opaque",
        "fragment",
        "name",
        "pass",
        "hidden",
        "anchor",
    )
    for secret in secrets:
        if secret in value:
            assert secret not in rendered


def test_full_mode_masks_lowercase_compound_secret_keys_without_renaming_keys():
    from gateway.full_progress_redaction import _redact_full_progress_args

    original = {
        "dbpassword": "raw-db-password-183",
        "oauth_token": "raw-oauth-token-294",
        "authtoken": "raw-auth-token-305",
        "clientcredentialsblob": "raw-client-credentials-416",
        "safevalue": "keep-visible",
    }

    redacted = _redact_full_progress_args(original)

    assert set(redacted) == set(original)
    assert redacted["safevalue"] == "keep-visible"
    for key in original.keys() - {"safevalue"}:
        assert redacted[key] == "***"
        assert original[key] not in json.dumps(redacted)


@pytest.mark.parametrize(
    "value, expected",
    [
        (
            "前custom+v1.2://url-user:url-pass@example.test/items"
            "?visible=opaque-visible&empty=#opaque-fragment 后",
            "前custom+v1.2://***@example.test/items"
            "?visible=***&empty=***#*** 后",
        ),
        (
            "git+ssh://git@example.test/repo?ref=opaque-ref&flag#opaque-anchor",
            "git+ssh://***@example.test/repo?ref=***&flag#***",
        ),
        (
            "s3://opaque-bucket-user@bucket.example.test/object?version=opaque-version",
            "s3://***@bucket.example.test/object?version=***",
        ),
        (
            "authorization://header-user:header-pass@example.test/session"
            "?scope=opaque-scope#opaque-fragment",
            "authorization://***@example.test/session?scope=***#***",
        ),
    ],
)
def test_full_mode_redacts_every_rfc_scheme_url(value, expected):
    from gateway.full_progress_redaction import _redact_full_progress_string

    assert _redact_full_progress_string(value) == expected


def test_full_mode_masks_dsn_and_connection_url_compound_keys_fail_closed():
    from gateway.full_progress_redaction import _redact_full_progress_args

    original = {
        "dsn": "opaque-dsn-100",
        "database_dsn": "opaque-database-dsn-200",
        "primaryDatabaseUrl": "opaque-primary-database-url-300",
        "readonly_connection_string": "opaque-readonly-connection-string-400",
        "db_uri": {"nested": "opaque-db-uri-500"},
        "monkey": "keep-monkey-visible",
        "monkey_business": "keep-business-visible",
        "database_name": "inventory",
        "connection_timeout": 30,
        "url": "https://example.test/health",
    }

    redacted = _redact_full_progress_args(original)

    for key in (
        "dsn",
        "database_dsn",
        "primaryDatabaseUrl",
        "readonly_connection_string",
        "db_uri",
    ):
        assert redacted[key] == "***"
        assert str(original[key]) not in json.dumps(redacted)
    assert redacted["monkey"] == "keep-monkey-visible"
    assert redacted["monkey_business"] == "keep-business-visible"
    assert redacted["database_name"] == "inventory"
    assert redacted["connection_timeout"] == 30
    assert redacted["url"] == "https://example.test/health"


def test_full_mode_masks_sensitive_string_and_list_headers():
    from gateway.full_progress_redaction import _redact_full_progress_args

    header_values = [
        "Authorization: Custom opaque-authorization-510",
        "Proxy-Authorization: Basic opaque-proxy-authorization-620",
        "Cookie: session=opaque-cookie-730; theme=dark",
        "Set-Cookie: session=opaque-set-cookie-840; HttpOnly; Secure",
        "X-API-Key: opaque-x-api-key-950",
        "API-Key: opaque-api-key-061",
        "X-Auth-Token: opaque-auth-token-172",
        "CSRF-Token: opaque-csrf-token-283",
    ]
    original = {
        "headers": header_values,
        "header_block": (
            "Authorization: Bearer opaque-block-authorization-394\n"
            "Set-Cookie: session=opaque-block-cookie-405; SameSite=Lax"
        ),
        "safe_headers": [
            "Accept: application/json",
            "Content-Type: application/json",
            "X-Request-ID: public-trace-123",
        ],
    }

    redacted = _redact_full_progress_args(original)

    assert redacted["headers"] == [
        f"{header.partition(':')[0]}: ***" for header in header_values
    ]
    assert redacted["header_block"] == (
        "Authorization: ***\nSet-Cookie: ***"
    )
    assert redacted["safe_headers"] == original["safe_headers"]
    assert all(secret not in json.dumps(redacted) for secret in (
        "opaque-authorization-510",
        "opaque-proxy-authorization-620",
        "opaque-cookie-730",
        "opaque-set-cookie-840",
        "opaque-x-api-key-950",
        "opaque-api-key-061",
        "opaque-auth-token-172",
        "opaque-csrf-token-283",
        "opaque-block-authorization-394",
        "opaque-block-cookie-405",
    ))


def test_full_mode_masks_sensitive_headers_inside_curl_strings_losslessly():
    from gateway.full_progress_redaction import _redact_full_progress_string

    command = (
        "curl -H 'Authorization: Bearer opaque-curl-auth-516' "
        "-H \"Proxy-Authorization: Basic opaque-curl-proxy-627\" "
        "-H 'Cookie: sid=opaque-curl-cookie-738; theme=dark' "
        "-H 'X-API-Key: opaque-curl-api-key-849' "
        "-H 'X-CSRF-Token: opaque-curl-csrf-950' "
        "-H 'Accept: application/json' https://example.test/health"
    )

    redacted = _redact_full_progress_string(command)

    assert redacted == (
        "curl -H 'Authorization: ***' "
        "-H \"Proxy-Authorization: ***\" "
        "-H 'Cookie: ***' "
        "-H 'X-API-Key: ***' "
        "-H 'X-CSRF-Token: ***' "
        "-H 'Accept: application/json' https://example.test/health"
    )


def test_full_mode_masks_cli_credentials_quote_safely():
    from gateway.full_progress_redaction import _redact_full_progress_string

    command = (
        "curl -u alice:opaque-short-password-101 "
        "--user bob:opaque-long-password-202 "
        "--user=carol:opaque-equals-password-303 "
        "--password 'opaque spaced password 404' "
        "--password=opaque-equals-password-505 "
        '"--token" "opaque quoted token 606" '
        "'--api-key=opaque api key 707' "
        "--jwt opaque-jwt-value-808 --bearer=opaque-bearer-value-909 "
        "https://example.test/health"
    )

    redacted = _redact_full_progress_string(command)

    assert redacted == (
        "curl -u *** "
        "--user *** "
        "--user=*** "
        "--password '***' "
        "--password=*** "
        '"--token" "***" '
        "'--api-key=***' "
        "--jwt *** --bearer=*** "
        "https://example.test/health"
    )
    for secret in (
        "opaque-short-password-101",
        "opaque-long-password-202",
        "opaque-equals-password-303",
        "opaque spaced password 404",
        "opaque-equals-password-505",
        "opaque quoted token 606",
        "opaque api key 707",
        "opaque-jwt-value-808",
        "opaque-bearer-value-909",
    ):
        assert secret not in redacted


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (
            "MY_DATABASE_PASSWORD='opaque database phrase 101' command",
            "MY_DATABASE_PASSWORD='***' command",
        ),
        (
            'export API_TOKEN="opaque exported token phrase 202"; command',
            'export API_TOKEN="***"; command',
        ),
        (
            "env CLIENT_SECRET='opaque client phrase 303' command",
            "env CLIENT_SECRET='***' command",
        ),
        (
            "'MY_DATABASE_PASSWORD=opaque whole word phrase 404' command",
            "'MY_DATABASE_PASSWORD=***' command",
        ),
        (
            "MY_DATABASE_PASSWORD=opaque-single-value command",
            "MY_DATABASE_PASSWORD=*** command",
        ),
        (
            "MY_DATABASE_LABEL='visible database label' command",
            "MY_DATABASE_LABEL='visible database label' command",
        ),
    ],
)
def test_full_mode_masks_quoted_secret_environment_assignments(command, expected):
    from gateway.full_progress_redaction import _redact_full_progress_string

    assert _redact_full_progress_string(command) == expected


def test_full_mode_masks_key_suffix_environment_assignment():
    from gateway.full_progress_redaction import _redact_full_progress_string

    env_name = "MATRIX_RECOVERY_" + "KEY"
    secret = "opaque recovery material 515"
    command = env_name + "='" + secret + "' command"
    expected = env_name + "='***' command"

    rendered = _redact_full_progress_string(command)

    assert rendered == expected
    assert secret not in rendered


def test_full_mode_keeps_benign_cli_flag_lookalikes():
    from gateway.full_progress_redaction import _redact_full_progress_string

    command = (
        "curl --user alice --username alice:visible --user-agent alice:visible "
        "--password-policy strict --password-file ./fixture.txt "
        "--token-count 4 --tokenizer local --api-key-file ./public.json "
        "--jwt-decoder local --bearer-format compact; "
        "tool --password ; echo still-visible"
    )

    assert _redact_full_progress_string(command) == command


def test_full_mode_masks_canonical_body_secret_keys_plus_bearer():
    _SENSITIVE_BODY_KEYS = {'private_key', 'api_key', 'key', 'apikey', 'token', 'access_token', 'password', 'authorization', 'secret', 'client_secret', 'id_token', 'jwt', 'refresh_token', 'auth'}
    from gateway.full_progress_redaction import _redact_full_progress_args

    original: dict[str, object] = {
        key: f"opaque-body-value-{index}"
        for index, key in enumerate(sorted(_SENSITIVE_BODY_KEYS))
    }
    original["bearer"] = "opaque-body-bearer-extra"
    original["safe_payload"] = {"path": "/tmp/visible", "count": 4}

    redacted = _redact_full_progress_args(original)

    for key in _SENSITIVE_BODY_KEYS | {"bearer"}:
        assert redacted[key] == "***"
        assert str(original[key]) not in json.dumps(redacted)
    assert redacted["safe_payload"] == original["safe_payload"]


def test_full_mode_masks_query_only_oauth_fragments():
    from gateway.full_progress_redaction import _redact_full_progress_string

    value = (
        "exchange ?client_id=public-client&code=opaque-oauth-code-111"
        "&state=opaque-oauth-state-222#opaque-oauth-fragment then continue"
    )

    assert _redact_full_progress_string(value) == (
        "exchange ?client_id=***&code=***&state=***#*** then continue"
    )


def test_full_mode_masks_relative_and_network_url_credentials():
    from gateway.full_progress_redaction import _redact_full_progress_string

    assert _redact_full_progress_string(
        "POST /callback?client_id=public-client&code=opaque-oauth-code"
        "&state=opaque-state#opaque-fragment"
    ) == "POST /callback?client_id=***&code=***&state=***#***"
    assert _redact_full_progress_string(
        "fetch //alice:opaque-password@example.test/path?token=opaque-token"
        "&visible=public-value"
    ) == "fetch //alice:***@example.test/path?token=***&visible=***"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("fetch //alice:opaque@host/path", "fetch //alice:***@host/path"),
        ("redirect /callback#opaque", "redirect /callback#***"),
        ("open callback?code=opaque", "open callback?code=***"),
        ("open callback#opaque", "open callback#***"),
    ],
)
def test_full_mode_masks_queryless_and_bare_relative_url_credentials(
    value, expected
):
    from gateway.full_progress_redaction import _redact_full_progress_string

    assert _redact_full_progress_string(value) == expected


def test_full_mode_keeps_noncredential_prose_and_invalid_query_syntax():
    from gateway.full_progress_redaction import _redact_full_progress_args

    original = {
        "monkey": "The monkey checks which code path handles state transitions.",
        "question": "Ready? code = example and state = documented.",
        "header_docs": "Authorization guide: public documentation",
        "safe_headers": ["CookieJar: local.txt", "X-API-Version: 2026-07-16"],
        "invalid_scheme": "1custom://user@example.test/path?visible=value",
    }

    assert _redact_full_progress_args(original) == original


@pytest.mark.asyncio
async def test_full_mode_terminal_shows_all_args_without_code_block(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        FullTerminalArgsAgent,
        session_id="sess-full-terminal-all-fields",
        config_data={
            "display": {
                "tool_progress": "off",
                "tool_preview_length": 8,
                "platforms": {"telegram": {"tool_progress": "full"}},
            }
        },
        adapter_cls=CodeBlockProgressAdapter,
    )

    assert result["final_response"] == "done"
    content, rendered_args = _first_progress_json(adapter)
    assert rendered_args == FullTerminalArgsAgent.ARGS
    assert "```" not in content
    all_progress = "\n".join(
        call["content"] for call in adapter.sent + adapter.edits
    )
    assert FullTerminalArgsAgent.RESULT_MARKER not in all_progress
    assert FullTerminalArgsAgent.THINKING_MARKER not in all_progress


@pytest.mark.asyncio
async def test_full_mode_ignores_tool_preview_length(monkeypatch, tmp_path):
    adapter, _ = await _run_with_agent(
        monkeypatch,
        tmp_path,
        FullTerminalArgsAgent,
        session_id="sess-full-ignores-preview-length",
        config_data={
            "display": {"tool_progress": "full", "tool_preview_length": 1}
        },
        adapter_cls=CodeBlockProgressAdapter,
    )

    _, rendered_args = _first_progress_json(adapter)
    assert rendered_args["command"] == FullTerminalArgsAgent.ARGS["command"]
    assert rendered_args["watch_patterns"] == ["ready", "finished"]


@pytest.mark.asyncio
async def test_full_mode_nonterminal_preserves_media_path_and_question(monkeypatch, tmp_path):
    adapter, _ = await _run_with_agent(
        monkeypatch,
        tmp_path,
        FullMediaArgsAgent,
        session_id="sess-full-media-args",
        config_data={"display": {"tool_progress": "full", "tool_preview_length": 2}},
    )

    _, rendered_args = _first_progress_json(adapter)
    assert rendered_args == FullMediaArgsAgent.ARGS


@pytest.mark.asyncio
async def test_full_mode_force_redacts_secret_and_preserves_structure(monkeypatch, tmp_path):
    from agent import redact

    # Full-mode chat is a hard safety boundary: force=True must win even when
    # the operator disabled ordinary global/log redaction.
    monkeypatch.setattr(redact, "_redact_enabled", lambda: False)
    assert redact._redact_enabled() is False
    adapter, _ = await _run_with_agent(
        monkeypatch,
        tmp_path,
        FullSecretArgsAgent,
        session_id="sess-full-force-redaction",
        config_data={"display": {"tool_progress": "full"}},
    )

    content, rendered_args = _first_progress_json(adapter)
    assert FullSecretArgsAgent.SECRET not in content
    assert rendered_args["token"] != FullSecretArgsAgent.SECRET
    assert rendered_args["path"] == FullSecretArgsAgent.ARGS["path"]
    assert rendered_args["nested"] == FullSecretArgsAgent.ARGS["nested"]
    assert set(rendered_args) == set(FullSecretArgsAgent.ARGS)


@pytest.mark.parametrize("alias", ["pass", "pw", "pwd", "passcode"])
@pytest.mark.parametrize("form", ["structured", "separate", "equals"])
def test_full_mode_exact_password_aliases_are_redacted_before_enqueue(
    monkeypatch, alias, form
):
    from copy import deepcopy
    from queue import Queue

    from agent import redact
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext

    monkeypatch.setattr(redact, "_redact_enabled", lambda: False)
    monkeypatch.setattr(redact, "_VAULT_REDACTION_VALUES", {})
    opaque = "mauve glacier 916"
    assert redact.redact_sensitive_text(opaque, force=True) == opaque
    prefix = "PWD='/tmp/visible workdir' tool "
    suffix = "--compass north --bypass none --pwd-hint public; pwd"
    args = {
        "body": [{alias: opaque}] if form == "structured" else [],
        "command": prefix + suffix,
        "safe": {"compass": "north", "bypass": "none", "pw_hint": "public",
                 "pwd_hint": "directory", "passcode_hint": "public"},
    }
    expected = deepcopy(args)
    if form == "structured":
        expected["body"] = [{alias: "***"}]
    else:
        separator = " " if form == "separate" else "="
        args["command"] = prefix + f"--{alias}{separator}'{opaque}' " + suffix
        expected["command"] = prefix + f"--{alias}{separator}'***' " + suffix
    original = deepcopy(args)
    ctx = TurnContext(
        source=SimpleNamespace(chat_id="password-alias-test"),
        _run_still_current=lambda: True,
        progress_mode="full", tool_progress_enabled=True, progress_queue=Queue(),
    )
    turn = TurnRunner(SimpleNamespace(_delivery_adapter_for=lambda _source: None), ctx)
    turn.progress_callback("tool.started", "credential_test_tool", "short", args)

    marker, content = ctx.progress_queue.get_nowait()
    assert marker == "__full__"
    assert ctx.progress_queue.empty()
    assert args == original
    assert opaque not in content
    assert json.loads(content.split("\n", 1)[1]) == expected


@pytest.mark.asyncio
async def test_full_mode_cli_and_body_credentials_are_redacted_before_enqueue(
    monkeypatch, tmp_path
):
    from agent import redact

    monkeypatch.setattr(redact, "_redact_enabled", lambda: False)
    assert redact._redact_enabled() is False
    adapter, _ = await _run_with_agent(
        monkeypatch,
        tmp_path,
        FullCliCredentialAgent,
        session_id="sess-full-cli-credential-redaction",
        config_data={"display": {"tool_progress": "full"}},
    )

    content, rendered_args = _first_progress_json(adapter)
    assert all(
        secret not in content for secret in FullCliCredentialAgent.SECRETS.values()
    )
    assert rendered_args["command"] == (
        "export MY_DATABASE_PASSWORD='***' "
        "export MATRIX_RECOVERY_KEY='***' "
        "curl -u *** --user=*** --password \"***\" "
        "--token='***' https://example.test/health"
    )
    assert rendered_args["body"] == {"bearer": "***", "jwt": "***"}
    assert rendered_args["safe"]["path"] == "/tmp/visible"


@pytest.mark.asyncio
async def test_full_mode_strict_redaction_happens_before_json_serialization(
    monkeypatch, tmp_path
):
    from agent import redact

    monkeypatch.setattr(redact, "_redact_enabled", lambda: False)
    assert redact._redact_enabled() is False
    adapter, _ = await _run_with_agent(
        monkeypatch,
        tmp_path,
        FullStrictRedactionAgent,
        session_id="sess-full-strict-structured-redaction",
        config_data={"display": {"tool_progress": "full"}},
    )

    content, rendered_args = _first_progress_json(adapter)
    raw_secrets = {
        FullStrictRedactionAgent.QUERY_SECRET,
        FullStrictRedactionAgent.QUERY_VISIBLE_VALUE,
        FullStrictRedactionAgent.URL_USER,
        FullStrictRedactionAgent.URL_PASSWORD,
        FullStrictRedactionAgent.URL_FRAGMENT,
        FullStrictRedactionAgent.AUTHORIZATION,
        FullStrictRedactionAgent.COOKIE,
        FullStrictRedactionAgent.QUOTED_SECRET,
        FullStrictRedactionAgent.TOKEN_PREFIX_VALUE,
        FullStrictRedactionAgent.REDACTED_KEY_ONE,
        FullStrictRedactionAgent.REDACTED_KEY_TWO,
        "opaque-after-quote-371",
    }
    assert all(secret not in content for secret in raw_secrets)
    assert rendered_args["headers"] == {
        "Authorization": "***",
        "Cookie": "***",
    }
    assert rendered_args["secret_value"] == "***"
    assert rendered_args["tokenPrefixData"] == "***"
    assert rendered_args["safe"] == {
        "path": "/tmp/visible",
        "items": ["alpha", ["beta"]],
    }
    assert rendered_args["request_url"] == (
        "prefix https://***@api.example.test/v1/items"
        "?access_token=***&visible=***#*** suffix"
    )
    assert "first-collision-value" in rendered_args.values()
    assert "second-collision-value" in rendered_args.values()
    assert len(rendered_args) == len(FullStrictRedactionAgent.ARGS)


@pytest.mark.asyncio
async def test_oversized_full_entry_splits_losslessly_in_separate_mode(
    monkeypatch, tmp_path
):
    adapter, _ = await _run_with_agent(
        monkeypatch,
        tmp_path,
        OversizedSingleFullEntryAgent,
        session_id="sess-full-overflow-separate",
        config_data={
            "display": {
                "tool_progress": "full",
                "tool_progress_grouping": "separate",
            }
        },
        adapter_cls=SmallLimitProgressAdapter,
    )

    expected = _expected_full_message(
        "full_payload_tool", OversizedSingleFullEntryAgent.ARGS
    )
    assert len(adapter.sent) > 1
    assert adapter.oversized_sends == []
    assert "".join(call["content"] for call in adapter.sent) == expected


@pytest.mark.asyncio
async def test_oversized_full_entry_splits_after_edit_failure(monkeypatch, tmp_path):
    adapter, _ = await _run_with_agent(
        monkeypatch,
        tmp_path,
        EditFailureThenOversizedFullAgent,
        session_id="sess-full-overflow-after-edit-failure",
        config_data={"display": {"tool_progress": "full"}},
        adapter_cls=FailFirstEditProgressAdapter,
    )

    expected = _expected_full_message(
        "full_payload_tool", EditFailureThenOversizedFullAgent.BIG_ARGS
    )
    start = next(
        idx
        for idx, call in enumerate(adapter.sent)
        if call["content"].startswith("⚙️ full_payload_tool\n")
    )
    assert adapter.failed_edit_ids
    assert adapter.oversized_sends == []
    assert "".join(call["content"] for call in adapter.sent[start:]) == expected


@pytest.mark.asyncio
async def test_full_split_uses_utf16_adapter_length_losslessly(monkeypatch, tmp_path):
    original_args = OversizedSingleFullEntryAgent.ARGS
    unicode_args = {"question": "astral-" + "🧪" * 150 + "-done"}
    monkeypatch.setattr(OversizedSingleFullEntryAgent, "ARGS", unicode_args)
    adapter, _ = await _run_with_agent(
        monkeypatch,
        tmp_path,
        OversizedSingleFullEntryAgent,
        session_id="sess-full-overflow-utf16",
        config_data={"display": {"tool_progress": "full"}},
        adapter_cls=Utf16SmallLimitProgressAdapter,
    )

    expected = _expected_full_message("full_payload_tool", unicode_args)
    assert original_args != unicode_args
    assert len(adapter.sent) > 1
    assert adapter.oversized_sends == []
    assert all(
        adapter.message_len_fn(call["content"]) <= adapter.MAX_MESSAGE_LENGTH
        for call in adapter.sent
    )
    assert "".join(call["content"] for call in adapter.sent) == expected


@pytest.mark.asyncio
async def test_full_split_finalizes_chunks_before_later_progress(monkeypatch, tmp_path):
    adapter, _ = await _run_with_agent(
        monkeypatch,
        tmp_path,
        OversizedThenSmallFullAgent,
        session_id="sess-full-overflow-state-reset",
        config_data={"display": {"tool_progress": "full"}},
        adapter_cls=TrackingSmallLimitProgressAdapter,
    )

    expected_big = _expected_full_message(
        "full_payload_tool", OversizedThenSmallFullAgent.BIG_ARGS
    )
    expected_small = _expected_full_message(
        "post_split_tool", OversizedThenSmallFullAgent.SMALL_ARGS
    )
    combined = "".join(call["content"] for call in adapter.sent)
    assert combined == expected_big + expected_small
    assert adapter.edited_message_ids == []


@pytest.mark.asyncio
async def test_exact_child_awaiter_propagates_direct_child_cancellation():
    from gateway.run_turn_progress import _await_exact_task_through_cancellation

    attempts = 0

    async def cancel_directly():
        nonlocal attempts
        attempts += 1
        raise asyncio.CancelledError

    child = asyncio.create_task(cancel_directly())
    waiter = asyncio.create_task(_await_exact_task_through_cancellation(child))
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(waiter, timeout=1.0)

    assert attempts == 1
    assert child.cancelled() is True
    assert waiter.cancelled() is True


@pytest.mark.asyncio
async def test_oversized_verbose_fence_keeps_upstream_single_entry_behavior(
    monkeypatch, tmp_path
):
    adapter, _ = await _run_with_agent(
        monkeypatch,
        tmp_path,
        OversizedVerboseTerminalAgent,
        session_id="sess-verbose-oversized-fence-unsplit",
        config_data={"display": {"tool_progress": "verbose"}},
        adapter_cls=SmallLimitCodeBlockProgressAdapter,
    )

    assert len(adapter.sent) == 1
    assert adapter.oversized_sends == [adapter.sent[0]["content"]]
    from agent.display import get_tool_emoji

    expected = (
        f"{get_tool_emoji('terminal', default='⚙️')} terminal\n```\n"
        f"{OversizedVerboseTerminalAgent.COMMAND}\n```"
    )
    assert adapter.sent[0]["content"] == expected
    assert all(call["content"] == expected for call in adapter.edits)


@pytest.mark.asyncio
async def test_oversized_single_full_entry_splits_losslessly(monkeypatch, tmp_path):
    adapter, result = await _run_with_agent(
        monkeypatch,
        tmp_path,
        OversizedSingleFullEntryAgent,
        session_id="sess-full-single-entry-overflow",
        config_data={"display": {"tool_progress": "full"}},
        adapter_cls=SmallLimitProgressAdapter,
    )

    expected = _expected_full_message(
        "full_payload_tool", OversizedSingleFullEntryAgent.ARGS
    )
    assert result["final_response"] == "done"
    assert isinstance(adapter, SmallLimitProgressAdapter)
    assert len(adapter.sent) > 1
    assert adapter.oversized_sends == []
    assert adapter.oversized_edits == []
    assert "".join(call["content"] for call in adapter.sent) == expected


@pytest.mark.asyncio
async def test_explicit_full_progress_survives_native_slack_task_card_opt_in(
    monkeypatch, tmp_path
):
    adapter, result = await _run_with_agent(
        monkeypatch, tmp_path, FullTerminalArgsAgent,
        session_id="sess-full-native-slack-cards",
        config_data={"display": {"tool_progress": "full"}},
        platform=Platform.SLACK,
        adapter_cls=NativeCardsProgressAdapter,
    )
    assert result["final_response"] == "done"
    _, rendered = _first_progress_json(adapter)
    assert rendered == FullTerminalArgsAgent.ARGS


@pytest.mark.parametrize("mode", ["full", "all"])
def test_full_uses_lossless_queue_while_compact_keeps_native_stream_owner(mode):
    from queue import Queue
    from gateway.run_turn_runner import TurnRunner
    from gateway.stream_consumer import GatewayStreamConsumer
    from gateway.turn_context import TurnContext

    consumer = GatewayStreamConsumer.__new__(GatewayStreamConsumer)
    consumer._use_native_streaming = True
    consumer._queue = Queue()
    ctx = TurnContext(
        source=SimpleNamespace(chat_id="native-chat"),
        _run_still_current=lambda: True,
        progress_mode=mode,
        tool_progress_enabled=True,
        progress_queue=Queue(),
        stream_consumer_holder=[consumer],
    )
    turn = TurnRunner(SimpleNamespace(_delivery_adapter_for=lambda _s: None), ctx)
    turn.progress_callback("tool.started", "terminal", "pwd", {"command": "pwd"})
    if mode == "full":
        marker, text = ctx.progress_queue.get_nowait()
        assert marker == "__full__"
        assert json.loads(text.split("\n", 1)[1]) == {"command": "pwd"}
        assert consumer._queue.empty()
    else:
        assert ctx.progress_queue.empty()
        assert "pwd" in consumer._queue.get_nowait()[1]


class FullReceiptProgressAdapter(SmallLimitProgressAdapter):
    """Capture native last-ID receipts and exact operation order."""

    partial = False
    multipart = True
    block_at = None
    instance = None

    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(platform=platform)
        type(self).instance = self
        self.operations = []
        self.receipts = []
        self.deleted = []
        self.cleanup_done = asyncio.Event()
        self.blocked = asyncio.Event()
        self.release = asyncio.Event()
        self.first_sent = threading.Event()
        self.send_attempts = []

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.send_attempts.append(content)
        if self.block_at == "chunk" and len(self.send_attempts) == 2:
            self.blocked.set()
            await self.release.wait()
        self.sent.append({"chat_id": chat_id, "content": content, "metadata": metadata})
        previous = (self._mint_id(),) if self.multipart else ()
        last = self._mint_id()
        receipt = SendResult(
            success=not self.partial, message_id=last,
            continuation_message_ids=previous,
            error="synthetic partial delivery" if self.partial else None,
            retryable=self.partial,
        )
        self.receipts.append(receipt)
        self.operations.append(("send", previous + (last,), content))
        self.first_sent.set()
        return receipt

    async def edit_message(self, chat_id, message_id, content):
        if self.block_at == "edit":
            self.blocked.set()
            await self.release.wait()
        self.operations.append(("edit", (message_id,), content))
        return await super().edit_message(chat_id, message_id, content)

    async def delete_message(self, chat_id, message_id):
        self.deleted.append(message_id)
        self.operations.append(("delete", (message_id,), None))
        expected_count = sum(len(r.continuation_message_ids) + 1 for r in self.receipts)
        if len(self.deleted) == expected_count:
            self.cleanup_done.set()
        return True

    async def confirm_final_delivery(self):
        """Exercise the native one-shot callback after simulated final delivery."""
        assert self.deleted == []
        assert len(self._post_delivery_callbacks) == 1
        session_key = next(iter(self._post_delivery_callbacks))
        callback = self.pop_post_delivery_callback(session_key)
        assert callable(callback)
        self.operations.append(("post_delivery", (), None))
        result = callback()
        if asyncio.iscoroutine(result):
            await result
        await asyncio.wait_for(self.cleanup_done.wait(), 3)
        assert self.pop_post_delivery_callback(session_key) is None


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.asyncio
async def test_full_cleanup_owns_continuations_and_partial_receipts_once(monkeypatch, tmp_path, partial):
    monkeypatch.setattr(FullReceiptProgressAdapter, "partial", partial)
    adapter, result = await _run_with_agent(
        monkeypatch, tmp_path, OversizedSingleFullEntryAgent,
        session_id="full-receipts", adapter_cls=FullReceiptProgressAdapter,
        config_data={"display": {"tool_progress": "full", "cleanup_progress": True}},
    )
    assert result["final_response"] == "done"
    acknowledged = [ident for receipt in adapter.receipts
                    for ident in (*receipt.continuation_message_ids, receipt.message_id)]
    assert acknowledged
    assert len(set(acknowledged)) == len(acknowledged)
    assert isinstance(adapter, FullReceiptProgressAdapter)
    await adapter.confirm_final_delivery()
    assert adapter.deleted == acknowledged
    assert len(adapter.send_attempts) == len(adapter.receipts)
    assert adapter.edits == []
    expected = _expected_full_message("full_payload_tool", OversizedSingleFullEntryAgent.ARGS)
    if partial:
        # The first partial receipt owns visible IDs, but must not replay or continue
        # a failed logical entry whose delivered text range is unknown.
        assert len(adapter.send_attempts) == 1
        assert expected.startswith(adapter.send_attempts[0])
    else:
        assert "".join(adapter.send_attempts) == expected
    assert all(op[0] == "send" for op in adapter.operations[:len(adapter.receipts)])


@pytest.mark.parametrize("block_at", ["edit", "chunk"])
@pytest.mark.asyncio
async def test_full_split_survives_two_parent_cancellations_without_retry(monkeypatch, tmp_path, block_at):
    from gateway.run_turn_runner import TurnRunner

    release_agent = threading.Event()
    ready = asyncio.Event()
    captured = {}
    original_sender = TurnRunner.send_progress_messages

    async def capture_sender(self):
        captured["task"] = asyncio.current_task()
        ready.set()
        return await original_sender(self)

    monkeypatch.setattr(TurnRunner, "send_progress_messages", capture_sender)
    monkeypatch.setattr(FullReceiptProgressAdapter, "block_at", block_at)
    monkeypatch.setattr(FullReceiptProgressAdapter, "multipart", False)
    small = {"question": "pending-before-cancel"}
    big = {"question": "cancel-" + "0123456789" * 70}

    class Agent(FakeAgent):
        def run_conversation(self, message, conversation_history=None, task_id=None):
            self.tool_progress_callback("tool.started", "pending_tool", "pending", small)
            assert FullReceiptProgressAdapter.instance.first_sent.wait(3)
            self.tool_progress_callback("tool.started", "full_payload_tool", "big", big)
            assert release_agent.wait(8)
            return {"final_response": "done", "messages": [], "api_calls": 1}

    run_task = asyncio.create_task(_run_with_agent(
        monkeypatch, tmp_path, Agent, session_id="full-double-cancel",
        adapter_cls=FullReceiptProgressAdapter,
        config_data={"display": {"tool_progress": "full", "cleanup_progress": True}},
    ))
    try:
        await asyncio.wait_for(ready.wait(), 3)
        adapter = FullReceiptProgressAdapter.instance
        await asyncio.wait_for(adapter.blocked.wait(), 4)
        parent = captured["task"]
        assert parent.cancel()
        await asyncio.sleep(0)
        assert not parent.done()
        assert parent.cancel()
        await asyncio.sleep(0)
        assert not parent.done()
        adapter.release.set()
        await asyncio.wait_for(asyncio.shield(parent), 3)
    finally:
        if FullReceiptProgressAdapter.instance:
            FullReceiptProgressAdapter.instance.release.set()
        release_agent.set()
        adapter, result = await asyncio.wait_for(run_task, 4)
    assert result["final_response"] == "done"
    expected_small = _expected_full_message("pending_tool", small)
    expected_big = _expected_full_message("full_payload_tool", big)
    assert adapter.send_attempts[0] == expected_small
    assert "".join(adapter.send_attempts[1:]) == expected_big
    assert [op[0] for op in adapter.operations[:3]] == ["send", "edit", "send"]
    assert [edit["content"] for edit in adapter.edits] == [expected_small]
    ids = [r.message_id for r in adapter.receipts]
    assert isinstance(adapter, FullReceiptProgressAdapter)
    await adapter.confirm_final_delivery()
    assert adapter.deleted == ids
    assert len(ids) == len(set(ids))


@pytest.mark.asyncio
@pytest.mark.parametrize("edit_path", ["rollover", "fallback"])
@pytest.mark.parametrize("state", ["current", "replacement", "interrupted"])
async def test_full_entry_rechecks_owner_after_suspended_edit(edit_path, state):
    from queue import Queue

    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext

    class SuspendedEditAdapter(FullReceiptProgressAdapter):
        multipart = False
        block_at = "edit"

        async def edit_message(self, chat_id, message_id, content):
            result = await super().edit_message(chat_id, message_id, content)
            if edit_path == "fallback":
                return SendResult(success=False, retryable=False, error="editing unavailable")
            return result

    adapter = SuspendedEditAdapter()
    owner = _make_runner(adapter)
    key = "suspended-full-edit"
    generation = owner._begin_session_run_generation(key)
    agent = SimpleNamespace(is_interrupted=False)
    ctx = TurnContext(
        source=SessionSource(platform=adapter.platform, chat_id="chat"),
        _run_still_current=lambda: owner._is_session_run_current(key, generation),
        agent_holder=[agent], progress_mode="full", tool_progress_enabled=True,
        progress_queue=Queue(), _cleanup_progress=True,
    )
    turn = TurnRunner(owner, ctx)
    entries = []
    for name, question in (("prefix_tool", "already acknowledged"), ("next_tool", "complete new arguments")):
        turn.progress_callback("tool.started", name, "short", {"question": question})
        marker, entry = ctx.progress_queue.get_nowait()
        assert marker == "__full__"
        entries.append(entry)
    prefix, entry = entries
    st = turn._progress_edit_state(adapter)
    combined = prefix + "\n" + entry
    st._PROGRESS_TEXT_LIMIT = (
        max(map(st._progress_len_fn, entries)) if edit_path == "rollover"
        else st._progress_len_fn(combined)
    )
    assert turn._split_full_entry(st, entry) == [entry]
    await turn._send_full_progress_entry(st, prefix)
    acknowledged = adapter.receipts[0].message_id
    assert ctx._cleanup_msg_ids == [acknowledged]

    pending = asyncio.create_task(turn._send_full_progress_entry(st, entry))
    try:
        await asyncio.wait_for(adapter.blocked.wait(), 1)
        assert adapter.send_attempts == [prefix]
        if state == "replacement":
            owner._begin_session_run_generation(key)
        elif state == "interrupted":
            agent.is_interrupted = True
        # The native shield must settle this exact edit, even if its caller is cancelled twice.
        for _ in range(2):
            pending.cancel()
            await asyncio.sleep(0)
            assert not pending.done()
        adapter.release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(pending, 2)
    finally:
        adapter.release.set()
        await asyncio.wait_for(asyncio.gather(pending, return_exceptions=True), 2)

    assert [edit["message_id"] for edit in adapter.edits] == [acknowledged]
    assert [edit["content"] for edit in adapter.edits] == [prefix if edit_path == "rollover" else combined]
    assert adapter.send_attempts == ([prefix, entry] if state == "current" else [prefix])
    ids = [receipt.message_id for receipt in adapter.receipts]
    assert ctx._cleanup_msg_ids == ids
    assert ids[0] == acknowledged and len(ids) == len(set(ids))
    if state != "current":
        assert ids == [acknowledged]


@pytest.mark.asyncio
async def test_log_mode_real_turn_remains_chat_silent(monkeypatch, tmp_path):
    class LogModeAgent(FullTerminalArgsAgent):
        def run_conversation(self, message, conversation_history=None, task_id=None):
            cb = self.tool_progress_callback
            assert cb is not None
            cb("tool.started", "terminal", self.ARGS["command"], dict(self.ARGS))
            return {"final_response": "done", "messages": [], "api_calls": 1}

    adapter, result = await _run_with_agent(
        monkeypatch, tmp_path, LogModeAgent, session_id="native-log-control",
        config_data={"display": {"tool_progress": "log"}},
    )
    assert result["final_response"] == "done"
    assert adapter.sent == [] and adapter.edits == []
    log_text = (tmp_path / "logs" / "tool_calls.log").read_text()
    assert "terminal:" in log_text
    assert FullTerminalArgsAgent.ARGS["command"] in log_text


def test_full_mode_nested_header_pairs_preserve_shape_and_inputs():
    from copy import deepcopy
    from gateway.full_progress_redaction import _redact_full_progress_args

    original = {"headers": [["Authorization", "opaque-pair-71"],
                            ("X-API-Key", "opaque-pair-72"),
                            ["Accept", "application/json"]],
                "lines": ["Cookie: opaque-cookie-73", "Accept: text/plain"]}
    before = deepcopy(original)
    safe = _redact_full_progress_args(original)
    assert safe["headers"] == [["Authorization", "***"], ("X-API-Key", "***"),
                               ["Accept", "application/json"]]
    assert safe["lines"] == ["Cookie: ***", "Accept: text/plain"]
    assert original == before


@pytest.mark.parametrize("secret_kind", ["vendor", "vault"])
def test_full_mode_untrusted_mask_markers_cannot_bypass_enqueue_redaction(
    monkeypatch, secret_kind
):
    from copy import deepcopy
    from queue import Queue

    from agent import redact
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext

    monkeypatch.setattr(redact, "_redact_enabled", lambda: False)
    monkeypatch.setattr(redact, "_VAULT_REDACTION_VALUES", {})
    secret = "ghp_" + "a" * 36 if secret_kind == "vendor" else "isolated-vault-value-519"
    if secret_kind == "vault":
        redact.register_vault_redaction_value(secret)
    marked = "***" + secret
    scanned = redact.redact_sensitive_text(marked, force=True)
    assert secret not in scanned
    literal_marker = "__HERMES_FULL_MASKED_WORD_0__"
    args = {
        "question": marked,
        "command": (
            f"{literal_marker} export API_TOKEN='opaque phrase' "
            f"echo '{marked}' --label='visible ***'"
        ),
        "nearby": ["visible ***", {literal_marker: "🧪  keep\n\twhitespace"}],
    }
    original = deepcopy(args)
    ctx = TurnContext(
        source=SimpleNamespace(chat_id="enqueue-test"),
        _run_still_current=lambda: True,
        progress_mode="full", tool_progress_enabled=True, progress_queue=Queue(),
    )
    turn = TurnRunner(SimpleNamespace(_delivery_adapter_for=lambda _s: None), ctx)
    turn.progress_callback("tool.started", "full_payload_tool", "short", args)

    marker, content = ctx.progress_queue.get_nowait()
    assert marker == "__full__"
    assert args == original
    assert secret not in content
    decoded = json.loads(content.split("\n", 1)[1])
    assert decoded == {
        "question": scanned,
        "command": (
            f"{literal_marker} export API_TOKEN='***' "
            f"echo '{scanned}' --label='visible ***'"
        ),
        "nearby": original["nearby"],
    }
    assert ctx.progress_queue.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("repeats", [1, 100], ids=["editable", "single-long-arg"])
async def test_full_mode_mattermost_http_preserves_json_fidelity(
    monkeypatch, tmp_path, repeats
):
    from copy import deepcopy
    from unittest.mock import AsyncMock, MagicMock

    from plugins.platforms.mattermost.adapter import MAX_POST_LENGTH, MattermostAdapter

    unit = '🧪  漢字\n\t![diagram](https://example.test/a.png)  '
    escaped = r'\![escaped](https://example.test/b.png) \\![double](https://example.test/c.png) \u0021[literal]'
    initial = {r'\![label](https://example.test/d.png)': [unit, escaped]}
    long_args = {"question": (unit + escaped + '  "quoted"  ') * repeats}
    originals = deepcopy([initial, long_args])
    requests = []
    posts = {}
    first_post = threading.Event()
    adapter = MattermostAdapter(PlatformConfig(
        enabled=True, token="synthetic-test-token",
        extra={"url": "https://mm.example.test", "reply_mode": "thread"},
    ))
    adapter._resolve_root_id = AsyncMock(return_value="thread-root")
    adapter.send_typing = AsyncMock()
    adapter.stop_typing = AsyncMock()
    adapter._session = MagicMock()

    def http_response(method, url, **kwargs):
        payload = deepcopy(kwargs["json"])
        requests.append((method, url, payload))
        if method == "POST":
            assert url.endswith("/api/v4/posts")
            message_id = f"post-{len(posts) + 1}"
        else:
            assert url.endswith("/patch")
            message_id = url.split("/")[-2]
            assert message_id in posts
        posts[message_id] = payload["message"]
        response = AsyncMock()
        response.status = 200
        response.json.return_value = {"id": message_id}
        response.__aenter__.return_value = response
        if method == "POST":
            first_post.set()
        return response

    adapter._session.post.side_effect = lambda url, **kw: http_response("POST", url, **kw)
    adapter._session.put.side_effect = lambda url, **kw: http_response("PUT", url, **kw)

    class Agent(FakeAgent):
        def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
            self.tool_progress_callback("tool.started", "initial_payload_tool", "short", initial)
            assert first_post.wait(2), "initial progress was not acknowledged"
            self.tool_progress_callback("tool.started", "full_payload_tool", "short", long_args)
            assert [initial, long_args] == originals
            return {"final_response": "done", "messages": [], "api_calls": 1}

    _, result = await _run_with_agent(
        monkeypatch, tmp_path, Agent, session_id=f"mm-json-{repeats}",
        platform=Platform.MATTERMOST, chat_id="channel-test", thread_id="thread-root",
        adapter_cls=lambda platform: adapter,
        config_data={"display": {"tool_progress": "full", "tool_preview_length": 1}},
    )

    assert result["final_response"] == "done"
    assert [initial, long_args] == originals
    assert {method for method, _, _ in requests} == {"POST", "PUT"}
    assert all(adapter.message_len_fn(body["message"]) <= MAX_POST_LENGTH
               for _, _, body in requests)
    assert all(body["root_id"] == "thread-root" for method, _, body in requests if method == "POST")
    if repeats > 1:
        assert adapter.message_len_fn(json.dumps(long_args, ensure_ascii=False)) > MAX_POST_LENGTH
        assert len(posts) > 2
    remaining = "".join(posts.values())
    for tool, expected in zip(("initial_payload_tool", "full_payload_tool"), originals):
        header, separator, body = remaining.partition("\n")
        assert separator and header.endswith(tool)
        decoded, end = json.JSONDecoder().raw_decode(body)
        assert decoded == expected
        remaining = body[end:].removeprefix("\n")
    assert remaining == ""
    # The adapter's ordinary image-Markdown policy remains native.
    assert adapter.format_message("![diagram](https://example.test/a.png)") == "https://example.test/a.png"
