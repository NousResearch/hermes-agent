"""Regression tests for the s1 extraction of ``plugins/platforms/telegram/adapter.py``.

Covers the PURE methods that moved into the mixin modules (wave-1 blind
implementer w1b, shard s1):

- ``text_format`` (c1): MarkdownV2 escape/strip/chunk-fence separation
- ``config_mixin`` (c8): ``_env_float_clamped`` / ``_coerce_bool_extra`` /
  ``_coerce_float_extra`` config parsing
- ``network_mixin`` (c11): transport-error classifiers
- ``topics_mixin`` (c12): metadata parsing + thread-id normalization +
  group/forum classification
- ``authz_mixin`` (c4): ``_scoped_gate_env`` (module-level)
- ``rich_mixin`` (c2): ``_rich_normalize_linebreaks`` (module-level) and
  ``_needs_rich_rendering`` group classification

No telegram network is touched: adapters are built with ``object.__new__`` +
stub config, matching the existing bare-adapter pattern used by
``tests/gateway/test_telegram_auth_check.py``.
"""

import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_telegram_mock():
    """Mirror the existing telegram mock so the adapter imports cleanly when
    python-telegram-bot is not installed (idempotent when it is)."""
    import sys

    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.constants.ChatType.GROUP = "group"
    telegram_mod.constants.ChatType.SUPERGROUP = "supergroup"
    telegram_mod.constants.ChatType.CHANNEL = "channel"
    telegram_mod.constants.ChatType.PRIVATE = "private"
    telegram_mod.error.NetworkError = type("NetworkError", (OSError,), {})
    telegram_mod.error.TimedOut = type("TimedOut", (OSError,), {})
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)
    sys.modules.setdefault("telegram.error", telegram_mod.error)


_ensure_telegram_mock()

from plugins.platforms.telegram.adapter import (  # noqa: E402
    TelegramAdapter,
    _escape_mdv2,
    _separate_chunk_indicator_from_fence,
    _strip_mdv2,
)
from plugins.platforms.telegram.authz_mixin import _scoped_gate_env  # noqa: E402
from plugins.platforms.telegram.rich_mixin import _rich_normalize_linebreaks  # noqa: E402
from plugins.platforms.telegram.text_format import (  # noqa: E402
    _MDV2_ESCAPE_RE,
    _CHUNK_INDICATOR_ON_FENCE_RE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _StubConfig:
    def __init__(self, extra=None):
        self.extra = dict(extra or {})


def _bare_adapter(**extra):
    """Bare adapter via object.__new__ + stub config (no __init__, no network)."""
    adapter = object.__new__(TelegramAdapter)
    adapter.config = _StubConfig(extra)
    adapter._bot = SimpleNamespace(do_api_request=None)
    return adapter


def _error(name, message, cause=None, context=None):
    """Synthetic exception with a PTB-style class name for classifier tests."""
    cls = type(name, (Exception,), {})
    err = cls(message)
    if cause is not None:
        err.__cause__ = cause
    if context is not None:
        err.__context__ = context
    return err


# ---------------------------------------------------------------------------
# c1 — text_format: MarkdownV2 escape/strip/fence separation
# ---------------------------------------------------------------------------

class TestMarkdownV2Formatting:
    def test_mdv2_escape_re_matches_special_characters(self):
        assert _MDV2_ESCAPE_RE.match("_")
        assert _MDV2_ESCAPE_RE.match("*")
        assert _MDV2_ESCAPE_RE.match("[")
        assert not _MDV2_ESCAPE_RE.match("a")
        assert not _MDV2_ESCAPE_RE.match(" ")

    def test_escape_mdv2_escapes_specials(self):
        assert _escape_mdv2("a_b") == r"a\_b"
        assert _escape_mdv2("v2.0") == r"v2\.0"
        assert _escape_mdv2("wow!") == r"wow\!"
        assert _escape_mdv2("plain") == "plain"
        assert _escape_mdv2("") == ""

    def test_strip_mdv2_removes_escapes_and_markers(self):
        assert _strip_mdv2(r"hello\.world\!") == "hello.world!"
        assert _strip_mdv2("*bold* and _italic_") == "bold and italic"
        assert _strip_mdv2("my_variable_name") == "my_variable_name"
        assert _strip_mdv2("~gone~") == "gone"
        assert _strip_mdv2("||spoiler||") == "spoiler"
        assert _strip_mdv2("plain text") == "plain text"

    def test_fence_re_matches_indicator_on_fence(self):
        m = _CHUNK_INDICATOR_ON_FENCE_RE.search("``` \\(1/2\\)")
        assert m is not None
        assert m.group("indicator") == r"\(1/2\)"
        assert _CHUNK_INDICATOR_ON_FENCE_RE.search("```") is None
        assert _CHUNK_INDICATOR_ON_FENCE_RE.search("```python") is None

    def test_separate_chunk_indicator_from_fence(self):
        out = _separate_chunk_indicator_from_fence("line\n``` \\(1/2\\)\nnext")
        assert out == "line\n```\n\\(1/2\\)\nnext"
        # No-op when there is no indicator on the fence line.
        assert _separate_chunk_indicator_from_fence("plain") == "plain"


# ---------------------------------------------------------------------------
# c8 — config parsing
# ---------------------------------------------------------------------------

class TestConfigParsing:
    def test_env_float_clamped_default(self, monkeypatch):
        monkeypatch.delenv("HERMES_TG_TEST_FLOAT", raising=False)
        adapter = _bare_adapter()
        assert adapter._env_float_clamped("HERMES_TG_TEST_FLOAT", 0.3) == 0.3

    def test_env_float_clamped_parses_and_clamps(self, monkeypatch):
        monkeypatch.setenv("HERMES_TG_TEST_FLOAT", "5.5")
        adapter = _bare_adapter()
        assert adapter._env_float_clamped(
            "HERMES_TG_TEST_FLOAT", 0.3, min_value=0.08, max_value=2.0
        ) == 2.0
        assert adapter._env_float_clamped(
            "HERMES_TG_TEST_FLOAT", 0.3, min_value=0.08, max_value=10.0
        ) == 5.5

    def test_env_float_clamped_rejects_garbage_and_nonfinite(self, monkeypatch):
        monkeypatch.setenv("HERMES_TG_TEST_FLOAT", "nan")
        adapter = _bare_adapter()
        assert adapter._env_float_clamped("HERMES_TG_TEST_FLOAT", 0.3) == 0.3
        monkeypatch.setenv("HERMES_TG_TEST_FLOAT", "not-a-number")
        assert adapter._env_float_clamped("HERMES_TG_TEST_FLOAT", 0.3) == 0.3

    def test_coerce_bool_extra(self):
        adapter = _bare_adapter(disable_link_previews=True)
        assert adapter._coerce_bool_extra("disable_link_previews", False) is True
        adapter = _bare_adapter(disable_link_previews="yes")
        assert adapter._coerce_bool_extra("disable_link_previews", False) is True
        adapter = _bare_adapter(disable_link_previews="off")
        assert adapter._coerce_bool_extra("disable_link_previews", True) is False
        adapter = _bare_adapter()
        assert adapter._coerce_bool_extra("missing_key", True) is True

    def test_coerce_float_extra(self):
        adapter = _bare_adapter(typing_cooldown_seconds="45")
        assert adapter._coerce_float_extra("typing_cooldown_seconds", 30.0) == 45.0
        adapter = _bare_adapter(typing_cooldown_seconds="999")
        assert adapter._coerce_float_extra(
            "typing_cooldown_seconds", 30.0, min_value=1.0, max_value=300.0
        ) == 300.0
        adapter = _bare_adapter(typing_cooldown_seconds="junk")
        assert adapter._coerce_float_extra("typing_cooldown_seconds", 30.0) == 30.0
        adapter = _bare_adapter()
        assert adapter._coerce_float_extra("missing_key", 30.0) == 30.0


# ---------------------------------------------------------------------------
# c11 — network error classification
# ---------------------------------------------------------------------------

class TestNetworkClassifiers:
    def test_looks_like_polling_conflict(self):
        conflict = _error("Conflict", "terminated by other getupdates request")
        assert TelegramAdapter._looks_like_polling_conflict(conflict) is True
        other = _error("BadRequest", "bad request")
        assert TelegramAdapter._looks_like_polling_conflict(other) is False

    def test_looks_like_network_error(self):
        assert TelegramAdapter._looks_like_network_error(_error("NetworkError", "conn reset")) is True
        assert TelegramAdapter._looks_like_network_error(_error("TimedOut", "timed out")) is True
        assert TelegramAdapter._looks_like_network_error(_error("BadRequest", "bad")) is False
        assert TelegramAdapter._looks_like_network_error(_error("Forbidden", "forbidden")) is False
        assert TelegramAdapter._looks_like_network_error(OSError("socket down")) is True

    def test_looks_like_connect_timeout_walks_cause_chain(self):
        connect = _error("ConnectTimeout", "connect timed out")
        wrapped = _error("TimedOut", "request timed out", cause=connect)
        assert TelegramAdapter._looks_like_connect_timeout(wrapped) is True
        assert TelegramAdapter._looks_like_connect_timeout(_error("TimedOut", "plain timeout")) is False
        assert TelegramAdapter._looks_like_connect_timeout(_error("TimedOut", "")) is False

    def test_looks_like_pool_timeout(self):
        pool = _error(
            "PoolTimeout",
            "Pool timeout: All connections in the connection pool are occupied. "
            "Request was *not* sent to Telegram.",
        )
        assert TelegramAdapter._looks_like_pool_timeout(pool) is True
        assert TelegramAdapter._looks_like_pool_timeout(_error("TimedOut", "plain timeout")) is False


# ---------------------------------------------------------------------------
# c12 — topics: metadata parsing, thread-id normalization, group classification
# ---------------------------------------------------------------------------

class TestTopicsRouting:
    def test_metadata_thread_id(self):
        assert TelegramAdapter._metadata_thread_id(None) is None
        assert TelegramAdapter._metadata_thread_id({}) is None
        assert TelegramAdapter._metadata_thread_id({"thread_id": "42"}) == "42"
        assert TelegramAdapter._metadata_thread_id({"message_thread_id": 7}) == "7"
        # thread_id takes precedence (checked first via ``or`` short-circuit)
        assert (
            TelegramAdapter._metadata_thread_id(
                {"thread_id": "42", "message_thread_id": "7"}
            )
            == "42"
        )

    def test_metadata_direct_messages_topic_id(self):
        assert TelegramAdapter._metadata_direct_messages_topic_id(None) is None
        assert (
            TelegramAdapter._metadata_direct_messages_topic_id(
                {"direct_messages_topic_id": "5"}
            )
            == "5"
        )
        assert (
            TelegramAdapter._metadata_direct_messages_topic_id(
                {"telegram_direct_messages_topic_id": 9}
            )
            == "9"
        )

    def test_metadata_reply_to_message_id(self):
        assert TelegramAdapter._metadata_reply_to_message_id(None) is None
        assert (
            TelegramAdapter._metadata_reply_to_message_id(
                {"telegram_reply_to_message_id": "11"}
            )
            == 11
        )

    def test_message_thread_id_for_send_normalizes_general_topic(self):
        # "1" (forum General topic) must be omitted on sends.
        assert TelegramAdapter._message_thread_id_for_send(None) is None
        assert TelegramAdapter._message_thread_id_for_send("1") is None
        assert TelegramAdapter._message_thread_id_for_send(1) is None
        assert TelegramAdapter._message_thread_id_for_send("42") == 42

    def test_message_thread_id_for_typing_keeps_general_topic(self):
        # Asymmetric with _message_thread_id_for_send on purpose: typing
        # bubbles need the real thread id (including "1").
        assert TelegramAdapter._message_thread_id_for_typing(None) is None
        assert TelegramAdapter._message_thread_id_for_typing("1") == 1
        assert TelegramAdapter._message_thread_id_for_typing("42") == 42

    def test_is_private_dm_topic_send_classification(self):
        # Reply-anchor fallback send → private DM topic send.
        metadata = {
            "telegram_dm_topic_reply_fallback": True,
            "telegram_reply_to_message_id": 3,
        }
        assert TelegramAdapter._is_private_dm_topic_send("c", "7", metadata) is True
        # Direct-messages topic id with fallback + anchor → True.
        metadata2 = {
            "direct_messages_topic_id": "9",
            "telegram_dm_topic_reply_fallback": True,
            "telegram_reply_to_message_id": 3,
        }
        assert TelegramAdapter._is_private_dm_topic_send("c", None, metadata2) is True
        # Hermes-created topic send marker → False.
        metadata3 = {"telegram_dm_topic_created_for_send": True}
        assert TelegramAdapter._is_private_dm_topic_send("c", "7", metadata3) is False
        # Plain group send → False.
        assert TelegramAdapter._is_private_dm_topic_send("c", "7", None) is False

    def test_reply_to_message_id_for_send(self):
        assert TelegramAdapter._reply_to_message_id_for_send("5", None) == 5
        metadata = {"telegram_dm_topic_reply_fallback": True, "telegram_reply_to_message_id": 9}
        assert TelegramAdapter._reply_to_message_id_for_send(None, metadata) == 9
        # reply_to_mode="off" suppresses the DM-topic fallback anchor.
        assert TelegramAdapter._reply_to_message_id_for_send(None, metadata, reply_to_mode="off") is None

    def test_thread_kwargs_for_send(self):
        assert TelegramAdapter._thread_kwargs_for_send("c", "42", None) == {
            "message_thread_id": 42
        }
        assert TelegramAdapter._thread_kwargs_for_send("c", "1", None) == {
            "message_thread_id": None
        }
        metadata = {"direct_messages_topic_id": "9"}
        assert TelegramAdapter._thread_kwargs_for_send("c", None, metadata) == {
            "message_thread_id": None,
            "direct_messages_topic_id": 9,
        }
        fallback = {
            "telegram_dm_topic_reply_fallback": True,
            "telegram_reply_to_message_id": 3,
        }
        assert TelegramAdapter._thread_kwargs_for_send("c", "7", fallback) == {
            "message_thread_id": 7
        }

    def test_is_bad_request_error(self):
        bad = _error("BadRequest", "bad request")
        assert TelegramAdapter._is_bad_request_error(bad) is True
        assert TelegramAdapter._is_bad_request_error(_error("Forbidden", "nope")) is False

    def test_is_thread_not_found_error(self):
        assert TelegramAdapter._is_thread_not_found_error(
            _error("BadRequest", "message thread not found")
        ) is True
        assert TelegramAdapter._is_thread_not_found_error(
            _error("BadRequest", "something else")
        ) is False

    def test_dm_topic_missing_anchor_error_message(self):
        assert "reply anchor" in TelegramAdapter._dm_topic_missing_anchor_error()

    def test_should_retry_without_dm_topic_reply_anchor(self):
        bad = _error("BadRequest", "message to be replied not found")
        metadata = {"telegram_dm_topic_reply_fallback": True}
        assert (
            TelegramAdapter._should_retry_without_dm_topic_reply_anchor(
                bad, metadata, 3
            )
            is True
        )
        # Non-BadRequest never retries.
        assert (
            TelegramAdapter._should_retry_without_dm_topic_reply_anchor(
                _error("NetworkError", "boom"), metadata, 3
            )
            is False
        )
        # No fallback marker → never retries.
        assert (
            TelegramAdapter._should_retry_without_dm_topic_reply_anchor(bad, None, None)
            is False
        )
        # direct_messages_topic_id rejected with topic marker → retry.
        topic_err = _error("BadRequest", "topic_deleted")
        metadata_topic = {
            "telegram_dm_topic_reply_fallback": True,
            "direct_messages_topic_id": "9",
        }
        assert (
            TelegramAdapter._should_retry_without_dm_topic_reply_anchor(
                topic_err, metadata_topic, None
            )
            is True
        )


# ---------------------------------------------------------------------------
# c4 — _scoped_gate_env (module-level helper)
# ---------------------------------------------------------------------------

class TestScopedGateEnv:
    def test_single_profile_fallback_to_os_environ(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111,222")
        assert _scoped_gate_env("TELEGRAM_ALLOWED_USERS") == "111,222"

    def test_missing_env_returns_default(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
        assert _scoped_gate_env("TELEGRAM_ALLOWED_USERS") == ""
        assert _scoped_gate_env("TELEGRAM_ALLOWED_USERS", "x") == "x"

    def test_exposed_from_adapter_module(self, monkeypatch):
        # Existing call sites (and tests) import _scoped_gate_env from the
        # adapter module; the re-import must resolve to the same function.
        from plugins.platforms.telegram.adapter import _scoped_gate_env as adapter_env

        monkeypatch.setenv("TELEGRAM_GROUP_ALLOWED_CHATS", "-1001")
        assert adapter_env("TELEGRAM_GROUP_ALLOWED_CHATS") == "-1001"


# ---------------------------------------------------------------------------
# c2 — rich helpers
# ---------------------------------------------------------------------------

class TestRichHelpers:
    def test_rich_normalize_linebreaks_single_newlines(self):
        assert _rich_normalize_linebreaks("a\nb") == "a  \nb"
        # Paragraph breaks stay untouched.
        assert _rich_normalize_linebreaks("a\n\nb") == "a\n\nb"
        # Fenced code block is protected.
        out = _rich_normalize_linebreaks("before\n```\nx\ny\n```\nafter")
        assert "```\nx\ny\n```" in out
        assert out.startswith("before  \n```")
        # Pipe table block is protected.
        table = "| a | b |\n| - | - |\n| 1 | 2 |\n"
        # Rows are protected; the trailing row separator newline is prose and
        # receives a Markdown hard break (matches live adapter behavior).
        assert _rich_normalize_linebreaks(table) == "| a | b |\n| - | - |\n| 1 | 2 |  \n"

    def test_needs_rich_rendering(self):
        adapter = _bare_adapter()
        # TABLE_SEPARATOR_RE matches delimiter rows (| - | - |), not headers.
        assert adapter._needs_rich_rendering("| - | - |") is True
        assert adapter._needs_rich_rendering("| a | b |") is False
        assert adapter._needs_rich_rendering("- [x] done") is True
        assert adapter._needs_rich_rendering("<details>") is True
        assert adapter._needs_rich_rendering("x $$ y") is True
        assert adapter._needs_rich_rendering("plain text") is False
        assert adapter._needs_rich_rendering("") is False

    def test_rich_eligible_gates(self):
        adapter = _bare_adapter()
        adapter._rich_messages_enabled = True
        adapter._rich_send_disabled = False
        adapter._bot = SimpleNamespace(
            do_api_request=MagicMock()  # sync callable → _bot_supports_rich False
        )
        assert adapter._bot_supports_rich() is False
        # Without an async do_api_request, rich is never eligible.
        assert adapter._rich_eligible("plain") is False

    def test_streaming_overflow_limit_none_when_rich_unavailable(self):
        adapter = _bare_adapter()
        adapter._rich_messages_enabled = True
        adapter._rich_send_disabled = False
        adapter._bot = SimpleNamespace(do_api_request=None)
        assert adapter.streaming_overflow_limit() is None

    def test_prefers_fresh_final_streaming_always_false(self):
        adapter = _bare_adapter()
        assert adapter.prefers_fresh_final_streaming("x", None) is False

    def test_rich_message_payload_builds_raw_markdown(self):
        adapter = _bare_adapter()
        payload = adapter._rich_message_payload("a\nb")
        assert payload["markdown"] == "a  \nb"
        payload2 = adapter._rich_message_payload("x", skip_entity_detection=True)
        assert payload2["skip_entity_detection"] is True

    def test_rich_fallback_error_classification(self):
        adapter = _bare_adapter()
        bad = _error("BadRequest", "unsupported block")
        assert adapter._is_rich_fallback_error(bad) is True
        cap = _error("EndpointNotFound", "method not found")
        assert adapter._is_rich_fallback_error(cap) is True
        trans = _error("NetworkError", "connection reset")
        assert adapter._is_rich_fallback_error(trans) is False
