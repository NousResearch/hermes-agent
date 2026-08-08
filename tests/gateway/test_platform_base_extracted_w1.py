"""Regression tests for the wave-1 extraction of ``gateway/platforms/base.py``.

Shard s1 move clusters covered (verbatim extraction):

* ``c6`` proxy resolution / NO_PROXY matching -> ``gateway/platforms/proxy_utils.py``
* ``c4`` UTF-16 text metrics                       -> ``gateway/platforms/text_metrics.py``
* ``c1`` thread/reply routing metadata             -> ``gateway/platforms/routing_metadata.py``

Two contracts are asserted here:

1. Behavior of the moved pure helpers (unchanged semantics).
2. Re-export parity: every moved function is still importable from
   ``gateway.platforms.base`` and is the *same object* as the one in its new
   module, so all existing ``from gateway.platforms.base import ...`` call
   sites (discord adapter, stream_consumer, helpers, tests, ...) keep working.
"""

import sys
from types import SimpleNamespace

import pytest

from gateway.platforms.base import (
    _custom_unit_to_cp,
    _mark_notify_metadata,
    _prefix_within_utf16_limit,
    _reply_anchor_for_event,
    _split_host_port,
    _thread_metadata_for_source,
    is_host_excluded_by_no_proxy,
    proxy_kwargs_for_aiohttp,
    proxy_kwargs_for_bot,
    resolve_proxy_url,
    should_bypass_proxy,
    utf16_len,
)
from gateway.platforms.proxy_utils import (
    _detect_macos_system_proxy,
    _no_proxy_entries,
    _no_proxy_entry_matches,
    is_host_excluded_by_no_proxy as proxy_is_host_excluded_by_no_proxy,
    proxy_kwargs_for_aiohttp as proxy_proxy_kwargs_for_aiohttp,
    proxy_kwargs_for_bot as proxy_proxy_kwargs_for_bot,
    resolve_proxy_url as proxy_resolve_proxy_url,
    should_bypass_proxy as proxy_should_bypass_proxy,
)
from gateway.platforms.routing_metadata import (
    _mark_notify_metadata as routing_mark_notify_metadata,
    _reply_anchor_for_event as routing_reply_anchor_for_event,
    _thread_metadata_for_source as routing_thread_metadata_for_source,
)
from gateway.platforms.text_metrics import (
    _custom_unit_to_cp as tm_custom_unit_to_cp,
    _prefix_within_utf16_limit as tm_prefix_within_utf16_limit,
    utf16_len as tm_utf16_len,
)


# ─── re-export parity ────────────────────────────────────────────────────────

def test_reexport_identity_proxy():
    assert should_bypass_proxy is proxy_should_bypass_proxy
    assert resolve_proxy_url is proxy_resolve_proxy_url
    assert proxy_kwargs_for_bot is proxy_proxy_kwargs_for_bot
    assert proxy_kwargs_for_aiohttp is proxy_proxy_kwargs_for_aiohttp
    assert is_host_excluded_by_no_proxy is proxy_is_host_excluded_by_no_proxy


def test_reexport_identity_text_metrics():
    assert utf16_len is tm_utf16_len
    assert _prefix_within_utf16_limit is tm_prefix_within_utf16_limit
    assert _custom_unit_to_cp is tm_custom_unit_to_cp


def test_reexport_identity_routing_metadata():
    assert _thread_metadata_for_source is routing_thread_metadata_for_source
    assert _mark_notify_metadata is routing_mark_notify_metadata
    assert _reply_anchor_for_event is routing_reply_anchor_for_event


# ─── c4 text metrics ─────────────────────────────────────────────────────────

def test_utf16_len_bmp_only():
    assert utf16_len("hello") == 5
    assert utf16_len("") == 0


def test_utf16_len_surrogate_pairs():
    # U+1F600 (😀) is outside the BMP: one code point, two UTF-16 code units.
    assert utf16_len("😀") == 2
    assert utf16_len("a😀b") == 4


def test_prefix_within_utf16_limit_under_limit():
    assert _prefix_within_utf16_limit("abcd", 10) == "abcd"
    assert _prefix_within_utf16_limit("", 0) == ""


def test_prefix_within_utf16_limit_respects_surrogate_boundary():
    s = "a😀b"
    assert _prefix_within_utf16_limit(s, 3) == "a😀"  # not "a" + half a surrogate
    assert _prefix_within_utf16_limit(s, 2) == "a"
    assert _prefix_within_utf16_limit(s, 4) == s


def test_custom_unit_to_cp_with_utf16_units():
    # len_fn measures UTF-16 units; codepoint offsets must not split pairs.
    assert _custom_unit_to_cp("a😀b", 3, utf16_len) == 2
    assert _custom_unit_to_cp("a😀b", 4, utf16_len) == 3
    assert _custom_unit_to_cp("abcd", 2, len) == 2


# ─── c6 proxy resolution / NO_PROXY ──────────────────────────────────────────

def test_should_bypass_proxy_no_env(monkeypatch):
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    assert should_bypass_proxy(None) is False
    assert should_bypass_proxy("api.example.com") is False


def test_should_bypass_proxy_matches_env(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "example.com, .internal")
    assert should_bypass_proxy("api.example.com") is True
    assert should_bypass_proxy("svc.internal") is True
    assert should_bypass_proxy("other.net") is False


def test_should_bypass_proxy_lowercase_env(monkeypatch):
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.setenv("no_proxy", "*.example.com")
    assert should_bypass_proxy("a.b.example.com") is True


def test_no_proxy_entries_merges_both_env_vars(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "a.com")
    if sys.platform == "win32":
        # Windows environment variables are case-insensitive, so the two
        # spellings alias a single variable; only one entry can exist.
        assert set(_no_proxy_entries()) == {"a.com"}
    else:
        monkeypatch.setenv("no_proxy", "b.com")
        assert set(_no_proxy_entries()) == {"a.com", "b.com"}


def test_no_proxy_entry_matches_cidr_and_ip():
    assert _no_proxy_entry_matches("10.0.0.0/8", "10.1.2.3") is True
    assert _no_proxy_entry_matches("10.0.0.0/8", "11.1.2.3") is False
    assert _no_proxy_entry_matches("192.168.1.5", "192.168.1.5") is True
    assert _no_proxy_entry_matches("192.168.1.5", "192.168.1.6") is False


def test_no_proxy_entry_matches_port_and_wildcard():
    assert _no_proxy_entry_matches("example.com:443", "example.com", 443) is True
    assert _no_proxy_entry_matches("example.com:443", "example.com", 80) is False
    assert _no_proxy_entry_matches("example.com:443", "example.com") is False
    assert _no_proxy_entry_matches("*", "anything.example") is True
    assert _no_proxy_entry_matches("", "anything.example") is False


def test_is_host_excluded_by_no_proxy():
    assert is_host_excluded_by_no_proxy("api.example.com", "example.com") is True
    assert is_host_excluded_by_no_proxy("example.com", "*.example.com") is True
    assert is_host_excluded_by_no_proxy("sub.example.com", "*.example.com") is True
    assert is_host_excluded_by_no_proxy("other.net", "*.example.com") is False
    assert is_host_excluded_by_no_proxy("anything.io", "*") is True
    assert is_host_excluded_by_no_proxy("api.example.com", " example.com , other.net ") is True


def test_is_host_excluded_by_no_proxy_uses_env_when_none(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "example.com")
    assert is_host_excluded_by_no_proxy("api.example.com") is True


def test_split_host_port():
    assert _split_host_port("") == ("", None)
    assert _split_host_port("http://Example.COM:8080/path") == ("example.com", 8080)
    assert _split_host_port("[::1]:8443") == ("::1", 8443)
    assert _split_host_port("host.example:9090") == ("host.example", 9090)
    assert _split_host_port("bare.example") == ("bare.example", None)


def test_resolve_proxy_url_env_priority(monkeypatch):
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.local:3128")
    monkeypatch.setenv("DISCORD_PROXY", "http://discord-proxy.local:3129")
    assert resolve_proxy_url() == "http://proxy.local:3128"
    assert resolve_proxy_url("DISCORD_PROXY") == "http://discord-proxy.local:3129"


def test_resolve_proxy_url_socks_alias(monkeypatch):
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setenv("ALL_PROXY", "socks://127.0.0.1:1080")
    assert resolve_proxy_url() == "socks5://127.0.0.1:1080"


def test_resolve_proxy_url_bypass(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.local:3128")
    monkeypatch.setenv("NO_PROXY", "api.example.com")
    assert resolve_proxy_url(target_hosts="api.example.com") is None


def test_resolve_proxy_url_none(monkeypatch):
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    assert resolve_proxy_url() is None


def test_proxy_kwargs_for_bot_none_and_http():
    assert proxy_kwargs_for_bot(None) == {}
    assert proxy_kwargs_for_bot("http://proxy.local:3128") == {"proxy": "http://proxy.local:3128"}


def test_proxy_kwargs_for_aiohttp_none_and_http():
    assert proxy_kwargs_for_aiohttp(None) == ({}, {})
    sess, req = proxy_kwargs_for_aiohttp("http://proxy.local:3128")
    assert sess == {}
    assert req == {"proxy": "http://proxy.local:3128"}


@pytest.mark.skipif(sys.platform == "darwin", reason="scutil exists on macOS")
def test_detect_macos_system_proxy_non_darwin():
    assert _detect_macos_system_proxy() is None


# ─── c1 routing metadata ─────────────────────────────────────────────────────

def _source(**kwargs):
    defaults = dict(
        thread_id=None,
        platform=None,
        scope_id=None,
        chat_type=None,
        message_id=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_thread_metadata_for_source_plain_thread_id():
    src = _source(thread_id="42")
    assert _thread_metadata_for_source(src) == {"thread_id": "42"}


def test_thread_metadata_for_source_none():
    assert _thread_metadata_for_source(_source()) is None


def test_thread_metadata_for_source_slack_team_id():
    src = _source(thread_id="1", platform="slack", scope_id="T123")
    meta = _thread_metadata_for_source(src)
    assert meta == {"thread_id": "1", "slack_team_id": "T123"}


def test_thread_metadata_for_source_telegram_dm():
    src = _source(thread_id="42", platform="telegram", chat_type="dm", message_id="7")
    meta = _thread_metadata_for_source(src)
    assert meta["telegram_dm_topic_reply_fallback"] is True
    assert meta["direct_messages_topic_id"] == "42"
    assert meta["telegram_reply_to_message_id"] == "7"
    assert meta["thread_id"] == "42"


def test_thread_metadata_for_source_telegram_dm_explicit_anchor():
    src = _source(thread_id="42", platform="telegram", chat_type="dm")
    meta = _thread_metadata_for_source(src, reply_to_message_id="99")
    assert meta["telegram_reply_to_message_id"] == "99"


def test_mark_notify_metadata_clones():
    original = {"thread_id": "1"}
    marked = _mark_notify_metadata(original)
    assert marked == {"thread_id": "1", "notify": True}
    assert "notify" not in original
    assert _mark_notify_metadata(None) == {"notify": True}


def test_reply_anchor_default_and_telegram_dm():
    event = SimpleNamespace(
        source=_source(platform="telegram", chat_type="dm", thread_id="42"),
        raw_message=None,
        message_id="m1",
        reply_to_message_id=None,
    )
    assert _reply_anchor_for_event(event) == "m1"


def test_reply_anchor_telegram_group_is_none():
    event = SimpleNamespace(
        source=_source(platform="telegram", chat_type="group", thread_id="42"),
        raw_message=None,
        message_id="m1",
        reply_to_message_id=None,
    )
    assert _reply_anchor_for_event(event) is None


def test_reply_anchor_slack_synthetic_is_none():
    event = SimpleNamespace(
        source=_source(platform="slack", thread_id=None),
        raw_message={"_hermes_no_thread_response": True},
        message_id="m1",
        reply_to_message_id=None,
    )
    assert _reply_anchor_for_event(event) is None


def test_reply_anchor_feishu_thread():
    event = SimpleNamespace(
        source=_source(platform="feishu", thread_id="42"),
        raw_message=None,
        message_id="m1",
        reply_to_message_id="r9",
    )
    assert _reply_anchor_for_event(event) == "r9"
