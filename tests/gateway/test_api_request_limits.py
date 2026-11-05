"""Request normalization reads supported facade limits when each request runs."""

import pytest


def test_text_limit_override_caps_string_and_parts(monkeypatch):
    pytest.importorskip("aiohttp")
    from gateway.platforms import api_server
    monkeypatch.setattr(api_server, "MAX_NORMALIZED_TEXT_LENGTH", 3)
    assert api_server._normalize_chat_content("abcdef") == "abc"
    assert len(api_server._normalize_chat_content(["abcdef", "gh"])) <= 3


def test_list_limit_override_caps_normalized_parts(monkeypatch):
    pytest.importorskip("aiohttp")
    from gateway.platforms import api_server
    monkeypatch.setattr(api_server, "MAX_CONTENT_LIST_SIZE", 1)
    assert api_server._normalize_chat_content(["first", "second"]) == "first"
