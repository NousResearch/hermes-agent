# -*- coding: utf-8 -*-
"""Tests for gateway.platforms.qqbot.api_client — REST API client."""

from __future__ import annotations

import pytest

from gateway.platforms.qqbot.core.api_client import QQApiClient


# ── build_text_body ───────────────────────────────────────────────────

class TestBuildTextBody:
    def test_markdown_mode(self):
        msg = QQApiClient.build_text_body("**bold**", markdown=True)
        assert msg.msg_type == 2
        assert msg.markdown is not None
        assert msg.markdown.content == "**bold**"

    def test_plain_text_mode(self):
        msg = QQApiClient.build_text_body("hello", markdown=False)
        assert msg.msg_type == 0
        assert msg.content == "hello"

    def test_truncation(self):
        msg = QQApiClient.build_text_body("x" * 10000, markdown=False, max_length=100)
        assert len(msg.content) == 100

    def test_reply_to_adds_reference(self):
        msg = QQApiClient.build_text_body("hi", reply_to="msg-1", markdown=False)
        assert msg.message_reference is not None
        assert msg.message_reference.message_id == "msg-1"

    def test_reply_to_no_reference_in_markdown(self):
        msg = QQApiClient.build_text_body("hi", reply_to="msg-1", markdown=True)
        assert msg.message_reference is None

    def test_msg_seq_present(self):
        msg = QQApiClient.build_text_body("hi", markdown=True)
        assert isinstance(msg.msg_seq, int)
        assert msg.msg_seq > 0

    def test_to_dict_roundtrip(self):
        """DTO should serialize correctly to dict for API submission."""
        msg = QQApiClient.build_text_body("hello", reply_to="r-1", markdown=False)
        d = msg.to_dict()
        assert d["msg_type"] == 0
        assert d["content"] == "hello"
        assert d["msg_id"] == "r-1"
        assert d["message_reference"]["message_id"] == "r-1"

    def test_markdown_to_dict(self):
        msg = QQApiClient.build_text_body("**bold**", markdown=True)
        d = msg.to_dict()
        assert d["msg_type"] == 2
        assert d["markdown"]["content"] == "**bold**"
        assert "message_reference" not in d


# ── next_msg_seq ──────────────────────────────────────────────────────

class TestNextMsgSeq:
    def test_range(self):
        for __ in range(100):
            seq = QQApiClient.next_msg_seq("test")
            assert 0 <= seq <= 65535

    def test_varies(self):
        seqs = {QQApiClient.next_msg_seq("test") for __ in range(20)}
        assert len(seqs) > 1


# ── clear_token ───────────────────────────────────────────────────────

class TestClearToken:
    def test_clears(self):
        client = QQApiClient("app", "secret")
        client._access_token = "tok"
        client._token_expires_at = 9999.0
        client.clear_token()
        assert client._access_token is None
        assert client._token_expires_at == 0.0


# ── media_headers ─────────────────────────────────────────────────────

class TestMediaHeaders:
    def test_with_token(self):
        client = QQApiClient("app", "secret")
        client._access_token = "tok-123"
        assert client.media_headers() == {"Authorization": "QQBot tok-123"}

    def test_without_token(self):
        client = QQApiClient("app", "secret")
        assert client.media_headers() == {}


# ── URI constants ─────────────────────────────────────────────────────

class TestURIConstants:
    def test_c2c_messages(self):
        from gateway.platforms.qqbot.core.api_client import _C2C_MESSAGES_URI as C2C_MESSAGES_URI
        assert "{user_id}" in C2C_MESSAGES_URI

    def test_group_messages(self):
        from gateway.platforms.qqbot.core.api_client import _GROUP_MESSAGES_URI as GROUP_MESSAGES_URI
        assert "{group_id}" in GROUP_MESSAGES_URI

    def test_guild_messages(self):
        from gateway.platforms.qqbot.core.api_client import _GUILD_MESSAGES_URI as GUILD_MESSAGES_URI
        assert "{channel_id}" in GUILD_MESSAGES_URI
