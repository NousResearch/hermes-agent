"""Tests for Discord mention-gating free-response policy."""

from gateway.platforms.discord import _is_free_response_channel


class DummyChannel:
    def __init__(self, channel_id, parent_id=None, parent=None):
        self.id = channel_id
        self.parent_id = parent_id
        self.parent = parent


def test_free_response_matches_channel_id_directly():
    channel = DummyChannel(channel_id="123")
    assert _is_free_response_channel(channel, {"123"}) is True


def test_thread_inherits_free_response_from_parent_id():
    thread = DummyChannel(channel_id="999", parent_id="123")
    assert _is_free_response_channel(thread, {"123"}) is True


def test_thread_inherits_free_response_from_parent_object():
    parent = DummyChannel(channel_id="123")
    thread = DummyChannel(channel_id="999", parent=parent)
    assert _is_free_response_channel(thread, {"123"}) is True


def test_non_free_channel_returns_false():
    channel = DummyChannel(channel_id="999", parent_id="888")
    assert _is_free_response_channel(channel, {"123"}) is False
