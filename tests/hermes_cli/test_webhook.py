from __future__ import annotations

import argparse
from unittest.mock import patch

import hermes_cli.webhook as webhook


class _Response:
    status = 202

    def __init__(self, body: bytes, reads: list[int]):
        self._body = body
        self._reads = reads

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, size=-1):
        self._reads.append(size)
        return self._body[:size] if size >= 0 else self._body


def _args(**overrides):
    values = {"name": "build", "payload": '{"ok":true}'}
    values.update(overrides)
    return argparse.Namespace(**values)


def test_webhook_test_reads_bounded_response(monkeypatch, capsys):
    monkeypatch.setattr(webhook, "_WEBHOOK_TEST_RESPONSE_BODY_MAX_BYTES", 8)
    monkeypatch.setattr(
        webhook,
        "_load_subscriptions",
        lambda: {"build": {"secret": "secret"}},
    )
    monkeypatch.setattr(webhook, "_get_webhook_base_url", lambda: "http://127.0.0.1:8644")
    reads: list[int] = []

    def fake_urlopen(req, timeout=0):
        assert req.full_url == "http://127.0.0.1:8644/webhooks/build"
        assert timeout == 10
        return _Response(b"accepted", reads)

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        webhook._cmd_test(_args())

    assert reads == [9]
    out = capsys.readouterr().out
    assert "Response (202): accepted" in out
    assert "truncated" not in out


def test_webhook_test_truncates_oversized_response(monkeypatch, capsys):
    monkeypatch.setattr(webhook, "_WEBHOOK_TEST_RESPONSE_BODY_MAX_BYTES", 8)
    monkeypatch.setattr(
        webhook,
        "_load_subscriptions",
        lambda: {"build": {"secret": "secret"}},
    )
    monkeypatch.setattr(webhook, "_get_webhook_base_url", lambda: "http://127.0.0.1:8644")
    reads: list[int] = []

    with patch(
        "urllib.request.urlopen",
        return_value=_Response(b"accepted-extra", reads),
    ):
        webhook._cmd_test(_args())

    assert reads == [9]
    out = capsys.readouterr().out
    assert "Response (202) (truncated after 8 bytes): accepted" in out
    assert "extra" not in out
