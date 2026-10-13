"""Gemini long decode: a dropped stream resumes from its last checkpoint token, never duplicating text."""

from __future__ import annotations

import json

import httpx
import pytest


def _scripted_stream_client(monkeypatch, streams):
    """A GeminiNativeClient whose Nth stream yields ``streams[N]``: event dicts, then httpx drops on ``"DROP"``."""
    from agent.gemini_native_adapter import GeminiNativeClient

    sent = []

    class _Stream:
        status_code = 200

        def __init__(self, script):
            self._script = script

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def iter_text(self):
            for item in self._script:
                if item == "DROP":
                    raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")
                yield f"data: {json.dumps(item)}\n\n"

    class DummyHTTP:
        def stream(self, method, url, json=None, headers=None, timeout=None):
            sent.append(json)
            return _Stream(streams[min(len(sent), len(streams)) - 1])

        def close(self):
            return None

    monkeypatch.setattr("agent.gemini_native_adapter.httpx.Client", lambda *a, **k: DummyHTTP())
    client = GeminiNativeClient(api_key="AQ.test", base_url="https://generativelanguage.googleapis.com/v1beta")
    kwargs = {"model": "gemini-4-argon", "messages": [{"role": "user", "content": "count"}], "stream": True}
    return client, kwargs, sent


def _ld_event(text, token=None, finish=None):
    cand = {"content": {"role": "model", "parts": [{"text": text}]}}
    return {"candidates": [{**cand, **({"continuationToken": token} if token else {}), **({"finishReason": finish} if finish else {})}]}


def test_dropped_long_decode_stream_resumes_from_its_last_checkpoint(monkeypatch):
    """Past 32K a stream checkpoints every ~5K tokens (the token rides the event whose text it commits).
    A drop after a checkpoint resends the SAME request plus that token: no restart, no duplicated text."""
    streams = [[_ld_event("1\n"), _ld_event("2\n", token="ckpt-2"), "DROP"], [_ld_event("3\n", finish="STOP")]]
    client, kwargs, sent = _scripted_stream_client(monkeypatch, streams)

    chunks = list(client.chat.completions.create(**kwargs))
    assert sent[1] == {**sent[0], "continuationToken": "ckpt-2"}
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "1\n2\n3\n"
    assert [c.choices[0].finish_reason for c in chunks if c.choices[0].finish_reason] == ["stop"]


@pytest.mark.parametrize("streams, requests", [
    ([[_ld_event("1\n"), "DROP"]], 1),  # before the first checkpoint: nothing to resume from
    ([[_ld_event("1\n", token="ckpt-1"), _ld_event("2\n"), "DROP"]], 1),  # "2" was shown but not committed
    ([[_ld_event("1\n", token="ckpt-1"), "DROP"], ["DROP"]], 4),  # bounded: one token rides at most 3 resends
])
def test_dropped_stream_without_a_clean_checkpoint_raises(monkeypatch, streams, requests):
    from agent.gemini_native_adapter import GeminiAPIError

    client, kwargs, sent = _scripted_stream_client(monkeypatch, streams)
    with pytest.raises(GeminiAPIError) as excinfo:
        list(client.chat.completions.create(**kwargs))
    assert excinfo.value.code == "gemini_stream_error"
    assert len(sent) == requests
