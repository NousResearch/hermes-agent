"""Invariant tests: MatrixAdapter.send_voice accepts the shared is_voice kwarg.

Regression for #102221 — upstream ccc367dce0 made the shared caller pass
``is_voice=`` unconditionally (gateway/platforms/base.py), but the Matrix
override took neither the kwarg nor ``**kwargs``, so every Matrix voice send
raised TypeError that the gateway swallowed as "Error sending media".
"""
import inspect

import pytest

from plugins.platforms.matrix.adapter import MatrixAdapter


def test_send_voice_signature_accepts_is_voice():
    """The override must accept the kwarg the shared caller passes — same contract
    as BasePlatformAdapter.send_voice. Proven red on pre-fix code: TypeError."""
    params = inspect.signature(MatrixAdapter.send_voice).parameters
    assert "is_voice" in params, (
        "MatrixAdapter.send_voice dropped the is_voice kwarg; the shared caller "
        "passes it unconditionally and every voice send will TypeError (#102221)"
    )
    assert params["is_voice"].default is True


@pytest.mark.asyncio
async def test_send_voice_forwards_is_voice_false_as_plain_audio(monkeypatch, tmp_path):
    """is_voice=False must reach _send_local_file so plain m.audio (no MSC3245
    voice metadata) is sent, matching base-class semantics."""
    audio = tmp_path / "clip.ogg"
    audio.write_bytes(b"OggS-fake")
    seen = {}

    async def fake_send_local_file(self, chat_id, path, msgtype, caption=None, reply_to=None,
                                   file_name=None, metadata=None, is_voice=True, **kwargs):
        seen.update(path=path, is_voice=is_voice, metadata=metadata)
        from gateway.platforms.base import SendResult
        return SendResult(success=True, message_id="$fake")

    monkeypatch.setattr(MatrixAdapter, "_send_local_file", fake_send_local_file)
    adapter = object.__new__(MatrixAdapter)  # name is a read-only property; not needed here
    await adapter.send_voice("!room:example.org", str(audio), caption=None,
                             reply_to=None, metadata={"k": "v"}, is_voice=False)
    assert seen == {"path": str(audio), "is_voice": False, "metadata": {"k": "v"}}
