"""Regression tests: send_voice must accept the gateway router's is_voice kwarg.

The gateway media dispatch loop (gateway/platforms/base.py) calls
``self.send_voice(chat_id=..., audio_path=..., metadata=..., is_voice=...)``
whenever ``should_send_media_as_audio`` routes an audio MEDIA: attachment to
the voice sender. Adapters whose ``send_voice`` lacks ``is_voice`` (and any
``**kwargs``) raise ``TypeError`` at argument-binding time, and the dispatch
loop swallows it, so the attachment silently never arrives.

Matrix was fixed for this in #99712; these tests pin the same contract for
the Mattermost and LINE adapters.
"""

from __future__ import annotations

import inspect

import pytest

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

mattermost_adapter = load_plugin_adapter("mattermost")
line_adapter = load_plugin_adapter("line")


def _send_voice_params(module, class_name):
    adapter_cls = getattr(module, class_name)
    return inspect.signature(adapter_cls.send_voice).parameters


class TestMattermostSendVoiceSignature:
    def test_accepts_is_voice_kwarg(self):
        params = _send_voice_params(mattermost_adapter, "MattermostAdapter")
        assert "is_voice" in params, (
            "gateway media router calls send_voice(is_voice=...); "
            "MattermostAdapter must accept it"
        )

    @pytest.mark.asyncio
    async def test_router_kwarg_set_binds(self):
        """The exact kwarg set from the base.py dispatch loop must bind."""
        adapter_cls = mattermost_adapter.MattermostAdapter
        instance = adapter_cls.__new__(adapter_cls)

        # No connection/file: the adapter either returns a controlled SendResult
        # or skips the missing local file — anything except a TypeError from
        # argument binding.
        adapter_cls = mattermost_adapter.MattermostAdapter
        instance = adapter_cls.__new__(adapter_cls)
        try:
            await adapter_cls.send_voice(
                instance,
                chat_id="C1",
                audio_path="/tmp/does-not-exist.ogg",
                metadata=None,
                is_voice=True,
            )
        except TypeError as e:
            pytest.fail(f"send_voice rejected the router's is_voice kwarg: {e}")


class TestLineSendVoiceSignature:
    def test_accepts_is_voice_kwarg(self):
        params = _send_voice_params(line_adapter, "LineAdapter")
        assert "is_voice" in params, (
            "gateway media router calls send_voice(is_voice=...); "
            "LineAdapter must accept it"
        )

    @pytest.mark.asyncio
    async def test_router_kwarg_set_binds(self):
        """The exact kwarg set from the base.py dispatch loop must bind."""
        adapter_cls = line_adapter.LineAdapter
        instance = adapter_cls.__new__(adapter_cls)

        result = await adapter_cls.send_voice(
            instance,
            chat_id="C1",
            audio_path="/tmp/does-not-exist.ogg",
            metadata=None,
            is_voice=True,
        )
        # Missing file returns a controlled failure, proving binding worked.
        assert result.success is False
        assert "not found" in result.error
