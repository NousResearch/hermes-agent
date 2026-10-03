"""Weixin ``send_voice``/``send_video`` must accept the router's ``is_voice`` kwarg.

The dispatcher (``run_turn.py`` / ``run_notifications.py``) passes ``is_voice=`` to
platform adapters. Adapters that don't support native voice/video bubbles accept the
kwarg and ignore it (PR #100021 for mattermost+line). The weixin adapter was the
last hold-out: the missing ``**kwargs`` caused a ``TypeError`` that was silently
swallowed by the dispatcher, dropping the audio attachment (#101380).
"""
import pytest

from gateway.config import PlatformConfig
from gateway.platforms.weixin import WeixinAdapter


def _make_adapter() -> WeixinAdapter:
    return WeixinAdapter(PlatformConfig(enabled=True, token="test-token"))


def test_send_voice_accepts_is_voice_kwarg():
    adapter = _make_adapter()
    import inspect
    sig = inspect.signature(adapter.send_voice)
    assert "kwargs" in str(sig), (
        "send_voice must accept **kwargs so the dispatcher's is_voice kwarg "
        "does not cause a TypeError (the router passes is_voice to all adapters)"
    )


def test_send_video_accepts_is_voice_kwarg():
    adapter = _make_adapter()
    import inspect
    sig = inspect.signature(adapter.send_video)
    assert "kwargs" in str(sig), (
        "send_video must accept **kwargs for the same reason as send_voice"
    )


def test_send_voice_signature_compatible_with_base_class():
    """Base class send_voice has **kwargs; weixin must be compatible."""
    import inspect
    from gateway.platforms.base import BasePlatformAdapter
    base_sig = inspect.signature(BasePlatformAdapter.send_voice)
    wx_sig = inspect.signature(WeixinAdapter.send_voice)
    base_params = set(base_sig.parameters) - {"self"}
    wx_params = set(wx_sig.parameters) - {"self"}
    missing = base_params - wx_params
    assert not missing, f"weixin send_voice missing params: {missing}"
    assert "kwargs" in wx_params, "weixin send_voice must have **kwargs"
