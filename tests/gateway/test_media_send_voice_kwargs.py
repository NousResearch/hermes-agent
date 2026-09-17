"""The shared media dispatcher passes ``is_voice`` to every adapter's ``send_voice``.

Adapter overrides that narrow the base signature without ``**kwargs`` raise
TypeError before any upload happens, so MEDIA: audio replies are silently
dropped on those platforms (run_turn.py wraps its call site in
``suppress(Exception)``). The invariant: dispatching media with
``is_voice=True`` reaches the adapter's send path regardless of how the
override narrows its signature.
"""
import inspect

import pytest

from gateway.platforms.base import BasePlatformAdapter

# Adapters whose send_voice override was missing **kwargs (regression guards).
_VOICE_OVERRIDES = [
    ("plugins.platforms.line.adapter", "LineAdapter"),
    ("plugins.platforms.matrix.adapter", "MatrixAdapter"),
    ("plugins.platforms.mattermost.adapter", "MattermostAdapter"),
]


def _adapter_class(module_name: str, class_name: str):
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


@pytest.mark.parametrize("module_name,class_name", _VOICE_OVERRIDES)
def test_send_voice_override_accepts_is_voice_kwarg(module_name: str, class_name: str):
    """send_voice overrides must tolerate the dispatcher's is_voice kwarg."""
    cls = _adapter_class(module_name, class_name)
    sig = inspect.signature(cls.send_voice)
    assert sig.parameters.get("kwargs").kind == inspect.Parameter.VAR_KEYWORD, (
        f"{class_name}.send_voice does not accept **kwargs; the shared media "
        "dispatcher passes is_voice= and the call would raise TypeError before upload"
    )


def test_dispatch_send_voice_with_is_voice_reaches_base_sender():
    """The base adapter's send_voice/_send_media_file path tolerates is_voice end-to-end.

    Asserts the dispatcher's exact call shape binds against the BASE class
    signature — the contract every override must preserve.
    """
    sig = inspect.signature(BasePlatformAdapter.send_voice)
    sig.bind(None, "chat", "/tmp/a.wav", caption=None, reply_to=None, metadata=None, is_voice=True)
