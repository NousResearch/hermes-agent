"""#102221: every adapter send_voice override must accept the kwargs the
shared media dispatch passes (chat_id, audio_path, metadata, is_voice).

Matrix (and same-shaped Mattermost) overrides lacked `is_voice`, so every
voice send raised TypeError and was swallowed as 'Error sending media' —
total loss of voice delivery on those platforms.
"""
import inspect

import pytest

from gateway.platforms.base import BasePlatformAdapter
from plugins.platforms.matrix.adapter import MatrixAdapter
from plugins.platforms.mattermost.adapter import MattermostAdapter

# Exactly what gateway/platforms/base.py passes in the send_voice branch.
_DISPATCH_KWARGS = {
    "chat_id": "chat-1",
    "audio_path": "note.ogg",
    "metadata": {},
    "is_voice": True,
}


def _bind(cls):
    sig = inspect.signature(cls.send_voice)
    bound = sig.bind(object(), **_DISPATCH_KWARGS)
    bound.apply_defaults()
    return bound


@pytest.mark.parametrize(
    "cls",
    [BasePlatformAdapter, MatrixAdapter, MattermostAdapter],
    ids=["base", "matrix", "mattermost"],
)
def test_send_voice_accepts_dispatch_kwargs(cls):
    # Binding the exact dispatch kwargs must not raise TypeError (that is
    # the reported crash). An explicit is_voice parameter must receive the
    # value; a **kwargs catch-all must capture it.
    bound = _bind(cls)
    if "is_voice" in bound.arguments:
        assert bound.arguments["is_voice"] is True
    else:
        assert bound.arguments.get("kwargs", {}).get("is_voice") is True
