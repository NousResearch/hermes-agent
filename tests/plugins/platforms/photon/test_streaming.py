"""Regression tests for Photon adapter streaming behavior."""
from plugins.platforms.photon.adapter import PhotonAdapter


def test_photon_adapter_does_not_support_message_editing() -> None:
    """PhotonAdapter.SUPPORTS_MESSAGE_EDITING must be False.

    Photon (iMessage) has no real edit API for already-sent messages.
    This attribute signals the gateway to suppress the streaming cursor
    instead of leaving a stale tofu square (▉) behind when edit attempts fail.
    """
    assert PhotonAdapter.SUPPORTS_MESSAGE_EDITING is False
