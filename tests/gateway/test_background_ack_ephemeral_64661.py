"""Regression tests for #64661 — /background and /btw acknowledgements trigger
duplicate file uploads when the prompt contains a bare local file path.

Background: ``_handle_background_command`` in gateway/slash_commands.py returns
a plain translated string containing the prompt preview. That string flows
through ``_process_message_background`` → ``extract_local_files`` → upload.
When the background task completes, the same file is uploaded again via
the task-completion path. The user receives two attachments for one task.

Fix: the slash-command handler should return ``EphemeralReply(...)`` instead
of a plain string. The ``is_ephemeral_response`` check at base.py:4608
already gates media extraction on this — we just need the slash command
to opt in.

This test asserts that the slash command's return value type is the
``EphemeralReply`` sentinel — not a plain string. A plain str return
falls through the gate and triggers the duplicate-upload bug.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.platforms.base import EphemeralReply


def test_ephemeral_reply_class_exists():
    """Sanity: the EphemeralReply sentinel class must exist on base.py."""
    from gateway.platforms.base import EphemeralReply

    assert isinstance(EphemeralReply("hello"), str)
    assert type(EphemeralReply("hello")) is EphemeralReply
    assert type(EphemeralReply("hello")) is not str


def test_is_ephemeral_response_check_uses_isinstance():
    """The gate at base.py:4608 uses isinstance(response, EphemeralReply).
    A plain str returns False (gate fires) and an EphemeralReply returns
    True (gate suppresses extraction).
    """
    from gateway.platforms.base import EphemeralReply

    plain = "Background task started: use C:\\temp\\hermes-test\\bg.png and return it"
    ephemeral = EphemeralReply(plain)
    assert not isinstance(plain, EphemeralReply), (
        "Plain str must NOT be an EphemeralReply (else the gate doesn't fire)"
    )
    assert isinstance(ephemeral, EphemeralReply), (
        "EphemeralReply instances must satisfy isinstance check"
    )


def test_handle_background_command_returns_ephemeral_reply():
    """The actual fix: _handle_background_command must return EphemeralReply.

    A plain str return is the bug — it falls through the media-extraction
    gate. After the fix, the return value should be an EphemeralReply so
    the gate fires.
    """
    from gateway.slash_commands import GatewaySlashCommandsMixin

    # Build a minimal event mock
    event = MagicMock()
    event.get_command_args.return_value = (
        "use C:\\temp\\hermes-test\\bg.png and return it as an image"
    )
    event.source = MagicMock()
    event.media_urls = []
    event.media_types = []

    handler = GatewaySlashCommandsMixin.__new__(GatewaySlashCommandsMixin)  # bypass __init__
    # Set up only the attrs the function touches
    handler._background_tasks = set()
    handler._reply_anchor_for_event = MagicMock(return_value=None)
    handler._run_background_task = AsyncMock()

    result = asyncio.run(handler._handle_background_command(event))

    assert isinstance(result, EphemeralReply), (
        f"#64661 regression: _handle_background_command returned {type(result).__name__}, "
        f"expected EphemeralReply. Plain str falls through the media-extraction gate "
        f"at base.py:4608, causing the duplicate-upload bug."
    )


def test_handle_btw_command_returns_ephemeral_reply():
    """The /btw command shares the same background handler. Same fix needed."""
    from gateway.slash_commands import GatewaySlashCommandsMixin

    event = MagicMock()
    event.get_command_args.return_value = (
        "use /home/alice/data.csv and return the chart"
    )
    event.source = MagicMock()
    event.media_urls = []
    event.media_types = []

    handler = GatewaySlashCommandsMixin.__new__(GatewaySlashCommandsMixin)
    handler._background_tasks = set()
    handler._reply_anchor_for_event = MagicMock(return_value=None)
    handler._run_background_task = AsyncMock()

    # /btw may route to _handle_background_command internally, or have
    # its own handler. Try the same handler first; skip if /btw path
    # routes through a different method.
    fn = getattr(handler, "_handle_btw_command", None) or getattr(
        handler, "_handle_background_command", None
    )
    if fn is None:
        pytest.skip("No /btw handler found")
    result = asyncio.run(fn(event))

    assert isinstance(result, EphemeralReply), (
        f"#64661 regression: /btw returned {type(result).__name__}, "
        f"expected EphemeralReply."
    )