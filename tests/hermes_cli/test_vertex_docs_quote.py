"""The Vertex guide quotes the credential error verbatim; the quote must not drift.

The docs promise the exact message Hermes emits. A rewritten error message silently
breaks that promise, so the quote is checked against the code that raises it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

GUIDE = Path(__file__).resolve().parents[2] / "website" / "docs" / "guides" / "google-vertex.md"


def _emitted_credential_error() -> str:
    """The exact AuthError text the Vertex resolution path raises."""
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.auth import AuthError

    import agent.vertex_adapter as va

    original = va.get_vertex_config
    va.get_vertex_config = lambda *a, **k: (None, None)
    try:
        with pytest.raises(AuthError) as exc:
            resolve_runtime_provider(requested="vertex")
    finally:
        va.get_vertex_config = original
    return str(exc.value)


def test_guide_quotes_the_credential_error_verbatim():
    """The troubleshooting section quotes the message the code actually raises."""
    message = _emitted_credential_error()
    guide = GUIDE.read_text(encoding="utf-8")
    assert message in guide, (
        "the Vertex guide no longer quotes the credential error verbatim; "
        f"the code now says: {message!r}")


def test_credential_error_names_both_accepted_file_kinds():
    """The message must not send a user with an ``authorized_user`` file back to the docs."""
    message = _emitted_credential_error()
    assert "service-account JSON" in message
    assert "authorized_user" in message, (
        "the error should name the gcloud authorized_user file, which the loader accepts")
