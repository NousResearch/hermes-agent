"""The api_server hint's file-delivery halves: the base truth and the enabled correction.

By default the api_server hint says non-image files are NOT intercepted anywhere. With
``gateway.api_server.file_delivery.enabled``, ``system_prompt._default_platform_hint`` appends
``API_SERVER_FILE_DELIVERY_HINT``, a correction that supersedes the document half — so a default
install's system prompt stays byte-identical (prompt caching) while an enabled one carries the
whole delivery contract: trigger, placement, deliverable types + cap, one-shot disclosure, and
honest degradation.
"""

import pytest

from agent import system_prompt
from agent.prompt_builder import API_SERVER_FILE_DELIVERY_HINT, PLATFORM_HINTS


def test_api_server_hint_scopes_media_tag_guidance():
    """api_server MEDIA: interception is partial (#68402, corrected):
    _resolve_media_to_data_urls (gateway/platforms/api_server.py) inlines
    small image MEDIA: tags as base64 data URLs on the chat, completions,
    and responses endpoints — but non-image files are never resolved
    (_MEDIA_IMG_EXT is image-only) and the /v1/runs handler never calls
    the resolver at all. The hint must teach BOTH halves: images work via
    MEDIA:, everything else needs a plain path in the response text.

    The base hint is the flag-off truth. With
    ``gateway.api_server.file_delivery.enabled`` the document half is
    superseded by the appended correction (``API_SERVER_FILE_DELIVERY_HINT``,
    see ``system_prompt._default_platform_hint``) — the runs gap stays true
    either way, and a default install's prompt is byte-identical."""
    hint = PLATFORM_HINTS["api_server"]
    # Images ARE intercepted: inlined as data URLs.
    assert "MEDIA:" in hint
    assert "inlined" in hint.lower()
    assert "data" in hint.lower()  # data URLs
    # The gaps: non-image files and the runs endpoint.
    assert "non-image" in hint.lower()
    assert "runs" in hint.lower()
    # Fallback guidance: plain file path in the response text.
    assert "plain" in hint.lower()


def test_api_server_file_delivery_hint_carries_the_whole_contract():
    """The flag-on correction must carry everything honest behaviour needs:
    the trigger (user asked for a file, or one is the natural deliverable),
    where to write it (workspace/tmp; credential+system paths refused),
    which types and cap deliver, that the link is ONE-SHOT (disclosed to
    the user), and the honest degradation (undeliverable → tag stays
    literal → state the plain path, never promise a link)."""
    hint = API_SERVER_FILE_DELIVERY_HINT.lower()
    # Supersedes the base paragraph's "non-image files are NOT intercepted".
    assert "no longer" in hint
    # Trigger and placement.
    assert "asks for a file" in hint
    assert "workspace" in hint
    assert "refused" in hint
    # The advertised vocabulary is the transport's vocabulary, exactly:
    # every type the second pass can produce is named (and nothing outside
    # it is implied — a type added to DELIVERY_DOCUMENT_EXT_MIME must be
    # added here, or the model never uses the tag for it).
    from gateway.platforms.api_server_file_delivery import DELIVERY_DOCUMENT_EXT_MIME
    for ext in DELIVERY_DOCUMENT_EXT_MIME:
        assert ext in hint
    assert "10 mb" in hint  # the byte cap, named
    # One-shot link, and the model says so.
    assert "once" in hint
    # Honest degradation.
    assert "literal" in hint
    assert "plain path" in hint
    assert "never promise a link" in hint
    # The /v1/runs gap is true with the flag on or off.
    assert "runs" in hint


def test_api_server_hint_is_unchanged_without_file_delivery(monkeypatch):
    """Default install: the appended correction must not appear (cache-stable prompt)."""
    monkeypatch.setattr(system_prompt, "_api_server_file_delivery_enabled", lambda: False)
    assert system_prompt._default_platform_hint("api_server") == PLATFORM_HINTS["api_server"]
    monkeypatch.setattr(system_prompt, "_api_server_file_delivery_enabled", lambda: True)
    enabled = system_prompt._default_platform_hint("api_server")
    assert enabled.startswith(PLATFORM_HINTS["api_server"])
    assert enabled.endswith(API_SERVER_FILE_DELIVERY_HINT)
