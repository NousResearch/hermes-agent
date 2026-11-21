"""Regression for #156745: a lone surrogate must not break session-id derivation."""

import hashlib

from gateway.platforms.api_server import _derive_chat_session_id


def test_lone_surrogate_still_derives_a_stable_id():
    """A lone UTF-16 surrogate must not raise while deriving the id (#156745).

    surrogatepass keeps the fingerprint deterministic. A real emoji hashes
    the same as strict UTF-8, so non-BMP conversations do not fork.
    """
    high = "hello \ud83d"
    low = "hello \udce7"
    from_user = _derive_chat_session_id("sys", high)
    from_system = _derive_chat_session_id(low, "hello")
    assert from_user.startswith("api-")
    assert from_system.startswith("api-")
    assert from_user == _derive_chat_session_id("sys", high)
    assert from_system == _derive_chat_session_id(low, "hello")
    emoji = "hello \U0001f600"
    seed = f"sys\n{emoji}"
    expected = "api-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    assert _derive_chat_session_id("sys", emoji) == expected
    research = _derive_chat_session_id("sys", high, "research")
    assert research != from_user
    assert research == _derive_chat_session_id("sys", high, "research")
