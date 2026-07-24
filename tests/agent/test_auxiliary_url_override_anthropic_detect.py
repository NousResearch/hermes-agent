"""Regression: auxiliary client Anthropic-detect must respect user base_url override.

The bug: ``resolve_provider_client`` built the OpenAI client with the user's
``explicit_base_url`` (e.g. ``https://api.minimax.chat/v1``) but then passed
the *raw* registry URL (e.g. ``https://api.minimax.io/anthropic``) to
``_wrap_if_needed``. ``_endpoint_speaks_anthropic_messages`` saw the
``/anthropic`` suffix and wrapped the client in ``AnthropicAuxiliaryClient``,
which then hit a real Anthropic endpoint with the user's key and 401'd.

The fix: pass the post-override ``base_url`` to ``_wrap_if_needed`` so the
Anthropic heuristic sees what the user actually configured.

If you re-introduce this regression, a user who routes the ``minimax``
provider through their own OpenAI-compatible gateway will get
``HTTP 401: invalid api key`` from the auxiliary tasks.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agent.auxiliary_client import resolve_provider_client


def test_user_base_url_override_skips_anthropic_wrap():
    """A user-supplied OpenAI-compatible base_url must NOT trigger the
    Anthropic wire wrap, even when the provider registry's raw URL is the
    /anthropic endpoint (minimax, minimax-cn, etc.)."""
    client, resolved = resolve_provider_client(
        provider="minimax",
        model="MiniMax-M3",
        explicit_base_url="https://api.minimax.chat/v1",
        explicit_api_key="dummy-key-for-shape-test",
        api_mode=None,  # the original buggy path; # type: ignore[arg-type]
    )
    assert client is not None
    assert resolved == "MiniMax-M3"
    assert type(client).__name__ == "OpenAI", (
        f"Expected plain OpenAI client (user base_url is OpenAI-compatible), "
        f"got {type(client).__name__} — the Anthropic detect heuristic "
        f"leaked the registry's hardcoded /anthropic URL into the wrap "
        f"decision (regression of url-override-anthropic-detect bug)."
    )
    # The actual base_url on the constructed client should be the user's URL,
    # not the registry's hardcoded /anthropic URL.
    actual = str(getattr(client, "base_url", "") or "")
    assert actual.startswith("https://api.minimax.chat"), (
        f"client.base_url leaked: got {actual!r}, expected to start with "
        f"'https://api.minimax.chat'"
    )


def test_user_explicit_anthropic_url_still_wraps_with_explicit_mode():
    """If the user genuinely wants the Anthropic wire transport, they
    declare it with ``api_mode='anthropic_messages'`` and the wrap MUST
    happen. The fix should not break this path."""
    client, resolved = resolve_provider_client(
        provider="minimax",
        model="MiniMax-M3",
        explicit_base_url="https://api.minimax.io/anthropic",
        explicit_api_key="dummy-key-for-shape-test",
        api_mode="anthropic_messages",
    )
    assert client is not None
    # Should be wrapped in AnthropicAuxiliaryClient (the wire adapter)
    assert type(client).__name__ in ("AnthropicAuxiliaryClient",), (
        f"Expected AnthropicAuxiliaryClient for api_mode=anthropic_messages, "
        f"got {type(client).__name__}"
    )


def test_explicit_anthropic_url_without_mode_uses_openai_shape():
    """A URL ending in ``/anthropic`` gets normalized to ``/v1`` by
    ``_to_openai_base_url`` (the OpenAI SDK needs ``/v1``). With no
    ``api_mode`` override, the client is plain OpenAI — to actually
    use the Anthropic wire, the user must set ``api_mode=anthropic_messages``.
    This documents the contract so a future refactor doesn't silently
    re-introduce the wrapping heuristic on /anthropic URLs."""
    client, resolved = resolve_provider_client(
        provider="minimax",
        model="MiniMax-M3",
        explicit_base_url="https://api.minimax.io/anthropic",
        explicit_api_key="dummy-key-for-shape-test",
        api_mode=None,  # type: ignore[arg-type]
    )
    assert client is not None
    assert type(client).__name__ == "OpenAI"
    actual = str(getattr(client, "base_url", "") or "")
    assert actual.endswith("/v1/") or actual.endswith("/v1"), (
        f"_to_openai_base_url should have normalised /anthropic -> /v1, "
        f"got {actual!r}"
    )
