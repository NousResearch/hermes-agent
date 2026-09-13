"""Regression for #109756: video_analyze (and any other async auxiliary call) 401s on the
keyless OpenCode Free tier while vision_analyze appears to work, because the SYNC->ASYNC
client rebuild in ``_to_async_client`` rebuilds ``default_headers`` from scratch via
``_endpoint_default_headers(base_url, inferred_provider)`` and never re-applies the
keyless-blanking check that ``_create_openai_client`` applies on the sync path.

``_infer_provider_from_url("https://opencode.ai/zen/v1")`` resolves to ``"opencode-go"``
(a KEYED sibling family sharing the same host) rather than ``"opencode-free"``, so the
rebuilt async client's ``default_headers`` carry no ``Authorization`` override at all —
the OpenAI SDK's own ``auth_headers`` property then sends ``Bearer
opencode-zen-free-keyless`` on the wire, which the relay 401s (any recognized-looking
bearer is rejected on the free tier; only an explicitly blank ``Authorization`` succeeds).

This bug is NOT specific to video or vision: ``_to_async_client`` is the sync->async
seam for every async auxiliary call (``async_call_llm``, used by vision_analyze,
video_analyze, and any other task run with ``async_mode=True``). A synchronous
auxiliary call on the same config is fine, and a *first* async call happening to reuse a
still-keyless cached sync client can look fine too, which is why the reporter's own
comparison (vision "worked", video "didn't") pointed at the wrong divergence — the real
seam is sync-vs-async client construction, not vision-vs-video routing.
"""

from hermes_cli.models import OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER

import agent.auxiliary_client as ac


class _FakeSyncClient:
    """Minimal stand-in for the OpenAI sync client ``_to_async_client`` reads from."""

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url


class TestOpencodeFreeKeylessSurvivesAsyncRebuild:
    def test_async_rebuild_blanks_authorization_for_keyless_placeholder(self):
        """The exact #109756 regression: rebuilding a keyless opencode-free sync client into
        its async counterpart must still carry ``Authorization: ""`` in default_headers,
        the same override ``_create_openai_client`` applies on the sync path."""
        sync_client = _FakeSyncClient(
            api_key=OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER,
            base_url="https://opencode.ai/zen/v1",
        )

        async_client, _model = ac._to_async_client(sync_client, "muse-spark-1.3-contributor-free")

        assert async_client._custom_headers.get("Authorization") == "", (
            "async-rebuilt client must blank Authorization for the keyless free-tier "
            "placeholder, or the OpenAI SDK sends a live Bearer the relay 401s"
        )

    def test_async_rebuild_preserves_real_credentials_unaffected(self):
        """A normal (non-keyless) provider's async rebuild must NOT gain a spurious blank
        Authorization override — the fix must be scoped to the keyless placeholder only,
        so real API-key providers (and thus vision_analyze on them) are unaffected."""
        sync_client = _FakeSyncClient(
            api_key="sk-real-provider-key",
            base_url="https://api.some-real-provider.example/v1",
        )

        async_client, _model = ac._to_async_client(sync_client, "some-model")

        headers = async_client._custom_headers or {}
        assert headers.get("Authorization") != "", (
            "a real credential must not be blanked by the keyless override"
        )

    def test_helper_is_a_noop_for_non_keyless_api_keys(self):
        """Direct unit test of the shared helper: only the exact keyless placeholder value
        triggers the header override; everything else passes through unchanged."""
        original = {"X-Title": "Hermes Agent"}
        assert ac._opencode_zen_free_header_override("sk-real-key", original) == original
        assert ac._opencode_zen_free_header_override(None, original) == original

    def test_helper_blanks_authorization_for_keyless_placeholder(self):
        original = {"X-Title": "Hermes Agent", "Authorization": "Bearer stale"}
        merged = ac._opencode_zen_free_header_override(
            OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER, original
        )
        assert merged["Authorization"] == ""
        assert merged["X-Title"] == "Hermes Agent"
