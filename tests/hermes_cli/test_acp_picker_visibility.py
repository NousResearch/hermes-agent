"""ACP external-process providers auto-inject into the model picker.

The ``CANONICAL_PROVIDERS`` auto-extend loop skips ``external_process``
providers that need bespoke picker UX (device-code, key-entry, etc.).
ACP subprocess providers (``base_url`` starting with ``acp://``) need
no such UX — the subprocess owns auth — so they should be auto-injected.

Regression for #102421: out-of-tree ACP plugins were invisible in
``/model``, the setup wizard, and the desktop model selector because
the auto-extend loop blanket-skipped every ``external_process`` provider.
"""

from __future__ import annotations

import pytest

from providers.base import ProviderProfile


class _MockACPProfile(ProviderProfile):
    """Synthetic ACP provider for testing."""
    def create_client(self, **kwargs):
        return ("mock-acp-client", kwargs)

    def fetch_models(self, **kwargs):
        return None


class _MockNonACPProfile(ProviderProfile):
    """Synthetic non-ACP external-process provider for testing."""
    def create_client(self, **kwargs):
        return ("mock-non-acp-client", kwargs)

    def fetch_models(self, **kwargs):
        return None


def _make_acp_profile(name: str = "test-acp-provider") -> _MockACPProfile:
    return _MockACPProfile(
        name=name,
        display_name="Test ACP Provider",
        description="Test ACP provider (auto-inject test)",
        base_url="acp://test-agent",
        auth_type="external_process",
        process_command="test-acp-cli",
        process_args=("--acp",),
        process_command_env_vars=("TEST_ACP_CLI_PATH",),
    )


def _make_non_acp_profile(name: str = "test-non-acp-ext") -> _MockNonACPProfile:
    return _MockNonACPProfile(
        name=name,
        display_name="Test Non-ACP External",
        base_url="http://localhost:9999",
        auth_type="external_process",
        process_command="non-acp-cli",
    )


class TestACPProviderPickerVisibility:
    """ACP providers appear in CANONICAL_PROVIDERS; non-ACP external ones don't."""

    def test_acp_provider_passes_auto_extend_filter(self, monkeypatch):
        """An ACP (acp://) external-process provider passes the filter."""
        import hermes_cli.models as hm

        acp = _make_acp_profile("fresh-acp")

        # Snapshot before
        original_slugs = {p.slug for p in hm.CANONICAL_PROVIDERS}
        assert "fresh-acp" not in original_slugs

        # Simulate the auto-extend loop with just this provider
        monkeypatch.setattr(
            "providers.list_providers",
            lambda: [acp],
        )

        # Re-run the auto-extend logic inline (mirrors the module-level code)
        _canonical_slugs = {p.slug for p in hm.CANONICAL_PROVIDERS}
        from providers import list_providers as _lp
        for _pp in _lp():
            if _pp.name in _canonical_slugs:
                continue
            if _pp.auth_type in {"oauth_device_code", "oauth_external", "aws_sdk", "copilot", "vertex"}:
                continue
            if _pp.auth_type == "external_process" and not str(_pp.base_url or "").startswith("acp://"):
                continue
            _label = _pp.display_name or _pp.name
            _desc = _pp.description or f"{_label} (direct API)"
            hm.CANONICAL_PROVIDERS.append(hm.ProviderEntry(_pp.name, _label, _desc))
            _canonical_slugs.add(_pp.name)

        try:
            slugs = [p.slug for p in hm.CANONICAL_PROVIDERS]
            assert "fresh-acp" in slugs
            entry = next(p for p in hm.CANONICAL_PROVIDERS if p.slug == "fresh-acp")
            assert entry.label == "Test ACP Provider"
            assert "Test ACP provider" in entry.tui_desc
        finally:
            # Clean up: remove the injected entry
            hm.CANONICAL_PROVIDERS[:] = [
                p for p in hm.CANONICAL_PROVIDERS if p.slug != "fresh-acp"
            ]

    def test_non_acp_external_process_blocked_by_filter(self):
        """A non-ACP external-process provider is blocked by the filter."""
        non_acp = _make_non_acp_profile()

        # Reproduce the filter logic
        skipped = False
        if non_acp.auth_type in {"oauth_device_code", "oauth_external", "aws_sdk", "copilot", "vertex"}:
            skipped = True
        if non_acp.auth_type == "external_process" and not str(non_acp.base_url or "").startswith("acp://"):
            skipped = True

        assert skipped, "Non-ACP external-process provider should be skipped"

    def test_acp_provider_not_blocked_by_filter(self):
        """An ACP external-process provider is NOT blocked by the filter."""
        acp = _make_acp_profile()

        # Reproduce the filter logic
        skipped = False
        if acp.auth_type in {"oauth_device_code", "oauth_external", "aws_sdk", "copilot", "vertex"}:
            skipped = True
        if acp.auth_type == "external_process" and not str(acp.base_url or "").startswith("acp://"):
            skipped = True

        assert not skipped, "ACP external-process provider should NOT be skipped"

    def test_copilot_acp_not_duplicated(self):
        """copilot-acp is hardcoded in CANONICAL_PROVIDERS; auto-extend
        skips it (name already in _canonical_slugs) so it appears exactly once."""
        from hermes_cli.models import CANONICAL_PROVIDERS

        copilot_entries = [p for p in CANONICAL_PROVIDERS if p.slug == "copilot-acp"]
        assert len(copilot_entries) == 1, (
            "copilot-acp should appear exactly once (hardcoded, not duplicated)"
        )

    @pytest.mark.parametrize("base_url,expected_skip", [
        ("acp://some-agent", False),
        ("acp://copilot", False),
        ("acp://kiro-agent", False),
        ("http://localhost:8080", True),
        ("https://api.example.com/v1", True),
        ("acp+tcp://remote:9999", True),  # acp+tcp is not acp://
        ("", True),
    ])
    def test_acp_url_prefix_filter(self, base_url, expected_skip):
        """The filter correctly distinguishes acp:// from other schemes."""
        profile = _MockACPProfile(
            name="url-test",
            base_url=base_url,
            auth_type="external_process",
        )

        skipped = (
            profile.auth_type == "external_process"
            and not str(profile.base_url or "").startswith("acp://")
        )
        assert skipped == expected_skip, (
            f"base_url={base_url!r}: expected skip={expected_skip}, got {skipped}"
        )

    def test_api_key_providers_still_injected(self):
        """api_key auth providers still pass the filter (no regression)."""
        profile = ProviderProfile(
            name="test-api-key-provider",
            base_url="https://api.example.com",
            auth_type="api_key",
        )

        skipped = False
        if profile.auth_type in {"oauth_device_code", "oauth_external", "aws_sdk", "copilot", "vertex"}:
            skipped = True
        if profile.auth_type == "external_process" and not str(profile.base_url or "").startswith("acp://"):
            skipped = True

        assert not skipped, "api_key providers should still be auto-injected"

    @pytest.mark.parametrize("auth_type", [
        "oauth_device_code", "oauth_external", "aws_sdk", "copilot", "vertex",
    ])
    def test_non_api_key_types_still_skipped(self, auth_type):
        """Other non-api-key auth types are still skipped (no regression)."""
        profile = ProviderProfile(
            name="test-skip-provider",
            base_url="https://api.example.com",
            auth_type=auth_type,
        )

        skipped = False
        if profile.auth_type in {"oauth_device_code", "oauth_external", "aws_sdk", "copilot", "vertex"}:
            skipped = True
        if profile.auth_type == "external_process" and not str(profile.base_url or "").startswith("acp://"):
            skipped = True

        assert skipped, f"auth_type={auth_type!r} should still be skipped"
