"""Regression tests for the Buzz requirement gate vs external secrets (#95216).

``check_requirements()`` runs at gateway startup BEFORE the per-profile
secret scope is installed, so a Bitwarden-managed ``BUZZ_PRIVATE_KEY`` (only
``BWS_ACCESS_TOKEN`` in ``.env``) was invisible to the bare env read and Buzz
was silently skipped. The fix adds a one-shot ``build_profile_secret_scope``
consultation to the unscoped fallback of ``_get_scoped_secret``.

The key values below are synthesized placeholders (never a usable secret).
"""

import os

import pytest

# Synthesized, non-credential placeholder (any non-empty string exercises
# the gate; no real key material is ever embedded here).
_STUB_KEY = os.environ.get("BUZZ_TEST_STUB_KEY") or ("k" * 8)


@pytest.fixture(autouse=True)
def _reset_unscoped_cache():
    import plugins.platforms.buzz.adapter as adapter

    prev = adapter._UNSCOPED_PROFILE_SECRETS
    adapter._UNSCOPED_PROFILE_SECRETS = None
    yield
    adapter._UNSCOPED_PROFILE_SECRETS = prev


def _install_fake_scope(monkeypatch, secrets):
    """Point build_profile_secret_scope at a fake external-secret snapshot."""
    import agent.secret_scope as secret_scope

    calls = []

    def fake_build(home):  # noqa: ANN001 - test double
        calls.append(home)
        return dict(secrets)

    monkeypatch.setattr(secret_scope, "build_profile_secret_scope", fake_build)
    return calls


class TestRequirementGateSeesExternalSecrets:
    def test_externally_managed_key_passes_gate(self, monkeypatch):
        import plugins.platforms.buzz.adapter as adapter

        monkeypatch.delenv("BUZZ_RELAY_URL", raising=False)
        monkeypatch.delenv("BUZZ_PRIVATE_KEY", raising=False)
        monkeypatch.setenv("BWS_ACCESS_TOKEN", "stub-token")
        _install_fake_scope(
            monkeypatch,
            {"BUZZ_RELAY_URL": "wss://relay.example", "BUZZ_PRIVATE_KEY": _STUB_KEY},
        )

        assert adapter.check_requirements() is True

    def test_gate_fails_cleanly_when_nothing_resolves(self, monkeypatch):
        import plugins.platforms.buzz.adapter as adapter

        monkeypatch.delenv("BUZZ_RELAY_URL", raising=False)
        monkeypatch.delenv("BUZZ_PRIVATE_KEY", raising=False)
        _install_fake_scope(monkeypatch, {})

        assert adapter.check_requirements() is False

    def test_relay_from_env_still_passes_with_external_key(self, monkeypatch):
        import plugins.platforms.buzz.adapter as adapter

        monkeypatch.setenv("BUZZ_RELAY_URL", "wss://relay.example")
        monkeypatch.delenv("BUZZ_PRIVATE_KEY", raising=False)
        _install_fake_scope(monkeypatch, {"BUZZ_PRIVATE_KEY": _STUB_KEY})

        assert adapter.check_requirements() is True

    def test_profile_scope_build_failure_degrades_to_not_configured(
        self, monkeypatch
    ):
        import agent.secret_scope as secret_scope
        import plugins.platforms.buzz.adapter as adapter

        monkeypatch.delenv("BUZZ_RELAY_URL", raising=False)
        monkeypatch.delenv("BUZZ_PRIVATE_KEY", raising=False)

        def boom(home):  # noqa: ANN001 - test double
            raise RuntimeError("external secret resolver unavailable")

        monkeypatch.setattr(secret_scope, "build_profile_secret_scope", boom)
        assert adapter.check_requirements() is False

    def test_scope_snapshot_is_built_once_and_cached(self, monkeypatch):
        import plugins.platforms.buzz.adapter as adapter

        monkeypatch.setenv("BUZZ_RELAY_URL", "wss://relay.example")
        monkeypatch.delenv("BUZZ_PRIVATE_KEY", raising=False)
        calls = _install_fake_scope(monkeypatch, {"BUZZ_PRIVATE_KEY": _STUB_KEY})

        assert adapter.check_requirements() is True
        assert adapter.check_requirements() is True
        assert adapter.validate_config(type("Cfg", (), {"extra": {}})()) is True
        assert len(calls) == 1, "the external-secret snapshot must be cached"


class TestScopedSemanticsUnchanged:
    def test_active_scope_miss_does_not_fall_through_to_unscoped_build(
        self, monkeypatch
    ):
        """A scoped miss must keep returning default: the unscoped build is
        only for the no-scope startup gate, never a cross-profile borrow."""
        import agent.secret_scope as secret_scope
        import plugins.platforms.buzz.adapter as adapter

        monkeypatch.setenv("BUZZ_RELAY_URL", "wss://relay.example")
        monkeypatch.delenv("BUZZ_PRIVATE_KEY", raising=False)
        calls = _install_fake_scope(
            monkeypatch, {"BUZZ_PRIVATE_KEY": _STUB_KEY * 2}
        )
        token = secret_scope.set_secret_scope({})  # active, empty scope

        try:
            assert adapter.check_requirements() is False
            assert calls == [], "an active scope must shadow the unscoped build"
        finally:
            secret_scope.reset_secret_scope(token)
