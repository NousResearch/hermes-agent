"""API-key displays must count credential-pool entries, not just .env.

``hermes auth add`` persists a credential to the pool in ``auth.json`` and never writes ``.env``.
Three displays resolved providers from env vars alone and so rendered "(not set)" for a provider
that was authed and serving traffic: ``hermes doctor`` (see tests/hermes_cli/test_doctor.py),
``hermes status`` (see tests/hermes_cli/test_status.py), and ``hermes config show`` here.

``auth.explicit_pool_providers`` is the one resolver behind all three; ``pool_credential_labels``
adds the env-var keying the two display sites need.
"""

import pytest


@pytest.fixture
def clean_anthropic_env(monkeypatch, tmp_path):
    """No Anthropic credential in either ``.env`` or the process env."""
    from hermes_cli.auth import PROVIDER_REGISTRY

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for var in PROVIDER_REGISTRY["anthropic"].api_key_env_vars:
        monkeypatch.delenv(var, raising=False)


def _set_pool(monkeypatch, pool):
    import hermes_cli.auth as auth_mod

    if isinstance(pool, Exception):
        def boom(provider_id=None):
            raise pool
        monkeypatch.setattr(auth_mod, "read_credential_pool", boom)
    else:
        monkeypatch.setattr(auth_mod, "read_credential_pool", lambda provider_id=None: pool)


class TestExplicitPoolProviders:
    """The shared primitive: which providers hold a credential the user deliberately added."""

    def test_returns_the_first_explicit_entry_per_provider(self, monkeypatch):
        from hermes_cli.auth import explicit_pool_providers

        _set_pool(monkeypatch, {"anthropic": [
            {"source": "claude_code", "label": "Ambient"},
            {"source": "manual", "label": "First explicit"},
            {"source": "manual", "label": "Second explicit"}]})

        assert explicit_pool_providers()["anthropic"]["label"] == "First explicit"

    def test_providers_with_only_ambient_entries_are_omitted(self, monkeypatch):
        from hermes_cli.auth import explicit_pool_providers

        _set_pool(monkeypatch, {
            "anthropic": [{"source": "claude_code"}], "copilot": [{"source": "gh_cli"}],
            "zai": [{"source": "device_code"}]})

        assert sorted(explicit_pool_providers()) == ["zai"]

    def test_malformed_provider_slices_are_skipped(self, monkeypatch):
        from hermes_cli.auth import explicit_pool_providers

        _set_pool(monkeypatch, {"anthropic": "not-a-list", "zai": [{"source": "manual"}]})

        assert sorted(explicit_pool_providers()) == ["zai"]

    def test_unreadable_store_yields_nothing_instead_of_raising(self, monkeypatch):
        from hermes_cli.auth import explicit_pool_providers

        _set_pool(monkeypatch, OSError("auth.json unreadable"))

        assert explicit_pool_providers() == {}


class TestDoctorAndDisplaysShareOneResolver:
    """doctor, status and config must agree — they now resolve through the same primitive."""

    def test_doctor_sees_what_the_displays_see(self, monkeypatch):
        from hermes_cli.auth import pool_credential_labels
        from hermes_cli.doctor_config import pooled_credential_providers

        _set_pool(monkeypatch, {
            "anthropic": [{"source": "manual", "label": "Agentic"}],
            "openrouter": [{"source": "claude_code", "label": "Ambient"}]})

        assert pooled_credential_providers() == ["anthropic"]
        assert pool_credential_labels()["ANTHROPIC_API_KEY"] == "Agentic"

    def test_both_go_quiet_on_an_unreadable_store(self, monkeypatch):
        from hermes_cli.auth import pool_credential_labels
        from hermes_cli.doctor_config import pooled_credential_providers

        _set_pool(monkeypatch, OSError("auth.json unreadable"))

        assert pooled_credential_providers() == []
        assert pool_credential_labels() == {}


class TestPoolCredentialLabels:
    def test_maps_provider_env_vars_to_its_pool_label(self, monkeypatch):
        from hermes_cli.auth import pool_credential_labels

        _set_pool(monkeypatch, {"anthropic": [{"source": "manual", "label": "Agentic"}]})

        assert pool_credential_labels()["ANTHROPIC_API_KEY"] == "Agentic"

    def test_falls_back_to_provider_id_when_entry_has_no_label(self, monkeypatch):
        from hermes_cli.auth import pool_credential_labels

        _set_pool(monkeypatch, {"anthropic": [{"source": "manual", "label": "   "}]})

        assert pool_credential_labels()["ANTHROPIC_API_KEY"] == "anthropic"

    def test_ambient_borrowed_entries_are_excluded(self, monkeypatch):
        from hermes_cli.auth import pool_credential_labels

        _set_pool(monkeypatch, {"anthropic": [{"source": "claude_code", "label": "Claude Code"}]})

        assert pool_credential_labels() == {}

    def test_first_explicit_entry_wins_for_a_shared_env_var(self, monkeypatch):
        from hermes_cli.auth import pool_credential_labels

        _set_pool(monkeypatch, {"anthropic": [
            {"source": "manual", "label": "First"}, {"source": "manual", "label": "Second"}]})

        assert pool_credential_labels()["ANTHROPIC_API_KEY"] == "First"

    def test_unreadable_store_yields_no_labels_instead_of_raising(self, monkeypatch):
        from hermes_cli.auth import pool_credential_labels

        _set_pool(monkeypatch, OSError("auth.json unreadable"))

        assert pool_credential_labels() == {}

    def test_pooled_secret_is_never_returned(self, monkeypatch):
        from hermes_cli.auth import pool_credential_labels

        sentinel = "NONSECRET_SENTINEL_POOL_VALUE_DO_NOT_RETURN_123456"
        _set_pool(monkeypatch, {
            "anthropic": [{"source": "manual", "label": "Agentic", "api_key": sentinel}]})

        assert sentinel not in str(pool_credential_labels())


class TestShowConfigApiKeys:
    """``hermes config show`` — the third copy of the env-only API-Keys rendering."""

    @staticmethod
    def _anthropic_row(out):
        return next(line for line in out.splitlines() if "Anthropic" in line)

    def test_pool_only_credential_renders_as_configured(
            self, monkeypatch, capsys, clean_anthropic_env):
        from hermes_cli.config import show_config

        _set_pool(monkeypatch, {"anthropic": [{"source": "manual", "label": "Agentic"}]})
        show_config()

        row = self._anthropic_row(capsys.readouterr().out)
        assert "credential pool (Agentic)" in row
        assert "(not set)" not in row

    def test_env_value_takes_precedence_over_pool_label(
            self, monkeypatch, capsys, clean_anthropic_env):
        from hermes_cli.config import show_config

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env-value-not-a-real-key")
        _set_pool(monkeypatch, {"anthropic": [{"source": "manual", "label": "Agentic"}]})
        show_config()

        row = self._anthropic_row(capsys.readouterr().out)
        assert "credential pool" not in row
        assert "sk-ant-env-value-not-a-real-key" not in row

    def test_pooled_secret_value_is_never_printed(
            self, monkeypatch, capsys, clean_anthropic_env):
        from hermes_cli.config import show_config

        sentinel = "NONSECRET_SENTINEL_POOL_VALUE_DO_NOT_PRINT_123456"
        _set_pool(monkeypatch, {
            "anthropic": [{"source": "manual", "label": "Agentic", "api_key": sentinel}]})
        show_config()

        assert sentinel not in capsys.readouterr().out

    def test_unpooled_provider_still_reads_not_set(
            self, monkeypatch, capsys, clean_anthropic_env):
        from hermes_cli.config import show_config

        _set_pool(monkeypatch, {})
        show_config()

        assert "(not set)" in self._anthropic_row(capsys.readouterr().out)
