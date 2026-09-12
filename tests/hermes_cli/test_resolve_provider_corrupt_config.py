"""Regression tests for issue #81952 (silent OpenRouter paid-default class fix).

When ``~/.hermes/config.yaml`` EXISTS but fails to parse, ``load_config()``
falls back to ``DEFAULT_CONFIG`` — so ``resolve_provider('auto')``'s tier-2
config check finds no ``model.provider`` and the tier-3 env sniff
(OPENROUTER_API_KEY / OPENAI_API_KEY) or tier-4 pool probe silently adopts the
PAID openrouter provider, even though the user's real (broken) config may name
a completely different provider (e.g. ``openai-codex``). Real money, zero
consent.

The fix: ``hermes_cli.config`` records active parse failures
(``get_active_config_parse_failure``), and ``resolve_provider`` refuses
env/pool auto-adoption with ``AuthError(code='corrupt_config')`` while the
active config is corrupt. Explicit provider requests and valid-config
env-sniff flows are untouched, and fixing the file in place clears the block.
"""

import uuid

import pytest


@pytest.fixture(autouse=True)
def _clean_inference_env(monkeypatch):
    for key in (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "NOUS_API_KEY",
        "HERMES_INFERENCE_PROVIDER",
    ):
        monkeypatch.delenv(key, raising=False)


CORRUPT_YAML = "model:\n  provider: openai-codex\n  default: gpt-5.5\n broken: [unterminated\n"
VALID_YAML = "gateway:\n  enabled: false\n"


def _setup_home(tmp_path, monkeypatch, config_text):
    home = tmp_path / "hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    cfg = home / "config.yaml"
    cfg.write_text(config_text)
    return home, cfg


def _load_config_fresh():
    """Call load_config so a corrupt file goes through the warn/record funnel."""
    from hermes_cli.config import load_config

    return load_config()


class TestParseFailureProbe:
    def test_probe_reports_active_corrupt_config(self, tmp_path, monkeypatch):
        _home, _cfg = _setup_home(tmp_path, monkeypatch, CORRUPT_YAML)
        _load_config_fresh()

        from hermes_cli.config import get_active_config_parse_failure

        err = get_active_config_parse_failure()
        assert err, "expected an active parse failure to be reported"

    def test_probe_clears_when_file_fixed_in_place(self, tmp_path, monkeypatch):
        _home, cfg = _setup_home(tmp_path, monkeypatch, CORRUPT_YAML)
        _load_config_fresh()

        from hermes_cli.config import get_active_config_parse_failure

        assert get_active_config_parse_failure()
        cfg.write_text(VALID_YAML)  # user fixes the YAML — different size/mtime
        assert get_active_config_parse_failure() is None

    def test_probe_none_for_valid_config(self, tmp_path, monkeypatch):
        _setup_home(tmp_path, monkeypatch, VALID_YAML)
        _load_config_fresh()

        from hermes_cli.config import get_active_config_parse_failure

        assert get_active_config_parse_failure() is None


class TestResolveProviderCorruptConfig:
    def test_corrupt_config_blocks_env_sniff_adoption(self, tmp_path, monkeypatch):
        """Corrupt config + OPENROUTER_API_KEY must NOT resolve to openrouter."""
        _setup_home(tmp_path, monkeypatch, CORRUPT_YAML)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-FAKE1234567890")
        _load_config_fresh()

        from hermes_cli.auth import AuthError, resolve_provider

        with pytest.raises(AuthError) as excinfo:
            resolve_provider("auto")
        assert excinfo.value.code == "corrupt_config"
        assert "config.yaml" in str(excinfo.value)

    def test_corrupt_config_blocks_pool_probe_adoption(self, tmp_path, monkeypatch):
        """Corrupt config + pool-only credential must NOT resolve to openrouter."""
        _setup_home(tmp_path, monkeypatch, CORRUPT_YAML)
        _load_config_fresh()

        from agent.credential_pool import (
            AUTH_TYPE_API_KEY,
            SOURCE_MANUAL,
            PooledCredential,
            load_pool,
        )

        pool = load_pool("openrouter")
        pool.add_entry(
            PooledCredential(
                provider="openrouter",
                id=uuid.uuid4().hex[:6],
                label="api-key-1",
                auth_type=AUTH_TYPE_API_KEY,
                priority=0,
                source=SOURCE_MANUAL,
                access_token="sk-or-FAKEKEY123",
                base_url="https://openrouter.ai/api/v1",
            )
        )

        from hermes_cli.auth import AuthError, resolve_provider

        with pytest.raises(AuthError) as excinfo:
            resolve_provider("auto")
        assert excinfo.value.code == "corrupt_config"

    def test_valid_config_env_sniff_keep_path_unbroken(self, tmp_path, monkeypatch):
        """KEEP: valid config that names no provider + env key still resolves."""
        _setup_home(tmp_path, monkeypatch, VALID_YAML)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-FAKE1234567890")
        _load_config_fresh()

        from hermes_cli.auth import resolve_provider

        assert resolve_provider("auto") == "openrouter"

    def test_fixed_config_clears_block(self, tmp_path, monkeypatch):
        """Rewriting the corrupt file valid clears the refusal immediately."""
        _home, cfg = _setup_home(tmp_path, monkeypatch, CORRUPT_YAML)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-FAKE1234567890")
        _load_config_fresh()

        from hermes_cli.auth import AuthError, resolve_provider

        with pytest.raises(AuthError):
            resolve_provider("auto")

        cfg.write_text(VALID_YAML)
        assert resolve_provider("auto") == "openrouter"

    def test_explicit_provider_request_untouched(self, tmp_path, monkeypatch):
        """Explicit user intent (requested != auto) resolves even with corrupt config."""
        _setup_home(tmp_path, monkeypatch, CORRUPT_YAML)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-FAKE1234567890")
        _load_config_fresh()

        from hermes_cli.auth import resolve_provider

        assert resolve_provider("openrouter") == "openrouter"
