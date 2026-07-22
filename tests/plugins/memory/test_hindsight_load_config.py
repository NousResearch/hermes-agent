"""Regression tests for `plugins.memory.hindsight._load_config`.

Covers the #51166 bug: the Hindsight plugin ignored the HINDSIGHT_BANK_ID
env var (and other HINDSIGHT_* env vars) once a `config.json` existed on
disk, because `_load_config()` short-circuited with `return json.loads(...)`
and never reached the env-var defaults. After the fix, env vars act as
**fallbacks** for the JSON-loaded config — so a per-profile `.env` like
`HINDSIGHT_BANK_ID=nihai-tcm` overrides the bank regardless of whether
the user has a `~/.hermes/hindsight/config.json` from a prior `hermes
memory status` run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plugins.memory.hindsight import _load_config


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Wipe all HINDSIGHT_* env vars so they cannot leak between tests."""
    for key in (
        "HINDSIGHT_API_KEY", "HINDSIGHT_API_URL", "HINDSIGHT_BANK_ID",
        "HINDSIGHT_BUDGET", "HINDSIGHT_MODE", "HINDSIGHT_TIMEOUT",
        "HINDSIGHT_IDLE_TIMEOUT", "HINDSIGHT_LLM_API_KEY",
        "HINDSIGHT_RETAIN_TAGS", "HINDSIGHT_RETAIN_OBSERVATION_SCOPES",
        "HINDSIGHT_RETAIN_SOURCE",
        "HINDSIGHT_RETAIN_USER_PREFIX", "HINDSIGHT_RETAIN_ASSISTANT_PREFIX",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def config_json(tmp_path, monkeypatch):
    """Write a config.json under the profile-scoped path and point
    HERMES_HOME at tmp_path so _load_config() finds it."""
    cfg_dir = tmp_path / "hindsight"
    cfg_dir.mkdir()
    cfg_path = cfg_dir / "config.json"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return cfg_path


# ---------------------------------------------------------------------------
# The #51166 regression: env var must win even when config.json exists.
# ---------------------------------------------------------------------------

class TestEnvVarOverridesConfigJson:
    """`HINDSIGHT_BANK_ID` must take effect regardless of config.json."""

    def test_bank_id_env_var_wins_over_config_json(self, config_json, monkeypatch):
        """The headline bug: a config.json with a different bank must be
        overridden by HINDSIGHT_BANK_ID in the .env."""
        config_json.write_text(json.dumps({
            "mode": "cloud",
            "apiKey": "json-stored-key",
            "banks": {"hermes": {"bankId": "from-json", "enabled": True}},
        }))
        monkeypatch.setenv("HINDSIGHT_BANK_ID", "nihai-tcm")

        cfg = _load_config()
        # The env-var path stores the override at top-level bank_id (read
        # site is `self._config.get("bank_id") or banks.get("bankId", ...)`).
        # Either representation is fine; the contract is that the
        # *resolved* bank is the env-var one, not the JSON one.
        assert cfg.get("bank_id") == "nihai-tcm"
        # The JSON-stored bank must NOT silently win.
        assert cfg.get("banks", {}).get("hermes", {}).get("bankId") != "nihai-tcm" or \
            cfg.get("bank_id") == "nihai-tcm"

    def test_other_hindsight_env_vars_overlay_config_json(self, config_json, monkeypatch):
        """Same fallback principle for the rest of the HINDSIGHT_* env vars:
        HINDSIGHT_MODE, HINDSIGHT_API_KEY, HINDSIGHT_BUDGET."""
        config_json.write_text(json.dumps({
            "mode": "cloud",
            "apiKey": "json-key",
            "banks": {"hermes": {"bankId": "json-bank", "budget": "mid"}},
        }))
        monkeypatch.setenv("HINDSIGHT_MODE", "local_embedded")
        monkeypatch.setenv("HINDSIGHT_API_KEY", "env-key")
        monkeypatch.setenv("HINDSIGHT_BUDGET", "high")
        monkeypatch.setenv("HINDSIGHT_BANK_ID", "env-bank")

        cfg = _load_config()
        assert cfg.get("mode") == "local_embedded"
        # API key — we store the env override at top-level `apiKey` and
        # keep the JSON value as a fallback under `json_apiKey` (or any
        # name that does not collide) so the JSON path is recoverable.
        assert cfg.get("apiKey") == "env-key"
        assert cfg.get("budget") == "high"
        assert cfg.get("bank_id") == "env-bank"

    def test_no_env_var_keeps_json_value(self, config_json):
        """When the user has NOT set the env var, the JSON value must
        still be the source of truth — no spurious defaults."""
        config_json.write_text(json.dumps({
            "mode": "cloud",
            "apiKey": "json-key",
            "banks": {"hermes": {"bankId": "json-bank", "enabled": True}},
        }))
        cfg = _load_config()
        # apiKey comes from JSON, mode from JSON, bank comes from JSON.
        assert cfg.get("apiKey") == "json-key"
        assert cfg.get("mode") == "cloud"


# ---------------------------------------------------------------------------
# Backward compat: when no config.json exists, env vars are the source.
# ---------------------------------------------------------------------------

class TestEnvVarsWhenNoConfigJson:
    """Without a config.json, env vars must seed the whole config dict."""

    def test_env_vars_seed_when_no_config_json(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HINDSIGHT_BANK_ID", "env-only-bank")
        monkeypatch.setenv("HINDSIGHT_MODE", "cloud")
        monkeypatch.setenv("HINDSIGHT_API_KEY", "env-only-key")

        cfg = _load_config()
        assert cfg.get("bank_id") == "env-only-bank"
        assert cfg.get("apiKey") == "env-only-key"
        assert cfg.get("mode") == "cloud"


# ---------------------------------------------------------------------------
# Bank-id resolution: the value must reach the read site in __init__.
# ---------------------------------------------------------------------------

class TestBankIdReachesReadSite:
    """End-to-end: the resolved bank id must be reachable via the same
    lookup the HindsightMemoryProvider.__init__ does at line 1287:
        static_bank_id = self._config.get("bank_id") or banks.get("bankId", "hermes")
    """

    def test_env_bank_id_is_first_class(self, config_json, monkeypatch):
        """After loading, `cfg.get("bank_id")` must be the env-var value
        so the read site's `or` short-circuits to it without falling
        through to the JSON-stored `banks["hermes"]["bankId"]`."""
        config_json.write_text(json.dumps({
            "mode": "cloud",
            "apiKey": "json-key",
            "banks": {"hermes": {"bankId": "STALE-JSON-BANK", "enabled": True}},
        }))
        monkeypatch.setenv("HINDSIGHT_BANK_ID", "fresh-env-bank")

        cfg = _load_config()
        static_bank_id = cfg.get("bank_id") or cfg.get("banks", {}).get("hermes", {}).get("bankId", "hermes")
        assert static_bank_id == "fresh-env-bank"
