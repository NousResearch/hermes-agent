"""Orchestrator-level profile secret handling.

Covers the two halves of the profile-clobber bug cluster:

- ``secrets.preserve_existing`` (#58073): named env vars keep their existing
  value even against a source with ``override_existing: true``.
- Profile aliasing (#51447): under a named profile, an applied
  ``FOO_<PROFILE>`` var also hydrates the canonical ``FOO`` so adapters and
  plugins that read fixed env names see the profile's value.

Both are implemented ONCE in ``apply_all()`` so every backend — bundled or
plugin — gets them for free.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agent.secret_sources import registry
from agent.secret_sources.base import ErrorKind, FetchResult, SecretSource


class _FakeBulk(SecretSource):
    name = "fakebulk"
    label = "Fake Bulk"
    shape = "bulk"

    def __init__(self, secrets):
        self._secrets = secrets

    def override_existing(self, cfg):
        return bool(cfg.get("override_existing", True))

    def fetch(self, cfg, home_path):
        res = FetchResult()
        res.secrets = dict(self._secrets)
        return res


@pytest.fixture(autouse=True)
def _clean_registry():
    registry._reset_registry_for_tests()
    registry._BUILTINS_LOADED = True  # keep real builtins out
    yield
    registry._reset_registry_for_tests()


def _apply(secrets, cfg_extra=None, home=Path("/tmp/x/.hermes"), env=None):
    registry.register_source(_FakeBulk(secrets), replace=True)
    cfg = {"fakebulk": {"enabled": True}}
    cfg.update(cfg_extra or {})
    env = env if env is not None else {}
    report = registry.apply_all(cfg, home, environ=env)
    return report, env


PROFILE_HOME = Path("/home/u/.hermes/profiles/milla")


# ---------------------------------------------------------------------------
# preserve_existing
# ---------------------------------------------------------------------------


def test_preserve_existing_beats_override():
    report, env = _apply(
        {"FEISHU_APP_SECRET": "shared", "OPENAI_API_KEY": "fresh"},
        cfg_extra={"preserve_existing": ["FEISHU_APP_SECRET"]},
        env={"FEISHU_APP_SECRET": "profile-local", "OPENAI_API_KEY": "stale"},
    )
    assert env["FEISHU_APP_SECRET"] == "profile-local"   # preserved
    assert env["OPENAI_API_KEY"] == "fresh"              # override still works
    sr = report.sources[0]
    assert "FEISHU_APP_SECRET" in sr.skipped_existing
    assert "OPENAI_API_KEY" in sr.applied


def test_preserve_existing_only_guards_set_vars():
    """A preserve-listed var with NO existing value still gets applied."""
    _, env = _apply(
        {"FEISHU_APP_SECRET": "shared"},
        cfg_extra={"preserve_existing": ["FEISHU_APP_SECRET"]},
        env={},
    )
    assert env["FEISHU_APP_SECRET"] == "shared"




# ---------------------------------------------------------------------------
# profile aliasing
# ---------------------------------------------------------------------------


def test_profile_suffixed_var_hydrates_canonical():
    report, env = _apply(
        {"TELEGRAM_BOT_TOKEN_MILLA": "123:tok"},
        home=PROFILE_HOME,
    )
    assert env["TELEGRAM_BOT_TOKEN_MILLA"] == "123:tok"
    assert env["TELEGRAM_BOT_TOKEN"] == "123:tok"
    assert "TELEGRAM_BOT_TOKEN" in report.provenance
    assert any("applied profile-scoped" in w
               for w in report.sources[0].result.warnings)












def test_hyphenated_profile_name_matches_underscore_suffix():
    _, env = _apply(
        {"SLACK_APP_TOKEN_MY_BOT": "xapp-1"},
        home=Path("/home/u/.hermes/profiles/my-bot"),
    )
    assert env["SLACK_APP_TOKEN"] == "xapp-1"


def test_source_fetch_reads_injected_environment_without_global_mutation(
    monkeypatch, tmp_path
):
    """Cold-profile bootstrap values reach sources through the local mapping."""
    from agent.secret_sources.base import get_source_environment

    class _BootstrapSource(SecretSource):
        name = "bootstrap"
        shape = "mapped"

        def fetch(self, cfg, home_path):
            result = FetchResult()
            result.secrets = {
                "RESOLVED_API_KEY": get_source_environment()["BOOTSTRAP_TOKEN"]
            }
            return result

    registry.register_source(_BootstrapSource())
    monkeypatch.delenv("BOOTSTRAP_TOKEN", raising=False)
    env = {"BOOTSTRAP_TOKEN": "profile-token"}
    _, applied = _apply(
        {},
        cfg_extra={"bootstrap": {"enabled": True}},
        home=tmp_path,
        env=env,
    )

    assert applied["RESOLVED_API_KEY"] == "profile-token"
    assert "BOOTSTRAP_TOKEN" not in __import__("os").environ


def test_empty_injected_environment_does_not_fall_back_to_process(monkeypatch, tmp_path):
    from agent.secret_sources.base import get_source_environment

    class _CanarySource(SecretSource):
        name = "canary"
        shape = "mapped"

        def fetch(self, cfg, home_path):
            result = FetchResult()
            assert get_source_environment().get("LEAK_CANARY") is None
            return result

    registry.register_source(_CanarySource())
    monkeypatch.setenv("LEAK_CANARY", "global-secret")
    registry.apply_all(
        {"canary": {"enabled": True}}, tmp_path, environ={}
    )




