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


def test_preserve_existing_junk_config_ignored():
    for junk in ("notalist", 42, {"a": 1}, [1, 2], None):
        _, env = _apply(
            {"K": "v"}, cfg_extra={"preserve_existing": junk}, env={"K": "old"}
        )
        assert env["K"] == "v"  # falls back to normal override semantics


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


def test_alias_requires_credential_suffix():
    _, env = _apply({"RANDOM_SETTING_MILLA": "x"}, home=PROFILE_HOME)
    assert "RANDOM_SETTING" not in env


def test_alias_never_shadows_directly_supplied_var():
    """If the project also carries the canonical name, the alias must not
    fight it — direct supply wins."""
    _, env = _apply(
        {"TELEGRAM_BOT_TOKEN_MILLA": "profile-tok",
         "TELEGRAM_BOT_TOKEN": "canonical-tok"},
        home=PROFILE_HOME,
    )
    assert env["TELEGRAM_BOT_TOKEN"] == "canonical-tok"


def test_alias_respects_existing_env_without_override():
    class _NoOverride(_FakeBulk):
        def override_existing(self, cfg):
            return False

    registry.register_source(
        _NoOverride({"TELEGRAM_BOT_TOKEN_MILLA": "123:tok"}), replace=True
    )
    env = {"TELEGRAM_BOT_TOKEN": "existing"}
    registry.apply_all({"fakebulk": {"enabled": True}}, PROFILE_HOME, environ=env)
    assert env["TELEGRAM_BOT_TOKEN"] == "existing"


def test_alias_disabled_by_config():
    _, env = _apply(
        {"TELEGRAM_BOT_TOKEN_MILLA": "123:tok"},
        cfg_extra={"profile_alias": False},
        home=PROFILE_HOME,
    )
    assert "TELEGRAM_BOT_TOKEN" not in env


def test_default_profile_never_aliases():
    _, env = _apply(
        {"TELEGRAM_BOT_TOKEN_MILLA": "123:tok"},
        home=Path("/home/u/.hermes"),
    )
    assert "TELEGRAM_BOT_TOKEN" not in env


def test_hyphenated_profile_name_matches_underscore_suffix():
    _, env = _apply(
        {"SLACK_APP_TOKEN_MY_BOT": "xapp-1"},
        home=Path("/home/u/.hermes/profiles/my-bot"),
    )
    assert env["SLACK_APP_TOKEN"] == "xapp-1"


def test_alias_never_touches_protected_vars():
    class _Protecting(_FakeBulk):
        def protected_env_vars(self, cfg):
            return frozenset({"BWS_ACCESS_TOKEN"})

    registry.register_source(
        _Protecting({"BWS_ACCESS_TOKEN_MILLA": "0.evil"}), replace=True
    )
    env = {"BWS_ACCESS_TOKEN": "0.real"}
    registry.apply_all({"fakebulk": {"enabled": True}}, PROFILE_HOME, environ=env)
    assert env["BWS_ACCESS_TOKEN"] == "0.real"


def test_alias_provenance_recorded():
    report, _ = _apply({"NOTION_TOKEN_MILLA": "sec"}, home=PROFILE_HOME)
    assert report.provenance["NOTION_TOKEN"].source == "fakebulk"
