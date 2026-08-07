"""Legacy `secrets:` shapes must warn instead of failing silently.

Two config shapes are read by nothing on current main and produce no
diagnostic at all — a credential that never loads, with no explanation:

* ``secrets.provider`` — the single-backend selector that predates source
  composition. Not read anywhere.
* ``secrets.<source>`` as a scalar instead of a mapping — coerced to ``{}``
  in ``_ordered_enabled_sources``, so ``is_enabled`` says False.

Both are what a config written against the earlier iteration looks like, so
the failure lands on exactly the users who configured secrets early. These
tests exercise the real orchestrator against a real temp HERMES_HOME with a
real helper subprocess — no mocks.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from agent.secret_sources import registry as reg


@pytest.fixture(autouse=True)
def _fresh_registry():
    reg._reset_registry_for_tests()
    yield
    reg._reset_registry_for_tests()


def _helper(tmp_path: Path, body: str) -> Path:
    """A real executable helper that prints a dotenv blob."""
    script = tmp_path / "helper.sh"
    script.write_text(f"#!/bin/sh\n{body}\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


class TestLegacyProviderKey:
    def test_warns_that_provider_is_ignored(self, tmp_path, caplog):
        env: dict[str, str] = {}
        with caplog.at_level("WARNING", logger=reg.logger.name):
            reg.apply_all({"provider": "command"}, tmp_path, environ=env)
        assert any("secrets.provider is no longer supported" in m
                   for m in caplog.messages), caplog.messages

    def test_warning_names_the_replacement(self, tmp_path, caplog):
        with caplog.at_level("WARNING", logger=reg.logger.name):
            reg.apply_all({"provider": "command"}, tmp_path, environ={})
        joined = " ".join(caplog.messages)
        assert "enabled" in joined, "warning must point at the working shape"

    def test_absent_provider_key_is_silent(self, tmp_path, caplog):
        with caplog.at_level("WARNING", logger=reg.logger.name):
            reg.apply_all({}, tmp_path, environ={})
        assert not any("provider" in m for m in caplog.messages)


class TestNonMappingSourceSection:
    def test_warns_when_section_is_a_string(self, tmp_path, caplog):
        cfg = {"command": "cat /run/user/1000/hermes-secrets.env"}
        with caplog.at_level("WARNING", logger=reg.logger.name):
            reg.apply_all(cfg, tmp_path, environ={})
        assert any("secrets.command must be a mapping" in m
                   for m in caplog.messages), caplog.messages

    def test_warning_reports_the_offending_type(self, tmp_path, caplog):
        with caplog.at_level("WARNING", logger=reg.logger.name):
            reg.apply_all({"command": ["a", "b"]}, tmp_path, environ={})
        assert any("is a list" in m for m in caplog.messages), caplog.messages

    def test_valid_mapping_does_not_warn(self, tmp_path, caplog):
        cfg = {"command": {"enabled": False, "command": "true"}}
        with caplog.at_level("WARNING", logger=reg.logger.name):
            reg.apply_all(cfg, tmp_path, environ={})
        assert not any("must be a mapping" in m for m in caplog.messages)

    def test_absent_section_does_not_warn(self, tmp_path, caplog):
        """A source the user never configured must stay quiet."""
        with caplog.at_level("WARNING", logger=reg.logger.name):
            reg.apply_all({}, tmp_path, environ={})
        assert not any("must be a mapping" in m for m in caplog.messages)


class TestBehaviorUnchanged:
    """The warnings are diagnostics only — resolution must not shift."""

    def test_legacy_config_still_applies_nothing(self, tmp_path):
        cfg = {"provider": "command", "command": "printf 'K=v\\n'"}
        env: dict[str, str] = {}
        reg.apply_all(cfg, tmp_path, environ=env)
        assert env == {}, "legacy shape must remain inert, not start working"

    def test_current_schema_still_applies_secrets(self, tmp_path):
        helper = _helper(tmp_path, "printf 'LEGACY_TEST_KEY=abc123\\n'")
        cfg = {"command": {"enabled": True, "command": str(helper)}}
        env: dict[str, str] = {}
        reg.apply_all(cfg, tmp_path, environ=env)
        assert env.get("LEGACY_TEST_KEY") == "abc123"

    def test_provider_key_alongside_valid_section_still_works(self, tmp_path):
        """A half-migrated config: the valid section must still load."""
        helper = _helper(tmp_path, "printf 'HALF_MIGRATED=yes\\n'")
        cfg = {"provider": "command",
               "command": {"enabled": True, "command": str(helper)}}
        env: dict[str, str] = {}
        reg.apply_all(cfg, tmp_path, environ=env)
        assert env.get("HALF_MIGRATED") == "yes"

    def test_disabled_source_stays_disabled(self, tmp_path):
        helper = _helper(tmp_path, "printf 'SHOULD_NOT_LOAD=x\\n'")
        cfg = {"command": {"enabled": False, "command": str(helper)}}
        env: dict[str, str] = {}
        reg.apply_all(cfg, tmp_path, environ=env)
        assert "SHOULD_NOT_LOAD" not in env


class TestEndToEndFromConfigFile:
    """Exercise the real env_loader path against a real HERMES_HOME."""

    def test_legacy_config_yaml_warns_and_loads_nothing(
            self, tmp_path, monkeypatch, caplog):
        yaml = pytest.importorskip("yaml")
        home = tmp_path / "hermes_home"
        home.mkdir()
        helper = _helper(tmp_path, "printf 'E2E_LEGACY_KEY=nope\\n'")
        (home / "config.yaml").write_text(yaml.safe_dump({
            "secrets": {"provider": "command", "command": str(helper)},
        }))

        monkeypatch.setenv("HERMES_HOME", str(home))
        env: dict[str, str] = {}
        cfg = yaml.safe_load((home / "config.yaml").read_text())["secrets"]
        with caplog.at_level("WARNING", logger=reg.logger.name):
            reg.apply_all(cfg, home, environ=env)

        assert "E2E_LEGACY_KEY" not in env
        assert any("no longer supported" in m or "must be a mapping" in m
                   for m in caplog.messages), caplog.messages

    def test_migrated_config_yaml_loads(self, tmp_path, monkeypatch):
        yaml = pytest.importorskip("yaml")
        home = tmp_path / "hermes_home"
        home.mkdir()
        helper = _helper(tmp_path, "printf 'E2E_NEW_KEY=works\\n'")
        (home / "config.yaml").write_text(yaml.safe_dump({
            "secrets": {"command": {"enabled": True, "command": str(helper)}},
        }))

        monkeypatch.setenv("HERMES_HOME", str(home))
        env: dict[str, str] = {}
        cfg = yaml.safe_load((home / "config.yaml").read_text())["secrets"]
        reg.apply_all(cfg, home, environ=env)
        assert env.get("E2E_NEW_KEY") == "works"


class TestNoSecretLeakage:
    def test_warning_never_contains_the_config_value(self, tmp_path, caplog):
        """The section value can be a command string — keep it out of logs."""
        secretish = "cat /run/keys/PRODUCTION_TOKEN_abcdef123456"
        with caplog.at_level("WARNING", logger=reg.logger.name):
            reg.apply_all({"command": secretish}, tmp_path, environ={})
        joined = " ".join(caplog.messages)
        assert "abcdef123456" not in joined
        assert "PRODUCTION_TOKEN" not in joined
