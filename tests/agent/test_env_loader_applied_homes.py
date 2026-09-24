"""Regression tests for #40597: _APPLIED_HOMES must be marked AFTER a real
fetch attempt, so early failures (malformed config, disabled sources) stay
retryable within the process instead of being permanently skipped."""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import env_loader


@pytest.fixture(autouse=True)
def _reset():
    env_loader.reset_secret_source_cache()
    yield
    env_loader.reset_secret_source_cache()
    from agent.secret_sources import registry
    registry._reset_registry_for_tests()


def _write_enabled_config(home: Path):
    (home / "config.yaml").write_text(
        "secrets:\n"
        "  bitwarden:\n"
        "    enabled: true\n"
        "    project_id: proj\n"
    )


def test_malformed_config_does_not_permanently_skip(tmp_path, monkeypatch):
    """Config error on first call → fixed config on second call must apply."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("secrets: [unclosed")  # malformed YAML

    env_loader._apply_external_secret_sources(home)
    assert str(home.resolve()) not in env_loader._APPLIED_HOMES

    # User fixes the config; same process must now attempt the fetch.
    _write_enabled_config(home)
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.t")

    import agent.secret_sources.bitwarden as bw
    calls = {"n": 0}

    def fake_fetch(**kwargs):
        calls["n"] += 1
        return {"NEW_KEY_40597": "val"}, []

    monkeypatch.setattr(bw, "find_bws", lambda install_if_missing=True: home / "bws")
    monkeypatch.setattr(bw, "fetch_bitwarden_secrets", fake_fetch)
    monkeypatch.delenv("NEW_KEY_40597", raising=False)

    from agent.secret_sources import registry
    registry._reset_registry_for_tests()

    env_loader._apply_external_secret_sources(home)
    assert calls["n"] == 1
    assert str(home.resolve()) in env_loader._APPLIED_HOMES
    monkeypatch.delenv("NEW_KEY_40597", raising=False)


def test_no_secrets_section_does_not_mark_applied(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("model:\n  provider: openrouter\n")
    env_loader._apply_external_secret_sources(home)
    assert str(home.resolve()) not in env_loader._APPLIED_HOMES


def test_disabled_sources_do_not_mark_applied(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "secrets:\n  bitwarden:\n    enabled: false\n    project_id: p\n"
    )
    from agent.secret_sources import registry
    registry._reset_registry_for_tests()
    env_loader._apply_external_secret_sources(home)
    assert str(home.resolve()) not in env_loader._APPLIED_HOMES


def test_fetch_error_still_marks_applied(tmp_path, monkeypatch):
    """A real fetch attempt that FAILS still marks the home — otherwise every
    import-time load_hermes_dotenv() would re-fetch and re-print the same
    error 3-5x per startup."""
    home = tmp_path / ".hermes"
    home.mkdir()
    _write_enabled_config(home)
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.dead")

    import agent.secret_sources.bitwarden as bw
    calls = {"n": 0}

    def boom(**kwargs):
        calls["n"] += 1
        raise RuntimeError("bws exited 1: network unreachable")

    monkeypatch.setattr(bw, "find_bws", lambda install_if_missing=True: home / "bws")
    monkeypatch.setattr(bw, "fetch_bitwarden_secrets", boom)

    from agent.secret_sources import registry
    registry._reset_registry_for_tests()

    env_loader._apply_external_secret_sources(home)
    env_loader._apply_external_secret_sources(home)  # second call = no-op
    assert calls["n"] == 1
    assert str(home.resolve()) in env_loader._APPLIED_HOMES


def test_success_marks_applied_and_second_call_noop(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    _write_enabled_config(home)
    monkeypatch.setenv("BWS_ACCESS_TOKEN", "0.t")
    monkeypatch.delenv("KEY_OK_40597", raising=False)

    import agent.secret_sources.bitwarden as bw
    calls = {"n": 0}

    def fake_fetch(**kwargs):
        calls["n"] += 1
        return {"KEY_OK_40597": "v"}, []

    monkeypatch.setattr(bw, "find_bws", lambda install_if_missing=True: home / "bws")
    monkeypatch.setattr(bw, "fetch_bitwarden_secrets", fake_fetch)

    from agent.secret_sources import registry
    registry._reset_registry_for_tests()

    env_loader._apply_external_secret_sources(home)
    env_loader._apply_external_secret_sources(home)
    assert calls["n"] == 1
    monkeypatch.delenv("KEY_OK_40597", raising=False)
