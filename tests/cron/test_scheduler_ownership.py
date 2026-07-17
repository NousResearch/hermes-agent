"""Behavior contracts for selecting one cron scheduler owner per Hermes home."""
from __future__ import annotations

import pytest


def _write_config(home, body: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(body, encoding="utf-8")


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, "auto"),
        ({"cron": {}}, "auto"),
        ({"cron": {"scheduler_owner": "auto"}}, "auto"),
        ({"cron": {"scheduler_owner": " gateway "}}, "gateway"),
        ({"cron": {"scheduler_owner": "DESKTOP"}}, "desktop"),
    ],
)
def test_scheduler_owner_selection_contract(config, expected):
    from cron.scheduler_provider import resolve_cron_scheduler_owner

    assert resolve_cron_scheduler_owner(config=config) == expected


@pytest.mark.parametrize("invalid", ["", "both", 42, None])
def test_invalid_scheduler_owner_fails_closed_without_logging_value(invalid, caplog):
    from cron.scheduler_provider import resolve_cron_scheduler_owner

    with caplog.at_level("ERROR"):
        assert resolve_cron_scheduler_owner(
            config={"cron": {"scheduler_owner": invalid}}
        ) is None
    assert "scheduler startup disabled" in caplog.text
    if isinstance(invalid, str) and invalid:
        assert invalid not in caplog.text


def test_default_config_uses_safe_auto_owner():
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["cron"]["scheduler_owner"] == "auto"


@pytest.mark.parametrize(
    "body",
    [
        'cron:\n  scheduler_owner: "unterminated\n',
        "cron: desktop\n",
        "cron:\n  scheduler_owner: both\n",
        "null\n",
    ],
)
def test_owner_file_errors_fail_closed(tmp_path, monkeypatch, body):
    from cron.scheduler_provider import resolve_cron_scheduler_owner

    _write_config(tmp_path, body)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert resolve_cron_scheduler_owner() is None


def test_named_profile_reads_its_registry_not_default_registry(tmp_path, monkeypatch):
    from cron.scheduler_provider import should_start_cron_scheduler

    default_home = tmp_path / ".hermes"
    profile_home = default_home / "profiles" / "worker"
    _write_config(default_home, "cron:\n  scheduler_owner: gateway\n")
    _write_config(profile_home, "cron:\n  scheduler_owner: desktop\n")
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    assert should_start_cron_scheduler("gateway") is False
    assert should_start_cron_scheduler("desktop") is True


def test_owner_env_reference_is_expanded(tmp_path, monkeypatch):
    from cron.scheduler_provider import resolve_cron_scheduler_owner

    _write_config(tmp_path, "cron:\n  scheduler_owner: ${CRON_OWNER}\n")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("CRON_OWNER", "desktop")
    assert resolve_cron_scheduler_owner() == "desktop"


def test_managed_owner_env_wins_and_uses_selected_profile_scope(tmp_path, monkeypatch):
    from cron.scheduler_provider import resolve_cron_scheduler_owner

    profile_home = tmp_path / ".hermes" / "profiles" / "worker"
    selected_managed = tmp_path / "managed-worker"
    other_managed = tmp_path / "managed-default"
    _write_config(profile_home, "cron:\n  scheduler_owner: desktop\n")
    _write_config(other_managed, "cron:\n  scheduler_owner: desktop\n")
    _write_config(selected_managed, "cron:\n  scheduler_owner: ${MANAGED_OWNER}\n")
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(selected_managed))
    monkeypatch.setenv("MANAGED_OWNER", "gateway")
    assert resolve_cron_scheduler_owner() == "gateway"


def test_unresolved_owner_env_reference_fails_closed_without_logging_value(
    tmp_path, monkeypatch, caplog
):
    from cron.scheduler_provider import resolve_cron_scheduler_owner

    ref = "${UNSET_CRON_OWNER_FOR_TEST}"
    _write_config(tmp_path, f"cron:\n  scheduler_owner: {ref}\n")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("UNSET_CRON_OWNER_FOR_TEST", raising=False)
    with caplog.at_level("ERROR"):
        assert resolve_cron_scheduler_owner() is None
    assert ref not in caplog.text


def test_managed_provider_precedence_is_read_with_owner(tmp_path, monkeypatch):
    from cron.scheduler_runtime import read_scheduler_ownership_policy_strict

    managed = tmp_path / "managed"
    _write_config(tmp_path, "cron:\n  scheduler_owner: desktop\n  provider: builtin\n")
    _write_config(managed, "cron:\n  scheduler_owner: gateway\n  provider: chronos\n")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    policy = read_scheduler_ownership_policy_strict()
    assert policy is not None
    assert (policy.mode, policy.configured_provider) == ("gateway", "chronos")
