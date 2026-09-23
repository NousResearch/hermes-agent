"""Apply-path tool-credential handles (#107700 Phase 2 first cut).

When ``secrets.tool_credentials: handles``, tool-facing names are registered
on the apply report instead of being written to process environ. Provider and
secret-source bootstrap keys still hydrate. Default remains the old apply-all
path (fail-open).
"""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from agent.secret_sources import registry
from agent.secret_sources.base import ErrorKind, FetchResult, SecretSource


class _FakeSource(SecretSource):
    name = "fake"
    label = "Fake"
    shape = "mapped"

    def __init__(self, secrets):
        self._secrets = secrets

    def fetch(self, cfg, home_path):
        res = FetchResult()
        res.secrets = dict(self._secrets)
        return res

    def override_existing(self, cfg):
        return bool(cfg.get("override_existing", True))


@pytest.fixture(autouse=True)
def _clean_registry():
    registry._reset_registry_for_tests()
    registry._BUILTINS_LOADED = True
    yield
    registry._reset_registry_for_tests()


def _apply(secrets, cfg_extra=None, env=None, home=Path("/tmp/x/.hermes")):
    registry.register_source(_FakeSource(secrets), replace=True)
    cfg = {"fake": {"enabled": True}}
    cfg.update(cfg_extra or {})
    env = env if env is not None else {}
    report = registry.apply_all(cfg, home, environ=env)
    return report, env


def _handle_names(report):
    return [h.name for h in report.handles]


# ---------------------------------------------------------------------------
# handles mode: withhold tool-facing, still apply provider keys
# ---------------------------------------------------------------------------


def test_handles_mode_withholds_github_token_applies_provider_key():
    report, env = _apply(
        {"OPENROUTER_API_KEY": "sk-or-test", "GITHUB_TOKEN": "ghp-test"},
        cfg_extra={"tool_credentials": "handles"},
    )
    assert env["OPENROUTER_API_KEY"] == "sk-or-test"
    assert "GITHUB_TOKEN" not in env
    assert _handle_names(report) == ["GITHUB_TOKEN"]
    handle = report.handles[0]
    assert handle.name == "GITHUB_TOKEN"
    assert handle.source == "fake"
    assert {f.name for f in fields(handle)} == {"name", "source"}
    assert "ghp-test" not in repr(handle)


def test_handles_mode_withholds_explicit_tool_facing_set():
    secrets = {
        "GITHUB_TOKEN": "ghp",
        "GH_TOKEN": "gho",
        "GITHUB_APP_ID": "app-id",
        "GITHUB_APP_PRIVATE_KEY_PATH": "/tmp/key.pem",
        "GITHUB_APP_INSTALLATION_ID": "123",
        "AWS_ACCESS_KEY_ID": "AKIATEST",
        "AWS_SECRET_ACCESS_KEY": "aws-secret",
        "AWS_SESSION_TOKEN": "aws-sess",
        "OPENAI_API_KEY": "sk-openai",
        "ANTHROPIC_API_KEY": "sk-ant",
        "ANTHROPIC_TOKEN": "at-token",
        "BWS_ACCESS_TOKEN": "bws-token",
        "OP_SERVICE_ACCOUNT_TOKEN": "op-token",
    }
    report, env = _apply(secrets, cfg_extra={"tool_credentials": "handles"})
    withheld = {
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "GITHUB_APP_ID",
        "GITHUB_APP_PRIVATE_KEY_PATH",
        "GITHUB_APP_INSTALLATION_ID",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    }
    for name in withheld:
        assert name not in env
        assert name in _handle_names(report)
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_TOKEN",
        "BWS_ACCESS_TOKEN",
        "OP_SERVICE_ACCOUNT_TOKEN",
    ):
        assert env[name] == secrets[name]
        assert name not in _handle_names(report)
    assert all(h.source == "fake" for h in report.handles)


def test_unknown_name_still_applies_in_handles_mode():
    report, env = _apply(
        {"MY_CUSTOM_KEY": "custom-val", "GITHUB_TOKEN": "ghp-test"},
        cfg_extra={"tool_credentials": "handles"},
    )
    assert env["MY_CUSTOM_KEY"] == "custom-val"
    assert "GITHUB_TOKEN" not in env
    assert "MY_CUSTOM_KEY" not in _handle_names(report)


# ---------------------------------------------------------------------------
# fail-open: missing / env / typo / non-string → old apply
# ---------------------------------------------------------------------------


def test_fail_open_omitted_tool_credentials_applies_both():
    _, env = _apply({"OPENROUTER_API_KEY": "sk-or-test", "GITHUB_TOKEN": "ghp-test"})
    assert env["OPENROUTER_API_KEY"] == "sk-or-test"
    assert env["GITHUB_TOKEN"] == "ghp-test"


@pytest.mark.parametrize("mode", ["env", "ENV", "bogus", "", True, 1, None])
def test_fail_open_non_handles_values_apply_github_token(mode):
    extra = {} if mode is None else {"tool_credentials": mode}
    report, env = _apply(
        {"OPENROUTER_API_KEY": "sk-or-test", "GITHUB_TOKEN": "ghp-test"},
        cfg_extra=extra,
    )
    assert env["OPENROUTER_API_KEY"] == "sk-or-test"
    assert env["GITHUB_TOKEN"] == "ghp-test"
    assert not getattr(report, "handles", [])


# ---------------------------------------------------------------------------
# process_plaintext override + ignore malformed key
# ---------------------------------------------------------------------------


def test_process_plaintext_forces_github_token_apply():
    report, env = _apply(
        {"GITHUB_TOKEN": "ghp-test"},
        cfg_extra={
            "tool_credentials": "handles",
            "process_plaintext": ["GITHUB_TOKEN"],
        },
    )
    assert env["GITHUB_TOKEN"] == "ghp-test"
    assert _handle_names(report) == []


def test_process_plaintext_strips_names():
    _, env = _apply(
        {"GITHUB_TOKEN": "ghp-test"},
        cfg_extra={
            "tool_credentials": "handles",
            "process_plaintext": ["  GITHUB_TOKEN  "],
        },
    )
    assert env["GITHUB_TOKEN"] == "ghp-test"


def test_process_plaintext_non_list_is_ignored():
    report, env = _apply(
        {"GITHUB_TOKEN": "ghp-test"},
        cfg_extra={
            "tool_credentials": "handles",
            "process_plaintext": "GITHUB_TOKEN",
        },
    )
    assert "GITHUB_TOKEN" not in env
    assert _handle_names(report) == ["GITHUB_TOKEN"]


# ---------------------------------------------------------------------------
# existing guards still win; no handle recorded
# ---------------------------------------------------------------------------


def test_preserve_existing_beats_handles_no_handle_recorded():
    report, env = _apply(
        {"GITHUB_TOKEN": "from-source"},
        cfg_extra={
            "tool_credentials": "handles",
            "preserve_existing": ["GITHUB_TOKEN"],
        },
        env={"GITHUB_TOKEN": "already-there"},
    )
    assert env["GITHUB_TOKEN"] == "already-there"
    assert _handle_names(report) == []
    assert "GITHUB_TOKEN" in report.sources[0].skipped_existing


def test_failed_fetch_registers_no_handle():
    class _Broken(SecretSource):
        name = "broken"
        shape = "mapped"

        def fetch(self, cfg, home_path):
            return FetchResult().fail("boom", ErrorKind.NETWORK)

    registry.register_source(_Broken())
    env: dict = {}
    report = registry.apply_all(
        {"broken": {"enabled": True}, "tool_credentials": "handles"},
        Path("/tmp/x/.hermes"),
        environ=env,
    )
    assert env == {}
    assert _handle_names(report) == []
