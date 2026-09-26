"""``hermes usage`` — the non-interactive /usage surface (issue #33094).

Drives the real ``hermes`` argparse entrypoint; only the network fetch is replaced with a snapshot
(``agent.account_usage.fetch_account_usage`` is what ``cmd_usage`` reads at call time).
"""

import json
import sys
import time
from datetime import datetime, timezone
from threading import Event
from unittest.mock import patch

import httpx

from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
from hermes_cli import main as hermes_main
from hermes_cli.subcommands.usage import usage_snapshot_document

_SNAPSHOT = AccountUsageSnapshot(
    provider="openai-codex", source="usage_api", fetched_at=datetime(2026, 9, 19, 12, 0, tzinfo=timezone.utc),
    plan="Plus",
    windows=(
        AccountUsageWindow(label="Session", used_percent=37.0, reset_at=datetime(2026, 9, 19, 21, 0, tzinfo=timezone.utc)),
        AccountUsageWindow(label="Weekly", used_percent=12.5, reset_at=None),
    ),
    details=("You have 1 reset banked - use /usage reset to activate",),
)


def _run(argv, fetch):
    with patch.object(hermes_main, "_plugin_cli_discovery_needed", return_value=False), \
         patch("agent.account_usage.fetch_account_usage", fetch), \
         patch.object(sys, "argv", ["hermes", *argv]):
        try:
            hermes_main.main()
        except SystemExit as exc:
            return int(exc.code or 0)
    return 0


def test_hermes_usage_json_is_one_stable_document(capsys):
    calls = []

    def fetch(provider, **kwargs):
        calls.append(provider)
        return _SNAPSHOT

    assert _run(["usage", "--json", "--provider", "openai-codex"], fetch) == 0
    out, err = capsys.readouterr()
    doc = json.loads(out)
    assert calls == ["openai-codex"] and err == ""
    assert doc["provider"] == "openai-codex" and doc["plan"] == "Plus"
    assert doc["fetched_at"] == "2026-09-19T12:00:00+00:00"
    assert doc["windows"] == [
        {"label": "Session", "used_percent": 37.0, "resets_at": "2026-09-19T21:00:00+00:00", "detail": None},
        {"label": "Weekly", "used_percent": 12.5, "resets_at": None, "detail": None},
    ]
    assert doc["details"] == ["You have 1 reset banked - use /usage reset to activate"]
    assert set(doc) == {"provider", "source", "title", "plan", "fetched_at", "windows", "details", "unavailable_reason"}


def test_hermes_usage_without_credential_exits_nonzero_with_one_stderr_line(capsys):
    # fetch_account_usage returns None when no credential resolves (or the fetch fails) — script-friendly failure.
    assert _run(["usage", "--json", "--provider", "openai-codex"], lambda provider, **kw: None) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert err.count("\n") == 1 and "openai-codex" in err


def test_all_credentials_fetches_each_persisted_row_without_selecting_or_leaking_tokens(capsys):
    rows = [
        {"id": "alpha", "access_token": "secret-alpha", "base_url": "https://chatgpt.com/backend-api/codex"},
        {"id": "beta", "access_token": "secret-beta", "last_status": "exhausted"},
    ]
    calls = []

    def fetch(provider, **kwargs):
        calls.append((provider, kwargs))
        return _SNAPSHOT if kwargs["api_key"] == "secret-alpha" else None

    with patch("agent.credential_pool.read_credential_pool", return_value=rows):
        assert _run(["usage", "--json", "--all-credentials", "--provider", "openai-codex"], fetch) == 0
    out, err = capsys.readouterr()
    doc = json.loads(out)
    assert err == ""
    assert calls == [
        ("openai-codex", {"api_key": "secret-alpha", "base_url": "https://chatgpt.com/backend-api/codex",
                          "allow_recovery": False}),
        ("openai-codex", {"api_key": "secret-beta", "base_url": None, "allow_recovery": False}),
    ]
    assert doc == {"provider": "openai-codex", "credentials": [
        {"id": "alpha", "usage": usage_snapshot_document(_SNAPSHOT)},
        {"id": "beta", "usage": None},
    ]}
    assert "secret-alpha" not in out and "secret-beta" not in out


def test_all_credentials_empty_or_failed_is_script_friendly(capsys):
    with patch("agent.credential_pool.read_credential_pool", return_value=[]):
        assert _run(["usage", "--json", "--all-credentials", "--provider", "openai-codex"], lambda *_a, **_kw: _SNAPSHOT) == 1
    out, err = capsys.readouterr()
    assert out == "" and "openai-codex" in err


def test_all_credentials_requires_json_and_rejects_nous(capsys):
    assert _run(["usage", "--all-credentials", "--provider", "openai-codex"], lambda *_a, **_kw: _SNAPSHOT) != 0
    assert _run(["usage", "--json", "--all-credentials", "--provider", "nous"], lambda *_a, **_kw: _SNAPSHOT) != 0
    # Anthropic's usage fetcher currently ignores its explicit api_key (see #20995).
    assert _run(["usage", "--json", "--all-credentials", "--provider", "anthropic"], lambda *_a, **_kw: _SNAPSHOT) != 0
    assert "secret" not in capsys.readouterr().out


def test_codex_all_credentials_probes_distinct_tokens_through_actual_fetcher(capsys):
    from agent.account_usage import fetch_account_usage

    rows = [{"id": "one", "access_token": "token-one"}, {"id": "two", "access_token": "token-two"}]
    seen = []

    def respond(url, headers, *, timeout):
        seen.append(headers["Authorization"])
        return {"plan_type": "plus", "rate_limit": {"primary_window": {
            "used_percent": 11 if headers["Authorization"] == "Bearer token-one" else 72,
        }}}

    with patch("agent.credential_pool.read_credential_pool", return_value=rows), \
         patch("agent.account_usage._get_json", side_effect=respond):
        assert _run(["usage", "--json", "--all-credentials", "--provider", "openai-codex"], fetch_account_usage) == 0
    out, err = capsys.readouterr()
    doc = json.loads(out)
    assert err == "" and seen == ["Bearer token-one", "Bearer token-two"]
    assert [c["usage"]["windows"][0]["used_percent"] for c in doc["credentials"]] == [11, 72]
    assert "token-one" not in out and "token-two" not in out


def test_codex_all_credentials_401_does_not_change_auth_state(tmp_path, monkeypatch, capsys):
    from agent import account_usage, credential_pool

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    original = {"version": 1, "providers": {"openai-codex": {"tokens": {
        "access_token": "rejected", "refresh_token": "revoked",
    }}}, "credential_pool": {"openai-codex": [
        {"id": "a", "source": "device_code", "auth_type": "oauth",
         "access_token": "rejected", "refresh_token": "revoked"},
    ]}}
    store = home / "auth.json"
    store.write_text(json.dumps(original))
    request = httpx.Request("GET", "https://chatgpt.com/backend-api/wham/usage")

    def unauthorized(*args, **kwargs):
        response = httpx.Response(401, request=request)
        raise httpx.HTTPStatusError("unauthorized", request=request, response=response)

    with patch.object(account_usage, "_get_json", side_effect=unauthorized), \
         patch("hermes_cli.auth._import_codex_cli_tokens", return_value=None), \
         patch.object(credential_pool.CredentialPool, "_post_tokens_refresh",
                      side_effect=RuntimeError("revoked")), \
         patch.object(credential_pool.auth_mod, "_is_terminal_codex_oauth_refresh_error", return_value=True):
        assert _run(["usage", "--json", "--all-credentials", "--provider", "openai-codex"],
                    account_usage.fetch_account_usage) == 1
    out, err = capsys.readouterr()
    assert err == ""
    assert json.loads(out)["credentials"] == [{"id": "a", "usage": None}]
    assert json.loads(store.read_text()) == original


def test_all_credentials_has_one_deadline_and_returns_complete_json(monkeypatch, capsys):
    from hermes_cli.subcommands import usage

    monkeypatch.setattr(usage, "ALL_CREDENTIALS_DEADLINE_S", 0.05, raising=False)
    release = Event()
    calls = []

    def slow_fetch(provider, **kwargs):
        calls.append(kwargs["api_key"])
        release.wait(3)
        return _SNAPSHOT

    rows = [{"id": str(i), "access_token": f"secret-{i}"} for i in range(3)]
    try:
        with patch("agent.credential_pool.read_credential_pool", return_value=rows):
            started = time.monotonic()
            assert _run(["usage", "--json", "--all-credentials", "--provider", "openai-codex"], slow_fetch) == 1
            assert time.monotonic() - started < 2
    finally:
        release.set()
    out, err = capsys.readouterr()
    assert err == ""
    assert calls == ["secret-0"]
    assert json.loads(out) == {"provider": "openai-codex", "credentials": [
        {"id": str(i), "usage": None} for i in range(3)
    ]}


def test_all_credentials_all_probes_unavailable_returns_json_and_nonzero(capsys):
    with patch("agent.credential_pool.read_credential_pool", return_value=[
        {"id": "one", "access_token": "secret-one"},
        {"id": "two", "access_token": ""},
    ]):
        assert _run(["usage", "--json", "--all-credentials", "--provider", "openrouter"],
                    lambda *_a, **_kw: None) == 1
    out, err = capsys.readouterr()
    assert err == "" and json.loads(out) == {"provider": "openrouter", "credentials": [
        {"id": "one", "usage": None}, {"id": "two", "usage": None},
    ]}
    assert "secret-one" not in out

