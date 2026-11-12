"""``hermes usage`` — the non-interactive /usage surface (issue #33094).

Drives the real ``hermes`` argparse entrypoint; only the network fetch is replaced with a snapshot
(``agent.account_usage.fetch_account_usage`` is what ``cmd_usage`` reads at call time).
"""

import json
import sys
from datetime import datetime, timezone, UTC
from unittest.mock import patch

from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
from hermes_cli import main as hermes_main

_SNAPSHOT = AccountUsageSnapshot(
    provider="openai-codex", source="usage_api", fetched_at=datetime(2026, 9, 19, 12, 0, tzinfo=UTC),
    plan="Plus",
    windows=(
        AccountUsageWindow(label="Session", used_percent=37.0, reset_at=datetime(2026, 9, 19, 21, 0, tzinfo=UTC)),
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


def test_hermes_usage_without_credential_reports_the_kind(capsys):
    # A None fetch with no exception means the provider can report usage but no credential resolved.
    assert _run(["usage", "--json", "--provider", "openai-codex"], lambda provider, **kw: None) == 1
    out, err = capsys.readouterr()
    doc = json.loads(out)
    assert doc["provider"] == "openai-codex"
    assert doc["error"] == "no_credential"
    assert "no credential is configured" in doc["detail"]
    assert err.count("\n") == 1 and "openai-codex" in err


def test_hermes_usage_names_a_failed_fetch_instead_of_the_generic_line(capsys):
    class _Boom(Exception):
        pass

    def fetch(provider, **kwargs):
        kwargs["on_error"](_Boom("HTTP 401 Unauthorized"))

    assert _run(["usage", "--json", "--provider", "anthropic"], fetch) == 1
    out, err = capsys.readouterr()
    doc = json.loads(out)
    assert doc["error"] == "fetch_failed"
    assert "_Boom" in doc["detail"] and "HTTP 401" in doc["detail"]
    assert "HTTP 401" in err and err.count("\n") == 1


def test_hermes_usage_reports_a_provider_without_a_usage_endpoint(capsys):
    with patch("agent.account_usage_cache.has_account_usage", return_value=False):
        assert _run(["usage", "--json", "--provider", "openai-codex"], lambda provider, **kw: None) == 1
    out, err = capsys.readouterr()
    doc = json.loads(out)
    assert doc["error"] == "no_usage_endpoint"
    assert "no usage endpoint" in err

