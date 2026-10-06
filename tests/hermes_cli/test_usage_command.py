"""``hermes usage`` — the non-interactive /usage surface (issue #33094).

Drives the real ``hermes`` argparse entrypoint; only the network fetch is replaced with a snapshot
(``agent.account_usage.fetch_account_usage`` is what ``cmd_usage`` reads at call time).
"""

import json
import sys
from datetime import datetime, timezone
from unittest.mock import patch

from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
from hermes_cli import main as hermes_main

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

