"""``hermes usage`` shows every account of every signed-in provider, and says why a blank one is blank.

Drives the real ``hermes`` entrypoint against a temp ``HERMES_HOME`` holding an Anthropic pool;
only the provider's usage endpoint is replaced. The idle seat's access token has expired, so a
read-only usage fetch with it fails the way the real endpoint does (401).
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from agent import account_usage, anthropic_credentials
from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
from agent.account_usage_cache import _snapshots
from hermes_cli import main as hermes_main


@pytest.fixture(autouse=True)
def _empty_usage_cache():
    _snapshots.clear()
    yield
    _snapshots.clear()


def _seat(number: int, access_token: str, *, expires_in_s: float, **extra) -> dict:
    return {
        "id": f"seat{number}", "label": f"seat-{number}", "auth_type": "oauth", "priority": number,
        "source": "manual:hermes_pkce", "access_token": access_token, "refresh_token": f"refresh-{number}",
        "expires_at_ms": int((time.time() + expires_in_s) * 1000), **extra,
    }


def _api_key_row(provider: str, number: int) -> dict:
    return {"id": f"{provider}{number}", "label": f"{provider}-{number}", "auth_type": "api_key",
            "priority": number, "source": "manual", "access_token": f"{provider}-key-{number}"}


def _usage(provider: str, **labels) -> AccountUsageSnapshot:
    soon = datetime.now(UTC) + timedelta(hours=2)
    return AccountUsageSnapshot(
        provider=provider, source="usage_api", fetched_at=datetime.now(UTC), **labels, windows=(
            AccountUsageWindow("Current session", used_percent=51.0, reset_at=soon),
            AccountUsageWindow("Current week", used_percent=12.0, reset_at=soon + timedelta(days=5)),
        ))


def _capped(provider: str) -> AccountUsageSnapshot:
    week_reset = datetime.now(UTC) + timedelta(days=2)
    return AccountUsageSnapshot(
        provider=provider, source="usage_api", fetched_at=datetime.now(UTC), plan="Max 20x",
        email="capped@example.com", windows=(
            AccountUsageWindow("Current session", used_percent=0.0),
            AccountUsageWindow("Current week", used_percent=100.0, reset_at=week_reset),
        ))


_TWO_SEATS = [
    _seat(1, "busy-access-token", expires_in_s=3 * 3600),
    _seat(2, "idle-access-token", expires_in_s=-3600),
]


@pytest.fixture
def pool_home(tmp_path, monkeypatch):
    """Returns ``write(pool)``: stores *pool* as the credential pool and returns the list of access
    tokens the Anthropic usage endpoint is then called with."""
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text("model:\n  provider: anthropic\n  default: claude-opus-4-7\n")
    used_tokens: list = []

    def anthropic_usage(base_url, api_key):
        used_tokens.append(api_key)
        if api_key == "idle-access-token":
            raise RuntimeError("HTTP 401: token expired")
        if api_key == "capped-access-token":
            return _capped("anthropic")
        return _usage("anthropic", plan="Max 5x", email=f"{api_key.split('-')[0]}@example.com")

    # A Claude Code login on the machine (~/.claude/.credentials.json) would join the pool as one more account.
    monkeypatch.setattr(anthropic_credentials, "read_claude_code_credentials", lambda: None)
    monkeypatch.setitem(account_usage._USAGE_FETCHERS, "anthropic", anthropic_usage)
    monkeypatch.setitem(account_usage._USAGE_FETCHERS, "openrouter", lambda base_url, api_key: _usage("openrouter"))
    # The command renews expired tokens first. By default the token endpoint is unreachable (a transient
    # failure); a test that needs a renewal replaces this.
    monkeypatch.setattr(anthropic_credentials, "refresh_anthropic_oauth_pure", _unreachable_token_endpoint)

    def write(pool: dict) -> list:
        (home / "auth.json").write_text(json.dumps({"version": 1, "credential_pool": pool}))
        return used_tokens

    return write


def _unreachable_token_endpoint(refresh_token, *, use_json=False):
    raise RuntimeError("token endpoint unreachable")


def _renew(refreshed: list):
    def refresh(refresh_token, *, use_json=False):
        refreshed.append(refresh_token)
        return {"access_token": "renewed-access-token", "refresh_token": f"{refresh_token}-next",
                "expires_at_ms": int(time.time() * 1000) + 3600_000}
    return refresh


def _stored(provider: str = "anthropic") -> list[dict]:
    return json.loads((Path(os.environ["HERMES_HOME"]) / "auth.json").read_text())["credential_pool"][provider]


def _usage_cmd(capsys, *argv: str) -> tuple[int, str, str]:
    with patch.object(hermes_main, "_plugin_cli_discovery_needed", return_value=False), \
         patch.object(sys, "argv", ["hermes", "usage", *argv]):
        try:
            hermes_main.main()
            code = 0
        except SystemExit as exc:
            code = int(exc.code or 0)
    out, err = capsys.readouterr()
    return code, out, err


def _table(text: str) -> dict[str, list[str]]:
    """Table rows keyed by account label, split on the column gaps."""
    import re

    lines = text.splitlines()
    return {cells[1]: cells for cells in (re.split(r"\s{2,}", line.strip()) for line in lines[1:lines.index("")])}


def test_the_table_lists_every_account_of_every_provider_usable_first(pool_home, capsys):
    capped = _seat(3, "capped-access-token", expires_in_s=3600)
    pool_home({"anthropic": [capped, *_TWO_SEATS], "openrouter": [_api_key_row("openrouter", 1)],
               "zai": [_api_key_row("zai", 1)]})

    code, text, _ = _usage_cmd(capsys)
    rows = _table(text)

    assert code == 0
    assert text.splitlines()[0].split() == ["Provider", "Account", "ID", "Email", "Plan", "5h", "Week", "Week",
                                            "resets", "Usable", "Note"]
    assert list(rows) == ["seat-1", "seat-2", "seat-3", "openrouter-1", "zai-1"]
    assert rows["seat-1"] == ["anthropic", "seat-1", "seat1", "busy@example.com", "Max 5x", "51%", "12%",
                              rows["seat-1"][7], "yes"]
    assert rows["seat-2"][-2:] == ["?", "token renewal failed: try again"]
    assert rows["seat-3"][5:7] == ["0%", "100%"] and rows["seat-3"][-2] == "no"
    assert rows["seat-3"][-1].startswith("back ")
    assert rows["zai-1"][-2:] == ["?", "provider reports no usage"]
    assert "anthropic: 1 of 3 accounts usable, 1 unknown" in text


def test_a_capped_account_whose_renewal_fails_says_why_its_row_is_blank(pool_home, capsys):
    reset_at = time.time() + 2 * 24 * 3600
    capped = _seat(2, "idle-access-token", expires_in_s=-3600, last_status="exhausted",
                   last_status_at=time.time() - 3600, last_error_code=429, last_error_reset_at=reset_at)
    pool_home({"anthropic": [_TWO_SEATS[0], capped]})

    code, text, _ = _usage_cmd(capsys)
    note = _table(text)["seat-2"][-1]

    assert code == 0
    assert note.startswith("back ") and note.endswith(", token renewal failed: try again")
    assert (_stored()[1]["last_status"], _stored()[1]["last_error_reset_at"]) == ("exhausted", reset_at)


def test_a_failed_renewal_does_not_bench_a_healthy_account(pool_home, capsys):
    pool_home({"anthropic": _TWO_SEATS})

    code, _, _ = _usage_cmd(capsys)

    assert code == 0
    assert [row.get("last_status") for row in _stored()] == [None, None]


def test_provider_narrows_the_table_to_one_provider(pool_home, capsys):
    pool_home({"anthropic": _TWO_SEATS, "openrouter": [_api_key_row("openrouter", 1)]})

    code, text, _ = _usage_cmd(capsys, "--provider", "openrouter")

    assert code == 0 and list(_table(text)) == ["openrouter-1"]


def test_every_pooled_account_is_listed_with_all_its_windows(pool_home, capsys):
    dead = _seat(3, "dead-access-token", expires_in_s=3600, last_status="dead", last_status_at=time.time())
    pool_home({"anthropic": [*_TWO_SEATS, dead]})

    code, text, _ = _usage_cmd(capsys, "--details")

    assert code == 0
    assert "1 of 3 accounts ready · 1 unknown · 1 unavailable" in text
    busy = text[text.index("seat-1"):text.index("seat-2")]
    assert "Current session: 49% remaining" in busy and "Current week: 88% remaining" in busy
    assert "token expires in" in busy
    assert "token expired" in text[text.index("seat-2"):text.index("seat-3")]
    assert "hermes auth add anthropic" in text[text.index("seat-3"):]


def test_json_keeps_the_single_account_keys_and_adds_the_accounts(pool_home, capsys):
    pool_home({"anthropic": _TWO_SEATS})

    code, out, _ = _usage_cmd(capsys, "--json")
    document = json.loads(out)

    assert code == 0
    assert {"provider", "source", "title", "plan", "fetched_at", "windows", "details", "unavailable_reason"} <= set(document)
    assert [(a["label"], a["state"]) for a in document["accounts"]] == [("seat-1", "ready"), ("seat-2", "unknown")]
    assert datetime.fromisoformat(document["accounts"][1]["token_expires_at"]) < datetime.now(UTC)
    assert document["pool"] == {"ready": 1, "limited": 0, "unknown": 1, "unavailable": 0, "total": 2}


def test_only_the_expired_token_is_renewed_and_that_account_reports_usage(pool_home, capsys, monkeypatch):
    pool_home({"anthropic": _TWO_SEATS})
    refreshed: list[str] = []
    monkeypatch.setattr(anthropic_credentials, "refresh_anthropic_oauth_pure", _renew(refreshed))

    code, text, _ = _usage_cmd(capsys, "--details")

    assert code == 0 and refreshed == ["refresh-2"]
    assert "2 of 2 accounts ready" in text


def test_a_benched_account_is_renewed_and_keeps_its_cooldown(pool_home, capsys, monkeypatch):
    # A capped account idles until its weekly reset, so its token expires while it is benched.
    reset_at = time.time() + 2 * 24 * 3600
    benched = _seat(2, "idle-access-token", expires_in_s=-3600, last_status="exhausted",
                    last_status_at=time.time() - 3600, last_error_code=429, last_error_reset_at=reset_at)
    pool_home({"anthropic": [_TWO_SEATS[0], benched]})
    monkeypatch.setattr(anthropic_credentials, "refresh_anthropic_oauth_pure", _renew([]))

    code, text, _ = _usage_cmd(capsys)

    assert code == 0
    assert _stored()[1]["access_token"] == "renewed-access-token"
    assert (_stored()[1]["last_status"], _stored()[1]["last_error_reset_at"]) == ("exhausted", reset_at)
    assert "renewal failed" not in text


def test_a_read_leaves_the_pool_alone_when_no_token_has_expired(pool_home, capsys):
    # An elapsed cooldown is what a session's pre-request sweep would clear and persist.
    rested = _seat(2, "rested-access-token", expires_in_s=3 * 3600,
                   last_status="exhausted", last_status_at=time.time() - 7 * 24 * 3600)
    pool_home({"anthropic": [_TWO_SEATS[0], rested]})

    code, _, _ = _usage_cmd(capsys)

    assert code == 0
    assert [row.get("last_status") for row in _stored()] == [None, "exhausted"]


def test_a_seat_without_a_token_is_never_read_with_another_credential(pool_home, capsys):
    used_tokens = pool_home({"anthropic": [*_TWO_SEATS, _seat(3, "", expires_in_s=3600)]})

    code, out, _ = _usage_cmd(capsys, "--json")

    assert code == 0
    assert sorted(used_tokens) == ["busy-access-token", "idle-access-token"]
    assert json.loads(out)["accounts"][2]["state"] == "unknown"


def test_details_and_json_show_other_pooled_providers_only_with_all(pool_home, capsys):
    pool_home({"anthropic": _TWO_SEATS, "openrouter": [_api_key_row("openrouter", 1)]})

    _, text, _ = _usage_cmd(capsys, "--details")
    _, out, _ = _usage_cmd(capsys, "--all", "--json")

    assert "seat-1" in text and "openrouter" not in text
    assert [document["provider"] for document in json.loads(out)["providers"]] == ["anthropic", "openrouter"]


def test_all_json_keeps_its_envelope_for_a_single_provider(pool_home, capsys):
    pool_home({"anthropic": _TWO_SEATS})

    code, out, _ = _usage_cmd(capsys, "--all", "--json")

    assert code == 0
    assert [document["provider"] for document in json.loads(out)["providers"]] == ["anthropic"]


def test_a_pool_without_a_usage_endpoint_still_reports_no_usage(pool_home, capsys):
    pool_home({"zai": [_api_key_row("zai", 1), _api_key_row("zai", 2)]})

    code, out, err = _usage_cmd(capsys, "--provider", "zai")

    assert code == 1 and out == ""
    assert err.count("\n") == 1 and "zai" in err
