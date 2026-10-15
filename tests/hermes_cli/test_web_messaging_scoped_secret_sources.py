"""Profile-scoped Messaging cards read the profile's own .env plus its external secret
sources — never the dashboard process's os.environ (which carries the ROOT install's
.env). op://-style refs resolve into a per-home snapshot (``get_secret_source_values``),
not into .env, so a connected bot whose token comes from 1Password used to show up as
Disabled / "Needs setup" while it was answering in the channel (#133442). The snapshot is
the same source ``get_secret`` resolves through, so the card now agrees with the adapter.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

TOKEN = "x" * 72


@pytest.fixture
def quiet_liveness(monkeypatch):
    from hermes_cli.web_routers import messaging

    # No live gateway in these tests: the card under test derives from config only.
    monkeypatch.setattr(
        messaging,
        "resolve_gateway_liveness",
        lambda **kw: SimpleNamespace(running=False),
    )
    monkeypatch.setattr(messaging, "load_config", lambda: {})
    return messaging


def _entry():
    return {
        "id": "discord",
        "name": "Discord",
        "description": "",
        "docs_url": "",
        "env_vars": ["DISCORD_BOT_TOKEN"],
        "required_env": ["DISCORD_BOT_TOKEN"],
    }


def _env_field(payload, key):
    return next(f for f in payload["env_vars"] if f["key"] == key)


def test_scoped_card_counts_a_secret_source_credential_as_configured(quiet_liveness):
    """op://-mapped token, absent from the profile's .env: the card must read the bot as
    enabled+configured, and the Test-connection path (same payload) stops claiming
    "Missing required setup"."""
    from hermes_cli.web_routers import messaging

    payload = messaging._messaging_platform_payload(
        _entry(),
        env_on_disk={},
        runtime={},
        scoped=True,
        profile_home=None,
        secret_values={"DISCORD_BOT_TOKEN": TOKEN},
    )
    assert payload["configured"] is True
    assert payload["enabled"] is True
    assert _env_field(payload, "DISCORD_BOT_TOKEN")["is_set"] is True
    assert _env_field(payload, "DISCORD_BOT_TOKEN")["redacted_value"]


def test_scoped_card_stays_unconfigured_without_either_source(quiet_liveness):
    from hermes_cli.web_routers import messaging

    payload = messaging._messaging_platform_payload(
        _entry(),
        env_on_disk={},
        runtime={},
        scoped=True,
        profile_home=None,
        secret_values={},
    )
    assert payload["configured"] is False
    assert payload["enabled"] is False
    assert _env_field(payload, "DISCORD_BOT_TOKEN")["is_set"] is False


def test_dotenv_value_still_wins_over_the_secret_source(quiet_liveness):
    """A key present in both surfaces reports the .env value — the snapshot only fills gaps."""
    from hermes_cli.web_routers import messaging

    payload = messaging._messaging_platform_payload(
        _entry(),
        env_on_disk={"DISCORD_BOT_TOKEN": "from-dotenv"},
        runtime={},
        scoped=True,
        profile_home=None,
        secret_values={"DISCORD_BOT_TOKEN": TOKEN},
    )
    assert payload["configured"] is True
    assert _env_field(payload, "DISCORD_BOT_TOKEN")[
        "redacted_value"
    ] == messaging.redacted_credential_preview("from-dotenv")


def test_unscoped_cards_do_not_consult_the_snapshot(quiet_liveness, monkeypatch):
    """Unscoped keeps its own ladder (.env, then this process's os.environ); the snapshot
    is a per-profile surface and must not leak into the default view."""
    from hermes_cli.web_routers import messaging

    def _boom(home):
        raise AssertionError(f"snapshot read on the unscoped path: {home}")

    monkeypatch.setattr(messaging, "get_secret_source_values", _boom)
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "from-process-env")
    payload = messaging._messaging_platform_payload(
        _entry(),
        env_on_disk={},
        runtime={},
        scoped=False,
        profile_home=None,
        secret_values={"DISCORD_BOT_TOKEN": TOKEN},
    )
    assert _env_field(payload, "DISCORD_BOT_TOKEN")["is_set"] is True


def test_platform_payloads_read_the_scoped_homes_snapshot(
    tmp_path, quiet_liveness, monkeypatch
):
    """``_platform_payloads`` is the single entry both /platforms and /test use: scoped, it
    must fetch THIS home's snapshot and thread it into enablement; unscoped it must not."""
    from hermes_cli.web_routers import messaging

    alpha = tmp_path / "profiles" / "alpha"
    alpha.mkdir(parents=True)
    monkeypatch.setattr(messaging, "load_env", lambda: {})
    monkeypatch.setattr(messaging, "read_runtime_status", lambda path=None: {})
    monkeypatch.setattr(
        messaging,
        "get_secret_source_values",
        lambda home: {"DISCORD_BOT_TOKEN": TOKEN} if Path(home) == alpha else {},
    )

    [scoped_payload] = messaging._platform_payloads(alpha, [_entry()])
    assert scoped_payload["configured"] is True
    assert _env_field(scoped_payload, "DISCORD_BOT_TOKEN")["is_set"] is True

    [unscoped_payload] = messaging._platform_payloads(None, [_entry()])
    assert unscoped_payload["configured"] is False
    assert _env_field(unscoped_payload, "DISCORD_BOT_TOKEN")["is_set"] is False
