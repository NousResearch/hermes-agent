"""Regression tests for the "leave empty = open access" gateway setup bug.

The interactive platform setup wizards told operators that leaving the user
allowlist empty meant "open access". The empty branch, however, only printed a
warning and never wrote an allow-all flag. Runtime auth
(``gateway/authz_mixin.py``) requires an explicit allow-all to treat unknown
senders as authorized, so the bot silently stayed in default-deny / DM-pairing
mode — the opposite of what setup advertised.

The fix routes every empty-allowlist branch through
``hermes_cli.setup._configure_open_access_or_pairing``, which offers an explicit
choice and only enables open access when the operator picks it (never a silent
fail-open, per SECURITY.md §2.6). These tests pin that behavior for the shared
helper and for every affected wizard.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli.config import get_env_value


# Every env var these wizards read ("already configured" sentinels) or write.
# ``save_env_value`` also mutates ``os.environ`` directly (config.py), which
# pytest's monkeypatch can't auto-revert, so clear them per-test to stop one
# wizard run from leaking into the next.
_PLATFORM_VARS = [
    "TELEGRAM_ALLOW_ALL_USERS", "DISCORD_ALLOW_ALL_USERS", "MATRIX_ALLOW_ALL_USERS",
    "MATTERMOST_ALLOW_ALL_USERS", "BLUEBUBBLES_ALLOW_ALL_USERS", "GATEWAY_ALLOW_ALL_USERS",
    "TELEGRAM_ALLOWED_USERS", "DISCORD_ALLOWED_USERS", "MATRIX_ALLOWED_USERS",
    "MATTERMOST_ALLOWED_USERS", "BLUEBUBBLES_ALLOWED_USERS",
    "TELEGRAM_BOT_TOKEN", "TELEGRAM_HOME_CHANNEL",
    "DISCORD_BOT_TOKEN", "DISCORD_HOME_CHANNEL",
    "MATTERMOST_TOKEN", "MATTERMOST_URL", "MATTERMOST_HOME_CHANNEL",
    "MATRIX_ACCESS_TOKEN", "MATRIX_PASSWORD", "MATRIX_HOMESERVER", "MATRIX_USER_ID",
    "MATRIX_ENCRYPTION", "MATRIX_HOME_ROOM",
    "BLUEBUBBLES_SERVER_URL", "BLUEBUBBLES_PASSWORD", "BLUEBUBBLES_HOME_CHANNEL",
    "BLUEBUBBLES_WEBHOOK_PORT",
]


@pytest.fixture(autouse=True)
def _isolated_platform_env(monkeypatch):
    """Start each test from a clean slate and never block on stdin.

    Clears the platform env vars (defeating the ``os.environ`` write-through in
    ``save_env_value``) and stubs every ``prompt_yes_no`` to a non-interactive
    ``False`` so optional/advanced wizard branches don't read from stdin.
    """
    import hermes_cli.setup as setup_mod
    import hermes_cli.cli_output as cli_output

    for var in _PLATFORM_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(setup_mod, "prompt_yes_no", lambda *a, **k: False)
    monkeypatch.setattr(cli_output, "prompt_yes_no", lambda *a, **k: False)


# ---------------------------------------------------------------------------
# Shared helper — the heart of the fix.
# ---------------------------------------------------------------------------

ALLOW_ALL_VARS = [
    "TELEGRAM_ALLOW_ALL_USERS",
    "DISCORD_ALLOW_ALL_USERS",
    "MATRIX_ALLOW_ALL_USERS",
    "MATTERMOST_ALLOW_ALL_USERS",
    "BLUEBUBBLES_ALLOW_ALL_USERS",
]


@pytest.mark.parametrize("allow_all_env", ALLOW_ALL_VARS)
def test_helper_open_access_writes_allow_all(monkeypatch, allow_all_env):
    """Picking "open access" (index 0) writes the platform allow-all flag."""
    from hermes_cli import setup as setup_mod

    monkeypatch.setattr(setup_mod, "prompt_choice", lambda *a, **k: 0)
    setup_mod._configure_open_access_or_pairing(allow_all_env)

    assert get_env_value(allow_all_env) == "true"


@pytest.mark.parametrize("choice_idx", [1, 2])
def test_helper_pairing_or_skip_never_fails_open(monkeypatch, choice_idx):
    """Pairing (the default) and Skip must NOT write an allow-all flag.

    This is the security-critical half: an operator who accepts the default or
    skips must never silently get an open bot (SECURITY.md §2.6).
    """
    from hermes_cli import setup as setup_mod

    monkeypatch.setattr(setup_mod, "prompt_choice", lambda *a, **k: choice_idx)
    setup_mod._configure_open_access_or_pairing("TELEGRAM_ALLOW_ALL_USERS")

    assert not get_env_value("TELEGRAM_ALLOW_ALL_USERS")


# ---------------------------------------------------------------------------
# End-to-end wizard wiring — the empty-allowlist branch reaches the helper.
# ---------------------------------------------------------------------------


def _make_prompt():
    """A stand-in for the interactive ``prompt`` that drives each wizard to the
    empty-allowlist branch: supplies required URLs/tokens, leaves the allowlist
    (and every optional field) empty."""

    def _prompt(question, default=None, password=False):
        q = str(question).lower()
        if "choice" in q:           # Telegram "Choice [1/2]" — pick automatic.
            return "1"
        if "url" in q:              # homeserver / server URL fields.
            return "https://example.test"
        if "token" in q or "password" in q:
            return "secret-token"
        return ""                   # allowlist + home channel + advanced → empty.

    return _prompt


def _run_telegram(monkeypatch):
    from hermes_cli import setup as setup_mod

    monkeypatch.setattr(setup_mod, "prompt", _make_prompt())
    monkeypatch.setattr(
        setup_mod,
        "_setup_telegram_auto_result",
        lambda: SimpleNamespace(token="123456789:token", owner_user_id=None),
    )
    monkeypatch.setattr(setup_mod, "_is_valid_telegram_bot_token", lambda *a, **k: True)
    setup_mod._setup_telegram()


def _run_bluebubbles(monkeypatch):
    from hermes_cli import setup as setup_mod

    monkeypatch.setattr(setup_mod, "prompt", _make_prompt())
    setup_mod._setup_bluebubbles()


def _run_discord(monkeypatch):
    import hermes_cli.cli_output as cli_output
    from plugins.platforms.discord import adapter

    monkeypatch.setattr(cli_output, "prompt", _make_prompt())
    adapter.interactive_setup()


def _run_mattermost(monkeypatch):
    import hermes_cli.cli_output as cli_output
    from plugins.platforms.mattermost import adapter

    monkeypatch.setattr(cli_output, "prompt", _make_prompt())
    adapter.interactive_setup()


def _run_matrix(monkeypatch):
    import hermes_cli.cli_output as cli_output
    import tools.lazy_deps as lazy_deps
    from plugins.platforms.matrix import adapter

    monkeypatch.setattr(cli_output, "prompt", _make_prompt())
    # Don't attempt the mautrix install when the access-token branch runs.
    monkeypatch.setattr(lazy_deps, "feature_missing", lambda *a, **k: ())
    adapter.interactive_setup()


WIZARDS = [
    ("telegram", "TELEGRAM_ALLOW_ALL_USERS", _run_telegram),
    ("discord", "DISCORD_ALLOW_ALL_USERS", _run_discord),
    ("matrix", "MATRIX_ALLOW_ALL_USERS", _run_matrix),
    ("mattermost", "MATTERMOST_ALLOW_ALL_USERS", _run_mattermost),
    ("bluebubbles", "BLUEBUBBLES_ALLOW_ALL_USERS", _run_bluebubbles),
]


@pytest.mark.parametrize(
    "name, allow_all_env, runner", WIZARDS, ids=[w[0] for w in WIZARDS]
)
def test_wizard_open_access_enables_allow_all(monkeypatch, name, allow_all_env, runner):
    """Empty allowlist + explicit "open access" → allow-all flag is written."""
    from hermes_cli import setup as setup_mod

    monkeypatch.setattr(setup_mod, "prompt_choice", lambda *a, **k: 0)
    runner(monkeypatch)

    assert get_env_value(allow_all_env) == "true", name


@pytest.mark.parametrize(
    "name, allow_all_env, runner", WIZARDS, ids=[w[0] for w in WIZARDS]
)
def test_wizard_pairing_default_does_not_fail_open(monkeypatch, name, allow_all_env, runner):
    """Empty allowlist + the default (DM pairing) → no allow-all flag.

    Reproduces the original bug's safe side: leaving the allowlist empty must
    not silently open the bot. Before the fix the wizard wrote nothing here too,
    but it also wrote nothing for the *open access* case — so "open access" was
    a broken promise. The companion test above covers that half.
    """
    from hermes_cli import setup as setup_mod

    monkeypatch.setattr(setup_mod, "prompt_choice", lambda *a, **k: 1)
    runner(monkeypatch)

    assert not get_env_value(allow_all_env), name
