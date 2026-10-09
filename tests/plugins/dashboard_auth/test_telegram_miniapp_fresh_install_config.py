"""Proves the Mini App's fresh-install config surface is genuinely fail-closed.

Not theorized: this copies the REAL cli-config.yaml.example (the literal
file scripts/install.sh and docker/stage2-hook.sh copy to a fresh
~/.hermes/config.yaml) into an isolated HERMES_HOME and drives the actual
plugins.dashboard_auth.telegram_miniapp.register() entry point against it,
exactly as a real install's plugin-loading pass would.
"""
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_CONFIG = PROJECT_ROOT / "cli-config.yaml.example"


class _FakeCtx:
    def __init__(self):
        self.registered = []

    def register_dashboard_auth_provider(self, provider):
        self.registered.append(provider)


@pytest.fixture(autouse=True)
def _seed_fresh_install_config(_isolate_hermes_home):
    """Copy the real example config into this test's isolated HERMES_HOME,
    exactly as scripts/install.sh:1788 / docker/stage2-hook.sh:390 do.

    No cache-busting needed: load_config() caches on (path, mtime, size)
    keyed by the config path, and _isolate_hermes_home gives every test a
    brand-new tmp_path-based HERMES_HOME, so this path has never been read
    before -- nothing stale to invalidate.
    """
    from hermes_constants import get_hermes_home

    assert EXAMPLE_CONFIG.is_file(), f"{EXAMPLE_CONFIG} must exist"
    dest = get_hermes_home() / "config.yaml"
    dest.write_text(EXAMPLE_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    yield


def _register_and_get(monkeypatch, *, bot_token: str | None):
    import plugins.dashboard_auth.telegram_miniapp as miniapp_plugin

    if bot_token is None:
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    else:
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", bot_token)

    ctx = _FakeCtx()
    miniapp_plugin.register(ctx)
    return ctx, miniapp_plugin.LAST_SKIP_REASON


def test_fresh_install_config_has_telegram_miniapp_present_but_disabled():
    """The literal claim: dashboard.telegram_miniapp exists in a fresh
    install's config.yaml, with enabled: false -- not absent, not True."""
    from hermes_cli.config import load_config, cfg_get

    cfg = load_config()
    section = cfg_get(cfg, "dashboard", "telegram_miniapp", default=None)
    assert section is not None, "telegram_miniapp section must be present, not absent"
    assert section["enabled"] is False


def test_fresh_install_env_example_has_admin_users_key_commented_unset():
    """TELEGRAM_DASHBOARD_ADMIN_USERS exists in .env.example, commented
    (unset by default) -- present in the template, inert until an operator
    uncomments and fills it in."""
    env_example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "TELEGRAM_DASHBOARD_ADMIN_USERS" in env_example
    for line in env_example.splitlines():
        stripped = line.strip()
        if "TELEGRAM_DASHBOARD_ADMIN_USERS=" in stripped:
            assert stripped.startswith("#"), "must ship commented-out (unset), not active"
            return
    pytest.fail("TELEGRAM_DASHBOARD_ADMIN_USERS= line not found")


def test_enabled_false_serves_nothing(monkeypatch):
    """Case A: the shipped default (enabled: false). Even with a bot token
    present, the plugin must not register."""
    ctx, skip_reason = _register_and_get(monkeypatch, bot_token="123456:fake-token")
    assert ctx.registered == []
    assert "not set to true" in skip_reason


def test_enabled_true_admin_unset_serves_feature_with_zero_admins(monkeypatch):
    """Case B, the one this whole prep step is about: flip enabled to true
    (simulating an operator opting in) with TELEGRAM_DASHBOARD_ADMIN_USERS
    left unset (the shipped default) and a bot token present. The plugin
    MUST register (feature is live) but MUST NOT fall open -- no user,
    paired or not, may resolve to the admin tier.
    """
    from hermes_constants import get_hermes_home

    config_path = get_hermes_home() / "config.yaml"
    text = config_path.read_text(encoding="utf-8")
    # Targeted, not a blind first-match replace: cli-config.yaml.example has
    # multiple unrelated "enabled: false" lines (e.g. streaming:) -- a naive
    # .replace(..., 1) silently flips the WRONG one and this test would
    # "pass" while proving nothing about telegram_miniapp at all.
    anchor = "telegram_miniapp:\n    enabled: false"
    assert anchor in text, "expected telegram_miniapp block shape not found"
    config_path.write_text(
        text.replace(anchor, "telegram_miniapp:\n    enabled: true", 1), encoding="utf-8"
    )

    monkeypatch.delenv("TELEGRAM_DASHBOARD_ADMIN_USERS", raising=False)

    ctx, skip_reason = _register_and_get(monkeypatch, bot_token="123456:fake-token")
    assert len(ctx.registered) == 1, f"expected the provider to register; skip_reason={skip_reason!r}"

    from plugins.dashboard_auth.telegram_miniapp.tiers import resolve_tier
    from types import SimpleNamespace

    # A genuinely paired/allowlisted user -- the ONLY thing standing between
    # them and admin, with the fix in place, is TELEGRAM_DASHBOARD_ADMIN_USERS.
    paired_store = SimpleNamespace(is_approved=lambda *_a, **_kw: True)
    assert resolve_tier("555001122", pairing_store=paired_store) == "paired"
    assert resolve_tier("999999999", pairing_store=paired_store) == "paired"
