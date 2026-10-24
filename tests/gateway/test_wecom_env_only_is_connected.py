"""The WeCom connectivity checkers must answer for the credentials each adapter reads.

Same class of defect as the Feishu plugin (#120870): ``hermes gateway setup``'s picker
(``hermes_cli.gateway_setup_wizard._platform_status``) hands every registry platform a
synthetic ``PlatformConfig(enabled=True)`` with an empty ``extra``, so a checker that
consults ``extra`` alone reports "not configured" for the env-only shape the wizard writes.

The AI Bot and Callback modes differ in what their adapters can actually start from, and
the checkers must match each one — a checker that out-runs its adapter lets the picker
claim a config that then fails to boot:

* ``WeComAdapter._setting`` reads ``extra`` then ``WECOM_BOT_ID``/``WECOM_SECRET`` from the
  owning profile's ``.env``, and ``connect()`` refuses to start without **both**
  (``wecom_missing_credentials``). Widening is honest.
* ``WecomCallbackAdapter._normalize_apps(extra)`` has no env rung at all, so its checker
  must stay ``extra``-only until the adapter gains one.
"""

import pytest

from gateway.config import PlatformConfig


@pytest.fixture(autouse=True)
def _clean_wecom_env(monkeypatch):
    """No ambient credentials: each case states its own."""
    for name in (
        "WECOM_BOT_ID",
        "WECOM_SECRET",
        "WECOM_CALLBACK_CORP_ID",
        "WECOM_CALLBACK_CORP_SECRET",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def checkers():
    pytest.importorskip("plugins.platforms.wecom.adapter")
    from plugins.platforms.wecom.adapter import _callback_is_connected, _is_connected

    return _is_connected, _callback_is_connected


# --------------------------------------------------------------------------- AI Bot mode


def test_bot_env_only_reads_connected_with_empty_config(checkers, monkeypatch):
    """The wizard's synthetic config + .env credentials must read as configured."""
    bot_check, _ = checkers
    monkeypatch.setenv("WECOM_BOT_ID", "bot_env_only")
    monkeypatch.setenv("WECOM_SECRET", "secret_env_only")

    assert bot_check(PlatformConfig(enabled=True)) is True


def test_bot_yaml_extra_still_reads_connected(checkers):
    """The pre-existing YAML path (credentials in config.extra) must keep working."""
    bot_check, _ = checkers

    assert bot_check(PlatformConfig(enabled=True, extra={"bot_id": "bot_yaml", "secret": "s"})) is True


def test_bot_env_beats_yaml_like_the_adapter(checkers, monkeypatch):
    """Env wins in the shared reader, so the checker reports the adapter's view."""
    bot_check, _ = checkers
    monkeypatch.setenv("WECOM_BOT_ID", "bot_from_env")
    monkeypatch.setenv("WECOM_SECRET", "secret_from_env")

    assert bot_check(PlatformConfig(enabled=True, extra={"bot_id": "", "secret": ""})) is True


def test_bot_half_configured_is_not_connected(checkers, monkeypatch):
    """``connect()`` raises ``wecom_missing_credentials`` without a secret — not ready."""
    bot_check, _ = checkers
    monkeypatch.setenv("WECOM_BOT_ID", "bot_half")

    assert bot_check(PlatformConfig(enabled=True, extra={"bot_id": "bot_half"})) is False


def test_bot_unconfigured_is_not_connected(checkers):
    """No env, no extra: still honestly "not configured"."""
    bot_check, _ = checkers

    assert bot_check(PlatformConfig(enabled=True)) is False


def test_bot_blank_values_are_unset(checkers, monkeypatch):
    """Blank env / blank YAML count as unset — never as credentials."""
    bot_check, _ = checkers
    monkeypatch.setenv("WECOM_BOT_ID", "   ")
    monkeypatch.setenv("WECOM_SECRET", "")

    assert bot_check(PlatformConfig(enabled=True, extra={"bot_id": "  ", "secret": ""})) is False


def test_bot_checker_agrees_with_adapter_settings(checkers, monkeypatch):
    """Agreement invariant: checker True means the adapter resolves both halves from env."""
    bot_check, _ = checkers
    monkeypatch.setenv("WECOM_BOT_ID", "bot_agreement")
    monkeypatch.setenv("WECOM_SECRET", "secret_agreement")

    from plugins.platforms.wecom.adapter import WeComAdapter

    adapter = WeComAdapter(PlatformConfig(enabled=True))
    assert bot_check(PlatformConfig(enabled=True)) is True
    assert adapter._bot_id and adapter._secret  # the invariant under test


# ---------------------------------------------------------------------- Callback mode


def test_callback_checker_stays_extra_only(checkers, monkeypatch):
    """Callback's adapter has no env rung, so its checker must not answer for env.

    Widening this one alone is the disagreement that stalled #120884: the picker would show
    "configured" while ``_normalize_apps(extra)`` yields no apps and ``connect()`` bails
    with "No callback apps configured".
    """
    _, callback_check = checkers
    monkeypatch.setenv("WECOM_CALLBACK_CORP_ID", "corp_env_only")
    monkeypatch.setenv("WECOM_CALLBACK_CORP_SECRET", "secret_env_only")

    assert callback_check(PlatformConfig(enabled=True)) is False


def test_callback_yaml_extra_reads_connected(checkers):
    """The YAML path the callback adapter does read must still report configured."""
    _, callback_check = checkers

    assert callback_check(PlatformConfig(enabled=True, extra={"corp_id": "corp_yaml"})) is True


def test_callback_apps_block_reads_connected(checkers):
    """Multi-app installs are configured by the `apps` block."""
    _, callback_check = checkers

    config = PlatformConfig(enabled=True, extra={"apps": [{"name": "a", "corp_id": "c"}]})

    assert callback_check(config) is True


def test_callback_agrees_with_adapter_apps(checkers):
    """Agreement invariant: checker False means the adapter would start with no apps."""
    _, callback_check = checkers

    from plugins.platforms.wecom.callback_adapter import WecomCallbackAdapter

    config = PlatformConfig(enabled=True)
    assert callback_check(config) is False
    assert WecomCallbackAdapter._normalize_apps(config.extra or {}) == []
