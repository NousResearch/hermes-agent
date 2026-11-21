"""Cron preflight must refuse a BARE ``deliver: <platform>`` lane that cannot resolve a target.

``_preflight_check_delivery`` asked two questions about a lane — is the platform known, and is it
connected — and neither of them is "can this lane name a destination". A bare platform token
resolves through that platform's home channel (``_get_home_target_chat_id``: env mirror first, then
``platforms.<p>.home_channel``), so a CONNECTED platform with no home channel passes preflight and
then dies at fire time with ``no delivery target resolved for deliver=<platform>``
(``_unresolved_delivery_outcome``). That failure is the worst shape available: the run itself
SUCCEEDED, so no incident opens and nothing alerts — the output is written to disk and read by
nobody.

The probe reuses ``_resolve_delivery_targets`` rather than re-deriving home lookup, so preflight and
delivery cannot disagree about whether a lane is deliverable.

Related precedent: #97476 (routed-satellite rescue — a bare lane must not be blocked merely because
this home holds no credentials; that rescue is preserved for the new arm) and NS-788 (the failure
lane is validated as strictly as ``deliver``).
"""

import textwrap
from unittest.mock import MagicMock, patch

import pytest
import hermes_yaml as yaml

from cron import scheduler_delivery as sched_delivery
from cron.scheduler_preflight import (
    _bare_lane_target_reason,
    _preflight_check_delivery,
)
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _gateway_config(connected_values):
    config = MagicMock()
    config.get_connected_platforms.return_value = [MagicMock(value=v) for v in connected_values]
    return config


@pytest.fixture
def connected_telegram_no_home(tmp_path, monkeypatch):
    """A REAL gateway config naming telegram connected with no home channel.

    No resolution is mocked here: ``load_gateway_config`` reads the config.yaml below, and the
    home-channel lookup runs its real env → config chain. This is the arm that proves the gate and
    delivery agree, not just that a mock was called.
    """
    home = tmp_path / "home"
    home.mkdir()
    config_path = home / "config.yaml"
    config_path.write_text(textwrap.dedent("""
        platforms:
          telegram:
            enabled: true
            token: "111111:AAA-test-token"
        """).strip() + "\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Ensure the env mirror cannot satisfy the home lookup from the host environment.
    for var in ("TELEGRAM_HOME_CHANNEL", "TELEGRAM_CRON_THREAD_ID",
                "TELEGRAM_HOME_CHANNEL_THREAD_ID", "GATEWAY_RELAY_PLATFORMS"):
        monkeypatch.delenv(var, raising=False)
    # Primary home == current home, so the multiplex rescue reports "nothing to consult".
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: home)
    token = set_hermes_home_override(str(home))
    yield home, config_path
    reset_hermes_home_override(token)


class TestBareLaneWithoutHomeBlocks:
    def test_connected_platform_bare_lane_no_home_is_blocked(self, connected_telegram_no_home):
        """The card's shape, on the REAL resolver: connected + bare + no home target → blocked."""
        _home, _config_path = connected_telegram_no_home
        reason = _preflight_check_delivery({"id": "pf-home", "deliver": "telegram"})
        assert reason is not None, (
            "a bare lane on a connected platform with no home target passed preflight; at fire "
            "time this dies with 'no delivery target resolved' on an otherwise-successful run"
        )
        assert "telegram" in reason

    def test_same_job_passes_once_the_home_target_resolves(self, connected_telegram_no_home):
        """Flip the fixture, not the assertion: give telegram a home channel → not blocked.

        This is the other direction of the same measurement, and it is what proves the arm keys on
        resolvability rather than on the platform's identity.
        """
        _home, config_path = connected_telegram_no_home
        config_path.write_text(yaml.safe_dump({
            "platforms": {"telegram": {
                "enabled": True,
                "token": "111111:AAA-test-token",
                "home_channel": {"platform": "telegram", "chat_id": "-100555"},
            }},
        }), encoding="utf-8")
        assert _preflight_check_delivery({"id": "pf-home", "deliver": "telegram"}) is None

    def test_env_mirror_home_target_also_satisfies_the_lane(self, connected_telegram_no_home,
                                                            monkeypatch):
        """The legacy env mirror is a first-class home target, not a second-class one."""
        _home, _config_path = connected_telegram_no_home
        monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "-100777")
        assert _preflight_check_delivery({"id": "pf-home", "deliver": "telegram"}) is None


class TestVerdictNamesTheRealRemedy:
    """AC2: the reason must name the fix that works, not ``hermes setup``."""

    def test_reason_names_the_home_env_var_and_never_hermes_setup(
            self, connected_telegram_no_home):
        _home, _config_path = connected_telegram_no_home
        reason = _preflight_check_delivery({"id": "pf-home", "deliver": "telegram"})
        assert reason is not None
        assert "TELEGRAM_HOME_CHANNEL" in reason
        assert "hermes setup" not in reason, (
            "hermes setup configures credentials, not a home channel — naming it here repeats the "
            "misdirection the credential-failure string already commits"
        )
        assert "/sethome" in reason

    def test_platform_without_a_home_channel_concept_names_the_target_form(self):
        """webhook has no home-channel concept: naming WEBHOOK_HOME_CHANNEL would be a lie."""
        reason = _bare_lane_target_reason("webhook", "")
        assert "webhook" in reason
        assert "HOME_CHANNEL" not in reason
        assert "webhook:<chat_id>" in reason

    def test_home_capable_platform_reason_offers_both_fixes(self):
        reason = _bare_lane_target_reason("telegram", "TELEGRAM_HOME_CHANNEL")
        assert "TELEGRAM_HOME_CHANNEL" in reason
        assert "telegram:<chat_id>" in reason


class TestNoFalseBlocks:
    """Existing accepted lanes must keep passing — the new arm is an addition, not a tightening."""

    def test_explicit_chat_target_needs_no_home(self, connected_telegram_no_home):
        """``platform:chat_id`` addresses a chat directly; a missing home must not block it."""
        _home, _config_path = connected_telegram_no_home
        assert _preflight_check_delivery(
            {"id": "pf-explicit", "deliver": "telegram:-1001234567890"}) is None

    def test_unrelated_token_does_not_mask_a_bare_hole(self, monkeypatch):
        """``telegram,slack:D0ABC``: the explicit slack half must not make the bare half pass."""
        monkeypatch.setattr(
            "gateway.config.load_gateway_config",
            lambda: _gateway_config({"telegram", "slack"}))
        monkeypatch.setattr(
            sched_delivery, "_get_home_target_chat_id",
            lambda p: "D0ABC" if p == "slack" else "")

        def _resolve_target(platform, rest, **kw):
            return rest, None, None

        monkeypatch.setattr("tools.send_message_tool.prepare_send_message_platforms", lambda: None)
        monkeypatch.setattr("tools.send_message_tool.resolve_send_target", _resolve_target)
        monkeypatch.setattr(
            "cron.scheduler_preflight._delivery_platform_routed_from_primary_gateway",
            lambda _p: False)

        reason = _preflight_check_delivery({"id": "pf-mixed", "deliver": "telegram,slack:D0ABC"})
        assert reason is not None and "telegram" in reason

    def test_targeted_lane_with_a_home_passes(self, monkeypatch):
        """Negative control for the arm's own key: give the resolver a target → no block."""
        monkeypatch.setattr(
            "gateway.config.load_gateway_config",
            lambda: _gateway_config({"telegram"}))
        monkeypatch.setattr(
            sched_delivery, "_get_home_target_chat_id", lambda p: "-100999")
        assert _preflight_check_delivery({"id": "pf-ok", "deliver": "telegram"}) is None

    def test_failure_lane_is_checked_too(self, monkeypatch):
        """NS-788 parity: a bare failure lane with no home is refused at preflight."""
        monkeypatch.setattr(
            "gateway.config.load_gateway_config",
            lambda: _gateway_config({"telegram", "discord"}))
        monkeypatch.setattr(
            sched_delivery, "_get_home_target_chat_id", lambda p: "" if p == "discord" else "-1")
        monkeypatch.setattr(
            "cron.scheduler_preflight._delivery_platform_routed_from_primary_gateway",
            lambda _p: False)
        reason = _preflight_check_delivery(
            {"id": "pf-fail", "deliver": "telegram", "failure_deliver": "discord"})
        assert reason is not None and "discord" in reason

    def test_local_and_routing_tokens_are_untouched(self, monkeypatch):
        """``local``/``origin``/``all`` expand at fire time and must never reach the probe."""
        monkeypatch.setattr(
            "gateway.config.load_gateway_config",
            lambda: _gateway_config({"telegram"}))
        monkeypatch.setattr(
            sched_delivery, "_get_home_target_chat_id", lambda p: "")
        for lane in ("local", "origin", "all", "origin,all"):
            assert _preflight_check_delivery({"id": "pf-x", "deliver": lane}) is None


class TestFailOpenAndSatelliteRescue:
    def test_gateway_config_load_failure_still_fails_open(self, monkeypatch):
        """AC3: an unavailable config returns None — only an affirmative verdict blocks."""
        def _boom():
            raise RuntimeError("config.yaml unreadable")

        monkeypatch.setattr("gateway.config.load_gateway_config", _boom)
        assert _preflight_check_delivery({"id": "pf-open", "deliver": "telegram"}) is None

    def test_routed_satellite_bare_lane_is_not_blocked(self, tmp_path, monkeypatch):
        """#97476 preserved for the new arm: the primary gateway owns a routed satellite's adapters.

        The satellite legitimately holds no home channel, so its bare lane resolves to nothing
        locally while still delivering through the primary — the same false block the credential
        branch already rescues.
        """
        home = tmp_path / "profiles" / "grant"
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(textwrap.dedent("""
            platforms:
              telegram:
                enabled: true
                token: "111111:AAA-test-token"
            """).strip() + "\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)
        monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: tmp_path)
        monkeypatch.setattr(
            "cron.scheduler_preflight._delivery_platform_routed_from_primary_gateway",
            lambda p: p == "telegram")
        token = set_hermes_home_override(str(home))
        try:
            assert _preflight_check_delivery(
                {"id": "pf-sat", "deliver": "telegram"}) is None
        finally:
            reset_hermes_home_override(token)

    def test_probe_exception_fails_open(self, monkeypatch):
        """A resolver that raises is an internal error, never a verdict (same contract as the
        sibling checks in ``_preflight_job_config``)."""
        monkeypatch.setattr(
            "gateway.config.load_gateway_config",
            lambda: _gateway_config({"telegram"}))

        def _boom(*_a, **_kw):
            raise RuntimeError("resolver exploded")

        monkeypatch.setattr(sched_delivery, "_resolve_delivery_targets", _boom)
        assert _preflight_check_delivery({"id": "pf-boom", "deliver": "telegram"}) is None
