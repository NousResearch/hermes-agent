"""Lifecycle guard must match the gateway's own service, not any label containing
the substring (#124700, Case 1).

The #30719 SIGTERM-respawn-loop guard anchors on ``hermes[.-]gateway`` with a word
boundary, so retiring an UNRELATED service whose label merely contains the prefix
(``ai.hermes.gateway-watchdog``) is blocked as if it were the gateway. #124716
handles the ``bash -n`` parse-only exemption and the cron prompt scan (Cases 2-3);
this keeps Case 1: match the bare label identity, and decide suffixed labels
against this install's real service set.
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from cron.lifecycle_guard import contains_gateway_lifecycle_command  # noqa: E402


def _suffixes(names):
    return lambda: frozenset(names)


def test_label_that_contains_gateway_prefix_is_allowed(monkeypatch):
    """`ai.hermes.gateway-watchdog` is a DIFFERENT service whose label merely
    contains the substring; retiring it must not trip the guard."""
    monkeypatch.setattr(
        "cron.lifecycle_guard._install_gateway_label_suffixes", _suffixes({"work"})
    )
    for command in (
        "launchctl bootout gui/$(id -u)/ai.hermes.gateway-watchdog",
        "launchctl bootout gui/$(id -u)/ai.hermes.gateway-watcher-helper",
        "launchctl kickstart -k gui/501/ai.hermes.gateway-monitor",
    ):
        assert not contains_gateway_lifecycle_command(command), command


def test_bare_identity_and_file_spellings_stay_blocked():
    """The gateway's actual label — bare, plist/service file spelling — is the
    #30719 foot-gun and stays blocked without needing any enumeration."""
    for blocked in (
        "launchctl bootout gui/$(id -u)/ai.hermes.gateway",
        "launchctl bootout gui/501/ai.hermes.gateway.plist",
        "launchctl unload ~/Library/LaunchAgents/ai.hermes.gateway.plist",
        "launchctl bootout gui/501/hermes-gateway",
        "systemctl stop hermes-gateway.service",
        "hermes gateway restart",
    ):
        assert contains_gateway_lifecycle_command(blocked), blocked


def test_suffixed_label_of_this_install_is_blocked(monkeypatch):
    """`ai.hermes.gateway-work` where `work` IS a profile of this install is the
    gateway's own service: blocked. The over-match fix must not open it."""
    monkeypatch.setattr(
        "cron.lifecycle_guard._install_gateway_label_suffixes",
        _suffixes({"profile", "work"}),
    )
    for command in (
        "launchctl bootout gui/$(id -u)/ai.hermes.gateway-work",
        "systemctl stop hermes-gateway-profile.service",
    ):
        assert contains_gateway_lifecycle_command(command), command


def test_suffixed_label_requires_a_lifecycle_command(monkeypatch):
    """Mentioning this install's label is harmless unless paired with a lifecycle verb."""
    monkeypatch.setattr(
        "cron.lifecycle_guard._install_gateway_label_suffixes", _suffixes({"work"})
    )
    for prose in (
        "echo ai.hermes.gateway-work",
        'git commit -m "fix ai.hermes.gateway-work handling"',
        "systemctl status ai.hermes.gateway-work",
    ):
        assert not contains_gateway_lifecycle_command(prose), prose
    for lifecycle_command in (
        "launchctl bootout gui/501/ai.hermes.gateway-work",
        "systemctl stop ai.hermes.gateway-work.service",
    ):
        assert contains_gateway_lifecycle_command(lifecycle_command), lifecycle_command


def test_suffixed_label_of_another_service_is_allowed(monkeypatch):
    """Same spellings, but the suffix belongs to no profile of this install:
    a different service — allowed (the over-match this PR fixes)."""
    monkeypatch.setattr(
        "cron.lifecycle_guard._install_gateway_label_suffixes",
        _suffixes({"profile"}),
    )
    for command in (
        "launchctl bootout gui/$(id -u)/ai.hermes.gateway-work",
        "systemctl stop hermes-gateway-legacy-tool.service",
    ):
        assert not contains_gateway_lifecycle_command(command), command


def test_legacy_hash_suffix_still_blocked(monkeypatch):
    """Pre-profile-scheme installs use `ai.hermes.gateway-<8hex>` labels
    (hermes_cli.gateway.legacy_launchd_labels_for_install); the suffix grammar
    keeps them blocked without any enumeration."""
    monkeypatch.setattr(
        "cron.lifecycle_guard._install_gateway_label_suffixes", _suffixes(set())
    )
    assert contains_gateway_lifecycle_command(
        "launchctl bootout gui/$(id -u)/ai.hermes.gateway-1a2b3c4d"
    )


def test_enumeration_failure_fails_closed(monkeypatch):
    """If the install's label set cannot be computed, suffixed labels keep the
    old behavior (blocked) — the guard never widens on an error."""
    monkeypatch.setattr(
        "cron.lifecycle_guard._install_gateway_label_suffixes", lambda: None
    )
    assert contains_gateway_lifecycle_command(
        "launchctl bootout gui/$(id -u)/ai.hermes.gateway-watchdog"
    )


def test_spliced_suffix_cannot_bypass(monkeypatch):
    """Quote-spliced suffixed label of this install still blocks (the tokenized
    pass resolves quotes before the membership check)."""
    monkeypatch.setattr(
        "cron.lifecycle_guard._install_gateway_label_suffixes", _suffixes({"work"})
    )
    assert contains_gateway_lifecycle_command(
        'launchctl bootout gui/$(id -u)/ai.hermes."ga"teway-work'
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
