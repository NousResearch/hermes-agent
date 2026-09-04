"""Regression coverage for #88654.

``hermes update`` relaunches manually-run profile gateways through
``_prepare_profile_gateway_update_restart``.  When the profile-derived
relaunch could not be armed the helper returned ``None``, and the update
path's response to ``None`` was a bare ``continue`` -- so the gateway was
neither relaunched, nor stopped, nor mentioned.  It kept serving from
pre-update modules while the new code sat on disk, and every lazy import
from that point mixed versions.

The helper now falls back to replaying the process's own captured command
line via ``launch_detached_gateway_restart_by_cmdline`` -- the companion
that already exists for gateways with no profile mapping, and that the
Windows post-update path already uses for exactly this case.
"""

import pytest

import hermes_cli.gateway as gateway


_ARGV = ["python", "-m", "hermes_cli.main", "gateway", "run"]


def _stub_argv(monkeypatch, argv):
    monkeypatch.setattr(gateway, "_capture_gateway_argv", lambda _pid: argv)


def test_profile_relaunch_wins_and_skips_the_cmdline_replay(monkeypatch):
    """The existing path is unchanged: a profile relaunch short-circuits."""
    _stub_argv(monkeypatch, list(_ARGV))
    monkeypatch.setattr(
        gateway, "launch_detached_profile_gateway_restart", lambda *_a: True
    )
    monkeypatch.setattr(
        gateway,
        "launch_detached_gateway_restart_by_cmdline",
        lambda *_a: pytest.fail("cmdline replay must not run when the profile path works"),
    )

    assert gateway._prepare_profile_gateway_update_restart("fitness", 4242) == "detached"


def test_falls_back_to_cmdline_replay_when_profile_relaunch_fails(monkeypatch):
    """The #88654 fix: an unarmable profile relaunch still gets the gateway back."""
    _stub_argv(monkeypatch, list(_ARGV))
    monkeypatch.setattr(
        gateway, "launch_detached_profile_gateway_restart", lambda *_a: False
    )
    seen = []

    def _by_cmdline(pid, argv):
        seen.append((pid, argv))
        return True

    monkeypatch.setattr(
        gateway, "launch_detached_gateway_restart_by_cmdline", _by_cmdline
    )

    assert (
        gateway._prepare_profile_gateway_update_restart("fitness", 4242)
        == "detached-cmdline"
    )
    # Replays the process's OWN argv, which is the whole point: the profile
    # could not be mapped back to a run argv, so the captured one is the only
    # faithful description of how to restart it.
    assert seen == [(4242, _ARGV)]


def test_returns_none_when_there_is_no_argv_to_replay(monkeypatch):
    """No captured argv means no honest way to relaunch; caller must be told."""
    _stub_argv(monkeypatch, [])
    monkeypatch.setattr(
        gateway, "launch_detached_profile_gateway_restart", lambda *_a: False
    )
    monkeypatch.setattr(
        gateway,
        "launch_detached_gateway_restart_by_cmdline",
        lambda *_a: pytest.fail("must not replay an empty argv"),
    )

    assert gateway._prepare_profile_gateway_update_restart("fitness", 4242) is None


def test_returns_none_when_both_relaunch_paths_fail(monkeypatch):
    """Both mechanisms failing is still reported as None, not a false success."""
    _stub_argv(monkeypatch, list(_ARGV))
    monkeypatch.setattr(
        gateway, "launch_detached_profile_gateway_restart", lambda *_a: False
    )
    monkeypatch.setattr(
        gateway, "launch_detached_gateway_restart_by_cmdline", lambda *_a: False
    )

    assert gateway._prepare_profile_gateway_update_restart("fitness", 4242) is None


def test_external_supervisor_still_short_circuits_before_any_replay(monkeypatch):
    """Guardrail: the supervisor hand-back must not gain a replay behind it.

    Replaying the argv for an externally supervised gateway would escape the
    manager and race its replacement process, which is the exact hazard the
    supervisor branch exists to avoid.
    """
    _stub_argv(monkeypatch, _ARGV + ["--external-supervisor"])
    monkeypatch.setattr(
        gateway,
        "launch_detached_profile_gateway_restart",
        lambda *_a: pytest.fail("detached watcher must not be launched"),
    )
    monkeypatch.setattr(
        gateway,
        "launch_detached_gateway_restart_by_cmdline",
        lambda *_a: pytest.fail("cmdline replay must not be launched"),
    )

    assert (
        gateway._prepare_profile_gateway_update_restart("fitness", 4242)
        == "external-supervisor"
    )
