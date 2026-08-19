"""Tests for ``hermes status`` surfacing the "plist exists but launchd has
unloaded the service" state on macOS.

Reproduction:
- ``~/Library/LaunchAgents/ai.hermes.gateway.plist`` is intact
- ``launchctl list | grep ai.hermes.gateway`` returns nothing
- ``hermes status`` previously reported the gateway as running because the
  ``serve`` (dashboard API) process was still alive — users only noticed
  the outage when their Telegram bot stopped responding.

Fix: in the macOS branch of the Gateway section, when the plist is present
but ``_probe_launchd_service_running()`` returns False, print a yellow ⚠
warning plus the exact ``launchctl bootstrap`` command to repair.

See hermes_cli/status.py (gateway service rendering block). Related upstream
issues: #55441, #28632, #42675 family.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.status import show_status


# ---------------------------------------------------------------------------
# Snapshot factories
# ---------------------------------------------------------------------------


def _snapshot(
    *,
    running: bool = False,
    manager: str = "launchd",
    pids: tuple = (),
    service_installed: bool = True,
    service_running: bool = False,
    mismatch: bool = False,
):
    return SimpleNamespace(
        running=running,
        manager=manager,
        gateway_pids=pids,
        service_installed=service_installed,
        service_running=service_running,
        has_process_service_mismatch=mismatch,
    )


# ---------------------------------------------------------------------------
# Test fixtures — patch all the heavy stuff so status() runs end-to-end
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_status_env(monkeypatch, tmp_path):
    """Make ``show_status`` runnable in a unit test without touching real
    auth, providers, or gateway subprocesses. Returns a context manager
    the individual tests can use to override gateway state."""

    from hermes_cli import status as status_mod
    import hermes_cli.auth as auth_mod
    import hermes_cli.gateway as gateway_mod

    # HERMES_HOME for any config-loading paths
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(status_mod, "get_env_path", lambda: tmp_path / ".env", raising=False)
    monkeypatch.setattr(status_mod, "get_hermes_home", lambda: tmp_path, raising=False)
    monkeypatch.setattr(status_mod, "load_config", lambda: {"model": "gpt-5.4"}, raising=False)
    monkeypatch.setattr(status_mod, "resolve_requested_provider", lambda requested=None: "minimax-cn", raising=False)
    monkeypatch.setattr(status_mod, "resolve_provider", lambda requested=None, **kwargs: "minimax-cn", raising=False)
    monkeypatch.setattr(status_mod, "provider_label", lambda provider: "MiniMax", raising=False)

    monkeypatch.setattr(auth_mod, "get_nous_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_codex_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_qwen_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status", lambda: {}, raising=False)

    class _GatewayState:
        def __init__(self):
            self.snapshot = _snapshot()
            self.plist = tmp_path / "ai.hermes.gateway.plist"
            self.plist_exists = True
            self.probe_running = False

        def set_snapshot(self, **kwargs):
            self.snapshot = _snapshot(**kwargs)
            return self

        def set_plist(self, *, exists: bool):
            self.plist_exists = exists
            return self

        def set_probe(self, *, running: bool):
            self.probe_running = running
            return self

        def install(self):
            state = self

            monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda exclude_pids=None, all_profiles=False: list(state.snapshot.gateway_pids), raising=False)
            monkeypatch.setattr(gateway_mod, "get_gateway_runtime_snapshot", lambda: state.snapshot, raising=False)
            monkeypatch.setattr(gateway_mod, "_format_gateway_pids", lambda pids: ",".join(str(p) for p in pids), raising=False)
            monkeypatch.setattr(gateway_mod, "get_launchd_label", lambda: "ai.hermes.gateway", raising=False)
            monkeypatch.setattr(gateway_mod, "get_launchd_plist_path", lambda: state.plist, raising=False)
            monkeypatch.setattr(gateway_mod, "_probe_launchd_service_running", lambda: state.probe_running, raising=False)

            # Make Path.exists honor our toggle for the plist path only.
            original_exists = Path.exists

            def _exists(self, *a, **kw):
                if self == state.plist:
                    return state.plist_exists
                return original_exists(self, *a, **kw)

            monkeypatch.setattr(Path, "exists", _exists)
            return state

    return _GatewayState()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStatusLaunchdUnloadWarning:
    """The macOS warning only fires when ALL of these are true:

    - We're on macOS (``sys.platform == 'darwin'``)
    - The snapshot reports ``service_installed=True`` and ``service_running=False``
      — the "installed but stopped" branch in the existing status rendering
    - The plist file exists on disk
    - ``_probe_launchd_service_running()`` returns False (launchd has unloaded)

    On Linux/Windows the warning must NOT appear even when the plist path is
    callable, because no launchd exists to query.
    """

    def test_warning_shown_on_macos_when_plist_exists_and_unloaded(
        self, monkeypatch, capsys, isolated_status_env
    ):
        monkeypatch.setattr(sys, "platform", "darwin")
        isolated_status_env.set_snapshot(
            running=False,
            manager="launchd",
            service_installed=True,
            service_running=False,
        ).set_plist(exists=True).set_probe(running=False).install()

        show_status(SimpleNamespace(all=False, deep=False))

        out = capsys.readouterr().out
        assert "plist found but service is not loaded into launchd" in out, (
            f"expected unload warning in status output, got:\n{out}"
        )
        assert "hermes gateway start" in out, (
            f"expected repair hint in status output, got:\n{out}"
        )
        assert str(isolated_status_env.plist) not in out, (
            f"status must not echo raw plist path when routing through the CLI; "
            f"got:\n{out}"
        )

    def test_warning_suppressed_when_launchd_is_running(
        self, monkeypatch, capsys, isolated_status_env
    ):
        monkeypatch.setattr(sys, "platform", "darwin")
        isolated_status_env.set_snapshot(
            running=True,
            manager="launchd",
            pids=(12345,),
            service_installed=True,
            service_running=True,
        ).set_plist(exists=True).set_probe(running=True).install()

        show_status(SimpleNamespace(all=False, deep=False))

        out = capsys.readouterr().out
        assert "plist found but service is not loaded" not in out, (
            f"warning must not fire when launchd is supervising; got:\n{out}"
        )

    def test_warning_suppressed_when_plist_missing(
        self, monkeypatch, capsys, isolated_status_env
    ):
        monkeypatch.setattr(sys, "platform", "darwin")
        isolated_status_env.set_snapshot(
            running=False,
            manager="launchd",
            service_installed=True,
            service_running=False,
        ).set_plist(exists=False).set_probe(running=False).install()

        show_status(SimpleNamespace(all=False, deep=False))

        out = capsys.readouterr().out
        assert "plist found but service is not loaded" not in out, (
            f"warning must not fire when plist is absent; got:\n{out}"
        )

    def test_warning_suppressed_when_service_running_per_snapshot(
        self, monkeypatch, capsys, isolated_status_env
    ):
        # Snapshot says service is running → we don't enter the
        # "installed but stopped" branch at all, so the warning can't fire.
        monkeypatch.setattr(sys, "platform", "darwin")
        isolated_status_env.set_snapshot(
            running=True,
            manager="launchd",
            service_installed=True,
            service_running=True,
        ).set_plist(exists=True).set_probe(running=False).install()

        show_status(SimpleNamespace(all=False, deep=False))

        out = capsys.readouterr().out
        assert "plist found but service is not loaded" not in out, (
            f"warning must not fire when snapshot reports running; got:\n{out}"
        )

    def test_warning_suppressed_on_linux(
        self, monkeypatch, capsys, isolated_status_env
    ):
        # Force the "not darwin" code path; warning must be silent.
        monkeypatch.setattr(sys, "platform", "linux")
        isolated_status_env.set_snapshot(
            running=False,
            manager="systemd (user)",
            service_installed=True,
            service_running=False,
        ).set_plist(exists=True).set_probe(running=False).install()

        show_status(SimpleNamespace(all=False, deep=False))

        out = capsys.readouterr().out
        assert "plist found but service is not loaded" not in out, (
            f"warning must not fire on Linux even with plist/unload state; got:\n{out}"
        )

    def test_warning_does_not_recommend_raw_launchctl_command(
        self, monkeypatch, capsys, isolated_status_env
    ):
        """Regression: status must not suggest a raw ``launchctl bootstrap``
        command for repair. The launchd domain (gui/<uid> vs user/<uid>) is
        resolved by ``_launchd_domain()`` and depends on the session (Aqua vs
        Background/SSH); hardcoding ``gui/<uid>`` misleads Background/SSH
        users. The CLI (``hermes gateway start``) handles all the cases.

        Surfaced by hermes-sweeper review on PR #73370:
        https://github.com/NousResearch/hermes-agent/pull/73370
        """
        monkeypatch.setattr(sys, "platform", "darwin")
        isolated_status_env.set_snapshot(
            running=False,
            manager="launchd",
            service_installed=True,
            service_running=False,
        ).set_plist(exists=True).set_probe(running=False).install()

        show_status(SimpleNamespace(all=False, deep=False))

        out = capsys.readouterr().out
        assert "launchctl bootstrap" not in out, (
            f"status must not print a raw launchctl command — domain resolution "
            f"is hermes's responsibility (see _launchd_domain()); got:\n{out}"
        )
        assert "launchctl bootout" not in out, (
            f"status must not print a raw bootout command either — sticky domain "
            f"resolution same reason; got:\n{out}"
        )


class TestStatusSnapshotLoading:
    """The status function must accept the new symbol imports without crashing.

    If this fails, the new imports added in status.py are wrong.
    """

    def test_imports_succeed(self):
        from hermes_cli.gateway import (  # noqa: F401
            get_launchd_plist_path,
            _probe_launchd_service_running,
            get_launchd_label,
        )