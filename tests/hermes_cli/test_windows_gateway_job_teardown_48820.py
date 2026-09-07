"""Regression tests for #48820 (4th repro): job-object teardown killed the
post-update respawned gateway silently, and the updater printed
"✓ Restarting Windows gateway profile(s)" anyway.

Two fixes under test:

1. ``_spawn_gateway_restart_watcher``'s inlined watcher source must
   (a) route the respawned gateway's stray stdout/stderr to
       ``logs/gateway-stdio.log`` (it was ``DEVNULL`` — a gateway killed by
       parent Job Object teardown left ZERO trace anywhere), and
   (b) stamp ``_HERMES_GATEWAY_BREAKAWAY`` =1/0 on the respawn env exactly
       like the canonical ``gateway_windows._spawn_detached``, so the
       lifecycle/exit-diag records show whether the gateway escaped the
       parent's Job Object.

2. ``_resume_windows_gateways_after_update`` must verify a stable gateway
   process actually exists (via ``gateway_windows._wait_for_gateway_ready``)
   before printing the ✓ — a truthy launch return only proves the watcher
   process was created, not that the respawned gateway survived the
   updater's Job Object teardown.
"""

from unittest.mock import patch

import pytest

import hermes_cli.gateway as gateway
import hermes_cli.gateway_windows as gateway_windows
import hermes_cli.main as hm
from hermes_cli._subprocess_compat import _WINDOWS_GATEWAY_BREAKAWAY_ENV
from hermes_cli.update_cmd import _resume_windows_gateways_after_update


# ---------------------------------------------------------------------------
# 1. Watcher template contract
# ---------------------------------------------------------------------------


def _captured_watcher_source(monkeypatch) -> str:
    """Spawn the watcher with a mocked Popen and return the inlined -c source."""
    captured = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs

        class _P:
            pid = 12345

        return _P()

    monkeypatch.setattr(gateway.subprocess, "Popen", fake_popen)
    assert gateway._spawn_gateway_restart_watcher(
        999999, ["python", "-m", "hermes_cli.main", "gateway", "run"]
    )
    argv = captured["argv"]
    assert argv[1] == "-c"
    return argv[2]


class TestWatcherRespawnTemplate:
    def test_respawn_stdio_routed_to_sidecar_log_not_devnull(self, monkeypatch):
        """DEVNULL swallowed the dying gateway's last words (#48820 4th
        repro: 'Zero trace anywhere ... because the watcher respawns with
        stdout=DEVNULL, stderr=DEVNULL')."""
        src = _captured_watcher_source(monkeypatch)
        assert "gateway-stdio.log" in src, (
            "watcher respawn must route stray stdout/stderr to the same "
            "sidecar log _spawn_detached uses, so a gateway killed moments "
            "after respawn leaves a trace"
        )
        # DEVNULL remains only as the fallback when the log dir is
        # unavailable — the popen kwargs must not be hardwired to it.
        assert '"stdout": _stdio_target' in src
        assert '"stderr": _stdio_target' in src

    def test_respawn_stamps_breakaway_state_like_spawn_detached(
        self, monkeypatch
    ):
        """The respawned gateway must carry _HERMES_GATEWAY_BREAKAWAY=1 on
        the primary (breakaway) spawn and =0 on the no-breakaway fallback,
        mirroring gateway_windows._spawn_detached — without the stamp, a
        job-teardown kill is indistinguishable from any other silent death
        in the exit diagnostics."""
        src = _captured_watcher_source(monkeypatch)
        assert "_WINDOWS_GATEWAY_BREAKAWAY_ENV" in src
        assert _WINDOWS_GATEWAY_BREAKAWAY_ENV == "_HERMES_GATEWAY_BREAKAWAY"
        # Primary stamps "1", the OSError fallback stamps "0".
        assert '_WINDOWS_GATEWAY_BREAKAWAY_ENV: "1"' in src
        assert '_WINDOWS_GATEWAY_BREAKAWAY_ENV: "0"' in src

    def test_respawn_source_compiles(self, monkeypatch):
        """The inlined -c template is built via str.format over a
        dedented literal — guard against brace/indentation regressions."""
        src = _captured_watcher_source(monkeypatch)
        compile(src, "<watcher>", "exec")

    def test_watcher_fallback_retry_preserved(self, monkeypatch):
        """The ERROR_ACCESS_DENIED retry without breakaway must survive."""
        src = _captured_watcher_source(monkeypatch)
        assert "windows_detach_flags_without_breakaway" in src


# ---------------------------------------------------------------------------
# 2. Post-update resume liveness gate
# ---------------------------------------------------------------------------


def _token(profiles: dict) -> dict:
    return {
        "resume_needed": True,
        "profiles": profiles,
        "unmapped_pids": [],
        "unmapped": [],
    }


class TestResumeLivenessGate:
    @pytest.fixture(autouse=True)
    def _windows(self, monkeypatch):
        monkeypatch.setattr(hm, "_is_windows", lambda: True)
        monkeypatch.setattr(hm, "_refresh_windows_gateway_launchers", lambda: None)
        monkeypatch.setattr(
            gateway, "launch_detached_profile_gateway_restart", lambda *_a: True
        )
        monkeypatch.setattr(
            gateway, "launch_detached_gateway_restart_by_cmdline", lambda *_a: True
        )

    def test_dead_respawn_fails_the_resume_instead_of_printing_check(
        self, monkeypatch
    ):
        """No stable gateway after the relaunch → the resume raises (update
        marked incomplete) instead of printing '✓ Restarting'. This is the
        exact #48820 3rd/4th-repro hole: spawn succeeded, gateway died
        within seconds, success was reported, platforms were offline for
        12.5 hours."""
        monkeypatch.setattr(
            gateway_windows, "_wait_for_gateway_ready", lambda **_kw: []
        )
        token = _token({"default": 1111})
        printed = []
        with patch("builtins.print", side_effect=lambda *a, **k: printed.append(a)):
            with pytest.raises(RuntimeError, match="not verified alive"):
                _resume_windows_gateways_after_update(token)

        text = " ".join(str(a) for a in printed)
        assert "✓ Restarting" not in text
        assert "could not be verified" in text
        # The profile stays on the token so retry/reporting still sees it.
        assert token["profiles"] == {"default": 1111}
        assert token["resume_needed"] is True

    def test_live_respawn_prints_check_and_writes_attestation(self, monkeypatch):
        monkeypatch.setattr(
            gateway_windows, "_wait_for_gateway_ready", lambda **_kw: [777]
        )
        attested = {}
        monkeypatch.setattr(
            gateway_windows,
            "_write_start_attestation",
            lambda pids, via: attested.update(pids=pids, via=via),
        )
        token = _token({"default": 1111})
        printed = []
        with patch("builtins.print", side_effect=lambda *a, **k: printed.append(a)):
            _resume_windows_gateways_after_update(token)

        text = " ".join(str(a) for a in printed)
        assert "✓ Restarting" in text
        assert attested == {"pids": [777], "via": "post-update relaunch"}
        assert token["resume_needed"] is False

    def test_liveness_poll_scans_all_profiles(self, monkeypatch):
        """The resume relaunches the whole fleet; the verification must not
        be scoped to the active profile."""
        seen = {}

        def fake_wait(**kwargs):
            seen.update(kwargs)
            return [777]

        monkeypatch.setattr(gateway_windows, "_wait_for_gateway_ready", fake_wait)
        monkeypatch.setattr(
            gateway_windows, "_write_start_attestation", lambda *_a, **_kw: None
        )
        with patch("builtins.print"):
            _resume_windows_gateways_after_update(_token({"work": 2222}))
        assert seen.get("all_profiles") is True
