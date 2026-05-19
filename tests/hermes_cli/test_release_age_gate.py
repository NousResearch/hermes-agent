"""Tests for the release-age gate on ``hermes update``.

The gate refuses to resolve PyPI versions published less than
``security.minimum_release_age_days`` ago by passing ``--exclude-newer``
to ``uv pip install``. This defends against supply-chain compromise
windows where a malicious release is detected and pulled within days.

The two live incidents that motivated this gate:

- **mistralai 2.4.6** (PyPI, 2026-05-12, Mini Shai-Hulud campaign):
  malicious release was live for ~15 minutes before PyPI quarantine.
- **node-ipc 9.1.6 / 9.2.3 / 12.0.1** (npm, 2026-05-14): malicious
  versions live for hours-to-days before detection-driven removal.

A 3-day gate would have refused both classes outright. These tests
exercise the helpers in isolation (no real PyPI, no real subprocess
calls).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from hermes_cli import main as cli_main


# ---------------------------------------------------------------------------
# _exclude_newer_date — pure date arithmetic
# ---------------------------------------------------------------------------


class TestExcludeNewerDate:
    def test_disabled_returns_none(self):
        assert cli_main._exclude_newer_date(0) is None

    def test_negative_returns_none(self):
        assert cli_main._exclude_newer_date(-1) is None
        assert cli_main._exclude_newer_date(-30) is None

    # Mid-day fixed clock so the tests prove exact N×24h arithmetic, not
    # just date truncation.
    _FIXED_NOW = datetime(2026, 5, 13, 12, 30, 45, tzinfo=timezone.utc)

    def test_one_day(self):
        cutoff = cli_main._exclude_newer_date(1, _now_utc=self._FIXED_NOW)
        assert cutoff == "2026-05-12T12:30:45Z"

    def test_three_days_matches_nanoclaw_default(self):
        # NanoClaw's pnpm minimumReleaseAge: 4320 (3 days)
        cutoff = cli_main._exclude_newer_date(3, _now_utc=self._FIXED_NOW)
        assert cutoff == "2026-05-10T12:30:45Z"

    def test_seven_days(self):
        cutoff = cli_main._exclude_newer_date(7, _now_utc=self._FIXED_NOW)
        assert cutoff == "2026-05-06T12:30:45Z"

    def test_rfc3339_format(self):
        """Output must be an RFC 3339 UTC timestamp (exact-window enforcement;
        uv would read a bare date as local midnight)."""
        cutoff = cli_main._exclude_newer_date(1)
        assert cutoff is not None
        assert cutoff.endswith("Z")
        # Parse round-trip — raises if not valid ISO 8601
        datetime.fromisoformat(cutoff)

    def test_non_utc_aware_clock_is_normalized(self):
        """An aware non-UTC "now" must still produce a UTC cutoff."""
        plus_two = timezone(timedelta(hours=2))
        now = datetime(2026, 5, 13, 14, 30, 45, tzinfo=plus_two)  # == 12:30:45Z
        cutoff = cli_main._exclude_newer_date(1, _now_utc=now)
        assert cutoff == "2026-05-12T12:30:45Z"


# ---------------------------------------------------------------------------
# _get_min_release_age_days — config read with safe default
# ---------------------------------------------------------------------------


class TestGetMinReleaseAgeDays:
    def test_default_when_config_missing_key(self):
        with patch("hermes_cli.config.load_config", return_value={"security": {}}):
            assert cli_main._get_min_release_age_days() == 0

    def test_default_when_security_block_missing(self):
        with patch("hermes_cli.config.load_config", return_value={}):
            assert cli_main._get_min_release_age_days() == 0

    def test_default_when_config_load_raises(self):
        # Failing closed (blocking the gate) would be wrong — it would
        # block all updates for any config error.
        with patch("hermes_cli.config.load_config", side_effect=RuntimeError("boom")):
            assert cli_main._get_min_release_age_days() == 0

    def test_reads_configured_value(self):
        with patch(
            "hermes_cli.config.load_config",
            return_value={"security": {"minimum_release_age_days": 3}},
        ):
            assert cli_main._get_min_release_age_days() == 3

    def test_coerces_str_int(self):
        with patch(
            "hermes_cli.config.load_config",
            return_value={"security": {"minimum_release_age_days": "7"}},
        ):
            assert cli_main._get_min_release_age_days() == 7

    def test_none_treated_as_zero(self):
        with patch(
            "hermes_cli.config.load_config",
            return_value={"security": {"minimum_release_age_days": None}},
        ):
            assert cli_main._get_min_release_age_days() == 0


# ---------------------------------------------------------------------------
# _install_python_dependencies_with_optional_fallback — flag injection
# ---------------------------------------------------------------------------


class TestExcludeNewerInjection:
    """Verify the install function actually passes --exclude-newer to uv.

    These tests assert the *behavior* of the install path with a fixture
    around `_run_install_with_heartbeat` — they catch regressions where
    `extra_args` is silently dropped or `exclude_newer` is forgotten by
    a future refactor.
    """

    def test_uv_command_gets_exclude_newer_when_provided(self, monkeypatch):
        """A uv-prefixed install command receives --exclude-newer <date>."""
        captured_args: list[list[str]] = []

        def fake_run(args, env=None):
            captured_args.append(list(args))

        monkeypatch.setattr(cli_main, "_run_install_with_heartbeat", fake_run)
        monkeypatch.setattr(cli_main, "_is_windows", lambda: False)

        cli_main._install_python_dependencies_with_optional_fallback(
            ["uv", "pip"], group="all", exclude_newer="2026-05-10"
        )

        assert len(captured_args) == 1
        assert captured_args[0] == [
            "uv", "pip", "install", "-e", ".[all]", "--exclude-newer", "2026-05-10"
        ]

    def test_uv_command_no_flag_when_none(self, monkeypatch):
        """When exclude_newer is None (gate disabled), no flag is added."""
        captured_args: list[list[str]] = []

        def fake_run(args, env=None):
            captured_args.append(list(args))

        monkeypatch.setattr(cli_main, "_run_install_with_heartbeat", fake_run)
        monkeypatch.setattr(cli_main, "_is_windows", lambda: False)

        cli_main._install_python_dependencies_with_optional_fallback(
            ["uv", "pip"], group="all", exclude_newer=None
        )

        assert "--exclude-newer" not in captured_args[0]

    def test_pip_command_ignores_exclude_newer(self, monkeypatch):
        """Plain pip does not support --exclude-newer; flag is suppressed."""
        captured_args: list[list[str]] = []

        def fake_run(args, env=None):
            captured_args.append(list(args))

        monkeypatch.setattr(cli_main, "_run_install_with_heartbeat", fake_run)
        monkeypatch.setattr(cli_main, "_is_windows", lambda: False)

        cli_main._install_python_dependencies_with_optional_fallback(
            ["/usr/bin/python", "-m", "pip"], group="all", exclude_newer="2026-05-10"
        )

        assert "--exclude-newer" not in captured_args[0]

    @pytest.mark.parametrize("name", ["pyuv", "uvloop-wrapper", "python-uv-shim", "uvicorn"])
    def test_uv_substring_does_not_false_positive(self, monkeypatch, name):
        """Binary basenames containing 'uv' but not equal to 'uv' / 'uv.exe'
        must not get the flag — they're not uv and would crash on unknown args."""
        captured_args: list[list[str]] = []

        def fake_run(args, env=None):
            captured_args.append(list(args))

        monkeypatch.setattr(cli_main, "_run_install_with_heartbeat", fake_run)
        monkeypatch.setattr(cli_main, "_is_windows", lambda: False)

        cli_main._install_python_dependencies_with_optional_fallback(
            [f"/some/path/{name}", "pip"], group="all", exclude_newer="2026-05-10"
        )

        assert "--exclude-newer" not in captured_args[0]

    def test_uv_exe_on_windows(self, monkeypatch):
        """Exact match also accepts the Windows ``uv.exe`` form."""
        captured_args: list[list[str]] = []

        def fake_run(args, env=None):
            captured_args.append(list(args))

        monkeypatch.setattr(cli_main, "_run_install_with_heartbeat", fake_run)
        monkeypatch.setattr(cli_main, "_is_windows", lambda: False)

        cli_main._install_python_dependencies_with_optional_fallback(
            ["C:/Users/x/uv.exe", "pip"], group="all", exclude_newer="2026-05-10"
        )

        assert "--exclude-newer" in captured_args[0]


# ---------------------------------------------------------------------------
# Regression scenarios — the two live incidents that motivated this gate
# ---------------------------------------------------------------------------


class TestRealIncidentScenarios:
    """Verify the install path would have refused real malicious releases.

    These tests freeze "now" via the ``_now_utc`` test seam on
    ``_exclude_newer_date``, then call the actual install function with a
    captured-arg fake for ``_run_install_with_heartbeat``. They assert the
    *exact* ``--exclude-newer <cutoff>`` value lands in the install argv —
    so removing the ``extra_args`` injection in production would fail these
    tests, not just the synthetic injection tests above.

    The asserted cutoff values are computed from each scenario's frozen
    "today" minus the gate window. uv's documented behavior is to refuse
    any artifact whose upload time is after ``--exclude-newer``, so when
    cutoff < release_date, the malicious version is refused.
    """

    @staticmethod
    def _run_with_frozen_clock(
        monkeypatch, today_iso: str, gate_days: int
    ) -> list[list[str]]:
        """Run the install path with frozen clock + captured argv; return
        the list of argv lists passed to ``_run_install_with_heartbeat``."""
        today = datetime.fromisoformat(today_iso).replace(tzinfo=timezone.utc)
        captured: list[list[str]] = []

        def fake_run(args, env=None):
            captured.append(list(args))

        monkeypatch.setattr(cli_main, "_run_install_with_heartbeat", fake_run)
        monkeypatch.setattr(cli_main, "_is_windows", lambda: False)

        cutoff = cli_main._exclude_newer_date(gate_days, _now_utc=today)
        cli_main._install_python_dependencies_with_optional_fallback(
            ["uv", "pip"], group="all", exclude_newer=cutoff
        )
        return captured

    def test_mistralai_2_4_6_refused_by_3_day_gate(self, monkeypatch):
        """mistralai 2.4.6 was published 2026-05-12.

        On 2026-05-13 with a 3-day gate, cutoff is 2026-05-10 — uv would
        refuse 2026-05-12 (>2026-05-10). The install argv must contain
        the exact flag/value pair so uv applies it.
        """
        argv = self._run_with_frozen_clock(monkeypatch, "2026-05-13", gate_days=3)
        assert len(argv) == 1
        assert "--exclude-newer" in argv[0]
        idx = argv[0].index("--exclude-newer")
        cutoff = argv[0][idx + 1]
        assert cutoff == "2026-05-10T00:00:00Z"
        # The actual refusal condition: cutoff < malicious-release date
        assert cutoff < "2026-05-12"

    def test_mistralai_2_4_6_still_refused_a_day_later(self, monkeypatch):
        """On 2026-05-14, cutoff = 2026-05-11; still < 2026-05-12 → refused."""
        argv = self._run_with_frozen_clock(monkeypatch, "2026-05-14", gate_days=3)
        idx = argv[0].index("--exclude-newer")
        cutoff = argv[0][idx + 1]
        assert cutoff == "2026-05-11T00:00:00Z"
        assert cutoff < "2026-05-12"

    def test_mistralai_2_4_6_allowed_after_window_expires(self, monkeypatch):
        """On 2026-05-16, cutoff = 2026-05-13 — no longer < 2026-05-12.
        In practice by this point the ecosystem has pulled the package so
        upstream resolution fails anyway; the gate's job is the early
        window, not perpetual protection."""
        argv = self._run_with_frozen_clock(monkeypatch, "2026-05-16", gate_days=3)
        idx = argv[0].index("--exclude-newer")
        cutoff = argv[0][idx + 1]
        assert cutoff == "2026-05-13T00:00:00Z"
        assert cutoff >= "2026-05-12"  # gate no longer refuses

    def test_node_ipc_9_1_6_refused_by_3_day_gate(self, monkeypatch):
        """node-ipc 9.1.6 was published 2026-05-14. On 2026-05-15 with a
        3-day gate, cutoff = 2026-05-12 — refused. Hermes doesn't pull
        node-ipc from PyPI (it's npm) but the gate's mechanism is
        ecosystem-agnostic; the test validates the same upload-date logic
        the npm ecosystem analog would use."""
        argv = self._run_with_frozen_clock(monkeypatch, "2026-05-15", gate_days=3)
        idx = argv[0].index("--exclude-newer")
        cutoff = argv[0][idx + 1]
        assert cutoff == "2026-05-12T00:00:00Z"
        assert cutoff < "2026-05-14"

    def test_no_gate_emits_no_flag(self, monkeypatch):
        """With gate disabled, --exclude-newer must be absent from argv —
        the install proceeds with normal resolver behavior."""
        argv = self._run_with_frozen_clock(monkeypatch, "2026-05-13", gate_days=0)
        assert "--exclude-newer" not in argv[0]

    def test_long_gate_window_for_audited_environments(self, monkeypatch):
        """30-day gate (audited/regulated environments). On 2026-06-01 with
        a 30-day gate, cutoff = 2026-05-02 — refuses anything from the
        past month, including the 2026-05-12 incident."""
        argv = self._run_with_frozen_clock(monkeypatch, "2026-06-01", gate_days=30)
        idx = argv[0].index("--exclude-newer")
        cutoff = argv[0][idx + 1]
        assert cutoff == "2026-05-02T00:00:00Z"
        assert cutoff < "2026-05-12"


class TestGateCoverageOfInternalInstallers:
    """The gate must also reach installs that run *inside* the update flow:
    the core-dep verifier's repair reinstall and the lazy-backend refresh."""

    def test_uv_exclude_newer_args_helper(self):
        assert cli_main._uv_exclude_newer_args(["/usr/bin/uv", "pip"], "2026-05-10") == [
            "--exclude-newer", "2026-05-10",
        ]
        # Plain pip never receives the uv-only flag.
        assert cli_main._uv_exclude_newer_args(["/usr/bin/python", "-m", "pip"], "2026-05-10") == []
        assert cli_main._uv_exclude_newer_args(["/usr/bin/uv", "pip"], None) == []

    def test_verifier_repair_command_carries_exclude_newer(self, monkeypatch):
        """When the verifier detects missing core deps and reinstalls, the
        repair command must carry the same cutoff as the original install."""
        from pathlib import Path
        from types import SimpleNamespace

        repaired: list[list[str]] = []

        # First dep-check reports one missing dep (triggers repair), second
        # reports none (repair "worked").
        checks = iter(["somepkg\n", ""])

        def _fake_run(cmd, **kw):
            return SimpleNamespace(returncode=0, stdout=next(checks), stderr="")

        monkeypatch.setattr(cli_main.subprocess, "run", _fake_run)
        monkeypatch.setattr(
            cli_main, "_resolve_install_target_python", lambda *a, **kw: Path("/x/python")
        )
        monkeypatch.setattr(
            cli_main,
            "_run_quarantined_install",
            lambda cmd, **kw: repaired.append(list(cmd)),
        )

        cli_main._verify_core_dependencies_installed(
            ["/usr/bin/uv", "pip"], exclude_newer="2026-05-10"
        )

        assert len(repaired) == 1
        assert repaired[0][-2:] == ["--exclude-newer", "2026-05-10"]
        assert "--reinstall" in repaired[0]

    def test_lazy_refresh_scopes_uv_exclude_newer_env(self, monkeypatch):
        """UV_EXCLUDE_NEWER must be set for the duration of the lazy refresh
        (uv's env equivalent of --exclude-newer) and restored afterwards."""
        import os

        seen: dict[str, str | None] = {}

        class _FakeLazyDeps:
            @staticmethod
            def active_features():
                return ["voice"]

            @staticmethod
            def refresh_active_features(prompt=False):
                seen["during"] = os.environ.get("UV_EXCLUDE_NEWER")
                return {"voice": "current"}

        monkeypatch.delenv("UV_EXCLUDE_NEWER", raising=False)
        monkeypatch.setattr(cli_main, "_get_min_release_age_days", lambda: 3)
        monkeypatch.setattr(
            cli_main, "_exclude_newer_date", lambda days, **kw: "2026-05-10"
        )
        import sys as _sys

        monkeypatch.setitem(_sys.modules, "tools.lazy_deps", _FakeLazyDeps())
        import tools

        monkeypatch.setattr(tools, "lazy_deps", _FakeLazyDeps(), raising=False)
        cli_main._refresh_active_lazy_features()

        assert seen["during"] == "2026-05-10"
        assert "UV_EXCLUDE_NEWER" not in os.environ

    def test_lazy_refresh_leaves_env_untouched_when_gate_off(self, monkeypatch):
        import os

        seen: dict[str, str | None] = {}

        class _FakeLazyDeps:
            @staticmethod
            def active_features():
                return ["voice"]

            @staticmethod
            def refresh_active_features(prompt=False):
                seen["during"] = os.environ.get("UV_EXCLUDE_NEWER")
                return {"voice": "current"}

        monkeypatch.delenv("UV_EXCLUDE_NEWER", raising=False)
        monkeypatch.setattr(cli_main, "_get_min_release_age_days", lambda: 0)
        import sys as _sys

        monkeypatch.setitem(_sys.modules, "tools.lazy_deps", _FakeLazyDeps())
        import tools

        monkeypatch.setattr(tools, "lazy_deps", _FakeLazyDeps(), raising=False)
        cli_main._refresh_active_lazy_features()

        assert seen["during"] is None
        assert "UV_EXCLUDE_NEWER" not in os.environ

    def test_verifier_per_package_fallback_carries_exclude_newer(self, monkeypatch):
        """The last-ditch per-package force-install must stay gated too —
        ranged requirements could otherwise resolve past the cutoff."""
        from pathlib import Path
        from types import SimpleNamespace

        heartbeat_cmds: list[list[str]] = []

        # check1: missing → full repair; check2: still missing → per-package
        # force-install; check3: resolved.
        checks = iter(["somepkg\n", "somepkg\n", ""])

        def _fake_run(cmd, **kw):
            return SimpleNamespace(returncode=0, stdout=next(checks), stderr="")

        monkeypatch.setattr(cli_main.subprocess, "run", _fake_run)
        monkeypatch.setattr(
            cli_main, "_resolve_install_target_python", lambda *a, **kw: Path("/x/python")
        )
        monkeypatch.setattr(
            cli_main, "_run_quarantined_install", lambda cmd, **kw: None
        )
        monkeypatch.setattr(
            cli_main,
            "_run_install_with_heartbeat",
            lambda cmd, **kw: heartbeat_cmds.append(list(cmd)),
        )

        cli_main._verify_core_dependencies_installed(
            ["/usr/bin/uv", "pip"], exclude_newer="2026-05-10"
        )

        assert len(heartbeat_cmds) == 1
        assert heartbeat_cmds[0][-2:] == ["--exclude-newer", "2026-05-10"]

    def test_lazy_install_fails_closed_instead_of_ungated_pip_fallback(self, monkeypatch):
        """With UV_EXCLUDE_NEWER set, _venv_pip_install must not fall back to
        plain pip (which ignores the cutoff): it fails closed so the backend
        keeps its previously-installed version."""
        from tools import lazy_deps

        monkeypatch.setenv("UV_EXCLUDE_NEWER", "2026-05-10")
        # uv absent → tier 1 skipped entirely.
        monkeypatch.setattr(lazy_deps.shutil, "which", lambda name: None)

        def _no_subprocess(*a, **kw):
            raise AssertionError("pip fallback must not run while gate env is set")

        monkeypatch.setattr(lazy_deps.subprocess, "run", _no_subprocess)

        result = lazy_deps._venv_pip_install(("somepkg==1.0",))
        assert result.success is False
        assert "release-age" in result.stderr

    def test_android_psutil_preinstall_respects_gate(self, monkeypatch, capsys):
        """The Termux psutil sdist is fetched directly (no uv resolution), so
        the gate compares the pin's recorded upload time to the cutoff."""
        installed: list[list[str]] = []
        monkeypatch.setattr(
            cli_main, "_run_install_with_heartbeat", lambda cmd, **kw: installed.append(cmd)
        )
        monkeypatch.setattr(
            "hermes_cli.psutil_android.PSUTIL_UPLOAD_TIME", "2026-05-12T09:00:00Z"
        )

        # Cutoff older than the pin's upload → refused, nothing downloaded.
        cli_main._install_psutil_android_compat(
            ["uv", "pip"], exclude_newer="2026-05-10T00:00:00Z"
        )
        assert installed == []
        assert "release-age gate" in capsys.readouterr().out

    def test_android_psutil_preinstall_proceeds_when_pin_old_enough(self, monkeypatch):
        installed: list[list[str]] = []
        monkeypatch.setattr(
            cli_main, "_run_install_with_heartbeat", lambda cmd, **kw: installed.append(cmd)
        )
        monkeypatch.setattr(
            "hermes_cli.psutil_android.PSUTIL_UPLOAD_TIME", "2026-01-28T18:14:54Z"
        )
        monkeypatch.setattr(
            "urllib.request.urlretrieve", lambda url, dst: Path(dst).write_bytes(b"")
        )
        monkeypatch.setattr(
            "hermes_cli.psutil_android.prepare_patched_psutil_sdist",
            lambda archive, tmp: tmp / "psutil-src",
        )
        from pathlib import Path

        cli_main._install_psutil_android_compat(
            ["uv", "pip"], exclude_newer="2026-05-10T00:00:00Z"
        )
        assert len(installed) == 1
        assert "--no-build-isolation" in installed[0]
