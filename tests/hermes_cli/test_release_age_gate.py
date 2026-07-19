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


class TestReleaseAgePreflight:
    """Option-A preflight: defer the update BEFORE source advances when the
    incoming core dependency set is blocked solely by the gate."""

    _PYPROJECT = '[project]\nname = "x"\nversion = "0"\ndependencies = ["foo==1.2.3"]\n'

    def _run(
        self,
        monkeypatch,
        *,
        ungated_rc,
        gated_offline_rc=0,
        gated_online_rc=0,
        gated_build_rc=0,
        uv="/usr/bin/uv",
    ):
        calls: list[list[str]] = []
        envs: list[dict | None] = []

        def fake_run(cmd, **kw):
            from types import SimpleNamespace

            if "--help" in cmd:
                # Capability probe: report a uv that supports all gate flags.
                return SimpleNamespace(
                    returncode=0,
                    stdout="--exclude-newer --no-build --offline",
                    stderr="",
                )
            calls.append(list(cmd))
            envs.append(kw.get("env"))
            if "--exclude-newer" not in cmd:
                rc = ungated_rc
            elif "--offline" in cmd:
                rc = gated_offline_rc
            elif "--no-build" in cmd:
                rc = gated_online_rc
            else:
                rc = gated_build_rc

            return SimpleNamespace(
                returncode=rc,
                stdout="",
                stderr="refused: only releases before cutoff allowed" if rc else "",
            )

        monkeypatch.setattr(cli_main.subprocess, "run", fake_run)
        monkeypatch.setattr("hermes_cli.managed_uv.ensure_uv", lambda: uv)
        result = cli_main._release_age_preflight_blocked(
            self._PYPROJECT, "2026-05-10T00:00:00Z"
        )
        return result, calls, envs

    def test_defers_when_only_the_gate_blocks_resolution(self, monkeypatch):
        """Ungated succeeds; gated fails offline, online, AND with builds
        permitted → confirmed rejection."""
        blocked, calls, _ = self._run(
            monkeypatch,
            ungated_rc=0,
            gated_offline_rc=1,
            gated_online_rc=1,
            gated_build_rc=1,
        )
        assert blocked is not None and "refused" in blocked
        assert len(calls) == 4
        assert "--exclude-newer" not in calls[0]
        assert "--offline" in calls[1]
        assert "--offline" not in calls[2] and "--exclude-newer" in calls[2]
        # The resolves that can encounter ungated candidates must never
        # execute a PEP 517 backend; only the final gate-confirmed check
        # permits builds (its candidates are all gate-passing artifacts).
        assert all("--no-build" in c for c in calls[:3])
        assert "--no-build" not in calls[3] and "--exclude-newer" in calls[3]

    def test_no_deferral_when_aged_sdist_satisfies_gated_build(self, monkeypatch):
        """Wheel newer than cutoff + aged sdist: the --no-build gated runs
        fail, but the build-permitted gated confirm succeeds (the real
        install builds) → fail open."""
        blocked, calls, _ = self._run(
            monkeypatch,
            ungated_rc=0,
            gated_offline_rc=1,
            gated_online_rc=1,
            gated_build_rc=0,
        )
        assert blocked is None
        assert len(calls) == 4

    def test_ambient_uv_exclude_newer_is_scrubbed(self, monkeypatch):
        """A stray UV_EXCLUDE_NEWER in the environment would silently gate
        the 'ungated' control, masking real rejections as broken-regardless."""
        monkeypatch.setenv("UV_EXCLUDE_NEWER", "2020-01-01T00:00:00Z")
        blocked, calls, envs = self._run(monkeypatch, ungated_rc=0)
        assert blocked is None
        assert envs and all(e is not None for e in envs)
        assert all("UV_EXCLUDE_NEWER" not in e for e in envs)

    def test_preflight_includes_build_system_requires(self, monkeypatch):
        """A too-new [build-system].requires pin strands the update the same
        way a runtime dep does — it must be part of the resolved set."""
        seen_reqs: dict[str, str] = {}

        def fake_run(cmd, **kw):
            from pathlib import Path
            from types import SimpleNamespace

            if "--help" in cmd:
                return SimpleNamespace(
                    returncode=0,
                    stdout="--exclude-newer --no-build --offline",
                    stderr="",
                )
            reqs_path = next(a for a in cmd if str(a).endswith("requirements.in"))
            seen_reqs["content"] = Path(reqs_path).read_text(encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(cli_main.subprocess, "run", fake_run)
        monkeypatch.setattr("hermes_cli.managed_uv.ensure_uv", lambda: "/usr/bin/uv")

        pyproject = (
            '[build-system]\nrequires = ["setuptools==80.0.0"]\n'
            '[project]\nname = "x"\nversion = "0"\ndependencies = ["foo==1.2.3"]\n'
        )
        assert (
            cli_main._release_age_preflight_blocked(pyproject, "2026-05-10T00:00:00Z")
            is None
        )
        assert "foo==1.2.3" in seen_reqs["content"]
        assert "setuptools==80.0.0" in seen_reqs["content"]

    def test_sdist_only_falls_back_to_gated_build_and_proceeds(self, monkeypatch):
        """Wheels-only control fails (sdist-only dep, e.g. Termux) but the
        gated build-permitted resolve succeeds → the real install will too."""
        blocked, calls, _ = self._run(
            monkeypatch, ungated_rc=1, gated_build_rc=0
        )
        assert blocked is None
        assert len(calls) == 2
        # The fallback resolve is gated (never an ungated build) so it can
        # only execute backends of aged, gate-passing artifacts.
        assert "--exclude-newer" in calls[1] and "--no-build" not in calls[1]

    def test_sdist_only_defers_conservatively_when_gated_build_fails(
        self, monkeypatch
    ):
        """If even the gated build can't resolve, advancing source would
        strand the install whether the cause is the gate or broken deps —
        defer, with wording that doesn't overclaim gate causality."""
        blocked, calls, _ = self._run(
            monkeypatch, ungated_rc=1, gated_build_rc=1
        )
        assert blocked is not None
        assert "may also indicate a resolution problem" in blocked
        assert len(calls) == 2

    def test_no_deferral_when_gated_resolution_succeeds(self, monkeypatch):
        blocked, calls, _ = self._run(monkeypatch, ungated_rc=0, gated_offline_rc=0)
        assert blocked is None
        assert len(calls) == 2  # ungated + gated --offline; online confirm not needed

    def test_no_deferral_when_offline_fails_but_online_succeeds(self, monkeypatch):
        """A cache miss can fail the --offline run; the online confirmation
        succeeding proves the gate is not the cause → fail open."""
        blocked, calls, _ = self._run(
            monkeypatch, ungated_rc=0, gated_offline_rc=1, gated_online_rc=0
        )
        assert blocked is None
        assert len(calls) == 3

    def test_fails_open_without_uv(self, monkeypatch):
        blocked, calls, _ = self._run(
            monkeypatch, ungated_rc=0, gated_offline_rc=1, gated_online_rc=1, uv=None
        )
        assert blocked is None
        assert calls == []

    def test_fails_open_on_unparsable_pyproject(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.managed_uv.ensure_uv", lambda: "/usr/bin/uv")

        def boom(cmd, **kw):
            raise AssertionError("must not resolve an unparsable pyproject")

        monkeypatch.setattr(cli_main.subprocess, "run", boom)
        assert (
            cli_main._release_age_preflight_blocked("not [ toml", "2026-05-10T00:00:00Z")
            is None
        )

    def test_deferral_message_names_cutoff_and_override(self, monkeypatch, capsys):
        monkeypatch.setattr(cli_main, "_get_min_release_age_days", lambda: 3)
        cli_main._print_release_age_deferral(
            "refused: foo too new", "2026-05-10T00:00:00Z"
        )
        out = capsys.readouterr().out
        assert "Update deferred" in out
        assert "2026-05-10T00:00:00Z" in out
        assert "minimum_release_age_days" in out
        assert "unchanged" in out

    def test_lazy_refresh_makes_managed_uv_discoverable(self, monkeypatch):
        """On managed installs uv lives off PATH; while the gate scopes
        UV_EXCLUDE_NEWER (which makes lazy_deps fail closed without uv), the
        managed uv's dir must be prepended to PATH — and restored after."""
        import os

        seen: dict[str, str | None] = {}

        class _FakeLazyDeps:
            @staticmethod
            def active_features():
                return ["voice"]

            @staticmethod
            def refresh_active_features(prompt=False):
                seen["path"] = os.environ.get("PATH")
                return {"voice": "current"}

        original_path = os.environ.get("PATH", "")
        monkeypatch.delenv("UV_EXCLUDE_NEWER", raising=False)
        monkeypatch.setattr(cli_main, "_get_min_release_age_days", lambda: 3)
        monkeypatch.setattr(
            cli_main, "_exclude_newer_date", lambda days, **kw: "2026-05-10T00:00:00Z"
        )
        monkeypatch.setattr(cli_main.shutil, "which", lambda name: None)
        monkeypatch.setattr(
            "hermes_cli.managed_uv.ensure_uv", lambda: "/fake/hermes-home/bin/uv"
        )
        import sys as _sys

        monkeypatch.setitem(_sys.modules, "tools.lazy_deps", _FakeLazyDeps())
        import tools

        monkeypatch.setattr(tools, "lazy_deps", _FakeLazyDeps(), raising=False)
        cli_main._refresh_active_lazy_features()

        assert seen["path"].startswith("/fake/hermes-home/bin" + os.pathsep)
        assert os.environ.get("PATH", "") == original_path

    def test_git_ref_preflight_fails_open_when_ref_has_no_pyproject(self, monkeypatch):
        from pathlib import Path
        from types import SimpleNamespace

        monkeypatch.setattr(
            cli_main.subprocess,
            "run",
            lambda cmd, **kw: SimpleNamespace(returncode=128, stdout="", stderr="bad ref"),
        )
        assert (
            cli_main._release_age_preflight_git_ref(
                ["git"], "origin/xyz", "2026-05-10T00:00:00Z", Path("/tmp")
            )
            is None
        )

    def test_git_ref_preflight_delegates_pyproject_to_blocked_check(self, monkeypatch):
        from pathlib import Path
        from types import SimpleNamespace

        monkeypatch.setattr(
            cli_main.subprocess,
            "run",
            lambda cmd, **kw: SimpleNamespace(
                returncode=0, stdout='[project]\nname="x"\n', stderr=""
            ),
        )
        captured = {}

        def fake_blocked(text, cutoff):
            captured["text"] = text
            captured["cutoff"] = cutoff
            return "blocked-reason"

        monkeypatch.setattr(cli_main, "_release_age_preflight_blocked", fake_blocked)
        assert (
            cli_main._release_age_preflight_git_ref(
                ["git"], "abc123", "2026-05-10T00:00:00Z", Path("/tmp")
            )
            == "blocked-reason"
        )
        assert captured["cutoff"] == "2026-05-10T00:00:00Z"

    def test_fails_open_when_uv_lacks_gate_flags(self, monkeypatch):
        """An older uv without --exclude-newer would fail the gated runs with
        a usage error while the ungated control succeeds — indistinguishable
        from a real rejection. The capability probe must fail open instead."""
        from types import SimpleNamespace

        compile_calls: list[list[str]] = []

        def fake_run(cmd, **kw):
            if "--help" in cmd:
                return SimpleNamespace(
                    returncode=0, stdout="an old uv with no such flags", stderr=""
                )
            compile_calls.append(list(cmd))
            return SimpleNamespace(returncode=2, stdout="", stderr="unexpected argument")

        monkeypatch.setattr(cli_main.subprocess, "run", fake_run)
        monkeypatch.setattr("hermes_cli.managed_uv.ensure_uv", lambda: "/usr/bin/uv")

        assert (
            cli_main._release_age_preflight_blocked(
                self._PYPROJECT, "2026-05-10T00:00:00Z"
            )
            is None
        )
        assert compile_calls == []  # probe failed → no resolution attempted

    def test_resolves_against_target_venv_python(self, monkeypatch, tmp_path):
        """Resolution must use the interpreter the update installs into —
        the uv on PATH may be bound to a different Python, skewing marker
        evaluation and wheel tags."""
        from types import SimpleNamespace

        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python").write_text("")
        monkeypatch.setattr(cli_main, "PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(cli_main, "_is_windows", lambda: False)

        calls: list[list[str]] = []

        def fake_run(cmd, **kw):
            if "--help" in cmd:
                return SimpleNamespace(
                    returncode=0,
                    stdout="--exclude-newer --no-build --offline",
                    stderr="",
                )
            calls.append(list(cmd))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(cli_main.subprocess, "run", fake_run)
        monkeypatch.setattr("hermes_cli.managed_uv.ensure_uv", lambda: "/usr/bin/uv")

        cli_main._release_age_preflight_blocked(
            self._PYPROJECT, "2026-05-10T00:00:00Z"
        )
        assert calls, "expected at least one resolve"
        assert all(
            "--python" in c and str(venv_bin / "python") in c for c in calls
        )


class TestGatewayModeStatus:
    """Gateway output is streamed to the messaging user: informational
    confirmations stay quiet, but a bypassed gate must still warn."""

    def test_gateway_mode_suppresses_active_confirmation(self, monkeypatch, capsys):
        monkeypatch.setattr(cli_main, "_get_min_release_age_days", lambda: 3)
        cli_main._print_release_age_status(uv_available=True, gateway_mode=True)
        assert capsys.readouterr().out == ""

    def test_gateway_mode_still_warns_when_gate_bypassed(self, monkeypatch, capsys):
        monkeypatch.setattr(cli_main, "_get_min_release_age_days", lambda: 3)
        cli_main._print_release_age_status(uv_available=False, gateway_mode=True)
        out = capsys.readouterr().out
        assert "release-age gate configured" in out


class TestReleaseAgeDaysClamp:
    def test_huge_value_clamped_instead_of_overflowing(self, monkeypatch):
        """A typo'd huge day count must not crash the datetime subtraction
        (OverflowError) and brick every update."""
        with patch(
            "hermes_cli.config.load_config",
            return_value={"security": {"minimum_release_age_days": 10**9}},
        ):
            days = cli_main._get_min_release_age_days()
        assert days == 36500
        # And the cutoff computation stays representable.
        assert cli_main._exclude_newer_date(days) is not None

    def test_lazy_refresh_prefers_managed_uv_over_system_uv(self, monkeypatch):
        """A stale system uv that predates UV_EXCLUDE_NEWER would ignore the
        gate env var — the managed uv dir must shadow it while gated."""
        import os

        seen: dict[str, str | None] = {}

        class _FakeLazyDeps:
            @staticmethod
            def active_features():
                return ["voice"]

            @staticmethod
            def refresh_active_features(prompt=False):
                seen["path"] = os.environ.get("PATH")
                return {"voice": "current"}

        monkeypatch.delenv("UV_EXCLUDE_NEWER", raising=False)
        monkeypatch.setattr(cli_main, "_get_min_release_age_days", lambda: 3)
        monkeypatch.setattr(
            cli_main, "_exclude_newer_date", lambda days, **kw: "2026-05-10T00:00:00Z"
        )
        # System uv IS discoverable — managed must still shadow it.
        monkeypatch.setattr(cli_main.shutil, "which", lambda name: "/usr/bin/uv")
        monkeypatch.setattr(
            "hermes_cli.managed_uv.ensure_uv", lambda: "/fake/hermes-home/bin/uv"
        )
        import sys as _sys

        monkeypatch.setitem(_sys.modules, "tools.lazy_deps", _FakeLazyDeps())
        import tools

        monkeypatch.setattr(tools, "lazy_deps", _FakeLazyDeps(), raising=False)
        cli_main._refresh_active_lazy_features()

        assert seen["path"].startswith("/fake/hermes-home/bin" + os.pathsep)
