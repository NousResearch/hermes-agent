"""Regression coverage for the updater self-lock deferral (#83569, #86735).

``_detect_venv_python_processes`` excludes the calling process and its
ancestors by design — a CLI ``hermes update`` IS the venv python.  Before
this guard, an updater that had already imported a native venv extension
(e.g. ``cryptography.hazmat.bindings._rust``) died mid-sync with
``os error 5`` when ``uv`` tried to replace the mapped ``.pyd``, stranding
the venv half-updated (#83569).

The first version of the guard (#86687) fired as a PRE-FETCH preflight and
keyed on nothing but "is the module in sys.modules" — which was ALWAYS true
while ``bitwarden.py`` imported cryptography at module level, so every
Windows ``hermes update`` exited 2 before even fetching and the update
looped forever (#86735 / #86780 / #86781).  The guard is now honest:

* it only reports a loaded module when the dependency sync would actually
  REWRITE that distribution (``_dependency_sync_would_rewrite`` — installed
  version vs the on-disk pyproject pins);
* it runs AFTER the code swap, immediately before the venv rewrite
  (``_abort_dependency_sync_if_self_locked``), so a deferral leaves the
  user on NEW code with only the dependency install pending.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import hermes_cli.main as cli_main
import hermes_cli.update_cmd as update_cmd

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# _dependency_sync_would_rewrite
# ---------------------------------------------------------------------------


class TestDependencySyncWouldRewrite:
    def _with_pyproject(self, tmp_path, body: str):
        (tmp_path / "pyproject.toml").write_text(body, encoding="utf-8")
        return patch.object(cli_main, "PROJECT_ROOT", tmp_path)

    def test_satisfied_pin_means_no_rewrite(self, tmp_path):
        with self._with_pyproject(
            tmp_path,
            '[project]\nname = "x"\ndependencies = ["cryptography==1.2.3"]\n',
        ), patch("importlib.metadata.version", return_value="1.2.3"):
            assert cli_main._dependency_sync_would_rewrite("cryptography") is False

    def test_stale_pin_means_rewrite(self, tmp_path):
        with self._with_pyproject(
            tmp_path,
            '[project]\nname = "x"\ndependencies = ["cryptography==2.0.0"]\n',
        ), patch("importlib.metadata.version", return_value="1.2.3"):
            assert cli_main._dependency_sync_would_rewrite("cryptography") is True

    def test_missing_distribution_means_rewrite(self, tmp_path):
        with self._with_pyproject(
            tmp_path,
            '[project]\nname = "x"\ndependencies = ["cryptography==2.0.0"]\n',
        ), patch(
            "importlib.metadata.version",
            side_effect=Exception("not installed"),
        ):
            assert cli_main._dependency_sync_would_rewrite("cryptography") is True

    def test_pin_in_optional_extra_is_honored(self, tmp_path):
        body = textwrap.dedent(
            """
            [project]
            name = "x"
            dependencies = []
            [project.optional-dependencies]
            all = ["cryptography>=2,<3"]
            """
        )
        with self._with_pyproject(tmp_path, body), patch(
            "importlib.metadata.version", return_value="2.5.0"
        ):
            assert cli_main._dependency_sync_would_rewrite("cryptography") is False
        with self._with_pyproject(tmp_path, body), patch(
            "importlib.metadata.version", return_value="1.0.0"
        ):
            assert cli_main._dependency_sync_would_rewrite("cryptography") is True

    def test_unpinned_distribution_is_unknown(self, tmp_path):
        with self._with_pyproject(
            tmp_path, '[project]\nname = "x"\ndependencies = ["requests"]\n'
        ), patch("importlib.metadata.version", return_value="1.0.0"):
            assert cli_main._dependency_sync_would_rewrite("pyyaml") is None

    def test_broken_pyproject_is_unknown_never_raises(self, tmp_path):
        with self._with_pyproject(tmp_path, "not [valid toml"), patch(
            "importlib.metadata.version", return_value="1.0.0"
        ):
            assert cli_main._dependency_sync_would_rewrite("cryptography") is None


# ---------------------------------------------------------------------------
# _detect_self_loaded_native_modules — version-gated detection
# ---------------------------------------------------------------------------


@patch.object(cli_main, "_is_windows", return_value=False)
def test_self_lock_detection_is_noop_off_windows(_winp):
    with patch.dict(sys.modules, {"cryptography.hazmat.bindings._rust": MagicMock()}):
        assert cli_main._detect_self_loaded_native_modules() == []


@patch.object(cli_main, "_is_windows", return_value=True)
def test_loaded_module_with_pending_version_change_is_flagged(_winp):
    with patch.dict(
        sys.modules, {"cryptography.hazmat.bindings._rust": MagicMock()}
    ), patch.object(cli_main, "_dependency_sync_would_rewrite", return_value=True):
        assert "cryptography (_rust.pyd)" in cli_main._detect_self_loaded_native_modules()


@patch.object(cli_main, "_is_windows", return_value=True)
def test_loaded_module_already_at_target_version_is_not_flagged(_winp):
    """#86735 core fix: loaded-but-not-changing must NOT trip the deferral.

    Installed cryptography already satisfies the on-disk pins → the sync
    will not touch ``_rust.pyd`` → no lock risk → the update must proceed.
    """
    with patch.dict(
        sys.modules, {"cryptography.hazmat.bindings._rust": MagicMock()}
    ), patch.object(cli_main, "_dependency_sync_would_rewrite", return_value=False):
        assert cli_main._detect_self_loaded_native_modules() == []


@patch.object(cli_main, "_is_windows", return_value=True)
def test_unknown_rewrite_risk_fails_open(_winp):
    """Uncertain rewrite risk must NOT defer (None → proceed).

    PyYAML is loaded by every CLI process; deferring on an unparseable
    pyproject would recreate the #86735 always-firing loop. The missed-
    deferral downside is the pre-existing mid-sync failure, which the
    marker recovery already handles.
    """
    with patch.dict(
        sys.modules, {"cryptography.hazmat.bindings._rust": MagicMock()}
    ), patch.object(cli_main, "_dependency_sync_would_rewrite", return_value=None):
        assert cli_main._detect_self_loaded_native_modules() == []


@patch.object(cli_main, "_is_windows", return_value=True)
def test_self_lock_detection_clean_when_rust_not_loaded(_winp):
    sys.modules.pop("cryptography.hazmat.bindings._rust", None)
    with patch.object(cli_main, "_dependency_sync_would_rewrite", return_value=True):
        assert (
            "cryptography (_rust.pyd)"
            not in cli_main._detect_self_loaded_native_modules()
        )


# ---------------------------------------------------------------------------
# _abort_dependency_sync_if_self_locked — deferral wiring
# ---------------------------------------------------------------------------


def test_abort_helper_noop_when_nothing_at_risk():
    with patch.object(
        cli_main, "_detect_self_loaded_native_modules", return_value=[]
    ), patch.object(cli_main, "_defer_update_for_self_lock") as defer:
        cli_main._abort_dependency_sync_if_self_locked()
    defer.assert_not_called()


def test_abort_helper_defers_and_exits_2(capsys):
    marker_writes = []
    with patch.object(
        cli_main,
        "_detect_self_loaded_native_modules",
        return_value=["cryptography (_rust.pyd)"],
    ), patch.object(
        cli_main,
        "_write_update_incomplete_marker",
        side_effect=lambda: marker_writes.append("written"),
    ), pytest.raises(SystemExit) as exc:
        cli_main._abort_dependency_sync_if_self_locked()
    assert exc.value.code == 2
    # Deferral contract: marker dropped so the next fresh launch completes
    # the dependency install (the code swap has already happened by now).
    assert marker_writes == ["written"]
    out = capsys.readouterr().out
    assert "cryptography (_rust.pyd)" in out
    assert "deferred" in out


def test_abort_helper_resumes_paused_gateways_before_exit():
    resume_calls = []
    sentinel = object()
    with patch.object(
        cli_main,
        "_detect_self_loaded_native_modules",
        return_value=["cryptography (_rust.pyd)"],
    ), patch.object(cli_main, "_write_update_incomplete_marker"), patch.object(
        cli_main,
        "_resume_windows_gateways_after_update",
        side_effect=lambda token: resume_calls.append(token),
    ), pytest.raises(SystemExit):
        cli_main._abort_dependency_sync_if_self_locked(sentinel)
    assert resume_calls == [sentinel]


# ---------------------------------------------------------------------------
# Placement: the deferral must NOT fire before the fetch (#86735 / #86780)
# ---------------------------------------------------------------------------


def test_pre_fetch_flow_has_no_self_lock_preflight():
    """#86780 regression: a loaded native module must not block the git fetch.

    The old preflight sat between the venv-holder sweep and the fetch, so a
    universally-loaded module (bitwarden's module-level cryptography import)
    deferred every update before any code was pulled — the Windows infinite
    update loop.  The deferral now lives at the dependency-sync boundaries
    only; the stretch of _cmd_update_impl between the venv-holder sweep and
    the fetch must not consult the detector at all.
    """
    import inspect

    src = inspect.getsource(update_cmd._cmd_update_impl)
    fetch_idx = src.index("Fetching updates")
    pre_fetch = src[:fetch_idx]
    assert "_detect_self_loaded_native_modules()" not in pre_fetch
    assert "_m()._abort_dependency_sync_if_self_locked" not in pre_fetch
    # ... and it must still guard the dependency sync after the code swap.
    post_fetch = src[fetch_idx:]
    assert "_m()._abort_dependency_sync_if_self_locked" in post_fetch


def test_zip_update_guards_dependency_sync():
    import inspect

    src = inspect.getsource(update_cmd._update_via_zip)
    swap_idx = src.index("Updating Python dependencies")
    assert "_abort_dependency_sync_if_self_locked" in src[:swap_idx]


# ---------------------------------------------------------------------------
# Import hygiene: the update entrypoint must not load cryptography at all
# ---------------------------------------------------------------------------


class TestUpdateEntrypointImportHygiene:
    """Subprocess-verified: the CLI/update path never eagerly loads crypto.

    Empirical sys.modules assertions in a clean interpreter — the technique
    that pinned #86735's root cause.  If any future module-level import of
    cryptography (or another self-locking native module) creeps back into
    the startup path, these fail before a Windows user ever hits the loop.
    """

    def _run(self, code: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=120,
        )

    def test_import_main_does_not_load_self_locking_modules(self):
        result = self._run(
            textwrap.dedent(
                """
                import sys
                import hermes_cli.main
                from hermes_cli.update_cmd import _SELF_LOCKING_NATIVE_MODULES
                loaded = [
                    p for p in _SELF_LOCKING_NATIVE_MODULES
                    if p in sys.modules and not p.startswith("yaml")
                ]
                # PyYAML is a base dep every CLI process needs (config parsing);
                # it is version-gated instead of import-gated. Everything else
                # in the registry must stay lazy.
                assert not loaded, f"eagerly loaded self-locking modules: {loaded}"
                assert "cryptography.hazmat.bindings._rust" not in sys.modules
                print("OK")
                """
            )
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "OK" in result.stdout

    def test_update_dispatch_does_not_load_cryptography(self):
        result = self._run(
            textwrap.dedent(
                """
                import sys
                from unittest.mock import patch
                sys.argv = ["hermes", "update", "--check"]
                import hermes_cli.main as m
                with patch("hermes_cli.main._cmd_update_check", lambda *a, **k: 0):
                    try:
                        m.main()
                    except SystemExit:
                        pass
                assert "cryptography.hazmat.bindings._rust" not in sys.modules, (
                    "update dispatch eagerly loaded cryptography._rust"
                )
                print("OK")
                """
            )
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "OK" in result.stdout
