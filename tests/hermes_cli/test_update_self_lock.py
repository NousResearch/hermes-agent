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
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import hermes_cli.main as cli_main
from hermes_cli import _early_recovery

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

    def test_unpinned_transitive_moves_with_its_pinned_parent(self, tmp_path):
        """#81594: ``_cffi_backend`` belongs to cffi, which no pyproject line pins; it moves
        only because cryptography's base pin moved and uv.lock resolves cffi elsewhere. Once the
        parent is at its pin the transitive stays put (fail-open, #86735) — even while an
        optional-extra parent (brotlicffi) is missing, which a skipped extra can leave forever."""
        (tmp_path / "uv.lock").write_text(
            textwrap.dedent(
                """
                version = 1
                [[package]]
                name = "brotlicffi"
                version = "1.2.0.2"
                dependencies = [{ name = "cffi" }]
                [[package]]
                name = "cffi"
                version = "2.0.0"
                [[package]]
                name = "cryptography"
                version = "50.0.0"
                dependencies = [{ name = "cffi" }]
                """
            ),
            encoding="utf-8",
        )
        pyproject = textwrap.dedent(
            """
            [project]
            name = "x"
            dependencies = ["cryptography==50.0.0"]
            [project.optional-dependencies]
            messaging = ["brotlicffi==1.2.0.2"]
            """
        )
        for crypto, expected in (("49.0.0", True), ("50.0.0", None)):
            installed = {"cffi": "1.17.1", "cryptography": crypto}
            with self._with_pyproject(tmp_path, pyproject), patch(
                "importlib.metadata.version", side_effect=installed.__getitem__
            ):
                assert cli_main._dependency_sync_would_rewrite("cffi") is expected


# ---------------------------------------------------------------------------
# _detect_self_loaded_native_modules — version-gated detection
# ---------------------------------------------------------------------------


def _native_module(site: Path, name: str, suffix: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = str(site.joinpath(*name.split(".")).with_name(name.rsplit(".", 1)[-1] + suffix))
    return module


def test_native_extension_scan_derives_every_loaded_venv_extension(tmp_path):
    """#81594: the report's locked file was ``_cffi_backend.pyd``, which no hand list named.
    Every loaded extension under site-packages maps to its distribution; stdlib extensions
    and pure-Python modules never do."""
    from hermes_cli.update_cmd_deps import _loaded_native_extension_dists

    site, stdlib = tmp_path / "site-packages", tmp_path / "DLLs"
    modules = {
        "_cffi_backend": _native_module(site, "_cffi_backend", ".pyd"),
        "yaml._yaml": _native_module(site, "yaml._yaml", ".pyd"),
        "yaml": _native_module(site, "yaml", ".py"),
        "_ssl": _native_module(stdlib, "_ssl", ".pyd"),
        "builtin_like": types.ModuleType("builtin_like"),
    }
    loaded = _loaded_native_extension_dists(
        modules, [site], (".pyd",),
        {"_cffi_backend": ["cffi"], "yaml": ["PyYAML"], "_ssl": ["nope"]})

    assert loaded == {"cffi": ["_cffi_backend.pyd"], "PyYAML": ["_yaml.pyd"]}


@pytest.mark.linux_only
def test_self_lock_detection_is_noop_off_windows():
    with patch.dict(sys.modules, {"cryptography.hazmat.bindings._rust": MagicMock()}):
        assert cli_main._detect_self_loaded_native_modules() == []


@pytest.mark.windows_only
def test_loaded_module_with_pending_version_change_is_flagged():
    import sysconfig

    site = Path(sysconfig.get_path("purelib"))
    with patch.dict(
        sys.modules, {"_cffi_backend": _native_module(site, "_cffi_backend", ".pyd")}
    ), patch(
        "importlib.metadata.packages_distributions", return_value={"_cffi_backend": ["cffi"]}
    ), patch.object(cli_main, "_dependency_sync_would_rewrite", return_value=True):
        assert "cffi (_cffi_backend.pyd)" in cli_main._detect_self_loaded_native_modules()


@pytest.mark.windows_only
def test_loaded_module_already_at_target_version_is_not_flagged():
    """#86735 core fix: loaded-but-not-changing must NOT trip the deferral.

    Installed cryptography already satisfies the on-disk pins → the sync
    will not touch ``_rust.pyd`` → no lock risk → the update must proceed.
    """
    with patch.dict(
        sys.modules, {"cryptography.hazmat.bindings._rust": MagicMock()}
    ), patch.object(cli_main, "_dependency_sync_would_rewrite", return_value=False):
        assert cli_main._detect_self_loaded_native_modules() == []


@pytest.mark.windows_only
def test_unknown_rewrite_risk_fails_open():
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


def test_recovered_update_retry_builds_parser_without_native_secret_modules(
    monkeypatch,
):
    """Parser registration must not re-lock a freshly recovered updater."""
    called = []
    monkeypatch.setattr(_early_recovery, "_UPDATE_RETRY_RECOVERED", True)
    monkeypatch.setattr(cli_main, "cmd_update", lambda args: called.append(args))
    monkeypatch.setattr(sys, "argv", ["hermes", "update", "--yes"])
    sys.modules.pop("cryptography.hazmat.bindings._rust", None)

    cli_main.main()

    assert len(called) == 1
    assert "cryptography.hazmat.bindings._rust" not in sys.modules


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
    assert "cryptography (_rust.pyd)" in capsys.readouterr().out


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
                # PyYAML is a base dep every CLI process needs (config parsing);
                # it is version-gated instead of import-gated. The crypto stack
                # (cryptography's _rust.pyd and cffi's _cffi_backend.pyd) must stay lazy.
                loaded = [
                    m for m in ("cryptography.hazmat.bindings._rust", "_cffi_backend")
                    if m in sys.modules
                ]
                assert not loaded, f"eagerly loaded self-locking modules: {loaded}"
                print("OK")
                """
            )
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "OK" in result.stdout

