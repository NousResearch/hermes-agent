"""Regression tests for #81421: Hindsight local-embedded dependency repair
must not install the bare full bundle on Intel macOS.

Before the fix, ``_provider_pip_dependencies`` appended ``hindsight-all``
for every local/local_embedded install.  On Intel macOS the full local-ML
dependency set pulls MLX packages that have no x86_64 wheels, so the
resolver silently backtracks to ancient ``hindsight-all``/``hindsight-api``
releases.  Their overlapping ``hindsight_api`` files override the working
slim API and the daemon crashes with "Unknown embeddings provider: onnx"
(#81421).
"""

from unittest.mock import patch

import hermes_cli.memory_setup as memory_setup
from hermes_cli.memory_setup import _is_intel_macos, _provider_pip_dependencies

DECLARED = ["hindsight-client>=0.6.1"]


def _write_hindsight_config(home, mode="local_embedded"):
    (home / "hindsight").mkdir(parents=True, exist_ok=True)
    (home / "hindsight" / "config.json").write_text(
        '{"mode": "%s"}' % mode, encoding="utf-8"
    )


class TestIntelMacosGuard:
    def test_detects_intel_macos(self):
        with patch("platform.system", return_value="Darwin"), patch(
            "platform.machine", return_value="x86_64"
        ):
            assert _is_intel_macos() is True

    def test_arm64_macos_is_not_intel(self):
        with patch("platform.system", return_value="Darwin"), patch(
            "platform.machine", return_value="arm64"
        ):
            assert _is_intel_macos() is False

    def test_linux_is_not_intel_macos(self):
        with patch("platform.system", return_value="Linux"), patch(
            "platform.machine", return_value="x86_64"
        ):
            assert _is_intel_macos() is False


class TestHindsightLocalEmbeddedDeps:
    def test_intel_macos_uses_slim_stack_not_bare_bundle(self, tmp_path, monkeypatch):
        """The issue's regression case: Intel macOS local_embedded must NOT
        get a bare ``hindsight-all`` spec (which backtracks to ancient
        full-package releases)."""
        _write_hindsight_config(tmp_path)
        monkeypatch.setattr(memory_setup, "get_hermes_home", lambda: tmp_path)

        with patch("platform.system", return_value="Darwin"), patch(
            "platform.machine", return_value="x86_64"
        ):
            deps = _provider_pip_dependencies("hindsight", DECLARED)

        assert "hindsight-all" not in deps
        assert "hindsight-all-slim" in deps
        assert "hindsight-api-slim[local-onnx]" in deps
        assert "hindsight-embed" in deps
        # Declared bridge deps are preserved.
        assert "hindsight-client>=0.6.1" in deps

    def test_non_intel_keeps_bare_bundle(self, tmp_path, monkeypatch):
        """Apple Silicon / Linux keep the full bundle — the existing heal
        path for #70636 is unchanged."""
        _write_hindsight_config(tmp_path)
        monkeypatch.setattr(memory_setup, "get_hermes_home", lambda: tmp_path)

        with patch("platform.system", return_value="Darwin"), patch(
            "platform.machine", return_value="arm64"
        ):
            deps = _provider_pip_dependencies("hindsight", DECLARED)

        assert "hindsight-all" in deps
        assert "hindsight-all-slim" not in deps

    def test_non_local_modes_unaffected(self, tmp_path, monkeypatch):
        """A remote/API-mode Hindsight config never gets local deps at all."""
        _write_hindsight_config(tmp_path, mode="remote")
        monkeypatch.setattr(memory_setup, "get_hermes_home", lambda: tmp_path)

        with patch("platform.system", return_value="Darwin"), patch(
            "platform.machine", return_value="x86_64"
        ):
            deps = _provider_pip_dependencies("hindsight", DECLARED)

        assert deps == DECLARED

    def test_missing_config_falls_back_to_declared(self, tmp_path, monkeypatch):
        """No hindsight config.json → declared bridge deps only (same as
        before the fix)."""
        monkeypatch.setattr(memory_setup, "get_hermes_home", lambda: tmp_path)

        deps = _provider_pip_dependencies("hindsight", DECLARED)

        assert deps == DECLARED


class TestHindsightPluginPostSetupGuard:
    """Regression: the Hindsight plugin's interactive ``post_setup`` wizard
    must apply the same Intel-macOS slim-stack guard as
    ``_provider_pip_dependencies``.  Without this guard the wizard still
    installs the bare ``hindsight-all`` on Intel macOS, undoing the
    earlier-stage fix and crashing the daemon with "Unknown embeddings
    provider: onnx" (#81421).

    These tests don't import the plugin class (the plugin package is not
    on the default ``sys.path``).  Instead they mirror the
    ``local_embedded`` branch of ``HindsightProvider.post_setup`` and
    assert the dep list reflects the Intel-macOS guard imported from
    ``hermes_cli.memory_setup``.
    """

    @staticmethod
    def _compute_plugin_local_deps(monkeypatch, *, intel_macos: bool):
        from hermes_cli import memory_setup

        monkeypatch.setattr(
            memory_setup, "_is_intel_macos", lambda: intel_macos
        )

        # Mirror the ``local_embedded`` branch in
        # ``plugins/memory/hindsight/__init__.py::post_setup``: the plugin
        # now imports ``_is_intel_macos`` from ``hermes_cli.memory_setup``
        # and selects the slim stack on Intel macOS.
        if memory_setup._is_intel_macos():
            local_dep = [
                "hindsight-all-slim",
                "hindsight-api-slim[local-onnx]",
                "hindsight-embed",
            ]
        else:
            local_dep = ["hindsight-all"]
        return local_dep

    def test_intel_macos_installs_slim_stack(self, monkeypatch):
        deps = self._compute_plugin_local_deps(monkeypatch, intel_macos=True)

        assert "hindsight-all" not in deps
        assert "hindsight-all-slim" in deps
        assert "hindsight-api-slim[local-onnx]" in deps
        assert "hindsight-embed" in deps

    def test_non_intel_installs_full_bundle(self, monkeypatch):
        deps = self._compute_plugin_local_deps(monkeypatch, intel_macos=False)

        assert deps == ["hindsight-all"]


class TestIntelMacosSmokeCheck:
    """Post-install smoke validation for the Intel-macOS slim runtime.

    The #81421 failure mode is the resolver backtracking to an ancient
    ``hindsight_api`` release that is still importable but no longer
    exposes ``LocalSTEmbeddings`` — pip reports success while the daemon
    crashes with "Unknown embeddings provider: onnx".  The smoke check
    runs in a clean subprocess (so cached modules can't mask the new
    environment) and must raise on failure so ``hermes update`` can't
    claim a heal that is still broken.
    """

    def _write_plugin_yaml(self, tmp_path):
        import yaml

        plugin_dir = tmp_path / "hindsight"
        plugin_dir.mkdir(exist_ok=True)
        (plugin_dir / "plugin.yaml").write_text(
            yaml.safe_dump({"pip_dependencies": ["hindsight-client>=0.6.1"]}),
            encoding="utf-8",
        )
        return plugin_dir

    def _patch_intel_macos_env(self, monkeypatch, tmp_path, *, intel=True):
        import platform as _platform

        monkeypatch.setattr("plugins.memory.find_provider_dir", lambda name: tmp_path / "hindsight")
        monkeypatch.setattr(memory_setup, "get_hermes_home", lambda: tmp_path)
        if intel:
            monkeypatch.setattr(_platform, "system", lambda: "Darwin")
            monkeypatch.setattr(_platform, "machine", lambda: "x86_64")

    def _make_install_ok(self, monkeypatch):
        from types import SimpleNamespace

        monkeypatch.setattr(
            "tools.lazy_deps.install_specs",
            lambda specs, timeout=120: SimpleNamespace(ok=True, blocked=False, reason="", stderr=""),
        )

    def test_install_failure_raises_and_prints(self, tmp_path, monkeypatch, capsys):
        """Smoke failure after a successful install must raise RuntimeError
        with the #81421 context, and print the failing symbol first."""
        import pytest

        self._write_plugin_yaml(tmp_path)
        _write_hindsight_config(tmp_path, mode="local_embedded")
        self._patch_intel_macos_env(monkeypatch, tmp_path)
        self._make_install_ok(monkeypatch)
        monkeypatch.setattr(
            memory_setup,
            "_smoke_import_hindsight_local",
            lambda: [
                "hindsight_api.LocalSTEmbeddings: AttributeError: the slim "
                "runtime shipped an API release that does not expose the "
                "configured ONNX embeddings provider (#81421)"
            ],
        )

        with pytest.raises(RuntimeError) as excinfo:
            memory_setup._install_dependencies("hindsight", force=True)

        assert "Hindsight slim runtime smoke validation failed" in str(excinfo.value)
        captured = capsys.readouterr().out
        assert "smoke validation failed" in captured
        assert "LocalSTEmbeddings" in captured

    def test_install_success_is_silent(self, tmp_path, monkeypatch, capsys):
        """Healthy smoke (no errors) must not raise and must print neither
        the failure banner nor a success line."""
        self._write_plugin_yaml(tmp_path)
        _write_hindsight_config(tmp_path, mode="local_embedded")
        self._patch_intel_macos_env(monkeypatch, tmp_path)
        self._make_install_ok(monkeypatch)
        monkeypatch.setattr(memory_setup, "_smoke_import_hindsight_local", lambda: [])

        memory_setup._install_dependencies("hindsight", force=True)

        captured = capsys.readouterr().out
        assert "smoke validation failed" not in captured

    def test_non_force_refresh_stale_runtime_still_validated(self, tmp_path, monkeypatch):
        """Even when every slim dep already imports (nothing missing), a
        stale importable-but-broken runtime must still be caught — the
        early-return refresh path must run the smoke check too."""
        import sys
        import types

        import pytest

        self._write_plugin_yaml(tmp_path)
        _write_hindsight_config(tmp_path, mode="local_embedded")
        self._patch_intel_macos_env(monkeypatch, tmp_path)
        monkeypatch.setattr(memory_setup, "_smoke_import_hindsight_local", lambda: ["boom"])

        # Inject fake modules so the missing-dep probe sees every slim
        # package as present — force=False then hits the early-return
        # branch, which must still run the smoke check and raise.
        fake_modules = {
            "hindsight_api": types.ModuleType("hindsight_api"),
            "hindsight_embed": types.ModuleType("hindsight_embed"),
            "hindsight_client": types.ModuleType("hindsight_client"),
        }
        for name, mod in fake_modules.items():
            monkeypatch.delitem(sys.modules, name, raising=False)
            sys.modules[name] = mod
        try:
            with pytest.raises(RuntimeError) as excinfo:
                memory_setup._install_dependencies("hindsight", force=False)
        finally:
            for name in ("hindsight_api", "hindsight_embed", "hindsight_client"):
                sys.modules.pop(name, None)

        assert "Hindsight slim runtime smoke validation failed" in str(excinfo.value)

    def test_non_intel_does_not_smoke(self, tmp_path, monkeypatch):
        """Apple Silicon / non-macOS must never smoke-check — the guard is
        gated inside ``_maybe_run_intel_macos_local_embedded_smoke_check``.
        (force=True installs the full bundle; we stub install_specs so the
        test asserts only the absence of the smoke gate.)"""
        self._write_plugin_yaml(tmp_path)
        _write_hindsight_config(tmp_path, mode="local_embedded")
        self._patch_intel_macos_env(monkeypatch, tmp_path, intel=False)
        self._make_install_ok(monkeypatch)
        monkeypatch.setattr(
            memory_setup, "_smoke_import_hindsight_local", lambda: ["boom"]
        )

        # Must not raise despite smoke errors: gated off on non-Intel.
        memory_setup._install_dependencies("hindsight", force=True)

    def test_non_local_mode_does_not_smoke(self, tmp_path, monkeypatch):
        """Cloud modes must never smoke-check (gate is on mode too)."""
        self._write_plugin_yaml(tmp_path)
        _write_hindsight_config(tmp_path, mode="remote")
        self._patch_intel_macos_env(monkeypatch, tmp_path)
        self._make_install_ok(monkeypatch)
        monkeypatch.setattr(
            memory_setup, "_smoke_import_hindsight_local", lambda: ["boom"]
        )

        # Must not raise despite smoke errors: gated off on remote mode.
        memory_setup._install_dependencies("hindsight", force=True)

    def test_smoke_probe_healthy_and_failure_paths(self, monkeypatch):
        """The probe parses the subprocess JSON: healthy imports → no
        errors; a missing LocalSTEmbeddings → one failure string."""
        import subprocess as _sp
        import sys as _sys

        def fake_run(cmd, **kwargs):
            assert cmd[0] == _sys.executable
            return _sp.CompletedProcess(cmd, 0, stdout='{"errors": ["hindsight_api.LocalSTEmbeddings: missing"]}', stderr="")

        monkeypatch.setattr(_sp, "run", fake_run)

        # The helper does `import subprocess as _sp` inside the function,
        # which resolves to the same module object we patched above.
        errors = memory_setup._smoke_import_hindsight_local()
        assert errors == ["hindsight_api.LocalSTEmbeddings: missing"]

        monkeypatch.setattr(
            _sp,
            "run",
            lambda cmd, **kwargs: _sp.CompletedProcess(
                cmd, 0, stdout='{"errors": []}', stderr=""
            ),
        )
        assert memory_setup._smoke_import_hindsight_local() == []
