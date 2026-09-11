"""C16: hindsight run_setup dependency install must route through pm — the
declared ``extra: hindsight`` (uv.lock pin) for all modes — and the
local_embedded wizard must ACTUALLY install the isolated side-env runtime via
embedded_runtime.ensure_sideenv (hindsight-embed + hindsight-api-slim[all],
pinned, built by PM's isolated-environment operation). Never the deleted ``tools.lazy_deps``."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def setup_mod():
    import plugins.memory.hindsight.setup as mod
    return mod


def _fake_provider(existing=None):
    return SimpleNamespace(_config=existing or {}, save_config=MagicMock())


def _no_network_setup(monkeypatch, setup_mod, tmp_path, mode):
    """Drive run_setup non-interactively: fixed selections, empty secrets,
    templates disabled, profile-env materialization stubbed."""
    monkeypatch.setattr(setup_mod, "_select", lambda title, items, values, current: mode)
    monkeypatch.setattr(setup_mod, "_secret_prompt", lambda label: "")
    monkeypatch.setattr(setup_mod, "_materialize_embedded_profile_env", lambda *a, **kw: None)
    import builtins
    monkeypatch.setattr(builtins, "input", lambda prompt="": "")
    import plugins.memory.hindsight.templates as templates
    monkeypatch.setattr(templates, "supported_for_mode", lambda mode: False)
    import hermes_cli.config as hermes_config
    monkeypatch.setattr(hermes_config, "save_config", lambda config: None)
    return _fake_provider(), str(tmp_path / "home"), {"memory": {}}


class TestClientModeUsesDeclaredExtra:
    def test_cloud_mode_syncs_hindsight_extra(self, monkeypatch, setup_mod, tmp_path):
        calls = []
        import pm
        monkeypatch.setattr(pm, "sync_venv", lambda extras=None, *, explicit=False: calls.append((extras, explicit)))
        provider, home, config = _no_network_setup(monkeypatch, setup_mod, tmp_path, "cloud")

        setup_mod.run_setup(provider, home, config)

        assert calls == [(["hindsight"], True)]

    def test_local_external_mode_syncs_hindsight_extra(self, monkeypatch, setup_mod, tmp_path):
        calls = []
        import pm
        monkeypatch.setattr(pm, "sync_venv", lambda extras=None, *, explicit=False: calls.append((extras, explicit)))
        provider, home, config = _no_network_setup(monkeypatch, setup_mod, tmp_path, "local_external")

        setup_mod.run_setup(provider, home, config)

        assert calls == [(["hindsight"], True)]


class TestSyncFailureSurfaces:
    def test_sealed_refusal_warns_not_crashes(self, monkeypatch, setup_mod, tmp_path, capsys):
        import pm

        def refuse(extras=None, *, explicit=False):
            raise RuntimeError("venv refuses: frozen feature set")

        monkeypatch.setattr(pm, "sync_venv", refuse)
        provider, home, config = _no_network_setup(monkeypatch, setup_mod, tmp_path, "cloud")

        setup_mod.run_setup(provider, home, config)

        out = capsys.readouterr().out
        assert "frozen feature set" in out
        assert "hermes pm install" in out


class TestEmbeddedInstallsIsolatedSideEnv:
    def test_embedded_installs_isolated_runtime_not_shared_venv(self, monkeypatch, setup_mod, tmp_path, capsys):
        calls = []
        import pm
        monkeypatch.setattr(pm, "sync_venv", lambda extras=None, *, explicit=False: calls.append(extras))
        installed = []
        import plugins.memory.hindsight.embedded_runtime as rt
        monkeypatch.setattr(rt, "ensure_sideenv", lambda: installed.append("gen") or rt.sideenv_root())
        provider, home, config = _no_network_setup(monkeypatch, setup_mod, tmp_path, "local_embedded")

        setup_mod.run_setup(provider, home, config)

        out = capsys.readouterr().out
        # The client extra goes through pm; the heavy stack installs into the
        # isolated side env — never the shared venv.
        assert calls == [["hindsight"]]
        assert installed == ["gen"]
        assert "Isolated Hindsight runtime installed" in out
        assert "not modified" in out

    def test_embedded_install_failure_is_truthful_and_non_fatal(self, monkeypatch, setup_mod, tmp_path, capsys):
        import pm
        monkeypatch.setattr(pm, "sync_venv", lambda extras=None, *, explicit=False: None)
        import plugins.memory.hindsight.embedded_runtime as rt
        monkeypatch.setattr(rt, "ensure_sideenv",
                            lambda: (_ for _ in ()).throw(RuntimeError("uv sync failed: disk full")))
        provider, home, config = _no_network_setup(monkeypatch, setup_mod, tmp_path, "local_embedded")

        setup_mod.run_setup(provider, home, config)

        out = capsys.readouterr().out
        assert "Isolated runtime install failed" in out
        assert "disk full" in out
        provider.save_config.assert_not_called()
        assert config["memory"] == {}

    def test_embedded_config_still_saved(self, monkeypatch, setup_mod, tmp_path):
        import pm
        monkeypatch.setattr(pm, "sync_venv", lambda extras=None, *, explicit=False: None)
        import plugins.memory.hindsight.embedded_runtime as rt
        monkeypatch.setattr(rt, "ensure_sideenv", lambda: rt.sideenv_root())
        provider, home, config = _no_network_setup(monkeypatch, setup_mod, tmp_path, "local_embedded")

        setup_mod.run_setup(provider, home, config)

        saved = provider.save_config.call_args[0][0]
        assert saved["mode"] == "local_embedded"
        assert saved["bank_id"] == "hermes"


class TestNoLazyDeps:
    def test_tools_lazy_deps_absent_after_setup(self, monkeypatch, setup_mod, tmp_path):
        import pm
        monkeypatch.setattr(pm, "sync_venv", lambda extras=None, *, explicit=False: None)
        provider, home, config = _no_network_setup(monkeypatch, setup_mod, tmp_path, "cloud")

        setup_mod.run_setup(provider, home, config)

        assert "tools.lazy_deps" not in sys.modules
