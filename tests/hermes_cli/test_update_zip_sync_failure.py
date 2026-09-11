"""A failed dependency transaction must stop the ZIP update, not report success."""
import pytest

import pm
import hermes_cli.main as main
from hermes_cli import update_cmd_zip


def test_zip_dependency_failure_propagates_before_followup_mutations(monkeypatch):
    monkeypatch.setattr(pm, "ensure", lambda *_args, **_kwargs: None)
    def fail(*args, **kwargs):
        raise pm.InstallError("venv", "network unavailable")
    monkeypatch.setattr(pm, "sync_venv", fail)
    def forbidden(*args, **kwargs):
        raise AssertionError("follow-up mutation after failed sync")
    monkeypatch.setattr(main, "_refresh_active_memory_provider_dependencies", forbidden)
    with pytest.raises(pm.InstallError, match="network unavailable"):
        update_cmd_zip._reinstall_python_deps_after_zip()


def test_pull_dependency_failure_keeps_recovery_marker(tmp_path, monkeypatch):
    from hermes_cli import update_cmd_deps, update_cmd

    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(main, "_refuse_update_for_contended_shims", lambda *a: None, raising=False)
    monkeypatch.setattr(update_cmd, "_write_update_incomplete_marker", lambda: None)
    def fail(*a, **k):
        raise pm.InstallError("venv", "sync stopped")
    monkeypatch.setattr(pm, "sync_venv", fail)
    def forbidden(*a, **k):
        raise AssertionError("must retain recovery marker")
    monkeypatch.setattr(main, "_clear_update_incomplete_marker", forbidden)
    with pytest.raises(pm.InstallError, match="sync stopped"):
        update_cmd_deps._sync_python_dependencies_after_pull(
            ["git"], "main", "a" * 40, active_lazy_features=[],
            _windows_gateway_resume=[], desktop_dir=tmp_path, had_desktop_app_before_update=False,
        )
