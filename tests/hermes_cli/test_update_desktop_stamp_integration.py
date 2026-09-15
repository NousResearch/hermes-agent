"""Desktop recovery uses the durable stamp even through an older updater facade.

Based on Realmagnum #92622 and eman717's #90495 comment5557504436.
"""

from types import SimpleNamespace

import pytest

from hermes_cli import update_cmd, update_cmd_deps


@pytest.mark.parametrize("build_exit", [0, 1])
def test_missing_desktop_with_surviving_stamp_attempts_build(tmp_path, monkeypatch, build_exit):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "desktop-build-stamp.json").write_text("{}", encoding="utf-8")
    desktop = tmp_path / "checkout" / "apps" / "desktop"
    desktop.mkdir(parents=True)
    (desktop / "package.json").write_text("{}", encoding="utf-8")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=build_exit, stdout="fixture build")

    # The old already-imported facade has no _desktop_stamp_path export.
    frozen = SimpleNamespace(
        PROJECT_ROOT=desktop.parent.parent,
        _desktop_packaged_executable=lambda _: None,
        _desktop_dist_exists=lambda _: False,
        _resolve_node_runtime_npm=lambda: "fixture-npm",
        _desktop_build_needed=lambda *a, **kw: True,
        _run_logged_subprocess=run,
    )
    monkeypatch.setattr(update_cmd, "_m", lambda: frozen)
    monkeypatch.setattr("hermes_constants.with_hermes_node_path", lambda: {})
    assert update_cmd_deps._rebuild_desktop_after_update(
        desktop, had_desktop_app_before_update=False) is (build_exit == 0)
    assert calls
    assert all(command[-2:] == ["desktop", "--build-only"] for command in calls)


def test_never_installed_desktop_does_not_build(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    desktop = tmp_path / "apps" / "desktop"
    desktop.mkdir(parents=True)
    (desktop / "package.json").write_text("{}", encoding="utf-8")
    frozen = SimpleNamespace(
        _desktop_packaged_executable=lambda _: None,
        _desktop_dist_exists=lambda _: False,
        _resolve_node_runtime_npm=lambda: "fixture-npm",
        _run_logged_subprocess=lambda *a, **kw: pytest.fail("never-installed Desktop built"),
    )
    monkeypatch.setattr(update_cmd, "_m", lambda: frozen)
    assert update_cmd_deps._rebuild_desktop_after_update(
        desktop, had_desktop_app_before_update=False)
