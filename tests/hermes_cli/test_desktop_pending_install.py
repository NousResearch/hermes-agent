"""A desktop install skipped for a running bundle is recorded and completable (#123737).

``_install_rebuilt_macos_bundles`` correctly refuses to swap a RUNNING ``Hermes.app`` under,
but the refusal used to be log-only: the rebuilt bundle stayed staged, the update reported
success, and nothing ever completed the install. These pin the contract: the skip records
a pending install under HERMES_HOME, a later successful pass clears it, and
``hermes desktop finish-update`` completes the recorded install with the same machinery.
"""

import shutil

import pytest

from hermes_cli import main_desktop


def _bundle(root, asar):
    app = root / "Hermes.app"
    (app / "Contents" / "MacOS").mkdir(parents=True)
    (app / "Contents" / "MacOS" / "Hermes").write_bytes(b"\xcf\xfa\xed\xfe")
    (app / "Contents" / "Resources").mkdir()
    (app / "Contents" / "Resources" / "app.asar").write_bytes(asar)
    return app


def _asar(app):
    return (app / "Contents" / "Resources" / "app.asar").read_bytes()


def _stage_copy(src, dst):
    shutil.copytree(src, dst, symlinks=True)


@pytest.fixture
def rebuilt(tmp_path, monkeypatch):
    monkeypatch.setattr(main_desktop, "_stage_macos_bundle_copy", _stage_copy)
    return _bundle(tmp_path / "apps" / "desktop" / "release" / "mac-arm64", b"rebuilt")


def _record_path():
    return main_desktop.pending_desktop_install_path()


def test_running_skip_records_pending_install(rebuilt, tmp_path, monkeypatch):
    stale = _bundle(tmp_path / "Applications", b"stale")
    running = _bundle(tmp_path / "Volumes" / "Applications", b"older")

    installed, problems = main_desktop._install_rebuilt_macos_bundles(
        rebuilt, [stale, running], running={running.resolve()})

    # The non-running stale copy still installs; only the running one is skipped.
    assert installed == [stale]
    assert len(problems) == 1 and str(running) in problems[0]
    # Installing the SIBLING bundle must not erase the running bundle's record.
    assert main_desktop.read_pending_desktop_install() is not None
    # The skip is no longer log-only: the staged install is recorded for later completion.
    record = main_desktop.read_pending_desktop_install()
    assert record is not None
    assert record["app"] == str(running)
    assert record["rebuilt_app"] == str(rebuilt)
    assert record["asar_hash"] == main_desktop._app_asar_hash(rebuilt)
    assert _record_path().exists()


def test_successful_install_clears_pending_install(rebuilt, tmp_path):
    stale = _bundle(tmp_path / "Applications", b"stale")
    main_desktop.record_pending_desktop_install(
        app=stale, rebuilt_app=rebuilt, asar_hash=main_desktop._app_asar_hash(rebuilt))
    assert _record_path().exists()

    installed, problems = main_desktop._install_rebuilt_macos_bundles(rebuilt, [stale], running=set())

    assert installed == [stale] and problems == []
    assert not _record_path().exists()


def test_already_current_install_self_heals_the_record(rebuilt, tmp_path):
    current = _bundle(tmp_path / "Applications", b"rebuilt")
    main_desktop.record_pending_desktop_install(
        app=current, rebuilt_app=rebuilt, asar_hash=main_desktop._app_asar_hash(rebuilt))

    installed, problems = main_desktop._install_rebuilt_macos_bundles(rebuilt, [current], running=set())

    assert installed == [] and problems == []
    # The installed copy already matches the recorded hash: healed, record goes away.
    assert not _record_path().exists()


def test_finish_update_without_record_is_a_noop(rebuilt, tmp_path, capsys):
    main_desktop.cmd_desktop_finish_update(type("Args", (), {})())
    assert "No pending desktop install" in capsys.readouterr().out
    assert not _record_path().exists()


def test_finish_update_completes_the_recorded_install(rebuilt, tmp_path, monkeypatch, capsys):
    stale = _bundle(tmp_path / "Applications", b"stale")
    main_desktop.record_pending_desktop_install(
        app=stale, rebuilt_app=rebuilt, asar_hash=main_desktop._app_asar_hash(rebuilt))
    monkeypatch.setattr(main_desktop, "_running_macos_app_bundles", set)
    import hermes_cli.gui_uninstall as gui_uninstall
    monkeypatch.setattr(gui_uninstall, "packaged_gui_app_paths", list)

    main_desktop.cmd_desktop_finish_update(type("Args", (), {})())

    out = capsys.readouterr().out
    assert str(stale) in out and "Installed" in out
    assert _asar(stale) == b"rebuilt"
    assert not _record_path().exists()


def test_finish_update_refuses_a_running_bundle_and_keeps_the_record(rebuilt, tmp_path, monkeypatch, capsys):
    stale = _bundle(tmp_path / "Applications", b"stale")
    main_desktop.record_pending_desktop_install(
        app=stale, rebuilt_app=rebuilt, asar_hash=main_desktop._app_asar_hash(rebuilt))
    monkeypatch.setattr(main_desktop, "_running_macos_app_bundles", lambda: {stale.resolve()})
    import hermes_cli.gui_uninstall as gui_uninstall
    monkeypatch.setattr(gui_uninstall, "packaged_gui_app_paths", list)

    main_desktop.cmd_desktop_finish_update(type("Args", (), {})())

    out = capsys.readouterr().out
    assert "running" in out
    assert _asar(stale) == b"stale"
    assert _record_path().exists()


def test_finish_update_reports_missing_staged_bundle(rebuilt, tmp_path, capsys):
    stale = _bundle(tmp_path / "Applications", b"stale")
    main_desktop.record_pending_desktop_install(
        app=stale, rebuilt_app=tmp_path / "gone" / "release" / "Hermes.app", asar_hash="deadbeef")

    main_desktop.cmd_desktop_finish_update(type("Args", (), {})())

    out = capsys.readouterr().out
    assert "gone or changed" in out
    # The stale record can never be completed; the command clears it instead of failing forever.
    assert not _record_path().exists()


def test_finish_update_clears_when_installed_already_matches(rebuilt, tmp_path, monkeypatch, capsys):
    current = _bundle(tmp_path / "Applications", b"rebuilt")
    main_desktop.record_pending_desktop_install(
        app=current, rebuilt_app=rebuilt, asar_hash=main_desktop._app_asar_hash(rebuilt))
    monkeypatch.setattr(main_desktop, "_running_macos_app_bundles", set)
    import hermes_cli.gui_uninstall as gui_uninstall
    monkeypatch.setattr(gui_uninstall, "packaged_gui_app_paths", list)

    main_desktop.cmd_desktop_finish_update(type("Args", (), {})())

    assert "already matches" in capsys.readouterr().out
    assert not _record_path().exists()
