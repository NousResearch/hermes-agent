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


def test_record_failure_is_contained_and_reports_honestly(rebuilt, tmp_path, monkeypatch, caplog):
    """A read-only/full HERMES_HOME must not abort the install loop, must clean its
    `.tmp`, and must NOT tell the user a recovery command exists when no record does."""
    import pytest
    stale = _bundle(tmp_path / "Applications", b"stale")
    running = _bundle(tmp_path / "Volumes" / "Applications", b"older")

    def failing_write(self, data, encoding=None):
        raise PermissionError(13, "Read-only file system")

    monkeypatch.setattr(main_desktop.Path, "write_text", failing_write)
    with caplog.at_level("WARNING", logger="hermes_cli.main_desktop"):
        ok = main_desktop.record_pending_desktop_install(
            app=running, rebuilt_app=rebuilt, asar_hash="deadbeef")
    assert ok is False
    assert "could not be written" in caplog.text
    # No dishonest "recorded it — run ..." advice when nothing was recorded.
    assert not any("recorded it" in r.getMessage() for r in caplog.records)
    assert not _record_path().exists()
    # The install loop itself is not aborted by the record failure: the sibling
    # bundle still refreshes (record_pending_... is called inside the candidate
    # loop; with write failing it must propagate nothing).
    monkeypatch.setattr(main_desktop.Path, "write_text", failing_write)
    installed, problems = main_desktop._install_rebuilt_macos_bundles(
        rebuilt, [stale, running], running={running.resolve()})
    assert installed == [stale], "the next installable bundle must still refresh"
    assert len(problems) == 1 and str(running) in problems[0]


def test_finish_update_installs_after_recorded_app_was_removed(rebuilt, tmp_path, monkeypatch, capsys):
    """A deleted installed bundle must not loop 'nothing to do' forever: the staged
    build installs at the standard locations and the record clears."""
    gone_root = tmp_path / "OldApplications"
    stale = _bundle(gone_root, b"stale")
    record_app = stale
    main_desktop.record_pending_desktop_install(
        app=record_app, rebuilt_app=rebuilt, asar_hash=main_desktop._app_asar_hash(rebuilt))
    # The recorded location disappears (user uninstalled from there)...
    shutil.rmtree(gone_root)
    # ...but a standard install location exists and is stale.
    fresh = _bundle(tmp_path / "Applications", b"older")
    monkeypatch.setattr(main_desktop, "_running_macos_app_bundles", set)
    import hermes_cli.gui_uninstall as gui_uninstall
    monkeypatch.setattr(gui_uninstall, "packaged_gui_app_paths", lambda: [fresh])

    main_desktop.cmd_desktop_finish_update(type("Args", (), {})())

    out = capsys.readouterr().out
    assert "Installed" in out and str(fresh) in out
    assert _asar(fresh) == b"rebuilt"
    assert not _record_path().exists()


def test_finish_update_removed_app_with_nowhere_to_install_clears(rebuilt, tmp_path, monkeypatch, capsys):
    """Gone recorded app AND no install location: say so once and clear, not an
    endless 'nothing to do' loop."""
    gone_root = tmp_path / "OldApplications"
    stale = _bundle(gone_root, b"stale")
    main_desktop.record_pending_desktop_install(
        app=stale, rebuilt_app=rebuilt, asar_hash=main_desktop._app_asar_hash(rebuilt))
    shutil.rmtree(gone_root)
    monkeypatch.setattr(main_desktop, "_running_macos_app_bundles", set)
    import hermes_cli.gui_uninstall as gui_uninstall
    monkeypatch.setattr(gui_uninstall, "packaged_gui_app_paths", list)

    main_desktop.cmd_desktop_finish_update(type("Args", (), {})())

    out = capsys.readouterr().out
    assert "no longer installed" in out
    assert not _record_path().exists()


def test_products_stage_warns_when_an_install_stays_pending(rebuilt, tmp_path, monkeypatch, capsys):
    """Requirement 4 (#123737): the products stage the bootstrap/repair/installer
    flows share must not end on plain success while the rebuilt app sits staged
    and uninstalled — the skip is surfaced with the completion command."""
    import hermes_cli.source_build as source_build
    import hermes_cli.update_receipt as update_receipt

    running = _bundle(tmp_path / "Applications", b"older")
    monkeypatch.setattr(main_desktop, "_stage_macos_bundle_copy", _stage_copy)
    main_desktop.record_pending_desktop_install(
        app=running, rebuilt_app=rebuilt, asar_hash=main_desktop._app_asar_hash(rebuilt))

    recorded = {}
    monkeypatch.setattr(update_receipt, "record_fact",
                        lambda key, value: recorded.update({key: value}))

    source_build._warn_pending_desktop_install()

    err = capsys.readouterr().err
    assert str(running) in err and "finish-update" in err
    assert recorded == {"pending_desktop_install": str(running)}


def test_products_stage_says_nothing_without_a_pending_install(capsys):
    import hermes_cli.source_build as source_build

    source_build._warn_pending_desktop_install()

    assert capsys.readouterr().err == ""
