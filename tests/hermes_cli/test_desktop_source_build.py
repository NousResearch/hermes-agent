"""Desktop launch prepares once, then stages and publishes the local pack."""
from argparse import Namespace
import subprocess

import pytest

from hermes_cli import main, main_desktop
from tests.hermes_cli.test_source_build import source_checkout, source_products, _events  # noqa: F401


@pytest.fixture
def desktop_source(source_products, monkeypatch):
    root, acquired = source_products
    monkeypatch.setattr(main, "PROJECT_ROOT", root)
    monkeypatch.setattr(main_desktop, "_desktop_launch_env", lambda args: ({}, []))
    monkeypatch.setattr(main_desktop, "_register_linux_desktop_entry", lambda: None)
    return root, acquired


@pytest.mark.platforms("linux")
def test_desktop_build_only_prepares_once_and_keeps_fresh_launch_fast(desktop_source):
    root, acquired = desktop_source
    main_desktop.cmd_gui(Namespace(build_only=True))
    app = root / "apps/desktop/release/linux-unpacked/hermes"
    assert app.read_text() == "desktop"
    assert [event["step"] for event in _events(root)] == ["deps", "icons", "desktop"]
    assert acquired == ["npm"]
    assert (root / "node_modules/ui-tui").exists()
    assert (root / "node_modules/web").exists()
    assert (root / "node_modules/apps-desktop").exists()
    assert not (root / "node_modules/unrelated").exists()
    main_desktop.cmd_gui(Namespace(build_only=True))
    assert acquired == ["npm"]
    assert len(_events(root)) == 3


@pytest.mark.platforms("linux")
def test_failed_pack_exits_without_launching_or_replacing_the_app(desktop_source):
    root, acquired = desktop_source
    app = root / "apps/desktop/release/linux-unpacked/hermes"
    app.parent.mkdir(parents=True)
    app.write_text("previous app")
    (root / "fail-desktop").touch()
    with pytest.raises(SystemExit) as error:
        main_desktop.cmd_gui(Namespace(build_only=True))
    assert error.value.code != 0
    assert app.read_text() == "previous app"
    assert acquired == ["npm"]
    assert [event["step"] for event in _events(root)] == ["deps", "icons", "desktop"]
    assert not list((root / "apps/desktop").glob(".staging-*"))


@pytest.mark.platforms("linux")
def test_skip_build_source_checks_artifacts_without_provisioning(desktop_source):
    root, acquired = desktop_source
    with pytest.raises(SystemExit):
        main_desktop.cmd_gui(Namespace(source=True, skip_build=True, build_only=True))
    assert acquired == []
    dist = root / "apps/desktop/dist"
    dist.mkdir()
    (dist / "index.html").write_text("prepared renderer")
    electron = root / "node_modules/electron"
    electron.mkdir(parents=True)
    (electron / "package.json").write_text('{}')
    main_desktop.cmd_gui(Namespace(source=True, skip_build=True, build_only=True))
    assert acquired == []
