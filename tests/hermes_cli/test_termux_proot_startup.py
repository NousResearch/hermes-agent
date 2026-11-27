from types import SimpleNamespace

import pytest


@pytest.mark.platforms("linux")
@pytest.mark.parametrize(
    "release,version,host_present,expected",
    [
        ("6.1.0-PRoot", "#1 SMP", True, True),
        ("6.1.0", "#1 PRoot", True, True),
        ("6.1.0-PRoot", "#1 SMP", False, False),
        ("6.1.0", "#1 SMP", True, False),
    ],
)
def test_proot_startup_requires_termux_host(
    monkeypatch, release, version, host_present, expected
):
    from hermes_cli import _startup_fast as startup

    monkeypatch.delenv("PREFIX", raising=False)
    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    monkeypatch.setattr(
        startup.os, "uname", lambda: SimpleNamespace(release=release, version=version)
    )
    isdir = startup.os.path.isdir
    monkeypatch.setattr(
        startup.os.path,
        "isdir",
        lambda path: host_present if path == "/data/data/com.termux" else isdir(path),
    )

    assert startup.is_termux_startup_environment() is expected


@pytest.mark.platforms("linux")
def test_proot_startup_skips_unchanged_skills_but_syncs_new_revision(
    tmp_path, monkeypatch
):
    from hermes_cli import main
    from tools import skills_sync

    monkeypatch.delenv("PREFIX", raising=False)
    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    monkeypatch.delenv("HERMES_TERMUX_FORCE_SKILLS_SYNC", raising=False)
    monkeypatch.setattr(main, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(
        main.os, "uname", lambda: SimpleNamespace(release="6.1.0-PRoot", version="")
    )
    isdir = main.os.path.isdir
    monkeypatch.setattr(
        main.os.path,
        "isdir",
        lambda path: path == "/data/data/com.termux" or isdir(path),
    )
    revision = ["first"]
    monkeypatch.setattr(main, "_termux_bundled_skills_fingerprint", lambda: revision[0])
    syncs = []
    monkeypatch.setattr(skills_sync, "sync_skills", lambda **kwargs: syncs.append(kwargs))

    assert main._sync_bundled_skills_for_startup() is True
    assert main._sync_bundled_skills_for_startup() is False
    revision[0] = "second"
    assert main._sync_bundled_skills_for_startup() is True
    assert syncs == [{"quiet": True}, {"quiet": True}]
