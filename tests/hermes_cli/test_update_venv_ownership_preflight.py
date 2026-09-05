"""Venv ownership preflight for ``hermes update`` (#83529).

A venv touched by ``sudo pip`` / ``sudo hermes`` contains root-owned files
(e.g. ``site-packages/hermes_agent-*.dist-info/INSTALLER``). A later normal
``hermes update`` then dies mid-mutation inside ``uv pip install -e .``
("Permission denied (os error 13)") with ``venv/bin/hermes`` already deleted,
bricking the CLI. The preflight refuses BEFORE the first venv mutation and
prints the exact chown recovery command.

The helper must be pure ``os.stat``/``os.scandir`` — no subprocess calls —
because update-path tests mock ``subprocess.run`` with sequenced side effects.
"""

import os

import pytest

from hermes_cli import update_cmd
import hermes_cli.update_cmd_deps as update_cmd_deps


def _make_fake_venv(tmp_path):
    """Minimal POSIX venv layout with a dist-info directory."""
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "hermes").write_text("#!stub\n")
    (venv / "bin" / "python").write_text("#!stub\n")
    sp = venv / "lib" / "python3.12" / "site-packages"
    dist_info = sp / "hermes_agent-1.0.0.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "INSTALLER").write_text("pip\n")
    (dist_info / "RECORD").write_text("\n")
    (sp / "some_pkg").mkdir()
    return venv


def test_all_owned_returns_empty(tmp_path):
    venv = _make_fake_venv(tmp_path)
    assert update_cmd._venv_foreign_owned_paths(venv) == []


def test_all_owned_preflight_proceeds(tmp_path, monkeypatch, capsys):
    """Gate is a no-op (no exit, no output) when everything is user-owned."""
    _make_fake_venv(tmp_path)
    update_cmd._refuse_update_if_venv_foreign_owned(tmp_path)
    assert capsys.readouterr().out == ""


def test_foreign_owned_dist_info_child_detected(tmp_path, monkeypatch):
    venv = _make_fake_venv(tmp_path)
    installer = str(
        venv / "lib" / "python3.12" / "site-packages"
        / "hermes_agent-1.0.0.dist-info" / "INSTALLER"
    )
    real_uid = update_cmd._path_uid

    def fake_uid(path):
        if str(path) == installer:
            return 0  # simulate root-owned sudo-pip residue
        return real_uid(path)

    monkeypatch.setattr(update_cmd, "_path_uid", fake_uid)
    monkeypatch.setattr(update_cmd_deps, "_path_uid", fake_uid)
    foreign = update_cmd._venv_foreign_owned_paths(venv)
    assert foreign == [(installer, 0)]


def test_foreign_owned_refuses_with_chown_hint(tmp_path, monkeypatch, capsys):
    venv = _make_fake_venv(tmp_path)
    hermes_bin = str(venv / "bin" / "hermes")
    real_uid = update_cmd._path_uid
    monkeypatch.setattr(
        update_cmd,
        "_path_uid",
        lambda p: 0 if str(p) == hermes_bin else real_uid(p),
    )
    monkeypatch.setattr(
        update_cmd_deps,
        "_path_uid",
        lambda p: 0 if str(p) == hermes_bin else real_uid(p),
    )
    with pytest.raises(SystemExit) as exc:
        update_cmd._refuse_update_if_venv_foreign_owned(tmp_path)
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert hermes_bin in out
    assert "owner uid 0" in out
    assert f"sudo chown -R $(id -un): {tmp_path}" in out
    assert "Nothing in the venv was modified." in out


def test_limit_caps_reported_paths(tmp_path, monkeypatch):
    venv = _make_fake_venv(tmp_path)
    bin_dir = venv / "bin"
    for i in range(10):
        (bin_dir / f"tool{i}").write_text("x")
    monkeypatch.setattr(
        update_cmd,
        "_path_uid",
        lambda p: 0 if str(p).startswith(str(bin_dir) + os.sep) else 12345,
    )
    monkeypatch.setattr(
        update_cmd_deps,
        "_path_uid",
        lambda p: 0 if str(p).startswith(str(bin_dir) + os.sep) else 12345,
    )
    monkeypatch.setattr(update_cmd.os, "geteuid", lambda: 12345, raising=False)
    foreign = update_cmd._venv_foreign_owned_paths(venv, limit=3)
    assert len(foreign) == 3


def test_no_geteuid_returns_empty(tmp_path, monkeypatch):
    """Windows (no os.geteuid) skips the preflight entirely."""
    venv = _make_fake_venv(tmp_path)

    class _NoGeteuidOS:
        def __getattr__(self, name):
            if name == "geteuid":
                raise AttributeError(name)
            return getattr(os, name)

    monkeypatch.setattr(update_cmd, "os", _NoGeteuidOS())
    assert update_cmd._venv_foreign_owned_paths(venv) == []


def test_running_as_root_returns_empty(tmp_path, monkeypatch):
    venv = _make_fake_venv(tmp_path)
    monkeypatch.setattr(update_cmd.os, "geteuid", lambda: 0, raising=False)
    # Even with foreign uids everywhere, root skips the gate.
    monkeypatch.setattr(update_cmd, "_path_uid", lambda p: 4242)
    monkeypatch.setattr(update_cmd_deps, "_path_uid", lambda p: 4242)
    assert update_cmd._venv_foreign_owned_paths(venv) == []


def test_never_raises_on_scandir_permission_error(tmp_path, monkeypatch):
    venv = _make_fake_venv(tmp_path)

    def boom(path):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(update_cmd.os, "scandir", boom)
    assert update_cmd._venv_foreign_owned_paths(venv) == []


def test_never_raises_on_missing_venv(tmp_path):
    assert update_cmd._venv_foreign_owned_paths(tmp_path / "no-venv") == []


def test_path_uid_none_on_oserror(tmp_path):
    assert update_cmd._path_uid(tmp_path / "does-not-exist") is None
