"""Git object ownership preflight for ``hermes update`` (#102172).

A checkout previously touched through ``sudo`` can contain root-owned fan-out
folders under ``.git/objects``. A later update then fails while autostash or
fetch tries to add an object. The preflight refuses before the backup and Git
mutation phases and prints the exact ownership repair command.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import update_cmd


def _make_object_db(root: Path) -> Path:
    objects = root / ".git" / "objects"
    (objects / "1c").mkdir(parents=True)
    (objects / "info").mkdir()
    (objects / "pack").mkdir()
    return objects


def test_foreign_owned_object_fanout_is_detected(tmp_path, monkeypatch):
    objects = _make_object_db(tmp_path)
    fanout = str(objects / "1c")

    monkeypatch.setattr(update_cmd.os, "geteuid", lambda: 12345, raising=False)
    monkeypatch.setattr(
        update_cmd,
        "_path_uid",
        lambda path: 0 if str(path) == fanout else 12345,
    )

    assert update_cmd._git_objects_foreign_owned_paths(tmp_path) == [(fanout, 0)]


def test_repository_ownership_gate_refuses_with_repair_hint(
    tmp_path, monkeypatch, capsys
):
    object_dir = str(tmp_path / ".git" / "objects" / "1c")
    monkeypatch.setattr(
        update_cmd,
        "_git_objects_foreign_owned_paths",
        lambda _root: [(object_dir, 0)],
    )

    with pytest.raises(SystemExit) as exc:
        update_cmd._refuse_update_if_git_objects_foreign_owned(tmp_path)

    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert object_dir in output
    assert "owner uid 0" in output
    assert f"sudo chown -R $(id -un): {tmp_path}" in output
    assert "before creating a backup or stashing local changes" in output


def test_repository_ownership_gate_is_noop_for_healthy_checkout(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        update_cmd,
        "_git_objects_foreign_owned_paths",
        lambda _root: [],
    )

    update_cmd._refuse_update_if_git_objects_foreign_owned(tmp_path)

    assert capsys.readouterr().out == ""


def test_update_runs_repository_ownership_gate_before_backup(tmp_path, monkeypatch):
    from hermes_cli import main as hermes_main

    calls = []
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(hermes_main, "_capture_active_lazy_features", lambda: set())
    monkeypatch.setattr(hermes_main, "_capture_active_tool_dependencies", lambda: set())
    monkeypatch.setattr(hermes_main, "_is_windows", lambda: False)
    monkeypatch.setattr(
        hermes_main,
        "_run_pre_update_backup",
        lambda _args: calls.append("backup"),
    )
    monkeypatch.setattr(update_cmd, "_read_project_version", lambda: "test")
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setattr("hermes_cli.update_receipt.begin_update_receipt", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.update_inventory.collect_runtime_inventory",
        lambda: SimpleNamespace(runtimes=[]),
    )
    monkeypatch.setattr(
        "hermes_cli.update_inventory.record_plan_in_receipt", lambda _plan: None
    )

    def refuse(_root):
        calls.append("preflight")
        raise SystemExit(37)

    monkeypatch.setattr(
        update_cmd, "_refuse_update_if_git_objects_foreign_owned", refuse
    )

    with pytest.raises(SystemExit) as exc:
        update_cmd._cmd_update_impl(SimpleNamespace(), gateway_mode=False)

    assert exc.value.code == 37
    assert calls == ["preflight"]


def test_object_scan_is_bounded_to_direct_entries(tmp_path, monkeypatch):
    objects = _make_object_db(tmp_path)
    loose_object = objects / "1c" / "deadbeef"
    loose_object.write_text("object")

    monkeypatch.setattr(update_cmd.os, "geteuid", lambda: 12345, raising=False)
    monkeypatch.setattr(
        update_cmd,
        "_path_uid",
        lambda path: 0 if Path(path) == loose_object else 12345,
    )

    assert update_cmd._git_objects_foreign_owned_paths(tmp_path) == []


def test_object_scan_skips_root_and_platforms_without_geteuid(
    tmp_path, monkeypatch
):
    real_os = update_cmd.os
    _make_object_db(tmp_path)
    monkeypatch.setattr(update_cmd, "_path_uid", lambda _path: 12345)
    monkeypatch.setattr(update_cmd.os, "geteuid", lambda: 0, raising=False)
    assert update_cmd._git_objects_foreign_owned_paths(tmp_path) == []

    class _NoGeteuidOS:
        def __getattr__(self, name):
            if name == "geteuid":
                raise AttributeError(name)
            return getattr(real_os, name)

    monkeypatch.setattr(update_cmd, "os", _NoGeteuidOS())
    assert update_cmd._git_objects_foreign_owned_paths(tmp_path) == []
