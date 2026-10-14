"""scripts/check_detached_spawns.py: detached spawns outside spawn_owned are flagged unless marked."""

import importlib.util
import textwrap
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_detached_spawns.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_detached_spawns", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "src",
    [
        "subprocess.Popen(argv, start_new_session=True)",
        "subprocess.Popen(argv, start_new_session=flag)",
        'kwargs = {"start_new_session": True}',
        "subprocess.Popen(argv, preexec_fn=os.setsid)",
        "subprocess.Popen(argv, preexec_fn=lambda: os.setpgid(0, 0))",
        "os.setpgrp()",
        "subprocess.Popen(argv, **windows_detach_popen_kwargs())",
        "subprocess.Popen(argv, **_subprocess_compat.windows_detach_popen_kwargs())",
    ],
)
def test_detached_shapes_are_flagged(src):
    assert _load().detach_lines(src) == [1]


@pytest.mark.parametrize(
    "src",
    [
        "subprocess.Popen(argv)",
        "subprocess.Popen(argv, start_new_session=False)",
        'kwargs = {"start_new_session": False}',
        "spawn_owned(argv, purpose='x')",
        "# start_new_session=True in a comment",
        'doc = "os.setsid in a string"',
    ],
)
def test_non_detached_shapes_are_not_flagged(src):
    assert _load().detach_lines(src) == []


def test_marker_on_the_line_or_above_opts_out_and_nothing_else_does(tmp_path):
    mod = _load()
    pkg = tmp_path / "tools"
    pkg.mkdir()
    f = pkg / "x.py"
    f.write_text(textwrap.dedent("""\
        import os, subprocess
        subprocess.Popen(a, start_new_session=True)  # detached-spawn: ok — awaited, group-killed
        # detached-spawn: ok — outlives its owner on purpose
        subprocess.Popen(b, start_new_session=True)
        # an unrelated comment
        subprocess.Popen(c, start_new_session=True)
    """), encoding="utf-8")
    hits = mod.scan([f], root=tmp_path)
    assert [n for n, _ in hits["tools/x.py"]] == [6]


def test_tests_and_the_helper_are_out_of_scope(tmp_path):
    mod = _load()
    for rel in ("tests/tools/test_x.py", "tools/test_y.py", "hermes_cli/process_identity.py"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("subprocess.Popen(a, start_new_session=True)\n", encoding="utf-8")
    assert mod.scan(list(tmp_path.rglob("*.py")), root=tmp_path) == {}


def test_a_new_unmarked_spawn_fails_main_and_a_marked_one_passes(tmp_path, monkeypatch, capsys):
    mod = _load()
    f = tmp_path / "tools" / "new.py"
    f.parent.mkdir()
    f.write_text("subprocess.Popen(a, start_new_session=True)\n", encoding="utf-8")
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    assert mod.main([str(f)]) == 1
    assert "tools/new.py:1" in capsys.readouterr().out
    f.write_text("subprocess.Popen(a, start_new_session=True)  # detached-spawn: ok — test\n", encoding="utf-8")
    assert mod.main([str(f)]) == 0


def test_the_repository_has_no_unmarked_detached_spawn():
    mod = _load()
    assert mod.scan() == {}, "a detached spawn outside spawn_owned lacks a `detached-spawn: ok — <why>` marker"
    assert mod._BASELINE == {}
