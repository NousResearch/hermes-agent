"""The runtime's own interpreter/venv is not agent-deletable (#58748).

A Hermes session asked to clean up "older Pythons" removed the base
interpreter its own venv pointed at; the next boot died with ``uv trampoline
failed to spawn Python child process``. These tests pin both defense layers:

* the approval floor (``_floor_block``) must hard-block shell commands that
  delete the running interpreter/venv/base — even under yolo;
* the file-safety write classifier must deny writes/deletes to them.

The ``_protected_snapshot`` cache is keyed on (executable, prefix), so tests
swap ``sys`` attributes and let the cache key change with them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

import pytest

from agent import runtime_self_protection as rsp


@pytest.fixture
def fake_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A venv whose pyvenv.cfg points at a uv-managed base interpreter."""
    venv = tmp_path / "hermes-agent" / "venv"
    venv.mkdir(parents=True)
    exe_dir = venv / ("Scripts" if sys.platform == "win32" else "bin")
    exe_dir.mkdir()
    exe = exe_dir / ("python.exe" if sys.platform == "win32" else "python")
    exe.write_text("", encoding="utf-8")

    uv_base = tmp_path / "uv" / "python" / "cpython-3.11.9-windows-x86_64-none"
    uv_base.mkdir(parents=True)
    base_exe_dir = uv_base / ("Scripts" if sys.platform == "win32" else "bin")
    base_exe_dir.mkdir()
    (base_exe_dir / ("python.exe" if sys.platform == "win32" else "python")).write_text("", encoding="utf-8")

    home_dir = str(base_exe_dir)
    (venv / "pyvenv.cfg").write_text(
        f"home = {home_dir}\nversion = 3.11.9\ninclude-system-site-packages = false\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "executable", str(exe))
    monkeypatch.setattr(sys, "prefix", str(venv))
    rsp._protected_snapshot.cache_clear()
    yield {"exe": str(exe), "venv": str(venv), "uv_base": str(uv_base), "base_dir": home_dir}
    rsp._protected_snapshot.cache_clear()


def test_pyvenv_home_is_parsed(fake_runtime):
    assert rsp._pyvenv_home(fake_runtime["venv"]) == fake_runtime["base_dir"]


def test_rm_of_running_interpreter_is_detected(fake_runtime):
    assert rsp.command_deletes_runtime(f'rm "{fake_runtime["exe"]}"') is not None


def test_rm_rf_of_own_venv_is_detected(fake_runtime):
    assert rsp.command_deletes_runtime(f"rm -rf {fake_runtime['venv']}") is not None


def test_rm_of_base_interpreter_is_detected(fake_runtime):
    base_exe = os.path.join(fake_runtime["base_dir"], os.listdir(fake_runtime["base_dir"])[0])
    assert rsp.command_deletes_runtime(f"sudo rm -f '{base_exe}'") is not None


def test_rm_of_whole_uv_install_dir_is_detected(fake_runtime):
    assert rsp.command_deletes_runtime(f"rm -rf {fake_runtime['uv_base']}") is not None


def test_uv_python_uninstall_of_running_version_is_detected(fake_runtime):
    assert rsp.command_deletes_runtime("uv python uninstall 3.11") is not None


def test_uv_python_uninstall_all_is_detected(fake_runtime):
    assert rsp.command_deletes_runtime("uv python uninstall --all") is not None


def test_uv_python_uninstall_of_other_version_is_allowed(fake_runtime):
    assert rsp.command_deletes_runtime("uv python uninstall 3.9") is None


def test_find_delete_over_own_venv_is_detected(fake_runtime):
    # Both are deletes under the protected root: an unfiltered -delete removes
    # the interpreter itself, and a filtered one still deletes protected files.
    assert rsp.command_deletes_runtime(f"find '{fake_runtime['venv']}' -delete") is not None
    assert rsp.command_deletes_runtime(f"find '{fake_runtime['venv']}' -name __pycache__ -delete") is not None
    assert rsp.command_deletes_runtime("find /tmp -name __pycache__ -delete") is None


def test_rm_of_unrelated_venv_is_allowed(fake_runtime, tmp_path):
    other = tmp_path / "project" / ".venv"
    other.mkdir(parents=True)
    assert rsp.command_deletes_runtime(f"rm -rf {other}") is None


def test_windows_del_and_powershell_spellings(fake_runtime, monkeypatch):
    with mock.patch.object(os, "name", "nt"):
        exe = fake_runtime["exe"]
        assert rsp.command_deletes_runtime(f'del "{exe}"') is not None
        assert rsp.command_deletes_runtime(f'Remove-Item -Recurse -Force "{exe}"') is not None
        assert rsp.command_deletes_runtime(f"rd /s /q {fake_runtime['venv']}") is not None


def test_mkdir_over_protected_path_not_flagged_as_delete(fake_runtime):
    # mkdir touches the venv dir but deletes nothing — command layer says fine
    # (the file-safety layer still denies the write).
    assert rsp.command_deletes_runtime(f"mkdir -p {fake_runtime['venv']}") is None


def test_is_protected_path_covers_venv_children_and_base(fake_runtime):
    site = os.path.join(fake_runtime["venv"], "lib", "site-packages", "x.py")
    assert rsp.is_protected_path(site) is not None
    assert rsp.is_protected_path(fake_runtime["base_dir"]) is not None
    assert rsp.is_protected_path("/tmp/scratch.txt") is None


def test_file_safety_denies_write_to_running_interpreter(fake_runtime):
    from agent.file_safety import _classify_write_denial

    assert _classify_write_denial(fake_runtime["exe"]) == "credential"
    assert _classify_write_denial(os.path.join(fake_runtime["venv"], "pyvenv.cfg")) == "credential"


def test_file_safety_allows_unrelated_paths(fake_runtime, tmp_path):
    from agent.file_safety import _classify_write_denial

    scratch = tmp_path / "scratch.txt"
    assert _classify_write_denial(str(scratch)) is None


def test_approval_floor_blocks_runtime_delete_even_under_yolo(fake_runtime, monkeypatch):
    from tools import approval

    monkeypatch.setattr(approval, "_yolo_active", lambda: True)
    result = approval._floor_block(f'rm -rf "{fake_runtime["venv"]}"')
    assert result is not None
    assert result.get("approved") is False


def test_check_all_guards_blocks_runtime_delete(fake_runtime, monkeypatch):
    from tools import approval

    # Outside CLI/gateway/ask contexts with yolo on, the floor is the only
    # thing standing; it must still block.
    monkeypatch.setattr(approval, "_yolo_active", lambda: True)
    result = approval.check_all_command_guards(f'rm "{fake_runtime["exe"]}"', "local")
    assert result.get("approved") is False


def test_normal_command_unaffected(fake_runtime, monkeypatch):
    from tools import approval

    monkeypatch.setattr(approval, "_yolo_active", lambda: True)
    assert approval.check_all_command_guards("ls /tmp", "local").get("approved") is True


# --- a relative delete target follows a preceding `cd` (#143691) -------------
#
# A relative target used to be resolved against the process cwd only, so the
# shell's own `cd` was ignored: `cd <project>; rm -rf .` was blocked whenever the
# process cwd happened to be an ancestor of the install (the gateway's cwd is
# usually the user's home), while `cd ~/.hermes/hermes-agent && rm -rf venv` —
# the delete this floor exists for — passed.


def test_relative_delete_after_cd_resolves_against_that_directory(fake_runtime, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the process cwd is an ancestor of the install
    project = tmp_path / "projects" / "x"
    project.mkdir(parents=True)
    assert rsp.command_deletes_runtime(f"cd {project}; find . -name __pycache__ -prune -exec rm -r {{}} +") is None
    assert rsp.command_deletes_runtime(f"cd {project} && rm -rf .") is None
    # Control: the same delete spelled absolutely was never a hit.
    assert rsp.command_deletes_runtime(f"find {project} -name __pycache__ -prune -exec rm -r {{}} +") is None


def test_cd_into_the_install_blocks_a_relative_venv_delete(fake_runtime, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    install = tmp_path / "hermes-agent"
    assert rsp.command_deletes_runtime(f"cd {install} && rm -rf venv") is not None
    assert rsp.command_deletes_runtime(f"cd {install}; rm -rf ./venv/bin") is not None
    assert rsp.command_deletes_runtime(f"cd {install} && find . -name '*.pyc' -delete") is not None
    # No `cd` at all: a relative target keeps its process-cwd reading.
    assert rsp.command_deletes_runtime("rm -rf hermes-agent") is not None


def test_cd_tracks_tilde_and_chained_segments(fake_runtime, tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "projects" / "x").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(tmp_path)
    assert rsp.command_deletes_runtime("cd ~/projects/x && rm -rf .") is None
    assert rsp.command_deletes_runtime(f"cd {tmp_path}; cd hermes-agent && rm -rf venv") is not None


def test_unresolvable_cd_falls_back_to_the_process_cwd(fake_runtime, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # `cd "$TARGET"` and `cd -` are not statically knowable: keep the process-cwd
    # reading rather than guessing a directory the shell may not enter.
    assert rsp.command_deletes_runtime('cd "$TARGET" && rm -rf .') is not None
    assert rsp.command_deletes_runtime("cd - && rm -rf .") is not None


def test_a_cd_or_pushd_that_does_not_go_home_keeps_the_old_directory(fake_runtime, tmp_path, monkeypatch):
    # A quoted or escaped `~` is a literal name, `~+` is $PWD, `cd ""` stays put, a bare `pushd`
    # or `pushd +N` / `-N` rotates the directory stack, `pushd -n` never moves, a glob is the
    # shell's to expand, a `;` inside quotes is no separator, and `cd -- -L` names a directory.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path / "hermes-agent")
    (tmp_path / "hermes-agent" / "child'").mkdir()
    for cd in ("cd '~'", 'cd "~"', "cd \\~", "pushd", f"cd '{tmp_path};missing'", "cd child';missing'", "cd -- -L"):
        assert rsp.command_deletes_runtime(f"{cd}; rm -rf venv") is not None, cd
    for cd in ("cd ~+", 'cd ""', "pushd +0", "pushd -0", "cd ../hermes-*", f"pushd -n {tmp_path}"):
        for joiner in (";", "&&"):
            assert rsp.command_deletes_runtime(f"{cd} {joiner} rm -rf venv") is not None, f"{cd} {joiner}"


def test_cd_dash_p_keeps_the_old_directory(fake_runtime, tmp_path, monkeypatch):
    # `cd -P` enters the symlink's target, so a later `cd ..` returns to the install.
    install = tmp_path / "hermes-agent"
    (install / "child").mkdir()
    (tmp_path / "link").symlink_to(install / "child")
    monkeypatch.chdir(install)
    for cd in ("cd -P", "cd -LP"):
        assert rsp.command_deletes_runtime(f"{cd} {tmp_path / 'link'}; cd ..; rm -rf ../hermes-agent/venv") is not None, cd


def test_a_cd_into_a_missing_dir_keeps_the_directory_the_shell_reached(fake_runtime, tmp_path, monkeypatch):
    # Each missing `cd` fails and leaves the shell in the install; enough of them used to
    # push the install out of the capped candidate set, leaving only the process cwd.
    install = tmp_path / "hermes-agent"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    for count in (3, 9, 40):
        misses = "; ".join(f"cd missing{i}" for i in range(count))
        assert rsp.command_deletes_runtime(f"cd {install} && {misses}; rm -rf venv") is not None, count


def test_a_reassigned_home_makes_a_bare_cd_unknowable(fake_runtime, tmp_path, monkeypatch):
    install = tmp_path / "hermes-agent"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setenv("HOME", str(elsewhere))
    monkeypatch.chdir(install)
    for cd in ("cd", "cd ~"):
        assert rsp.command_deletes_runtime(f"cd {elsewhere}; HOME={install}; {cd}; rm -rf venv") is not None, cd
    assert rsp.command_deletes_runtime(f"cd {install}; cd; rm -rf venv") is None


def test_cd_does_not_hide_an_absolute_or_dollar_home_target(fake_runtime, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    assert rsp.command_deletes_runtime(f"cd {elsewhere} && rm -rf $HOME/hermes-agent/venv") is not None
    assert rsp.command_deletes_runtime(f"cd {elsewhere} && rm -rf ${{HOME}}/hermes-agent/venv") is not None


def test_a_cd_that_may_not_move_the_shell_keeps_the_old_directory(fake_runtime, tmp_path, monkeypatch):
    # `||` runs the delete only when the cd FAILED, a piped cd runs in a subshell,
    # and after `;` a cd into a missing directory leaves the shell where it was.
    install = tmp_path / "hermes-agent"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(install)
    assert rsp.command_deletes_runtime(f"cd {elsewhere} || rm -rf venv") is not None
    assert rsp.command_deletes_runtime(f"cd {elsewhere} | rm -rf venv") is not None
    assert rsp.command_deletes_runtime(f"cd {tmp_path / 'missing'}; rm -rf venv") is not None
    # A cd that certainly moved the shell elsewhere does clear the relative target.
    assert rsp.command_deletes_runtime(f"cd {elsewhere}; rm -rf venv") is None
    assert rsp.command_deletes_runtime(f"cd {tmp_path / 'missing'} && rm -rf venv") is None


def test_a_failed_and_chain_does_not_carry_its_cd_past_the_list(fake_runtime, tmp_path, monkeypatch):
    # A failing `cd` skips the rest of its `&&` chain; a later `;`/`||` segment
    # then runs in the directory the shell was in before the chain.
    install = tmp_path / "hermes-agent"
    monkeypatch.chdir(install)
    missing = tmp_path / "missing"
    assert rsp.command_deletes_runtime(f"cd {missing} && echo ok; rm -rf venv") is not None
    assert rsp.command_deletes_runtime(f"cd {missing} && true || rm -rf venv") is not None
    assert rsp.command_deletes_runtime(f"cd {missing} && true\nrm -rf venv") is not None
    # After a `||` the cd itself may be skipped while the delete still runs.
    (tmp_path / "elsewhere").mkdir()
    assert rsp.command_deletes_runtime(f"true || cd {tmp_path / 'elsewhere'} && rm -rf venv") is not None


def test_only_a_bare_cd_builtin_moves_the_shell(fake_runtime, tmp_path, monkeypatch):
    install = tmp_path / "hermes-agent"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(install)
    assert rsp.command_deletes_runtime(f"true | cd {elsewhere} && rm -rf venv") is not None
    for prefix in ("sudo cd", "env cd", "/usr/bin/cd"):
        assert rsp.command_deletes_runtime(f"{prefix} {elsewhere}; rm -rf venv") is not None, prefix
    if os.name != "nt":
        assert rsp.command_deletes_runtime(f"CD {elsewhere}; rm -rf venv") is not None
    assert rsp.command_deletes_runtime(f"cd {elsewhere} {tmp_path}; rm -rf venv") is not None


def test_a_pipeline_continued_on_the_next_line_keeps_its_cd_in_a_subshell(fake_runtime, tmp_path, monkeypatch):
    install = tmp_path / "hermes-agent"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(install)
    for gap in ("\n", "\n\n", " # note\n", " \\\n"):
        assert rsp.command_deletes_runtime(f"true |{gap}cd {elsewhere} && rm -rf venv") is not None, repr(gap)
    # A comment hides its own separators: the cd below never runs.
    assert rsp.command_deletes_runtime(f"# note; cd {elsewhere}\nrm -rf venv") is not None


def test_a_cd_the_walk_does_not_model_brings_back_the_process_cwd(fake_runtime, tmp_path, monkeypatch):
    # After a certain `cd`, a later directory change the walk cannot follow still moves the shell.
    install = tmp_path / "hermes-agent"
    (install / "child").mkdir()
    monkeypatch.chdir(install)
    for move in ("if true; then cd ..; fi", "builtin cd ..", "command cd ..", "{ cd ..; }", "eval 'cd ..'", "up() { cd ..; }; up",
                 "if true; then pushd ..; fi"):
        assert rsp.command_deletes_runtime(f"cd child; {move}; rm -rf venv") is not None, move


def test_a_cd_target_removed_earlier_in_the_command_does_not_count(fake_runtime, tmp_path, monkeypatch):
    install = tmp_path / "hermes-agent"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(install)
    assert rsp.command_deletes_runtime(f"rm -rf {elsewhere}; cd {elsewhere}; rm -rf venv") is not None


def test_cdpath_makes_a_relative_cd_unknowable(fake_runtime, tmp_path, monkeypatch):
    # With CDPATH set, `cd hermes-agent` from inside the install re-enters it
    # instead of failing on the missing `./hermes-agent`.
    monkeypatch.chdir(tmp_path / "hermes-agent")
    assert rsp.command_deletes_runtime(f"export CDPATH={tmp_path}; cd hermes-agent && rm -rf venv") is not None
    monkeypatch.setenv("CDPATH", str(tmp_path))
    assert rsp.command_deletes_runtime("cd hermes-agent && rm -rf venv") is not None


def test_many_cds_into_missing_dirs_stay_linear(fake_runtime, tmp_path, monkeypatch):
    import time

    monkeypatch.chdir(tmp_path / "hermes-agent")
    cds = "; ".join(f"cd d{i}" for i in range(40))
    began = time.monotonic()
    assert rsp.command_deletes_runtime(f"{cds}; rm -rf venv") is not None
    assert rsp.command_deletes_runtime(cds.replace("; ", "\n") + "\nrm -rf zz") is None
    assert time.monotonic() - began < 2
