"""The run-elsewhere marker module's own contracts: which marker a home reads, what an unreadable
marker folder means, who may change a marker, and the names a plugin may import.

The host-side wiring (``profiles_to_serve``, the gateway, cron, Kanban, the dashboard) is covered in
``test_profiles_run_elsewhere.py``.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli import profiles_run_elsewhere as pre


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A Hermes root with the default profile and two named profiles, ``ops`` and ``dev``."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    for name in ("ops", "dev"):
        profile = home / "profiles" / name
        profile.mkdir(parents=True)
        (profile / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _alias(link: Path, target: Path) -> Path:
    """*link* as another path to the folder *target*: a symlink, or on Windows a junction when
    this account may not create symlinks. Skips where the OS offers neither."""
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        if os.name != "nt":
            pytest.skip(f"cannot create a directory symlink here: {exc}")
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
    return link


# ── which marker a home reads ──────────────────────────────────────────────────────────────────


def test_an_alias_of_a_named_profile_home_reads_that_profiles_marker(root, tmp_path):
    """``HERMES_HOME`` may be a symlink or junction to ``<root>/profiles/<id>``: the path names no
    ``profiles`` folder, the home is still that profile's."""
    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin")
    ops_link = _alias(tmp_path / "ops-link", root / "profiles" / "ops")
    dev_link = _alias(tmp_path / "dev-link", root / "profiles" / "dev")

    entry = pre.run_elsewhere_for_home(ops_link)
    assert entry is not None and (entry.profile, entry.plugin) == ("ops", "sandbox-plugin")
    with pytest.raises(pre.ProfileRunsElsewhere, match="'ops'"):
        pre.require_home_runs_here(ops_link)
    assert pre.run_elsewhere_for_home(dev_link) is None


def test_an_alias_of_the_root_reads_the_default_profiles_marker(root, tmp_path):
    root_link = _alias(tmp_path / "root-link", root)
    assert pre.run_elsewhere_for_home(root_link) is None
    pre.mark_profile_run_elsewhere("default", "sandbox-plugin")
    assert pre.run_elsewhere_for_home(root_link).profile == "default"
    # A named profile reached through the aliased root is still that named profile.
    assert pre.run_elsewhere_for_home(root_link / "profiles" / "ops") is None
    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin")
    assert pre.run_elsewhere_for_home(root_link / "profiles" / "ops").profile == "ops"


def test_a_home_spelled_with_dot_segments_or_relative_reads_its_profiles_marker(root, monkeypatch):
    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin")
    dotted = root / "profiles" / "dev" / ".." / "ops"
    assert pre.run_elsewhere_for_home(dotted).profile == "ops"
    monkeypatch.chdir(root / "profiles")
    assert pre.run_elsewhere_for_home("ops").profile == "ops"
    assert pre.run_elsewhere_for_home("dev") is None


def test_a_profile_home_spelled_in_another_letter_case_reads_its_profiles_marker(root):
    """Profile ids are stored lowercase, and a case-insensitive volume opens ``profiles/OPS`` as
    ``profiles/ops``."""
    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin")
    assert pre.run_elsewhere_for_home(root / "profiles" / "OPS").profile == "ops"
    assert pre.run_elsewhere_for_home(root / "profiles" / "Dev") is None
    pre.mark_profile_run_elsewhere("qa", "sandbox-plugin")  # no folder on disk: nothing to resolve to
    assert pre.run_elsewhere_for_home(root / "profiles" / "QA").profile == "qa"


@pytest.mark.platforms("windows")  # only Windows resolves a path to the letter case on disk
def test_a_profiles_folder_spelled_in_another_letter_case_reads_the_marker_on_windows(root):
    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin")
    assert pre.run_elsewhere_for_home(root / "PROFILES" / "OPS").profile == "ops"
    assert pre.run_elsewhere_for_home(root / "Profiles" / "dev") is None


def test_a_default_home_whose_real_path_looks_like_a_profile_keeps_its_default_marker(tmp_path, monkeypatch):
    """A root that is an alias of ``<somewhere>/profiles/<id>`` is still read as the root it is
    addressed as: either reading being marked refuses, so neither can hide the other."""
    real = tmp_path / "srv" / "profiles" / "main"
    real.mkdir(parents=True)
    home = _alias(tmp_path / "hermes-home", real)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert pre.run_elsewhere_for_home(home) is None
    pre.mark_profile_run_elsewhere("default", "sandbox-plugin", root=home)
    assert pre.run_elsewhere_for_home(home).profile == "default"


# ── what an unreadable marker folder means ─────────────────────────────────────────────────────


def test_a_file_where_the_marker_folder_belongs_runs_no_profile(root):
    """Only a missing folder means "nothing is marked". A regular file in its place is a folder that
    cannot be read: the host cannot tell which profiles are marked, so it runs none."""
    from hermes_cli.profiles import profiles_to_serve

    assert pre.run_elsewhere("ops") is None  # missing folder: nothing is marked
    (root / "profiles_run_elsewhere").write_text("not a folder\n", encoding="utf-8")

    entry = pre.run_elsewhere("ops")
    assert entry is not None and entry.folder_error and "cannot tell" in entry.message()
    assert pre.run_elsewhere_for_home(root).folder_error
    assert profiles_to_serve(True) == []
    with pytest.raises(pre.ProfileRunsElsewhere):
        pre.require_home_runs_here(root / "profiles" / "dev")


def test_reading_the_marker_folder_gives_one_kind_of_answer(root):
    """The marked ids and the error that made the folder unreadable are separate fields of one
    result, so a failed read is never the same value as "nothing marked"."""
    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin")
    listed = pre._read_folder(root / "profiles_run_elsewhere")
    missing = pre._read_folder(root / "no-such-folder")
    (root / "a-file").write_text("", encoding="utf-8")
    failed = pre._read_folder(root / "a-file")

    assert type(listed) is type(missing) is type(failed)
    assert (listed.names, listed.error) == (frozenset({"ops"}), None)
    assert (missing.names, missing.error) == (frozenset(), None)
    assert failed.names == frozenset() and isinstance(failed.error, OSError)


# ── who may change a marker ────────────────────────────────────────────────────────────────────


def _between_look_and_change(monkeypatch, profile: str, action) -> None:
    """Run *action* once, right after the module's next look at *profile*'s marker (the listing, stat
    or read a writer does before it changes anything): what another writer does in that gap.

    Only the module's own call counts. The suite's home guard wraps ``os.lstat`` and resolves the path
    first, and POSIX ``realpath`` lstats every component through ``os``: a look made inside a look is
    part of it, so *action* waits for the outermost one to finish."""
    armed = [True]
    looking = [0]
    real_listdir, real_lstat, real_read = os.listdir, os.lstat, pre._read_entry

    def look(real, is_the_marker, *args, **kwargs):
        looking[0] += 1
        try:
            return real(*args, **kwargs)
        finally:
            looking[0] -= 1
            if armed and not looking[0] and is_the_marker:
                armed.clear()
                action()

    def listdir(path):
        return look(real_listdir, Path(path).name == pre.RUN_ELSEWHERE_DIRNAME, path)

    def lstat(path, *args, **kwargs):
        return look(real_lstat, Path(os.fspath(path)).name == f"{profile}.json", path, *args, **kwargs)

    def read_entry(folder, name):
        return look(real_read, name == profile, folder, name)

    monkeypatch.setattr(pre.os, "listdir", listdir)
    monkeypatch.setattr(pre.os, "lstat", lstat)
    monkeypatch.setattr(pre, "_read_entry", read_entry)


def _marker_files(root: Path) -> list[str]:
    return sorted(os.listdir(root / "profiles_run_elsewhere"))


def test_two_plugins_that_both_saw_no_marker_cannot_both_claim_the_profile(root, monkeypatch):
    _between_look_and_change(monkeypatch, "ops", lambda: pre.mark_profile_run_elsewhere("ops", "other"))

    with pytest.raises(ValueError, match="'other'"):
        pre.mark_profile_run_elsewhere("ops", "sandbox-plugin")

    assert pre.run_elsewhere("ops").plugin == "other"
    assert _marker_files(root) == ["ops.json"]  # the loser's staged file is gone


def test_a_clear_that_found_no_marker_never_removes_one_published_after_it_looked(root, monkeypatch):
    (root / "profiles_run_elsewhere").mkdir()
    _between_look_and_change(monkeypatch, "ops", lambda: pre.mark_profile_run_elsewhere("ops", "other"))

    assert pre.clear_profile_run_elsewhere("ops", "sandbox-plugin") is False

    assert pre.run_elsewhere("ops").plugin == "other"


def test_the_same_plugin_marking_from_two_processes_at_once_keeps_one_marker(root, monkeypatch):
    _between_look_and_change(monkeypatch, "ops",
                             lambda: pre.mark_profile_run_elsewhere("ops", "sandbox-plugin"))
    entry = pre.mark_profile_run_elsewhere("ops", "sandbox-plugin")
    assert pre.run_elsewhere("ops") == entry and entry.plugin == "sandbox-plugin"
    assert _marker_files(root) == ["ops.json"]


def test_mark_still_publishes_where_the_filesystem_has_no_hard_links(root, monkeypatch):
    """FAT volumes and Android app storage refuse ``os.link``: the marker is then renamed into place."""
    import errno

    def no_links(src, dst, **kwargs):
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(pre.os, "link", no_links)
    entry = pre.mark_profile_run_elsewhere("ops", "sandbox-plugin")
    assert pre.run_elsewhere("ops") == entry and entry.plugin == "sandbox-plugin"
    assert _marker_files(root) == ["ops.json"]
    assert pre.clear_profile_run_elsewhere("ops", "sandbox-plugin") is True


# ── the names a plugin may import ──────────────────────────────────────────────────────────────


def _private_import_warnings(tmp_path: Path, name: str, source: str) -> list[str]:
    import hermes_yaml as yaml
    from hermes_cli.plugin_validate import validate_plugin_dir

    plugin = tmp_path / name
    plugin.mkdir()
    (plugin / "plugin.yaml").write_text(
        yaml.safe_dump({"name": name, "version": "1.0.0", "description": "x"}), encoding="utf-8")
    (plugin / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")
    (plugin / "marker.py").write_text(source, encoding="utf-8")
    report = validate_plugin_dir(plugin)
    assert report.ok, report.failures
    return [w for w in report.warnings if w.startswith("private core imports")]


def test_the_documented_marker_functions_are_public_plugin_api(tmp_path):
    """The user guide tells host-side plugin code to call these, so ``hermes plugins validate`` must
    not call them private. The rest of the module stays internal."""
    assert _private_import_warnings(tmp_path, "documented", (
        "from hermes_cli.profiles_run_elsewhere import (\n"
        "    clear_profile_run_elsewhere, mark_profile_run_elsewhere, run_elsewhere)\n")) == []

    [warning] = _private_import_warnings(tmp_path, "internal", (
        "from hermes_cli.profiles_run_elsewhere import own_profile_dir_when_unlistable\n"
        "from hermes_cli import profiles_run_elsewhere\n"))
    assert "marker.py:1 hermes_cli.profiles_run_elsewhere.own_profile_dir_when_unlistable" in warning
    assert "marker.py:2 hermes_cli.profiles_run_elsewhere" in warning


def test_the_user_guides_marker_example_validates_clean(tmp_path):
    """A plugin that copies the guide's example gets no private-import warning."""
    import ast
    import re

    guide = Path(__file__).resolve().parents[2] / "website/docs/user-guide/multi-profile-gateways.md"
    [block] = [b for b in re.findall(r"```python\n(.*?)```", guide.read_text(encoding="utf-8"), re.DOTALL)
               if "profiles_run_elsewhere" in b]
    imported = {alias.name for node in ast.walk(ast.parse(block)) if isinstance(node, ast.ImportFrom)
                and node.module == "hermes_cli.profiles_run_elsewhere" for alias in node.names}
    assert imported == {"mark_profile_run_elsewhere", "clear_profile_run_elsewhere", "run_elsewhere"}
    assert _private_import_warnings(tmp_path, "guide", block) == []
