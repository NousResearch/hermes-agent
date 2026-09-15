"""Shipped plugin callers keep their contract without owning PM publication."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _plugin(home, name, *, dependencies=True):
    path = home / "plugins" / name
    path.mkdir(parents=True)
    (path / "plugin.yaml").write_text(f"name: {name}\n", encoding="utf-8")
    if dependencies:
        (path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("active", [None, "default", "profile"])
def test_candidate_member_dirs_preserves_proposed_home_order_and_extras(isolated_home, monkeypatch, active):
    from hermes_cli import plugins_admission

    home = isolated_home
    profile = home / "profiles" / "coder"
    profile.mkdir(parents=True)
    for directory in (home, profile):
        (directory / "config.yaml").write_text("plugins:\n  enabled: [old]\n", encoding="utf-8")
        _plugin(directory, "old")
        _plugin(directory, "new")
        _plugin(directory, "blocked")
        _plugin(directory, "plain", dependencies=False)
    extra = _plugin(home, "extra")
    selected_home = profile if active == "profile" else home
    expected = [home / "plugins" / "old", profile / "plugins" / "old"]
    if active is not None:
        expected[active == "profile"] = selected_home / "plugins" / "new"
    extra_existing = expected[0]
    before = {p: p.read_bytes() for p in home.rglob("*") if p.is_file()}
    # Discovery must use PM's independent manifest reader, not application config/UI.
    monkeypatch.setitem(sys.modules, "hermes_cli.config", None)
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins_cmd", None)
    result = plugins_admission.candidate_member_dirs(
        iter(["new", "blocked", "plain", "new"]), iter(["blocked"]),
        active_plugins_dir=str(selected_home / "plugins") if active else None,
        extra_dirs=iter([str(extra_existing), str(extra), str(extra), home / "plugins" / "plain", home / "missing"]),
    )
    assert isinstance(result, list)
    assert result == [*expected, extra]
    assert {p: p.read_bytes() for p in home.rglob("*") if p.is_file()} == before


@pytest.fixture
def publication(isolated_home, tmp_path):
    from hermes_cli.runtime_paths import install_state_dir

    project = tmp_path / "checkout"
    project.mkdir()
    target = _plugin(isolated_home, "published")
    (target / "version").write_bytes(b"new code")
    backup = _plugin(isolated_home, ".previous-historic")
    (backup / "version").write_bytes(b"old code")
    metadata = target.parent / ".install-metadata.json"
    metadata.write_bytes(b'{"published": "new"}\n')
    # Old callers pass the already-decoded row and exact journal path. The
    # adapter must not rediscover/reparse a different current journal.
    journal = tmp_path / "historic-publication.json"
    journal.write_bytes(b"already decoded by the old caller")
    canonical = install_state_dir(project) / "publication.json"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"unrelated publication; do not read or remove")
    row = {
        "kind": "plugin", "target": str(target), "backup": str(backup),
        "metadata": str(metadata), "target_existed": True, "facts_before": None,
        "metadata_before": base64.b64encode(b'{"published": "old"}\n').decode(),
        "metadata_after": base64.b64encode(metadata.read_bytes()).decode(),
    }
    return project, row, journal, canonical


def _recover_in_stdlib(publication):
    project, row, journal, _ = publication
    script = """
import json
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
# No selected application environment, no PM, and no CLI config/UI reader.
for module in ('pm', 'hermes_cli.config', 'hermes_cli.plugins_cmd'):
    sys.modules[module] = None
from hermes_cli import plugins_transaction
row = json.loads(sys.stdin.read())
plugins_transaction.recover_plugin_publication(
    project=Path(sys.argv[2]), row=row, journal=Path(sys.argv[3]),
)
"""
    return subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(Path(__file__).resolve().parents[1]), str(project), str(journal)],
        input=json.dumps(row), text=True, capture_output=True, check=False,
        env={**os.environ, "HERMES_HOME": str(Path(row["metadata"]).parent.parent)},
    )


@pytest.mark.parametrize("commit", ["rollback", "explicit", "facts-changed"])
def test_old_publication_recovers_in_stdlib_using_supplied_row_and_journal(publication, commit):
    from hermes_cli.runtime_paths import runtime_facts_path

    project, row, journal, canonical = publication
    target, backup, metadata = (Path(row[key]) for key in ("target", "backup", "metadata"))
    if commit == "explicit":
        row["committed"] = True
    elif commit == "facts-changed":
        runtime_facts_path(project).write_bytes(b"new facts")
    result = _recover_in_stdlib(publication)
    assert result.returncode == 0, result.stderr
    assert (target / "version").read_bytes() == (b"old code" if commit == "rollback" else b"new code")
    expected = row["metadata_before"] if commit == "rollback" else row["metadata_after"]
    assert metadata.read_bytes() == base64.b64decode(expected)
    assert not backup.exists()
    assert not journal.exists()
    assert canonical.read_bytes() == b"unrelated publication; do not read or remove"


def test_old_publication_rolls_back_a_first_install(publication):
    from hermes_cli import plugins_transaction
    import shutil

    project, row, journal, _ = publication
    shutil.rmtree(row["backup"])
    row.update(target_existed=False, metadata_before=None)
    plugins_transaction.recover_plugin_publication(project, row, journal)
    assert not Path(row["target"]).exists()
    assert not Path(row["metadata"]).exists()
    assert not journal.exists()


@pytest.mark.parametrize("invalid", ["target", "backup", "metadata", "edited-metadata"])
def test_old_publication_refuses_unsafe_or_changed_state_without_writes(publication, tmp_path, invalid):
    from hermes_cli import plugins_transaction

    project, row, journal, _ = publication
    if invalid == "edited-metadata":
        Path(row["metadata"]).write_bytes(b"independent user edit")
    else:
        outside = tmp_path / "outside" / "plugins"
        outside.mkdir(parents=True)
        row[invalid] = str(outside / Path(row[invalid]).name)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="paths escape|metadata changed"):
        plugins_transaction.recover_plugin_publication(project, row, journal)
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    assert journal.exists()
