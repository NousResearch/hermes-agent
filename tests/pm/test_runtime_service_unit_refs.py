"""A generation a supervisor definition still names is never reclaimed (#133184).

A hand-written systemd unit whose Exec directives hard-code a generation path
keeps starting long after the update that superseded it: reclaiming that tree is
what turns a stale reference into a ``203/EXEC`` restart loop, and the update
that reclaimed it reported success throughout. Collection now skips any
generation a definition under the user/system unit directories still names (and
warns which one), instead of deleting the tree out from under the service.
"""

import json
import logging
from pathlib import Path

from hermes_cli.runtime_state import collect_generations

SELECTED = "aa" * 16
REFERENCED = "bb" * 16
SPARE = "cc" * 16


def _install(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    from pm.environments import install_state_dir

    return repo, install_state_dir(repo)


def _generation(repo, name):
    from pm.environments import install_state_dir, site_packages

    venv = install_state_dir(repo) / "environments" / name / "venv"
    venv.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("version = 3.11", encoding="utf-8")
    site_packages(venv).mkdir(parents=True)
    (venv.parent / ".lease-managed").touch()
    return venv.parent


def _select(repo, name):
    from pm.environments import install_state_dir, runtime_facts_path

    venv = install_state_dir(repo) / "environments" / name / "venv"
    runtime_facts_path(repo).write_text(
        json.dumps({"packages": {"venv": {"environment": str(venv)}}}), encoding="utf-8")


def _unit(directory: Path, unit_name: str, exec_path: Path, *, dropin: bool = False) -> Path:
    path = directory / (f"{unit_name}.service.d/10-override.conf" if dropin
                        else f"{unit_name}.service")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"[Service]\nExecStart={exec_path}/venv/bin/hermes gateway run\n",
                    encoding="utf-8")
    return path


def test_collector_keeps_a_generation_a_service_unit_still_references(
        tmp_path, monkeypatch, caplog):
    """The generation a unit pins is exactly the tree collection must not reclaim."""
    from hermes_cli import runtime_state

    repo, state = _install(tmp_path, monkeypatch)
    _generation(repo, SELECTED)
    _generation(repo, REFERENCED)
    _generation(repo, SPARE)
    _select(repo, SELECTED)

    unit_dir = tmp_path / "systemd-user"
    unit = _unit(unit_dir, "hermes-gateway", state / "environments" / REFERENCED)
    monkeypatch.setattr(runtime_state, "_service_unit_directories", lambda: [unit_dir])

    with caplog.at_level(logging.WARNING, logger="hermes_cli.runtime_state"):
        removed = [p.resolve() for p in collect_generations(repo, min_age_seconds=0)]

    assert removed == [(state / "environments" / SPARE).resolve()]
    assert (state / "environments" / REFERENCED).is_dir()
    assert (state / "environments" / SELECTED).is_dir()
    assert unit.is_file(), "the definition itself is never rewritten here"
    assert any(REFERENCED in record.getMessage()
               for record in caplog.records if record.levelno == logging.WARNING), \
        "the kept reference is reported, not silently skipped"


def test_collector_reclaims_normally_when_no_unit_references_the_generation(tmp_path, monkeypatch):
    """A unit naming some other tree must not shield a dead generation."""
    from hermes_cli import runtime_state

    repo, state = _install(tmp_path, monkeypatch)
    _generation(repo, SELECTED)
    _generation(repo, SPARE)
    _select(repo, SELECTED)

    unit_dir = tmp_path / "systemd-user"
    _unit(unit_dir, "hermes-gateway", tmp_path / "elsewhere")
    monkeypatch.setattr(runtime_state, "_service_unit_directories", lambda: [unit_dir])

    removed = [p.resolve() for p in collect_generations(repo, min_age_seconds=0)]

    assert removed == [(state / "environments" / SPARE).resolve()]


def test_reference_scan_reads_service_and_dropin_files_in_both_directories(tmp_path, monkeypatch):
    """Drop-ins count too: one can replace ExecStart for a live unit."""
    from hermes_cli import runtime_state

    repo, state = _install(tmp_path, monkeypatch)
    _generation(repo, REFERENCED)
    _generation(repo, SPARE)
    user_dir = tmp_path / "systemd-user"
    system_dir = tmp_path / "systemd-system"
    from_user = _unit(user_dir, "hermes-gateway", state / "environments" / REFERENCED)
    from_dropin = _unit(system_dir, "custom-name", state / "environments" / SPARE, dropin=True)
    _unit(system_dir, "unrelated", tmp_path / "unrelated")
    monkeypatch.setattr(
        runtime_state, "_service_unit_directories", lambda: [user_dir, system_dir])

    references = runtime_state.service_unit_generation_references(repo)

    assert references == {REFERENCED: from_user, SPARE: from_dropin}


def test_reference_matches_a_bare_path_and_rejects_lookalike_directories(tmp_path, monkeypatch):
    """A directive that ends exactly at the generation path (e.g. ``WorkingDirectory=``)
    still names it, while a ``<name>-backup`` sibling is a different directory."""
    from hermes_cli import runtime_state

    repo, state = _install(tmp_path, monkeypatch)
    _generation(repo, SELECTED)
    _generation(repo, REFERENCED)
    _generation(repo, SPARE)
    _select(repo, SELECTED)

    unit_dir = tmp_path / "systemd-user"
    unit = _unit(unit_dir, "hermes-gateway", state / "environments" / REFERENCED)
    with unit.open("a", encoding="utf-8") as handle:
        handle.write(f"WorkingDirectory={state / 'environments' / REFERENCED}\n")
    lookalike = state / "environments" / f"{SPARE}-backup"
    lookalike.mkdir()
    (unit_dir / "lookalike.service").write_text(
        f"[Service]\nExecStart={lookalike}/venv/bin/hermes\n", encoding="utf-8")
    monkeypatch.setattr(runtime_state, "_service_unit_directories", lambda: [unit_dir])

    removed = [p.name for p in collect_generations(repo, min_age_seconds=0)]

    assert removed == [SPARE], "the bare path reference is kept; its -backup sibling does not shield it"
    assert (state / "environments" / REFERENCED).is_dir()


def test_another_installs_definition_does_not_shield_this_installs_generation(tmp_path, monkeypatch):
    """Only paths under THIS install's state dir count: a unit naming some other
    install's identically-shaped tree must not block this install's collection."""
    from hermes_cli import runtime_state

    repo, state = _install(tmp_path, monkeypatch)
    _generation(repo, SELECTED)
    _generation(repo, REFERENCED)
    _select(repo, SELECTED)

    unit_dir = tmp_path / "systemd-user"
    other_state = tmp_path / "other-home" / "installs" / "0000000000000000"
    _unit(unit_dir, "other-install", other_state / "environments" / REFERENCED)
    monkeypatch.setattr(runtime_state, "_service_unit_directories", lambda: [unit_dir])

    removed = [p.name for p in collect_generations(repo, min_age_seconds=0)]

    assert removed == [REFERENCED]
