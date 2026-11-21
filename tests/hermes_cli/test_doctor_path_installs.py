"""`hermes doctor` names every `hermes` on PATH and warns when the first one is another install."""

from argparse import Namespace

import pytest

from hermes_cli import doctor, doctor_path_installs


def _command(bin_dir, install_dir, version="Hermes Agent v9.9.9 (2026.1.1)", code=0):
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / "hermes"
    script.write_text(
        f"#!/bin/sh\necho '{version}'\necho 'Install directory: {install_dir}'\nexit {code}\n", encoding="utf-8"
    )
    script.chmod(0o755)
    return script


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("other_first", [True, False])
def test_first_hermes_on_path_must_be_this_install(tmp_path, monkeypatch, capsys, other_first):
    ours, other = tmp_path / "ours", tmp_path / "old-pip"
    ours.mkdir(), other.mkdir()
    mine = _command(tmp_path / "bin-ours", ours, "Hermes Agent v1.0.0 (2026.11.1)")
    stale = _command(tmp_path / "bin-old", other, "Hermes Agent v0.3.0 (2025.1.1)")
    alias = tmp_path / "bin-alias"
    alias.mkdir()
    (alias / "hermes").symlink_to(mine)  # same target twice is one command
    order = [stale.parent, mine.parent, alias] if other_first else [mine.parent, alias, stale.parent]
    monkeypatch.setenv("PATH", ":".join(map(str, order)))
    monkeypatch.setattr(doctor, "PROJECT_ROOT", ours)
    monkeypatch.setattr(doctor, "DOCTOR_CHECKS", (("Hermes Commands on PATH", doctor_path_installs._check_path_installs),))

    code = doctor.run_doctor(Namespace(fix=False))

    out = capsys.readouterr().out
    assert "v0.3.0" in out and "v1.0.0" in out
    assert str(alias / "hermes") not in out
    assert ("runs first on PATH, but is not this install" in out) is other_first
    assert ("shadowed" in out) is not other_first
    assert code == int(other_first)


@pytest.mark.platforms("posix")
def test_broken_first_command_is_reported_not_raised(tmp_path, monkeypatch, capsys):
    ours = tmp_path / "ours"
    ours.mkdir()
    broken = _command(tmp_path / "bin-broken", "", "", code=1)
    monkeypatch.setenv("PATH", str(broken.parent))
    monkeypatch.setattr(doctor, "PROJECT_ROOT", ours)
    monkeypatch.setattr(doctor, "DOCTOR_CHECKS", ((None, doctor_path_installs._check_path_installs),))

    assert doctor.run_doctor(Namespace(fix=False)) == 1
    assert "--version failed: exit code 1" in capsys.readouterr().out
