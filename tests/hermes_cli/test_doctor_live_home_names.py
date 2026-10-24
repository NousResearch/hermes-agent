"""``doctor.HERMES_HOME`` / ``doctor._DHH`` stay live after a test patches them.

``monkeypatch.setattr`` restores the value it read as a real module attribute, which would
shadow the module ``__getattr__`` for the rest of the session; the autouse fixture in this
directory's conftest drops the names again. These two tests run in file order: the first
patches, the second must still see the live home.
"""

from hermes_cli import doctor
from hermes_cli.config import get_hermes_home


def test_a_patching_doctor_home_names(tmp_path, monkeypatch):
    monkeypatch.setattr(doctor, "HERMES_HOME", tmp_path / "patched")
    monkeypatch.setattr(doctor, "_DHH", str(tmp_path / "patched"))
    assert doctor.HERMES_HOME == tmp_path / "patched"


def test_b_names_resolve_live_after_the_patch_is_undone(tmp_path, monkeypatch):
    home = tmp_path / "live_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert "HERMES_HOME" not in vars(doctor) and "_DHH" not in vars(doctor)
    assert doctor.HERMES_HOME == get_hermes_home() == home
