"""A fresh (non-cloned) profile keeps the skills the creating profile turned off.

Regression for #119088: a profile created without a clone source got a ``model`` block and
nothing else, so every skill disabled in the active profile came back on in the new one, even
though the Desktop create dialog's Fresh profile preview showed them disabled.
"""
from pathlib import Path

import hermes_yaml as yaml
import pytest

from hermes_cli.profiles import create_profile
from hermes_cli.skills_config import get_disabled_skills


@pytest.fixture()
def default_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _profile_cfg(profile_dir: Path) -> dict:
    path = profile_dir / "config.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}


def test_fresh_profile_keeps_the_active_profiles_disabled_skills(default_home):
    (default_home / "config.yaml").write_text(
        "model:\n  provider: nous\n  default: some/model\n"
        "skills:\n  disabled:\n    - airtable\n    - notion\n    - hermes-agent\n",
        encoding="utf-8",
    )

    cfg = _profile_cfg(create_profile("writer", no_alias=True))

    # Essential skills can never be disabled, so they are not carried over either.
    assert get_disabled_skills(cfg) == {"airtable", "notion"}
    assert cfg["model"] == {"provider": "nous", "default": "some/model"}


def test_fresh_profile_skill_list_is_copied_not_linked(default_home):
    (default_home / "config.yaml").write_text("skills:\n  disabled: [airtable]\n", encoding="utf-8")
    profile_dir = create_profile("writer", no_alias=True)

    (default_home / "config.yaml").write_text("skills:\n  disabled: [notion]\n", encoding="utf-8")

    assert get_disabled_skills(_profile_cfg(profile_dir)) == {"airtable"}


def test_nothing_disabled_writes_no_skills_section(default_home):
    (default_home / "config.yaml").write_text(
        "model:\n  provider: nous\n  default: some/model\nskills:\n  disabled: []\n", encoding="utf-8"
    )

    assert "skills" not in _profile_cfg(create_profile("writer", no_alias=True))
