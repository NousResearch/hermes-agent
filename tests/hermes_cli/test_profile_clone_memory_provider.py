"""RED test for #120115: light ``--clone`` must carry the memory provider's
in-profile config dir (e.g. ``hindsight/``), not just ``memory.provider`` in config.yaml.
"""

from pathlib import Path

import pytest
import yaml

from hermes_cli import profiles
from hermes_cli.profiles import create_profile


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return tmp_path


def _source_with_hindsight(default_home: Path):
    (default_home / "config.yaml").write_text(
        yaml.safe_dump({"memory": {"provider": "hindsight", "hindsight": {"mode": "local"}}}),
        encoding="utf-8",
    )
    (default_home / "hindsight").mkdir(exist_ok=True)
    (default_home / "hindsight" / "config.json").write_text('{"mode": "local_embedded"}', encoding="utf-8")


def test_clone_copies_memory_provider_config_dir(profile_env):
    """Clone keeps memory.provider AND copies hindsight/config.json."""
    default_home = profile_env / ".hermes"
    _source_with_hindsight(default_home)

    clone = create_profile("work", clone_config=True, no_alias=True)

    cloned_cfg = yaml.safe_load((clone / "config.yaml").read_text(encoding="utf-8"))
    assert cloned_cfg["memory"]["provider"] == "hindsight"
    assert (clone / "hindsight" / "config.json").is_file()
    assert (clone / "hindsight" / "config.json").read_text(encoding="utf-8") == '{"mode": "local_embedded"}'


def test_clone_without_provider_dir_still_works(profile_env):
    """Builtin provider (no <provider>/ dir) clones fine and creates nothing extra."""
    default_home = profile_env / ".hermes"
    (default_home / "config.yaml").write_text(
        yaml.safe_dump({"memory": {"provider": "builtin"}}), encoding="utf-8"
    )

    clone = create_profile("plain", clone_config=True, no_alias=True)

    assert (clone / "config.yaml").is_file()
    assert not (clone / "builtin").exists()


def test_clone_ignores_malicious_provider_name(profile_env):
    """A traversal-shaped memory.provider must not escape the source profile."""
    default_home = profile_env / ".hermes"
    (default_home / "config.yaml").write_text(
        yaml.safe_dump({"memory": {"provider": "../evil"}}), encoding="utf-8"
    )
    (profile_env / "evil").mkdir(exist_ok=True)
    (profile_env / "evil" / "pwned.txt").write_text("x", encoding="utf-8")

    clone = create_profile("safe", clone_config=True, no_alias=True)

    assert (clone / "config.yaml").is_file()
    assert (profile_env / "evil" / "pwned.txt").read_text() == "x"  # source untouched
    assert not list(clone.glob("evil*"))
