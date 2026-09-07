"""Home initialization must respect operator-owned links and diagnose storage."""
import shutil
from pathlib import Path

import pytest

from hermes_cli import config


@pytest.mark.linux_only
@pytest.mark.parametrize("subdir", (".", *config._HERMES_HOME_SUBDIRS))
def test_unavailable_directory_links_are_diagnosed_without_creating_targets(tmp_path, monkeypatch, subdir):
    home = tmp_path / "hermes"
    link = home / subdir
    link.parent.mkdir(parents=True, exist_ok=True)
    target = tmp_path / "unmounted" / "external"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(config, "get_hermes_home", lambda: home)
    monkeypatch.setattr(config, "is_managed", lambda: False)
    config._HERMES_HOME_ENSURED.pop(str(home), None)

    issues = config.validate_config_structure()

    assert issues
    text = " ".join(str(issue) for issue in issues)
    assert str(link) in text and str(target) in text
    assert "mount" in text.lower() and "setup" not in text.lower()
    assert link.is_symlink() and link.readlink() == target
    assert not target.parent.exists()
    assert str(home) not in config._HERMES_HOME_ENSURED

    target.mkdir(parents=True, mode=0o750)
    config.ensure_hermes_home()
    assert link.is_symlink() and target.stat().st_mode & 0o777 == 0o750
    assert str(home) in config._HERMES_HOME_ENSURED


@pytest.mark.linux_only
@pytest.mark.parametrize("linked", ("plain", "logs", "home"))
def test_initialization_preserves_external_directory_modes(tmp_path, monkeypatch, linked):
    home = tmp_path / "hermes"
    target = tmp_path / "shared"
    target.mkdir(mode=0o750)
    if linked == "home":
        home.symlink_to(target, target_is_directory=True)
    else:
        home.mkdir()
    curator = target / "curator"
    curator.mkdir(mode=0o750)
    if linked == "logs":
        (home / "logs").symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(config, "get_hermes_home", lambda: home)
    monkeypatch.setattr(config, "is_managed", lambda: False)
    config._HERMES_HOME_ENSURED.pop(str(home), None)

    config.ensure_hermes_home()
    monkeypatch.setattr(config, "get_hermes_home", lambda: home.resolve())
    config.ensure_hermes_home()

    assert all((home / name).is_dir() for name in config._HERMES_HOME_SUBDIRS)
    assert (home / "SOUL.md").is_file()
    if linked != "plain":
        assert (home if linked == "home" else home / "logs").is_symlink()
        assert target.stat().st_mode & 0o777 == 0o750
        assert curator.stat().st_mode & 0o777 == 0o750
    else:
        assert (home / "logs").stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("managed", (False, True))
def test_named_profile_disappearance_during_initialization_never_recreates_home(
    tmp_path, monkeypatch, managed,
):
    from hermes_cli.config_home import HomeInitializationError

    root = tmp_path / ".hermes"
    home = root / "profiles" / "worker"
    for subdir in ("cron", "sessions", "logs", "memories"):
        (home / subdir).mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(config, "get_hermes_home", lambda: home)
    monkeypatch.setattr(config, "is_managed", lambda: managed)
    original_mkdir = Path.mkdir
    removed = False

    def remove_before_mkdir(path, *args, **kwargs):
        nonlocal removed
        if not removed and path == home / "logs" / "curator":
            removed = True
            shutil.rmtree(home)
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", remove_before_mkdir)
    with pytest.raises(HomeInitializationError, match="Cannot initialize Hermes directory"):
        config.ensure_hermes_home()

    assert not home.exists()
    assert str(home) not in config._HERMES_HOME_ENSURED
