"""Home initialization must respect operator-owned links and diagnose storage."""
import stat
import sys
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
    config._HERMES_HOME_ENSURED.discard(str(home))

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
    config._HERMES_HOME_ENSURED.discard(str(home))

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


@pytest.mark.linux_only
@pytest.mark.parametrize("existing", (False, True))
def test_symlinked_parent_above_home_is_not_an_operator_home_link(
    tmp_path, monkeypatch, existing
):
    """A link above HERMES_HOME is not an operator-owned home link.

    macOS aliases ``/tmp`` -> ``/private/tmp`` and ``/var`` -> ``/private/var``, so a home
    under the default temp root arrives with a symlinked parent; the same happens for any
    user whose own directory is aliased. The home and its subdirectories are still ours to
    secure: they must end up 0o700, both when created fresh and when a previous run left them
    at the 0o755 default.
    """
    real_root = tmp_path / "real"
    real_root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real_root, target_is_directory=True)
    home = alias / "hermes"
    if existing:
        home.mkdir(parents=True, mode=0o755)
        for name in config._HERMES_HOME_SUBDIRS:
            (home / name).mkdir(mode=0o755)
        assert stat.S_IMODE(home.stat().st_mode) == 0o755

    monkeypatch.setattr(config, "get_hermes_home", lambda: home)
    monkeypatch.setattr(config, "is_managed", lambda: False)
    config._HERMES_HOME_ENSURED.discard(str(home))
    config._HERMES_HOME_ENSURED.discard(str(home.resolve()))

    config.ensure_hermes_home()

    assert stat.S_IMODE(home.stat().st_mode) == 0o700
    for name in config._HERMES_HOME_SUBDIRS:
        mode = stat.S_IMODE((home / name).stat().st_mode)
        assert mode == 0o700, f"{name} should be 0700, got 0o{mode:o}"


@pytest.mark.linux_only
def test_aliased_parent_still_leaves_an_operator_home_link_alone(tmp_path, monkeypatch):
    """An operator-owned link at the home boundary keeps owning the mode, aliased parent or not."""
    real_root = tmp_path / "real"
    real_root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real_root, target_is_directory=True)
    shared = real_root / "shared"
    shared.mkdir(mode=0o750)
    (shared / "curator").mkdir(mode=0o750)
    home = alias / "hermes"
    home.symlink_to(shared, target_is_directory=True)

    monkeypatch.setattr(config, "get_hermes_home", lambda: home)
    monkeypatch.setattr(config, "is_managed", lambda: False)
    config._HERMES_HOME_ENSURED.discard(str(home))
    config._HERMES_HOME_ENSURED.discard(str(home.resolve()))

    config.ensure_hermes_home()

    assert home.is_symlink() and home.readlink() == shared
    assert stat.S_IMODE(shared.stat().st_mode) == 0o750
    assert stat.S_IMODE((shared / "curator").stat().st_mode) == 0o750


def test_initialization_refuses_the_callers_account_home(tmp_path, monkeypatch):
    """A HERMES_HOME that IS the account root must fail loudly, never chmod it."""
    import os

    from hermes_cli import config_home

    account_root = tmp_path / "acct"
    account_root.mkdir(mode=0o755)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: account_root))
    monkeypatch.setattr(config, "is_managed", lambda: False)
    config._HERMES_HOME_ENSURED.discard(str(account_root))

    with pytest.raises(config_home.HomeInitializationError, match="account root"):
        config_home.initialize_home(
            account_root, config._HERMES_HOME_SUBDIRS, config._HERMES_HOME_ENSURED
        )

    assert not (account_root / "SOUL.md").exists()
    assert not any((account_root / name).exists() for name in config._HERMES_HOME_SUBDIRS)
    assert str(account_root) not in config._HERMES_HOME_ENSURED
    if os.name == "posix":
        assert account_root.stat().st_mode & 0o777 == 0o755


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX account-root layout")
@pytest.mark.parametrize(
    "bad_home",
    ("/", "/Users", "/Users/other-acct", "/home/other-acct", "/root", "/var/root"),
)
def test_other_account_home_roots_are_refused(bad_home):
    from hermes_cli.config_home import _is_account_home_root

    assert _is_account_home_root(Path(bad_home))


def test_dedicated_home_directories_are_allowed(tmp_path):
    from hermes_cli.config_home import _is_account_home_root

    assert not _is_account_home_root(tmp_path / "hermes")
    assert not _is_account_home_root(Path.home() / ".hermes")
    assert not _is_account_home_root(Path.home() / "hermes-tenant")
