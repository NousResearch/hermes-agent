"""Source probes retain home security without importing config or seeding it."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hermes_cli.config_home import HomeInitializationError, initialize_probe_home


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory permissions")
def test_probe_home_is_private_and_import_light(tmp_path, monkeypatch):
    home = tmp_path / "probe"
    monkeypatch.setenv("HERMES_HOME", str(home))
    for key in ("HERMES_MANAGED", "HERMES_HOME_MODE", "HERMES_SKIP_CHMOD", "HERMES_CONTAINER"):
        monkeypatch.delenv(key, raising=False)
    script = """
import os, stat, sys
from pathlib import Path
from hermes_cli.config_home import initialize_probe_home
home = Path(os.environ['HERMES_HOME'])
os.umask(0o022)
initialize_probe_home(home)
assert 'hermes_cli.config' not in sys.modules
assert stat.S_IMODE(home.stat().st_mode) == 0o700
assert stat.S_IMODE((home / 'cache').stat().st_mode) == 0o700
assert sorted(p.name for p in home.iterdir()) == ['cache']
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[2],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(os.name == "nt", reason="Symlinks require elevated Windows privileges")
@pytest.mark.parametrize("location", ["ancestor", "home", "cache"])
def test_source_probe_rejects_missing_link_target(tmp_path, location):
    from hermes_cli.source_check import check_for_updates

    missing = tmp_path / "missing-mount"
    if location == "ancestor":
        link = tmp_path / "mount"
        link.symlink_to(missing, target_is_directory=True)
        home = link / "profile"
    elif location == "home":
        home = tmp_path / "profile"
        link = home
        link.symlink_to(missing, target_is_directory=True)
    else:
        home = tmp_path / "profile"
        home.mkdir()
        link = home / "cache"
        link.symlink_to(missing, target_is_directory=True)
    with pytest.raises(HomeInitializationError, match="Directory link is unavailable"):
        check_for_updates(install_root=tmp_path / "checkout", home=home)
    assert link.is_symlink()
    assert not missing.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink and permission semantics")
def test_probe_preserves_operator_owned_link_modes(tmp_path):
    import stat

    target = tmp_path / "mounted-home"
    target.mkdir(mode=0o750)
    cache = target / "cache"
    cache.mkdir(mode=0o750)
    home = tmp_path / "profile"
    home.symlink_to(target, target_is_directory=True)
    initialize_probe_home(home)
    assert stat.S_IMODE(target.stat().st_mode) == 0o750
    assert stat.S_IMODE(cache.stat().st_mode) == 0o750
