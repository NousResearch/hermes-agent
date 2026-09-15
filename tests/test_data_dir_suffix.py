"""Default-root isolation through the real Python resolution chain."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.platforms("linux", "macos", "windows")
@pytest.mark.parametrize("suffix", ["", "-asdfasdf", "magic-test", " spaced "])
def test_suffix_scopes_default_home_and_profiles(tmp_path, suffix):
    env = dict(os.environ)
    env.pop("HERMES_HOME", None)
    env.update(
        HOME=str(tmp_path), USERPROFILE=str(tmp_path),
        LOCALAPPDATA=str(tmp_path / "AppData" / "Local"),
        HERMES_DATA_DIR_SUFFIX=suffix,
    )
    script = """
import json
import os
from pathlib import Path
from hermes_constants import get_hermes_home, get_process_hermes_home, get_default_hermes_root
from hermes_cli.profiles import _get_profiles_root, _get_active_profile_path, resolve_profile_env
root = get_default_hermes_root()
profile = root / 'profiles' / 'coder'
profile.mkdir(parents=True)
result = [str(get_hermes_home()), str(get_process_hermes_home()), str(root),
          str(_get_profiles_root()), str(_get_active_profile_path()), resolve_profile_env('coder')]
os.environ['HERMES_HOME'] = str(profile)
result.extend([str(get_hermes_home()), str(get_default_hermes_root())])
os.environ.pop('HERMES_HOME')
os.environ['HERMES_DATA_DIR_SUFFIX'] += '-changed'
result.append(str(get_default_hermes_root()))
print(json.dumps(result))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], env=env,
        cwd=Path(__file__).resolve().parents[1],
        text=True, capture_output=True, check=True,
    )
    base = tmp_path / "AppData" / "Local" / "hermes" if sys.platform == "win32" else tmp_path / ".hermes"
    root = Path(str(base) + suffix)
    profile = root / "profiles" / "coder"
    assert json.loads(result.stdout) == list(map(str, [
        root, root, root, root / "profiles", root / "active_profile", profile,
        profile, root, Path(str(root) + "-changed"),
    ]))


@pytest.mark.platforms("linux", "macos", "windows")
def test_startup_readers_use_the_suffixed_home(tmp_path, monkeypatch):
    from hermes_constants import get_process_hermes_home
    from hermes_cli.dashboard_procs import _hermes_home_dir
    from hermes_cli.env_loader import load_hermes_dotenv
    from hermes_startup_watchdog import get_startup_watchdog_dump_path

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("HERMES_DATA_DIR_SUFFIX", "magic-test")
    monkeypatch.delenv("SUFFIX_TEST_CREDENTIAL", raising=False)
    home = get_process_hermes_home()
    home.mkdir(parents=True)
    (home / ".env").write_text("SUFFIX_TEST_CREDENTIAL=suffixed\n", encoding="utf-8")

    load_hermes_dotenv(project_env=tmp_path / "absent.env", load_external_secrets=False)
    assert os.environ.get("SUFFIX_TEST_CREDENTIAL") == "suffixed"
    assert _hermes_home_dir() == home
    assert get_startup_watchdog_dump_path() == home / "logs" / "gateway-startup-watchdog.log"


def test_suffix_does_not_change_explicit_or_context_home(tmp_path, monkeypatch):
    from hermes_constants import (
        get_default_hermes_root, get_hermes_home, get_process_hermes_home,
        reset_hermes_home_override, set_hermes_home_override,
    )

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
    explicit = tmp_path / "explicit"
    scoped = tmp_path / "scoped"
    monkeypatch.setenv("HERMES_HOME", str(explicit))
    monkeypatch.setenv("HERMES_DATA_DIR_SUFFIX", "magic-test")
    assert get_hermes_home() == get_process_hermes_home() == get_default_hermes_root() == explicit
    token = set_hermes_home_override(scoped)
    try:
        assert get_hermes_home() == scoped
        assert get_process_hermes_home() == get_default_hermes_root() == explicit
    finally:
        reset_hermes_home_override(token)
