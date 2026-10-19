"""Profile-targeted ``.env`` writer: explicit path, locked atomic write, private mode, no publish."""

import os
import stat
from pathlib import Path

import pytest

from hermes_cli.config_env_write import (
    env_file_defines_key, profile_env_path, save_env_value_to_profile)

SENTINEL = "sentinel-value-7f3a9c"


@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.mark.platforms("posix")
def test_new_env_file_gets_the_key_and_mode_0600(profile_env):
    save_env_value_to_profile(profile_env, "STRIPE_SECRET_KEY", SENTINEL)

    env = profile_env / ".env"
    assert env.read_text() == f"STRIPE_SECRET_KEY={SENTINEL}\n"
    assert _mode(env) == 0o600


@pytest.mark.platforms("posix")
def test_existing_lines_comments_and_order_are_kept(profile_env):
    env = profile_env / ".env"
    env.write_text("# comment\nA=1\nSTRIPE_SECRET_KEY=old\nB=2\n")
    os.chmod(env, 0o600)

    save_env_value_to_profile(profile_env, "STRIPE_SECRET_KEY", SENTINEL)
    save_env_value_to_profile(profile_env, "NEW_KEY", "x")

    assert env.read_text() == f"# comment\nA=1\nSTRIPE_SECRET_KEY={SENTINEL}\nB=2\nNEW_KEY=x\n"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("before, after", [(0o640, 0o640), (0o400, 0o400), (0o660, 0o600), (0o644, 0o600)])
def test_existing_mode_is_kept_only_when_private(profile_env, before, after):
    env = profile_env / ".env"
    env.write_text("A=1\n")
    os.chmod(env, before)

    save_env_value_to_profile(profile_env, "K", "v")

    assert _mode(env) == after


def test_value_is_not_published_to_the_process_environment(profile_env, monkeypatch):
    monkeypatch.delenv("PUBLISH_PROBE_KEY", raising=False)
    save_env_value_to_profile(profile_env, "PUBLISH_PROBE_KEY", SENTINEL)
    assert "PUBLISH_PROBE_KEY" not in os.environ


@pytest.mark.parametrize("value", ["a\nb", "a\rb", "a\0b"])
def test_values_with_line_breaks_or_nul_are_refused_not_stripped(profile_env, value):
    with pytest.raises(ValueError) as exc:
        save_env_value_to_profile(profile_env, "K", value)
    assert value not in str(exc.value)
    assert not (profile_env / ".env").exists()


@pytest.mark.parametrize("key", ["PATH", "LD_PRELOAD", "1BAD", "A-B"])
def test_denied_or_invalid_key_names_are_refused(profile_env, key):
    with pytest.raises(ValueError):
        save_env_value_to_profile(profile_env, key, "v")
    assert not (profile_env / ".env").exists()


@pytest.mark.platforms("posix")
def test_symlinked_env_outside_the_profile_is_refused(profile_env, tmp_path):
    outside = tmp_path / "outside.env"
    outside.write_text("A=1\n")
    (profile_env / ".env").symlink_to(outside)

    with pytest.raises(ValueError):
        profile_env_path(profile_env)
    with pytest.raises(ValueError):
        save_env_value_to_profile(profile_env, "K", "v")
    assert outside.read_text() == "A=1\n"


def test_env_file_defines_key_reads_plain_and_export_lines(profile_env):
    env = profile_env / ".env"
    assert env_file_defines_key(env, "K") is False
    env.write_text("export K=1\nOTHER=2\n")
    assert env_file_defines_key(env, "K") is True
    assert env_file_defines_key(env, "MISSING") is False
