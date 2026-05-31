"""Regression tests for doctor security checks."""

import os

from hermes_cli import doctor


def test_iter_env_files_for_permission_check_finds_root_and_profiles(tmp_path):
    root_env = tmp_path / ".env"
    root_env.write_text("ROOT=1", encoding="utf-8")
    profile_env = tmp_path / "profiles" / "legacy" / ".env"
    profile_env.parent.mkdir(parents=True)
    profile_env.write_text("PROFILE=1", encoding="utf-8")

    env_files = doctor._iter_env_files_for_permission_check(tmp_path, root_env)

    assert root_env in env_files
    assert profile_env in env_files


def test_env_file_permission_check_detects_group_or_other_bits(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("TOKEN=value", encoding="utf-8")
    os.chmod(env_path, 0o644)

    assert doctor._env_file_is_group_or_other_readable(env_path)

    assert doctor._tighten_env_file_permissions(env_path)
    assert not doctor._env_file_is_group_or_other_readable(env_path)
    assert env_path.stat().st_mode & 0o777 == 0o600
