"""Regression tests: the embedded Hindsight profile env file carries the
plaintext ``HINDSIGHT_API_LLM_API_KEY`` and must be created/kept owner-only
(0600), and must not survive a failed post-write permission validation.
"""

import os
import stat
from pathlib import Path

import pytest

from plugins.memory.hindsight import (
    _embedded_profile_env_path,
    _materialize_embedded_profile_env,
)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    isolated_home = tmp_path / "user-home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: isolated_home))
    return isolated_home


_CONFIG = {
    "profile": "hermes",
    "llm_provider": "openai",
    "llm_model": "gpt-4o-mini",
}


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not enforced on Windows")
def test_fresh_profile_env_is_owner_only_despite_permissive_umask():
    old_umask = os.umask(0o022)
    try:
        profile_env = _materialize_embedded_profile_env(
            _CONFIG, llm_api_key="sk-hindsight-secret"
        )
    finally:
        os.umask(old_umask)

    assert profile_env.exists()
    assert stat.S_IMODE(profile_env.stat().st_mode) == 0o600
    assert "HINDSIGHT_API_LLM_API_KEY=sk-hindsight-secret\n" in profile_env.read_text(
        encoding="utf-8"
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not enforced on Windows")
def test_rewrite_tightens_existing_world_readable_profile_env():
    profile_env = _embedded_profile_env_path(_CONFIG)
    profile_env.parent.mkdir(parents=True)
    profile_env.write_text("HINDSIGHT_API_LLM_API_KEY=stale\n", encoding="utf-8")
    os.chmod(profile_env, 0o644)

    _materialize_embedded_profile_env(_CONFIG, llm_api_key="sk-current")

    assert stat.S_IMODE(profile_env.stat().st_mode) == 0o600
    assert "HINDSIGHT_API_LLM_API_KEY=sk-current\n" in profile_env.read_text(
        encoding="utf-8"
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not enforced on Windows")
def test_secret_file_removed_when_permission_validation_fails(monkeypatch):
    """If the post-write permission check cannot verify 0600, the plaintext
    key file must not be left behind."""
    import plugins.memory.hindsight as hs

    def _fail_validation(profile_env):
        raise PermissionError(f"not owner-only: {profile_env}")

    monkeypatch.setattr(hs, "_validate_profile_env_permissions", _fail_validation)

    with pytest.raises(PermissionError):
        _materialize_embedded_profile_env(_CONFIG, llm_api_key="sk-doomed")

    assert not _embedded_profile_env_path(_CONFIG).exists(), (
        "secret env file must be cleaned up when validation fails"
    )
