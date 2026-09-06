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
from plugins.memory.hindsight.embedded import (
    _build_embedded_profile_env,
    _load_simple_env,
    _may_rewrite_profile_env,
    _on_disk_llm_api_key,
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
    import plugins.memory.hindsight.embedded as hs_embedded

    def _fail_validation(profile_env):
        raise PermissionError(f"not owner-only: {profile_env}")

    monkeypatch.setattr(hs_embedded, "_validate_profile_env_permissions", _fail_validation)

    with pytest.raises(PermissionError):
        _materialize_embedded_profile_env(_CONFIG, llm_api_key="sk-doomed")

    assert not _embedded_profile_env_path(_CONFIG).exists(), (
        "secret env file must be cleaned up when validation fails"
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not enforced on Windows")
def test_scopeless_worker_reuses_on_disk_key(monkeypatch):
    """Durability core: with no secret scope, the key resolves from disk.

    The daemon-start worker usually has no scope; without the disk fallback
    the client would be built keyless and the upstream manager merge
    (ensure_running -> _register_profile -> create_profile rewrite) would
    overwrite the file's good key with emptiness.
    """
    from agent import secret_scope

    profile_env = _embedded_profile_env_path(_CONFIG)
    profile_env.parent.mkdir(parents=True)
    profile_env.write_text(
        "HINDSIGHT_API_LLM_PROVIDER=openai\n"
        "HINDSIGHT_API_LLM_API_KEY=sk-test-live-key\n"
        "HINDSIGHT_API_LLM_MODEL=gpt-4o-mini\n"
        "HINDSIGHT_API_LOG_LEVEL=info\n",
        encoding="utf-8",
    )
    os.chmod(profile_env, 0o600)

    token = secret_scope.set_secret_scope(None)
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    monkeypatch.delenv("HINDSIGHT_API_LLM_API_KEY", raising=False)
    try:
        from plugins.memory.hindsight.embedded import _embedded_llm_api_key

        assert _embedded_llm_api_key(_CONFIG) == "sk-test-live-key"
        # ...so the build carries a key and the rewrite gate passes.
        assert _build_embedded_profile_env(_CONFIG)["HINDSIGHT_API_LLM_API_KEY"] == "sk-test-live-key"
        assert _may_rewrite_profile_env(_CONFIG) is True
    finally:
        secret_scope.reset_secret_scope(token)


def test_gate_refuses_keyless_build_against_keyed_disk(monkeypatch):
    """_may_rewrite_profile_env False branch: keyless build + keyed disk."""
    import plugins.memory.hindsight.embedded as hs_embedded

    monkeypatch.setattr(hs_embedded, "_build_embedded_profile_env",
                        lambda config, **kw: {"HINDSIGHT_API_LLM_API_KEY": ""})
    monkeypatch.setattr(hs_embedded, "_on_disk_llm_api_key", lambda config: "sk-test-live-key")
    assert hs_embedded._may_rewrite_profile_env(_CONFIG) is False


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits are not enforced on Windows")
def test_on_disk_key_survives_empty_build(monkeypatch):
    """Fail-closed core: a keyed file survives a key-destroying rewrite attempt.

    Uses the worker's real compare inputs — on-disk file vs a keyless build
    (llm_api_key="" models the pre-fix scopeless build shape) — and the
    gated worker body (rewrite only when the gate passes).
    """
    from agent import secret_scope

    profile_env = _embedded_profile_env_path(_CONFIG)
    profile_env.parent.mkdir(parents=True)
    profile_env.write_text(
        "HINDSIGHT_API_LLM_PROVIDER=openai\n"
        "HINDSIGHT_API_LLM_API_KEY=sk-test-live-key\n"
        "HINDSIGHT_API_LLM_MODEL=gpt-4o-mini\n"
        "HINDSIGHT_API_LOG_LEVEL=info\n",
        encoding="utf-8",
    )
    os.chmod(profile_env, 0o600)

    token = secret_scope.set_secret_scope(None)
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    monkeypatch.delenv("HINDSIGHT_API_LLM_API_KEY", raising=False)
    try:
        # The worker's real compare inputs: on-disk file vs a keyless build
        # (llm_api_key="" models the pre-fix scopeless build shape).
        keyless = _build_embedded_profile_env(_CONFIG, llm_api_key="")
        assert keyless["HINDSIGHT_API_LLM_API_KEY"] == ""
        assert _load_simple_env(profile_env) != keyless

        # The gate predicate on those same inputs: keyless build + keyed disk
        # -> refuse. Inline the predicate (same two reads the gate uses) so
        # the test pins behavior, not the helper's name.
        build_has_key = bool(keyless.get("HINDSIGHT_API_LLM_API_KEY"))
        disk_has_key = bool(_on_disk_llm_api_key(_CONFIG))
        assert not build_has_key and disk_has_key

        before = profile_env.read_bytes()
        # Gated worker body: rewrite only when the gate passes.
        if build_has_key or not disk_has_key:
            _materialize_embedded_profile_env(_CONFIG, llm_api_key="")
        assert profile_env.read_bytes() == before
        assert stat.S_IMODE(profile_env.stat().st_mode) == 0o600
    finally:
        secret_scope.reset_secret_scope(token)


def test_rewrite_allowed_when_build_carries_key():
    """A build WITH key material (rotation, model drift) must still rewrite."""
    profile_env = _embedded_profile_env_path(_CONFIG)
    profile_env.parent.mkdir(parents=True)
    profile_env.write_text("HINDSIGHT_API_LLM_API_KEY=sk-old\n", encoding="utf-8")

    cfg = dict(_CONFIG, llm_api_key="sk-new")
    assert _may_rewrite_profile_env(cfg) is True
    _materialize_embedded_profile_env(cfg)
    assert "HINDSIGHT_API_LLM_API_KEY=sk-new\n" in profile_env.read_text(encoding="utf-8")


def test_api_prefixed_vault_name_resolves(monkeypatch):
    """The vault item is HINDSIGHT_API_LLM_API_KEY; the wizard name alone misses."""
    from agent import secret_scope
    import plugins.memory.hindsight.embedded as hs_embedded

    token = secret_scope.set_secret_scope({"HINDSIGHT_API_LLM_API_KEY": "sk-test-live-key"})
    monkeypatch.delenv("HINDSIGHT_LLM_API_KEY", raising=False)
    monkeypatch.delenv("HINDSIGHT_API_LLM_API_KEY", raising=False)
    try:
        assert hs_embedded._embedded_llm_api_key(_CONFIG) == "sk-test-live-key"
    finally:
        secret_scope.reset_secret_scope(token)


def test_rewrite_allowed_when_no_key_anywhere(monkeypatch):
    """Keyless providers (ollama) must not brick: empty build + empty disk rewrites."""
    from agent import secret_scope

    token = secret_scope.set_secret_scope(None)
    monkeypatch.delenv("HINDSIGHT_API_LLM_API_KEY", raising=False)
    try:
        cfg = dict(_CONFIG, llm_provider="ollama", llm_model="qwen3:8b")
        # No key on disk either -> gate passes, non-key drift still converges.
        assert _on_disk_llm_api_key(cfg) == ""
        assert _may_rewrite_profile_env(cfg) is True
    finally:
        secret_scope.reset_secret_scope(token)
