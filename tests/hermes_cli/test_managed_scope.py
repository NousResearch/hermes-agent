"""Unit tests for hermes_cli.managed_scope (resolver + loaders + key helpers)."""
import textwrap
import os


import pytest


# ── Directory resolver ───────────────────────────────────────────────────────






# ── Loaders + key helpers ────────────────────────────────────────────────────


def _write_managed(tmp_path, monkeypatch, *, config=None, env=None):
    from hermes_cli import managed_scope

    managed = tmp_path / "managed"
    managed.mkdir(exist_ok=True)
    if config is not None:
        (managed / "config.yaml").write_text(textwrap.dedent(config), encoding="utf-8")
    if env is not None:
        (managed / ".env").write_text(textwrap.dedent(env), encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    managed_scope.invalidate_managed_cache()
    return managed








def test_load_managed_env_and_is_env_managed(tmp_path, monkeypatch):
    from hermes_cli import managed_scope

    _write_managed(
        tmp_path, monkeypatch, env="OPENAI_API_BASE=https://org.example/v1\n"
    )
    assert managed_scope.load_managed_env() == {
        "OPENAI_API_BASE": "https://org.example/v1"
    }
    assert managed_scope.is_env_managed("OPENAI_API_BASE") is True
    assert managed_scope.is_env_managed("OTHER") is False

def test_same_metadata_rewrite_invalidates_managed_cache(tmp_path, monkeypatch):
    from hermes_cli import managed_scope

    managed = _write_managed(
        tmp_path,
        monkeypatch,
        config="delegation:\n  routes:\n    alpha: {}\n",
    )
    path = managed / "config.yaml"
    original = path.stat()
    assert set(managed_scope.load_managed_config()["delegation"]["routes"]) == {"alpha"}

    replacement = "delegation:\n  routes:\n    bravo: {}\n"
    assert len(replacement.encode()) == original.st_size
    path.write_text(replacement, encoding="utf-8")
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    rewritten = path.stat()
    assert (rewritten.st_mtime_ns, rewritten.st_size) == (
        original.st_mtime_ns,
        original.st_size,
    )
    assert set(managed_scope.load_managed_config()["delegation"]["routes"]) == {"bravo"}




def test_managed_dir_env_scrubbed_by_default():
    """conftest must scrub HERMES_MANAGED_DIR so a dev-shell value can't leak in."""
    import os

    assert "HERMES_MANAGED_DIR" not in os.environ
