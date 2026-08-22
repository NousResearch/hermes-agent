"""Unit tests for hermes_cli.managed_scope (resolver + loaders + key helpers)."""

import textwrap

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


def test_invalid_skills_section_is_ignored_and_warned_once(
    tmp_path,
    monkeypatch,
    caplog,
):
    """Invalid managed skills policy is normalized once at the parse boundary."""
    from hermes_cli import managed_scope

    _write_managed(
        tmp_path,
        monkeypatch,
        config="display:\n  skin: managed\nskills: []\n",
    )

    with caplog.at_level("WARNING"):
        first = managed_scope.load_managed_config()
        second = managed_scope.load_managed_config()

    warnings = [
        record
        for record in caplog.records
        if "skills must be a mapping" in record.getMessage()
    ]
    assert first == {"display": {"skin": "managed"}}
    assert second == {"display": {"skin": "managed"}}
    assert managed_scope.is_key_managed("display.skin") is True
    assert managed_scope.is_key_managed("skills") is False
    assert len(warnings) == 1


def test_managed_dir_env_scrubbed_by_default():
    """conftest must scrub HERMES_MANAGED_DIR so a dev-shell value can't leak in."""
    import os

    assert "HERMES_MANAGED_DIR" not in os.environ
