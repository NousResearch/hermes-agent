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


def test_existing_empty_managed_dir_is_successful_config_noop(tmp_path, monkeypatch):
    from hermes_cli import managed_scope

    _write_managed(tmp_path, monkeypatch)

    config, loaded = managed_scope.load_managed_config_with_status()

    assert config == {}
    assert loaded is True


@pytest.mark.parametrize(
    "body",
    ["[]\n", "- item\n", "scalar\n", "null\n", "key: [unterminated\n"],
    ids=["empty-list", "list", "scalar", "null", "malformed"],
)
def test_invalid_managed_config_reports_unsuccessful_load(
    tmp_path, monkeypatch, body
):
    from hermes_cli import managed_scope

    _write_managed(tmp_path, monkeypatch, config=body)

    config, loaded = managed_scope.load_managed_config_with_status()

    assert config == {}
    assert loaded is False








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




def test_managed_dir_env_scrubbed_by_default():
    """conftest must scrub HERMES_MANAGED_DIR so a dev-shell value can't leak in."""
    import os

    assert "HERMES_MANAGED_DIR" not in os.environ
