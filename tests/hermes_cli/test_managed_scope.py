"""Unit tests for hermes_cli.managed_scope (resolver + loaders + key helpers)."""
import logging
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

_SENTINEL = "hunter2-managed-sentinel"


@pytest.mark.parametrize(
    "config, where",
    [
        (f"model: x\nldap_password: {_SENTINEL}: oops\n", "line 2"),
        (f"token: {_SENTINEL}-old\ntoken: {_SENTINEL}\n", "line 2"),
        (f"timeout: !!timestamp {_SENTINEL}\n", "line 1"),
        (f"port: !!int {_SENTINEL}\n", None),
    ],
    ids=["secret-on-bad-line", "duplicate-key", "bad-timestamp", "bad-int"],
)
def test_malformed_managed_config_warns_without_its_contents(tmp_path, monkeypatch, caplog, config, where):
    from hermes_cli import managed_scope

    managed = _write_managed(tmp_path, monkeypatch, config=config)
    with caplog.at_level(logging.WARNING, logger="hermes_cli.managed_scope"):
        assert managed_scope.load_managed_config() == {}

    warnings = [r.getMessage() for r in caplog.records if r.name == "hermes_cli.managed_scope"]
    assert len(warnings) == 1
    assert str(managed / "config.yaml") in warnings[0]
    assert "IGNORING" in warnings[0]
    if where:
        assert where in warnings[0]
    assert _SENTINEL not in warnings[0]
