"""Invariants for ``hermes_cli.config_effective.load_user_config_effective`` — the one loader every
defaults-free config reader (gateway runtime, TUI gateway, cron, ``hermes send`` bridge, doctor,
bootstrap modules) goes through."""
import textwrap

import pytest


@pytest.fixture
def homes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("FIXTURE_USER_KEY", "user-secret")
    monkeypatch.setenv("FIXTURE_MANAGED_URL", "https://managed.example")
    _reset_caches()
    return home, managed


def _reset_caches():
    import hermes_cli.config as cfg
    from hermes_cli import config_effective, managed_scope

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    config_effective._EFFECTIVE_CACHE.clear()
    config_effective._LAST_GOOD_USER_RAW.clear()
    managed_scope.invalidate_managed_cache()


def _write(path, body):
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    _reset_caches()


USER_YAML = """
    model:
      name: user/model
      api_key: ${FIXTURE_USER_KEY}
    provider: custom
    display:
      skin: user-skin
    """
MANAGED_YAML = """
    model:
      base_url: ${FIXTURE_MANAGED_URL}
    display:
      skin: managed-skin
    """


def _legacy_gateway_pipeline(config_path):
    """The pre-unification gateway sequence (raw read → overlay → model-key canon → ${VAR} expansion)."""
    from hermes_cli import managed_scope
    from hermes_cli.config import _expand_env_vars, _normalize_root_model_keys, read_user_config_raw

    raw = managed_scope.apply_managed_overlay(read_user_config_raw(config_path))
    return _expand_env_vars(_normalize_root_model_keys(raw))


def test_effective_equals_legacy_gateway_pipeline_and_carries_no_defaults(homes):
    """Contract: the shared loader returns byte-for-byte what the gateway's hand-rolled pipeline did
    for a user file with a managed overlay and ``${VAR}`` refs on both layers — so per-message
    gateway config reads (and the system prompt built from them) do not change — while never
    merging DEFAULT_CONFIG (a missing key stays missing)."""
    from hermes_cli.config import DEFAULT_CONFIG
    from hermes_cli.config_effective import load_user_config_effective

    home, managed = homes
    _write(home / "config.yaml", USER_YAML)
    _write(managed / "config.yaml", MANAGED_YAML)

    effective = load_user_config_effective(home / "config.yaml")

    assert effective == _legacy_gateway_pipeline(home / "config.yaml")
    assert effective["model"] == {
        "default": "user/model", "provider": "custom", "api_key": "user-secret",
        "base_url": "https://managed.example"}
    assert effective["display"]["skin"] == "managed-skin"
    assert "provider" not in effective  # root key migrated under ``model`` (canonicalization applied)
    assert "agent" not in effective and "agent" in DEFAULT_CONFIG  # no DEFAULT_CONFIG merge


def test_broken_yaml_serves_last_good_and_fail_closed_raises(homes):
    """A torn mid-edit write must not silently drop user overrides: the fail-open path serves the last
    successfully parsed user file through the same pipeline; ``fail_closed`` surfaces the error to
    callers that keep their own last-good state."""
    from hermes_cli.config_effective import load_user_config_effective

    home, _ = homes
    _write(home / "config.yaml", USER_YAML)
    good = load_user_config_effective(home / "config.yaml")

    (home / "config.yaml").write_text("model: [unterminated", encoding="utf-8")
    _reset_caches_keep_last_good()

    assert load_user_config_effective(home / "config.yaml") == good
    with pytest.raises(Exception):
        load_user_config_effective(home / "config.yaml", fail_closed=True)


def _reset_caches_keep_last_good():
    import hermes_cli.config as cfg
    from hermes_cli import config_effective

    cfg._RAW_CONFIG_CACHE.clear()
    config_effective._EFFECTIVE_CACHE.clear()
