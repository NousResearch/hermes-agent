"""Invariants for ``hermes_cli.config_effective.load_user_config_effective`` — the one loader every
defaults-free config reader (gateway runtime, TUI gateway, cron, ``hermes send`` bridge, doctor,
bootstrap modules) goes through."""
import textwrap

import pytest

import hermes_yaml as yaml


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


def test_effective_is_user_plus_managed_plus_env_with_no_defaults(homes):
    """Contract as a fixture: given user config.yaml X, managed overlay Y and env Z, the effective
    dict is exactly this literal — ``${VAR}`` expanded on both layers, managed keys winning,
    root ``provider`` migrated under ``model``, and no DEFAULT_CONFIG key introduced (a missing
    key stays missing). Per-message gateway reads (and the system prompt built from them) are
    pinned by this shape, not by re-running the implementation's primitives."""
    from hermes_cli.config_effective import load_user_config_effective

    home, managed = homes
    _write(home / "config.yaml", USER_YAML)
    _write(managed / "config.yaml", MANAGED_YAML)

    effective = load_user_config_effective(home / "config.yaml")

    assert effective == {
        "model": {
            "default": "user/model",
            "provider": "custom",
            "api_key": "user-secret",
            "base_url": "https://managed.example",
        },
        "display": {"skin": "managed-skin"},
    }


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
    with pytest.raises(yaml.YAMLError):  # the type _refresh_fallback_model's own last-good path keys on
        load_user_config_effective(home / "config.yaml", fail_closed=True)


def test_good_backup_is_written_only_for_the_active_home(homes, tmp_path):
    """Reading ANOTHER profile's config (doctor, TUI cwd lookup) is a read: it must not create
    ``backups/config/`` inside that profile. The active home keeps the last-good copy."""
    from hermes_cli.config_effective import load_user_config_effective

    home, _ = homes
    other = tmp_path / "other-profile"
    other.mkdir()
    _write(home / "config.yaml", USER_YAML)
    _write(other / "config.yaml", USER_YAML)

    load_user_config_effective(other / "config.yaml")
    load_user_config_effective(home / "config.yaml")

    assert not (other / "backups").exists()
    assert list((home / "backups" / "config").glob("config.yaml.good.*"))


def _reset_caches_keep_last_good():
    import hermes_cli.config as cfg
    from hermes_cli import config_effective

    cfg._RAW_CONFIG_CACHE.clear()
    config_effective._EFFECTIVE_CACHE.clear()


def _good_backups(home):
    return sorted((home / "backups" / "config").glob("config.yaml.good.*"))


def test_non_mapping_root_is_invalid_not_empty_config(homes):
    """A syntactically valid list/scalar root is an INVALID user config, not an empty one. Serving it
    through the success branch would record the bad bytes as last-known-good and back them up as
    ``good``, so a mid-edit ``[\"unfinished edit\"]`` would silently drop every override instead of
    being reported. Fail-open keeps the last-known-good mapping; ``fail_closed`` raises the mapping
    error ``require_readable_config_before_write`` already uses for the same document."""
    from hermes_cli.config_effective import load_user_config_effective

    home, _ = homes
    config_path = home / "config.yaml"
    _write(config_path, USER_YAML)
    good = load_user_config_effective(config_path)
    backups_before = _good_backups(home)
    assert good["display"] == {"skin": "user-skin"}

    config_path.write_text('["unfinished edit"]\n', encoding="utf-8")
    _reset_caches_keep_last_good()

    assert load_user_config_effective(config_path) == good
    with pytest.raises(TypeError, match="top-level YAML must be a mapping, got list"):
        load_user_config_effective(config_path, fail_closed=True)

    # The invalid root never became a ``good`` copy (only the corrupt-file snapshot the parse-failure
    # path keeps), and every surviving good copy is still a mapping.
    assert _good_backups(home) == backups_before
    for backup in _good_backups(home):
        assert isinstance(yaml.safe_load(backup.read_text(encoding="utf-8")), dict)

    # A cold process (no in-process last-known-good) recovers from the newest ``good`` backup.
    _reset_caches()
    assert load_user_config_effective(config_path) == good


def test_empty_and_null_roots_stay_the_documented_empty_layer(homes):
    """The invalid-root check must not swallow the documented empty cases: an absent file, an empty
    file and an explicit ``null`` document are the empty user layer, not a config error — including
    under ``fail_closed``."""
    from hermes_cli.config_effective import load_user_config_effective

    home, _ = homes
    config_path = home / "config.yaml"

    for body in ("", "null\n", "# only a comment\n"):
        _write(config_path, body)
        assert load_user_config_effective(config_path) == {}
        assert load_user_config_effective(config_path, fail_closed=True) == {}

    config_path.unlink()
    _reset_caches()
    assert load_user_config_effective(config_path) == {}

