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
    config_effective._LAST_GOOD_EFFECTIVE.clear()
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


def test_effective_named_profile_inherits_default_raw_config(homes):
    """Defaults-free readers see the same named-profile inheritance as load_config()."""
    from hermes_cli.config_effective import load_user_config_effective

    home, _ = homes
    profile = home / "profiles" / "work"
    profile.mkdir(parents=True)
    _write(home / "config.yaml", """
        model:
          default: default/model
        display:
          skin: default-skin
        """)
    _write(profile / "config.yaml", """
        model:
          provider: work-provider
        """)

    effective = load_user_config_effective(profile / "config.yaml")

    assert effective == {
        "model": {"default": "default/model", "provider": "work-provider"},
        "display": {"skin": "default-skin"},
    }


def test_effective_fail_closed_rejects_a_malformed_inheritance_base(homes):
    """Strict readers must not treat a corrupt base as an empty inheritance layer."""
    from hermes_cli.config_effective import load_user_config_effective

    home, _ = homes
    profile = home / "profiles" / "work"
    profile.mkdir(parents=True)
    _write(home / "config.yaml", "model:\n  provider: shared\n")
    _write(profile / "config.yaml", "display:\n  skin: work\n")
    assert load_user_config_effective(profile / "config.yaml")["model"]["provider"] == "shared"

    _write(home / "config.yaml", "model: [unterminated\n")
    with pytest.raises(yaml.YAMLError):
        load_user_config_effective(profile / "config.yaml", fail_closed=True)


def test_effective_recovers_a_broken_inheritance_base_from_its_own_backup(homes):
    """A cold effective reader reconstructs inherited policy from the base's good backup."""
    from hermes_cli import config as cfg
    from hermes_cli import config_effective
    from hermes_cli.config_backups import backup_config
    from hermes_cli.config_read_errors import FailedConfigRead
    from hermes_cli.config_effective import load_user_config_effective

    home, _ = homes
    profile = home / "profiles" / "work"
    profile.mkdir(parents=True)
    _write(home / "config.yaml", "approvals:\n  deny:\n    - rm -rf *\n")
    backup_config(home / "config.yaml", "good")
    _write(profile / "config.yaml", "display:\n  skin: work\n")
    _write(home / "config.yaml", "approvals: [unterminated\n")
    cfg._LOAD_CONFIG_CACHE.clear()
    config_effective._EFFECTIVE_CACHE.clear()
    config_effective._LAST_GOOD_EFFECTIVE.clear()

    recovered = load_user_config_effective(profile / "config.yaml")

    assert isinstance(recovered, FailedConfigRead)
    assert recovered["approvals"]["deny"] == ["rm -rf *"]
    assert recovered["display"]["skin"] == "work"


def test_effective_broken_inheritance_base_without_a_backup_returns_a_marked_child_fallback(homes):
    """Non-strict effective reads remain safe before a base backup has been created."""
    from hermes_cli import config as cfg
    from hermes_cli.config_read_errors import FailedConfigRead
    from hermes_cli.config_effective import load_user_config_effective

    home, _ = homes
    profile = home / "profiles" / "work"
    profile.mkdir(parents=True)
    _write(home / "config.yaml", "model: [unterminated\n")
    _write(profile / "config.yaml", "display:\n  skin: work\n")
    cfg._LOAD_CONFIG_CACHE.clear()

    recovered = load_user_config_effective(profile / "config.yaml")

    assert isinstance(recovered, FailedConfigRead)
    assert recovered["display"]["skin"] == "work"


def test_effective_child_moa_presets_override_the_parent_mapping(homes):
    """Defaults-free readers share load_config's nearest-explicit MoA preset contract."""
    from hermes_cli.config_effective import load_user_config_effective

    home, _ = homes
    profile = home / "profiles" / "work"
    profile.mkdir(parents=True)
    _write(home / "config.yaml", "moa:\n  presets:\n    council: {}\n")
    _write(profile / "config.yaml", "moa:\n  presets:\n    work: {}\n")

    effective = load_user_config_effective(profile / "config.yaml")

    assert set(effective["moa"]["presets"]) == {"work"}


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
