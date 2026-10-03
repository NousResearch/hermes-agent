import logging

import pytest
import hermes_yaml as yaml


@pytest.fixture
def policy_homes(tmp_path, monkeypatch):
    home, managed = tmp_path / "home", tmp_path / "managed"
    for path in (home, managed):
        path.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    import hermes_cli.config as config
    from hermes_cli import config_effective, managed_scope
    for cache in (config._LOAD_CONFIG_CACHE, config._RAW_CONFIG_CACHE,
                  config_effective._EFFECTIVE_CACHE, config_effective._LAST_GOOD_USER_RAW):
        cache.clear()
    managed_scope.invalidate_managed_cache()
    return home, managed


@pytest.mark.parametrize(("strict", "expected"), [(False, {"grants": ["user"]}), (True, None)])
def test_strict_overlay_preserves_explicit_null(policy_homes, strict, expected):
    from hermes_cli.config_effective import load_user_config_effective
    home, managed = policy_homes
    (home / "config.yaml").write_text("authorization: {nested: {grants: [user]}}\n")
    (managed / "config.yaml").write_text("authorization: {nested: null}\n")
    result = load_user_config_effective(fail_closed=strict)
    assert result["authorization"]["nested"] == expected


@pytest.mark.parametrize(
    ("managed_body", "error"),
    [("kanban: [unterminated", yaml.YAMLError), ("- kanban\n", ValueError)],
)
def test_strict_managed_root_never_recovers_user_grant(policy_homes, managed_body, error):
    from hermes_cli.config_effective import load_user_config_effective
    home, managed = policy_homes
    (home / "config.yaml").write_text("kanban: {dispatch_profiles: [default]}\n")
    (managed / "config.yaml").write_text(managed_body)
    with pytest.raises(error):
        load_user_config_effective(fail_closed=True)


@pytest.mark.parametrize("managed_body", ["", "null\n"])
def test_strict_read_treats_empty_managed_root_as_no_policy(policy_homes, managed_body):
    """An empty or ``null`` managed file carries no policy, like an absent one."""
    from hermes_cli.config_effective import load_user_config_effective
    home, managed = policy_homes
    (home / "config.yaml").write_text("kanban: {dispatch_profiles: [default]}\n")
    (managed / "config.yaml").write_text(managed_body)
    assert load_user_config_effective(fail_closed=True)["kanban"] == {"dispatch_profiles": ["default"]}


def test_strict_read_skips_fail_open_managed_snapshot_read(policy_homes, caplog):
    """The env-ref snapshot read honours ``fail_closed``: a strict read of a broken managed file
    raises without first logging the fail-open "IGNORING" warning; ordinary reads still warn."""
    from hermes_cli.config_effective import load_user_config_effective
    home, managed = policy_homes
    (home / "config.yaml").write_text("kanban: {dispatch_profiles: [default]}\n")
    (managed / "config.yaml").write_text("kanban: [unterminated")
    caplog.set_level(logging.WARNING, logger="hermes_cli.managed_scope")

    def ignoring():
        return [r for r in caplog.records if "IGNORING this managed file" in r.getMessage()]

    with pytest.raises(yaml.YAMLError):
        load_user_config_effective(fail_closed=True)
    assert ignoring() == []
    caplog.clear()
    assert load_user_config_effective()["kanban"] == {"dispatch_profiles": ["default"]}
    assert ignoring()  # the ordinary read still fails open, loudly


def test_strict_null_section_denies_fallback_chain_without_crashing(policy_homes):
    """A managed ``null`` over a whole section is a denial for strict readers, and the strict
    fallback-chain consumer reads it as an empty chain; ordinary reads keep the user section (#58277)."""
    from hermes_cli.config_effective import load_user_config_effective
    from hermes_cli.fallback_config import get_fallback_chain
    home, managed = policy_homes
    (home / "config.yaml").write_text("fallback_model: {provider: openrouter, model: user/fallback}\n")
    (managed / "config.yaml").write_text("fallback_model:\n")
    strict = load_user_config_effective(fail_closed=True)
    assert strict["fallback_model"] is None and get_fallback_chain(strict) == []
    ordinary = load_user_config_effective()
    assert [e["model"] for e in get_fallback_chain(ordinary)] == ["user/fallback"]
