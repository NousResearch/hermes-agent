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


@pytest.mark.parametrize("warm", [False, True])
def test_strict_overlay_preserves_explicit_null(policy_homes, warm):
    from hermes_cli.config_effective import load_user_config_effective
    home, managed = policy_homes
    (home / "config.yaml").write_text("authorization: {nested: {grants: [user]}}\n")
    (managed / "config.yaml").write_text("authorization: {nested: null}\n")
    if warm:
        assert load_user_config_effective()["authorization"]["nested"] == {"grants": ["user"]}
    assert load_user_config_effective(fail_closed=True, strict_policy=True)["authorization"]["nested"] is None
    assert load_user_config_effective()["authorization"]["nested"] == {"grants": ["user"]}
    assert load_user_config_effective(fail_closed=True, strict_policy=True)["authorization"]["nested"] is None


@pytest.mark.parametrize(
    ("managed_body", "error"),
    [("kanban: [unterminated", yaml.YAMLError), ("- kanban\n", ValueError)],
)
def test_strict_managed_root_never_recovers_user_grant(policy_homes, managed_body, error):
    from hermes_cli.config_effective import load_user_config_effective
    home, managed = policy_homes
    (home / "config.yaml").write_text("kanban: {dispatch_profiles: [default]}\n")
    # Absent, empty and null roots carry no policy, even for strict readers.
    assert load_user_config_effective(fail_closed=True, strict_policy=True)["kanban"] == {"dispatch_profiles": ["default"]}
    for empty_body in ("", "null\n"):
        (managed / "config.yaml").write_text(empty_body)
        assert load_user_config_effective(fail_closed=True, strict_policy=True)["kanban"] == {"dispatch_profiles": ["default"]}
    (managed / "config.yaml").write_text(managed_body)
    # A fail-open cache must never hide a malformed root from a later strict read.
    assert load_user_config_effective()["kanban"] == {"dispatch_profiles": ["default"]}
    with pytest.raises(error):
        load_user_config_effective(fail_closed=True, strict_policy=True)


def test_strict_read_skips_fail_open_managed_snapshot_read(policy_homes, caplog):
    """The env-ref snapshot read honours ``strict_policy``: a strict read of a broken managed file
    raises without first logging the fail-open "IGNORING" warning; ordinary reads still warn."""
    from hermes_cli.config_effective import load_user_config_effective
    home, managed = policy_homes
    (home / "config.yaml").write_text("kanban: {dispatch_profiles: [default]}\n")
    (managed / "config.yaml").write_text("kanban: [unterminated")
    caplog.set_level(logging.WARNING, logger="hermes_cli.managed_scope")

    def ignoring():
        return [r for r in caplog.records if "IGNORING this managed file" in r.getMessage()]

    with pytest.raises(yaml.YAMLError):
        load_user_config_effective(fail_closed=True, strict_policy=True)
    assert ignoring() == []
    caplog.clear()
    assert load_user_config_effective()["kanban"] == {"dispatch_profiles": ["default"]}
    assert ignoring()  # the ordinary read still fails open, loudly
