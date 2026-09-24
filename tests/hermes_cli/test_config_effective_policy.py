"""Strict authorization reads retain the effective loader's ordinary semantics."""
import pytest
import yaml
import textwrap
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('effective_fixture', Path(__file__).with_name('test_config_effective.py'))
assert spec is not None and spec.loader is not None
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)
homes, USER_YAML, MANAGED_YAML = f.homes, f.USER_YAML, f.MANAGED_YAML


@pytest.mark.parametrize('broken', ['kanban: [unterminated', '- invalid', 'false'])
def test_strict_policy_does_not_trust_fail_open_managed_cache(homes, broken):
    from hermes_cli.config_effective import load_user_config_effective
    home, managed = homes
    (home / 'config.yaml').write_text('kanban: {decision_grants: [user]}')
    overlay = managed / 'config.yaml'
    overlay.write_text('kanban: {decision_grants: []}')
    assert load_user_config_effective(fail_closed=True)['kanban']['decision_grants'] == []
    overlay.write_text(broken)
    assert load_user_config_effective()['kanban']['decision_grants'] == ['user']
    with pytest.raises((yaml.YAMLError, ValueError)):
        load_user_config_effective(fail_closed=True, side_effect_free=True)
    overlay.write_text('kanban: {decision_grants: [repaired]}')
    assert load_user_config_effective(fail_closed=True)['kanban']['decision_grants'] == ['repaired']


def test_read_opt_out_preserves_overlay_env_cache_and_deferred_backup(homes, monkeypatch):
    from hermes_cli.config_effective import load_user_config_effective
    home, managed = homes
    path = home / 'config.yaml'
    path.write_text(USER_YAML)
    overlay = managed / 'config.yaml'
    overlay.write_text(MANAGED_YAML)
    first = load_user_config_effective(fail_closed=True, side_effect_free=True)
    assert first['model']['base_url'] == 'https://managed.example'
    assert not (home / 'backups').exists()
    monkeypatch.setenv('FIXTURE_USER_KEY', 'changed-synthetic')
    assert load_user_config_effective(fail_closed=True, side_effect_free=True)['model']['api_key'] == 'changed-synthetic'
    # Force an effective miss while preserving the shared raw cache.
    overlay.write_text('display: {skin: changed}')
    load_user_config_effective()
    backups = home / 'backups/config'
    assert list(backups.glob('config.yaml.good.*'))
    for backup in backups.iterdir():
        backup.rename(backup.with_name(backup.name + '.retained'))
    path.write_text(textwrap.dedent(USER_YAML) + '\nnew_valid_revision: true\n')
    before = {p.name: p.read_bytes() for p in backups.iterdir()}
    load_user_config_effective(fail_closed=True, side_effect_free=True)
    assert {p.name: p.read_bytes() for p in backups.iterdir()} == before
    load_user_config_effective()
    assert any(p.name not in before for p in backups.iterdir())


@pytest.mark.parametrize('warm_strict', [False, True])
def test_strict_null_overlay_does_not_share_ordinary_cache(homes, warm_strict):
    from hermes_cli.config_effective import load_user_config_effective
    home, managed = homes
    user = {'authorization': {'nested': {'grants': ['user']}}, 'terminal': {'cwd': 'user'}}
    (home / 'config.yaml').write_text(yaml.safe_dump(user))
    (managed / 'config.yaml').write_text('authorization: {nested: null}\nterminal: null\n')
    strict = {'authorization': {'nested': None}, 'terminal': None}
    # Both cache orders, repeated hits, and deepcopy isolation. No policy names in the merge.
    for mode in (warm_strict, not warm_strict, warm_strict, not warm_strict):
        result = load_user_config_effective(fail_closed=mode, side_effect_free=True)
        assert result == (strict if mode else user)
        result['authorization'] = 'caller mutation'
    assert not (home / 'backups').exists()
    assert load_user_config_effective() == user
    assert list((home / 'backups/config').glob('config.yaml.good.*'))


@pytest.mark.parametrize('overlay, expected', [
    ('{}', ['user']),
    ('authorization: {}', ['user']),
    ('authorization: {grants: []}', []),
    ('authorization: {grants: [managed]}', ['managed']),
    ('authorization: {grants: null}', None),
    ('authorization: {grants: invalid}', 'invalid'),
    ('authorization: {grants: {invalid: true}}', {'invalid': True}),
])
def test_strict_overlay_preserves_absence_and_present_leaf_values(homes, overlay, expected):
    from hermes_cli.config_effective import load_user_config_effective
    home, managed = homes
    (home / 'config.yaml').write_text('authorization: {grants: [user]}')
    assert load_user_config_effective(fail_closed=True)['authorization']['grants'] == ['user']
    (managed / 'config.yaml').write_text(overlay)
    load_user_config_effective()  # warm the fail-open cache without resetting either cache
    assert load_user_config_effective(fail_closed=True)['authorization']['grants'] == expected


_ROOT_CASES = [
    pytest.param("missing-dir", None, True, {}, {}, id="missing-managed-dir"),
    pytest.param("missing-file", None, False, {}, {}, id="missing-managed-file"),
    pytest.param("empty-map", "{}\n", False, {}, {}, id="explicit-empty-map"),
    pytest.param("empty-section", "kanban: {}\n", False, {"kanban": {}}, {"kanban": {}}, id="empty-section-map"),
    pytest.param(
        "valid-map",
        "kanban:\n  decision_grants: []\n",
        False,
        {"kanban": {"decision_grants": []}},
        {"kanban": {"decision_grants": []}},
        id="valid-map",
    ),
    pytest.param("null", "null\n", False, {}, ValueError, id="null"),
    pytest.param("tilde-null", "~\n", False, {}, ValueError, id="tilde-null"),
    pytest.param("tagged-null", "!!null 'null'\n", False, {}, ValueError, id="tagged-null"),
    pytest.param("document-end-null", "---\n", False, {}, ValueError, id="document-end-null"),
    pytest.param("blank-null", "\n# comment only\n", False, {}, ValueError, id="blank-comment-only"),
    pytest.param("false", "false\n", False, {}, ValueError, id="scalar-false"),
    pytest.param("zero", "0\n", False, {}, ValueError, id="scalar-zero"),
    pytest.param("string", "managed\n", False, {}, ValueError, id="scalar-string"),
    pytest.param("list", "[]\n", False, {}, ValueError, id="root-list"),
]


@pytest.mark.parametrize("warm_ordinary", [False, True], ids=["cold", "ordinary-warm"])
@pytest.mark.parametrize("label, body, missing_dir, ordinary_expected, strict_expected", _ROOT_CASES)
def test_managed_root_type_matrix(
    homes, monkeypatch, label, body, missing_dir, ordinary_expected, strict_expected, warm_ordinary
):
    from hermes_cli import managed_scope

    _home, managed = homes
    if missing_dir:
        monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed / "does-not-exist"))
    elif body is not None:
        (managed / "config.yaml").write_text(body, encoding="utf-8")
    managed_scope.invalidate_managed_cache()

    if warm_ordinary:
        assert managed_scope.load_managed_config() == ordinary_expected
    if strict_expected is ValueError:
        with pytest.raises(ValueError, match="must be a mapping"):
            managed_scope.load_managed_config(fail_closed=True)
    else:
        assert managed_scope.load_managed_config(fail_closed=True) == strict_expected
    # The opposite loader must retain its normal semantics after the strict/cached read.
    assert managed_scope.load_managed_config() == ordinary_expected
    if strict_expected is ValueError:
        with pytest.raises(ValueError, match="must be a mapping"):
            managed_scope.load_managed_config(fail_closed=True)
    else:
        assert managed_scope.load_managed_config(fail_closed=True) == strict_expected


@pytest.mark.parametrize("warm_ordinary", [False, True], ids=["cold", "ordinary-warm"])
@pytest.mark.parametrize("label, body, missing_dir, ordinary_expected, strict_expected", _ROOT_CASES)
def test_effective_root_type_matrix_preserves_strict_root_errors(
    homes, monkeypatch, label, body, missing_dir, ordinary_expected, strict_expected, warm_ordinary
):
    from hermes_cli.config_effective import load_user_config_effective

    home, managed = homes
    user = {"kanban": {"decision_grants": ["user"]}, "terminal": {"cwd": "user"}}
    (home / "config.yaml").write_text(yaml.safe_dump(user), encoding="utf-8")
    if missing_dir:
        monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed / "does-not-exist"))
    elif body is not None:
        (managed / "config.yaml").write_text(body, encoding="utf-8")

    ordinary_no_effective_override = ordinary_expected in ({}, {"kanban": {}})
    ordinary_expected_effective = (
        user if ordinary_no_effective_override else {**user, **ordinary_expected}
    )
    no_effective_override = strict_expected in ({}, {"kanban": {}})
    strict_expected_effective = (
        None if strict_expected is ValueError else (
            user if not strict_expected or no_effective_override else {**user, **strict_expected}
        )
    )
    if warm_ordinary:
        assert load_user_config_effective(side_effect_free=True) == ordinary_expected_effective
    if strict_expected is ValueError:
        with pytest.raises(ValueError, match="must be a mapping"):
            load_user_config_effective(fail_closed=True, side_effect_free=True)
    else:
        assert load_user_config_effective(fail_closed=True, side_effect_free=True) == strict_expected_effective
    # Strict validation must not be bypassed by an ordinary effective-cache warmup, and
    # absence/empty mappings must continue to restore the documented user layer.
    assert load_user_config_effective(side_effect_free=True) == ordinary_expected_effective
    if strict_expected is ValueError:
        with pytest.raises(ValueError, match="must be a mapping"):
            load_user_config_effective(fail_closed=True, side_effect_free=True)
    else:
        assert load_user_config_effective(fail_closed=True, side_effect_free=True) == strict_expected_effective


def test_deleted_managed_file_transitions_to_documented_absence(homes):
    from hermes_cli.config_effective import load_user_config_effective

    home, managed = homes
    (home / "config.yaml").write_text("kanban: {decision_grants: [user]}\n", encoding="utf-8")
    path = managed / "config.yaml"
    path.write_text("kanban: {decision_grants: []}\n", encoding="utf-8")
    assert load_user_config_effective(fail_closed=True, side_effect_free=True)["kanban"]["decision_grants"] == []
    path.unlink()
    assert load_user_config_effective()["kanban"]["decision_grants"] == ["user"]
    assert load_user_config_effective(fail_closed=True, side_effect_free=True)["kanban"]["decision_grants"] == ["user"]
