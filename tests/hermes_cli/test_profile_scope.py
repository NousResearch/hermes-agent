
import pytest


def _fixture_profiles(monkeypatch, tmp_path, scope_text=None):
    import hermes_cli.profiles as profiles

    root = tmp_path / "root"
    root.mkdir()
    config = "gateway:\n"
    if scope_text is not None:
        config += f"  profile_scope: {scope_text}\n"
    (root / "config.yaml").write_text(config, encoding="utf-8")
    default = root / "default"
    named = [root / name for name in ("main", "staging")]
    default.mkdir()
    for home in named:
        home.mkdir()
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default)
    monkeypatch.setattr(profiles, "get_active_profile_name", lambda: "default")
    monkeypatch.setattr(profiles, "_iter_named_profile_dirs", lambda: iter(named))
    profiles._PROFILE_SCOPE_CACHE.clear()
    return profiles, root


def _names(result):
    return [name for name, _home in result]


def test_profile_scope_absent_preserves_multiplex_discovery(monkeypatch, tmp_path):
    profiles, _ = _fixture_profiles(monkeypatch, tmp_path)
    assert _names(profiles.profiles_to_serve(True)) == ["default", "main", "staging"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("[default, main]", ["default", "main"]),
        ("[default]", ["default"]),
        ("[main]", ["main"]),
        ("[]", []),
        ("[main, main, default]", ["default", "main"]),
    ],
)
def test_profile_scope_selects_deterministic_existing_profiles(monkeypatch, tmp_path, value, expected):
    profiles, _ = _fixture_profiles(monkeypatch, tmp_path, value)
    assert _names(profiles.profiles_to_serve(True)) == expected


def test_unknown_profile_warns_and_valid_entries_survive(monkeypatch, tmp_path, caplog):
    profiles, _ = _fixture_profiles(monkeypatch, tmp_path, "[main, missing]")
    with caplog.at_level("WARNING"):
        assert _names(profiles.profiles_to_serve(True)) == ["main"]
    assert "unknown profiles" in caplog.text


def test_all_unknown_is_fail_closed(monkeypatch, tmp_path, caplog):
    profiles, _ = _fixture_profiles(monkeypatch, tmp_path, "[missing]")
    with caplog.at_level("WARNING"):
        assert profiles.profiles_to_serve(True) == []


@pytest.mark.parametrize("value", ['"main"', "{name: main}", "1", "true"])
def test_malformed_profile_scope_fails_closed(monkeypatch, tmp_path, value, caplog):
    profiles, _ = _fixture_profiles(monkeypatch, tmp_path, value)
    with caplog.at_level("WARNING"):
        assert profiles.profiles_to_serve(True) == []
    assert "must be" in caplog.text


def test_null_profile_scope_is_treated_as_absent(monkeypatch, tmp_path):
    profiles, _ = _fixture_profiles(monkeypatch, tmp_path, "null")
    assert _names(profiles.profiles_to_serve(True)) == ["default", "main", "staging"]


def test_scope_only_applies_to_multiplex_path(monkeypatch, tmp_path):
    profiles, _ = _fixture_profiles(monkeypatch, tmp_path, "[main]")
    assert _names(profiles.profiles_to_serve(False)) == ["default"]


def test_scope_cache_invalidates_on_config_change(monkeypatch, tmp_path):
    profiles, root = _fixture_profiles(monkeypatch, tmp_path, "[main]")
    assert _names(profiles.profiles_to_serve(True)) == ["main"]
    (root / "config.yaml").write_text("gateway:\n  profile_scope: [default]\n", encoding="utf-8")
    assert _names(profiles.profiles_to_serve(True)) == ["default"]


def test_scope_keys_are_host_owner_keys_only():
    from hermes_cli.profile_channels import _GATEWAY_OWNER_KEYS

    assert "profile_scope" in _GATEWAY_OWNER_KEYS
    assert "cron_profile_scope" in _GATEWAY_OWNER_KEYS
