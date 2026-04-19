import textwrap

from hermes_cli.config import load_config, save_config


def _write_config(tmp_path, body: str):
    (tmp_path / "config.yaml").write_text(textwrap.dedent(body), encoding="utf-8")


def _read_config(tmp_path) -> str:
    return (tmp_path / "config.yaml").read_text(encoding="utf-8")


def test_save_config_preserves_env_refs_on_unrelated_change(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TU_ZI_API_KEY", "sk-realsecret")
    monkeypatch.setenv("ALT_SECRET", "alt-secret")
    _write_config(
        tmp_path,
        """\
        custom_providers:
          - name: tuzi
            base_url: https://api.tu-zi.com
            api_key: ${TU_ZI_API_KEY}
            headers:
              Authorization: Bearer ${ALT_SECRET}
            model: claude-opus-4-6
        model:
          default: claude-opus-4-6
        """,
    )

    config = load_config()
    config["model"]["default"] = "doubao-pro"
    save_config(config)

    saved = _read_config(tmp_path)
    assert "api_key: ${TU_ZI_API_KEY}" in saved
    assert "Authorization: Bearer ${ALT_SECRET}" in saved
    assert "sk-realsecret" not in saved
    assert "alt-secret" not in saved


def test_save_config_preserves_unresolved_env_refs(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("MISSING_SECRET", raising=False)
    _write_config(
        tmp_path,
        """\
        custom_providers:
          - name: unresolved
            api_key: ${MISSING_SECRET}
            model: claude-opus-4-6
        model:
          default: claude-opus-4-6
        """,
    )

    config = load_config()
    config["display"]["compact"] = True
    save_config(config)

    assert "api_key: ${MISSING_SECRET}" in _read_config(tmp_path)


def test_save_config_allows_intentional_secret_value_change(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TU_ZI_API_KEY", "sk-old-secret")
    _write_config(
        tmp_path,
        """\
        custom_providers:
          - name: tuzi
            api_key: ${TU_ZI_API_KEY}
            model: claude-opus-4-6
        model:
          default: claude-opus-4-6
        """,
    )

    config = load_config()
    config["custom_providers"][0]["api_key"] = "sk-new-secret"
    save_config(config)

    saved = _read_config(tmp_path)
    assert "api_key: sk-new-secret" in saved
    assert "${TU_ZI_API_KEY}" not in saved


def test_save_config_preserves_template_when_env_rotates_after_load(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TU_ZI_API_KEY", "sk-old-secret")
    _write_config(
        tmp_path,
        """\
        custom_providers:
          - name: tuzi
            api_key: ${TU_ZI_API_KEY}
            model: claude-opus-4-6
        model:
          default: claude-opus-4-6
        """,
    )

    config = load_config()
    monkeypatch.setenv("TU_ZI_API_KEY", "sk-rotated-secret")
    config["model"]["default"] = "doubao-pro"
    save_config(config)

    saved = _read_config(tmp_path)
    assert "api_key: ${TU_ZI_API_KEY}" in saved
    assert "sk-old-secret" not in saved
    assert "sk-rotated-secret" not in saved


def test_save_config_keeps_edited_partial_template_strings_literal(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("ALT_SECRET", "alt-secret")
    _write_config(
        tmp_path,
        """\
        custom_providers:
          - name: tuzi
            headers:
              Authorization: Bearer ${ALT_SECRET}
            model: claude-opus-4-6
        model:
          default: claude-opus-4-6
        """,
    )

    config = load_config()
    config["custom_providers"][0]["headers"]["Authorization"] = "Token alt-secret"
    save_config(config)

    saved = _read_config(tmp_path)
    assert "Authorization: Token alt-secret" in saved
    assert "Authorization: Bearer ${ALT_SECRET}" not in saved


def test_save_config_falls_back_to_positional_matching_for_duplicate_names(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("FIRST_SECRET", "first-secret")
    monkeypatch.setenv("SECOND_SECRET", "second-secret")
    _write_config(
        tmp_path,
        """\
        custom_providers:
          - name: duplicate
            api_key: ${FIRST_SECRET}
            model: claude-opus-4-6
          - name: duplicate
            api_key: ${SECOND_SECRET}
            model: doubao-pro
        model:
          default: claude-opus-4-6
        """,
    )

    config = load_config()
    config["display"]["compact"] = True
    save_config(config)

    saved = _read_config(tmp_path)
    assert saved.count("name: duplicate") == 2
    assert "api_key: ${FIRST_SECRET}" in saved
    assert "api_key: ${SECOND_SECRET}" in saved
    assert "first-secret" not in saved
    assert "second-secret" not in saved


def test_save_config_caches_expanded_form_not_raw_templates(monkeypatch, tmp_path):
    """save_config() must cache the expanded form so subsequent saves still
    restore ${VAR} templates, even when the first save got raw templates."""
    from hermes_cli.config import load_config, save_config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("SECRET_KEY", "super-secret-value")
    _write_config(
        tmp_path,
        """\
        _config_version: 19
        model:
          default: claude-opus-4-6
        custom_providers:
          - name: example
            api_key: ${SECRET_KEY}
        """,
    )

    # First load + save: populate cache with expanded value, restore template
    config = load_config()
    assert config["custom_providers"][0]["api_key"] == "super-secret-value"
    save_config(config)
    assert "api_key: ${SECRET_KEY}" in _read_config(tmp_path)

    # Second save (no prior load): cache must hold expanded form so template
    # is still recognised and restored, not replaced with plaintext
    config2 = load_config()
    config2["model"]["default"] = "gpt-4o"
    save_config(config2)
    saved = _read_config(tmp_path)
    assert "api_key: ${SECRET_KEY}" in saved
    assert "super-secret-value" not in saved


def test_migrate_config_preserves_env_refs(monkeypatch, tmp_path):
    """migrate_config() must call load_config() first so the expanded-form
    cache is populated before any migration step calls save_config().
    Without that, save_config() would cache raw templates and write
    expanded plaintext secrets to disk on the first migration save."""
    from hermes_cli.config import migrate_config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("MIGRATION_SECRET", "migration-secret-value")
    # Write a config at version 13 (below the latest 19) so migration runs.
    # Version 13→14 migration adds a `toolsets` field; subsequent migrations
    # up to 19 touch other fields but all call save_config() internally.
    _write_config(
        tmp_path,
        """\
        _config_version: 13
        model:
          default: claude-opus-4-6
        custom_providers:
          - name: old-provider
            base_url: https://api.example.com
            api_key: ${MIGRATION_SECRET}
        """,
    )

    results = migrate_config(interactive=False, quiet=True)

    saved = _read_config(tmp_path)
    # Env var template must be preserved — no plaintext leak
    assert "api_key: ${MIGRATION_SECRET}" in saved
    assert "migration-secret-value" not in saved
    # Migration should have updated the version
    assert "_config_version: 19" in saved or "_config_version: 1" not in saved
