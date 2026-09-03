import textwrap

from hermes_cli.config import load_config, save_config


def _write_config(tmp_path, body: str):
    (tmp_path / "config.yaml").write_text(textwrap.dedent(body), encoding="utf-8")


def _read_config(tmp_path) -> str:
    return (tmp_path / "config.yaml").read_text(encoding="utf-8")




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


# --- Runtime-authority provenance proof (#98717) ---------------------------
#
# A live credential can reach the config object through a runtime authority
# (profile scope via agent.secret_scope.get_secret, or ~/.hermes/.env) that
# never exposes the value through os.environ. The save-side template guard
# must still recognize the raw ${VAR} template as the value's origin instead
# of persisting the live secret as plaintext, while a caller-typed literal no
# authority can recompute stays an intentional replacement.


def _patch_scope_secrets(monkeypatch, secrets: dict):
    import agent.secret_scope as secret_scope

    monkeypatch.setattr(
        secret_scope,
        "get_secret",
        lambda name, default=None: secrets.get(name, default),
    )


def test_save_config_preserves_template_when_scope_supplies_live_secret(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("TEST_BEARER_TOKEN", raising=False)
    _patch_scope_secrets(monkeypatch, {"TEST_BEARER_TOKEN": "opaque-live-secret-value"})
    _write_config(
        tmp_path,
        """\
        gateway:
          authorization: Bearer ${TEST_BEARER_TOKEN}
        """,
    )

    config = load_config()
    # Runtime authority resolved the live bearer into the config object;
    # an unrelated setting is then persisted by a whole-document save.
    config["gateway"]["authorization"] = "Bearer opaque-live-secret-value"
    config["display"]["compact"] = True
    save_config(config)

    saved = _read_config(tmp_path)
    assert "authorization: Bearer ${TEST_BEARER_TOKEN}" in saved
    assert "opaque-live-secret-value" not in saved


def test_save_config_preserves_template_when_dotenv_supplies_value(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("TEST_DOTENV_BEARER", raising=False)
    _patch_scope_secrets(monkeypatch, {})
    (tmp_path / ".env").write_text(
        "TEST_DOTENV_BEARER=dotenv-live-secret\n", encoding="utf-8"
    )
    _write_config(
        tmp_path,
        """\
        gateway:
          authorization: Bearer ${TEST_DOTENV_BEARER}
        """,
    )

    config = load_config()
    config["gateway"]["authorization"] = "Bearer dotenv-live-secret"
    config["display"]["compact"] = True
    save_config(config)

    saved = _read_config(tmp_path)
    assert "authorization: Bearer ${TEST_DOTENV_BEARER}" in saved
    assert "dotenv-live-secret" not in saved


def test_save_config_still_allows_intentional_replacement_under_scope(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("TEST_BEARER_TOKEN", raising=False)
    _patch_scope_secrets(monkeypatch, {"TEST_BEARER_TOKEN": "opaque-live-secret-value"})
    _write_config(
        tmp_path,
        """\
        gateway:
          authorization: Bearer ${TEST_BEARER_TOKEN}
        """,
    )

    config = load_config()
    # Positive control: a user-owned edit that no authority can recompute
    # from the template must survive as a literal replacement.
    config["gateway"]["authorization"] = "Bearer user-typed-rotation"
    save_config(config)

    saved = _read_config(tmp_path)
    assert "authorization: Bearer user-typed-rotation" in saved
    assert "${TEST_BEARER_TOKEN}" not in saved


def test_save_config_preserves_non_secret_env_ref_from_scope(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("TEST_MODEL_ALIAS", raising=False)
    _patch_scope_secrets(monkeypatch, {"TEST_MODEL_ALIAS": "claude-opus-4-6"})
    _write_config(
        tmp_path,
        """\
        model:
          default: ${TEST_MODEL_ALIAS}
        """,
    )

    config = load_config()
    config["model"]["default"] = "claude-opus-4-6"
    config["display"]["compact"] = True
    save_config(config)

    saved = _read_config(tmp_path)
    assert "default: ${TEST_MODEL_ALIAS}" in saved


def test_save_config_allows_clearing_templated_value(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("TEST_BEARER_TOKEN", raising=False)
    _patch_scope_secrets(monkeypatch, {"TEST_BEARER_TOKEN": "opaque-live-secret-value"})
    _write_config(
        tmp_path,
        """\
        gateway:
          authorization: Bearer ${TEST_BEARER_TOKEN}
        """,
    )

    config = load_config()
    config["gateway"]["authorization"] = ""
    save_config(config)

    saved = _read_config(tmp_path)
    assert "${TEST_BEARER_TOKEN}" not in saved
    assert "opaque-live-secret-value" not in saved






