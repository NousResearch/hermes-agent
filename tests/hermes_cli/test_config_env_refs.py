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






