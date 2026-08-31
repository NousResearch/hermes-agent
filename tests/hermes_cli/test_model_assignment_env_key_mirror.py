"""#88990: Desktop model assignment must not copy env-backed keys into model.api_key.

``POST /api/model/set`` (``_apply_model_assignment_sync``) mirrors a custom
provider entry's ``api_key`` into ``model.api_key``. ``load_config()`` expands
``${VAR}`` env refs to plaintext, so before the fix the RESOLVED secret was
written into config.yaml — and re-written on every re-apply even after the
user deleted it by hand. The mirror must be skipped when the on-disk provider
entry references the environment (``${VAR}`` template or ``key_env``), and
preserved when the entry holds a literal key.
"""

import importlib
import os

import pytest
import yaml


@pytest.fixture()
def _hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_cli.config as config_mod

    importlib.reload(config_mod)
    yield home
    importlib.reload(config_mod)


def _write_config(home, providers):
    cfg = {
        "model": {"provider": "openrouter", "default": "some/model"},
        "providers": providers,
    }
    (home / "config.yaml").write_text(yaml.safe_dump(cfg))


def _apply(provider, model="local/model"):
    import hermes_cli.web_server as ws

    return ws._apply_model_assignment_sync("main", provider, model, "", "")


def _raw_model_cfg(home):
    return (yaml.safe_load((home / "config.yaml").read_text()) or {}).get("model", {})


def test_env_ref_key_not_copied_into_model_api_key(_hermes_home, monkeypatch):
    monkeypatch.setenv("MY_LOCAL_KEY", "sk-super-secret-value")
    _write_config(
        _hermes_home,
        {"mylocal": {"base_url": "http://localhost:1234/v1", "api_key": "${MY_LOCAL_KEY}"}},
    )
    _apply("mylocal")
    model_cfg = _raw_model_cfg(_hermes_home)
    assert "sk-super-secret-value" not in yaml.safe_dump(model_cfg)
    assert not model_cfg.get("api_key")


def test_key_env_entry_not_copied(_hermes_home, monkeypatch):
    monkeypatch.setenv("MYLOCAL_API_KEY", "sk-keyenv-secret")
    _write_config(
        _hermes_home,
        {"mylocal": {"base_url": "http://localhost:1234/v1", "key_env": "MYLOCAL_API_KEY"}},
    )
    _apply("mylocal")
    assert "sk-keyenv-secret" not in (_hermes_home / "config.yaml").read_text()


def test_literal_key_still_mirrored(_hermes_home):
    _write_config(
        _hermes_home,
        {"mylocal": {"base_url": "http://localhost:1234/v1", "api_key": "literal-key-abc"}},
    )
    _apply("mylocal")
    assert _raw_model_cfg(_hermes_home).get("api_key") == "literal-key-abc"
