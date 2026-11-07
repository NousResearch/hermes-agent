"""Doctor must validate against the merged config, not the raw user file.

Managed-scope ``providers.<id>`` definitions live in ``$HERMES_MANAGED_DIR/config.yaml``
and are invisible to ``read_user_config_raw``: doctor reported a managed-defined provider
as unknown while the runtime resolved it fine, and silently skipped managed-defined
auxiliary blocks instead of validating them.
"""
import textwrap

import pytest

from hermes_cli import doctor_config


@pytest.fixture
def homes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    import hermes_cli.config as cfg
    from hermes_cli import managed_scope

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()
    return home, managed


def _write(path, body):
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    import hermes_cli.config as cfg
    from hermes_cli import managed_scope

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()


def test_managed_provider_definition_is_not_reported_unknown(homes):
    """The rig symptom: ``model.provider`` names a provider defined only in the managed
    scope. The runtime resolves it; doctor must agree."""
    home, managed = homes
    _write(home / "config.yaml", """
        model:
          provider: llama-swap
          default: llama-3.1-8b-instruct
    """)
    _write(managed / "config.yaml", """
        providers:
          llama-swap:
            base_url: http://localhost:8081/v1
            api_key: sk-doctor-fixture
    """)
    issues = []
    doctor_config._validate_model_config(issues)
    assert not [i for i in issues if "unknown" in i or "recognised" in i]


def test_managed_auxiliary_block_is_validated(homes, capsys):
    """Inverse symptom: an auxiliary block defined only in the managed scope must be
    seen and routed, not silently skipped."""
    home, managed = homes
    _write(home / "config.yaml", """
        model:
          provider: llama-swap
    """)
    _write(managed / "config.yaml", """
        providers:
          llama-swap:
            base_url: http://localhost:8081/v1
            api_key: fixture-token
        auxiliary:
          compression:
            provider: llama-swap
            model: llama-3.1-8b-instruct
    """)
    issues = []
    doctor_config._validate_auxiliary_config(issues)
    assert not [i for i in issues if "cannot be resolved" in i]
    assert "compression" in capsys.readouterr().out
