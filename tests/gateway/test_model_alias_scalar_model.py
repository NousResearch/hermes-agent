"""A scalar ``model: <name>`` config is valid; the alias credential reader must not crash on it (N105)."""
from pathlib import Path
from types import SimpleNamespace


def test_alias_credential_reader_accepts_a_scalar_model_entry(tmp_path):
    from gateway.session_mutation_model import _alias_credential
    from hermes_cli.model_switch import ModelSwitchResult

    (tmp_path / 'config.yaml').write_text('model: far-model\nmodel_aliases: {}\n')
    authority = SimpleNamespace(db=SimpleNamespace(db_path=str(Path(tmp_path) / 'state.db')))
    # A built-in direct alias the profile does not redeclare: both config sections are scanned.
    result = ModelSwitchResult(success=True, new_model='far-model', target_provider='custom',
                               base_url='http://127.0.0.3:9/v1', resolved_via_alias='far')
    assert _alias_credential(authority, result) == (None, None, None)
