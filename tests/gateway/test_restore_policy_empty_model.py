"""Regression tests for #148540 - restore_policy rejects model=None/'' our writers persist.

``gateway/session_local.py`` resolves an unresolvable model to ``None``
(``_resolve_gateway_model(...) or None``) and ``gateway/session_kanban.py``
persists the raw ``''`` resolution; both receipts then failed restore_policy's
non-blank model clause with ``storage_unavailable`` on every recovery tick. The
launch path already treats an empty policy model as "resolve from config"
(``session_hosted_rpc`` falls back to ``_resolve_gateway_model``; #35381 made
the runtime recover the same value), so restore accepts what the writers
persist and rejects only non-string models.
"""
from dataclasses import asdict

import pytest


def _base_policy(tmp_path):
    from gateway.session_policy import build_policy

    cfg = {'platform_toolsets': {'cli': ['terminal']}}
    return build_policy({'cwd': str(tmp_path), 'model': 'm'}, cfg)


def test_restore_accepts_writer_persisted_empty_models(tmp_path):
    from gateway.session_policy import restore_policy

    base = _base_policy(tmp_path)
    for model in (None, ''):
        restored = restore_policy({**asdict(base), 'model': model})
        assert restored.model == model


def test_restore_still_rejects_non_string_model(tmp_path):
    from gateway.session_policy import restore_policy
    from hermes_state_runtime import RuntimeStoreError

    base = _base_policy(tmp_path)
    with pytest.raises(RuntimeStoreError, match='storage_unavailable'):
        restore_policy({**asdict(base), 'model': 123})
