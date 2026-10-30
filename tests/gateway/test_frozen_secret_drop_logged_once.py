"""#145704: a frozen config secret the profile dropped is reported once per session, not every turn.

The ref is never pruned once its entries are dropped, so every turn re-reads the same entry. The value
is still dropped on every call; only the WARNING is deduplicated per (session, path).
"""
import json
import logging
from types import SimpleNamespace

from gateway import session_policy_credentials as creds


def _policy(session, path):
    entries = [[list(path), 'fingerprint-of-the-frozen-value']]
    ref = creds.PREFIX + json.dumps({'session': session, 'entries': entries})
    return SimpleNamespace(config_secret_ref=ref, follows_profile=False, config_json='{}')


def _recover(monkeypatch, tmp_path, policy):
    from gateway import run
    monkeypatch.setattr(creds, '_check_scope', lambda *a, **k: None)
    monkeypatch.setattr(creds, '_route_paths', lambda policy: set())
    monkeypatch.setattr(run, '_load_gateway_config', lambda *a, **k: {})  # the block was removed
    authority = SimpleNamespace(db=SimpleNamespace(db_path=str(tmp_path / 'state.db')), _local_config_secrets={})
    return creds.recover_config_secrets(authority, policy, skip=())


def test_a_dropped_secret_is_reported_once_per_session(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(creds, '_REPORTED_DROPS', set())
    path = ('mcp_servers', 'demo', 'headers')
    caplog.set_level(logging.WARNING, logger=creds.__name__)

    for _ in range(3):
        values = _recover(monkeypatch, tmp_path, _policy('s1', path))
        assert values.dropped == (path,)  # still dropped on every turn
        assert path not in values

    assert sum('no longer matches the profile config' in r.getMessage() for r in caplog.records) == 1


def test_another_session_with_the_same_drop_is_reported_too(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(creds, '_REPORTED_DROPS', set())
    path = ('mcp_servers', 'demo', 'headers')
    caplog.set_level(logging.WARNING, logger=creds.__name__)

    _recover(monkeypatch, tmp_path, _policy('s1', path))
    _recover(monkeypatch, tmp_path, _policy('s2', path))

    assert sum('no longer matches the profile config' in r.getMessage() for r in caplog.records) == 2
