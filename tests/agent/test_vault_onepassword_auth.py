"""Existing op service/session routes remain isolated when Connect is absent."""
import json
import subprocess
from unittest.mock import patch

from agent.secret_scope import set_secret_scope, reset_secret_scope, set_multiplex_active
from agent.vault_backends import unlock
from agent.vault_backends.onepassword import OnePasswordLoginBackend


def test_service_session_and_connect_auth_boundaries(monkeypatch):
    set_multiplex_active(True)
    scope = set_secret_scope({'OP_SERVICE_ACCOUNT_TOKEN': 'synthetic-service-token'})
    try:
        backend = OnePasswordLoginBackend({'binary_path': '/synthetic/op'})
        with patch.object(backend, '_op', return_value='/synthetic/op'), patch(
                'agent.vault_backends.onepassword.run_cli', return_value=subprocess.CompletedProcess([], 0, '[]', '')) as cli:
            assert backend.is_unlocked() and backend.list_items() == []
            assert cli.call_args.kwargs['env']['OP_SERVICE_ACCOUNT_TOKEN'] == 'synthetic-service-token'
            assert 'synthetic-service-token' not in repr(cli.call_args.args)
        blank = set_secret_scope({})
        try:
            backend = OnePasswordLoginBackend()
            unlock.lock('onepassword')
            assert not backend.is_unlocked()
            generation = unlock.begin_unlock('onepassword')
            assert unlock.store_session_token('onepassword', 'synthetic-session-token', generation)
            assert backend.is_unlocked()
            with patch.object(backend, '_op', return_value='/synthetic/op'), patch(
                    'agent.vault_backends.onepassword.run_cli', return_value=subprocess.CompletedProcess([], 0, '[]', '')) as cli:
                assert backend.list_items() == []
                assert cli.call_args.kwargs['env']['OP_SESSION'] == 'synthetic-session-token'
        finally:
            unlock.lock('onepassword'); reset_secret_scope(blank)
        partial = set_secret_scope({'OP_CONNECT_TOKEN': 'synthetic-connect-token'})
        try:
            import pytest
            assert OnePasswordLoginBackend().is_unlocked() is False
            with pytest.raises(RuntimeError, match='both host and token'):
                OnePasswordLoginBackend().list_items()
        finally: reset_secret_scope(partial)
    finally:
        reset_secret_scope(scope); set_multiplex_active(False)
