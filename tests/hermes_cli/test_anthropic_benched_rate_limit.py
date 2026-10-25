"""A model the Anthropic pool benched is a quota wall, not a missing credential.

When the credential pool benches the only Anthropic key for one model (a model-scoped 429
cooldown), ``_anthropic_token_or_raise`` raises an ``AuthError`` that says "rate-limited for
<model>". Without the rate-limit code, a Kanban worker started on that model exits 1, and the
dispatcher counts the run as a spawn failure; repeated runs then trip the failure breaker and
block the card although the cooldown lifts on its own. With the code, the worker exits
EX_TEMPFAIL and the dispatcher requeues the card without counting a failure.
"""
import os
from unittest import mock

from hermes_cli import runtime_provider as rp
from hermes_cli.auth import AuthError, is_rate_limited_auth_error


def _raise_for(resolver):
    with mock.patch("agent.anthropic_credentials.resolve_anthropic_token", resolver):
        try:
            rp._anthropic_token_or_raise(model="claude-test-model")
        except AuthError as exc:
            return exc
    raise AssertionError("no AuthError raised")


def _benched_for_model(*, model=None):
    return None if model else "token-for-other-models"


def test_benched_model_raises_rate_limited_auth_error():
    exc = _raise_for(_benched_for_model)
    assert is_rate_limited_auth_error(exc)
    assert not exc.relogin_required


def test_benched_model_maps_to_kanban_rate_limit_exit():
    from hermes_cli.cli_single_query import _single_query_exit_code
    from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE
    exc = _raise_for(_benched_for_model)
    with mock.patch.dict(os.environ, {"HERMES_KANBAN_TASK": "t_test"}):
        code = _single_query_exit_code(None, credentials_rate_limited=is_rate_limited_auth_error(exc))
    assert code == KANBAN_RATE_LIMIT_EXIT_CODE


def test_missing_credentials_stay_non_rate_limited():
    exc = _raise_for(lambda *, model=None: None)
    assert not is_rate_limited_auth_error(exc)
