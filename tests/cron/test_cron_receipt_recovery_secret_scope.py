"""Regression tests for cron receipt recovery deferring forever in a multiplex gateway.

``agent.secret_scope.get_secret`` fails closed in a multiplexing process when no
profile secret scope is bound: it raises ``UnscopedSecretError`` rather than
resolve another profile's value.

The cron receipt-recovery path (``cron.scheduler_authority.reconcile_pending``)
runs OUTSIDE a cron tick, so no scope was bound when it resolved the delivery
target. The chain was::

    reconcile_pending -> _save_compose_deliver -> _resolve_delivery_targets
        -> _resolve_single_delivery_target -> _get_home_target_chat_id
        -> _env_home_target_chat_id -> _home_env_lookup -> get_secret

``UnscopedSecretError`` propagated out of that chain and was swallowed by
``reconcile_pending``'s ``except Exception``, which logged "Cron receipt
recovery deferred" and retried forever - so the receipt could never be booked.
Every retry hit the same wall.

These tests cover the delivery-side contract that makes that impossible:

* an unscoped read is "no env override", not a failure - the caller then falls
  through to the config.yaml home channel, the documented next step;
* an unscoped read must NOT fall back to ``os.environ`` (that is the
  cross-profile leak ``get_secret`` exists to prevent);
* with a scope bound, the env override still resolves.
"""
import os

from agent import secret_scope as ss
from cron.scheduler_delivery import _home_env_lookup


ENV_VAR = "MATRIX_HOME_ROOM"
SCOPED_VALUE = "room-abc-123"


def _scoped_read(env_var: str) -> str:
    return _home_env_lookup(env_var)


def test_unscoped_read_returns_no_override_instead_of_raising(monkeypatch):
    """An unscoped read in a multiplex process yields '' - it must not raise.

    Raising here aborts the whole delivery, which is what stranded the receipts.
    """
    token = ss.set_multiplex_context(True)
    try:
        assert _scoped_read(ENV_VAR) == ""
    finally:
        ss.reset_multiplex_context(token)


def test_unscoped_read_does_not_fall_back_to_process_environ(monkeypatch):
    """The host environ may hold ANOTHER profile's value - it must not be read."""
    monkeypatch.setenv(ENV_VAR, "another-profiles-room")
    token = ss.set_multiplex_context(True)
    try:
        assert _scoped_read(ENV_VAR) == ""
    finally:
        ss.reset_multiplex_context(token)


def test_scoped_read_still_resolves_the_override(monkeypatch):
    """With the owning profile's scope bound, the env override still wins."""
    monkeypatch.delenv(ENV_VAR, raising=False)
    scope_token = ss.set_secret_scope({ENV_VAR: SCOPED_VALUE}, profile_home=os.getcwd())
    try:
        assert _scoped_read(ENV_VAR) == SCOPED_VALUE
    finally:
        ss.reset_secret_scope(scope_token)


def test_scope_is_not_leaked_after_reset(monkeypatch):
    """After the scope is reset the read is unscoped again - '' - not the old value."""
    scope_token = ss.set_secret_scope({ENV_VAR: SCOPED_VALUE}, profile_home=os.getcwd())
    ss.reset_secret_scope(scope_token)
    token = ss.set_multiplex_context(True)
    try:
        assert _scoped_read(ENV_VAR) == ""
    finally:
        ss.reset_multiplex_context(token)
