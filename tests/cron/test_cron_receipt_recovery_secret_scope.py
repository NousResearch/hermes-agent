"""Regression tests for cron receipt recovery deferring forever in a multiplex gateway.

``agent.secret_scope.get_secret`` fails closed in a multiplexing process when no profile
secret scope is bound: it raises ``UnscopedSecretError`` rather than resolve another
profile's value.

The cron receipt-recovery path (``cron.scheduler_authority.reconcile_pending``) runs
OUTSIDE a cron tick, so no scope was bound when it resolved the delivery target. The
chain was::

    reconcile_pending -> _save_compose_deliver -> _resolve_delivery_targets
        -> _resolve_single_delivery_target -> _get_home_target_chat_id
        -> _env_home_target_chat_id -> _home_env_lookup -> get_secret

``UnscopedSecretError`` propagated out of that chain and was swallowed by
``reconcile_pending``'s ``except Exception``, which logged "Cron receipt recovery
deferred" and retried forever - so the receipt could never be booked. Every retry hit
the same wall.

THE FIX is to run the recovery delivery under the firing profile's secret scope
(``cron.scheduler.fire_secret_scope``), exactly as ``_run_one_job_body`` does.

What these tests pin:

* with the firing profile's scope bound, the home-target env override RESOLVES under
  multiplex semantics - the production fix, exercised through the same public helper the
  recovery path uses;
* an unscoped read still RAISES. It must not degrade to "unset" (""): an empty target
  list under ``deliver=origin`` books the receipt and deletes the admission journal while
  the output is never delivered - a silent drop, strictly worse than the loud deferral;
* an unscoped read never falls back to ``os.environ`` (that is another profile's value);
* the scope does not LEAK when the install fails after binding it.

An earlier version of this patch made the unscoped read return "" and shipped four
tests that pinned only that behaviour. Those tests passed even with the production fix
reverted, and the "" behaviour silently dropped deliveries. The tests below are written
so that reverting either half of the fix breaks them.
"""
import hashlib
import json
import os

import pytest

from agent import secret_scope as ss
from cron import scheduler
from cron.scheduler_delivery import (
    _env_home_target_chat_id,
    _home_env_lookup,
    _resolve_delivery_targets,
)


ENV_VAR = "TESTHOME_HOME_CHANNEL"
SCOPED_VALUE = "room-abc-123"


def _bind(mapping, profile_home="/tmp/test-profile"):
    """Install a secret scope and force multiplex semantics."""
    token = ss.set_secret_scope(mapping, profile_home=profile_home)
    context = ss.set_multiplex_context(True)
    return token, context


def _release(token, context):
    ss.reset_multiplex_context(context)
    ss.reset_secret_scope(token)


# --- the production fix -------------------------------------------------


def test_scoped_read_resolves_the_owning_profile_value():
    """With the firing profile's scope bound, the home-target env override resolves."""
    token, context = _bind({ENV_VAR: SCOPED_VALUE})
    try:
        assert _home_env_lookup(ENV_VAR) == SCOPED_VALUE
    finally:
        _release(token, context)


def test_fire_secret_scope_makes_the_home_target_readable_under_multiplex(monkeypatch):
    """THE RECOVERY PATH'S CONTRACT.

    ``reconcile_pending`` delivers under ``fire_secret_scope()``. Inside that scope the
    home-target read must resolve even though the process is multiplexing - this is the
    exact mechanism the production fix relies on, so it fails if the fix is reverted.
    """
    import tui_gateway.launch_profile_policy as launch_policy

    monkeypatch.setattr(
        launch_policy, "owner_secret_scope", lambda home: {ENV_VAR: SCOPED_VALUE}
    )
    context = ss.set_multiplex_context(True)
    try:
        with scheduler.fire_secret_scope():
            assert _home_env_lookup(ENV_VAR) == SCOPED_VALUE
    finally:
        ss.reset_multiplex_context(context)


# --- the silent-drop hazard --------------------------------------------


def test_unscoped_read_raises_instead_of_degrading_to_unset():
    """An unscoped read must RAISE, not return "".

    "" would resolve an EMPTY target list. With ``deliver=origin`` and no origin that
    makes ``_save_compose_deliver`` return without queueing anything while
    ``finish_completed_run`` books the receipt and unlinks the journal - the output is
    never delivered and nothing is logged. A loud deferral is the safe failure.
    """
    context = ss.set_multiplex_context(True)
    try:
        with pytest.raises(ss.UnscopedSecretError):
            _home_env_lookup(ENV_VAR)
    finally:
        ss.reset_multiplex_context(context)


def test_delivery_resolution_raises_rather_than_returning_an_empty_target_list():
    """The silent-drop shape itself: the target list must not come back empty."""
    context = ss.set_multiplex_context(True)
    try:
        with pytest.raises(ss.UnscopedSecretError):
            _resolve_delivery_targets({"deliver": "origin", "origin": None})
    finally:
        ss.reset_multiplex_context(context)


def test_unscoped_read_never_falls_back_to_the_process_environ():
    """The host environ may hold ANOTHER profile's value. It must not be read."""
    os.environ[ENV_VAR] = "another-profiles-room"
    context = ss.set_multiplex_context(True)
    try:
        with pytest.raises(ss.UnscopedSecretError):
            _home_env_lookup(ENV_VAR)
    finally:
        ss.reset_multiplex_context(context)
        os.environ.pop(ENV_VAR, None)


# --- scope lifetime ------------------------------------------------------


def test_scope_is_not_leaked_when_the_install_fails_after_binding(monkeypatch):
    """If the install raises AFTER the scope is bound, the scope must be released.

    A multiplex gateway's ticker thread services several profiles in one context, so a
    leaked scope would let the NEXT profile's tick read the PREVIOUS profile's secrets.
    """
    from cron import scheduler_provider
    import tui_gateway.launch_profile_policy as launch_policy

    monkeypatch.setattr(
        launch_policy, "owner_secret_scope", lambda home: {"LEAKED": "another-profile"}
    )

    def _boom():
        raise RuntimeError("install failed after the scope was bound")

    monkeypatch.setattr(scheduler_provider, "routed_profile_fire", _boom)
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: "/nonexistent-profile-home")

    with pytest.raises(RuntimeError):
        with scheduler.fire_secret_scope():
            pass

    assert ss.current_secret_scope() is None


def test_scope_is_released_after_a_normal_block(monkeypatch):
    """The happy path releases the scope too."""
    import tui_gateway.launch_profile_policy as launch_policy

    monkeypatch.setattr(
        launch_policy, "owner_secret_scope", lambda home: {"LEAKED": "another-profile"}
    )
    with scheduler.fire_secret_scope():
        assert ss.current_secret_scope() is not None
    assert ss.current_secret_scope() is None


# --- the recovery path, executed end to end -----------------------------


def _journal_file(home, job_id, request_id):
    """The exact path ``cron.scheduler_authority.journal_path`` computes."""
    key = hashlib.sha256(json.dumps([job_id, request_id]).encode()).hexdigest()
    path = home / "cron" / "admissions" / (key + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_reconcile_pending_delivers_with_a_profile_scope_bound(tmp_path, monkeypatch):
    """THE PRODUCTION FIX, executed end to end.

    ``reconcile_pending`` recovers a stuck receipt OUTSIDE a cron tick, so no profile
    secret scope is bound. If the recovery delivery does not bind one, the home-target
    read fails closed in a multiplex gateway and the receipt is deferred forever - every
    retry hitting the same wall. This is the bug this patch exists to close.

    This drives the real ``reconcile_pending`` over a real admission journal and asserts
    that the delivery runs with a scope bound AND that the delivery target RESOLVES.
    Reverting the scope binding in ``reconcile_pending`` fails this test.
    """
    from cron import executions
    import cron.scheduler as scheduler_mod
    import cron.scheduler_bookkeeping as bookkeeping
    from gateway import session_cron
    from hermes_cli import gateway_client
    import hermes_constants
    import tui_gateway.launch_profile_policy as launch_policy

    home = tmp_path / "profile"
    (home / "cron" / "admissions").mkdir(parents=True)
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: home)

    job_id, request_id = "job-under-test", "req-under-test"
    journal = _journal_file(home, job_id, request_id)
    journal.write_text(
        json.dumps({"params": {"job_id": job_id, "request_id": request_id}}), encoding="utf-8"
    )

    monkeypatch.setattr(executions, "execution_owner_live", lambda *a, **k: False)
    monkeypatch.setattr(executions, "get_execution", lambda *a, **k: None)
    monkeypatch.setattr(session_cron, "owner_for_home", lambda h: None)
    monkeypatch.setattr(
        launch_policy,
        "owner_secret_scope",
        lambda h: {"TELEGRAM_HOME_CHANNEL": "123456", "TELEGRAM_HOME_CHANNEL_THREAD_ID": ""},
    )

    terminal = {
        "status": "terminal",
        "job": {"id": job_id, "deliver": "origin", "origin": None},
        "result": [True, "# Cron Job: recovered\n", "# Cron Job: recovered\n", ""],
    }

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def rpc(self, method, **params):
            return terminal

    class _FakeConnect:
        async def __aenter__(self):
            return _FakeClient()

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(gateway_client, "connect_gateway", lambda: _FakeConnect())

    captured = {}

    def _fake_save(delivery, fence, answer, output, **kwargs):
        scope = ss.current_secret_scope()
        captured["scope_bound"] = scope is not None
        captured["scope_keys"] = sorted(scope.keys()) if scope else scope
        try:
            # The exact read that used to fail closed and defer this receipt forever.
            captured["home_target"] = _home_env_lookup("TELEGRAM_HOME_CHANNEL")
            captured["job"] = delivery.job
        except Exception as exc:
            captured["error"] = f"{type(exc).__name__}: {exc}"

    monkeypatch.setattr(scheduler_mod, "_save_compose_deliver", _fake_save)
    monkeypatch.setattr(bookkeeping, "fail_empty_response", lambda *a, **k: None)
    monkeypatch.setattr(bookkeeping, "finish_completed_run", lambda *a, **k: False)

    context = ss.set_multiplex_context(True)
    try:
        from cron.scheduler_authority import reconcile_pending

        reconcile_pending(allow_connect=True)
    finally:
        ss.reset_multiplex_context(context)

    assert captured.get("scope_bound") is True, (
        "the recovery delivery ran with NO profile secret scope bound - the home-target "
        "read would fail closed and this receipt would be deferred forever"
    )
    assert captured.get("home_target") == "123456", (
        "the recovery delivery could NOT read the home target under its scope - this "
        f"receipt would be deferred forever | captured={captured}"
    )
