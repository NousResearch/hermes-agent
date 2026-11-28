"""Regression for #146428: recovered cron output must retain the owning home target.

An unscoped multiplex lookup cannot return "unset": a deliver=origin job with
no origin can then queue nothing and finish_completed_run unlinks its receipt.
Recovery must bind the owner's secret scope before resolving home targets.
"""

import hashlib
import json

import pytest

from agent import secret_scope


@pytest.fixture()
def multiplex_no_scope():
    assert secret_scope.current_secret_scope() is None
    context = secret_scope.set_multiplex_context(True)
    try:
        yield
    finally:
        secret_scope.reset_multiplex_context(context)


@pytest.mark.parametrize("name", ["MATRIX_HOME_ROOM", "TELEGRAM_HOME_CHANNEL"])
def test_unscoped_home_lookup_fails_closed(multiplex_no_scope, name):
    from cron.scheduler_delivery import _home_env_lookup

    with pytest.raises(secret_scope.UnscopedSecretError):
        _home_env_lookup(name)


def test_unscoped_lookup_never_reads_sibling_process_env(multiplex_no_scope, monkeypatch):
    from cron.scheduler_delivery import _home_env_lookup

    monkeypatch.setenv("MATRIX_HOME_ROOM", "sibling-profile-room")
    with pytest.raises(secret_scope.UnscopedSecretError):
        _home_env_lookup("MATRIX_HOME_ROOM")


def test_scoped_home_lookup_uses_owning_profile(tmp_path, monkeypatch):
    from cron.scheduler_delivery import _home_env_lookup

    home = tmp_path / "profiles" / "ops"
    home.mkdir(parents=True)
    (home / ".env").write_text('MATRIX_HOME_ROOM="owner-room"\n', encoding="utf-8")
    monkeypatch.setenv("MATRIX_HOME_ROOM", "sibling-room")
    scope = secret_scope.set_secret_scope(
        secret_scope.build_profile_secret_scope(home), profile_home=str(home)
    )
    context = secret_scope.set_multiplex_context(True)
    try:
        assert _home_env_lookup("MATRIX_HOME_ROOM") == "owner-room"
    finally:
        secret_scope.reset_multiplex_context(context)
        secret_scope.reset_secret_scope(scope)


def test_reconcile_pending_resolves_owning_profile_home_target(tmp_path, monkeypatch):
    """Drive real admission-journal recovery through its delivery-scope boundary."""
    from cron import executions
    import cron.scheduler as scheduler
    import cron.scheduler_bookkeeping as bookkeeping
    from cron.scheduler_authority import reconcile_pending
    from gateway import session_cron
    from hermes_cli import gateway_client
    import hermes_constants
    import tui_gateway.launch_profile_policy as launch_policy

    home = tmp_path / "profile"
    admission_dir = home / "cron" / "admissions"
    admission_dir.mkdir(parents=True)
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: home)
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: home)

    job_id, request_id = "recovery-scope-job", "recovery-scope-request"
    digest = hashlib.sha256(json.dumps([job_id, request_id]).encode()).hexdigest()
    (admission_dir / f"{digest}.json").write_text(
        json.dumps({"params": {"job_id": job_id, "request_id": request_id}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(executions, "execution_owner_live", lambda *a, **kw: False)
    monkeypatch.setattr(executions, "get_execution", lambda *a, **kw: None)
    monkeypatch.setattr(session_cron, "owner_for_home", lambda home: None)
    monkeypatch.setattr(
        launch_policy,
        "owner_secret_scope",
        lambda home: {"TELEGRAM_HOME_CHANNEL": "owner-chat"},
    )

    terminal = {
        "status": "terminal",
        "job": {"id": job_id, "deliver": "origin", "origin": None},
        "result": [True, "# Cron Job: recovered\n", "# Cron Job: recovered\n", ""],
    }

    class FakeGateway:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def rpc(self, method, **params):
            assert method == "cron.recover"
            return terminal

    monkeypatch.setattr(gateway_client, "connect_gateway", FakeGateway)
    captured = {}

    def capture_delivery(delivery, fence, answer, output, **kwargs):
        from cron.scheduler_delivery import _home_env_lookup

        captured["bound"] = secret_scope.current_secret_scope() is not None
        captured["target"] = _home_env_lookup("TELEGRAM_HOME_CHANNEL")

    monkeypatch.setattr(scheduler, "_save_compose_deliver", capture_delivery)
    monkeypatch.setattr(bookkeeping, "fail_empty_response", lambda *a, **kw: None)
    monkeypatch.setattr(bookkeeping, "finish_completed_run", lambda *a, **kw: False)

    context = secret_scope.set_multiplex_context(True)
    try:
        reconcile_pending(allow_connect=True)
    finally:
        secret_scope.reset_multiplex_context(context)

    assert captured == {"bound": True, "target": "owner-chat"}
