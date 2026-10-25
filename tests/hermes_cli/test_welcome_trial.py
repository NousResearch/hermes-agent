"""The desktop's welcome trial (``hermes_cli.welcome_trial``): who may claim it, what a claim leaves
behind, which guests keep the free model, and that a connector guest never asks for inference."""

import contextlib

import pytest

from hermes_cli import anon_auth, install_state, welcome_trial
from tests.hermes_cli.anon_portal import as_local_desktop_backend, claimed_trial, install_portal, off_local_desktop


@pytest.fixture
def portal(monkeypatch, tmp_path):
    return install_portal(monkeypatch, tmp_path)


def test_a_first_install_on_the_local_desktop_backend_claims_the_trial(portal, monkeypatch):
    state = claimed_trial(monkeypatch)

    assert state[anon_auth.WELCOME_CLAIM_FLAG] is True
    assert anon_auth.free_inference_active()
    assert install_state.existing_install()  # the machine is marked: never claimed twice here
    assert welcome_trial.claim_blocker() is not None
    assert portal.creates() == 1


def test_the_claim_is_stamped_by_the_mint_write_itself(portal, monkeypatch):
    def lost_second_write(_flag):
        raise OSError("disk full")
    monkeypatch.setattr(anon_auth, "set_guest_flag", lost_second_write)
    as_local_desktop_backend(monkeypatch)

    with contextlib.suppress(OSError):
        welcome_trial.claim_desktop_free_tier()

    assert anon_auth.free_inference_active()


BLOCKED = {
    "the CLI": lambda mp, home: None,
    "an SSH backend": lambda mp, home: (mp.setenv("HERMES_DESKTOP", "1"), mp.delenv("HERMES_DASHBOARD_SESSION_TOKEN", raising=False)),
    "guests switched off": lambda mp, home: (as_local_desktop_backend(mp),
                                              (home / "config.yaml").write_text("nous:\n  guest: false\n", encoding="utf-8")),
    "a provider key": lambda mp, home: (as_local_desktop_backend(mp), mp.setenv("OPENROUTER_API_KEY", "sk-or-test")),
    "a used machine": lambda mp, home: (as_local_desktop_backend(mp), install_state.mark_machine_used()),
}


@pytest.mark.parametrize("case", list(BLOCKED))
def test_the_trial_is_never_claimed_outside_a_first_desktop_install(portal, monkeypatch, case):
    from hermes_constants import get_hermes_home

    BLOCKED[case](monkeypatch, get_hermes_home())

    assert welcome_trial.claim_desktop_free_tier() is None
    assert portal.creates() == 0
    assert not anon_auth.free_inference_active()


def test_a_connector_guest_neither_claims_nor_carries_inference(portal, monkeypatch):
    as_local_desktop_backend(monkeypatch)
    anon_auth.ensure_portal_identity(explicit=True)  # the first hosted connector action

    assert welcome_trial.claim_desktop_free_tier() is None
    assert not anon_auth.free_inference_active()
    from hermes_cli.auth import resolve_nous_access_token
    resolve_nous_access_token()
    assert [r["body"]["purpose"] for r in portal.token_requests] == ["connectors"]


def test_a_claimed_trial_carries_inference_outside_the_desktop_too(portal, monkeypatch):
    state = claimed_trial(monkeypatch)
    off_local_desktop(monkeypatch)  # the same home, read by a CLI process

    assert anon_auth.free_inference_active()
    assert not anon_auth.guest_without_free_tier(state)


def _chat(db_path, model: str) -> None:
    from hermes_state import SessionDB

    db = SessionDB(db_path)
    db.create_session("s1", "desktop", model=model)
    db.append_message("s1", "user", "hi")
    db.close()


def test_a_lost_claim_is_not_replaced(portal, monkeypatch):
    claimed_trial(monkeypatch)
    anon_auth.clear_dead_guest("anon_credential_dead")

    assert welcome_trial.claim_desktop_free_tier() is None
    assert portal.creates() == 1


def test_a_connector_guest_adopted_from_the_shared_store_carries_no_trial(portal, monkeypatch):
    from hermes_cli.auth import _load_auth_store, _save_auth_store

    as_local_desktop_backend(monkeypatch)
    anon_auth.ensure_portal_identity(explicit=True)  # connector mint: welcome_claim False, mirrored
    store = _load_auth_store()
    store["providers"].pop("nous")  # a profile whose own identity differs adopts the shared one
    _save_auth_store(store)

    adopted = anon_auth.ensure_portal_identity(explicit=True)

    assert adopted[anon_auth.WELCOME_CLAIM_FLAG] is False
    assert not anon_auth.free_inference_active()
    assert portal.creates() == 1


LOSSES = {
    "reaped": (lambda: anon_auth.clear_dead_guest("anon_credential_dead"), 1),
    "replaced_by_connector": (lambda: (anon_auth.clear_dead_guest("anon_credential_dead"),
                                       anon_auth.ensure_portal_identity(explicit=True)), 2),
}


@pytest.mark.parametrize("loss", list(LOSSES))
def test_a_lost_trial_is_never_claimed_again_after_a_relaunch(portal, monkeypatch, loss):
    from hermes_cli import free_tier_bootstrap
    from hermes_cli.auth import resolve_provider
    from hermes_cli.auth_constants import AuthError
    from hermes_constants import get_hermes_home

    lose, creates = LOSSES[loss]
    claimed_trial(monkeypatch)
    _chat(get_hermes_home() / "state.db", anon_auth.GUEST_MODEL)
    lose()
    install_state.probe.cache_clear()  # the relaunch
    free_tier_bootstrap.reset_for_tests()
    free_tier_bootstrap.run_bootstrap(announce=False)

    assert welcome_trial.claim_desktop_free_tier() is None
    assert portal.creates() == creates
    assert not anon_auth.free_inference_active()
    with pytest.raises(AuthError):
        resolve_provider("auto")
    record = free_tier_bootstrap.current_record()
    assert (record.provider_configured, record.free_tier_account) == (False, False)



def test_a_leftover_guest_pool_row_is_not_reported_as_a_nous_login(portal, monkeypatch):
    from hermes_cli.auth import _load_auth_store, _save_auth_store, resolve_nous_runtime_credentials
    from hermes_cli.auth_nous import get_nous_auth_status_local

    claimed_trial(monkeypatch)
    resolve_nous_runtime_credentials()  # the trial ran, so the pool holds the guest's exchanged token
    store = _load_auth_store()
    store["providers"].pop("nous")  # the claim is lost; only the pool row is left
    _save_auth_store(store)

    # Provider setup would otherwise tag Nous Portal "Connected" for a credential that runs nothing.
    assert get_nous_auth_status_local()["logged_in"] is False


@pytest.mark.parametrize("desktop", [True, False], ids=["local_desktop_backend", "elsewhere"])
def test_a_connector_guest_pinned_to_nous_gets_no_inference_token(portal, monkeypatch, desktop):
    from hermes_cli.auth_constants import AuthError
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_constants import get_hermes_home

    if desktop:
        as_local_desktop_backend(monkeypatch)
    anon_auth.ensure_portal_identity(explicit=True)  # a connector action's guest, never claimed
    (get_hermes_home() / "config.yaml").write_text("model:\n  provider: nous\n  default: nous/welcome\n",
                                                   encoding="utf-8")
    with contextlib.suppress(AuthError):
        resolve_runtime_provider(requested="nous")

    assert [r["body"].get("purpose") for r in portal.token_requests if r["body"].get("purpose") != "connectors"] == []


def test_the_token_seam_refuses_an_inference_token_to_a_guest_without_the_trial(portal, monkeypatch):
    import httpx

    from hermes_cli.auth_constants import AuthError

    as_local_desktop_backend(monkeypatch)
    state = dict(anon_auth.ensure_portal_identity(explicit=True))  # a connector guest
    with httpx.Client() as client, pytest.raises(AuthError):
        anon_auth.refresh_guest_state(state, client, purpose="inference")

    assert not [r for r in portal.token_requests if r["body"].get("purpose") != "connectors"]


def _root_run():
    from hermes_cli.config_effective import load_user_config_effective
    from hermes_constants import get_default_hermes_root

    section = load_user_config_effective(get_default_hermes_root() / "config.yaml").get("onboarding") or {}
    return section.get("run")


def _relaunch():
    from hermes_cli import free_tier_bootstrap

    install_state.probe.cache_clear()
    free_tier_bootstrap.reset_for_tests()
    free_tier_bootstrap.run_bootstrap(announce=False)


@pytest.mark.parametrize("used", [False, True], ids=["fresh", "used"])
def test_the_first_desktop_boot_records_that_onboarding_is_due(portal, monkeypatch, used):
    from hermes_cli.onboarding_run import set_run

    as_local_desktop_backend(monkeypatch)
    if used:
        install_state.mark_machine_used()
    _relaunch()
    assert _root_run() is (None if used else True)

    if not used:  # once: a run that ended is not made due again by a later boot
        set_run(False)
        _relaunch()
        assert _root_run() is False


def test_a_missed_first_launch_stays_due_and_keeps_the_trial(portal, monkeypatch):
    from hermes_cli.onboarding_run import should_run
    from hermes_constants import get_hermes_home

    as_local_desktop_backend(monkeypatch)
    _relaunch()                                         # first boot; the app never opened onboarding
    _chat(get_hermes_home() / "state.db", "anthropic/claude")   # then the user chatted
    install_state.probe.cache_clear()

    assert install_state.probe().existing
    assert should_run() is True
    assert welcome_trial.claim_blocker() is None
    assert welcome_trial.claim_desktop_free_tier() is not None
    assert welcome_trial.claim_blocker() == "a Nous identity exists"


def test_a_claim_cut_off_before_its_marker_write_never_claims_twice(portal, monkeypatch):
    as_local_desktop_backend(monkeypatch)
    _relaunch()                                         # first boot: due, the trial reserved
    mark = install_state.mark_machine_used

    def die():
        raise RuntimeError("the process died here")
    monkeypatch.setattr(install_state, "mark_machine_used", die)  # the stamped guest is saved, then the crash
    with pytest.raises(RuntimeError):
        welcome_trial.claim_desktop_free_tier()
    monkeypatch.setattr(install_state, "mark_machine_used", mark)
    _relaunch()
    anon_auth.clear_dead_guest("anon_credential_dead")   # the claimed guest is reaped later

    assert welcome_trial.claim_desktop_free_tier() is None
    assert portal.creates() == 1


def test_two_installs_claiming_at_once_mint_one_trial(portal, monkeypatch, tmp_path):
    """Stable and canary started together: two data roots (each its own auth and shared store, so
    their auth locks never meet), one OS user, one marker. The canary claims while stable mints."""
    import threading

    from hermes_cli import auth_nous
    from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override

    as_local_desktop_backend(monkeypatch)
    _relaunch()                                          # the trial is reserved for this OS user
    monkeypatch.setattr(auth_nous, "_nous_shared_auth_dir", lambda: get_hermes_home() / "shared-nous")
    canary_root = tmp_path / "hermes-canary"
    canary_root.mkdir()
    canary: dict = {}
    canary_done = threading.Event()

    def canary_claims():
        token = set_hermes_home_override(canary_root)
        try:
            canary["state"] = welcome_trial.claim_desktop_free_tier()
        finally:
            reset_hermes_home_override(token)
            canary_done.set()

    stable, canary_thread = threading.current_thread(), threading.Thread(target=canary_claims)
    real_mint = anon_auth.ensure_portal_identity

    def stable_mints(**kwargs):
        if threading.current_thread() is stable and not canary_thread.is_alive() and not canary_done.is_set():
            canary_thread.start()
            canary_done.wait(timeout=2.0)                # unguarded, the canary claims in this gap
        return real_mint(**kwargs)
    monkeypatch.setattr(anon_auth, "ensure_portal_identity", stable_mints)

    assert welcome_trial.claim_desktop_free_tier() is not None
    canary_thread.join(timeout=30.0)
    assert canary == {"state": None}
    assert portal.creates() == 1


def test_one_due_answer_walks_the_provider_ladder_once(portal, monkeypatch):
    from hermes_cli import auth

    as_local_desktop_backend(monkeypatch)
    real, calls = auth.resolve_provider, []
    monkeypatch.setattr(auth, "resolve_provider", lambda *a, **kw: (calls.append(kw), real(*a, **kw))[1])

    assert welcome_trial.inference_available() is True  # fresh: the trial is claimable
    assert len(calls) == 1


def _cron_model() -> str:
    from cron.scheduler import _load_cron_job_config

    return _load_cron_job_config({"id": "j", "name": "j", "prompt": "x"}, "j", "j").model


def _kanban_worker_model(monkeypatch, tmp_path) -> str:
    from dataclasses import asdict
    from types import SimpleNamespace
    from contextlib import closing
    from hermes_cli import kanban_db as kb
    from hermes_cli.config import load_config
    from hermes_cli.kanban_db_connect import connect
    from hermes_constants import get_hermes_home
    from gateway.session_contract import Principal
    from gateway.session_kanban import build_kanban_policy
    from gateway.session_policy import restore_policy

    home = str(get_hermes_home())
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "kanban"))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.setattr("hermes_cli.profiles.resolve_profile_env", lambda name: home)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    kb.create_board("trial")
    with closing(connect(board="trial")) as conn:
        task_id = kb.create_task(conn, title="t", assignee="default", workspace_kind="dir", workspace_path=str(workspace))
        kb.recompute_ready(conn)
        task = kb.claim_task(conn, task_id)
    connection = SimpleNamespace(authority=SimpleNamespace(profile_id=home), native_owner=True,
                                 actor=Principal("owner", home, frozenset({"session:create"}), "transport"))
    params = dict(board="trial", task_id=task.id, run_id=task.current_run_id, claim_lock=task.claim_lock)
    policy, _ = build_kanban_policy(connection, params, load_config())
    return restore_policy(asdict(policy)).model


@pytest.fixture
def unconfigured_trial(portal, monkeypatch):
    """A fresh install whose Desktop claimed the trial, read by the scheduler or the Kanban
    dispatcher (not Desktop work), with no model configured anywhere."""
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    claimed_trial(monkeypatch)
    off_local_desktop(monkeypatch)
    yield
    from hermes_cli.config import load_config
    assert not (load_config().get("model") or {}).get("default")  # never persisted: a later provider wins


def test_a_claimed_trial_runs_a_default_cron_job_when_no_model_is_configured(unconfigured_trial):
    assert _cron_model() == anon_auth.GUEST_MODEL


def test_a_claimed_trial_runs_a_kanban_worker_when_no_model_is_configured(unconfigured_trial, monkeypatch, tmp_path):
    assert _kanban_worker_model(monkeypatch, tmp_path) == anon_auth.GUEST_MODEL


def test_a_configured_model_beats_the_trial_for_cron_jobs_and_kanban_workers(portal, monkeypatch, tmp_path):
    from hermes_constants import get_hermes_home

    monkeypatch.delenv("HERMES_MODEL", raising=False)
    claimed_trial(monkeypatch)
    off_local_desktop(monkeypatch)
    (get_hermes_home() / "config.yaml").write_text(
        "model:\n  provider: openrouter\n  default: anthropic/claude-sonnet-4\n", encoding="utf-8")

    assert _cron_model() == "anthropic/claude-sonnet-4"
    assert _kanban_worker_model(monkeypatch, tmp_path) == "anthropic/claude-sonnet-4"
