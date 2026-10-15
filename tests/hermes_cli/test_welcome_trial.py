"""The desktop's welcome trial (``hermes_cli.welcome_trial``): who may claim it, what a claim leaves
behind, which guests keep the free model, and that a connector guest never asks for inference."""

import contextlib

import pytest

from hermes_cli import anon_auth, install_state, welcome_trial
from tests.hermes_cli.anon_portal import as_local_desktop_backend, claimed_trial, install_portal


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


def test_a_claimed_guest_carries_inference_only_on_the_local_desktop_backend(portal, monkeypatch):
    state = claimed_trial(monkeypatch)
    monkeypatch.delenv("HERMES_DASHBOARD_SESSION_TOKEN")  # the same home, read by a CLI process

    assert not anon_auth.free_inference_active()
    assert anon_auth.guest_without_free_tier(state)


def _chat(db_path, model: str) -> None:
    from hermes_state import SessionDB

    db = SessionDB(db_path)
    db.create_session("s1", "desktop", model=model)
    db.append_message("s1", "user", "hi")
    db.close()


def _legacy_guest() -> None:
    """A guest minted before the claim existed: no welcome_claim key at all, in either store."""
    from hermes_cli.auth import _load_auth_store, _load_provider_state, _save_auth_store, _store_provider_state
    from hermes_cli.auth_nous import _write_shared_nous_state

    anon_auth.ensure_portal_identity(explicit=True)
    store = _load_auth_store()
    state = dict(_load_provider_state(store, "nous"))
    state.pop(anon_auth.WELCOME_CLAIM_FLAG, None)
    _store_provider_state(store, "nous", state, set_active=False)
    _save_auth_store(store)
    _write_shared_nous_state(state)


@pytest.mark.parametrize(("model", "kept"), [(anon_auth.GUEST_MODEL, True), ("anthropic/claude", False)])
def test_a_guest_that_chatted_on_the_free_model_keeps_it(portal, monkeypatch, model, kept):
    from hermes_constants import get_hermes_home

    as_local_desktop_backend(monkeypatch)
    _legacy_guest()  # grandfathering is only for guests minted before the key existed
    _chat(get_hermes_home() / "state.db", model)

    assert welcome_trial.grandfather_existing_guest() is kept
    assert anon_auth.free_inference_active() is kept


def test_grandfathering_never_overwrites_a_sign_in_from_another_profile(portal, monkeypatch):
    from hermes_cli.auth_nous import _read_shared_nous_state, _write_shared_nous_state
    from hermes_constants import get_hermes_home
    from tests.hermes_cli.anon_portal import PORTAL

    as_local_desktop_backend(monkeypatch)
    _legacy_guest()                                    # this profile still holds the old guest
    _chat(get_hermes_home() / "state.db", anon_auth.GUEST_MODEL)
    _write_shared_nous_state({"auth_method": "oauth_device_code", "portal_base_url": PORTAL,
                              "access_token": "at-signed-in", "refresh_token": "rt-signed-in"})

    assert welcome_trial.grandfather_existing_guest() is False
    assert _read_shared_nous_state()["refresh_token"] == "rt-signed-in"
    assert anon_auth.WELCOME_CLAIM_FLAG not in anon_auth.current_nous_state()


def test_a_lost_claim_is_not_replaced(portal, monkeypatch):
    claimed_trial(monkeypatch)
    anon_auth.clear_dead_guest("anon_credential_dead")

    assert welcome_trial.claim_desktop_free_tier() is None
    assert portal.creates() == 1


def test_a_reaped_claimant_replaced_by_a_connector_guest_is_not_grandfathered(portal, monkeypatch):
    from hermes_constants import get_hermes_home

    claimed_trial(monkeypatch)
    _chat(get_hermes_home() / "state.db", anon_auth.GUEST_MODEL)
    anon_auth.clear_dead_guest("anon_credential_dead")
    anon_auth.ensure_portal_identity(explicit=True)  # a connector action mints the replacement

    assert welcome_trial.grandfather_existing_guest() is False
    assert not anon_auth.free_inference_active()


def test_a_connector_guest_adopted_from_the_shared_store_is_not_grandfathered(portal, monkeypatch):
    from hermes_cli.auth import _load_auth_store, _save_auth_store
    from hermes_constants import get_hermes_home

    as_local_desktop_backend(monkeypatch)
    anon_auth.ensure_portal_identity(explicit=True)  # connector mint: welcome_claim False, mirrored
    store = _load_auth_store()
    store["providers"].pop("nous")  # a profile whose own identity differs adopts the shared one
    _save_auth_store(store)
    _chat(get_hermes_home() / "state.db", anon_auth.GUEST_MODEL)

    adopted = anon_auth.ensure_portal_identity(explicit=True)

    assert adopted[anon_auth.WELCOME_CLAIM_FLAG] is False
    assert welcome_trial.grandfather_existing_guest() is False
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
    assert welcome_trial.grandfather_existing_guest() is False
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
