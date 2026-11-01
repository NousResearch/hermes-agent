"""Cron model resolution across a NAMED profile with no ``model:`` of its own (card t_92b4a684).

Two arms, one symptom: a one-shot cron job created in a named profile's cron store dies before it
can spawn when that profile's ``config.yaml`` carries no ``model:`` block.

- ARM 1 — the scheduler resolved the run's model from the ACTIVE profile's config only, then RAISED
  ("has no model configured") instead of falling back to the HOST root default the root profile
  runs on.
- ARM 2 — ``create_job(pinned=True)`` locked ``_main_model_pin()``'s ``(None, None)`` onto the job,
  so it returned ``success`` while persisting a DEAD, unpinned job — the silent-failure class the
  platform bans.

These tests drive the resolution functions directly (no ``run_agent`` import, so no gateway / early
recovery machinery). The profile home is laid out the way the host does it —
``<root>/profiles/<name>`` — so ``get_default_hermes_root`` resolves ``<root>`` as the host.
"""
import pytest

from cron import jobs as cron_jobs
from cron.scheduler import _load_cron_job_config

HOST_MODEL = "host-default-model"
HOST_PROVIDER = "deepseek"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    yield


def _make_homes(tmp_path, *, profile_model=None, host_model=HOST_MODEL, host_provider=HOST_PROVIDER):
    """``(root, profile)`` with ``profile = root/profiles/lanename``; each config.yaml per request."""
    root = tmp_path / "hermes-root"
    profile = root / "profiles" / "lanename"
    profile.mkdir(parents=True)
    host_lines = []
    if host_model:
        host_lines.append(f"model:\n  default: {host_model}\n  provider: {host_provider}\n")
    (root / "config.yaml").write_text("".join(host_lines), encoding="utf-8")
    if profile_model:
        (profile / "config.yaml").write_text(
            f"model:\n  default: {profile_model}\n  provider: deepseek\n", encoding="utf-8")
    else:
        (profile / "config.yaml").write_text("agent:\n  reasoning_effort: high\n", encoding="utf-8")
    return root, profile


def _activate(monkeypatch, profile):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(str(profile))
    monkeypatch.setattr("cron.scheduler._hermes_home", None)
    return token, reset_hermes_home_override


def _stub_provider(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kw: {"provider": kw.get("requested") or HOST_PROVIDER,
                      "api_key": "k", "base_url": "https://example.invalid/v1"})


# ---------------------------------------------------------------- ARM 1: scheduler resolution

def test_scheduler_unpinned_job_inherits_host_model(tmp_path, monkeypatch):
    """A profile with no model block runs unpinned jobs on the HOST root default, not a raise."""
    _root, profile = _make_homes(tmp_path)
    token, reset = _activate(monkeypatch, profile)
    try:
        jc = _load_cron_job_config({"id": "job", "name": "job", "model": None}, "job", "job")
    finally:
        reset(token)
    assert jc.model == HOST_MODEL
    assert jc.cron_default_provider == HOST_PROVIDER


def test_scheduler_profile_model_wins_over_host(tmp_path, monkeypatch):
    """A profile WITH its own model block keeps using it (the host is only a fallback)."""
    _root, profile = _make_homes(tmp_path, profile_model="profile-model")
    token, reset = _activate(monkeypatch, profile)
    try:
        jc = _load_cron_job_config({"id": "job", "name": "job", "model": None}, "job", "job")
    finally:
        reset(token)
    assert jc.model == "profile-model"


def test_scheduler_raises_and_names_profile_store_when_no_model_anywhere(tmp_path, monkeypatch):
    """No model in the profile OR the host: raise, and name whose store it is."""
    _root, profile = _make_homes(tmp_path, host_model=None)
    token, reset = _activate(monkeypatch, profile)
    try:
        with pytest.raises(RuntimeError) as exc:
            _load_cron_job_config({"id": "job", "name": "job", "model": None}, "job", "job")
    finally:
        reset(token)
    assert "has no model configured" in str(exc.value)
    assert "lanename" in str(exc.value)


# ---------------------------------------------------------------- ARM 2: pinned lock resolution

def test_main_model_pin_inherits_host_model_and_provider(tmp_path, monkeypatch):
    _root, profile = _make_homes(tmp_path)
    token, reset = _activate(monkeypatch, profile)
    _stub_provider(monkeypatch)
    try:
        assert cron_jobs._main_model_pin() == (HOST_PROVIDER, HOST_MODEL)
    finally:
        reset(token)


def test_main_model_pin_none_when_nothing_resolves(tmp_path, monkeypatch):
    _root, profile = _make_homes(tmp_path, host_model=None)
    token, reset = _activate(monkeypatch, profile)
    try:
        assert cron_jobs._main_model_pin() == (None, None)
    finally:
        reset(token)


def test_create_job_pinned_refuses_instead_of_persisting_a_dead_job(tmp_path, monkeypatch):
    """The old behavior created an UNPINNED, dead job and returned success. Now it refuses."""
    root, profile = _make_homes(tmp_path, host_model=None)
    token, reset = _activate(monkeypatch, profile)
    try:
        with cron_jobs.use_cron_store(profile):
            with pytest.raises(ValueError) as exc:
                cron_jobs.create_job(prompt="hi", schedule="30m", pinned=True)
            assert not cron_jobs.load_jobs(), "a refused pinned job must persist NOTHING"
    finally:
        reset(token)
    assert "lanename" in str(exc.value)
    assert "dead" in str(exc.value).lower()


def test_create_job_pinned_locks_host_model_when_profile_has_none(tmp_path, monkeypatch):
    root, profile = _make_homes(tmp_path)
    token, reset = _activate(monkeypatch, profile)
    _stub_provider(monkeypatch)
    try:
        with cron_jobs.use_cron_store(profile):
            job = cron_jobs.create_job(prompt="hi", schedule="30m", pinned=True)
    finally:
        reset(token)
    assert job["model"] == HOST_MODEL
    assert job["provider"] == HOST_PROVIDER
