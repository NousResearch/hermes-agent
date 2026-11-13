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
from cron.scheduler import (
    _cron_route_request, _load_cron_job_config, _resolve_job_runtime,
)
from cron.scheduler_preflight import _preflight_check_provider_key

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
    _root, profile = _make_homes(tmp_path, host_model=None)
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
    _root, profile = _make_homes(tmp_path)
    token, reset = _activate(monkeypatch, profile)
    _stub_provider(monkeypatch)
    try:
        with cron_jobs.use_cron_store(profile):
            job = cron_jobs.create_job(prompt="hi", schedule="30m", pinned=True)
    finally:
        reset(token)
    assert job["model"] == HOST_MODEL
    assert job["provider"] == HOST_PROVIDER


# ------------------------------------- ARM 3: ONE route, TWO readers (preflight + the run)

HOST_BASE_URL = "http://127.0.0.1:9/v1"
UNPINNED_JOB = {"id": "job", "name": "job", "prompt": "x"}


def _make_inherited_route_homes(tmp_path, *, profile_model=None):
    """Host root block carries a FULL route (provider + endpoint); profile carries no model block.

    ``providers.lab`` sits on the PROFILE home because that is the active home the resolver reads;
    the route itself (base_url included) lives only on the host root.
    """
    root = tmp_path / "hermes-root"
    profile = root / "profiles" / "lanename"
    profile.mkdir(parents=True)
    (root / "config.yaml").write_text(
        "model:\n"
        f"  default: {HOST_MODEL}\n"
        "  provider: custom:lab\n"
        f"  base_url: {HOST_BASE_URL}\n", encoding="utf-8")
    profile_cfg = (
        "providers:\n"
        "  lab:\n"
        f"    base_url: {HOST_BASE_URL}\n"
        '    api_key: ""\n'
        f"    models: [{HOST_MODEL}]\n")
    if profile_model:
        profile_cfg += f"model:\n  default: {profile_model}\n  provider: custom:lab\n"
    (profile / "config.yaml").write_text(profile_cfg, encoding="utf-8")
    return root, profile


def _capture_route(monkeypatch):
    """Record the kwargs the resolver is called with (callers import it late, so patch the module)."""
    import hermes_cli.runtime_provider as rp
    seen = {}

    def fake(**kwargs):
        seen.clear()
        seen.update(kwargs)
        return {"provider": kwargs.get("requested") or HOST_PROVIDER, "api_key": "k",
                "base_url": HOST_BASE_URL}

    monkeypatch.setattr(rp, "resolve_runtime_provider", fake)
    return seen


def test_unpinned_job_route_carries_the_inherited_host_block(tmp_path, monkeypatch):
    """The inherited route travels as a model block — base_url included — not as a bare name."""
    _root, profile = _make_inherited_route_homes(tmp_path)
    token, reset = _activate(monkeypatch, profile)
    try:
        jc = _load_cron_job_config(UNPINNED_JOB, "job", "job")
    finally:
        reset(token)
    assert jc.model_cfg_inherited is True
    route = _cron_route_request(UNPINNED_JOB, jc, jc.cfg)
    assert route["requested"] == "custom:lab"
    assert route["target_model"] == HOST_MODEL
    assert route["model_cfg_override"]["base_url"] == HOST_BASE_URL


def test_profile_own_block_is_not_handed_over_as_an_override(tmp_path, monkeypatch):
    """A profile with its own block resolves from its own home exactly as it does today."""
    _root, profile = _make_inherited_route_homes(tmp_path, profile_model="profile-model")
    token, reset = _activate(monkeypatch, profile)
    try:
        jc = _load_cron_job_config(UNPINNED_JOB, "job", "job")
    finally:
        reset(token)
    assert jc.model_cfg_inherited is False
    route = _cron_route_request(UNPINNED_JOB, jc, jc.cfg)
    assert route["target_model"] == "profile-model"
    assert "model_cfg_override" not in route


def test_preflight_probes_the_same_route_the_run_resolves(tmp_path, monkeypatch):
    """ONE computation, TWO readers: preflight's request is byte-for-byte the run's."""
    _root, profile = _make_inherited_route_homes(tmp_path)
    token, reset = _activate(monkeypatch, profile)
    try:
        jc = _load_cron_job_config(UNPINNED_JOB, "job", "job")
        seen = _capture_route(monkeypatch)
        assert _preflight_check_provider_key(UNPINNED_JOB, jc.cfg, jc) is None
        preflight_route = dict(seen)
        _resolve_job_runtime(UNPINNED_JOB, "job", jc)
        runtime_route = dict(seen)
    finally:
        reset(token)
    assert preflight_route == runtime_route
    assert runtime_route["requested"] == "custom:lab"
    assert runtime_route["target_model"] == HOST_MODEL


def test_route_without_jc_keeps_the_cfg_only_computation(tmp_path, monkeypatch):
    """``jc=None`` is the pre-fix judgment verbatim: no caller or test changes meaning."""
    _root, profile = _make_inherited_route_homes(tmp_path)
    token, reset = _activate(monkeypatch, profile)
    try:
        jc = _load_cron_job_config(UNPINNED_JOB, "job", "job")
    finally:
        reset(token)
    assert _cron_route_request(UNPINNED_JOB, None, jc.cfg) == {"requested": None, "target_model": ""}


def test_preflight_stops_false_blocking_a_fully_configured_inherited_route(tmp_path, monkeypatch):
    """The crux: the pre-fix probe judged ``auto``/"" and refused a home that resolves fine.

    Driven against the REAL resolver (no stub): the named ``custom:lab`` provider entry supplies a
    no-auth endpoint, so a healthy inherited route must produce no verdict at all.
    """
    _root, profile = _make_inherited_route_homes(tmp_path)
    token, reset = _activate(monkeypatch, profile)
    try:
        jc = _load_cron_job_config(UNPINNED_JOB, "job", "job")
        assert _preflight_check_provider_key(UNPINNED_JOB, jc.cfg, jc) is None
    finally:
        reset(token)


def _make_bare_custom_homes(tmp_path, *, profile_carried: bool):
    """Bare ``custom`` whose endpoint + model live in ONE ``model:`` block.

    ``profile_carried=True`` writes that block on the profile (today's working shape — the control);
    ``False`` puts it on the HOST root and leaves the profile without a ``model:`` block (the shape
    that used to die ``AuthError: provider 'custom' resolved without credentials``).
    """
    root = tmp_path / ("profile-carried" if profile_carried else "root-inherited")
    profile = root / "profiles" / "lanename"
    profile.mkdir(parents=True)
    block = (
        "model:\n"
        f"  default: {HOST_MODEL}\n"
        "  provider: custom\n"
        f"  base_url: {HOST_BASE_URL}\n")
    (root / "config.yaml").write_text("" if profile_carried else block, encoding="utf-8")
    (profile / "config.yaml").write_text(block if profile_carried else "{}\n", encoding="utf-8")
    return root, profile


def test_bare_custom_inherited_block_resolves_the_root_endpoint(tmp_path, monkeypatch):
    """Inheritance carries the FULL route: the endpoint in the root block is reached, not dropped."""
    _root, profile = _make_bare_custom_homes(tmp_path, profile_carried=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    token, reset = _activate(monkeypatch, profile)
    try:
        jc = _load_cron_job_config(UNPINNED_JOB, "job", "job")
        runtime, model = _resolve_job_runtime(UNPINNED_JOB, "job", jc)
    finally:
        reset(token)
    assert jc.model_cfg_inherited is True
    assert model == HOST_MODEL
    assert runtime["base_url"] == HOST_BASE_URL


def test_bare_custom_profile_carried_block_is_the_control(tmp_path, monkeypatch):
    """The same block written on the profile resolves identically — no behaviour change there."""
    _root, profile = _make_bare_custom_homes(tmp_path, profile_carried=True)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    token, reset = _activate(monkeypatch, profile)
    try:
        jc = _load_cron_job_config(UNPINNED_JOB, "job", "job")
        runtime, model = _resolve_job_runtime(UNPINNED_JOB, "job", jc)
    finally:
        reset(token)
    assert jc.model_cfg_inherited is False
    assert model == HOST_MODEL
    assert runtime["base_url"] == HOST_BASE_URL


def test_pinned_job_route_is_unchanged(tmp_path, monkeypatch):
    """A pinned job names its own provider + model: no host block rides along."""
    _root, profile = _make_inherited_route_homes(tmp_path)
    pinned = {"id": "job", "name": "job", "prompt": "x",
              "model": "pinned-model", "provider": "deepseek"}
    token, reset = _activate(monkeypatch, profile)
    try:
        jc = _load_cron_job_config(pinned, "job", "job")
    finally:
        reset(token)
    assert jc.model_cfg_inherited is False
    assert _cron_route_request(pinned, jc, jc.cfg) == {
        "requested": "deepseek", "target_model": "pinned-model"}
