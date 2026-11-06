"""Q3 of the dispatch preflight: a pin the assignee profile cannot serve is refused, not spawned.

Criterion this file pins (t_c7d92b03), one test per acceptance condition:

* a card pinned to a provider the assignee's own home holds no credential for is refused **before
  a worker is spawned**, and the refusal names the provider;
* a card pinned to a provider that home *can* serve still dispatches;
* a pin whose provider shape cannot be decided from the home (OAuth, custom, unknown) is admitted —
  a refusal has to be provable;
* a probe that fails is admitted too: a broken checker must never park a card;
* the probe reads the assignee's home, not the dispatcher's environment — the whole bug was that a
  pin resolved in the wrong place.

Everything runs against an isolated ``HERMES_HOME`` with two profiles: ``unkeyed`` holds no
credential at all, ``keyed`` holds an OpenRouter key in its own ``.env``. No live board, profile or
credential is touched, and the probe child reads the fixture home only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_route_credentials as rrc
from hermes_cli.kanban_db_connect import connect

PIN_MODEL = "deepseek/deepseek-v4.1-flash-1017"


def _write_profile(home: Path, name: str, dotenv: str = "", config: str = "model: {}\n") -> Path:
    profile = home / "profiles" / name
    profile.mkdir(parents=True, exist_ok=True)
    (profile / "config.yaml").write_text(config, encoding="utf-8")
    if dotenv:
        (profile / ".env").write_text(dotenv, encoding="utf-8")
    return profile


@pytest.fixture
def fleet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """``unkeyed`` has no credential; ``keyed`` has one for OpenRouter in its own ``.env``."""
    home = tmp_path / ".hermes"
    home.mkdir()
    _write_profile(home, "unkeyed")
    _write_profile(home, "keyed", "OPENROUTER_API_KEY=sk-or-v1-fixture-key-not-real\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def no_profile_gate(monkeypatch: pytest.MonkeyPatch):
    """Every assignee counts as installed; the credential gate is what is under test."""
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)


@pytest.fixture
def verdict(monkeypatch: pytest.MonkeyPatch):
    """Answer the probe without a child process, for the dispatch-level cases."""

    def _set(present: bool, kind: str = "api_key"):
        def _probe(profile: str, provider: str, *, timeout: int = 0, runner=None) -> rrc.RouteProbe:
            return rrc.RouteProbe(profile, provider, kind, present,
                                  ("OPENROUTER_API_KEY",) if kind == "api_key" else ())

        monkeypatch.setattr(rrc, "probe", _probe)

    return _set


def _dispatch(conn, spawn_fn):
    return kbd.dispatch_once(conn, spawn_fn=spawn_fn, failure_limit=3)


# --- the verdict -------------------------------------------------------------------------------

def test_a_pin_the_profile_cannot_serve_is_refused_and_names_the_provider(fleet, no_profile_gate, verdict):
    verdict(False)
    spawned: list[str] = []

    def spawn(task, workspace, board=None):
        spawned.append(task.id)
        return 4242

    with connect() as conn:
        tid = kb.create_task(conn, title="pinned", assignee="unkeyed",
                             model_override=PIN_MODEL, provider_override="openrouter")
        res = _dispatch(conn, spawn)
        task = kb.get_task(conn, tid)
        events = [e for e in kb.list_events(conn, tid) if e.kind == "routing_refused"]
        run = kb.latest_run(conn, tid)

    assert spawned == [], "a card whose worker cannot start must not be spawned"
    assert [entry[0] for entry in res.route_refused] == [tid]
    assert task is not None and task.status == "blocked" and task.block_kind == "capability"
    # The provider is named on the board, not only in this tick's return value.
    assert events and events[0].payload["question"] == "Q3"
    assert events[0].payload["provider"] == "openrouter"
    assert events[0].payload["profile"] == "unkeyed"
    assert "openrouter" in (run.summary or "")


def test_a_pin_the_profile_can_serve_still_dispatches(fleet, no_profile_gate, verdict):
    verdict(True)
    spawned: list[str] = []

    def spawn(task, workspace, board=None):
        spawned.append(task.id)
        return 4242

    with connect() as conn:
        tid = kb.create_task(conn, title="pinned", assignee="keyed",
                             model_override=PIN_MODEL, provider_override="openrouter")
        res = _dispatch(conn, spawn)
        task = kb.get_task(conn, tid)

    assert spawned == [tid], "a servable pin must dispatch"
    assert res.route_refused == []
    assert task is not None and task.status == "running" and task.model_override == PIN_MODEL


def test_an_unpinned_card_never_pays_for_the_gate(fleet, no_profile_gate, monkeypatch):
    def _explode(*args, **kwargs):
        raise AssertionError("an unpinned card must not be probed")

    monkeypatch.setattr(rrc, "probe", _explode)
    spawned: list[str] = []

    def spawn(task, workspace, board=None):
        spawned.append(task.id)
        return 4242

    with connect() as conn:
        tid = kb.create_task(conn, title="plain", assignee="unkeyed")
        res = _dispatch(conn, spawn)

    assert spawned == [tid] and res.route_refused == []


def test_an_undecidable_provider_shape_is_admitted(fleet, no_profile_gate):
    # OAuth grants can arrive from the global-root pool, custom/local endpoints may need no key:
    # neither is provable from the home, so neither may be refused.
    assert rrc.provider_shape("openrouter")[0] == "api_key"
    for provider in ("xai-oauth", "custom:house", "not-a-real-provider", "", None,
                     # anthropic's token also resolves from the Claude Code / Hermes OAuth store,
                     # and these local servers are served a no-auth placeholder: "nothing in
                     # .env/auth.json" is not proof a worker would die.
                     "anthropic", "lmstudio", "actual"):
        assert rrc.provider_shape(provider)[0] == "undecidable", provider
        assert rrc.route_credential_refusal("unkeyed", provider) is None

    spawned: list[str] = []

    def spawn(task, workspace, board=None):
        spawned.append(task.id)
        return 4242

    with connect() as conn:
        tid = kb.create_task(conn, title="oauth pin", assignee="unkeyed",
                             model_override="grok-4.7", provider_override="xai-oauth")
        res = _dispatch(conn, spawn)

    assert spawned == [tid] and res.route_refused == []


def test_a_failed_probe_admits_the_card(fleet, no_profile_gate, monkeypatch):
    def _explode(profile, provider, *, timeout=0, runner=None):
        raise RuntimeError("probe interpreter missing")

    monkeypatch.setattr(rrc, "probe", _explode)
    assert rrc.route_credential_refusal("unkeyed", "openrouter") is None

    spawned: list[str] = []

    def spawn(task, workspace, board=None):
        spawned.append(task.id)
        return 4242

    with connect() as conn:
        tid = kb.create_task(conn, title="pinned", assignee="unkeyed",
                             model_override=PIN_MODEL, provider_override="openrouter")
        res = _dispatch(conn, spawn)

    assert spawned == [tid] and res.route_refused == []


def test_the_refusal_names_the_remedy_and_the_env_var(fleet):
    refusal = rrc.route_credential_refusal("unkeyed", "openrouter", PIN_MODEL,
                                           probe_fn=lambda profile, provider, *, timeout=0:
                                           rrc.RouteProbe(profile, provider, "api_key", False,
                                                          ("OPENROUTER_API_KEY",)))
    assert refusal is not None
    text = refusal.as_text()
    assert rrc.ROUTE_REFUSAL_MARKER in text
    assert "OPENROUTER_API_KEY" in text and "openrouter" in text and "unkeyed" in text
    assert PIN_MODEL in text, "the pinned model is part of what was refused"
    # A model-less pin still refuses; the text just has no model clause.
    plain = rrc.route_credential_refusal("unkeyed", "openrouter",
                                         probe_fn=lambda profile, provider, *, timeout=0:
                                         rrc.RouteProbe(profile, provider, "api_key", False))
    assert plain is not None and PIN_MODEL not in plain.as_text()


# --- the probe reads the assignee's home -------------------------------------------------------

def test_probe_payload_reads_this_home_only(fleet, monkeypatch):
    from hermes_cli.profiles import resolve_profile_env
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    def payload(profile: str) -> dict:
        token = set_hermes_home_override(resolve_profile_env(profile))
        try:
            return rrc.probe_payload("openrouter")
        finally:
            reset_hermes_home_override(token)

    keyed = payload("keyed")
    unkeyed = payload("unkeyed")
    assert keyed["present"] is True and "env:OPENROUTER_API_KEY" in keyed["sources"]
    assert unkeyed["present"] is False and unkeyed["sources"] == []
    assert "fixture-key-not-real" not in json.dumps(unkeyed), "a probe never emits the credential"


def test_the_probe_child_answers_for_the_assignee_home(fleet):
    """End-to-end through the real child process: same verdicts, no dispatcher environment."""
    keyed = rrc.probe("keyed", "openrouter")
    unkeyed = rrc.probe("unkeyed", "openrouter")

    assert keyed.kind == "api_key" and keyed.present is True, keyed.detail
    assert unkeyed.kind == "api_key" and unkeyed.present is False, unkeyed.detail
    assert unkeyed.decisive is True


# --- the probe's "present" matches the runtime credential ladder ---------------------------------
#
# Round-1 review of t_c7d92b03: the first probe hand-rolled a three-source ladder (registry env
# vars, the pool, ``model.api_key``) that did not match ``auth._resolve_api_key_provider_secret``,
# so it admitted a profile with only an inline ``model.api_key`` (runtime reads none) and refused a
# Desktop-saved ``model.key_env`` pointer and a keyless local server. One regression test per
# direction, each checked against the resolver the worker itself runs.

from contextlib import contextmanager


@contextmanager
def _profile_home(profile: str):
    """Run in *profile*'s own home, the way a dispatched worker would."""
    from hermes_cli.profiles import resolve_profile_env
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(resolve_profile_env(profile))
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _payload_for(profile: str, provider: str) -> dict:
    with _profile_home(profile):
        return rrc.probe_payload(provider)


def test_model_key_env_pointer_is_a_credential(fleet, monkeypatch):
    """A Desktop-saved ``model.key_env`` pointer is read by runtime, so the probe must see it."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    _write_profile(
        fleet, "pointed",
        "HERMES_CUSTOM_DEEPSEEK_API_KEY=sk-deepseek-fixture\n",
        "model:\n  default: deepseek-chat\n  provider: deepseek\n"
        "  key_env: HERMES_CUSTOM_DEEPSEEK_API_KEY\n",
    )

    payload = _payload_for("pointed", "deepseek")
    assert payload["present"] is True, payload
    assert "env:HERMES_CUSTOM_DEEPSEEK_API_KEY" in payload["sources"]
    assert rrc.route_credential_refusal("pointed", "deepseek") is None

    # The runtime agrees for the same home; the probe is not asserting something it does not do.
    with _profile_home("pointed"):
        from hermes_cli.auth import resolve_api_key_provider_credentials
        creds = resolve_api_key_provider_credentials("deepseek")
    assert creds["source"] == "HERMES_CUSTOM_DEEPSEEK_API_KEY" and creds["api_key"]


def test_inline_model_api_key_is_not_a_credential(fleet, monkeypatch):
    """``model.api_key`` is not on the registry api_key ladder: counting it admitted a dead route."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    _write_profile(
        fleet, "inline", "",
        "model:\n  default: deepseek-chat\n  provider: deepseek\n  api_key: sk-deepseek-fixture\n",
    )

    payload = _payload_for("inline", "deepseek")
    assert payload["present"] is False, payload
    assert "config.yaml" not in payload["sources"]
    assert rrc.route_credential_refusal("inline", "deepseek", "deepseek-chat") is not None

    # ...and runtime reads no key from that home, which is exactly the crash this gate prevents.
    with _profile_home("inline"):
        from hermes_cli.auth import resolve_api_key_provider_credentials
        creds = resolve_api_key_provider_credentials("deepseek")
    assert creds["api_key"] == "" and creds["source"] == "default"


def test_keyless_local_provider_is_admitted(fleet, monkeypatch):
    """LM Studio is served a no-auth placeholder, so an empty home is not a dead route."""
    monkeypatch.delenv("LM_API_KEY", raising=False)

    assert rrc.provider_shape("lmstudio")[0] == "undecidable"
    assert rrc.route_credential_refusal("unkeyed", "lmstudio") is None

    with _profile_home("unkeyed"):
        from hermes_cli.auth import LMSTUDIO_NOAUTH_PLACEHOLDER, resolve_api_key_provider_credentials
        creds = resolve_api_key_provider_credentials("lmstudio")
    assert creds["api_key"] == LMSTUDIO_NOAUTH_PLACEHOLDER


def test_a_malformed_key_does_not_count_as_a_credential(fleet, monkeypatch):
    """Runtime skips a key whose shape is wrong (#93593); the probe must not call it present."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    _write_profile(fleet, "malformed", "OPENROUTER_API_KEY=not-an-openrouter-key\n")

    payload = _payload_for("malformed", "openrouter")
    assert payload["present"] is False, payload
    assert rrc.route_credential_refusal("malformed", "openrouter") is not None


def test_a_legacy_openai_key_shaped_like_openrouter_counts(fleet, monkeypatch):
    """``OPENAI_API_KEY`` is a legacy OpenRouter home; runtime accepts it when it is sk-or-shaped."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    _write_profile(fleet, "legacy", "OPENAI_API_KEY=sk-or-v1-legacy-fixture\n")

    payload = _payload_for("legacy", "openrouter")
    assert payload["present"] is True, payload
    assert "env:OPENAI_API_KEY" in payload["sources"]
    assert rrc.route_credential_refusal("legacy", "openrouter") is None


def test_an_env_backed_pool_row_without_its_key_is_not_a_credential(fleet, monkeypatch):
    """An ``env:`` pool row is persisted WITHOUT its secret and re-hydrated from that env var.

    This is the live researcher shape: auth.json holds ``env:OPENROUTER_API_KEY`` but the home
    resolves no such var, so the row is an empty shell and the worker would die.
    """
    import hermes_cli.auth as auth

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(auth, "read_credential_pool",
                        lambda provider=None: [{"id": "abc123", "auth_type": "api_key",
                                                "source": "env:OPENROUTER_API_KEY"}])
    payload = _payload_for("unkeyed", "openrouter")
    assert payload["present"] is False, payload
    assert rrc.route_credential_refusal("unkeyed", "openrouter") is not None


def test_an_env_backed_pool_row_is_a_credential_when_its_key_resolves(fleet, monkeypatch):
    import hermes_cli.auth as auth

    monkeypatch.setattr(auth, "read_credential_pool",
                        lambda provider=None: [{"id": "abc123", "auth_type": "api_key",
                                                "source": "env:OPENROUTER_API_KEY"}])
    _write_profile(fleet, "pooled", "OPENROUTER_API_KEY=sk-or-v1-pooled-fixture\n")

    payload = _payload_for("pooled", "openrouter")
    assert payload["present"] is True, payload
    assert "env:OPENROUTER_API_KEY" in payload["sources"]
    assert rrc.route_credential_refusal("pooled", "openrouter") is None
