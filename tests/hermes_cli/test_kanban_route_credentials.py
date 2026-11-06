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


def _write_profile(home: Path, name: str, dotenv: str = "") -> Path:
    profile = home / "profiles" / name
    profile.mkdir(parents=True, exist_ok=True)
    (profile / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    if dotenv:
        (profile / ".env").write_text(dotenv, encoding="utf-8")
    return profile


@pytest.fixture
def fleet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """``unkeyed`` has no credential; ``keyed`` has one for OpenRouter in its own ``.env``."""
    home = tmp_path / ".hermes"
    home.mkdir()
    _write_profile(home, "unkeyed")
    _write_profile(home, "keyed", "OPENROUTER_API_KEY=fixture-key-not-real\n")
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
    for provider in ("xai-oauth", "custom:house", "not-a-real-provider", "", None):
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
