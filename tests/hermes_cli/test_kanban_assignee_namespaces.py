"""Assignee namespaces: whose ``default`` is it? (proven shared-board race)

``default`` is INSTALL-RELATIVE — it means "the top-level agent of the install
doing the claiming", and ``profiles.profile_exists("default")`` answers True in
EVERY install. So on a board two installs sweep, a ``default`` card was spawnable
in both and ran in whichever dispatcher ticked first (probe ``t_45f8aaac`` ran
here instead of in the owning install). Same defect family as the
``HERMES_KANBAN_DB`` pin work: a name resolved without the ownership context it
depends on.

The fix puts the disambiguation in the assignee itself. A profile's identity is
the PAIR ``(namespace, name)``, never the bare name alone:

* ``ns:name`` — canonical, explicit; claimed only by the install owning ``ns``.
* bare name — resolves within the BOARD's declared namespace, so bare
  ``default`` on a board whose namespace is the other install means THAT
  install's top-level agent, and this one refuses it.
* bare name on a board that declares no namespace — unchanged name
  partitioning (the compatibility guarantee), EXCEPT an install-relative name
  (``default``), which is ambiguous by construction and therefore refused.

Every refusal is VISIBLE: a ``skipped_namespace`` bucket on ``DispatchResult``,
a ``dispatch_skipped`` task event carrying a remedy, and a warning on the tick.
A silent skip is the bug class this feature exists to prevent, so the regression
guard here asserts the durable record, not just the boolean.

Test mechanics: one temp ``HERMES_HOME`` per test holds the board(s) and the
profile directories; "which install is ticking" is chosen by
``HERMES_KANBAN_NAMESPACE``, exactly as deploy sets it via ``kanban.namespace``.
Profile *existence* is necessarily shared in a single-home test, so cases that
turn on "this namespace has no such profile" use a profile name that is absent
from the fixture's profile list.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from types import SimpleNamespace

import pytest

from hermes_cli.kanban_db_dispatch import (
    REASON_ASSIGNEE_INVALID,
    REASON_NAMESPACE_MISMATCH,
    REASON_NAMESPACE_UNDECLARED,
    REASON_NAMESPACE_UNRESOLVED,
    REASON_PROFILE_MISSING,
    assignee_profile,
    install_namespaces,
    resolve_assignee,
    split_assignee,
)


def _fake_spawn(*args, **kwargs):
    return 12345


@pytest.fixture()
def fleet(monkeypatch):
    """Build a fresh install home and hand back the imported modules.

    ``fleet(profiles=(...), boards={"slug": namespace-or-None})``. The returned
    namespace has ``.as_install(ns)`` to choose which install's dispatcher the
    next tick belongs to.
    """
    def build(*, profiles=("default", "admin"), boards=(("shared", None),)):
        home = tempfile.mkdtemp(prefix="kanban_assignee_ns_test_")
        for prof in profiles:
            os.makedirs(os.path.join(home, "profiles", prof), exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", home)
        monkeypatch.setenv("HERMES_KANBAN_NAMESPACE", "em")
        for mod in list(sys.modules):
            if (mod.startswith("hermes_cli") or mod.startswith("hermes_state")
                    or mod == "hermes_constants"):
                del sys.modules[mod]
        from hermes_cli import kanban_db as kb
        from hermes_cli import kanban_db_connect as kbc
        from hermes_cli import kanban_db_dispatch as kbd
        for slug, namespace in boards.items():
            kb.create_board(slug=slug, name=slug, namespace=namespace)
        # The install namespace is resolved once per process and memoised, so a
        # fresh module (above) is a fresh startup. Clearing explicitly keeps a
        # test that runs after an in-process env swap honest either way.
        kbd._reset_namespace_cache()
        env = SimpleNamespace(
            home=home, kb=kb, kbc=kbc, kbd=kbd, boards=dict(boards),
            monkeypatch=monkeypatch,
        )

        def as_install(ns):
            """Tick as though THIS install's dispatcher were the one running.

            ``HERMES_KANBAN_NAMESPACE`` is part of the namespace memo key, so the
            swap is picked up anyway; the documented reset hook is called too, so
            the tests exercise it (and so a config-derived namespace, which is
            only read once per process, would be re-read).
            """
            monkeypatch.setenv("HERMES_KANBAN_NAMESPACE", ns)
            kbd._reset_namespace_cache()

        env.as_install = as_install
        return env

    return build


def _tick(env, board, **kwargs):
    with env.kbc.connect_closing(board=board) as conn:
        return env.kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, board=board, **kwargs,
        )


def _create(env, board, *, title, assignee):
    with env.kbc.connect_closing(board=board) as conn:
        return env.kb.create_task(conn, title=title, assignee=assignee)


def _task(env, board, task_id):
    with env.kbc.connect_closing(board=board) as conn:
        return conn.execute(
            "SELECT status, assignee, claim_lock FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()


def _events(env, board, task_id, kind):
    with env.kbc.connect_closing(board=board) as conn:
        return [
            json.loads(row["payload"] or "{}")
            for row in conn.execute(
                "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? "
                "ORDER BY id", (task_id, kind),
            )
        ]


def _run_profiles(env, board, task_id):
    with env.kbc.connect_closing(board=board) as conn:
        return [
            row["profile"] for row in conn.execute(
                "SELECT profile FROM task_runs WHERE task_id = ? ORDER BY id", (task_id,),
            )
        ]


def _skip_reasons(result):
    return {tid: reason for (tid, _who, reason) in result.skipped_namespace}


# ---------------------------------------------------------------------------
# The race itself
# ---------------------------------------------------------------------------


def test_bare_default_claimed_only_by_the_board_namespace_owner(fleet):
    """A bare ``default`` on a board whose namespace is the other install is
    claimed by THAT install and refused here — with a durable record."""
    env = fleet(boards={"shared": "yummi"})
    task_id = _create(env, "shared", title="her top-level", assignee="default")

    env.as_install("em")
    res = _tick(env, "shared")
    assert res.spawned == []
    assert _skip_reasons(res) == {task_id: REASON_NAMESPACE_MISMATCH}
    assert task_id not in res.skipped_nonspawnable
    assert _task(env, "shared", task_id)["status"] == "ready"
    # Visible, not silent: a durable event the dashboard can show.
    skips = _events(env, "shared", task_id, "dispatch_skipped")
    assert [s["reason"] for s in skips] == [REASON_NAMESPACE_MISMATCH]
    assert skips[0]["board_namespace"] == "yummi"
    assert skips[0]["install_namespace"] == "em"
    assert "yummi" in skips[0]["remedy"]

    # Her dispatcher (same DB, her namespace) claims it, as plain ``default``.
    env.as_install("yummi")
    res = _tick(env, "shared")
    assert [s[0] for s in res.spawned] == [task_id]
    assert res.skipped_namespace == []
    assert _task(env, "shared", task_id)["status"] == "running"
    # The run row names the profile that ran, not the routing string.
    assert _run_profiles(env, "shared", task_id) == ["default"]


def test_explicit_namespaced_default_claimed_only_by_matching_install(fleet):
    """``yummi:default`` is the canonical form: it works on ANY board, declared
    or not, and only the owning install may claim it."""
    env = fleet(boards={"shared": None})
    task_id = _create(env, "shared", title="explicit", assignee="yummi:default")

    env.as_install("em")
    res = _tick(env, "shared")
    assert res.spawned == []
    assert _skip_reasons(res) == {task_id: REASON_NAMESPACE_MISMATCH}
    assert _events(env, "shared", task_id, "dispatch_skipped")[0]["reason"] == (
        REASON_NAMESPACE_MISMATCH
    )

    env.as_install("yummi")
    res = _tick(env, "shared")
    assert [s[0] for s in res.spawned] == [task_id]
    # Namespace stripped for the spawn: the profile is ``default``.
    assert _run_profiles(env, "shared", task_id) == ["default"]


def test_namespace_token_parsed_case_insensitively(fleet):
    """The human form ``Yummi:default`` means ``yummi:default``."""
    assert split_assignee("Yummi:default") == ("yummi", "default")
    assert split_assignee("  EM : admin ") == ("em", "admin")
    assert split_assignee("default") == (None, "default")
    assert split_assignee(None) == (None, "")
    assert assignee_profile("Yummi:default") == "default"

    env = fleet(boards={"shared": None})
    task_id = _create(env, "shared", title="case", assignee="Yummi:default")
    env.as_install("yummi")
    assert [s[0] for s in _tick(env, "shared").spawned] == [task_id]


# ---------------------------------------------------------------------------
# No regression on our own boards
# ---------------------------------------------------------------------------


def test_bare_default_on_our_own_board_unchanged(fleet):
    """``default`` on a board this install owns keeps working exactly as before.
    A large amount of live work depends on it."""
    env = fleet(boards={"em-admin": "em"})
    task_id = _create(env, "em-admin", title="ours", assignee="default")

    env.as_install("em")
    res = _tick(env, "em-admin")
    assert [s[0] for s in res.spawned] == [task_id]
    assert res.skipped_namespace == []
    assert _run_profiles(env, "em-admin", task_id) == ["default"]
    # Another install sweeping the same board does not take it.
    other = _create(env, "em-admin", title="ours too", assignee="default")
    env.as_install("yummi")
    res = _tick(env, "em-admin")
    assert other not in [s[0] for s in res.spawned]
    assert _skip_reasons(res)[other] == REASON_NAMESPACE_MISMATCH


def test_default_board_needs_no_declaration(fleet):
    """The ``default`` board's DB is ``<this install root>/kanban.db`` by
    construction, so no other install can reach it and no declaration is needed
    — that is a property of the slug→path mapping, not a symlink inference."""
    env = fleet(boards={"default": None})
    task_id = _create(env, "default", title="home board", assignee="default")
    for ns in ("em", "yummi", "whoever"):
        env.as_install(ns)
        res = _tick(env, "default", dry_run=True)
        assert task_id in [s[0] for s in res.spawned], ns
        assert res.skipped_namespace == [], ns


# ---------------------------------------------------------------------------
# No new silent skip — the regression guard
# ---------------------------------------------------------------------------


def test_bare_default_on_undeclared_board_refused_VISIBLY(fleet):
    """An install-relative bare name on a board that declares no namespace is
    ambiguous by construction: refused, never raced — and the refusal leaves a
    trace (log line, ``skipped_namespace`` bucket, one durable event)."""
    env = fleet(boards={"shared": None})
    task_id = _create(env, "shared", title="ambiguous", assignee="default")

    for ns in ("em", "yummi"):
        env.as_install(ns)
        res = _tick(env, "shared")
        assert res.spawned == [], ns
        assert _skip_reasons(res) == {task_id: REASON_NAMESPACE_UNDECLARED}, ns
        # NEVER the invisible bucket: that is the bug class being fixed.
        assert task_id not in res.skipped_nonspawnable, ns
        assert _task(env, "shared", task_id)["status"] == "ready", ns

    skips = _events(env, "shared", task_id, "dispatch_skipped")
    assert [s["reason"] for s in skips] == [REASON_NAMESPACE_UNDECLARED]
    assert skips[0]["board"] == "shared"
    assert "set-namespace" in skips[0]["remedy"]
    # One event per (task, assignee, reason) — not one per tick.
    env.as_install("em")
    _tick(env, "shared")
    assert len(_events(env, "shared", task_id, "dispatch_skipped")) == 1


def test_unresolvable_install_namespace_refused_visibly(fleet):
    """This install cannot prove which namespace it is: refuse rather than
    guess, and say so."""
    env = fleet(boards={"shared": "yummi"})
    task_id = _create(env, "shared", title="who am i", assignee="default")
    env.monkeypatch.setenv("HERMES_KANBAN_NAMESPACE", "not a token!")
    assert install_namespaces() == (None, frozenset())

    res = _tick(env, "shared")
    assert res.spawned == []
    assert _skip_reasons(res) == {task_id: REASON_NAMESPACE_UNRESOLVED}
    assert _events(env, "shared", task_id, "dispatch_skipped")[0]["reason"] == (
        REASON_NAMESPACE_UNRESOLVED
    )


def test_profile_missing_in_namespace_refused_visibly(fleet):
    """The name resolved through a namespace and this install has no such
    profile — operator-actionable (place it explicitly), so it must be visible
    rather than joining the invisible control-plane bucket."""
    env = fleet(profiles=("default",), boards={"em-admin": "em"})
    task_id = _create(env, "em-admin", title="ghost", assignee="quantconnect-dev-test")

    env.as_install("em")
    res = _tick(env, "em-admin")
    assert res.spawned == []
    assert _skip_reasons(res) == {task_id: REASON_PROFILE_MISSING}
    assert task_id not in res.skipped_nonspawnable
    event = _events(env, "em-admin", task_id, "dispatch_skipped")[0]
    assert event["reason"] == REASON_PROFILE_MISSING
    assert "quantconnect-dev-test" in event["remedy"]


def test_empty_assignee_is_refused_visibly(fleet):
    """``ns:`` with no profile behind it is a routing mistake, not a lane."""
    env = fleet(boards={"shared": None})
    task_id = _create(env, "shared", title="no profile", assignee="em:")
    env.as_install("em")
    res = _tick(env, "shared")
    assert _skip_reasons(res) == {task_id: REASON_ASSIGNEE_INVALID}


# ---------------------------------------------------------------------------
# Compatibility: bare unique names keep partitioning by name
# ---------------------------------------------------------------------------


def test_bare_unique_names_unchanged_on_undeclared_board(fleet):
    """No namespace declared ⇒ behaviour identical to today: the profile set
    decides, and bare unique names are not namespace-gated at all (that path is
    only reachable once a board declares a namespace). In one home both installs
    see the same profile directories, so this asserts the gate does not fire in
    either direction rather than partition ownership — ownership by name is what
    an undeclared board still relies on."""
    env = fleet(profiles=("default", "admin"), boards={"shared": None})
    ours = _create(env, "shared", title="ours", assignee="admin")
    # A name this install does not have takes today's immutable path: the
    # non-spawnable bucket, unchanged — deliberately NOT a namespace refusal,
    # because the board declares no namespace for the name to be foreign to.
    absent = _create(env, "shared", title="absent", assignee="her-bot")

    env.as_install("em")
    res = _tick(env, "shared")
    assert [s[0] for s in res.spawned] == [ours]
    assert res.skipped_namespace == []
    assert absent in res.skipped_nonspawnable

    env.as_install("yummi")
    other = _create(env, "shared", title="hers", assignee="admin")
    res = _tick(env, "shared")
    assert [s[0] for s in res.spawned] == [other]
    assert res.skipped_namespace == []


def test_bare_name_resolves_through_the_board_namespace(fleet):
    """On a board that declares a namespace, EVERY bare name resolves in it —
    not just install-relative ones (two identically named profiles in two
    namespaces would otherwise collide exactly like ``default`` did)."""
    env = fleet(profiles=("default", "admin"), boards={"shared": "yummi"})
    task_id = _create(env, "shared", title="bare", assignee="admin")

    env.as_install("em")
    res = _tick(env, "shared")
    assert res.spawned == []
    assert _skip_reasons(res) == {task_id: REASON_NAMESPACE_MISMATCH}

    env.as_install("yummi")
    assert [s[0] for s in _tick(env, "shared").spawned] == [task_id]


# ---------------------------------------------------------------------------
# Explicit cross-namespace placement
# ---------------------------------------------------------------------------


def test_explicit_override_places_work_across_namespaces(fleet):
    """``em:admin`` on her board runs here. The card keeps the assignee it was
    given; only the run row records the resolved profile."""
    env = fleet(profiles=("default", "admin"), boards={"shared": "yummi"})
    task_id = _create(env, "shared", title="borrowed", assignee="em:admin")

    env.as_install("em")
    res = _tick(env, "shared")
    assert [s[0] for s in res.spawned] == [task_id]
    assert _run_profiles(env, "shared", task_id) == ["admin"]
    assert _task(env, "shared", task_id)["assignee"] == "em:admin"


# ---------------------------------------------------------------------------
# Fallbacks and telemetry
# ---------------------------------------------------------------------------


def test_default_assignee_fallback_not_written_onto_a_foreign_board(fleet):
    """``kanban.default_assignee`` must not stamp an unresolvable
    install-relative name onto another install's card — that would mutate their
    board and claim a routing decision this install has no standing to make."""
    env = fleet(boards={"shared": "yummi"})
    with env.kbc.connect_closing(board="shared") as conn:
        task_id = env.kb.create_task(conn, title="unassigned", assignee=None)

    env.as_install("em")
    res = _tick(env, "shared", default_assignee="default")
    assert res.auto_assigned_default == []
    assert _skip_reasons(res) == {task_id: REASON_NAMESPACE_MISMATCH}
    assert _task(env, "shared", task_id)["assignee"] is None
    assert _events(env, "shared", task_id, "assigned") == []
    assert _events(env, "shared", task_id, "dispatch_skipped")[0]["reason"] == (
        REASON_NAMESPACE_MISMATCH
    )

    # On our own board the same fallback still assigns and spawns as before.
    env = fleet(boards={"em-admin": "em"})
    with env.kbc.connect_closing(board="em-admin") as conn:
        task_id = env.kb.create_task(conn, title="unassigned", assignee=None)
    env.as_install("em")
    res = _tick(env, "em-admin", default_assignee="default")
    assert res.auto_assigned_default == [task_id]
    assert _task(env, "em-admin", task_id)["assignee"] == "default"


def test_has_spawnable_is_board_aware(fleet):
    """Another install's queue must not be reported as spawnable work here, or
    a healthy board reads as stuck."""
    env = fleet(boards={"shared": "yummi", "em-admin": "em"})
    _create(env, "shared", title="hers", assignee="default")
    _create(env, "em-admin", title="ours", assignee="default")

    env.as_install("em")
    with env.kbc.connect_closing(board="shared") as conn:
        assert env.kbd.has_spawnable_ready(conn, board="shared") is False
    with env.kbc.connect_closing(board="em-admin") as conn:
        assert env.kbd.has_spawnable_ready(conn, board="em-admin") is True

    env.as_install("yummi")
    with env.kbc.connect_closing(board="shared") as conn:
        assert env.kbd.has_spawnable_ready(conn, board="shared") is True


def test_per_profile_cap_keyed_on_resolved_profile(fleet):
    """``em:default`` and a bare ``default`` in this namespace share one budget
    instead of pretending to be two profiles."""
    env = fleet(boards={"em-admin": "em"})
    _create(env, "em-admin", title="bare", assignee="default")
    _create(env, "em-admin", title="explicit", assignee="em:default")

    env.as_install("em")
    res = _tick(env, "em-admin", dry_run=True, max_in_progress_per_profile=1)
    assert len(res.spawned) == 1
    assert len(res.skipped_per_profile_capped) == 1
    assert res.skipped_per_profile_capped[0][2] == 1


def test_resolve_assignee_is_pure_and_inspectable(fleet):
    """The resolution is a value, not a side effect — the whole gate is one
    question (``claimable``) plus a remedy for the human."""
    env = fleet(boards={"shared": "yummi", "em-admin": "em"})

    env.as_install("em")
    assert resolve_assignee("default", "em-admin").claimable
    own = resolve_assignee("default", "em-admin")
    assert own.profile == "default" and own.board_namespace == "em"
    foreign = resolve_assignee("default", "shared")
    assert not foreign.claimable and foreign.reason == REASON_NAMESPACE_MISMATCH
    override = resolve_assignee("yummi:admin", "em-admin")
    assert not override.claimable and override.reason == REASON_NAMESPACE_MISMATCH
    assert resolve_assignee("em:admin", "shared").profile == "admin"
    assert resolve_assignee("nobody:admin", "em-admin").reason == REASON_NAMESPACE_MISMATCH
    undeclared = resolve_assignee("default", "default")
    assert undeclared.claimable and undeclared.board_namespace is None


def test_display_name_never_derives_the_namespace(fleet):
    """The token lives in config and is authoritative; ``display_name`` is
    cosmetic and user-editable, so editing it must not re-scope anything."""
    env = fleet(boards={"shared": "em"})
    os.makedirs(os.path.join(env.home, "profiles", "default"), exist_ok=True)
    with open(os.path.join(env.home, "profiles", "default", "profile.yaml"), "w") as fh:
        fh.write("display_name: Yummi\n")
    with open(os.path.join(env.home, "config.yaml"), "w") as fh:
        fh.write("kanban:\n  namespace: em\n")
    env.monkeypatch.delenv("HERMES_KANBAN_NAMESPACE", raising=False)

    assert install_namespaces() == ("em", frozenset({"em"}))
    # The mismatch between display_name and token is expected and harmless.
    assert resolve_assignee("default", "shared").claimable
    assert not resolve_assignee("yummi:default", "shared").claimable


def test_namespace_aliases_accept_configured_tokens(fleet):
    """Comma-separated tokens are accepted as aliases (the OS-user fallback is
    a real case: an install configured as ``em`` may still be swept while a
    stale token is in play) while ONE canonical token is named in messages."""
    env = fleet(boards={"shared": "deprecated"})
    env.monkeypatch.setenv("HERMES_KANBAN_NAMESPACE", "Em, deprecated")
    assert install_namespaces() == ("em", frozenset({"em", "deprecated"}))
    assert resolve_assignee("default", "shared").claimable
    assert resolve_assignee("Deprecated:default", "shared").claimable
