"""Housekeeping chores that read a profile's home, config or credentials run under the OWNING
profile's runtime scope on a multiplexed gateway.

The housekeeping thread has no turn on the stack, so nothing bound a profile for it: under
``gateway.multiplex_profiles`` the skills-sync pulls resolved Nous credentials through the
fail-closed reader and logged ``no profile secret scope on a multiplexed call`` four times per
hourly tick, per chore, while the launch profile's home/credentials leaked into every served
profile's pull. The MCP config reconciler already iterated the served profiles under
``_profile_runtime_scope``; the sync/curator ticks now ride the same iteration.
"""

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import gateway.run as gateway_run


class _Ticks:
    """Stop event that lets the housekeeping loop run exactly ``n`` ticks with no sleeping."""

    def __init__(self, n: int):
        self.n, self.left = 0, n

    def is_set(self):
        return self.n >= self.left

    def wait(self, timeout=None):
        self.n += 1
        return True


def _profile(home: Path, base_url: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("model:\n  provider: nous\n", encoding="utf-8")
    (home / ".env").write_text(f"NOUS_INFERENCE_BASE_URL={base_url}\n", encoding="utf-8")
    (home / "auth.json").write_text(json.dumps({"version": 1, "providers": {"nous": {
        "access_token": "x.y.z", "refresh_token": "r", "expires_at": 0,
        "portal_base_url": "https://portal.nousresearch.com", "client_id": "c"}}}), encoding="utf-8")


@pytest.fixture
def two_homes(tmp_path, monkeypatch):
    """Launch home A (= multiplex ``default``) and served named profile B under ``A/profiles/b``."""
    fake_home = tmp_path / "home"
    a = fake_home / ".hermes"
    b = a / "profiles" / "b"
    _profile(a, "https://a.example/v1")
    _profile(b, "https://b.example/v1")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setenv("HERMES_HOME", str(a))
    monkeypatch.delenv("NOUS_INFERENCE_BASE_URL", raising=False)
    # The hermetic conftest pins ``hermes_state.DEFAULT_DB_PATH`` at one sandbox store whenever
    # hermes_state is already imported, and that pin WINS over ``get_hermes_home()`` inside
    # ``_default_db_path()`` — exactly the per-profile resolution these tests exist to prove.
    # Restore the import-time sentinel so an argless ``acquire()`` resolves through the scope.
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    # Disabling the hermetic pin is only safe while the sentinel still resolves INSIDE the sandbox:
    # a resolution that escaped to the real home would have these tests writing the live store.
    resolved = Path(hermes_state._default_db_path())
    assert resolved.is_relative_to(tmp_path), f"unpinned store escaped the sandbox: {resolved}"
    return a, b


def _record_credential_chores(monkeypatch):
    """Replace the three credential-reading chores with recorders of (home, Nous override) they see."""
    import agent.curator as curator
    import tools.skills_sync_client as ssc
    import tools.skills_sync_client_org as sso
    from hermes_cli.auth_nous import _nous_inference_env_override
    from hermes_constants import get_hermes_home

    seen: dict = {"sync": [], "org": [], "curator": []}

    def _rec(key):
        return lambda *a, **k: seen[key].append((get_hermes_home().name, _nous_inference_env_override()))

    monkeypatch.setattr(ssc, "maybe_pull_skills", _rec("sync"))
    monkeypatch.setattr(sso, "maybe_pull_org_skills", _rec("org"))
    monkeypatch.setattr(curator, "maybe_run_curator", _rec("curator"))
    return seen


def _run_60_ticks(runner):
    gateway_run._start_gateway_housekeeping(_Ticks(60), interval=0, runner=runner)


def test_multiplexed_sync_ticks_run_once_per_profile_in_its_own_scope(two_homes, monkeypatch, caplog):
    """Under multiplex every credential-reading chore visits each served profile inside ITS scope:
    A's tick reads A's override, B's reads B's (B never sees A's), and no fail-closed credential
    read fires the ``no profile secret scope`` warning. The ambient home is untouched afterwards."""
    from agent.secret_scope import set_multiplex_active
    from hermes_constants import get_hermes_home

    a, b = two_homes
    seen = _record_credential_chores(monkeypatch)
    set_multiplex_active(True)
    try:
        with caplog.at_level(logging.WARNING):
            _run_60_ticks(SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True)))
    finally:
        set_multiplex_active(False)

    expected = [(a.name, "https://a.example/v1"), (b.name, "https://b.example/v1")]
    assert seen == {"sync": expected, "org": expected, "curator": expected}
    assert not [r for r in caplog.records if "no profile secret scope" in r.getMessage()]
    assert get_hermes_home() == a


def test_multiplexed_auto_archive_tick_sweeps_every_served_profile_store(two_homes, monkeypatch):
    """The auto-archive sweep reaches each served profile's OWN state.db.

    ``acquire()`` resolves through ``get_hermes_home()``, so an unscoped tick archived the
    launch profile's store only — and `hermes serve`/the dashboard defer to the gateway for
    every profile it owns, so a served secondary would have had no archiver at all.
    """
    from agent.secret_scope import set_multiplex_active
    from hermes_state import SessionDB

    a, b = two_homes
    swept: list = []
    monkeypatch.setattr(
        SessionDB, "maybe_auto_archive", lambda self, **kw: swept.append(Path(self.db_path)))
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda *args, **kwargs: {"sessions": {"auto_archive": True, "min_interval_hours": 0}})

    set_multiplex_active(True)
    try:
        _run_60_ticks(SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True)))
    finally:
        set_multiplex_active(False)

    assert swept == [a / "state.db", b / "state.db"]


def test_multiplexed_maintenance_tick_prunes_every_served_profile_store(two_homes, monkeypatch):
    """Prune/VACUUM reaches each served profile's OWN state.db, under its OWN ``sessions:`` config.

    Prune and VACUUM ran once in the gateway constructor against a handle pinned to the launch
    home, so a multiplexed secondary's store was never pruned or vacuumed by anybody — it grew
    without bound while the launch profile's ``retention_days`` decided whether it happened at all.
    Real stores, real config files: nothing here is patched.
    """
    from agent.secret_scope import set_multiplex_active
    from hermes_state import SessionDB

    homes = two_homes
    for home in homes:
        (home / "config.yaml").write_text(
            "model:\n  provider: nous\n"
            "sessions:\n"
            "  auto_prune: true\n"
            "  retention_days: 0\n"
            "  min_interval_hours: 0\n"
            "  vacuum_after_prune: false\n",
            encoding="utf-8")
        db = SessionDB(db_path=home / "state.db")
        db.create_session("old", "cli")
        db.end_session("old", "done")
        db.close()

    set_multiplex_active(True)
    try:
        _run_60_ticks(SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True)))
    finally:
        set_multiplex_active(False)

    for home in homes:
        db = SessionDB(db_path=home / "state.db")
        try:
            assert db.get_session("old") is None, f"{home.name}'s store was never pruned"
        finally:
            db.close()


def test_a_failing_profile_does_not_strand_the_profiles_after_it(two_homes, monkeypatch):
    """One served profile's broken store must not cost every profile after it its maintenance.

    ``_housekeeping_chore`` catches at the tick level only, so the launch store raising in
    ``acquire()`` (reachable: ``_init_session_db`` tolerates a failed primary store and keeps
    running) ended the per-profile loop before B, on every tick. Serve defers each served
    profile's sweep to this loop, so B had no archiver at all.
    """
    import hermes_state_registry as registry
    from agent.secret_scope import set_multiplex_active
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB

    a, b = two_homes
    swept: list = []
    real_acquire = registry.acquire

    def _acquire(*args, **kwargs):
        if get_hermes_home() == a:
            raise OSError("launch store unavailable")
        return real_acquire(*args, **kwargs)

    monkeypatch.setattr(registry, "acquire", _acquire)
    monkeypatch.setattr(
        SessionDB, "maybe_auto_archive", lambda self, **kw: swept.append(Path(self.db_path)))
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda *args, **kwargs: {"sessions": {"auto_archive": True, "min_interval_hours": 0}})

    set_multiplex_active(True)
    try:
        _run_60_ticks(SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True)))
    finally:
        set_multiplex_active(False)

    assert swept == [b / "state.db"]
    assert get_hermes_home() == a


def test_profile_scope_setup_failure_restores_the_callers_home(two_homes, monkeypatch):
    """A profile scope whose secret hydration raises must not leave its home installed.

    The home override was set before hydration and only reset in the ``finally`` around the
    ``yield``, so a raising ``.env`` load left the housekeeping thread (or a turn's context)
    resolving ``get_hermes_home()`` to the failed profile for every later unscoped read.
    """
    from agent.secret_scope import current_secret_scope
    from hermes_constants import get_hermes_home

    a, b = two_homes
    scope_before = current_secret_scope()

    def _boom(home):
        raise OSError(f"cannot read {home}/.env")

    monkeypatch.setattr(gateway_run, "_load_profile_secret_scope", _boom)

    with pytest.raises(OSError):
        with gateway_run._profile_runtime_scope(b):
            pass

    assert get_hermes_home() == a
    assert current_secret_scope() == scope_before


def test_prune_unlinks_transcripts_under_the_configured_sessions_dir(two_homes, tmp_path):
    """``gateway.sessions_dir`` governs the LAUNCH profile's transcripts; others use their own home.

    Hardcoding ``<home>/sessions`` made the prune unlink under a directory nothing writes to, so an
    override left every pruned session's ``.json``/``.jsonl``/``request_dump_*`` orphaned forever.
    """
    from agent.secret_scope import set_multiplex_active
    from hermes_state import SessionDB

    a, b = two_homes
    override = tmp_path / "custom-transcripts"
    override.mkdir()
    for home, transcripts in ((a, override), (b, b / "sessions")):
        (home / "config.yaml").write_text(
            "model:\n  provider: nous\n"
            "sessions:\n"
            "  auto_prune: true\n"
            "  retention_days: 0\n"
            "  min_interval_hours: 0\n"
            "  vacuum_after_prune: false\n",
            encoding="utf-8")
        db = SessionDB(db_path=home / "state.db")
        db.create_session("old", "cli")
        db.end_session("old", "done")
        db.close()
        transcripts.mkdir(parents=True, exist_ok=True)
        (transcripts / "old.jsonl").write_text("{}\n", encoding="utf-8")

    set_multiplex_active(True)
    try:
        _run_60_ticks(SimpleNamespace(config=SimpleNamespace(
            multiplex_profiles=True, sessions_dir=override)))
    finally:
        set_multiplex_active(False)

    assert not (override / "old.jsonl").exists(), "launch profile's configured transcript survived"
    assert not (b / "sessions" / "old.jsonl").exists(), "profile b's transcript survived"


def test_single_profile_sync_ticks_run_once_against_the_process_home(two_homes, monkeypatch):
    """Control: a single-profile gateway (multiplex off) still runs each chore exactly once against
    the process home — the named profile directory on disk is not visited."""
    a, _b = two_homes
    seen = _record_credential_chores(monkeypatch)

    _run_60_ticks(SimpleNamespace(config=SimpleNamespace(multiplex_profiles=False)))

    assert {k: [h for h, _ in v] for k, v in seen.items()} == {
        "sync": [a.name], "org": [a.name], "curator": [a.name]}


def test_multiplexed_plugin_update_check_visits_every_served_profiles_plugins(two_homes, monkeypatch):
    """The plugin update check runs once per served profile, against THAT profile's plugins dir
    and marker, under that profile's ``plugins:`` config. Unscoped it checked the launch home's
    plugins only, so B's plugins were never checked and B's ``plugins.auto_apply`` was ignored."""
    from agent.secret_scope import set_multiplex_active
    from hermes_cli import plugins_cadence

    a, b = two_homes
    (a / "config.yaml").write_text("plugins:\n  auto_update_check_hours: 1\n", encoding="utf-8")
    (b / "config.yaml").write_text(
        "plugins:\n  auto_update_check_hours: 1\n  auto_apply: true\n", encoding="utf-8")
    checked: list = []

    def _run_checks(plugins_dir, *a_, **k_):
        checked.append((plugins_dir, plugins_cadence.auto_apply_enabled()))
        return []

    import hermes_cli.plugins_updates as updates
    monkeypatch.setattr(updates, "run_checks", _run_checks)

    set_multiplex_active(True)
    try:
        _run_60_ticks(SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True)))
    finally:
        set_multiplex_active(False)

    assert checked == [(a / "plugins", False), (b / "plugins", True)]
    assert (a / "plugin-update-checks" / "last-run").exists()
    assert (b / "plugin-update-checks" / "last-run").exists()
