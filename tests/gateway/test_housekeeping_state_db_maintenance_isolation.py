"""One profile's broken store must not abandon the profiles after it in the same tick.

``_for_each_served_profile`` runs the maintenance chore once per served profile with no
boundary between them, and ``_housekeeping_chore`` only catches at the tick level — so an
unreadable store would strand every profile that follows it. That state is reachable:
``GatewayRunner._init_session_db()`` deliberately tolerates a failed primary-store init and
keeps running, and the dashboard stands down for served satellites (#109727), leaving the
multiplexer as their only sweeper. Review P2 on #110405.
"""
from pathlib import Path

import pytest


class _FakeDB:
    def __init__(self, path, swept):
        self.path, self._swept = path, swept

    def maybe_auto_archive(self, **kwargs):
        self._swept.append((self.path, kwargs["idle_days"]))

    def maybe_auto_prune_and_vacuum(self, **kwargs):
        pass


class _Runner:
    class config:
        multiplex_profiles = True
        sessions_dir = None


def _write_profile(home: Path, days) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        f"sessions:\n  auto_archive: true\n  auto_archive_days: {days}\n", encoding="utf-8")


@pytest.fixture
def homes(tmp_path, monkeypatch):
    launch, sat = tmp_path / "launch", tmp_path / "profiles" / "work"
    _write_profile(launch, 3)
    _write_profile(sat, 9)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    return launch, sat


def _serve(monkeypatch, *profiles):
    import gateway.run as run_mod

    monkeypatch.setattr(run_mod, "_multiplex_profile_homes", lambda config: list(profiles))


def _tick(runner=None):
    from gateway.run import _housekeeping_state_db_maintenance
    from gateway.run_profile_reconcile import profile_scoped_chore

    profile_scoped_chore(runner or _Runner(), _housekeeping_state_db_maintenance)()


def test_a_broken_store_does_not_strand_the_profiles_after_it(homes, monkeypatch):
    launch, sat = homes
    swept = []
    _serve(monkeypatch, ("default", launch), ("work", sat))

    import hermes_state_registry as reg

    from hermes_constants import get_hermes_home

    def _acquire(*a, **k):
        home = get_hermes_home()
        if home == launch:
            raise OSError("launch store unavailable")
        return _FakeDB(home / "state.db", swept)

    monkeypatch.setattr(reg, "acquire", _acquire)
    monkeypatch.setattr(reg, "release_or_close", lambda db: None)

    _tick()

    assert [p for p, _ in swept] == [sat / "state.db"], \
        f"the satellite after the broken launch store must still be swept; swept={swept}"


def test_the_chore_never_raises(homes, monkeypatch):
    """Asserted directly on the chore, not just through the loop."""
    import hermes_state_registry as reg

    from gateway.run import _housekeeping_state_db_maintenance

    def _boom(*a, **k):
        raise RuntimeError("store exploded")

    monkeypatch.setattr(reg, "acquire", _boom)

    _housekeeping_state_db_maintenance()  # must not raise


def test_healthy_profiles_are_each_swept_with_their_own_config(homes, monkeypatch):
    """The isolation must not swallow the normal path: both profiles still sweep, each under
    its own config (the scoping itself is main's `profile_scoped_chore`)."""
    launch, sat = homes
    swept = []
    _serve(monkeypatch, ("default", launch), ("work", sat))

    import hermes_state_registry as reg

    from hermes_constants import get_hermes_home

    monkeypatch.setattr(reg, "acquire", lambda *a, **k: _FakeDB(get_hermes_home() / "state.db", swept))
    monkeypatch.setattr(reg, "release_or_close", lambda db: None)

    _tick()

    by_path = {p: d for p, d in swept}
    assert by_path.get(launch / "state.db") == 3.0
    assert by_path.get(sat / "state.db") == 9.0, "satellite must use its OWN auto_archive_days"
