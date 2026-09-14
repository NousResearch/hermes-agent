"""The multiplexer must sweep every served profile, not just its launch home (#109727).

The dashboard stands down for a served satellite (it has no gateway.pid of its own and
the multiplexer holds its writer), so if the gateway also skipped it, a satellite with
sessions.auto_archive enabled would never archive at all.
"""
from pathlib import Path

import pytest


class _FakeDB:
    def __init__(self, path, swept):
        self.path, self._swept = path, swept

    def maybe_auto_archive(self, **kwargs):
        self._swept.append((self.path, kwargs["idle_days"]))


class _Runner:
    def __init__(self, homes):
        self._served_profile_homes = homes


@pytest.fixture
def homes(tmp_path, monkeypatch):
    launch = tmp_path / "launch"
    sat = tmp_path / "profiles" / "work"
    for home, days in ((launch, 3), (sat, 9)):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(
            f"sessions:\n  auto_archive: true\n  auto_archive_days: {days}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    return launch, sat


def _patch_registry(monkeypatch, swept):
    import hermes_state_registry as reg

    from hermes_constants import get_hermes_home

    monkeypatch.setattr(
        reg, "acquire",
        lambda db_path=None: _FakeDB(Path(db_path) if db_path else get_hermes_home() / "state.db", swept))
    monkeypatch.setattr(reg, "release_or_close", lambda db: None)


def test_served_satellite_is_swept_with_its_own_config(homes, monkeypatch):
    launch, sat = homes
    swept = []
    _patch_registry(monkeypatch, swept)

    from gateway.run import _housekeeping_auto_archive

    _housekeeping_auto_archive(_Runner({"default": launch, "work": sat}))

    by_path = {p: d for p, d in swept}
    assert launch / "state.db" in by_path, "launch profile must still be swept"
    assert by_path[sat / "state.db"] == 9.0, "satellite must use its OWN auto_archive_days, not the launch profile's"


def test_launch_home_is_not_swept_twice(homes, monkeypatch):
    launch, _sat = homes
    swept = []
    _patch_registry(monkeypatch, swept)

    from gateway.run import _housekeeping_auto_archive

    _housekeeping_auto_archive(_Runner({"default": launch}))

    assert len(swept) == 1, f"launch home swept more than once: {swept}"


def test_satellite_with_auto_archive_disabled_is_skipped(homes, monkeypatch):
    launch, sat = homes
    (sat / "config.yaml").write_text("sessions:\n  auto_archive: false\n", encoding="utf-8")
    swept = []
    _patch_registry(monkeypatch, swept)

    from gateway.run import _housekeeping_auto_archive

    _housekeeping_auto_archive(_Runner({"default": launch, "work": sat}))

    assert [p for p, _ in swept] == [launch / "state.db"]


def test_one_broken_satellite_does_not_stop_the_others(homes, monkeypatch):
    launch, sat = homes
    broken = launch.parent / "profiles" / "broken"
    broken.mkdir(parents=True)
    swept = []
    _patch_registry(monkeypatch, swept)

    import hermes_state_registry as reg

    real_acquire = reg.acquire

    def _acquire(db_path=None):
        if db_path and Path(db_path).parent == broken:
            raise OSError("satellite store unreadable")
        return real_acquire(db_path)

    monkeypatch.setattr(reg, "acquire", _acquire)

    from gateway.run import _housekeeping_auto_archive

    _housekeeping_auto_archive(_Runner({"default": launch, "broken": broken, "work": sat}))

    assert sat / "state.db" in {p for p, _ in swept}, "a broken sibling must not abort the sweep"


def test_no_runner_still_sweeps_the_launch_home(homes, monkeypatch):
    launch, _sat = homes
    swept = []
    _patch_registry(monkeypatch, swept)

    from gateway.run import _housekeeping_auto_archive

    _housekeeping_auto_archive()

    assert [p for p, _ in swept] == [launch / "state.db"]


def test_a_broken_launch_store_does_not_strand_the_satellites(homes, monkeypatch):
    """GatewayRunner._init_session_db() tolerates a failed primary-store init and keeps
    running, so the multiplexer can serve healthy satellites while its own store is down.
    The dashboard has already stood down for those satellites — if a launch-side raise
    escaped here they would never archive at all. Review P2 on #110405."""
    launch, sat = homes
    swept = []
    _patch_registry(monkeypatch, swept)

    import hermes_state_registry as reg

    real_acquire = reg.acquire

    def _acquire(db_path=None):
        if db_path is None or Path(db_path).parent == launch:
            raise OSError("launch store unavailable")
        return real_acquire(db_path)

    monkeypatch.setattr(reg, "acquire", _acquire)

    from gateway.run import _housekeeping_auto_archive

    _housekeeping_auto_archive(_Runner({"default": launch, "work": sat}))

    assert [p for p, _ in swept] == [sat / "state.db"], "satellite must still be swept"


def test_satellite_env_var_resolves_against_its_own_secret_scope(tmp_path, monkeypatch):
    """load_config() expands ${VAR} through agent.secret_scope. Installing only
    HERMES_HOME leaves the expansion falling back to the LAUNCH process environment,
    so a satellite's own .env value is ignored. Review P2 on #110405."""
    launch = tmp_path / "launch"
    sat = tmp_path / "profiles" / "work"
    launch.mkdir(parents=True)
    sat.mkdir(parents=True)
    (launch / "config.yaml").write_text(
        "sessions:\n  auto_archive: true\n  auto_archive_days: 3\n", encoding="utf-8")
    (sat / "config.yaml").write_text(
        "sessions:\n  auto_archive: true\n  auto_archive_days: ${ARCHIVE_DAYS}\n", encoding="utf-8")
    (sat / ".env").write_text("ARCHIVE_DAYS=9\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("ARCHIVE_DAYS", "3")  # the launch process value must NOT win

    swept = []
    _patch_registry(monkeypatch, swept)

    from gateway.run import _housekeeping_auto_archive

    _housekeeping_auto_archive(_Runner({"default": launch, "work": sat}))

    by_path = {p: d for p, d in swept}
    assert by_path.get(sat / "state.db") == 9.0, (
        "satellite must resolve ${ARCHIVE_DAYS} from its OWN .env (9), not the launch "
        f"environment (3); swept={swept}")
