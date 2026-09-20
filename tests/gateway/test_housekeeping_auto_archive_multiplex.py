"""The multiplexer must sweep every served profile, not just its launch home (#109727).

The dashboard stands down for a served satellite (it has no gateway.pid of its own and the
multiplexer holds its writer), so if the gateway also skipped it, a satellite with
sessions.auto_archive enabled would never archive at all.

These drive the real `profile_scoped_chore` / `_for_each_served_profile` primitive rather than
a hand-rolled loop, so the per-profile config AND secret scoping are actually exercised.
"""
from pathlib import Path

import pytest


class _FakeDB:
    def __init__(self, path, swept):
        self.path, self._swept = path, swept

    def maybe_auto_archive(self, **kwargs):
        self._swept.append((self.path, kwargs["idle_days"]))


class _Config:
    multiplex_profiles = True


class _Runner:
    config = _Config()


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


def _patch_registry(monkeypatch, swept):
    import hermes_state_registry as reg

    from hermes_constants import get_hermes_home

    monkeypatch.setattr(reg, "acquire", lambda *a, **k: _FakeDB(get_hermes_home() / "state.db", swept))
    monkeypatch.setattr(reg, "release_or_close", lambda db: None)


def _serve(monkeypatch, *profiles):
    """Make the multiplexer serve exactly ``profiles`` (name, home) pairs."""
    import gateway.run as run_mod

    monkeypatch.setattr(run_mod, "_multiplex_profile_homes", lambda config: list(profiles))


def _tick(runner):
    from gateway.run import _housekeeping_auto_archive
    from gateway.run_profile_reconcile import profile_scoped_chore

    profile_scoped_chore(runner, _housekeeping_auto_archive)()


def test_each_served_profile_is_swept_with_its_own_config(homes, monkeypatch):
    launch, sat = homes
    swept = []
    _patch_registry(monkeypatch, swept)
    _serve(monkeypatch, ("default", launch), ("work", sat))

    _tick(_Runner())

    by_path = {p: d for p, d in swept}
    assert by_path.get(launch / "state.db") == 3.0, "launch profile must still be swept"
    assert by_path.get(sat / "state.db") == 9.0, \
        "satellite must use its OWN auto_archive_days, not the launch profile's"


def test_satellite_env_var_resolves_against_its_own_secret_scope(homes, monkeypatch):
    """load_config() expands ${VAR} through agent.secret_scope. Without the profile's secret
    scope the value resolves against the LAUNCH process environment instead."""
    launch, sat = homes
    (sat / "config.yaml").write_text(
        "sessions:\n  auto_archive: true\n  auto_archive_days: ${ARCHIVE_DAYS}\n", encoding="utf-8")
    (sat / ".env").write_text("ARCHIVE_DAYS=9\n", encoding="utf-8")
    monkeypatch.setenv("ARCHIVE_DAYS", "3")  # the launch process value must NOT win

    swept = []
    _patch_registry(monkeypatch, swept)
    _serve(monkeypatch, ("default", launch), ("work", sat))

    _tick(_Runner())

    by_path = {p: d for p, d in swept}
    assert by_path.get(sat / "state.db") == 9.0, (
        "satellite must resolve ${ARCHIVE_DAYS} from its OWN .env (9), not the launch "
        f"environment (3); swept={swept}")


def test_a_profile_with_auto_archive_disabled_is_skipped(homes, monkeypatch):
    launch, sat = homes
    (sat / "config.yaml").write_text("sessions:\n  auto_archive: false\n", encoding="utf-8")
    swept = []
    _patch_registry(monkeypatch, swept)
    _serve(monkeypatch, ("default", launch), ("work", sat))

    _tick(_Runner())

    assert [p for p, _ in swept] == [launch / "state.db"]


def test_one_broken_store_does_not_strand_the_others(homes, monkeypatch):
    """_for_each_served_profile does not isolate its profiles, so the chore itself must never
    raise — otherwise one unreadable store abandons every profile after it. Review P2 on #110405.
    GatewayRunner._init_session_db() tolerates a failed primary-store init and keeps running, so
    a broken launch store beside healthy satellites is a state the gateway genuinely reaches."""
    launch, sat = homes
    swept = []
    _patch_registry(monkeypatch, swept)
    _serve(monkeypatch, ("default", launch), ("work", sat))

    import hermes_state_registry as reg

    from hermes_constants import get_hermes_home

    def _acquire(*a, **k):
        home = get_hermes_home()
        if home == launch:
            raise OSError("launch store unavailable")
        return _FakeDB(home / "state.db", swept)

    monkeypatch.setattr(reg, "acquire", _acquire)

    _tick(_Runner())

    assert [p for p, _ in swept] == [sat / "state.db"], "satellite must still be swept"


def test_the_chore_never_raises(homes, monkeypatch):
    """The isolation contract, asserted directly on the chore rather than through the loop."""
    import hermes_state_registry as reg

    from gateway.run import _housekeeping_auto_archive

    def _boom(*a, **k):
        raise RuntimeError("store exploded")

    monkeypatch.setattr(reg, "acquire", _boom)

    _housekeeping_auto_archive()  # must not raise


def test_single_profile_gateway_still_sweeps(homes, monkeypatch):
    """Non-multiplex gateways run the chore once, unscoped — they must not lose auto-archive."""
    launch, _sat = homes
    swept = []
    _patch_registry(monkeypatch, swept)

    class _Single:
        class config:
            multiplex_profiles = False

    _tick(_Single())

    assert [p for p, _ in swept] == [launch / "state.db"]


def test_the_tick_is_registered_per_served_profile():
    """The wiring is the fix: an unwrapped chore runs once, against the launch home only, and
    every served satellite silently stops archiving (#109727). Guards that one line."""
    import inspect
    import re

    from gateway.run import _start_gateway_housekeeping

    source = inspect.getsource(_start_gateway_housekeeping)
    match = re.search(r'"Auto-archive tick",\s*([^)]*\))', source)
    assert match, "the auto-archive chore registration moved; update this test"
    assert "profile_scoped_chore" in match.group(1), (
        f"auto-archive must be registered per served profile, got: {match.group(1)}")
