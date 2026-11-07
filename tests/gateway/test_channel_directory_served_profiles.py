"""A multiplexed gateway writes one channel directory per served profile home.

The directory is the gateway's cache of what ITS bot can see, so one bot's view is not another's.
The housekeeping chore built it once per process, so only ``get_hermes_home()``'s copy (the launch
home) was ever refreshed: every named profile's file stayed frozen at whatever a previous
single-profile gateway last saw, and a room only that profile's own bot had ever seen never reached
that profile's directory. These tests drive the real housekeeping thread and pin the per-home write,
its failure isolation, and the unchanged single-profile path.
"""

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import channel_directory
import gateway.run as gateway_run
from gateway.config import Platform


@pytest.fixture(autouse=True)
def _isolate_channel_aliases(tmp_path, monkeypatch):
    """A real ~/.hermes/channel_aliases.json must never leak into these builds."""
    monkeypatch.setattr(channel_directory, "CHANNEL_ALIASES_PATH", tmp_path / "none.json")


@pytest.fixture
def two_homes(tmp_path, monkeypatch):
    """Launch home A (= multiplex ``default``) and served named profile B under ``A/profiles/b``."""
    a = tmp_path / "home" / ".hermes"
    b = a / "profiles" / "b"
    for home in (a, b):
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("model:\n  provider: nous\n", encoding="utf-8")
        (home / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(a))
    # The suite conftest pins ``hermes_state.DEFAULT_DB_PATH`` at one sandbox store whenever
    # hermes_state is already imported, and that pin WINS over ``get_hermes_home()`` inside
    # ``_default_db_path()`` -- exactly the per-profile resolution these tests exist to prove.
    # Restore the import-time sentinel so an argless ``acquire()`` resolves through the scope.
    import hermes_state
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    resolved = Path(hermes_state._default_db_path())
    assert resolved.is_relative_to(tmp_path), f"unpinned store escaped the sandbox: {resolved}"
    return a, b


class _Bot:
    """Adapter with no ``list_channels`` hook, so its platform is discovered from sessions only."""


class _Ticks:
    """Stop event that lets the housekeeping loop run exactly ``n`` ticks with no sleeping."""

    def __init__(self, n: int):
        self.n, self.left = 0, n

    def is_set(self):
        return self.n >= self.left

    def wait(self, timeout=None):
        self.n += 1
        return True


def _seed_room(home: Path, *, chat_id: str, name: str) -> None:
    """One gateway session row in *home*'s own store: the room only that profile's bot has run in."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session(
            f"sess-{chat_id}", "telegram", session_key=f"telegram:{chat_id}", chat_id=chat_id,
            chat_type="group", display_name=name,
            origin_json=json.dumps({"platform": "telegram", "chat_id": chat_id,
                                    "chat_name": name, "chat_type": "group"}))
    finally:
        db.close()


def _multiplex_runner(a: Path, b: Path, profile_adapters) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(multiplex_profiles=True,
                               _runtime_profile_homes=(("default", a), ("b", b))),
        _primary_profile_name="default",
        _profile_adapters=profile_adapters,
    )


def _tick(runner, adapters, monkeypatch) -> None:
    """Run the real housekeeping thread for one 5-tick cycle with only the directory chore live."""
    monkeypatch.setattr(
        gateway_run, "_housekeeping_chore",
        lambda label, fn, *args, **kwargs:
        fn(*args, **kwargs) if label == "Channel directory refresh" else None)

    async def _run():
        await asyncio.to_thread(
            gateway_run._start_gateway_housekeeping, _Ticks(5),
            adapters=adapters, loop=asyncio.get_running_loop(), interval=0, runner=runner)

    asyncio.run(_run())


def _room_ids(home: Path) -> list:
    directory = json.loads((home / "channel_directory.json").read_text(encoding="utf-8"))
    return [entry["id"] for entry in directory["platforms"]["telegram"]]


def test_multiplexed_gateway_writes_a_directory_per_served_home(two_homes, monkeypatch):
    """Every served home gets its own file, holding ITS bot's rooms and nothing else.

    Before the per-home write only A's file was ever refreshed, so B's rooms -- and the thread that
    ran solely on B -- were absent from B's own directory while A's file could not supply them
    either (they are not in A's store).
    """
    a, b = two_homes
    _seed_room(a, chat_id="-1001", name="A room")
    _seed_room(b, chat_id="-2002", name="B room")

    _tick(_multiplex_runner(a, b, {"b": {Platform.TELEGRAM: _Bot()}}),
          {Platform.TELEGRAM: _Bot()}, monkeypatch)

    assert (b / "channel_directory.json").exists(), "profile b's own directory was never written"
    assert _room_ids(b) == ["-2002"]
    assert _room_ids(a) == ["-1001"]
    for home in (a, b):
        directory = json.loads((home / "channel_directory.json").read_text(encoding="utf-8"))
        assert directory["updated_at"], f"{home.name}'s directory was written unstamped"


def test_one_unwritable_home_is_logged_and_the_others_still_land(two_homes, monkeypatch, caplog):
    """The default home is written FIRST; its failed write must not cost B its directory.

    The tick-level catcher would otherwise abandon every profile after the first broken one, on
    every tick -- and each one's file is the only thing that tells its bot which rooms it can reach.
    """
    a, b = two_homes
    _seed_room(a, chat_id="-1001", name="A room")
    _seed_room(b, chat_id="-2002", name="B room")
    real_write = channel_directory.atomic_json_write

    def _refuse_default_home(path, data, **kwargs):
        if Path(path).parent == a:
            raise OSError(f"[Errno 13] Permission denied: '{path}'")
        return real_write(path, data, **kwargs)

    monkeypatch.setattr(channel_directory, "atomic_json_write", _refuse_default_home)

    with caplog.at_level(logging.WARNING):
        _tick(_multiplex_runner(a, b, {"b": {Platform.TELEGRAM: _Bot()}}),
              {Platform.TELEGRAM: _Bot()}, monkeypatch)

    assert _room_ids(b) == ["-2002"], "the readable profile lost its directory to a sibling's failure"
    assert not (a / "channel_directory.json").exists()
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(str(a / "channel_directory.json") in message for message in warnings), warnings


def test_a_served_profile_this_process_does_not_run_keeps_its_file(two_homes, monkeypatch):
    """A served profile with no adapter map here is served by someone else: leave its file alone.

    Writing an empty directory for it would blank rooms that its own gateway had discovered and
    that nobody in this process can see.
    """
    a, b = two_homes
    _seed_room(a, chat_id="-1001", name="A room")
    previous = {"updated_at": "2026-09-24T10:05:00",
                "platforms": {"telegram": [{"id": "-999", "name": "not ours", "type": "group",
                                            "thread_id": None}]}}
    (b / "channel_directory.json").write_text(json.dumps(previous), encoding="utf-8")

    _tick(_multiplex_runner(a, b, {}), {Platform.TELEGRAM: _Bot()}, monkeypatch)

    assert json.loads((b / "channel_directory.json").read_text(encoding="utf-8")) == previous
    assert _room_ids(a) == ["-1001"]


def test_single_profile_gateway_writes_only_the_launch_home(two_homes, monkeypatch):
    """Control: with multiplex off, one directory for the launch home and no profile file at all."""
    a, b = two_homes
    _seed_room(a, chat_id="-1001", name="A room")
    _seed_room(b, chat_id="-2002", name="B room")

    _tick(SimpleNamespace(config=SimpleNamespace(multiplex_profiles=False)),
          {Platform.TELEGRAM: _Bot()}, monkeypatch)

    assert _room_ids(a) == ["-1001"]
    assert not (b / "channel_directory.json").exists()
