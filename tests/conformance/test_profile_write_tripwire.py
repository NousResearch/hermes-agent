"""Profile-isolation write tripwire: no cross-profile writes to the default home.

Invariant under test
--------------------
When a non-default profile is active (``HERMES_HOME`` points at
``<root>/profiles/testprof``), NO subsystem write may land anywhere under the
default profile's tree — ``<root>/state.db``, ``<root>/config.yaml``,
``<root>/memories/``, ``<root>/cron/``, or any other path directly under
``<root>`` outside the ``profiles/`` subtree.

This is a recurring bug *class*, not a single bug.  Recent regressions:

* #88532 — gateway sessions physically stored in the root ``state.db`` even
  though the profile scope was active (rows carried the right profile_name,
  so the only symptom was the desktop listing them under the default bot).
* #92662 / #89190 — settings and API keys saved while a profile was active
  were written to the base profile's ``config.yaml`` / ``.env``.
* #89625 — cron listing/writing the default profile's jobs.json because the
  store paths were frozen at import time.
* #92156 — terminal env cache leaked across profiles (same import-time-
  snapshot shape of bug).

Each test here exercises a REAL persistence surface through its real code
path (no mocks of the subject) with the ``testprof`` profile active, then
walks the entire default-profile tree and fails if anything was created,
modified, or deleted there.  The walk is deliberately whole-tree rather than
per-known-file so the *next* variant of this bug class (a new subsystem, a
new sidecar file) trips it too.

Reusability
-----------
``ProfileTripwire`` and the ``profile_tripwire`` fixture are module-level and
documented so future surfaces (terminal env cache, skills, plugins, logs …)
can add one small exercise function + a ``pytest.param`` to ``SURFACES``.
"""

import os
import sqlite3
from pathlib import Path

import pytest

import hermes_state
from hermes_constants import (
    reset_hermes_home_override,
    set_hermes_home_override,
)


class ProfileTripwire:
    """Snapshot of the default profile's tree + the active testprof home.

    Construction snapshots every path under ``root`` EXCEPT the
    ``root/profiles`` subtree (that subtree is exactly where writes are
    supposed to go).  ``assert_default_untouched()`` re-walks and fails on
    any created / modified / deleted entry.

    Attributes:
        root:     the default profile's home (``<tmp>/hermes``).
        profile:  the active profile home (``<root>/profiles/testprof``).
    """

    def __init__(self, root: Path, profile: Path):
        self.root = root
        self.profile = profile
        self._baseline = self._snapshot()

    def _snapshot(self) -> dict:
        """Map of relpath -> (kind, mtime_ns, size) for the default tree."""
        snap = {}
        profiles_subtree = self.root / "profiles"
        for dirpath, dirnames, filenames in os.walk(self.root):
            d = Path(dirpath)
            if d == self.root and "profiles" in dirnames:
                dirnames.remove("profiles")  # never descend into profiles/
            for name in dirnames:
                p = d / name
                if p == profiles_subtree:
                    continue
                snap[str(p.relative_to(self.root))] = ("dir", None, None)
            for name in filenames:
                p = d / name
                st = p.stat()
                snap[str(p.relative_to(self.root))] = (
                    "file", st.st_mtime_ns, st.st_size,
                )
        return snap

    def assert_default_untouched(self) -> None:
        """Fail if the default profile's tree changed in any way."""
        now = self._snapshot()
        created = sorted(set(now) - set(self._baseline))
        deleted = sorted(set(self._baseline) - set(now))
        modified = sorted(
            rel for rel in set(now) & set(self._baseline)
            if now[rel] != self._baseline[rel]
        )
        problems = []
        if created:
            problems.append(f"created in default profile: {created}")
        if modified:
            problems.append(f"modified in default profile: {modified}")
        if deleted:
            problems.append(f"deleted from default profile: {deleted}")
        assert not problems, (
            "Cross-profile write leak (bug class #88532/#92662/#89190/"
            "#89625/#92156): the testprof profile was active but the default "
            f"profile tree at {self.root} changed: " + "; ".join(problems)
        )


@pytest.fixture
def profile_tripwire(tmp_path, monkeypatch):
    """Activate ``<root>/profiles/testprof`` and arm the default-tree tripwire.

    Sets up:

    1. A temp hermes root acting as the *default* profile, seeded with the
       files a real install has (config.yaml, memories/, cron/jobs.json) so
       both "new file created" and "existing file modified" leaks are
       detectable.
    2. ``HERMES_HOME`` (env var AND the context-local override) pointed at
       the testprof home — the exact activation shape ``--profile`` uses.
    3. ``hermes_state.DEFAULT_DB_PATH`` restored to its import-time snapshot.
       The suite conftest deliberately re-points that constant at its own
       fake home, which trips the escape hatch in ``_default_db_path()``
       (a re-pointed constant wins over everything).  Closing the hatch
       makes resolution flow through ``get_hermes_home()`` — the production
       path, and the one #88532 regressed.

    Yields a :class:`ProfileTripwire`; teardown re-asserts the invariant so
    a surface test that forgets the explicit check still trips the wire.
    """
    root = tmp_path / "hermes"
    profile = root / "profiles" / "testprof"
    profile.mkdir(parents=True)

    # Seed the default profile the way a real install looks.
    (root / "config.yaml").write_text("model:\n  default: default-model\n")
    (root / "memories").mkdir()
    (root / "memories" / "MEMORY.md").write_text("default profile memory\n")
    (root / "cron").mkdir()
    (root / "cron" / "jobs.json").write_text("[]\n")
    (root / ".env").write_text("")

    monkeypatch.setenv("HERMES_HOME", str(profile))
    token = set_hermes_home_override(str(profile))

    # Close the conftest's DEFAULT_DB_PATH escape hatch (see docstring).
    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH
    )

    tripwire = ProfileTripwire(root, profile)
    try:
        yield tripwire
        # Safety net: re-check even if the test body forgot to.
        tripwire.assert_default_untouched()
    finally:
        reset_hermes_home_override(token)


# ---------------------------------------------------------------------------
# Surface exercisers — each drives ONE real persistence surface end to end.
# Add new surfaces here as the bug class produces new variants.
# ---------------------------------------------------------------------------

def _exercise_session_db(tripwire: ProfileTripwire) -> None:
    """SessionDB() argless construction + session + message (#88532)."""
    db = hermes_state.SessionDB()
    try:
        db.create_session("20260823_000000_tripwire", "cli")
        db.append_message("20260823_000000_tripwire", "user", "hello")
    finally:
        db.close()
    profile_state_db = tripwire.profile / "state.db"
    assert profile_state_db.exists(), (
        f"SessionDB() did not write to the active profile home "
        f"({profile_state_db} missing)"
    )


def _exercise_config_save(tripwire: ProfileTripwire) -> None:
    """save_config()/load_config() while a profile is active (#92662, #89190)."""
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    cfg["model"] = "testprof-model"  # bare-string alias form is canonical
    save_config(cfg)
    assert (tripwire.profile / "config.yaml").exists(), (
        "save_config() did not write the active profile's config.yaml"
    )


def _exercise_memory_store(tripwire: ProfileTripwire) -> None:
    """MemoryStore.add() persistence path (same class as #92662)."""
    from tools.memory_tool import MemoryStore

    store = MemoryStore()
    store.load_from_disk()
    result = store.add("memory", "testprof remembers something")
    assert result.get("success"), f"memory add failed: {result}"
    mem_file = tripwire.profile / "memories" / "MEMORY.md"
    assert mem_file.exists(), (
        "MemoryStore did not persist to the active profile's memories/"
    )
    assert "testprof remembers something" in mem_file.read_text()


def _exercise_cron_store(tripwire: ProfileTripwire) -> None:
    """cron save_jobs()/load_jobs() store resolution (#89625)."""
    from cron.jobs import load_jobs, save_jobs

    job = {
        "id": "tripwire-job",
        "name": "tripwire",
        "prompt": "noop",
        "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
        "enabled": True,
    }
    save_jobs([job])
    jobs_file = tripwire.profile / "cron" / "jobs.json"
    assert jobs_file.exists(), (
        "save_jobs() did not write the active profile's cron/jobs.json"
    )
    loaded = load_jobs()
    assert any(j.get("id") == "tripwire-job" for j in loaded), (
        "load_jobs() did not read back from the active profile's store "
        "(#89625 shape: reading the default profile's jobs instead)"
    )
    # And the default profile's jobs.json content is untouched (the tree
    # walk would also catch this; the explicit read documents the contract).
    assert (tripwire.root / "cron" / "jobs.json").read_text() == "[]\n"


# NOTE: the terminal env cache surface (#92156) is not exercised here: the
# terminal backend cache requires spawning a real terminal session, which is
# not hermetic in this suite.  Its bug shape (import-time path snapshot) is
# the same one the four surfaces above pin.

SURFACES = [
    pytest.param(_exercise_session_db, id="session_db"),
    pytest.param(_exercise_config_save, id="config_save"),
    pytest.param(_exercise_memory_store, id="memory_store"),
    pytest.param(_exercise_cron_store, id="cron_store"),
]


@pytest.mark.parametrize("exercise", SURFACES)
def test_no_default_profile_writes(profile_tripwire, exercise):
    """Exercising a persistence surface under testprof must not touch <root>.

    Regression tripwire for the cross-profile write bug class:
    #88532 (sessions in root state.db), #92662/#89190 (settings/API keys in
    the base profile), #89625 (cron using the default profile's store),
    #92156 (terminal env cache leaking cross-profile).
    """
    exercise(profile_tripwire)
    profile_tripwire.assert_default_untouched()


def test_session_db_file_lands_in_profile_home(profile_tripwire):
    """The state.db written by an argless SessionDB() is PHYSICALLY the
    testprof file — rows are readable from ``profiles/testprof/state.db``
    and no root ``state.db`` exists at all (#88532's exact symptom).
    """
    db = hermes_state.SessionDB()
    try:
        db.create_session("20260823_111111_tripwire", "cli")
        db.append_message("20260823_111111_tripwire", "user", "profile hello")
    finally:
        db.close()

    profile_db = profile_tripwire.profile / "state.db"
    assert profile_db.exists()
    conn = sqlite3.connect(str(profile_db))
    try:
        ids = {r[0] for r in conn.execute("SELECT id FROM sessions")}
    finally:
        conn.close()
    assert "20260823_111111_tripwire" in ids

    assert not (profile_tripwire.root / "state.db").exists(), (
        "#88532 regression: a state.db appeared in the DEFAULT profile home "
        "while testprof was active"
    )
    profile_tripwire.assert_default_untouched()


def test_config_round_trips_through_profile_home(profile_tripwire):
    """load_config() under testprof reads testprof's config, not the root's.

    #92662/#89190 shape: the base profile's config.yaml carried the default
    model 'default-model'; a load under testprof must not see or mutate it.
    """
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    cfg["model"] = "round-trip-model"  # bare-string alias form is canonical
    save_config(cfg)

    reread = load_config()
    model = reread.get("model")
    resolved = model if isinstance(model, str) else (model or {}).get("default")
    assert resolved == "round-trip-model"
    # The default profile's config still holds its own value.
    assert "default-model" in (profile_tripwire.root / "config.yaml").read_text()
    profile_tripwire.assert_default_untouched()
