"""Unit tests for the user-state upgrade-preservation verifier.

tests/install/e2e-assets/verify-user-state.py is the standalone hook the
install/update E2E drivers call around a real upgrade. These tests exercise it
against a real temp HERMES_HOME (real files, real sqlite databases) — no
source-reading, no filesystem mocks.

The contract under test: an upgrade may ADD state and may rewrite config.yaml
(config migration) and the bundled skills tree (skills sync), but it may not
delete or modify the user's own durable state, and it may not shrink state.db.
plugins/** is deliberately not owned here — verify-plugin-preservation.py owns
it — so this tool must not fail on it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
VERIFIER = os.path.join(_HERE, "..", "install", "e2e-assets", "verify-user-state.py")

_spec = importlib.util.spec_from_file_location("verify_user_state", VERIFIER)
vus = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vus)


def _make_db(path, sessions=2, messages=3):
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY)")
    con.execute("CREATE TABLE IF NOT EXISTS messages (id TEXT)")
    con.executemany("INSERT OR REPLACE INTO sessions VALUES (?)",
                    [(f"s{i}",) for i in range(sessions)])
    con.executemany("INSERT INTO messages VALUES (?)",
                    [(f"m{i}",) for i in range(messages)])
    con.commit()
    con.close()


def _home(tmp_path):
    home = tmp_path / "home"
    for rel in ("memories", "cron", "sessions", "skills/foo", "skills/.archive/mine",
                "plugins/demo", "photon/sidecar", "profiles/work"):
        (home / rel).mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("timezone: utc\n", encoding="utf-8")
    (home / ".env").write_text("NOUS_API_KEY=xxx\n", encoding="utf-8")
    (home / "auth.json").write_text('{"tokens":{}}\n', encoding="utf-8")
    (home / "memories" / "note.md").write_text("remember\n", encoding="utf-8")
    (home / "cron" / "jobs.json").write_text('{"jobs":[]}\n', encoding="utf-8")
    (home / "sessions" / "2026.jsonl").write_text('{"t":1}\n', encoding="utf-8")
    (home / "skills" / "foo" / "SKILL.md").write_text("bundled\n", encoding="utf-8")
    (home / "skills" / ".archive" / "mine" / "SKILL.md").write_text("mine\n", encoding="utf-8")
    (home / "plugins" / "demo" / "plugin.yaml").write_text("name: demo\n", encoding="utf-8")
    (home / "photon" / "sidecar" / "package-lock.json").write_text("{}\n", encoding="utf-8")
    (home / "profiles" / "work" / "config.yaml").write_text("x: 1\n", encoding="utf-8")
    _make_db(home / "state.db")
    return home


def _snapshot(home):
    return vus.snapshot_home(str(home))


def _verify(home, snap):
    return vus.verify_home(str(home), snap)


# --- shape ------------------------------------------------------------------

def test_snapshot_records_rows_not_bytes_for_state_db(tmp_path):
    home = _home(tmp_path)
    snap = _snapshot(home)
    record = snap["entries"]["state.db"]
    assert record["kind"] == "file"
    assert record["rows"] == {"sessions": 2, "messages": 3}
    assert "sha256" not in record, "a live db must be judged by rows, not bytes"


def test_snapshot_is_read_only(tmp_path):
    home = _home(tmp_path)
    before = sorted(os.path.relpath(os.path.join(dp, n), home)
                    for dp, _, ns in os.walk(home) for n in ns)
    snap = _snapshot(home)
    after = sorted(os.path.relpath(os.path.join(dp, n), home)
                   for dp, _, ns in os.walk(home) for n in ns)
    assert before == after
    assert snap["entries"], "a used home must produce judged entries"


def test_plugins_and_bundled_skills_are_not_judged(tmp_path):
    home = _home(tmp_path)
    snap = _snapshot(home)
    assert not [k for k in snap["entries"] if k.startswith("plugins/")], \
        "plugins/** is owned by verify-plugin-preservation.py"
    assert "skills/foo/SKILL.md" in snap["advisory"]
    assert "skills/foo/SKILL.md" not in snap["entries"]
    assert "skills/.archive/mine/SKILL.md" in snap["entries"], \
        "the curator's archive holds USER skills and is judged"


# --- the contract -----------------------------------------------------------

def test_verify_passes_when_nothing_changed(tmp_path):
    home = _home(tmp_path)
    report = _verify(home, _snapshot(home))
    assert report["ok"] is True
    assert report["deleted"] == [] and report["modified"] == {}


def test_tolerates_added_files(tmp_path):
    home = _home(tmp_path)
    snap = _snapshot(home)
    (home / "memories" / "another.md").write_text("new\n", encoding="utf-8")
    report = _verify(home, snap)
    assert report["ok"] is True, "an upgrade may add files"
    assert "memories/another.md" in report["added"]


def test_fails_when_a_user_file_is_deleted(tmp_path):
    home = _home(tmp_path)
    snap = _snapshot(home)
    (home / "memories" / "note.md").unlink()
    report = _verify(home, snap)
    assert report["ok"] is False
    assert "memories/note.md" in report["deleted"]


def test_fails_when_env_is_modified(tmp_path):
    home = _home(tmp_path)
    snap = _snapshot(home)
    (home / ".env").write_text("CLOBBERED=1\n", encoding="utf-8")
    report = _verify(home, snap)
    assert report["ok"] is False
    assert ".env" in report["modified"]


def test_tolerates_config_yaml_rewrite(tmp_path):
    home = _home(tmp_path)
    snap = _snapshot(home)
    (home / "config.yaml").write_text("timezone: utc\nnew_key: 1\n", encoding="utf-8")
    (home / "profiles" / "work" / "config.yaml").write_text("x: 1\ny: 2\n", encoding="utf-8")
    report = _verify(home, snap)
    assert report["ok"] is True, "config migration rewrites config.yaml additively"
    assert sorted(report["tolerated_modified"]) == sorted(
        ["config.yaml", "profiles/work/config.yaml"])


def test_fails_when_state_db_loses_rows(tmp_path):
    home = _home(tmp_path)
    snap = _snapshot(home)
    con = sqlite3.connect(str(home / "state.db"))
    con.execute("DELETE FROM sessions WHERE id='s1'")
    con.commit()
    con.close()
    report = _verify(home, snap)
    assert report["ok"] is False
    assert report["rows_shrank"] == ["state.db"]


def test_tolerates_state_db_growth(tmp_path):
    home = _home(tmp_path)
    snap = _snapshot(home)
    con = sqlite3.connect(str(home / "state.db"))
    con.execute("INSERT INTO sessions VALUES ('s9')")
    con.commit()
    con.close()
    report = _verify(home, snap)
    assert report["ok"] is True, "a later turn adds rows; that is not loss"


def test_bundled_skill_resync_is_advisory_only(tmp_path):
    home = _home(tmp_path)
    snap = _snapshot(home)
    # A release adds a bundled skill and updates an existing one.
    (home / "skills" / "bar").mkdir()
    (home / "skills" / "bar" / "SKILL.md").write_text("new bundled\n", encoding="utf-8")
    (home / "skills" / "foo" / "SKILL.md").write_text("updated bundled\n", encoding="utf-8")
    report = _verify(home, snap)
    assert report["ok"] is True, "skills sync rewrites the bundled tree by design"
    assert "skills/foo/SKILL.md" in report["advisory"]["modified"]


def test_bundled_skill_resync_inside_a_profile_is_advisory_only(tmp_path):
    """A second profile's bundled skills re-sync on update exactly like the
    root's, so judging them would fail every leg that owns a profile."""
    home = _home(tmp_path)
    skills = home / "profiles" / "work" / "skills" / "foo"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text("bundled\n", encoding="utf-8")
    snap = _snapshot(home)
    # The update re-syncs every profile: bundled skill content moves.
    (skills / "SKILL.md").write_text("updated bundled\n", encoding="utf-8")
    report = _verify(home, snap)
    assert report["ok"] is True, "a per-profile skills sync is not a user-state change"
    assert "profiles/work/skills/foo/SKILL.md" in report["advisory"]["modified"]
    assert "profiles/work/skills/foo/SKILL.md" not in report["modified"]


def test_fails_when_a_profile_skill_archive_is_lost(tmp_path):
    """skills/.archive/** holds restorable USER skills at both roots."""
    home = _home(tmp_path)
    archive = home / "profiles" / "work" / "skills" / ".archive" / "mine"
    archive.mkdir(parents=True)
    (archive / "SKILL.md").write_text("mine\n", encoding="utf-8")
    snap = _snapshot(home)
    (archive / "SKILL.md").unlink()
    report = _verify(home, snap)
    assert report["ok"] is False
    assert "profiles/work/skills/.archive/mine/SKILL.md" in report["deleted"]


def test_fails_when_archived_user_skill_is_lost(tmp_path):
    home = _home(tmp_path)
    snap = _snapshot(home)
    (home / "skills" / ".archive" / "mine" / "SKILL.md").unlink()
    report = _verify(home, snap)
    assert report["ok"] is False
    assert "skills/.archive/mine/SKILL.md" in report["deleted"]


def test_fails_when_a_profile_directory_disappears(tmp_path):
    home = _home(tmp_path)
    snap = _snapshot(home)
    (home / "profiles" / "work" / "config.yaml").unlink()
    report = _verify(home, snap)
    assert report["ok"] is False
    assert "profiles/work/config.yaml" in report["deleted"]


# --- CLI contract -----------------------------------------------------------

def _run(args):
    return subprocess.run([sys.executable, VERIFIER, *args],
                          capture_output=True, text=True)


def test_cli_snapshot_then_verify_round_trip(tmp_path):
    home = _home(tmp_path)
    snap = tmp_path / "snap.json"
    report = tmp_path / "report.json"
    first = _run(["snapshot", "--home", str(home), "--out", str(snap)])
    assert first.returncode == 0, first.stderr
    assert snap.exists()
    ok = _run(["verify", "--home", str(home), "--snapshot", str(snap),
               "--report", str(report)])
    assert ok.returncode == 0, ok.stderr
    assert json.loads(report.read_text(encoding="utf-8"))["ok"] is True

    (home / "memories" / "note.md").unlink()
    bad = _run(["verify", "--home", str(home), "--snapshot", str(snap)])
    assert bad.returncode == 1
    assert "USER-STATE PRESERVATION FAILED" in bad.stderr


def test_cli_refuses_an_empty_snapshot_as_inconclusive(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    snap = tmp_path / "snap.json"
    result = _run(["snapshot", "--home", str(empty), "--out", str(snap)])
    assert result.returncode == 3
    assert "INCONCLUSIVE" in result.stderr


def test_cli_rejects_a_missing_home(tmp_path):
    result = _run(["snapshot", "--home", str(tmp_path / "nope"),
                   "--out", str(tmp_path / "s.json")])
    assert result.returncode == 2


def test_a_rewritten_dotenv_names_the_moved_variables_and_never_their_values(tmp_path):
    """An equal-size .env rewrite is invisible in the whole-file hash alone.

    The app-driven upgrade rewrote .env at an identical byte count with a
    different sha256, leaving the leg holding two hashes of a secrets file and no
    lead. The report must name the variable -- and carry no value.
    """
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text(
        "OPENAI_API_KEY=aaaa\n"
        "OPENAI_BASE_URL=http://127.0.0.1:9001/v1\n"
        "# a comment\n"
        "export ANTHROPIC_API_KEY=cccc\n",
        encoding="utf-8",
    )
    snap = vus.snapshot_home(str(home))

    (home / ".env").write_text(
        "OPENAI_API_KEY=bbbb\n"
        "OPENAI_BASE_URL=http://127.0.0.1:9001/v1\n"
        "# a comment\n"
        "export ANTHROPIC_API_KEY=cccc\n"
        "NEW_KEY=dddd\n",
        encoding="utf-8",
    )
    report = vus.verify_home(str(home), snap)

    diff = report["modified"][".env"]["key_diff"]
    assert diff["keys_changed"] == ["OPENAI_API_KEY"]
    assert diff["keys_added"] == ["NEW_KEY"]
    assert diff["keys_removed"] == []
    rendered = vus._render(report)
    assert "changed=OPENAI_API_KEY" in rendered
    assert "added=NEW_KEY" in rendered
    # Names only: a secrets file's content must never reach the report.
    for leaked in ("aaaa", "bbbb", "cccc", "dddd", "http://127.0.0.1:9001/v1"):
        assert leaked not in rendered
        assert leaked not in json.dumps(report)


def test_cron_ticker_stamps_move_but_user_cron_state_still_fails(tmp_path):
    """A live ticker rewrites its own liveness stamps; that is not user state.

    On a cold home those files appear as additions (tolerated). Once the ticker
    exists the same files register as modifications, which used to fail the user's
    upgrade outward for the harness's own background process.
    """
    home = tmp_path / "home"
    (home / "cron").mkdir(parents=True)
    (home / "profiles" / "p" / "cron").mkdir(parents=True)
    (home / "cron" / "ticker_heartbeat").write_text("100\n", encoding="utf-8")
    (home / "profiles" / "p" / "cron" / "ticker_last_success").write_text("100\n", encoding="utf-8")
    (home / "cron" / "jobs.json").write_text('{"jobs": []}\n', encoding="utf-8")
    snap = vus.snapshot_home(str(home))

    (home / "cron" / "ticker_heartbeat").write_text("200\n", encoding="utf-8")
    (home / "profiles" / "p" / "cron" / "ticker_last_success").write_text("200\n", encoding="utf-8")
    (home / "cron" / "jobs.json").write_text('{"jobs": [1]}\n', encoding="utf-8")
    report = vus.verify_home(str(home), snap)

    assert set(report["modified"]) == {"cron/jobs.json"}
    assert report["tolerated_modified"] == [
        "cron/ticker_heartbeat", "profiles/p/cron/ticker_last_success"]
    # The user's own cron DEFINITIONS are still state that must not move.
    assert report["ok"] is False


def test_a_dotenv_changed_only_in_comments_is_reported_as_such(tmp_path):
    """The per-key digests ignore comments, order and blanks by design.

    So a .env rewritten only in those parts reports as modified with no `variables`
    line -- the real "0 deleted, 1 modified" with nothing named. The report must say
    that explicitly, and still carry no values.
    """
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text("# one\nOPENAI_BASE_URL=http://127.0.0.1:9000/v1\n", encoding="utf-8")
    snap = vus.snapshot_home(str(home))

    (home / ".env").write_text("# one\n# two\nOPENAI_BASE_URL=http://127.0.0.1:9000/v1\n", encoding="utf-8")
    report = vus.verify_home(str(home), snap)

    assert set(report["modified"]) == {".env"}
    assert report["modified"][".env"]["key_diff"]["keys_changed"] == []
    rendered = vus._render(report)
    assert "no key differs" in rendered
    assert "http://127.0.0.1:9000/v1" not in rendered


def test_sqlite_sidecars_may_vanish_but_the_database_may_not(tmp_path):
    """A WAL/shm pair exists only while a connection is open.

    A snapshot taken with one live sees them; they vanish when it closes, and the
    report then reads as the upgrade DELETING user state (seen live on
    cron/executions.db-shm and -wal). The database itself must still fail if lost.
    """
    home = tmp_path / "home"
    (home / "cron").mkdir(parents=True)
    for name in ("executions.db", "executions.db-wal", "executions.db-shm"):
        (home / "cron" / name).write_text(f"{name}\n", encoding="utf-8")
    snap = vus.snapshot_home(str(home))

    os.remove(home / "cron" / "executions.db-wal")
    os.remove(home / "cron" / "executions.db-shm")
    report = vus.verify_home(str(home), snap)
    assert report["deleted"] == []
    assert report["tolerated_deleted"] == ["cron/executions.db-shm", "cron/executions.db-wal"]
    assert report["ok"] is True

    os.remove(home / "cron" / "executions.db")
    report = vus.verify_home(str(home), snap)
    assert report["deleted"] == ["cron/executions.db"]
    assert report["ok"] is False


def test_any_sqlite_database_is_judged_by_rows_not_bytes(tmp_path):
    """Live SQLite files churn bytes constantly; their contract is the rows.

    state.db was already handled that way; cron/executions.db was not, so a run
    writing rows mid-window would have failed the leg as a byte modification.
    """
    home = tmp_path / "home"
    (home / "cron").mkdir(parents=True)
    db = home / "cron" / "executions.db"
    with sqlite3.connect(db) as conn:
        # A table the verifier actually counts (COUNTED_TABLES), not an arbitrary one.
        conn.execute("create table cron_jobs (id integer primary key, at text)")
        conn.execute("insert into cron_jobs (at) values ('one')")
    snap = vus.snapshot_home(str(home))

    with sqlite3.connect(db) as conn:          # rows added: bytes move, contract holds
        conn.execute("insert into cron_jobs (at) values ('two')")
    report = vus.verify_home(str(home), snap)
    assert report["modified"] == {}
    assert report["rows_shrank"] == []
    assert report["ok"] is True

    with sqlite3.connect(db) as conn:          # rows lost: still fatal
        conn.execute("delete from cron_jobs")
    report = vus.verify_home(str(home), snap)
    assert report["rows_shrank"] == ["cron/executions.db"]
    assert report["ok"] is False


# --- retired provider vars ---------------------------------------------------

def test_retired_env_vars_come_from_the_migration_source(tmp_path):
    """The set is read from hermes_cli/config_migrations.py, not invented here.

    The 12 -> 13 migration clears LLM_MODEL/OPENAI_MODEL: the old setup wizard
    wrote them and nothing reads them now. A verifier carrying its own copy of
    that list would silently stop following the tree the moment it changes.
    """
    assert vus.retired_env_vars() == {"LLM_MODEL", "OPENAI_MODEL"}
    # Unreadable or unparseable source retires nothing, so every .env change
    # stays fatal rather than quietly becoming tolerated.
    assert vus.retired_env_vars(tmp_path / "absent.py") == frozenset()
    broken = tmp_path / "broken.py"
    broken.write_text("def (:\n", encoding="utf-8")
    assert vus.retired_env_vars(broken) == frozenset()


def test_a_retired_var_the_upgrade_empties_is_tolerated_and_named(tmp_path):
    """An old release's .env holds LLM_MODEL; the migration empties it.

    This verifier exists to catch an upgrade taking the user's state away. A key
    the tree itself retires is not that, so it is reported -- by name, never by
    value -- instead of failing the leg.
    """
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text(
        "LLM_MODEL=anthropic/claude-opus-4.6\nOPENROUTER_API_KEY=secret-value\n",
        encoding="utf-8",
    )
    snap = vus.snapshot_home(str(home))
    # Exactly the migration's write: same key, cleared value.
    (home / ".env").write_text(
        "LLM_MODEL=\nOPENROUTER_API_KEY=secret-value\n", encoding="utf-8"
    )

    report = vus.verify_home(str(home), snap)

    assert report["ok"] is True, vus._render(report)
    assert report["retired_env_cleared"] == {".env": ["LLM_MODEL"]}
    rendered = vus._render(report)
    assert "tolerated (retired var cleared: LLM_MODEL) .env" in rendered
    assert "secret-value" not in rendered
    assert "claude-opus" not in rendered


def test_a_live_key_cleared_by_an_upgrade_still_fails(tmp_path):
    """The exception is retired keys only -- emptying a real key is still loss."""
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text("OPENROUTER_API_KEY=secret-value\n", encoding="utf-8")
    snap = vus.snapshot_home(str(home))
    (home / ".env").write_text("OPENROUTER_API_KEY=\n", encoding="utf-8")

    report = vus.verify_home(str(home), snap)

    assert report["ok"] is False
    assert report["modified"][".env"]["key_diff"]["keys_changed"] == ["OPENROUTER_API_KEY"]


def test_a_retired_var_removed_outright_still_fails(tmp_path):
    """Clearing a retired key is the migration; deleting the line is not it."""
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text("LLM_MODEL=anthropic/claude-opus-4.6\n", encoding="utf-8")
    snap = vus.snapshot_home(str(home))
    (home / ".env").write_text("", encoding="utf-8")

    report = vus.verify_home(str(home), snap)

    assert report["ok"] is False
    assert report["modified"][".env"]["key_diff"]["keys_removed"] == ["LLM_MODEL"]
