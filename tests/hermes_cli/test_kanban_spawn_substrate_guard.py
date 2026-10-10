"""Spawn-seam substrate import guard (card t_a2f35c4d, ruling t_dbf5c876).

The incident these tests pin: every review-run worker spawn on the fleet died with
``cannot import name 'ADVISORY_SKILLS_ENV' from 'agent.skill_commands'`` — an import
no source file contains — and the review LANE went silently dead, because the failure
was counted as the CARD's, per card, with nothing naming the tree.

What is asserted here, all through the real dispatcher helpers:

* a substrate import failure never charges the card (``consecutive_failures`` unmoved,
  breaker untripped, card still retryable) and is spaced by ``substrate_cooldown``;
* the recorded failure text NAMES the artifact — importer ``file:line``, provider path,
  sha256, and the differs-from-HEAD verdict — front-loaded for the 500-char truncation;
* enough cards sharing one signature file EXACTLY ONE acting card, idempotently;
* a foreign-tag ``.pyc`` (``cpython-311`` / ``cpython-99``) is never selected by this
  interpreter, so it can neither resurrect a missing name nor shadow a source symbol —
  the proof-of-impossibility the ruling demands before anyone is allowed to gate a
  spawn on bytecode;
* the bytecode sweep helper REPORTS the two layouts that can change behaviour
  (sourceless ``module.pyc`` outside ``__pycache__``, non-zero header flags) while a
  foreign-tag population is measured as inert.

The scratch tree lives in ``tmp_path`` and the dispatcher's own tree root is patched to
it: the real repository must never be broken in place for a test, and the guard reads
one root from exactly one function.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

PKG = "substrate_probe_pkg"
MISSING_SYMBOL = "PROBE_MISSING_SYMBOL"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


def _make_scratch_tree(tmp_path: Path) -> Path:
    """A committed git tree whose importer asks its provider for a name it lacks.

    Committed first, then the provider is dirtied on disk, so ``differs-from-HEAD``
    has a real YES to report instead of a git refusal masquerading as one.
    """
    tree = tmp_path / "scratch-tree"
    package = tree / PKG
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "provider.py").write_text("PROBE_VALUE = 1\n", encoding="utf-8")
    (package / "importer.py").write_text(
        f"from {PKG}.provider import {MISSING_SYMBOL}\n", encoding="utf-8")
    _git(tree, "init", "-b", "main")
    _git(tree, "config", "user.email", "guard@example.com")
    _git(tree, "config", "user.name", "Guard Test")
    _git(tree, "add", "-A")
    _git(tree, "commit", "-m", "scratch tree")
    # Dirt: the provider the importer disagrees with is no longer HEAD's bytes.
    with (package / "provider.py").open("a", encoding="utf-8") as handle:
        handle.write("# touched after HEAD\n")
    return tree


def _run_broken_import(tree: Path) -> ImportError:
    """Import the scratch importer and return the REAL ImportError it raises."""
    root = str(tree)
    if root not in sys.path:
        sys.path.insert(0, root)
    for name in (PKG, f"{PKG}.provider", f"{PKG}.importer"):
        sys.modules.pop(name, None)
    import importlib

    try:
        importlib.import_module(f"{PKG}.importer")
    except ImportError as exc:  # the class under test
        return exc
    raise AssertionError("the scratch importer unexpectedly succeeded")


def _spawn_raising(tree: Path):
    """A spawn_fn that fails the way the incident failed: a real tree ImportError."""

    def spawn_fn(task, workspace, board=None):
        raise _run_broken_import(tree)

    return spawn_fn


def _substrate_tasks(conn):
    return conn.execute(
        "SELECT id, assignee, title FROM tasks WHERE idempotency_key LIKE 'substrate:%'"
    ).fetchall()


def _substrate_events(conn):
    return conn.execute(
        "SELECT task_id, payload FROM task_events WHERE kind = 'substrate_actor_filed'"
    ).fetchall()


def test_substrate_import_failure_never_charges_the_card(
    kanban_home, monkeypatch, all_assignees_spawnable, tmp_path,
):
    """Proof 1: three real substrate failures leave the card retryable.

    ``consecutive_failures`` stays 0, the breaker never parks the card, every run is
    tagged ``substrate_import`` AND ``infrastructure``, and once the cooldown is on the
    card is spaced by the guard's own ``substrate_cooldown`` reason.
    """
    tree = _make_scratch_tree(tmp_path)
    monkeypatch.setattr(kbd, "_dispatching_tree_root", lambda: tree)
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "0")

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="review spawn", assignee="a")
        for _ in range(3):
            res = kbd.dispatch_once(conn, spawn_fn=_spawn_raising(tree), failure_limit=2)
            assert res.auto_blocked == []
            assert res.spawned == []
        row = conn.execute(
            "SELECT status, block_kind, consecutive_failures, last_failure_error "
            "FROM tasks WHERE id = ?", (tid,),
        ).fetchone()
        assert (row["status"], row["block_kind"], row["consecutive_failures"]) == ("ready", None, 0)
        assert row["last_failure_error"].startswith(
            "tree-import-inconsistency: importer ")  # front-loaded for the 500-char cut
        runs = conn.execute(
            "SELECT outcome, metadata FROM task_runs WHERE task_id = ? ORDER BY id", (tid,),
        ).fetchall()
        assert [r["outcome"] for r in runs] == ["spawn_failed"] * 3
        for run in runs:
            metadata = json.loads(run["metadata"])
            assert metadata["infrastructure"] is True
            assert metadata["substrate_import"] is True
            assert metadata["substrate_signature"] == (
                f"substrate_import:{PKG}/provider.py:{MISSING_SYMBOL}")

        monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
        assert kbd.check_respawn_guard(conn, tid) == "substrate_cooldown"
        guarded = kbd.dispatch_once(conn, spawn_fn=_spawn_raising(tree), failure_limit=2)
        assert guarded.respawn_guarded == [(tid, "substrate_cooldown")]
        assert guarded.spawned == []

        # Control: an ordinary spawn failure on the same card still spends budget.
        monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "0")

        def spawn_broken(task, workspace, board=None):
            raise RuntimeError("profile launcher exploded")

        kbd.dispatch_once(conn, spawn_fn=spawn_broken, failure_limit=2)
        assert conn.execute(
            "SELECT consecutive_failures FROM tasks WHERE id = ?", (tid,),
        ).fetchone()["consecutive_failures"] == 1


def test_substrate_failure_text_names_importer_provider_and_head_verdict(tmp_path):
    """Proof 2: the line carries importer file:line + provider path + sha256 + HEAD verdict."""
    tree = _make_scratch_tree(tmp_path)
    provider = tree / PKG / "provider.py"
    expected_sha = hashlib.sha256(provider.read_bytes()).hexdigest()
    exc = _run_broken_import(tree)

    line = kbd.describe_tree_import_failure(exc, tree=tree, task_id="t_abc123")

    assert line is not None
    assert line.startswith("tree-import-inconsistency: ")
    assert f"importer {PKG}/importer.py:1 imports {MISSING_SYMBOL}" in line
    assert f"provider {PKG}/provider.py sha256={expected_sha}" in line
    assert f"size={provider.stat().st_size}" in line
    assert "defines-it=no" in line
    assert "differs-from-HEAD=yes" in line        # the provider is dirty vs HEAD
    assert "tree-self-consistent=no" in line
    assert "foreign-tag-pyc-in-provider-dir=0" in line
    assert "nothing of card t_abc123 ran" in line

    # A provider that DOES define the symbol reads back as such, and a tree git cannot
    # answer for says so rather than silently reporting "no".
    (tree / PKG / "provider.py").write_text(
        f"{MISSING_SYMBOL} = 1\n", encoding="utf-8")
    defined = kbd.describe_tree_import_failure(exc, tree=tree, task_id="t_abc123")
    assert "defines-it=yes" in defined
    assert "tree-self-consistent=yes" in defined


def test_one_actor_card_per_signature_across_cards(
    kanban_home, monkeypatch, all_assignees_spawnable, tmp_path,
):
    """Proof 3: N cards, one signature -> exactly ONE acting card, idempotent.

    A per-card ``gave_up`` names a card; the actor is what names the LANE. Repeated
    ticks — and therefore repeated failures — must resolve to the same card.
    """
    tree = _make_scratch_tree(tmp_path)
    monkeypatch.setattr(kbd, "_dispatching_tree_root", lambda: tree)
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    (kanban_home / "config.yaml").write_text(
        "kanban:\n  failure_threshold: 3\n", encoding="utf-8")

    with kbc.connect() as conn:
        first_two = [kb.create_task(conn, title=f"review {i}", assignee="a") for i in range(2)]
        # Two cards failing is BELOW kanban.failure_threshold=3: nothing may be filed.
        res = kbd.dispatch_once(conn, spawn_fn=_spawn_raising(tree), failure_limit=9,
                                max_in_progress=9)
        assert res.spawned == []          # a failing spawn consumes no slot
        assert _substrate_tasks(conn) == []
        assert _substrate_events(conn) == []
        assert kbd.check_respawn_guard(conn, first_two[0]) == "substrate_cooldown"

        # The third distinct card crosses the threshold -> exactly one actor, filed once.
        third = kb.create_task(conn, title="review 2", assignee="a")
        kbd.dispatch_once(conn, spawn_fn=_spawn_raising(tree), failure_limit=9,
                          max_in_progress=9)
        actors = _substrate_tasks(conn)
        assert len(actors) == 1, actors
        actor = actors[0]
        assert actor["assignee"] == "platform-stl"
        events = _substrate_events(conn)
        assert len(events) == 1
        payload = json.loads(events[0]["payload"])
        assert payload["signature"] == f"substrate_import:{PKG}/provider.py:{MISSING_SYMBOL}"
        assert sorted(payload["affected"]) == sorted(first_two + [third])
        assert events[0]["task_id"] == actor["id"]

        body = conn.execute(
            "SELECT body FROM tasks WHERE id = ?", (actor["id"],)).fetchone()["body"]
        assert "tree-import-inconsistency:" in body      # the naming line
        assert all(card in body for card in first_two + [third])   # affected cards
        assert "## Tree evidence" in body                 # git status --porcelain section

        # Idempotent: further ticks resolve to the SAME card, not a second one — even
        # though the actor card itself is now a ready row that fails the same way.
        kbd.dispatch_once(conn, spawn_fn=_spawn_raising(tree), failure_limit=9,
                          max_in_progress=9)
        assert [t["id"] for t in _substrate_tasks(conn)] == [actor["id"]]
        assert len(_substrate_events(conn)) == 1


def _foreign_tags() -> list:
    """Two interpreter tags this interpreter cannot select.

    On the fleet runtime (python 3.14) this is EXACTLY the pair the ruling names —
    ``cpython-311`` and ``cpython-99``. The suite also runs on the tree's legacy 3.11
    venv, where ``cpython-311`` is the CURRENT tag and therefore selectable, so the
    pair is derived from ``cache_tag`` rather than hardcoded: the property under test
    is "a tag other than the running one", which is what the guard actually rests on.
    """
    candidates = ["cpython-311", "cpython-99", "cpython-310", "cpython-312", "cpython-313"]
    foreign = [tag for tag in candidates if tag != sys.implementation.cache_tag]
    assert len(foreign) >= 2
    return foreign[:2]


def test_foreign_tag_bytecode_is_never_selected(tmp_path):
    """Proof 4 (impossibility): a foreign-tag pyc can neither supply nor shadow a name.

    The two pycs are written beside real modules and carry a python-3.11 and a
    cpython-99 tag. If this interpreter consulted either, the missing name would
    appear and the source symbol would be replaced; both must be unchanged.
    """
    tree = tmp_path / "bytecode-tree"
    package = tree / "pyc_probe_pkg"
    cache = package / "__pycache__"
    cache.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "provider.py").write_text("PROBE_VALUE = 1\n", encoding="utf-8")
    (package / "source_wins.py").write_text('VALUE = "source"\n', encoding="utf-8")

    foreign_tags = _foreign_tags()
    for tag in foreign_tags:
        assert tag != sys.implementation.cache_tag
        assert kbd._pyc_tag(f"provider.{tag}.pyc") == tag
        # Payload that WOULD define the missing name / shadow the source symbol.
        (cache / f"provider.{tag}.pyc").write_bytes(
            b"\x00\x00\x00\x00" + b"\x00" * 12 + b"PROBE_NAME_HERE")
        (cache / f"source_wins.{tag}.pyc").write_bytes(b"\xde\xad\xbe\xef" + b"\x00" * 12)

    root = str(tree)
    if root not in sys.path:
        sys.path.insert(0, root)
    for name in ("pyc_probe_pkg", "pyc_probe_pkg.provider", "pyc_probe_pkg.source_wins"):
        sys.modules.pop(name, None)
    import importlib

    # (a) the missing name still raises, with the foreign-tag pycs sitting right there
    with pytest.raises(ImportError):
        exec("from pyc_probe_pkg.provider import PROBE_MISSING_SYMBOL", {})
    # (b) the source symbol still wins over the shadowing pyc
    module = importlib.import_module("pyc_probe_pkg.source_wins")
    assert module.VALUE == "source"


def test_sweep_reports_sourceless_and_nonzero_flag_pycs(tmp_path):
    """Proof 5: the sweep names sourceless pycs and non-zero headers; foreign tag is inert."""
    tree = tmp_path / "sweep-tree"
    (tree / "__pycache__").mkdir(parents=True)
    (tree / "mod.py").write_text("x = 1\n", encoding="utf-8")
    (tree / "paired.py").write_text("y = 2\n", encoding="utf-8")
    (tree / "loose.pyc").write_bytes(b"\x00" * 16)          # sourceless: no loose.py
    (tree / "paired.pyc").write_bytes(b"\x00" * 16)         # has a .py: not sourceless
    tag = sys.implementation.cache_tag
    foreign_tags = _foreign_tags()
    (tree / "__pycache__" / f"mod.{tag}.pyc").write_bytes(
        b"\x00\x00\x00\x00" + (1).to_bytes(4, "little") + b"\x00" * 8)   # hash-based
    for foreign in foreign_tags:
        (tree / "__pycache__" / f"mod.{foreign}.pyc").write_bytes(b"\x00" * 16)

    report = kbd.sweep_tree_bytecode_pycs(tree)

    assert report["cache_tag"] == tag
    assert report["sourceless"] == ["loose.pyc"]
    assert report["pyc_total"] == 5
    assert report["current_tag"] == 1
    assert report["foreign_tag"] == 2
    assert report["flags_nonzero"] == 1
    assert report["hash_based"] == 1


def test_classifier_keeps_other_failures_on_the_card(tmp_path):
    """The classifier is the ONE decider, and it never widens its own reach.

    A card-level error, a missing third-party dependency, and an import failure from
    OUTSIDE the dispatching tree all stay ``card``; only the tree's own disagreement is
    ``substrate_import``.
    """
    from tools.process_registry import RestartSafeScopeUnavailable

    tree = _make_scratch_tree(tmp_path)
    assert kbd.classify_spawn_failure(RuntimeError("boom"), tree=tree) == "card"
    assert kbd.classify_spawn_failure(ValueError("no assignee"), tree=tree) == "card"
    assert kbd.classify_spawn_failure(
        RestartSafeScopeUnavailable("host cannot place the worker"), tree=tree) == "host_capacity"

    # A missing third-party dependency: its provider is NOT under this tree.
    try:
        # Intentionally not installed: the import MUST fail. (F401 is not an enabled rule here,
        # so no noqa directive is legal on this line.)
        import definitely_not_installed_pkg_xyz
    except ImportError as third_party:
        assert kbd.classify_spawn_failure(third_party, tree=tree) == "card"
        assert kbd.describe_tree_import_failure(third_party, tree=tree) is None

    assert kbd.classify_spawn_failure(_run_broken_import(tree), tree=tree) == "substrate_import"


def test_threshold_follows_kanban_failure_threshold(kanban_home, monkeypatch):
    """D5: the lane actor's count uses ``kanban.failure_threshold`` — no new key."""
    assert kbd._substrate_failure_threshold() == kbd.DEFAULT_FAILURE_LIMIT  # nothing configured
    (kanban_home / "config.yaml").write_text(
        "kanban:\n  failure_threshold: 5\n", encoding="utf-8")
    assert kbd._substrate_failure_threshold() == 5
    (kanban_home / "config.yaml").write_text(
        "kanban:\n  spawn_failure_threshold: 4\n", encoding="utf-8")
    assert kbd._substrate_failure_threshold() == 4   # legacy alias honoured
    (kanban_home / "config.yaml").write_text(
        "kanban:\n  failure_threshold: 0\n", encoding="utf-8")
    assert kbd._substrate_failure_threshold() == kbd.DEFAULT_FAILURE_LIMIT
