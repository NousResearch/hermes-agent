"""The destructive kanban stress scripts must never reach a live board.

Incident: a ``tests/stress/test_concurrency*.py`` run set a temporary
``HERMES_HOME`` but left the ambient ``HERMES_KANBAN_DB`` in place. That env
var outranks ``HERMES_HOME`` in :func:`kanban_db.kanban_db_path` (it is
resolution step 1, the home is step 3), so every "isolated" synthetic worker
claimed, failed and completed cards on the real default board.

``HERMES_HOME`` alone is therefore not isolation. The shared guard under test
is fail-closed: a stress entry point must *prove* its resolved DB is a fresh
file inside a sandbox home it just created, or refuse to run.

These tests exercise the guard entirely against ``tmp_path`` — with
``tempfile.tempdir`` repointed there, so even the guard's own ``mkdtemp``
lands under pytest's cleanup. No real DB is opened and **no stress script is
executed**.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STRESS_DIR = REPO_ROOT / "tests" / "stress"
GUARD_PATH = STRESS_DIR / "_isolation.py"

#: Every destructive concurrency entry point that must be wired to the guard.
CONCURRENCY_SCRIPTS = (
    "test_concurrency.py",
    "test_concurrency_mixed.py",
    "test_concurrency_parent_gate.py",
    "test_concurrency_reclaim_race.py",
)


def _load_guard() -> ModuleType:
    """Import ``tests/stress/_isolation.py`` by path.

    ``tests/stress`` is deliberately not a package (its conftest sets
    ``collect_ignore_glob = ["*.py"]`` so pytest never collects the scripts),
    so this cannot be a plain import.
    """
    name = "hermes_stress_isolation_under_test"
    spec = importlib.util.spec_from_file_location(name, GUARD_PATH)
    assert spec and spec.loader, f"cannot load guard from {GUARD_PATH}"
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules[cls.__module__],
    # so the module has to be registered before it is executed.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@pytest.fixture
def guard() -> ModuleType:
    return _load_guard()


@pytest.fixture(autouse=True)
def _restore_environ():
    """Snapshot the whole environment.

    The guard mutates ``os.environ`` process-wide by design (the stress
    scripts import ``kanban_db`` after it runs), so ``monkeypatch.setenv``
    is not enough — it only restores names it was told about.
    """
    saved = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


@pytest.fixture
def sandbox_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Repoint the system temp root at a subdir of ``tmp_path``.

    A subdir, not ``tmp_path`` itself, so the tests still have somewhere
    that is *outside* the temp root to use for the negative cases.
    """
    root = tmp_path / "tmp"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


@pytest.fixture
def live_board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A stand-in for the operator's real ``~/.hermes`` board.

    Nothing in these tests may read or write it; it exists so the guard has
    a concrete "live board" to refuse.
    """
    real_home = tmp_path / "realhome"
    hermes = real_home / ".hermes"
    hermes.mkdir(parents=True)
    db = hermes / "kanban.db"
    db.write_bytes(b"SQLite format 3\x00")  # a board that already exists
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: real_home))
    monkeypatch.setenv("HERMES_REAL_HOME", str(real_home))
    return db


# ---------------------------------------------------------------------------
# the happy path: an isolated sandbox, pinned explicitly
# ---------------------------------------------------------------------------


def test_isolate_pins_an_explicit_db_inside_the_new_temp_home(
    guard, sandbox_root, live_board,
):
    """The DB must be an explicit path inside the home the guard just made."""
    sandbox = guard.isolate_stress_home("unit_guard_")

    assert sandbox.home.parent.resolve() == sandbox_root.resolve()
    assert sandbox.db.parent.resolve() == sandbox.home.resolve()
    # Pinned explicitly, not left to HERMES_HOME resolution.
    assert os.environ["HERMES_KANBAN_DB"] == str(sandbox.db)
    assert os.environ["HERMES_HOME"] == str(sandbox.home)
    # The guard isolates; creating the schema stays the script's job.
    assert not sandbox.db.exists()


def test_isolated_db_is_what_kanban_db_actually_resolves(
    guard, sandbox_root, live_board,
):
    """End-to-end: the production resolver must agree with the sandbox.

    This is the check that would have caught the incident — the scripts'
    ``HERMES_HOME`` looked isolated while ``kanban_db_path()`` returned the
    live board.
    """
    sandbox = guard.isolate_stress_home("unit_guard_")

    from hermes_cli import kanban_db as kb

    assert kb.kanban_db_path().resolve() == sandbox.db.resolve()


def test_isolate_overrides_an_ambient_live_kanban_db(
    guard, sandbox_root, live_board, monkeypatch: pytest.MonkeyPatch,
):
    """The incident, reproduced: a live ``HERMES_KANBAN_DB`` is inherited.

    Setting ``HERMES_HOME`` does not dislodge it — the guard must.
    """
    monkeypatch.setenv("HERMES_KANBAN_DB", str(live_board))

    sandbox = guard.isolate_stress_home("unit_guard_")

    from hermes_cli import kanban_db as kb

    assert os.environ["HERMES_KANBAN_DB"] != str(live_board)
    assert kb.kanban_db_path().resolve() == sandbox.db.resolve()
    assert kb.kanban_db_path().resolve() != live_board.resolve()


def test_isolate_scrubs_every_kanban_redirect_var(
    guard, sandbox_root, live_board, monkeypatch: pytest.MonkeyPatch,
):
    """No inherited kanban var may still point at the live tree.

    ``HERMES_KANBAN_WORKSPACES_ROOT`` and friends have the same
    env-beats-home precedence as the DB, and the per-worker identity vars
    (``HERMES_KANBAN_TASK`` / ``_RUN_ID`` / ``_CLAIM_LOCK``) name real rows
    on the live board.
    """
    live = live_board.parent
    ambient = {
        "HERMES_KANBAN_HOME": str(live),
        "HERMES_KANBAN_BOARD": "default",
        "HERMES_KANBAN_WORKSPACES_ROOT": str(live / "kanban" / "workspaces"),
        "HERMES_KANBAN_ATTACHMENTS_ROOT": str(live / "kanban" / "attachments"),
        "HERMES_KANBAN_TASK": "t_bdf1001a",
        "HERMES_KANBAN_RUN_ID": "255",
        "HERMES_KANBAN_CLAIM_LOCK": "some-host:80006",
        "HERMES_KANBAN_WORKSPACE": str(live / "worktrees" / "t_bdf1001a"),
        "HERMES_PROFILE": "backend",
    }
    for key, value in ambient.items():
        monkeypatch.setenv(key, value)

    sandbox = guard.isolate_stress_home("unit_guard_")

    home_str = str(sandbox.home.resolve())
    for key in ambient:
        value = os.environ.get(key)
        if value is None:
            continue
        assert str(Path(value).resolve()).startswith(home_str) or (
            not Path(value).is_absolute()
        ), f"{key}={value!r} still escapes the sandbox"
    for key in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID",
                "HERMES_KANBAN_CLAIM_LOCK"):
        assert key not in os.environ, f"{key} still names a live-board row"


def test_child_env_carries_only_the_isolated_db(
    guard, sandbox_root, live_board, monkeypatch: pytest.MonkeyPatch,
):
    """Spawned workers must inherit the sandbox, never the live board."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(live_board))

    sandbox = guard.isolate_stress_home("unit_guard_")
    child = sandbox.child_env()

    assert child["HERMES_KANBAN_DB"] == str(sandbox.db)
    assert child["HERMES_HOME"] == str(sandbox.home)
    assert child["HOME"] == str(sandbox.home)
    leaked = [
        f"{k}={v}" for k, v in child.items()
        if k.startswith("HERMES_KANBAN") and str(live_board.parent) in str(v)
    ]
    assert leaked == []


# ---------------------------------------------------------------------------
# the operator's real home must survive the HOME rewrite
# ---------------------------------------------------------------------------


def test_isolate_preserves_the_operator_real_home(
    guard, sandbox_root, live_board, tmp_path: Path,
):
    """``HERMES_REAL_HOME`` names the operator's home, not the sandbox.

    The guard rewrites ``HOME`` at the sandbox, which is precisely why
    ``HERMES_REAL_HOME`` exists: it is the only surviving pointer at the
    tree that must never be written. Overwriting it with the sandbox blinds
    :func:`_live_roots` in every process downstream of isolation.
    """
    real_home = tmp_path / "realhome"

    sandbox = guard.isolate_stress_home("unit_guard_")

    assert os.environ["HERMES_REAL_HOME"] == str(real_home)
    assert sandbox.env["HERMES_REAL_HOME"] == str(real_home)
    assert sandbox.child_env()["HERMES_REAL_HOME"] == str(real_home)
    # ...while everything that actually resolves a board stays sandboxed.
    assert os.environ["HOME"] == str(sandbox.home)
    assert os.environ["HERMES_HOME"] == str(sandbox.home)
    assert os.environ["HERMES_KANBAN_HOME"] == str(sandbox.home)
    assert os.environ["HERMES_KANBAN_DB"] == str(sandbox.db)


def test_isolate_falls_back_to_the_actual_home_when_real_home_is_unset(
    guard, sandbox_root, live_board, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Un-dispatched runs have no ``HERMES_REAL_HOME`` to inherit.

    The value has to be captured *before* ``HOME`` is rewritten, or the
    fallback records the sandbox and the live-board rule matches nothing.
    """
    monkeypatch.delenv("HERMES_REAL_HOME", raising=False)

    sandbox = guard.isolate_stress_home("unit_guard_")

    assert os.environ["HERMES_REAL_HOME"] == str(tmp_path / "realhome")
    assert sandbox.child_env()["HERMES_REAL_HOME"] == str(tmp_path / "realhome")


def test_a_child_of_the_sandbox_still_refuses_the_operator_live_board(
    guard, sandbox_root, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """The teeth: ``_live_roots`` must survive being inherited.

    A spawned worker sees the sandbox as its ``HOME``. If the parent also
    handed it ``HERMES_REAL_HOME=<sandbox>``, both live-root candidates
    collapse onto the sandbox and the "is this the real board?" rule can
    never fire again — the one check aimed at the incident becomes a no-op
    in exactly the processes that caused it.

    The stand-in real home lives *under* the temp root so the containment
    and marker rules pass; the live-board rule is then the only thing left
    that can refuse.
    """
    real_home = sandbox_root / "realhome"
    hermes = real_home / ".hermes"
    hermes.mkdir(parents=True)
    (hermes / guard.SANDBOX_MARKER).write_text("test")
    monkeypatch.setattr(
        Path, "home", classmethod(lambda cls: Path(os.environ["HOME"])),
    )
    monkeypatch.setenv("HERMES_REAL_HOME", str(real_home))

    sandbox = guard.isolate_stress_home("unit_guard_")
    child = guard  # a fresh interpreter re-reads the same module
    os.environ.clear()
    os.environ.update(sandbox.child_env())

    with pytest.raises(child.StressIsolationError):
        child.assert_disposable_sandbox(hermes, hermes / "kanban.db")


# ---------------------------------------------------------------------------
# fail-closed: it must refuse anything it cannot prove is disposable
# ---------------------------------------------------------------------------


def test_guard_refuses_a_home_outside_the_temp_root(
    guard, sandbox_root, live_board, tmp_path: Path,
):
    """A home that is not under the system temp root is not a sandbox.

    Marked and DB-free, so the temp-root rule is the only thing left to
    refuse on.
    """
    outside = tmp_path / "not_temp"
    outside.mkdir()
    (outside / guard.SANDBOX_MARKER).write_text("test")

    with pytest.raises(guard.StressIsolationError):
        guard.assert_disposable_sandbox(outside, outside / "kanban.db")


def test_guard_refuses_the_live_default_board(guard, sandbox_root, live_board):
    """``~/.hermes/kanban.db`` is the exact DB the incident wrote to."""
    with pytest.raises(guard.StressIsolationError):
        guard.assert_disposable_sandbox(live_board.parent, live_board)


def test_guard_refuses_a_db_outside_its_home(guard, sandbox_root, live_board):
    """Containment: the DB has to be inside the sandbox, not merely named."""
    home = Path(tempfile.mkdtemp(prefix="unit_guard_"))
    (home / guard.SANDBOX_MARKER).write_text("test")

    with pytest.raises(guard.StressIsolationError):
        guard.assert_disposable_sandbox(home, live_board)


def test_guard_refuses_an_already_existing_board(
    guard, sandbox_root, live_board,
):
    """A DB file that already exists is somebody's board, not a fresh one."""
    home = Path(tempfile.mkdtemp(prefix="unit_guard_"))
    (home / guard.SANDBOX_MARKER).write_text("test")
    db = home / "kanban.db"
    db.write_bytes(b"SQLite format 3\x00")

    with pytest.raises(guard.StressIsolationError):
        guard.assert_disposable_sandbox(home, db)


def test_guard_refuses_an_unmarked_home(guard, sandbox_root, live_board):
    """Only a home this guard created carries the marker.

    Without it, a child process handed an arbitrary path would adopt it.
    """
    home = Path(tempfile.mkdtemp(prefix="unit_guard_"))

    with pytest.raises(guard.StressIsolationError):
        guard.assert_disposable_sandbox(
            home, home / "kanban.db", require_marker=True,
        )


def test_guard_refuses_when_the_resolver_still_disagrees(
    guard, sandbox_root, live_board, monkeypatch: pytest.MonkeyPatch,
):
    """The post-set verification is what makes this future-proof.

    If a new env var, or a change to ``kanban_db_path``'s precedence, ever
    out-votes the pin again, the guard must refuse rather than run.
    """
    from hermes_cli import kanban_db as kb

    monkeypatch.setattr(kb, "kanban_db_path", lambda *a, **k: live_board)

    with pytest.raises(guard.StressIsolationError):
        guard.isolate_stress_home("unit_guard_")


# ---------------------------------------------------------------------------
# the child half
# ---------------------------------------------------------------------------


def test_adopt_reasserts_the_sandbox_in_a_child(
    guard, sandbox_root, live_board, monkeypatch: pytest.MonkeyPatch,
):
    """A spawned worker re-pins rather than trusting what it inherited."""
    sandbox = guard.isolate_stress_home("unit_guard_")
    sandbox.db.write_bytes(b"SQLite format 3\x00")  # parent ran init_db

    # A child that inherited a hostile environment despite everything.
    monkeypatch.setenv("HERMES_KANBAN_DB", str(live_board))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")

    adopted = guard.adopt_stress_home(str(sandbox.home))

    from hermes_cli import kanban_db as kb

    assert adopted.db.resolve() == sandbox.db.resolve()
    assert kb.kanban_db_path().resolve() == sandbox.db.resolve()


def test_adopt_refuses_a_non_sandbox_home(guard, sandbox_root, live_board):
    """A child pointed at the live tree must die, not adopt it."""
    with pytest.raises(guard.StressIsolationError):
        guard.adopt_stress_home(str(live_board.parent))


# ---------------------------------------------------------------------------
# every entry point is wired — a new script cannot quietly skip the guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", CONCURRENCY_SCRIPTS)
def test_every_concurrency_script_uses_the_guard(script: str):
    """All four destructive entry points must go through the helper."""
    source = (STRESS_DIR / script).read_text()

    assert "_isolation" in source, f"{script} does not import the guard"
    assert "isolate_stress_home(" in source, (
        f"{script} does not create an isolated sandbox"
    )


@pytest.mark.parametrize("script", CONCURRENCY_SCRIPTS)
def test_no_concurrency_script_sets_hermes_home_by_hand(script: str):
    """Bypass detector.

    Hand-rolled ``os.environ["HERMES_HOME"] = ...`` is exactly what looked
    like isolation and was not. Every home/DB assignment goes through the
    guard, in the parent and in every child loop.
    """
    source = (STRESS_DIR / script).read_text()

    for name in ("HERMES_HOME", "HOME", "HERMES_KANBAN_DB"):
        assert f'os.environ["{name}"] =' not in source, (
            f"{script} still assigns {name} by hand"
        )
        assert f"os.environ['{name}'] =" not in source


@pytest.mark.parametrize("script", CONCURRENCY_SCRIPTS)
def test_every_spawned_child_entry_point_adopts_the_sandbox(script: str):
    """Child *processes* re-assert; they never trust what they inherited.

    Thread-based scripts share the parent's ``os.environ`` and are covered
    by the parent's ``isolate_stress_home`` alone — spawned processes get a
    fresh interpreter, so each one re-runs the guard.
    """
    source = (STRESS_DIR / script).read_text()

    if "Process(" not in source:
        pytest.skip(f"{script} spawns no child processes")
    assert "adopt_stress_home(" in source, (
        f"{script} spawns workers that never re-assert isolation"
    )
