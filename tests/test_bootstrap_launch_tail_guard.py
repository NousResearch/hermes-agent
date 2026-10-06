"""The launch tail of ``hermes_bootstrap`` must run only when an entry point is LAUNCHED.

``hermes_bootstrap`` is dual-use: entry-point modules import it, so its module body also
runs for a plain ``import``. Two things in that body are only correct for a process that
is *becoming* the CLI — finishing an armed self-update and ``os.execv``-ing the relaunched
interpreter. A library import inside another long-running program must do neither.

Measured incident (2026-10-01): a unit-test module imported the live CLI, the bootstrap
finished a real ``hermes update`` against the live checkout and then ``os.execv``-ed the
pytest session away; instrumentation saw three such routes from one test module
(2026-10-04).

The guard reads the IMPORTER's frame, not ``__name__`` of the bootstrap itself (that is
``hermes_bootstrap`` for both). An entry point imports ``hermes_bootstrap`` either as the
program itself (``python main.py``, ``-m hermes_cli.main``, ``runpy ... run_name="__main__"``)
or from a module the program imported directly — the console script's
``from hermes_cli.main import main`` and ``python -c "import <launcher>"``, a launch mode the
product itself preserves across a relaunch (``hermes_cli/venv_sync.py``: ``if argv[0] == "-c"``).
Deeper than that the caller is a library inside a longer-running process, and the tail is
withheld.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = REPO_ROOT / "hermes_bootstrap.py"

# ── stub modules the real bootstrap imports at module level ──────────────────
HERMES_CONSTANTS = "def export_scratch_tmp_env(*args, **kwargs):\n    return None\n"
PM_ENVIRONMENTS = """\
def activate_dependencies(project_root):
    return None


def install_state_permission_message(project_root, exc):
    return None


def install_state_dir(project_root):
    from pathlib import Path

    return Path(project_root) / ".hermes-install-state"
"""
EARLY_RECOVERY = "def recover_if_needed(project_root):\n    return None\n"
PARSER = "def command_argv(argv):\n    return list(argv)\n"
MAIN = "import hermes_bootstrap  # noqa: F401\n"
# The incident's shape: a module inside a longer-running program imports the CLI.
INCIDENT_HARNESS = "import hermes_cli.main  # noqa: F401\n"
UPDATE_HANDOFF = """\
def _continue_legacy_post_swap(handoff_path, argv_tail=None):
    return 0
"""
# The recorder: every call to the launch tail leaves a line in $REC_FILE, and
# REC_RETURN_PY makes the tail believe a relaunch is armed.
VENV_SYNC = """\
import os
from pathlib import Path


def completion_pending_path(project_root):
    return Path(project_root) / ".hermes-install-state" / "source-completion-pending"


def prepare_launch(project_root, argv):
    record = os.environ.get("REC_FILE")
    if record:
        with open(record, "a", encoding="utf-8") as handle:
            handle.write("prepare_launch\\n")
    if os.environ.get("REC_RETURN_PY"):
        return Path(os.environ["FAKE_PY"])
    return None


def relaunch_command(python, root, argv, orig_argv, module):
    return [str(python), "-c", "print('relaunched')"]
"""

_GUARD_MARKER = "def _launched_as_entry_point() -> bool:"


def _write(tree: Path, relpath: str, body: str) -> None:
    path = tree / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def _build_tree(tmp_path: Path, bootstrap_source: str) -> Path:
    tree = tmp_path / "tree"
    tree.mkdir()
    _write(tree, "hermes_bootstrap.py", bootstrap_source)
    _write(tree, "hermes_constants.py", HERMES_CONSTANTS)
    _write(tree, "incident_harness.py", INCIDENT_HARNESS)
    _write(tree, "pm/__init__.py", "")
    _write(tree, "pm/environments.py", PM_ENVIRONMENTS)
    _write(tree, "hermes_cli/__init__.py", "")
    _write(tree, "hermes_cli/_early_recovery.py", EARLY_RECOVERY)
    _write(tree, "hermes_cli/_parser.py", PARSER)
    _write(tree, "hermes_cli/main.py", MAIN)
    _write(tree, "hermes_cli/update_handoff.py", UPDATE_HANDOFF)
    _write(tree, "hermes_cli/venv_sync.py", VENV_SYNC)
    # Real sibling directories the bootstrap probes for private app surfaces.
    for dirname in ("delivery", "gui", "native"):
        (tree / "hermes_cli" / dirname).mkdir()
    return tree


def _run_child(tree: Path, args: list[str], home: Path,
               extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Launch ``args`` from inside ``tree`` with a clean environment.

    ``-S`` is load-bearing: the test interpreter is editable-installed, and its
    meta-path finder would otherwise win over the fake tree's modules.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if key in ("PATH", "LANG", "LC_ALL", "TZ", "TMPDIR", "SYSTEMROOT", "WINDIR")
    }
    env.pop("PYTEST_CURRENT_TEST", None)
    env.update({
        "PYTHONPATH": str(tree),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HERMES_HOME": str(home),
        "REC_FILE": str(home / "calls.log"),
    })
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-S", *args],
        cwd=str(tree), env=env, capture_output=True, text=True, timeout=60,
    )


def _calls(home: Path) -> list[str]:
    record = home / "calls.log"
    if not record.is_file():
        return []
    return record.read_text(encoding="utf-8").splitlines()


def _guarded_source() -> str:
    return BOOTSTRAP.read_text(encoding="utf-8")


def _unguarded(source: str) -> str:
    """Reproduce the pre-fix tail: force ``_launched_as_entry_point`` open.

    The pre-fix bytes live at the branch base. Reconstructing the behaviour from
    the guarded source keeps this control hermetic — it needs no commit that
    exists only in the fork and none that CI's shallow checkout would lack.
    """
    lines = source.splitlines(keepends=True)
    assert any(line.startswith(_GUARD_MARKER) for line in lines), (
        "the guard helper is absent from hermes_bootstrap.py — the negative control "
        "needs the guarded source to open"
    )
    start = next(i for i, line in enumerate(lines) if line.startswith(_GUARD_MARKER))
    end = next(i for i in range(start + 1, len(lines)) if lines[i] and not lines[i][0].isspace())
    replacement = (
        "def _launched_as_entry_point() -> bool:\n"
        '    """Forced open: the pre-fix tail ran on every import."""\n'
        "    return True\n\n\n"
    )
    return "".join([*lines[:start], replacement, *lines[end:]])


@pytest.fixture()
def guarded_tree(tmp_path: Path) -> Path:
    return _build_tree(tmp_path, _guarded_source())


@pytest.mark.platforms("posix")
def test_import_from_inside_another_program_does_not_run_the_launch_tail(
    tmp_path: Path, guarded_tree: Path
) -> None:
    """Arm 1 — the incident: a module inside a running program imports the CLI."""
    home = tmp_path / "home"
    home.mkdir()
    result = _run_child(guarded_tree, ["-c", "import incident_harness"], home)
    assert result.returncode == 0, result.stderr
    assert _calls(home) == []


@pytest.mark.platforms("posix")
def test_module_entry_point_still_runs_the_launch_tail(tmp_path: Path, guarded_tree: Path) -> None:
    """Arm 2 — the real CLI (``-m hermes_cli.main``) keeps the tail."""
    home = tmp_path / "home"
    home.mkdir()
    result = _run_child(guarded_tree, ["-m", "hermes_cli.main"], home)
    assert result.returncode == 0, result.stderr
    assert _calls(home) == ["prepare_launch"]


@pytest.mark.platforms("posix")
def test_command_launcher_still_runs_the_launch_tail(tmp_path: Path, guarded_tree: Path) -> None:
    """Arm 2b — ``-c "import <launcher>"`` is a launch mode the product preserves.

    ``hermes_cli/venv_sync.py`` reconstructs a relaunch for ``argv[0] == "-c"``, so a
    launcher imported directly by the program must keep the tail.
    """
    home = tmp_path / "home"
    home.mkdir()
    result = _run_child(guarded_tree, ["-c", "import hermes_cli.main"], home)
    assert result.returncode == 0, result.stderr
    assert _calls(home) == ["prepare_launch"]


@pytest.mark.platforms("posix")
def test_library_import_never_re_execs_even_when_a_relaunch_is_armed(
    tmp_path: Path, guarded_tree: Path
) -> None:
    """Arm 3 — with a relaunch armed, an in-process import must not re-exec nor warn."""
    home = tmp_path / "home"
    home.mkdir()
    result = _run_child(
        guarded_tree,
        ["-c", "import incident_harness\nprint('SENTINEL')"],
        home,
        {"REC_RETURN_PY": "1", "FAKE_PY": str(tmp_path / "fake-python")},
    )
    assert result.returncode == 0, result.stderr
    assert "SENTINEL" in result.stdout
    assert "hermes:" not in result.stderr


@pytest.mark.platforms("posix")
def test_withheld_path_still_announces_a_pending_update(tmp_path: Path, guarded_tree: Path) -> None:
    """A withheld import is not a silent one: armed completion state is reported once."""
    home = tmp_path / "home"
    home.mkdir()
    state = guarded_tree / ".hermes-install-state"
    state.mkdir()
    (state / "source-completion-pending").write_text("", encoding="utf-8")
    result = _run_child(guarded_tree, ["-c", "import incident_harness"], home)
    assert result.returncode == 0, result.stderr
    notices = [line for line in result.stderr.splitlines() if line.startswith("hermes:")]
    assert len(notices) == 1, result.stderr
    assert "hermes update" in notices[0]


@pytest.mark.platforms("posix")
def test_pending_notice_is_suppressed_inside_a_pytest_run(tmp_path: Path, guarded_tree: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    state = guarded_tree / ".hermes-install-state"
    state.mkdir()
    (state / "source-completion-pending").write_text("", encoding="utf-8")
    result = _run_child(
        guarded_tree, ["-c", "import incident_harness"], home,
        {"PYTEST_CURRENT_TEST": "tests/test_bootstrap_launch_tail_guard.py::x (call)"},
    )
    assert result.returncode == 0, result.stderr
    assert "hermes:" not in result.stderr


@pytest.mark.platforms("posix")
def test_negative_control_unguarded_source_reaches_the_tail_on_an_import(tmp_path: Path) -> None:
    """Arm 4 — proof the assertions bite: without the guard this route DOES run the tail."""
    tree = _build_tree(tmp_path, _unguarded(_guarded_source()))
    home = tmp_path / "home"
    home.mkdir()
    result = _run_child(tree, ["-c", "import incident_harness"], home)
    assert result.returncode == 0, result.stderr
    assert _calls(home) == ["prepare_launch"]
