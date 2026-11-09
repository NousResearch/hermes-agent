"""scripts/move/migrate_branch.py carries a branch across move chunks on its target."""

import subprocess
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[2] / "scripts" / "move" / "migrate_branch.py"

# Stand-in for the import codemod with its CLI contract: renames come from commit A, and
# `import old` / `from old` lines are rewritten by longest dotted prefix.
STAND_IN_CODEMOD = '''
import argparse, re, subprocess
from pathlib import Path

def module(path):
    parts = path.removesuffix(".py").split("/")
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)

parser = argparse.ArgumentParser()
parser.add_argument("--from-commit", required=True)
parser.add_argument("--paths", nargs="+", default=[])
args = parser.parse_args()
status = subprocess.run(["git", "diff", "-M", "--name-status", f"{args.from_commit}^", args.from_commit],
                        check=True, capture_output=True, text=True).stdout
renames = {module(old): module(new) for kind, old, new in (line.split("\\t") for line in status.splitlines())
           if kind.startswith("R") and old.endswith(".py")}
rewritten = 0
for path in args.paths:
    text = Path(path).read_text()
    new = text
    for old in sorted(renames, key=len, reverse=True):
        new = re.sub(rf"(?m)^(\\s*(?:from|import)\\s+){re.escape(old)}(?=[\\s.]|$)", rf"\\g<1>{renames[old]}", new)
    if new != text:
        Path(path).write_text(new)
        rewritten += 1
moves = Path("hermes_agent/compat/moves.py")
moves.write_text(moves.read_text() + "".join(f"# {old} -> {new}\\n" for old, new in sorted(renames.items())))
print(f"rewritten={rewritten} flagged=0 failed=0")
'''


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, text=True, capture_output=True).stdout.strip()


def write(repo, path, text):
    (repo / path).parent.mkdir(parents=True, exist_ok=True)
    (repo / path).write_text(text)


def commit(repo, message):
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", message)


def migrate(repo, *args):
    return subprocess.run([sys.executable, str(TOOL), "--onto", "main", *args],
                          cwd=repo, check=False, capture_output=True, text=True, timeout=120)


@pytest.fixture
def repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "commit.gpgsign", "false")
    write(repo, "leaf.py", "def value():\n    return 1\n")
    write(repo, "user.py", "from leaf import value\nTOTAL = value()\n")
    write(repo, "tools_pkg/__init__.py", "")
    write(repo, "tools_pkg/core.py", "from leaf import value\n")
    write(repo, "app.py", "from tools_pkg.core import value\n")
    write(repo, "hermes_agent/__init__.py", "")
    write(repo, "hermes_agent/compat/moves.py", "MOVES = {}\n")
    write(repo, "notes.txt", "one\ntwo\n")
    write(repo, "scripts/move/rewrite_imports.py", STAND_IN_CODEMOD)
    commit(repo, "base")
    git(repo, "branch", "feature")
    return repo


def land_move(repo, name, moves, hand_fix=None):
    """Land a chunk on main: commit A renames, commit B rewrites imports, then a merge commit."""
    git(repo, "checkout", "-q", "-b", f"chunk-{name}", "main")
    for old, new in moves.items():
        (repo / new).parent.mkdir(parents=True, exist_ok=True)
        git(repo, "mv", old, new)
    commit(repo, f"move: {name}\n\nHermes-Move: {name}")
    rename = git(repo, "rev-parse", "HEAD")
    python_files = [path for path in git(repo, "ls-files").splitlines() if path.endswith(".py")]
    subprocess.run([sys.executable, "scripts/move/rewrite_imports.py", "--from-commit", rename,
                    "--paths", *python_files], cwd=repo, check=True, capture_output=True)
    commit(repo, f"move: rewrite imports for {name}\n\nHermes-Move: {name}")
    if hand_fix:
        write(repo, *hand_fix)
        commit(repo, f"move: hand fixes for {name}")
    git(repo, "checkout", "-q", "main")
    git(repo, "merge", "-q", "--no-ff", "-m", f"Merge chunk {name}", f"chunk-{name}")
    git(repo, "checkout", "-q", "feature")


def on_feature(repo, changes, message="feature work"):
    git(repo, "checkout", "-q", "feature")
    for path, text in changes.items():
        write(repo, path, text)
    commit(repo, message)


def read(repo, path):
    return (repo / path).read_text()


def test_branch_edit_follows_the_moved_file(repo):
    on_feature(repo, {"leaf.py": "def value():\n    return 2\n"})
    land_move(repo, "02-leaf", {"leaf.py": "hermes_agent/leaf.py"})

    result = migrate(repo)

    assert result.returncode == 0, result.stderr
    assert read(repo, "hermes_agent/leaf.py") == "def value():\n    return 2\n"
    assert not (repo / "leaf.py").exists()
    assert read(repo, "user.py").startswith("from hermes_agent.leaf import value\n")
    assert "Moves applied: 02-leaf" in result.stdout
    assert read(repo, "hermes_agent/compat/moves.py") == "MOVES = {}\n# leaf -> hermes_agent.leaf\n"
    assert git(repo, "status", "--porcelain") == ""


def test_new_file_importing_a_moved_module_is_rewritten(repo):
    on_feature(repo, {"extra.py": "import leaf\nfrom leaf import value\n"})
    land_move(repo, "02-leaf", {"leaf.py": "hermes_agent/leaf.py"})

    assert migrate(repo).returncode == 0
    assert read(repo, "extra.py") == "import hermes_agent.leaf\nfrom hermes_agent.leaf import value\n"
    assert "rewrite imports for move 02-leaf" in git(repo, "log", "--format=%s", "-5")


def test_edit_next_to_a_rewritten_import_keeps_the_branch_side(repo):
    on_feature(repo, {"user.py": "from leaf import value\nTOTAL = value() + 1\n"})
    land_move(repo, "02-leaf", {"leaf.py": "hermes_agent/leaf.py"})

    result = migrate(repo)

    assert result.returncode == 0, result.stderr
    assert read(repo, "user.py") == "from hermes_agent.leaf import value\nTOTAL = value() + 1\n"


def test_hand_fix_on_the_target_that_conflicts_stops_on_that_file(repo):
    on_feature(repo, {"user.py": "from leaf import value\nTOTAL = value() + 1\n"})
    land_move(repo, "02-leaf", {"leaf.py": "hermes_agent/leaf.py"},
              hand_fix=("user.py", "from hermes_agent.leaf import value\nTOTAL = value() * 2\n"))

    result = migrate(repo)

    assert result.returncode == 1
    assert "merge main" in result.stderr
    assert "user.py" in result.stderr


def test_hand_fix_deleting_a_line_the_branch_edited_stops(repo):
    write(repo, "user.py", "from leaf import value\nimport legacy\nTOTAL = value()\n")
    commit(repo, "legacy import")
    git(repo, "branch", "-f", "feature")
    on_feature(repo, {"user.py": "from leaf import value\nimport legacy  # still needed\nTOTAL = value()\n"})
    land_move(repo, "02-leaf", {"leaf.py": "hermes_agent/leaf.py"},
              hand_fix=("user.py", "from hermes_agent.leaf import value\nTOTAL = value()\n"))

    result = migrate(repo)

    assert result.returncode == 1
    assert "user.py" in result.stderr


def test_unrelated_conflict_stops_before_the_move_and_continue_finishes(repo):
    on_feature(repo, {"notes.txt": "one\nbranch\n", "extra.py": "from leaf import value\n"})
    git(repo, "checkout", "-q", "main")
    write(repo, "notes.txt", "one\nmain\n")
    commit(repo, "unrelated change")
    land_move(repo, "02-leaf", {"leaf.py": "hermes_agent/leaf.py"})

    stopped = migrate(repo)
    assert stopped.returncode == 1
    assert "merge the base of move 02-leaf" in stopped.stderr
    assert "notes.txt" in stopped.stderr

    write(repo, "notes.txt", "one\nboth\n")
    git(repo, "add", "notes.txt")
    resumed = migrate(repo, "--continue")

    assert resumed.returncode == 0, resumed.stderr
    assert read(repo, "notes.txt") == "one\nboth\n"
    assert read(repo, "extra.py") == "from hermes_agent.leaf import value\n"


def test_abort_returns_the_branch_to_where_it_started(repo):
    on_feature(repo, {"notes.txt": "one\nbranch\n"})
    start = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-q", "main")
    write(repo, "notes.txt", "one\nmain\n")
    commit(repo, "unrelated change")
    land_move(repo, "02-leaf", {"leaf.py": "hermes_agent/leaf.py"})
    assert migrate(repo).returncode == 1

    assert migrate(repo, "--abort").returncode == 0
    assert git(repo, "rev-parse", "HEAD") == start
    assert git(repo, "status", "--porcelain") == ""


def test_two_chunks_in_a_row(repo):
    on_feature(repo, {"extra.py": "from leaf import value\nfrom tools_pkg.core import value as v\n",
                      "tools_pkg/added.py": "import leaf\n"})
    land_move(repo, "02-leaf", {"leaf.py": "hermes_agent/leaf.py"})
    land_move(repo, "03-tools", {"tools_pkg": "hermes_agent/tools_pkg"})

    result = migrate(repo)

    assert result.returncode == 0, result.stderr
    assert "Moves applied: 02-leaf, 03-tools" in result.stdout
    assert read(repo, "extra.py") == ("from hermes_agent.leaf import value\n"
                                      "from hermes_agent.tools_pkg.core import value as v\n")
    assert read(repo, "hermes_agent/tools_pkg/added.py") == "import hermes_agent.leaf\n"
    assert not (repo / "tools_pkg").exists()
    assert git(repo, "rev-parse", "main") in git(repo, "rev-list", "HEAD").splitlines()


def test_rerun_after_success_changes_nothing(repo):
    on_feature(repo, {"extra.py": "from leaf import value\n"})
    land_move(repo, "02-leaf", {"leaf.py": "hermes_agent/leaf.py"})
    assert migrate(repo).returncode == 0
    done = git(repo, "rev-parse", "HEAD")

    again = migrate(repo)

    assert again.returncode == 0, again.stderr
    assert "Moves applied: none" in again.stdout
    assert git(repo, "rev-parse", "HEAD") == done


def test_codemod_that_cannot_run_stops_and_continue_runs_it_again(repo):
    on_feature(repo, {"extra.py": "from leaf import value\n"})
    land_move(repo, "02-leaf", {"leaf.py": "hermes_agent/leaf.py"})
    git(repo, "checkout", "-q", "main")
    codemod = read(repo, "scripts/move/rewrite_imports.py")
    write(repo, "scripts/move/rewrite_imports.py", "raise SystemExit('missing dependency')\n")
    commit(repo, "codemod needs a dependency")
    git(repo, "checkout", "-q", "feature")

    stopped = migrate(repo)
    assert stopped.returncode == 1
    assert "did not run" in stopped.stderr

    extracted = Path(git(repo, "rev-parse", "--git-path", "migrate-branch/codemod/scripts/move/rewrite_imports.py"))
    (repo / extracted).write_text(codemod)
    resumed = migrate(repo, "--continue")

    assert resumed.returncode == 0, resumed.stderr
    assert read(repo, "extra.py") == "from hermes_agent.leaf import value\n"


def test_abort_refuses_when_another_branch_is_checked_out(repo):
    on_feature(repo, {"notes.txt": "one\nbranch\n"})
    git(repo, "checkout", "-q", "main")
    write(repo, "notes.txt", "one\nmain\n")
    commit(repo, "unrelated change")
    land_move(repo, "02-leaf", {"leaf.py": "hermes_agent/leaf.py"})
    assert migrate(repo).returncode == 1
    git(repo, "merge", "--abort")
    main = git(repo, "rev-parse", "main")
    git(repo, "checkout", "-q", "main")

    result = migrate(repo, "--abort")

    assert result.returncode == 1
    assert "for feature" in result.stderr
    assert git(repo, "rev-parse", "main") == main


def test_two_chunks_landed_in_one_merge_are_both_applied(repo):
    on_feature(repo, {"extra.py": "from leaf import value\nfrom tools_pkg.core import value as v\n"})
    git(repo, "checkout", "-q", "-b", "stack", "main")
    for name, old, new in (("02-leaf", "leaf.py", "hermes_agent/leaf.py"),
                           ("03-tools", "tools_pkg", "hermes_agent/tools_pkg")):
        git(repo, "mv", old, new)
        commit(repo, f"move: {name}\n\nHermes-Move: {name}")
        rename = git(repo, "rev-parse", "HEAD")
        python_files = [path for path in git(repo, "ls-files").splitlines() if path.endswith(".py")]
        subprocess.run([sys.executable, "scripts/move/rewrite_imports.py", "--from-commit", rename,
                        "--paths", *python_files], cwd=repo, check=True, capture_output=True)
        commit(repo, f"move: rewrite imports for {name}\n\nHermes-Move: {name}")
    git(repo, "checkout", "-q", "main")
    git(repo, "merge", "-q", "--no-ff", "-m", "Merge chunks 02 and 03", "stack")
    git(repo, "checkout", "-q", "feature")

    result = migrate(repo)

    assert result.returncode == 0, result.stderr
    assert read(repo, "extra.py") == ("from hermes_agent.leaf import value\n"
                                      "from hermes_agent.tools_pkg.core import value as v\n")


def test_reverted_chunk_is_not_applied(repo):
    on_feature(repo, {"extra.py": "from leaf import value\n"})
    land_move(repo, "02-leaf", {"leaf.py": "hermes_agent/leaf.py"})
    git(repo, "checkout", "-q", "main")
    git(repo, "revert", "--no-edit", "-m", "1", "HEAD")
    git(repo, "checkout", "-q", "feature")

    result = migrate(repo)

    assert result.returncode == 0, result.stderr
    assert "Moves applied: none" in result.stdout
    assert read(repo, "extra.py") == "from leaf import value\n"
    assert (repo / "leaf.py").exists()
