#!/usr/bin/env python3
"""Carry the current branch across the module moves that landed on its target.

A move chunk on the target is a set of commits carrying the trailer ``Hermes-Move: NN-<name> <role>``:
commit A (role ``move``) renames files with ``git mv`` and updates ``hermes_agent/compat/moves.py``, commit B
(role ``codemod``) is the import rewrite, and hand fixes (role ``fix``, or no trailer) follow. Trailers without a
role are read as: the first commit is A, only its direct child carrying the trailer is B, and any later trailer
commit is a hand fix. For every chunk the branch has not merged yet, in landing order, this script

1. merges the commit before A, so ordinary conflicts surface apart from the move;
2. merges A with directory rename detection, so the branch's edits follow the moved files, then moves any file
   the branch added under a directory A moved whole (git leaves files in brand-new subdirectories behind);
3. runs ``scripts/move/rewrite_imports.py --from-commit A`` on the Python files the branch changed and commits
   the result for those files only;
4. merges B.

It then merges the target itself, which brings in the chunk's hand fixes. In steps 4 and 5 a conflicting hunk is
resolved only when one side changed nothing but module names there (both sides read with the moved names spelled
the old way); every other conflicting hunk stops with its file and line. Before it reports success it rewrites
moved names on the lines the branch added to docs (``scripts/move/rewrite_docs.py``) and lists every line the
branch added to a non-Python file that still names a moved module or path; those stop the run unless
``--accept-flags`` is given. Resolve a stop, commit or ``git add``, then run ``--continue``, or ``--abort`` to put
the branch back where it started. Merges only, no history rewrite. The remote is never touched unless ``--push``
is given; it pushes to the branch's own name on its remote and never to the remote's default branch.

Usage:
    python scripts/move/migrate_branch.py [--onto origin/main] [--push] [--accept-flags]
    python scripts/move/migrate_branch.py --continue [--accept-flags]
    python scripts/move/migrate_branch.py --abort
"""

from __future__ import annotations

import argparse
import difflib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

TRAILER = "Hermes-Move"
ROLES = ("move", "codemod", "fix")
CODEMOD = "scripts/move/rewrite_imports.py"
GIT_TIMEOUT_S = 600
CODEMOD_TIMEOUT_S = 1800
_TOKEN = re.compile(r"[\w.]+")
_SHEBANG_PYTHON = re.compile(rb"#!.*\bpython")
_DATA_DIRS = ("scripts/move/", "hermes_agent/compat/")


class Stop(Exception):
    """A step needs the user; the message says what to do."""


@dataclass(frozen=True)
class Chunk:
    name: str
    rename: str  # commit A
    rewrite: str | None  # commit B; None when the chunk has no import rewrite commit
    last: str  # the chunk's last commit carrying the trailer


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(["git", *args], check=False, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=GIT_TIMEOUT_S)
    if check and result.returncode != 0:
        raise Stop(f"git {' '.join(args)} failed:\n{result.stdout}{result.stderr}")
    return result


def out(*args: str) -> str:
    return git(*args).stdout.strip()


def is_ancestor(commit: str, of: str = "HEAD") -> bool:
    return git("merge-base", "--is-ancestor", commit, of, check=False).returncode == 0


def state_dir() -> Path:
    return Path(out("rev-parse", "--git-path", "migrate-branch")).resolve()


def load_state() -> dict | None:
    path = state_dir() / "state.json"
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else None


def save_state(state: dict) -> None:
    state_dir().mkdir(parents=True, exist_ok=True)
    (state_dir() / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def move_module(name: str):
    """A module of the target's ``scripts/move`` (renames, rewrite_docs), else the one beside this script."""
    tools = str(state_dir() / "codemod" / "scripts" / "move")
    if tools not in sys.path:
        sys.path[:0] = [tools]
        sys.path.append(str(Path(__file__).resolve().parent))
    try:
        return __import__(name)
    except ImportError:
        if name == "renames":
            raise Stop("The target's scripts/move/ has no renames.py; this migrate-branch needs a newer target.")
        return None


# Chunks.

def find_chunks(base: str, onto: str) -> list[Chunk]:
    """Move chunks between base and onto, in the order they landed on onto."""
    log = git("log", "--reverse", "--topo-order", "--no-merges",
              f"--format=%H%x09%P%x09%(trailers:key={TRAILER},valueonly,separator=%x2C)", f"{base}..{onto}").stdout
    commits: dict[str, list[tuple[str, str, list[str]]]] = {}
    for line in log.splitlines():
        sha, parents, value = (line.split("\t", 2) + ["", ""])[:3]
        words = value.strip().split()
        if not words:
            continue
        role = words.pop() if len(words) > 1 and words[-1] in ROLES else ""
        commits.setdefault(" ".join(words), []).append((sha, role, parents.split()))
    chunks = [chunk for name, found in commits.items() if _present((chunk := _chunk(name, found)).rename, onto)]
    mainline = out("rev-list", "--first-parent", "--reverse", f"{base}..{onto}").splitlines()
    return sorted(chunks, key=lambda chunk: mainline.index(_landing_commit(chunk.last, mainline, onto)))


def _chunk(name: str, found: list[tuple[str, str, list[str]]]) -> Chunk:
    if any(role for _, role, _ in found):
        moves = [sha for sha, role, _ in found if role == "move"]
        if len(moves) != 1:
            raise Stop(f"Move chunk {name} has {len(moves)} commits with `{TRAILER}: {name} move`; it needs one.")
        codemods = [sha for sha, role, _ in found if role == "codemod"]
        return Chunk(name, moves[0], codemods[-1] if codemods else None, found[-1][0])
    rename = found[0][0]
    child = next((sha for sha, _, parents in found[1:] if rename in parents), None)
    return Chunk(name, rename, child, found[-1][0])


def _landing_commit(last: str, mainline: list[str], onto: str) -> str:
    descendants = set(out("rev-list", "--ancestry-path", f"{last}..{onto}").splitlines())
    return next(sha for sha in mainline if sha == last or sha in descendants)


def _present(rename: str, onto: str) -> bool:
    """False for a reverted chunk: onto has every old path of commit A back and none of the new ones."""
    status = [line.split("\t") for line in out("diff", "-M", "--name-status", f"{rename}^", rename).splitlines()]
    renames = [paths for kind, *paths in status if kind.startswith("R")]
    edits = [paths for kind, *paths in status if kind.startswith("M")]
    if not renames or any(paths != ["hermes_agent/compat/moves.py"] for paths in edits):
        raise Stop(f"Commit {rename[:12]} is commit A of a move chunk but is not a rename; commit A must be "
                   "`git mv` only, plus the update of hermes_agent/compat/moves.py.")
    old_paths, new_paths = zip(*renames)
    def held(paths: tuple[str, ...]) -> int:
        return len(out("ls-tree", "-r", "--name-only", onto, "--", *paths).splitlines())
    return not (held(old_paths) == len(old_paths) and held(new_paths) == 0)


# Merging, with conflicts resolved only where one side changed nothing but module names.

def rename_config(commit: str) -> list[str]:
    limit = len(out("ls-tree", "-r", "--name-only", commit).splitlines()) + 1
    return ["-c", "merge.renames=true", "-c", "merge.directoryRenames=true",
            "-c", f"merge.renameLimit={limit}", "-c", f"diff.renameLimit={limit}",
            "-c", "merge.conflictStyle=diff3"]


def merge(commit: str, message: str, state: dict | None = None) -> None:
    """Merge ``commit``; with ``state``, resolve the hunks where one side only renamed modules."""
    if is_ancestor(commit):
        return
    result = git(*rename_config(commit), "merge", "--no-edit", "-m", message, commit, check=False)
    if result.returncode == 0:
        return
    unmerged = out("diff", "--name-only", "--diff-filter=U").splitlines()
    if not unmerged:
        raise Stop(f"Merge of {commit[:12]} failed without conflicts:\n{result.stdout}{result.stderr}")
    normalize = old_spelling(state) if state is not None else None
    left = [problem for path in unmerged if (problem := resolve_file(path, normalize)) is not None]
    if left:
        raise Stop(f"{message}: conflicts in\n  " + "\n  ".join(left)
                   + "\nResolve them, `git add` the files, then run --continue (or --abort).")
    git("commit", "--no-edit")


def old_spelling(state: dict):
    """A function that spells the moved names of every chunk in ``state`` with their old names again."""
    renames = move_module("renames")
    pairs: dict[str, str] = {}
    for rename in state["chunks"]:
        for old, new in renames.module_renames(Path.cwd(), rename).items():
            pairs[new] = old
            pairs[new.replace(".", "/")] = old.replace(".", "/")
        for old, new in renames.renamed_paths(Path.cwd(), rename):
            pairs[new] = old
    if not pairs:
        return lambda line: line
    pattern = re.compile(r"(?<![\w.])(" + "|".join(map(re.escape, sorted(pairs, key=len, reverse=True))) + r")(?!\w)")
    return lambda line: pattern.sub(lambda match: pairs[match.group(1)], line)


def resolve_file(path: str, normalize) -> str | None:
    """None when every conflicting hunk of ``path`` was resolved and added; otherwise what is left, for the user."""
    stages = {line.split()[2] for line in out("ls-files", "-u", "--", path).splitlines()}
    if normalize is None or stages != {"1", "2", "3"} or not Path(path).is_file():
        return path
    try:
        lines = Path(path).read_bytes().decode("utf-8").splitlines(keepends=True)
    except UnicodeDecodeError:
        return path
    merged: list[str] = []
    hunks, left, index = 0, [], 0
    while index < len(lines):
        if not lines[index].startswith("<<<<<<< "):
            merged.append(lines[index])
            index += 1
            continue
        hunk = _read_hunk(lines, index)
        if hunk is None:
            return f"{path} (conflict markers it cannot read)"
        end, ours, base, theirs = hunk
        hunks += 1
        resolved = resolve_hunk(base, ours, theirs, normalize)
        if resolved is None:
            left.append(len(merged) + 1)
            merged.extend(lines[index:end])
        else:
            merged.extend(resolved)
        index = end
    if not hunks:
        return path
    Path(path).write_bytes("".join(merged).encode("utf-8"))
    if left:
        where = ", ".join(map(str, left))
        return f"{path}: {len(left)} of {hunks} conflicting hunks change more than module names (line {where})"
    git("add", "--", path)
    print(f"resolved {hunks} conflicting hunk(s) in {path}: one side only renamed modules there")
    return None


def _read_hunk(lines: list[str], start: int):
    """(index after the hunk, ours, base, theirs) of the diff3 hunk at ``start``; None when malformed."""
    sections: list[list[str]] = [[], [], []]
    part = 0
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line.startswith("<<<<<<< "):
            return None
        if part == 0 and line.startswith("||||||| "):
            part = 1
        elif part == 1 and line.rstrip("\r\n") == "=======":
            part = 2
        elif part == 2 and line.startswith(">>>>>>> "):
            return index + 1, *sections
        else:
            sections[part].append(line)
    return None


def resolve_hunk(base: list[str], ours: list[str], theirs: list[str], normalize) -> list[str] | None:
    """The hunk's resolution when one side changed nothing but module names; None otherwise."""
    plain_base, plain_ours, plain_theirs = ([normalize(line) for line in side] for side in (base, ours, theirs))
    if plain_theirs == plain_base:
        return _one_sided(base, theirs, ours, plain_base, plain_ours)
    if plain_ours == plain_base:
        return _one_sided(base, ours, theirs, plain_base, plain_theirs)
    return ours if ours == theirs else None


def _one_sided(base: list[str], renamed: list[str], edited: list[str], plain_base: list[str],
               plain_edited: list[str]) -> list[str] | None:
    """Merge ``edited`` (any change) with ``renamed`` (only module names changed, line for line with ``base``).

    A line ``edited`` kept takes ``renamed``'s spelling; a line ``edited`` changed is kept unless it still spells
    a name ``renamed`` replaced on the base lines it changed.
    """
    merged: list[str] = []
    matcher = difflib.SequenceMatcher(None, plain_base, plain_edited, autojunk=False)
    for tag, first, last, edit_first, edit_last in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(last - first):
                old, new, mine = base[first + offset], renamed[first + offset], edited[edit_first + offset]
                if new == mine or old == new:
                    merged.append(mine)
                elif old == mine:
                    merged.append(new)
                else:
                    return None
            continue
        gone = {token for old, new in zip(base[first:last], renamed[first:last]) if old != new
                for token in set(_TOKEN.findall(old)) - set(_TOKEN.findall(new))}
        added = edited[edit_first:edit_last]
        if any(token in gone for line in added for token in _TOKEN.findall(line)):
            if last - first != edit_last - edit_first:
                return None
            added = [_rename_like(old, new, mine) for old, new, mine in zip(base[first:last], renamed[first:last], added)]
            if None in added or any(token in gone for line in added for token in _TOKEN.findall(line)):
                return None
        merged.extend(added)
    return merged


def _rename_like(old: str, new: str, mine: str) -> str | None:
    """``mine`` with the whole-token renames that turned ``old`` into ``new``; None unless that is all that changed."""
    old_parts, new_parts = _TOKEN.split(old), _TOKEN.split(new)
    old_tokens, new_tokens = _TOKEN.findall(old), _TOKEN.findall(new)
    if old_parts != new_parts or len(old_tokens) != len(new_tokens):
        return None
    mapping: dict[str, str] = {}
    for before, after in zip(old_tokens, new_tokens):
        if mapping.setdefault(before, after) != after:
            return None
    return _TOKEN.sub(lambda match: mapping.get(match.group(0), match.group(0)), mine)


# The branch's own files.

def relocate(chunk: Chunk, state: dict) -> None:
    """Move files the branch added under a directory commit A moved whole; git leaves new subdirectories behind."""
    if chunk.rename in state["relocated"]:
        return
    directories: dict[str, str] = {}
    for old, new in move_module("renames").renamed_paths(Path.cwd(), chunk.rename):
        old_dir, new_dir = PurePosixPath(old).parent, PurePosixPath(new).parent
        while str(old_dir) not in (".", str(new_dir)) and old_dir.name == new_dir.name:
            directories.setdefault(str(old_dir), str(new_dir))
            old_dir, new_dir = old_dir.parent, new_dir.parent
    moved = []
    for old_dir, new_dir in sorted(directories.items(), key=lambda item: len(item[0])):
        if out("ls-tree", "-r", "--name-only", chunk.rename, "--", old_dir):
            continue  # the directory did not move whole
        for path in out("ls-files", "--", f"{old_dir}/").splitlines():
            target = new_dir + path[len(old_dir):]
            if Path(target).exists():
                raise Stop(f"{path} is under {old_dir}/, which move {chunk.name} moved to {new_dir}/, but {target} "
                           "already exists. Move or merge it by hand, commit, then run --continue.")
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            git("mv", "--", path, target)
            moved.append(f"{path} -> {target}")
    if moved:
        git("commit", "-q", "-m", f"migrate-branch: move files left under old paths by move {chunk.name}")
        print("moved files git left under old paths:\n  " + "\n  ".join(moved))
    state["relocated"].append(chunk.rename)
    save_state(state)


def branch_python_files(rename: str) -> list[str]:
    changed = out("diff", "--name-only", "--diff-filter=AMR", rename, "HEAD").splitlines()
    return [path for path in changed if (path.endswith(".py") and Path(path).is_file()) or _python_script(path)]


def rewrite(chunk: Chunk, state: dict) -> None:
    if chunk.rename in state["rewritten"]:
        return
    paths = branch_python_files(chunk.rename)
    if paths:
        run_codemod(chunk, paths, state)
        commit_changes(paths, f"migrate-branch: rewrite imports for move {chunk.name}")
        git("checkout", "--", ".")  # the target's own commit B carries every other rewrite
    state["rewritten"].append(chunk.rename)
    save_state(state)


def run_codemod(chunk: Chunk, paths: list[str], state: dict) -> None:
    tool = state_dir() / "codemod"
    env = {**os.environ, "PYTHONPATH": str(tool)}
    result = subprocess.run([sys.executable, str(tool / CODEMOD), "--from-commit", chunk.rename, "--paths", *paths],
                            check=False, capture_output=True, text=True, encoding="utf-8", errors="replace",
                            env=env, timeout=CODEMOD_TIMEOUT_S)
    print(result.stdout + result.stderr, end="")
    summary = re.search(r"^rewritten=\d+ flagged=\d+ failed=(\d+)$", result.stdout, re.MULTILINE)
    if summary is None:
        raise Stop(f"The import rewrite for move {chunk.name} did not run (output above). "
                   "Fix the cause, then run --continue to run it again.")
    if int(summary.group(1)):
        state["pending_rewrite"] = chunk.rename
        save_state(state)
        raise Stop(f"The import rewrite for move {chunk.name} could not parse some files (above). Rewrite "
                   "their imports by hand, then run --continue (it commits the changes to tracked files).")


def commit_changes(paths: list[str], message: str) -> None:
    git("add", "--", *paths) if paths else git("add", "-u")
    if git("diff", "--cached", "--quiet", check=False).returncode != 0:
        git("commit", "-q", "-m", message)


def extract_codemod(onto: str) -> None:
    tool = state_dir() / "codemod"
    shutil.rmtree(tool, ignore_errors=True)
    archive = subprocess.run(["git", "archive", "--format=tar", onto, "scripts/move"], check=False,
                             capture_output=True, timeout=GIT_TIMEOUT_S)
    if archive.returncode != 0:
        raise Stop(f"{onto} has no scripts/move/: {archive.stderr.decode()}")
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        tar.extractall(tool, filter="data")


# Lines the branch added outside Python.

def added_lines(onto: str) -> dict[str, dict[int, str]]:
    """Path -> line number -> text of every line the branch adds over ``onto``, in non-Python files."""
    diff = out("diff", "-U0", "--no-color", "--no-renames", "--diff-filter=AM", onto, "HEAD")
    found: dict[str, dict[int, str]] = {}
    path, number = None, 0
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
            if path and (path.endswith(".py") or path.startswith(_DATA_DIRS) or _python_script(path)):
                path = None
        elif line.startswith("@@") and path:
            number = int(re.match(r"@@ -\S+ \+(\d+)", line).group(1))
        elif line.startswith("+") and path:
            found.setdefault(path, {})[number] = line[1:]
            number += 1
    return found


def _python_script(path: str) -> bool:
    file = Path(path)
    if file.suffix or not file.is_file():
        return False
    with file.open("rb") as handle:
        return bool(_SHEBANG_PYTHON.match(handle.readline()))


def rewrite_docs(state: dict) -> None:
    """Rewrite moved names on the lines the branch added to docs, with the target's docs rewriter."""
    docs = move_module("rewrite_docs")
    if docs is None:
        print("The target has no scripts/move/rewrite_docs.py; docs are checked below but not rewritten.")
        return
    changed = []
    for path, lines in added_lines(state["onto"]).items():
        if not path.endswith(docs.TEXT_SUFFIXES + docs.YAML_SUFFIXES):
            continue
        try:
            source = Path(path).read_bytes().decode("utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        text = source
        for rename in state["chunks"]:
            text = docs.rewrite_file(path, text, docs.commit_plan(Path.cwd(), rename), docs.Stats())
        before, after = source.split("\n"), text.split("\n")
        if len(before) != len(after):
            continue  # the check below lists its lines
        keep = [new if number in lines else old for number, (old, new) in enumerate(zip(before, after), 1)]
        if keep != before:
            Path(path).write_bytes("\n".join(keep).encode("utf-8"))
            changed.append(path)
    if changed:
        commit_changes(changed, "migrate-branch: rewrite moved names in docs the branch changed")
        print("rewrote moved names on the branch's lines in:\n  " + "\n  ".join(changed))


def stale_names(state: dict) -> list[str]:
    """``path:line`` for every non-Python line the branch added that names a moved module or path."""
    renames = move_module("renames")
    names: set[str] = set()
    paths: set[str] = set()
    for rename in state["chunks"]:
        names |= set(renames.moved_modules(Path.cwd(), rename)) | set(renames.module_renames(Path.cwd(), rename))
        for old, new in renames.renamed_paths(Path.cwd(), rename):
            paths.add(old)
            old_dir, new_dir = PurePosixPath(old).parent, PurePosixPath(new).parent
            if str(old_dir) not in (".", str(new_dir)):
                paths.add(f"{old_dir}/")
    def alternation(items) -> str:
        return "|".join(map(re.escape, sorted(items, key=len, reverse=True)))
    patterns = []
    if spelled := [name for name in names if "." in name or "_" in name]:
        patterns.append(re.compile(rf"(?<![\w./-])({alternation(spelled)})(?![\w-])"))
    if names:
        patterns.append(re.compile(rf"(?:\bfrom\s+|\bimport\s+|-m\s+|-m[\"'],\s*[\"'])({alternation(names)})(?![\w/-])"))
    if paths:
        patterns.append(re.compile(rf"(?<![\w./~$-])({alternation(paths)})"))
    hits = []
    for path, lines in sorted(added_lines(state["onto"]).items()):
        for number, text in lines.items():
            found = next((match.group(1) for pattern in patterns if (match := pattern.search(text))), None)
            if found:
                hits.append(f"{path}:{number}  names {found}: {text.strip()[:160]}")
    return hits


# Driving it.

def carry(state: dict) -> None:
    branch = state["branch"]
    for chunk in find_chunks(state["base"], state["onto"]):
        if is_ancestor(chunk.last):
            continue
        if chunk.rename not in state["chunks"]:
            state["chunks"].append(chunk.rename)
            save_state(state)
        merge(f"{chunk.rename}^", f"migrate-branch: merge the base of move {chunk.name} into {branch}")
        merge(chunk.rename, f"migrate-branch: merge the renames of move {chunk.name} into {branch}")
        relocate(chunk, state)
        rewrite(chunk, state)
        if chunk.rewrite:
            merge(chunk.rewrite, f"migrate-branch: merge the import rewrite of move {chunk.name} into {branch}", state)
    merge(state["onto"], f"migrate-branch: merge {state['onto_ref']} into {branch}", state)


def current_branch() -> str:
    branch = git("symbolic-ref", "--short", "-q", "HEAD", check=False).stdout.strip()
    if not branch:
        raise Stop("HEAD is detached. Check out the branch to migrate first (`git switch <branch>`).")
    return branch


def push_target(branch: str) -> list[str]:
    """The remote and ref ``--push`` writes; refuses up front rather than after the work."""
    remote = (git("config", f"branch.{branch}.pushRemote", check=False).stdout.strip()
              or git("config", "remote.pushDefault", check=False).stdout.strip()
              or git("config", f"branch.{branch}.remote", check=False).stdout.strip() or "origin")
    if remote == "." or git("remote", "get-url", remote, check=False).returncode != 0:
        raise Stop(f"--push: {branch} has no remote to push to (found {remote!r}); push it yourself afterwards.")
    head = git("symbolic-ref", "-q", f"refs/remotes/{remote}/HEAD", check=False).stdout.strip()
    if branch in {head.rsplit("/", 1)[-1] if head else "", "main", "master"}:
        raise Stop(f"--push: {branch} is the default branch of {remote}; migrate a feature branch instead.")
    return [remote, f"refs/heads/{branch}"]


def start(onto_ref: str, push: bool, accept_flags: bool) -> dict:
    if load_state() is not None:
        raise Stop("A migration is in progress: run --continue or --abort.")
    branch = current_branch()
    if out("status", "--porcelain", "--untracked-files=no"):
        raise Stop("The working tree has uncommitted changes; commit or stash them first.")
    onto = out("rev-parse", "--verify", f"{onto_ref}^{{commit}}")
    state = {"onto": onto, "onto_ref": onto_ref, "base": out("merge-base", "HEAD", onto),
             "orig_head": out("rev-parse", "HEAD"), "branch": branch, "push": push_target(branch) if push else None,
             "accept_flags": accept_flags, "chunks": [], "relocated": [], "rewritten": []}
    if find_chunks(state["base"], onto):
        extract_codemod(onto)
    save_state(state)
    return state


def load_own_state() -> dict:
    state = load_state()
    if state is None:
        raise Stop("No migration in progress.")
    if current_branch() != state["branch"] or not is_ancestor(state["orig_head"]):
        raise Stop(f"The migration in progress is for {state['branch']}; check it out first.")
    return state


def resume(accept_flags: bool) -> dict:
    state = load_own_state()
    state["accept_flags"] = state["accept_flags"] or accept_flags
    if out("diff", "--name-only", "--diff-filter=U"):
        raise Stop("Unmerged files remain; resolve them and `git add` them first.")
    if git("rev-parse", "-q", "--verify", "MERGE_HEAD", check=False).returncode == 0:
        git("commit", "--no-edit")
    elif pending := state.pop("pending_rewrite", None):
        commit_changes([], "migrate-branch: rewrite imports (completed by hand)")
        state["rewritten"].append(pending)
    save_state(state)
    return state


def abort() -> None:
    state = load_own_state()
    if git("rev-parse", "-q", "--verify", "MERGE_HEAD", check=False).returncode == 0:
        git("merge", "--abort")
    git("reset", "-q", "--hard", state["orig_head"])
    shutil.rmtree(state_dir())
    print(f"Restored {state['branch']} to {state['orig_head'][:12]}.")


def finish(state: dict) -> None:
    if state["chunks"]:
        rewrite_docs(state)
        hits = stale_names(state)
        if hits:
            print("Lines this branch adds outside Python that still name a moved module or path:\n  "
                  + "\n  ".join(hits), file=sys.stderr)
            if not state["accept_flags"]:
                raise Stop("Needs hand fix: rewrite the lines above to the new names (or keep them on purpose), "
                           "commit, then run --continue; --continue --accept-flags keeps them as they are.")
    applied = [chunk.name for chunk in find_chunks(state["base"], state["onto"])
               if not is_ancestor(chunk.last, state["orig_head"])]
    shutil.rmtree(state_dir())
    print(f"{state['branch']} is merged with {state['onto_ref']}. Moves applied: {', '.join(applied) or 'none'}.")
    if state["push"]:
        remote, ref = state["push"]
        git("push", remote, f"HEAD:{ref}")
        print(f"Pushed to {remote} {ref}.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--onto", default="origin/main", help="target that holds the moves (default origin/main)")
    parser.add_argument("--push", action="store_true", help="push to the branch's own name when done (never forced)")
    parser.add_argument("--accept-flags", action="store_true",
                        help="finish even when non-Python lines the branch adds still name moved modules")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--continue", dest="resume", action="store_true", help="resume after resolving a stop")
    action.add_argument("--abort", action="store_true", help="return the branch to where it started")
    args = parser.parse_args(argv)
    os.chdir(out("rev-parse", "--show-toplevel"))
    try:
        if args.abort:
            abort()
            return 0
        state = resume(args.accept_flags) if args.resume else start(args.onto, args.push, args.accept_flags)
        carry(state)
        finish(state)
    except Stop as stop:
        print(stop, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
