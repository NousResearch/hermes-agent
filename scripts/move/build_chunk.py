#!/usr/bin/env python3
"""Build the mechanical part of one move chunk: commit A (git mv + MOVES) and commit B (generated).

usage: python3 scripts/move/build_chunk.py NN-name ROOT... [--log-dir DIR] [--only-a] [--from-a]

Run it with any Python 3.11+ (``just move-chunk`` does), never through ``scripts/hermes-python``: the
repository's environment tooling imports packages this command moves, so nothing that syncs the
environment can run once ``git mv`` has happened. Order:

  1. resolve the repository environment's interpreter and a libcst interpreter (before anything moves)
  2. git mv each ROOT into hermes_agent/, commit A, add the renames to MOVES (amends A)
  3. write the old-path files (generate_shims.py), then the codemod, the docs pass and the generators
  4. stage tracked changes plus the files the tools report they wrote, commit B

Every tool runs by absolute interpreter with PYTHONPATH set to the worktree root. A tree whose HEAD
is A finishes B when run again (``--from-a`` also discards the changes and untracked files a failed run left).
Exit status: 0 done; 1 refused, or the shim writer, codemod or docs pass crashed (no B is committed);
2 usage; 3 B is committed but a generator could not run (read its log: it is a hand-fix item).
The codemod exits 1 when it flags sites; that is not a crash. A crash is any other status, a missing
summary line, or ``failed`` above zero.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

EXIT_GENERATOR_FAILED = 3
TOOL_TIMEOUT = 3600  # seconds; the codemod takes minutes on the largest chunk
GIT_ENV = {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "core.hooksPath", "GIT_CONFIG_VALUE_0": os.devnull}
FLAG_LINE = re.compile(r"^[^ ]+:[0-9]+  ")
GENERATORS = [
    ("scripts/gen_gateway_contracts.py", [], False),
    ("scripts/gen_serve_handler_ledger.py", [], False),
    ("scripts/gen_plugin_api_docs.py", [], False),
    ("website/scripts/generate-skill-docs.py", [], False),  # its inputs are the SKILL.md files the docs pass rewrites
    ("scripts/gen_gateway_command_parity.py", ["--check"], True),
]


class Failure(Exception):
    pass


def run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None, log: Path | None = None,
        check: bool = True) -> subprocess.CompletedProcess[str]:
    full_env = {**os.environ, **GIT_ENV, **(env or {})}
    result = subprocess.run(cmd, cwd=cwd, env=full_env, text=True, encoding="utf-8", errors="replace",
                            capture_output=True, check=False, timeout=TOOL_TIMEOUT)
    if log is not None:
        log.write_text(result.stdout + result.stderr, encoding="utf-8")
    if check and result.returncode:
        raise Failure(f"{' '.join(cmd[:4])} exited {result.returncode}: {(result.stderr or result.stdout).strip()[-600:]}")
    return result


def git(root: Path, *args: str, check: bool = True) -> str:
    return run(["git", *args], cwd=root, check=check).stdout


def last_line(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-1].strip() if lines else ""


def resolve_interpreters(root: Path, saved: dict | None, python: str | None, cst_python: str | None) -> dict:
    """The repository environment (interpreter and variables) and a Python that has libcst, while pm still imports.

    The environment is a base interpreter plus PYTHONPATH pointing at the dependencies, so the variables
    the activation adds are kept with the interpreter.
    """
    resolved = dict(saved or {})
    if python:
        resolved["python"], resolved["env"] = python, resolved.get("env", {})
    if "python" not in resolved:
        code = "import json, os, sys; print(json.dumps({'python': sys.executable, 'env': dict(os.environ)}))"
        probe = run([str(root / "scripts/run-in-hermes-env"), "python", "-c", code], cwd=root)
        found = json.loads(last_line(probe.stdout))
        omitted = {"PWD", "OLDPWD", "SHLVL", "_"}
        # The test environment's venv interpreter carries its own packages; the base interpreter does not.
        resolved["python"] = found["env"].get("__HERMES_TEST_PYTHON") or found["python"]
        resolved["env"] = {k: v for k, v in found["env"].items() if os.environ.get(k) != v and k not in omitted}
    if not Path(resolved["python"]).exists():
        raise Failure(f"could not resolve the repository interpreter (got {resolved['python']!r}); pass --python")
    if cst_python:
        resolved["cst_python"] = cst_python
    if "cst_python" not in resolved:
        code = "import libcst, sys; print(sys.executable)"
        attempts = [
            [resolved["python"], "-m", "scripts.ci.python_packages", "libcst==1.9.0", "--", "-c", code],
            ["uv", "run", "--no-project", "--python", resolved["python"], "--with", "libcst==1.9.0", "python", "-c", code],
        ]
        errors = []
        for attempt in attempts:
            try:
                found = last_line(run(attempt, cwd=root, env=tool_env(root, resolved)).stdout)
            except (Failure, FileNotFoundError) as error:
                errors.append(str(error))
                continue
            if found and Path(found).exists():
                resolved["cst_python"] = found
                break
        if "cst_python" not in resolved:
            raise Failure("could not resolve an interpreter with libcst: " + " | ".join(errors))
    return resolved


def tool_env(root: Path, resolved: dict, *, isolated: bool = False) -> dict[str, str]:
    """Variables for a tool run: the resolved environment with the worktree root first on PYTHONPATH.

    ``isolated`` is for the codemod, which must see only its own libcst environment and the worktree.
    """
    env = {} if isolated else dict(resolved["env"])
    entries = [] if isolated else [e for e in env.get("PYTHONPATH", "").split(os.pathsep) if e and Path(e) != root]
    env["PYTHONPATH"] = os.pathsep.join([str(root), *entries])
    return env


def interpreters_file(log_dir: Path) -> Path:
    return log_dir / "interpreters.json"


def commit_trailer(root: Path, ref: str) -> str | None:
    match = re.search(r"^Hermes-Move: (.+)$", git(root, "log", "-1", "--format=%B", ref, check=False), re.MULTILINE)
    return match.group(1) if match else None


def clear_destination(root: Path, target: Path) -> None:
    """Delete ``target`` when it holds only ``__pycache__`` folders; git mv would nest the moved folder inside it.

    Anything else there is an ignored file git cannot restore, so the move stops and names it instead.
    """
    if target.is_dir():
        for cache in list(target.rglob("__pycache__")):
            shutil.rmtree(cache, ignore_errors=True)
        left = sorted(str(p.relative_to(root)) for p in target.rglob("*") if not p.is_dir())
        if not left:
            shutil.rmtree(target)
            return
    else:
        left = [str(target.relative_to(root))]
    raise Failure(f"{target.relative_to(root)} already holds files git does not track; move or delete them first: "
                  + ", ".join(left[:5]))


def move_roots(root: Path, name: str, roots: list[str]) -> str:
    for rel in roots:
        if Path(rel).anchor or ".." in Path(rel).parts:  # a Windows "\outside" has a root but no drive
            raise Failure(f"ROOT must be a path relative to the repository root: {rel}")
        if not (root / rel).exists():
            raise Failure(f"no such root: {rel}")
        target = root / "hermes_agent" / rel
        if git(root, "ls-files", "--", f"hermes_agent/{rel}").strip():
            raise Failure(f"hermes_agent/{rel} is already tracked")
        if target.exists():
            clear_destination(root, target)
    for rel in roots:
        (root / "hermes_agent" / rel).parent.mkdir(parents=True, exist_ok=True)
        git(root, "mv", rel, f"hermes_agent/{rel}")
        old = root / rel
        if old.is_dir():
            for cache in list(old.rglob("__pycache__")):
                shutil.rmtree(cache, ignore_errors=True)
            for folder in sorted((p for p in old.rglob("*") if p.is_dir()), reverse=True):
                try:
                    folder.rmdir()
                except OSError:
                    pass
            try:
                old.rmdir()
            except OSError:
                pass
    git(root, "commit", "-q", "-m", f"refactor: move {' '.join(roots)} into hermes_agent", "-m", f"Hermes-Move: {name} move")
    summary = update_moves_table(root)
    git(root, "add", "hermes_agent/compat/moves.py")
    git(root, "commit", "-q", "--amend", "--no-edit")
    return summary


def update_moves_table(root: Path) -> str:
    """Add the renames of the move commit at HEAD to MOVES; entries that pointed at a name moved again follow it."""
    sys.path.insert(0, str(root / "scripts/move"))
    try:
        from renames import current_name, module_renames
    finally:
        sys.path.pop(0)
    table_file = root / "hermes_agent/compat/moves.py"
    source = table_file.read_text(encoding="utf-8-sig")
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", None) == "MOVES")
    old = ast.literal_eval(node.value)
    new = module_renames(root, "HEAD")
    table = {key: current_name(value, new) or value for key, value in old.items()} | new
    body = "".join(f'    "{key}": "{value}",\n' for key, value in sorted(table.items()))
    lines = source.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = [f"MOVES: dict[str, str] = {{\n{body}}}\n"]
    table_file.write_text("".join(lines), encoding="utf-8")
    return f"MOVES += {len(new)}"


def untracked(root: Path) -> set[str]:
    return set(git(root, "ls-files", "--others", "--exclude-standard").splitlines())


def build_b(root: Path, name: str, a: str, log_dir: Path, resolved: dict) -> int:
    python, cst_python = resolved["python"], resolved["cst_python"]
    env = tool_env(root, resolved)
    before_untracked = untracked(root)
    written: set[str] = set()
    times: dict[str, float] = {}

    def step(label: str, cmd: list[str], log_name: str, ok_codes: tuple[int, ...] = (0,), step_env: dict[str, str] = env) -> subprocess.CompletedProcess[str]:
        started = time.time()
        result = run(cmd, cwd=root, env=step_env, log=log_dir / log_name, check=False)
        times[label] = time.time() - started
        print(f"{label} rc={result.returncode}")
        if result.returncode not in ok_codes:
            tail = (result.stderr or result.stdout).strip().splitlines()[-8:]
            raise Failure(f"{label} crashed (rc={result.returncode}); see {log_dir / log_name}\n" + "\n".join(tail))
        return result

    def write_shims(log_name: str) -> subprocess.CompletedProcess[str]:
        result = step("shims", [python, "scripts/move/generate_shims.py"], log_name)
        written.update(line[len("wrote "):] for line in result.stdout.splitlines() if line.startswith("wrote "))
        return result

    write_shims("shims.log")
    codemod = step("codemod", [cst_python, "scripts/move/rewrite_imports.py", "--from-commit", a], "codemod.log", (0, 1),
                   tool_env(root, resolved, isolated=True))
    # Exit 1 is how the codemod reports flags; a crash has no summary line.
    summary = re.search(r"^rewritten=(\d+) flagged=(\d+) failed=(\d+)$", codemod.stdout, re.MULTILINE)
    if codemod.returncode not in (0, 1) or summary is None or int(summary.group(3)):
        raise Failure(f"codemod crashed or failed on files; see {log_dir / 'codemod.log'}")
    docs_script = root / "scripts/move/rewrite_docs.py"
    if docs_script.is_file():
        step("docs", [python, "scripts/move/rewrite_docs.py", "--from-commit", a], "docs.log")
    else:
        (log_dir / "docs.log").write_text("skipped: scripts/move/rewrite_docs.py is not in this tree\n", encoding="utf-8")
        print("docs skipped: scripts/move/rewrite_docs.py is not in this tree")
    generate_log = []
    generate_failed = []
    started = time.time()
    for script, args, check_only in GENERATORS:
        if not (root / script).is_file():
            generate_log.append(f"== {script}\nskipped: not in this tree\n")
            continue
        result = run([python, script, *args], cwd=root, env=env, check=False)
        generate_log.append(f"== {script} {' '.join(args)}\n{result.stdout}{result.stderr}rc={result.returncode}\n")
        written.update(line[len("wrote "):] for line in result.stdout.splitlines() if line.startswith("wrote "))
        # A --check generator exits 1 for a table that needs a hand fix; anything else is a crash.
        if result.returncode and not (check_only and result.returncode == 1):
            generate_failed.append(script)
    times["generate"] = time.time() - started
    (log_dir / "generate.log").write_text("\n".join(generate_log), encoding="utf-8")
    ok = sum(1 for entry in generate_log if entry.rstrip().endswith("rc=0"))
    print(f"generate: {ok}/{len(GENERATORS)} rc=0 (a failing --check line here is a hand-fix item)")
    write_shims("shims-final.log")

    stray = sorted(untracked(root) - before_untracked - written)
    if stray:
        print(f"not staged (untracked files no tool reported writing): {', '.join(stray[:10])}"
              + (f" and {len(stray) - 10} more" if len(stray) > 10 else ""))
    git(root, "add", "-u")
    existing = sorted(rel for rel in written if (root / rel).exists())
    if existing:
        git(root, "add", "--", *existing)
    # --allow-empty: a chunk whose tools change nothing (a leaf nothing imports) still gets its commit B.
    git(root, "commit", "-q", "--allow-empty", "-m", f"refactor: rewrite old module names for the {name} move",
        "-m", f"Generated by scripts/move/build_chunk.py from {a}: scripts/move/generate_shims.py, rewrite_imports.py, "
              "rewrite_docs.py and the generators `just move-generate` runs.",
        "-m", f"Hermes-Move: {name} codemod")
    b = git(root, "rev-parse", "HEAD").strip()
    report(root, name, a, b, log_dir, codemod.stdout, times)
    if generate_failed:
        # B is committed: a generator that cannot run usually reads a path the codemod only flags (D47).
        print(f"generators failed, a hand-fix item: {', '.join(generate_failed)}; see {log_dir / 'generate.log'}",
              file=sys.stderr)
        return EXIT_GENERATOR_FAILED
    return 0


def tail(path: Path, count: int) -> str:
    return " ".join(path.read_text(encoding="utf-8-sig").splitlines()[-count:]) if path.exists() else ""


def report(root: Path, name: str, a: str, b: str, log_dir: Path, codemod_out: str, times: dict[str, float]) -> None:
    flags = [line for line in codemod_out.splitlines() if FLAG_LINE.match(line)]
    moved = sum(1 for line in git(root, "diff", "-M", "--name-status", f"{a}^", a).splitlines() if line.startswith("R"))
    changed = git(root, "diff", "--name-only", a, b).splitlines()
    shims = (log_dir / "shims.log").read_text(encoding="utf-8-sig").splitlines()
    lines = [
        f"NN={name} A={a} B={b}",
        f"moved={moved} files_changed_B={len(changed)} py_changed_B={sum(f.endswith('.py') for f in changed)} flagged={len(flags)}",
        f"codemod: {tail(log_dir / 'codemod.log', 2)}",
        f"docs: {tail(log_dir / 'docs.log', 3)}",
        f"shims: {sum(s.startswith('wrote') for s in shims)} written, {sum(s.startswith('deleted') for s in shims)} deleted",
        "time: " + " ".join(f"{k}={v:.0f}s" for k, v in times.items()),
    ]
    (log_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    classes: dict[str, int] = {}
    for line in flags:
        classes[line.split()[1]] = classes.get(line.split()[1], 0) + 1
    (log_dir / "flag-classes.txt").write_text(
        "".join(f"{n:7d} {k}\n" for k, n in sorted(classes.items(), key=lambda kv: -kv[1])), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("name", metavar="NN-name")
    parser.add_argument("roots", nargs="*", metavar="ROOT", help="paths at the repository root: a folder or a module file")
    parser.add_argument("--log-dir", type=Path, help="one log file per tool (default ~/chunk-logs/<worktree>/NN-name)")
    parser.add_argument("--only-a", action="store_true", help="stop after commit A")
    parser.add_argument("--from-a", action="store_true", help="HEAD is A: discard the changes and untracked files a failed run left, then build B")
    parser.add_argument("--python", help="the repository environment's interpreter (default: resolved before the move)")
    parser.add_argument("--cst-python", help="an interpreter with libcst (default: resolved before the move)")
    args = parser.parse_args(argv)
    try:
        root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").strip())
        log_dir = args.log_dir or Path(os.environ.get("CHUNK_LOG_ROOT", Path.home() / "chunk-logs" / root.name)) / args.name
        log_dir.mkdir(parents=True, exist_ok=True)
        trailer = commit_trailer(root, "HEAD")
        if trailer == f"{args.name} codemod":
            print(f"HEAD is already commit B of {args.name}")
            return 0
        resuming = trailer == f"{args.name} move"
        if args.from_a and not resuming:
            raise Failure(f"--from-a: HEAD has no 'Hermes-Move: {args.name} move' trailer")
        if not resuming and not args.roots:
            parser.error("at least one ROOT is required")
        remembered = interpreters_file(log_dir)
        saved = json.loads(remembered.read_text(encoding="utf-8-sig")) if resuming and remembered.is_file() else None
        dirty = git(root, "status", "--porcelain")
        if resuming:
            if dirty.strip() and args.from_a:
                git(root, "reset", "-q", "--hard", "HEAD")
                git(root, "clean", "-fdq")
            elif dirty.strip():
                raise Failure("HEAD is commit A and the tree is dirty; rerun with --from-a to discard what the failed run left\n" + dirty[:500])
        elif dirty.strip():
            raise Failure("dirty tree; commit or stash first\n" + dirty[:500])
        resolved = resolve_interpreters(root, saved, args.python, args.cst_python)
        remembered.write_text(json.dumps(resolved, indent=1), encoding="utf-8")
        print(f"interpreters: repo={resolved['python']} libcst={resolved['cst_python']}")
        started = time.time()
        if resuming:
            print(f"resuming from commit A {git(root, 'rev-parse', 'HEAD').strip()}")
        else:
            print(move_roots(root, args.name, args.roots))
        a = git(root, "rev-parse", "HEAD").strip()
        print(f"A={a} ({time.time() - started:.0f}s)")
        if args.only_a:
            return 0
        return build_b(root, args.name, a, log_dir, resolved)
    except Failure as error:
        print(f"build_chunk: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
