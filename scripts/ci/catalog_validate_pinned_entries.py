#!/usr/bin/env python3
"""Clone each changed catalog entry at its pinned sha and validate the plugin it names.

The supply-chain half of the plugin-catalog admission gate. For every entry path in
``$CHANGED_FILES`` (newline-separated) it refuses, in order:

* an entry file that is a symlink or missing;
* unusable PR-controlled fields (``catalog_entry_fields.py``);
* a repo that does not clone, or a pin that is not a reachable commit object;
* a subdir or any symlink in the clone that resolves outside the pinned clone;
* a subdir with no ``plugin.yaml``/``plugin.yml``/``plugin.json``;
* a failing ``hermes plugins validate --install-deps``;
* an unscannable payload: compiled binary, bytecode or archive hiding either (README rule 17);
* a self-updater (README rule 3).

Entries arrive in the environment, not on stdin, and every child runs with stdin closed:
``hermes plugins validate`` executes plugin-declared install code that could otherwise read stdin
and silently drain the remaining entry list. Run it with ``python3 -I``: cwd is the PR tree.

Exit 0 = every entry passed, 1 = at least one failed (one ``::error file=<entry>::`` line each).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_FIELDS_SCRIPT = Path(__file__).resolve().with_name("catalog_entry_fields.py")
_MANIFESTS = ("plugin.yaml", "plugin.yml", "plugin.json")
_SELF_UPDATE_FETCH = r"releases/latest|releases/download|raw\.githubusercontent\.com|codeload\.github\.com"
_SELF_UPDATE_WRITE = (r"writeTextFile|renamePath|writeFile\(|writeFileSync|renameSync|copyFileSync"
                      r"|cpSync|createWriteStream")
_SCRIPT_GLOBS = ("*.js", "*.mjs", "*.cjs", "*.ts", "*.jsx", "*.tsx", "*.mts", "*.cts")
# 600s: listed repos reach >1 GB (hermes-jackal-verified 1.43 GB, hindsight 741 MB); 120s would
# false-fail a re-pin on a slow runner.
_CLONE_SECONDS = 600
_VALIDATE_SECONDS = 600


class EntryRejected(Exception):
    """The entry fails the gate; the message is the ``::error`` text."""


def _run(argv: list[str], timeout: int | None = None, **kwargs) -> subprocess.CompletedProcess:
    if timeout is not None:
        argv = ["timeout", str(timeout), *argv]
    try:
        return subprocess.run(argv, stdin=subprocess.DEVNULL, check=False, **kwargs)
    except OSError:
        return subprocess.CompletedProcess(argv, 127, b"", b"")


def _substituted(output: bytes) -> str:
    # Shell command-substitution semantics the log lines were built with: NULs dropped, trailing
    # newlines stripped.
    return os.fsdecode(output.replace(b"\0", b"")).rstrip("\n")


_FIELDS = ("REPO", "SHA", "SUBDIR", "BAD")


def _read_fields(entry: str) -> dict[str, str]:
    """The entry's clone source from the isolated field reader; any malformed answer rejects it."""
    failed = dict.fromkeys(_FIELDS, "") | {"BAD": "catalog entry parser failed"}
    parsed = _run(["python3", "-I", str(_FIELDS_SCRIPT), entry], stdout=subprocess.PIPE)
    if parsed.returncode != 0:
        return failed
    try:
        fields = json.loads(parsed.stdout)
    except ValueError:
        return failed
    if not isinstance(fields, dict) or set(fields) != set(_FIELDS) \
            or not all(isinstance(value, str) for value in fields.values()):
        return failed
    return fields


def is_confined(path: str, root: str) -> bool:
    """Whether ``path``, with every symlink resolved, is ``root`` or lies under it."""
    resolved = os.path.realpath(path)
    return resolved == root or resolved.startswith(root + "/")


def escaping_symlink(clone_dir: str) -> str | None:
    """The first symlink anywhere in the clone (``.git`` aside) that resolves outside it.

    An unreadable directory counts as escaping: the walk cannot prove it confined.

    Confinement is per-symlink across the whole clone: the walk does not descend into a
    symlinked directory, so an allowed in-clone link like plugin/ext -> ../lib would otherwise
    hide lib/evil -> /outside from a plugin-dir-only scan while reads through it still escape
    the pin.
    """
    unreadable: list[str] = []
    for parent, dirnames, filenames in os.walk(clone_dir, onerror=lambda e: unreadable.append(e.filename)):
        names = dirnames + filenames
        if parent == clone_dir:
            names = [name for name in names if name != ".git"]
            dirnames[:] = [name for name in dirnames if name != ".git"]
        for name in names:
            link = os.path.join(parent, name)
            if os.path.islink(link) and not is_confined(link, clone_dir):
                return link
    return unreadable[0] if unreadable else None


def _checkout_pin(repo: str, sha: str, clone_dir: str) -> None:
    # Full clone (no --depth 1): the pinned sha may not be the branch tip.
    if _run(["git", "clone", repo, clone_dir], timeout=_CLONE_SECONDS).returncode != 0:
        raise EntryRejected(f"clone failed for {repo}")
    # The pin must name a commit object -- a tag-object sha would peel to a commit the reviewer
    # never declared as the pin.
    kind = _run(["git", "-C", clone_dir, "cat-file", "-t", sha],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if _substituted(kind.stdout) != "commit":
        raise EntryRejected(f"pinned sha {sha} is not a commit object in {repo}")
    # SUPPLY-CHAIN GATE: the pinned sha must be reachable in the repo.
    if _run(["git", "-C", clone_dir, "checkout", "--detach", sha]).returncode != 0:
        raise EntryRejected(f"pinned sha {sha} is not reachable in {repo}")


def _check_confinement(clone_dir: str, plugin_dir: str, subdir: str) -> None:
    # A symlinked directory inside the clone could still point outside it, so confine the
    # resolved path under the clone, not the lexical one.
    if not is_confined(plugin_dir, clone_dir):
        raise EntryRejected(f"subdir '{subdir}' resolves outside the pinned clone")
    if escaping_symlink(clone_dir) is not None:
        raise EntryRejected("a symlink in the repo resolves outside the pinned clone")


def _check_payloads(plugin_dir: str) -> None:
    # UNSCANNABLE-PAYLOAD GATE (README rule 17). A reviewer and the install scanner must be able
    # to read everything shipped at the pin as source: no compiled binaries (by header),
    # committed bytecode, or archives hiding either.
    scanner = os.path.join(os.environ["GITHUB_WORKSPACE"], "scripts/scan_plugin_payloads.py")
    scan = _run(["python3", scanner, plugin_dir], stdout=subprocess.PIPE)
    if scan.returncode != 0:
        # Filenames are repo-controlled: strip newlines so none can forge a workflow command in
        # the log.
        found = _substituted(scan.stdout).replace("\n", ";")
        raise EntryRejected(f"unscannable executable payload (README rule 17): {found}")


def _check_self_updater(plugin_dir: str) -> None:
    # SELF-UPDATER GATE (README rule 3): the pin is the only update path. A desktop bundle that
    # both fetches from GitHub AND writes/renames plugin files is a self-updater; either half
    # alone is fine (a plugin may read the API, or manage its own data files). NUL-delimited
    # both ways: committed filenames with spaces or newlines must not split into grep options
    # or forged log lines.
    includes = [f"--include={glob}" for glob in _SCRIPT_GLOBS]
    fetchers = _run(["grep", "-rlZE", _SELF_UPDATE_FETCH, *includes, plugin_dir],
                    stdout=subprocess.PIPE).stdout
    writers = subprocess.run(["xargs", "-r0", "grep", "-lZE", _SELF_UPDATE_WRITE],
                             input=fetchers, stdout=subprocess.PIPE, check=False,
                             timeout=_VALIDATE_SECONDS).stdout
    found = _substituted(writers).replace("\n", " ")
    if found:
        raise EntryRejected("self-updating code in catalog build (fetches GitHub AND writes "
                            f"plugin files): {found}")


def validate_entry(entry: str, workroot: str) -> None:
    """Run every gate on one entry; raises EntryRejected on the first failure."""
    # A symlinked or missing entry file is not a catalog entry.
    if os.path.islink(entry) or not os.path.isfile(entry):
        raise EntryRejected("catalog entry must be a regular file")
    fields = _read_fields(entry)
    # Check before echo: a rejected value must never reach the log, where a newline in it would
    # smuggle workflow commands like ::error::.
    if fields.get("BAD"):
        raise EntryRejected(fields["BAD"])
    repo, sha, subdir = fields.get("REPO", ""), fields.get("SHA", ""), fields.get("SUBDIR", "")
    print(f"repo={repo} sha={sha} subdir={subdir}")

    clone_dir = os.path.realpath(tempfile.mkdtemp(prefix="tmp.", dir=workroot))
    _checkout_pin(repo, sha, clone_dir)
    plugin_dir = f"{clone_dir}/{subdir}" if subdir else clone_dir
    _check_confinement(clone_dir, plugin_dir, subdir)
    # Native plugin.yaml OR portable Agent Plugins v1 plugin.json (#81196; native manifest wins
    # when both exist).
    if not any(os.path.isfile(f"{plugin_dir}/{name}") for name in _MANIFESTS):
        raise EntryRejected(f"no plugin.yaml or plugin.json at subdir '{subdir}' of {repo}@{sha}")
    # Manifest schema + declared-vs-registered capability check.
    validate = ["hermes", "plugins", "validate", "--install-deps", plugin_dir]
    if _run(validate, timeout=_VALIDATE_SECONDS).returncode != 0:
        raise EntryRejected("hermes plugins validate failed")
    _check_payloads(plugin_dir)
    _check_self_updater(plugin_dir)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Validate changed catalog entries at their pins.").parse_args(argv)
    # Line-buffered so log lines interleave with the children's output in order.
    sys.stdout.reconfigure(line_buffering=True, errors="surrogateescape")  # type: ignore[attr-defined]
    failed = False
    workroot = os.path.realpath(tempfile.mkdtemp(prefix="tmp."))
    for entry in filter(None, os.environ["CHANGED_FILES"].split("\n")):
        print(f"::group::validate {entry}")
        try:
            validate_entry(entry, workroot)
            print(f"✅ PASS: {entry}")
        except EntryRejected as rejected:
            print(f"::error file={entry}::{rejected}")
            failed = True
        print("::endgroup::")
    shutil.rmtree(workroot)

    if failed:
        print("❌ FAIL: one or more catalog entries failed pinned-source validation")
        return 1
    print("✅ PASS: all changed catalog entries validated at their pinned shas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
