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
* an ``attested_binaries`` path that is not a regular file, or has no build-provenance attestation
  from the entry's repo at the pinned sha (README rule 17);
* an unscannable payload: compiled binary, bytecode or archive hiding either (README rule 17);
* a self-updater (README rule 3).

Entries arrive in the environment, not on stdin, and every child runs with stdin closed:
``hermes plugins validate`` executes plugin-declared install code that could otherwise read stdin
and silently drain the remaining entry list. Run it with ``python3 -I``: cwd is the PR tree.

Exit 0 = every entry passed, 1 = at least one failed (one ``::error file=<entry>::`` line each).
"""

from __future__ import annotations

import argparse
import hashlib
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
# A GitHub fetch. The pattern itself is plain (grep -E): the ajv exemption is applied in Python
# afterward, in _file_fetches_github, so it cannot be gamed by quoting/spacing of the ajv
# declaration (grep lookbehinds were brittle across minifiers) and cannot hide a real fetch that
# merely shares a line with the inert ajv id.
_SELF_UPDATE_FETCH = r"releases/latest|releases/download|raw\.githubusercontent\.com|codeload\.github\.com"
# ajv's bundled meta-schema declares its own identifier as `$id`/`$ref` = this URL. The string is
# matched by value against a schema ajv already holds in memory; it never drives a request. It is
# inert ONLY as a JSON-schema declaration -- `[$ or "]id/ref : "<url>"` -- so the same literal
# reused as an executable fetch base (new URL('/evil', ajvUrl)) is not exempt. The match is
# lexical: a declared id read back as a fetch base (new URL(p, s.$id)) is not caught, and
# escaped-JSON or backtick spellings stay flagged. "#" is the only terminator, so a traversal like
# .../data.json#/../../evil (or out of lib/refs) is a different, non-exempt URL and still counts.
_AJV_ID = "https://raw.githubusercontent.com/ajv-validator/ajv/master/lib/refs/data.json#"
_AJV_DECLARATION = re.compile(
    r"""["$](?:id|ref)["']?\s*:\s*["']""" + re.escape(_AJV_ID) + r"""["']""")
_GITHUB_FETCH = re.compile(_SELF_UPDATE_FETCH)


def _file_fetches_github(path: str) -> bool:
    # True when the file fetches from GitHub for real. ajv's inert meta-schema DECLARATION is
    # removed first (any quoting/spacing); whatever raw.githubusercontent / release URL remains is
    # a real fetch. Reading the file as text mirrors what grep -l would have matched.
    try:
        text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return True  # unreadable here means the grep already flagged it; keep it flagged
    return bool(_GITHUB_FETCH.search(_AJV_DECLARATION.sub("", text)))


_SELF_UPDATE_WRITE = (r"writeTextFile|renamePath|writeFile\(|writeFileSync|renameSync|copyFileSync"
                      r"|cpSync|createWriteStream")
_SCRIPT_GLOBS = ("*.js", "*.mjs", "*.cjs", "*.ts", "*.jsx", "*.tsx", "*.mts", "*.cts")
# Maintainer-reviewed exceptions to the self-updater gate: a file that downloads a THIRD-PARTY tool
# pinned by checksum (not the plugin's own code) and unpacks it into its cache. Keyed by repo +
# pinned sha + path + the file's sha256, so a re-pin or any edit to the file is gated again. It
# lives under scripts/ci/, which an entry PR may not touch (catalog_changed_entries.py).
_SELF_UPDATER_ALLOWLIST = Path(__file__).resolve().with_name("catalog_self_updater_allowlist.json")
# 600s: listed repos reach >1 GB (hermes-jackal-verified 1.43 GB, hindsight 741 MB); 120s would
# false-fail a re-pin on a slow runner.
_CLONE_SECONDS = 600
_VALIDATE_SECONDS = 600
_ATTEST_SECONDS = 120
_PROVENANCE = "https://slsa.dev/provenance/v1"


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


_FIELDS = ("REPO", "SHA", "SUBDIR", "ATTESTED", "BAD")


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


def _check_attested(attested: str, plugin_dir: str, repo: str, sha: str) -> None:
    # A plugin that needs a native component lists it under attested_binaries (README rule 17).
    # Each must have been built by THIS repo at the pinned commit; a verified one is removed from
    # the clone, so the payload scan that follows ignores it and any other binary still fails.
    # Provenance says where a file came from, not what it does: every check that reads the
    # plugin's files (the self-updater grep) must run before this removes any of them.
    owner_repo = repo.removeprefix("https://github.com/").removesuffix(".git").removesuffix("/")
    for path in filter(None, attested.split("\n")):
        binary = f"{plugin_dir}/{path}"
        if os.path.islink(binary) or not os.path.isfile(binary):
            raise EntryRejected(f"attested_binaries path is not a regular file: {path}")
        verify = ["gh", "attestation", "verify", binary, "--repo", owner_repo,
                  "--source-digest", sha, "--predicate-type", _PROVENANCE]
        if _run(verify, timeout=_ATTEST_SECONDS).returncode != 0:
            raise EntryRejected(f"no build-provenance attestation for {path} at "
                                f"{owner_repo}@{sha} (README rule 17)")
        os.remove(binary)


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


def _allowlisted(path: str, plugin_dir: str, repo: str, sha: str) -> bool:
    rel = os.path.relpath(path, plugin_dir)
    try:
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return False
    return any(row.get("repo") == repo and row.get("sha") == sha and row.get("path") == rel
               and row.get("sha256") == digest
               for row in json.loads(_SELF_UPDATER_ALLOWLIST.read_text(encoding="utf-8-sig")))


def _check_self_updater(plugin_dir: str, repo: str = "", sha: str = "") -> None:
    # SELF-UPDATER GATE (README rule 3): the pin is the only update path. A desktop bundle that
    # both fetches from GitHub AND writes/renames plugin files is a self-updater; either half
    # alone is fine (a plugin may read the API, or manage its own data files). NUL-delimited
    # both ways: committed filenames with spaces or newlines must not split into grep options
    # or forged log lines.
    includes = [f"--include={glob}" for glob in _SCRIPT_GLOBS]
    # grep -E gives a cheap first cut of files mentioning any GitHub-fetch URL; _file_fetches_github
    # then drops those whose only match is ajv's inert meta-schema DECLARATION. NUL-delimited so
    # committed filenames with spaces or newlines cannot split into grep options or forge log lines.
    candidates = _run(["grep", "-rlZE", _SELF_UPDATE_FETCH, *includes, plugin_dir],
                      stdout=subprocess.PIPE).stdout
    real = b"\0".join(p for p in candidates.split(b"\0")
                      if p and _file_fetches_github(os.fsdecode(p)))
    if not real:
        return
    writers = subprocess.run(["xargs", "-r0", "grep", "-lZE", _SELF_UPDATE_WRITE],
                             input=real, stdout=subprocess.PIPE, check=False,
                             timeout=_VALIDATE_SECONDS).stdout
    flagged = [p for p in writers.split(b"\0")
               if p and not _allowlisted(os.fsdecode(p), plugin_dir, repo, sha)]
    found = _substituted(b"\0".join(flagged)).replace("\n", " ")
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
    _check_self_updater(plugin_dir, repo, sha)
    _check_attested(fields["ATTESTED"], plugin_dir, repo, sha)
    _check_payloads(plugin_dir)


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
