"""Trusted snapshot preparation: builds the approved snapshot and its receipt.

Launch only as:  python3 -I -S scripts/isolated_tests/prepare.py --inputs ... (stdlib only).
Order (approved-snapshot draft steps 1-11): load parent-owned trusted inputs
-> new 0700 dir -> `git ls-tree` of the base under an allowlisted env ->
raw blobs through one bounded `git cat-file --batch` stream (no archive, no
attributes, filters or export-subst; every body re-hashed to its blob id) ->
on-disk re-check against the ls-tree list -> candidate patches limited to
their manifest paths and hashes -> runner files copied by exact allowlist ->
exceptions only by the approved policy -> full re-walk and manifest ->
read-only -> re-verify -> receipt in a parent-owned location. Every failure
raises; nothing is skipped.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # append: stdlib wins

from isolated_tests import inputs  # noqa: E402
from isolated_tests.policy import PREPARE_LIMITS  # noqa: E402

Refused = inputs.Refused
_DIFF_HEADER = re.compile(r"^diff --git a/(\S+) b/(\S+)$")
_BAD_PATCH_LINES = ("rename from ", "rename to ", "copy from ", "copy to ",
                    "GIT binary patch", "Binary files ")
_LINK_MODES = ("120000", "160000")
_OID = re.compile(r"^[0-9a-f]{40}$")
_HEADER_CAP = 128  # one "<40-hex oid> blob <size>\n" header line, with room to spare


def _git_env(home: Path, cwd: Path) -> dict[str, str]:
    """Empty allowlisted env: no system/global config, prompts or helpers, and
    no discovery of an enclosing repository above cwd."""
    return {"PATH": "/usr/bin:/bin", "HOME": str(home), "LC_ALL": "C",
            "GIT_CEILING_DIRECTORIES": str(cwd.parent),
            "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_ATTR_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "/bin/false",
            "GIT_NO_REPLACE_OBJECTS": "1", "GIT_OPTIONAL_LOCKS": "0"}


_GIT_HARDENING = ("-c", "core.hooksPath=/dev/null", "-c", "credential.helper=",
                  "-c", "protocol.allow=never", "-c", "core.fsmonitor=false",
                  "-c", "core.attributesFile=/dev/null")


class Git:
    """The only process boundary. Tests substitute a fake with the same methods."""

    def __init__(self, git: str, repo: Path, home: Path):
        self.git, self.repo, self.home = git, repo, home

    def _run(self, args: list[str], cwd: Path, stdin: bytes | None, cap: int) -> bytes:
        argv = [self.git, *_GIT_HARDENING, *args]
        proc = subprocess.run(argv, cwd=cwd, env=_git_env(self.home, cwd), input=stdin,
                              capture_output=True, check=False, shell=False,
                              timeout=PREPARE_LIMITS["git_timeout_seconds"])
        if proc.returncode != 0:
            raise Refused(f"git {args[0]} failed rc={proc.returncode}")
        if len(proc.stdout) > cap:
            raise Refused(f"git {args[0]} output exceeds {cap} bytes")
        return proc.stdout

    def ls_tree(self, commit: str) -> bytes:
        return self._run(["--git-dir", str(self.repo), "ls-tree", "-r", "-z", "--full-tree",
                          commit], self.home, None, PREPARE_LIMITS["max_json_bytes"] * 4)

    @contextlib.contextmanager
    def cat_file_batch(self, oids: list[str]):
        """Yields git's stdout as a stream; the caller reads it with bounded reads, so
        nothing is buffered whole. A timer kills git at the git timeout (reads then see
        EOF and the caller refuses a truncated object)."""
        argv = [self.git, *_GIT_HARDENING, "--git-dir", str(self.repo), "cat-file", "--batch"]
        proc = subprocess.Popen(argv, cwd=self.home, env=_git_env(self.home, self.home),
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, shell=False)
        request = b"".join(oid.encode() + b"\n" for oid in oids)
        feeder = threading.Thread(target=_feed, args=(proc.stdin, request), daemon=True)
        timer = threading.Timer(PREPARE_LIMITS["git_timeout_seconds"], proc.kill)
        feeder.start()
        timer.start()
        try:
            yield proc.stdout
            rc = proc.wait(timeout=PREPARE_LIMITS["git_timeout_seconds"])
            if rc != 0:
                raise Refused(f"git cat-file failed rc={rc}")
        finally:
            timer.cancel()
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            proc.stdout.close()
            feeder.join()

    def apply(self, workdir: Path, patch: bytes) -> None:
        self._run(["apply", "--whitespace=nowarn", "-"], workdir, patch, 1 << 16)


def _feed(pipe, data: bytes) -> None:
    """Writes the oid list from a thread so a full stdout pipe cannot deadlock it."""
    try:
        pipe.write(data)
        pipe.close()
    except OSError:  # git was killed early (refusal/timeout); the reader reports why
        pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


class _Budget:
    def __init__(self):
        self.files = 0
        self.bytes = 0
        self.deadline = time.monotonic() + PREPARE_LIMITS["prepare_wall_seconds"]

    def charge(self, rel: str, size: int) -> None:
        self.files += 1
        self.bytes += size
        if size > PREPARE_LIMITS["max_file_bytes"]:
            raise Refused(f"prepare: file too large: {rel}")
        if self.files > PREPARE_LIMITS["max_files"] or self.bytes > PREPARE_LIMITS["max_total_bytes"]:
            raise Refused("prepare: file count or total bytes over the limit")
        self.check_time()

    def check_time(self) -> None:
        if time.monotonic() > self.deadline:
            raise Refused("prepare: wall-clock limit exceeded")


def _check_rel(rel: str) -> str:
    if (not inputs._safe_rel(rel) or "\\" in rel or "\0" in rel
            or len(rel.encode()) > PREPARE_LIMITS["max_path_bytes"]):
        raise Refused(f"prepare: unsafe path: {rel!r}")
    return rel


def _write_new(path: Path, data: bytes, executable: bool) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                 0o700 if executable else 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)


def parse_ls_tree(raw: bytes) -> list[tuple[str, bool, str]]:
    """(path, executable, blob oid) per entry; regular blobs only, paths checked."""
    entries: list[tuple[str, bool, str]] = []
    seen: set[str] = set()
    for rec in raw.split(b"\0"):
        if not rec:
            continue
        meta, sep, path = rec.decode("utf-8", "strict").partition("\t")
        fields = meta.split(" ")
        if not sep or len(fields) != 3:
            raise Refused(f"base tree: malformed ls-tree record: {rec[:80]!r}")
        mode, kind, oid = fields
        if kind != "blob" or mode in _LINK_MODES or mode not in ("100644", "100755"):
            raise Refused(f"base tree: non-regular entry refused: {path} ({mode} {kind})")
        if not _OID.match(oid):
            raise Refused(f"base tree: malformed object id for {path}")
        rel = _check_rel(path)
        if rel in seen:
            raise Refused(f"base tree: duplicate path: {rel}")
        seen.add(rel)
        entries.append((rel, mode == "100755", oid))
    return entries


def _batch_size(header: bytes, oid: str, rel: str) -> int:
    """Strict `<oid> blob <size>` header for the oid that was asked for."""
    if not header.endswith(b"\n"):
        raise Refused(f"cat-file: truncated or oversized header for {rel}")
    fields = header[:-1].split(b" ")
    if len(fields) == 2 and fields[0] == oid.encode() and fields[1] == b"missing":
        raise Refused(f"cat-file: missing object for {rel}")
    if (len(fields) != 3 or fields[0] != oid.encode() or fields[1] != b"blob"
            or not fields[2].isdigit()):
        raise Refused(f"cat-file: unexpected header for {rel}: {header[:80]!r}")
    return int(fields[2])


def extract_blobs(entries: list[tuple[str, bool, str]], stream, dest: Path,
                  budget: _Budget) -> dict[str, str]:
    """Raw blob bytes in ls-tree order; returns path -> sha256 (bodies are not kept)."""
    cap = PREPARE_LIMITS["max_total_bytes"] + _HEADER_CAP * (len(entries) + 1)
    consumed = 0
    tree: dict[str, str] = {}
    for rel, executable, oid in entries:
        header = stream.readline(_HEADER_CAP)
        size = _batch_size(header, oid, rel)
        budget.charge(rel, size)  # file size, count, total and wall: before the body is read
        consumed += len(header) + size + 1
        if consumed > cap:
            raise Refused("cat-file: response exceeds the total byte limit")
        body = stream.read(size)
        if len(body) != size or stream.read(1) != b"\n":
            raise Refused(f"cat-file: truncated or misframed object for {rel}")
        if _blob_id(body) != oid:
            raise Refused(f"base tree: bytes differ from git object id: {rel}")
        _write_new(dest / rel, body, executable)
        tree[rel] = _sha256(body)
    if stream.read(1):
        raise Refused("cat-file: unexpected trailing output")
    return tree


def verify_base_tree(root: Path, entries: list[tuple[str, bool, str]], budget: _Budget) -> None:
    """Re-walk after extraction: the files on disk are exactly the ls-tree list, each
    re-read and re-hashed to its blob id with the mode's exec bit. Not an independent
    source: the list and the bytes both come from the same object store, so this proves
    faithful extraction and writing, not that the object store is honest."""
    on_disk = _snapshot_tree(root, budget)
    listed = {rel: (executable, oid) for rel, executable, oid in entries}
    if set(on_disk) != set(listed):
        extra, missing = sorted(set(on_disk) - set(listed)), sorted(set(listed) - set(on_disk))
        raise Refused(f"base tree: disk != ls-tree (extra={extra[:5]}, missing={missing[:5]})")
    for rel, (executable, oid) in listed.items():
        if on_disk[rel][0] != executable or _blob_id((root / rel).read_bytes()) != oid:
            raise Refused(f"base tree: bytes or mode differ from git: {rel}")


def patch_paths(patch: bytes) -> set[str]:
    """Paths a patch touches; renames, copies, binary and link modes refused."""
    paths: set[str] = set()
    for line in patch.decode("utf-8", "strict").splitlines():
        if line.startswith(_BAD_PATCH_LINES) or any(
                line.startswith(f"{k} mode {m}") for k in ("new", "old", "new file", "deleted file")
                for m in _LINK_MODES):
            raise Refused(f"patch: unsupported header: {line[:80]!r}")
        found = _DIFF_HEADER.match(line)
        if found:
            if found.group(1) != found.group(2):
                raise Refused(f"patch: path change refused: {line[:80]!r}")
            paths.add(_check_rel(found.group(1)))
    if not paths:
        raise Refused("patch: no file headers")
    return paths


def _snapshot_tree(root: Path, budget: _Budget) -> dict[str, tuple[bool, str]]:
    """Re-walk: regular files only, single link, no setuid; returns exec bit + sha256."""
    out: dict[str, tuple[bool, str]] = {}
    budget.files = budget.bytes = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames + filenames:
            full = Path(dirpath) / name
            rel = _check_rel(full.relative_to(root).as_posix())
            st = os.lstat(full)
            if stat.S_ISDIR(st.st_mode):
                continue
            if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or st.st_mode & 0o6000:
                raise Refused(f"prepare: non-regular, linked or setuid file: {rel}")
            budget.charge(rel, st.st_size)
            out[rel] = (bool(st.st_mode & 0o111), inputs._sha256_file(full))
    return out


def apply_patches(root: Path, git: Git, trusted: inputs.TrustedInputs, budget: _Budget) -> None:
    for patch in trusted.patches:
        data = _read_pinned(patch.path, patch.sha256, PREPARE_LIMITS["max_patch_bytes"])
        allowed = dict(patch.manifest)
        touched = patch_paths(data)
        if touched != set(allowed):
            raise Refused(f"patch: touched paths != manifest ({sorted(touched ^ set(allowed))[:5]})")
        before = _snapshot_tree(root, budget)
        git.apply(root, data)
        after = _snapshot_tree(root, budget)
        for rel in set(before) | set(after):
            if rel in allowed:
                if rel not in after or after[rel][1] != allowed[rel]:
                    raise Refused(f"patch: result differs from manifest: {rel}")
            elif before.get(rel) != after.get(rel):
                raise Refused(f"patch: changed a path outside its manifest: {rel}")


def _read_pinned(raw: str, digest: str, cap: int) -> bytes:
    data = inputs._owned_regular_file(raw, "pinned input", cap)
    if len(data) > cap or _sha256(data) != digest:
        raise Refused(f"pinned input: sha256 mismatch or too large: {raw}")
    return data


def copy_runner(root: Path, source: Path, trusted: inputs.TrustedInputs, budget: _Budget) -> None:
    """Exactly the allowlisted new files; never a directory copy or glob."""
    for rel, digest in trusted.runner_allowlist:
        target = root / _check_rel(rel)
        if os.path.lexists(target):
            raise Refused(f"runner: allowlisted path already exists in the base: {rel}")
        data = _read_pinned(str(source / rel), digest, PREPARE_LIMITS["max_file_bytes"])
        budget.charge(rel, len(data))
        _write_new(target, data, rel.endswith(".sh"))


def check_final_tree(final: dict[str, tuple[bool, str]], expected: dict[str, str],
                     trusted: inputs.TrustedInputs) -> dict[str, str]:
    """Final path set == base + patches + runner; denied names only by policy."""
    if set(final) != set(expected):
        extra, missing = sorted(set(final) - set(expected)), sorted(set(expected) - set(final))
        raise Refused(f"final tree: unexpected paths (extra={extra[:5]}, missing={missing[:5]})")
    approved = dict(trusted.exceptions)
    applied: dict[str, str] = {}
    for rel, (_, digest) in final.items():
        if digest != expected[rel]:
            raise Refused(f"final tree: content differs: {rel}")
        if any(inputs._denied(part) for part in rel.split("/")):
            if inputs.forbidden_exception(rel) or approved.get(rel) != digest:
                raise Refused(f"final tree: denied name outside the approved policy: {rel}")
            applied[rel] = digest
    if len(applied) > trusted.max_exceptions:
        raise Refused("final tree: more exceptions than the policy allows")
    return applied


def _make_read_only(root: Path) -> None:
    for dirpath, dirnames, filenames in os.walk(root, topdown=False, followlinks=False):
        for name in filenames:
            full = Path(dirpath) / name
            os.chmod(full, 0o555 if os.lstat(full).st_mode & 0o100 else 0o444,
                     follow_symlinks=False)
        os.chmod(dirpath, 0o555)


def prepare(trusted_path: str, git: Git, runner_source: str, scratch: str,
            receipt_out: str) -> dict:
    trusted = inputs.load_trusted_inputs(trusted_path)                 # 1
    parent = inputs.real_dir(scratch, "scratch")
    root = parent / f"snapshot-{trusted.base_commit[:12]}-{os.urandom(4).hex()}"
    os.mkdir(root, 0o700)                                              # 2
    receipt_path = _receipt_target(receipt_out, root)
    budget = _Budget()
    entries = parse_ls_tree(git.ls_tree(trusted.base_commit))          # 3
    with git.cat_file_batch([oid for _, _, oid in entries]) as stream:
        base = extract_blobs(entries, stream, root, budget)            # 4
    verify_base_tree(root, entries, budget)                            # 5
    apply_patches(root, git, trusted, budget)                          # 6
    copy_runner(root, inputs.real_dir(runner_source, "runner source"), trusted, budget)  # 7
    expected = dict(base)
    for patch in trusted.patches:
        expected.update(dict(patch.manifest))
    expected.update(dict(trusted.runner_allowlist))
    final = _snapshot_tree(root, budget)                               # 9
    applied = check_final_tree(final, expected, trusted)               # 8
    manifest = {"schema": inputs.MANIFEST_SCHEMA, "source": "git-ls-files",
                "base_commit": trusted.base_commit,
                "candidate_patch_sha256": inputs.candidate_set_digest(trusted.patch_digests) or None,
                "candidate_patches_sha256": trusted.patch_digests,
                "files": {rel: {"sha256": d, "size": os.lstat(root / rel).st_size, "executable": x}
                          for rel, (x, d) in sorted(final.items())}}
    _write_new(root / inputs.MANIFEST_NAME,
               json.dumps(manifest, indent=1, sort_keys=True).encode(), False)
    _make_read_only(root)                                              # 10
    snap = inputs.verify_snapshot(str(root), applied)
    budget.check_time()
    receipt = {"schema": inputs.RECEIPT_SCHEMA, "result": "VERIFIED",
               "inputs_sha256": trusted.digest, "base_commit": trusted.base_commit,
               "candidate_patches_sha256": trusted.patch_digests,
               "no_candidate": not trusted.patches,
               "runner_allowlist_sha256": trusted.runner_digest(),
               "exception_policy_sha256": trusted.exception_policy_digest(),
               "prepare_limits_sha256": inputs._canon_sha(dict(PREPARE_LIMITS)),
               "applied_exceptions": applied, "manifest_sha256": snap.manifest_sha256,
               "snapshot_root": str(root), "file_count": snap.file_count}
    fd = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as fh:                                    # 11
        fh.write(json.dumps(receipt, indent=2, sort_keys=True).encode())
    return receipt


def _receipt_target(raw: str, snapshot: Path) -> Path:
    if not raw or not os.path.isabs(raw) or os.path.lexists(raw):
        raise Refused(f"receipt out: absolute path that does not exist yet required: {raw!r}")
    parent = inputs.real_dir(os.path.dirname(os.path.normpath(raw)), "receipt parent")
    if parent == snapshot or snapshot in parent.parents:
        raise Refused("receipt out: must be outside the snapshot")
    return parent / os.path.basename(os.path.normpath(raw))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="isolated-tests-prepare", allow_abbrev=False)
    p.add_argument("--inputs", required=True, help="parent-owned trusted inputs JSON")
    p.add_argument("--git", required=True, help="absolute git binary")
    p.add_argument("--repo", required=True, help="absolute git dir holding the base commit")
    p.add_argument("--runner-source", required=True, help="dir holding the allowlisted runner files")
    p.add_argument("--scratch", required=True, help="parent-owned dir for the new snapshot")
    p.add_argument("--receipt-out", required=True, help="new receipt path outside the snapshot")
    a = p.parse_args(argv)
    try:
        git_bin = inputs.docker_binary(a.git)  # same absolute executable-file rule
        repo = inputs.real_dir(a.repo, "repo")
        home = inputs.real_dir(a.scratch, "scratch")
        receipt = prepare(a.inputs, Git(str(git_bin), repo, home), a.runner_source,
                          a.scratch, a.receipt_out)
    except inputs.ProvenanceInvalid as exc:
        print(f"invalid provenance: {exc}", file=sys.stderr)
        return 8
    except (Refused, UnicodeDecodeError, subprocess.TimeoutExpired,
            OSError, ValueError) as exc:
        print(f"refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"manifest_sha256": receipt["manifest_sha256"],
                      "file_count": receipt["file_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
