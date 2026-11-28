"""Host-side validation of every path the launcher touches.

A path is accepted only after realpath + lstat checks; any symlink, special
file, broad root or mismatch with the snapshot manifest is a refusal, never a
warning.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import pwd
import re
import stat
from dataclasses import dataclass
from pathlib import Path

from isolated_tests.policy import PREPARE_LIMITS, SUITES


class Refused(Exception):
    """Input rejected; the message names the offending path and rule."""


MANIFEST_NAME = "ISOLATION-MANIFEST.json"
MANIFEST_SCHEMA = "hermes-isolated-snapshot/1"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

# Names never allowed inside a snapshot, matched against every path component.
# Matching is case-insensitive (patterns are lower-case) so `.ENV` or `ID_RSA`
# cannot slip past on a case-folding host; DENY_ALLOW stays an exact match.
DENY_NAMES = (
    ".git", ".env", ".env.*", ".netrc", ".npmrc", ".pypirc", ".docker",
    "auth.json", "credentials*", "*.pem", "*.key", "*.p12", "*.pfx",
    "id_rsa*", "id_ed25519*", "id_ecdsa*", "id_dsa*", "*.sqlite", "*.db", "state.db*",
    ".hermes", "sessions", "logs",
)
DENY_ALLOW = (".env.example",)


def account_home() -> Path:
    """The OS account home from the password database, never from $HOME."""
    return Path(os.path.realpath(pwd.getpwuid(os.getuid()).pw_dir))


def _broad_roots() -> tuple[Path, ...]:
    home = account_home()
    return (Path("/"), Path("/Users"), Path("/home"), Path("/private"),
            Path("/var"), Path("/tmp"), Path("/private/tmp"), home,
            home / ".hermes")


def real_dir(raw: str, what: str) -> Path:
    if not raw or not os.path.isabs(raw):
        raise Refused(f"{what}: must be an absolute path: {raw!r}")
    if os.path.islink(raw):
        raise Refused(f"{what}: must not be a symlink: {raw}")
    path = Path(os.path.realpath(raw))
    if path != Path(os.path.normpath(raw)):
        raise Refused(f"{what}: path traverses a symlink: {raw} -> {path}")
    try:
        st = os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        raise Refused(f"{what}: does not exist: {path}") from None
    if not stat.S_ISDIR(st.st_mode):
        raise Refused(f"{what}: not a directory: {path}")
    if path in _broad_roots():
        raise Refused(f"{what}: refusing broad root mount: {path}")
    if any(path == p or path in p.parents for p in _protected_homes()):
        raise Refused(f"{what}: contains a protected home: {path}")
    return path


def _protected_homes() -> tuple[Path, ...]:
    home = account_home()
    return (home / ".hermes" / "config.yaml", home / ".hermes" / ".env",
            home / ".ssh", home / ".docker")


def unix_socket(raw: str) -> Path:
    if not raw or not os.path.isabs(raw):
        raise Refused(f"docker socket: must be an absolute path: {raw!r}")
    path = Path(os.path.realpath(raw))
    try:
        st = os.stat(path)
    except FileNotFoundError:
        raise Refused(f"docker socket: does not exist: {path}") from None
    if not stat.S_ISSOCK(st.st_mode):
        raise Refused(f"docker socket: not a Unix socket: {path}")
    if st.st_uid not in (0, os.getuid()):
        raise Refused(f"docker socket: owned by uid {st.st_uid}, expected root or self")
    return path


def docker_binary(raw: str) -> Path:
    if not raw or not os.path.isabs(raw):
        raise Refused(f"docker binary: must be an absolute path: {raw!r}")
    path = Path(os.path.realpath(raw))
    try:
        st = os.stat(path)
    except FileNotFoundError:
        raise Refused(f"docker binary: does not exist: {path}") from None
    if not stat.S_ISREG(st.st_mode) or not os.access(path, os.X_OK):
        raise Refused(f"docker binary: not an executable regular file: {path}")
    if st.st_mode & stat.S_IWOTH:
        raise Refused(f"docker binary: world-writable: {path}")
    return path


def new_output_dir(raw: str, snapshot: Path) -> Path:
    if not raw or not os.path.isabs(raw):
        raise Refused(f"output dir: must be an absolute path: {raw!r}")
    if os.path.lexists(raw):
        raise Refused(f"output dir: must not exist yet: {raw}")
    parent = real_dir(os.path.dirname(os.path.normpath(raw)), "output parent")
    out = parent / os.path.basename(os.path.normpath(raw))
    if out == snapshot or snapshot in out.parents or out in snapshot.parents:
        raise Refused("output dir: must be disjoint from the source snapshot")
    return out


def _denied(name: str) -> bool:
    if name in DENY_ALLOW:
        return False
    return any(fnmatch.fnmatchcase(name.casefold(), pat) for pat in DENY_NAMES)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class Snapshot:
    root: Path
    manifest_sha256: str
    base_commit: str
    candidate_patch_sha256: str
    file_count: int


def _load_manifest(root: Path) -> tuple[dict, str]:
    mpath = root / MANIFEST_NAME
    if os.path.islink(mpath) or not mpath.is_file():
        raise Refused(f"snapshot: {MANIFEST_NAME} missing or not a regular file")
    raw = mpath.read_bytes()
    try:
        doc = json.loads(raw)
    except ValueError as exc:
        raise Refused(f"snapshot: manifest is not JSON: {exc}") from None
    if not isinstance(doc, dict) or doc.get("schema") != MANIFEST_SCHEMA:
        raise Refused(f"snapshot: manifest schema must be {MANIFEST_SCHEMA}")
    if doc.get("source") != "git-ls-files":
        raise Refused("snapshot: manifest source must be 'git-ls-files'")
    if not _HEX40.match(str(doc.get("base_commit", ""))):
        raise Refused("snapshot: base_commit must be a 40-hex commit")
    patch = doc.get("candidate_patch_sha256")
    if patch is not None and not _HEX64.match(str(patch)):
        raise Refused("snapshot: candidate_patch_sha256 must be 64-hex or null")
    files = doc.get("files")
    if not isinstance(files, dict) or not files:
        raise Refused("snapshot: manifest 'files' must be a non-empty object")
    return doc, hashlib.sha256(raw).hexdigest()


class ProvenanceInvalid(Refused):
    """Trusted inputs or the prepare receipt are malformed or do not match;
    the snapshot's provenance is unproven, so the run is INVALID."""


TRUSTED_INPUTS_SCHEMA = "hermes-isolated-prepare-inputs/1"
RECEIPT_SCHEMA = "hermes-isolated-prepare-receipt/1"
# Never exceptable, even by an approved policy: the policy cannot promote these.
FORBIDDEN_EXCEPTIONS = (
    ".git", ".env", ".env.*", "id_rsa*", "id_ed25519*", "id_ecdsa*", "id_dsa*",
    "*.pem", "*.key", "*.p12", "*.pfx", "auth.json", "*.sqlite", "*.db", "state.db*",
    ".netrc", ".pypirc", ".docker",
)


def _hex64_map(doc, what: str) -> dict[str, str]:
    if not isinstance(doc, dict):
        raise ProvenanceInvalid(f"{what}: must be an object of path -> sha256")
    for rel, digest in doc.items():
        if not isinstance(rel, str) or not _safe_rel(rel) or not _HEX64.match(str(digest)):
            raise ProvenanceInvalid(f"{what}: bad entry {rel!r}")
    return dict(doc)


def _safe_rel(rel: str) -> bool:
    parts = rel.split("/")
    return bool(rel) and not rel.startswith("/") and all(p not in ("", ".", "..") for p in parts)


def forbidden_exception(rel: str) -> bool:
    for name in rel.split("/"):
        if name in DENY_ALLOW:
            continue
        if any(fnmatch.fnmatchcase(name.casefold(), pat) for pat in FORBIDDEN_EXCEPTIONS):
            return True
    return False


@dataclass(frozen=True)
class CandidatePatch:
    path: str
    sha256: str
    manifest: tuple[tuple[str, str], ...]  # (repo path, final sha256)


@dataclass(frozen=True)
class TrustedInputs:
    """Parent-owned fixed inputs, kept outside the snapshot. Values are
    parameters (the base moves), never constants in this code."""
    digest: str
    base_commit: str
    patches: tuple[CandidatePatch, ...]
    runner_allowlist: tuple[tuple[str, str], ...]
    exceptions: tuple[tuple[str, str], ...]
    max_exceptions: int

    @property
    def patch_digests(self) -> list[str]:
        return [p.sha256 for p in self.patches]

    def runner_digest(self) -> str:
        return _canon_sha(dict(self.runner_allowlist))

    def exception_policy_digest(self) -> str:
        return _canon_sha({"allowed": dict(self.exceptions), "max_count": self.max_exceptions})


def _canon_sha(doc) -> str:
    raw = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _owned_regular_file(raw: str, what: str, cap: int) -> bytes:
    """Absolute, not a symlink, regular, owned by this uid, not group/world
    writable, bounded. Read through O_NOFOLLOW so a swap after lstat fails."""
    if not raw or not os.path.isabs(raw) or os.path.islink(raw):
        raise ProvenanceInvalid(f"{what}: absolute non-symlink path required: {raw!r}")
    try:
        fd = os.open(raw, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ProvenanceInvalid(f"{what}: cannot open {raw}: {exc.strerror}") from None
    with os.fdopen(fd, "rb") as fh:
        st = os.fstat(fh.fileno())
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid():
            raise ProvenanceInvalid(f"{what}: must be a regular file owned by this account")
        if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH) or st.st_size > cap:
            raise ProvenanceInvalid(f"{what}: group/world-writable or larger than {cap} bytes")
        return fh.read(cap + 1)


def parse_trusted_inputs(raw: bytes) -> TrustedInputs:
    try:
        doc = json.loads(raw)
    except ValueError:
        raise ProvenanceInvalid("trusted inputs: not JSON") from None
    if not isinstance(doc, dict) or doc.get("schema") != TRUSTED_INPUTS_SCHEMA:
        raise ProvenanceInvalid(f"trusted inputs: schema must be {TRUSTED_INPUTS_SCHEMA}")
    if not _HEX40.match(str(doc.get("base_commit", ""))):
        raise ProvenanceInvalid("trusted inputs: base_commit must be a 40-hex commit")
    raw_patches = doc.get("candidate_patches")
    if not isinstance(raw_patches, list):  # zero or more patches
        raise ProvenanceInvalid("trusted inputs: candidate_patches must be a list")
    patches = []
    for entry in raw_patches:
        if not isinstance(entry, dict) or not _HEX64.match(str(entry.get("sha256", ""))):
            raise ProvenanceInvalid("trusted inputs: each patch needs a 64-hex sha256")
        path = entry.get("path")
        if not isinstance(path, str) or not os.path.isabs(path):
            raise ProvenanceInvalid("trusted inputs: each patch needs an absolute path")
        manifest = _hex64_map(entry.get("manifest"), "trusted inputs: patch manifest")
        if not manifest:
            raise ProvenanceInvalid("trusted inputs: a patch manifest must list its paths")
        patches.append(CandidatePatch(path, entry["sha256"], tuple(sorted(manifest.items()))))
    runner = _hex64_map(doc.get("runner_allowlist"), "trusted inputs: runner_allowlist")
    policy_doc = doc.get("exception_policy")
    if not isinstance(policy_doc, dict):
        raise ProvenanceInvalid("trusted inputs: exception_policy must be an object")
    allowed = _hex64_map(policy_doc.get("allowed"), "trusted inputs: exception_policy.allowed")
    max_count = policy_doc.get("max_count")
    if type(max_count) is not int or not 0 <= max_count <= PREPARE_LIMITS["max_exceptions"]:
        raise ProvenanceInvalid("trusted inputs: exception_policy.max_count out of range")
    if len(allowed) > max_count:
        raise ProvenanceInvalid("trusted inputs: more approved exceptions than max_count")
    bad = sorted(rel for rel in allowed if forbidden_exception(rel))
    if bad:
        raise ProvenanceInvalid(f"trusted inputs: never-exceptable paths in policy: {bad[:5]}")
    return TrustedInputs(hashlib.sha256(raw).hexdigest(), doc["base_commit"], tuple(patches),
                         tuple(sorted(runner.items())), tuple(sorted(allowed.items())), max_count)


def load_trusted_inputs(raw: str) -> TrustedInputs:
    return parse_trusted_inputs(
        _owned_regular_file(raw, "trusted inputs", PREPARE_LIMITS["max_json_bytes"]))


def read_receipt(raw: str, trusted: TrustedInputs) -> dict:
    """Parse the prepare receipt and bind it to the trusted inputs. Any parse
    error is INVALID, never an empty default. Must run before the snapshot
    walk: the exceptions it returns come from here only."""
    try:
        doc = json.loads(_owned_regular_file(raw, "receipt", PREPARE_LIMITS["max_json_bytes"]))
    except ValueError:
        raise ProvenanceInvalid("receipt: not JSON") from None
    if not isinstance(doc, dict) or doc.get("schema") != RECEIPT_SCHEMA:
        raise ProvenanceInvalid(f"receipt: schema must be {RECEIPT_SCHEMA}")
    if doc.get("result") != "VERIFIED":
        raise ProvenanceInvalid("receipt: prepare result is not VERIFIED")
    expect = {
        "inputs_sha256": trusted.digest,
        "base_commit": trusted.base_commit,
        "candidate_patches_sha256": trusted.patch_digests,
        "runner_allowlist_sha256": trusted.runner_digest(),
        "exception_policy_sha256": trusted.exception_policy_digest(),
        "prepare_limits_sha256": _canon_sha(dict(PREPARE_LIMITS)),
    }
    for key, want in expect.items():
        if doc.get(key) != want:
            raise ProvenanceInvalid(f"receipt: {key} does not match the trusted inputs")
    if not trusted.patches and doc.get("no_candidate") is not True:
        raise ProvenanceInvalid("receipt: zero candidate patches requires no_candidate=true")
    if trusted.patches and doc.get("no_candidate") is not False:
        raise ProvenanceInvalid("receipt: no_candidate must be false when patches are listed")
    if not _HEX64.match(str(doc.get("manifest_sha256", ""))):
        raise ProvenanceInvalid("receipt: manifest_sha256 must be 64-hex")
    applied = _hex64_map(doc.get("applied_exceptions"), "receipt: applied_exceptions")
    approved = dict(trusted.exceptions)
    for rel, digest in applied.items():
        if approved.get(rel) != digest or forbidden_exception(rel):
            raise ProvenanceInvalid(f"receipt: exception outside the approved policy: {rel}")
    if len(applied) > trusted.max_exceptions:
        raise ProvenanceInvalid("receipt: applied exceptions exceed max_count")
    return {**doc, "applied_exceptions": applied}


def verify_receipt(doc: dict, trusted: TrustedInputs, snap: "Snapshot") -> dict:
    """Bind a parsed receipt to the verified snapshot. The snapshot's own
    manifest values are compared against the trusted inputs, never the
    other way round: self-reported fields prove nothing on their own."""
    if doc.get("manifest_sha256") != snap.manifest_sha256:
        raise ProvenanceInvalid("receipt: manifest_sha256 does not match the snapshot")
    if snap.base_commit != trusted.base_commit:
        raise ProvenanceInvalid("snapshot: base_commit differs from the trusted inputs")
    if snap.candidate_patch_sha256 != candidate_set_digest(trusted.patch_digests):
        raise ProvenanceInvalid("snapshot: candidate patch digest differs from the trusted inputs")
    files = json.loads((snap.root / MANIFEST_NAME).read_bytes())["files"]
    expected = dict(trusted.runner_allowlist)
    for patch in trusted.patches:
        expected.update(dict(patch.manifest))
    for rel, digest in expected.items():
        entry = files.get(rel)
        if not isinstance(entry, dict) or entry.get("sha256") != digest:
            raise ProvenanceInvalid(f"snapshot: {rel} does not match the trusted inputs")
    return doc


def candidate_set_digest(patch_digests: list[str]) -> str:
    """Manifest field for 0..n patches: '' (no candidate) or a digest of the list."""
    return _canon_sha(list(patch_digests)) if patch_digests else ""


def _walk(root: Path, exceptions: dict | None = None) -> dict[str, os.stat_result]:
    found: dict[str, os.stat_result] = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames + filenames:
            full = Path(dirpath) / name
            rel = full.relative_to(root).as_posix()
            st = os.lstat(full)
            if _denied(name) and not _excepted(rel, full, st, exceptions or {}):
                raise Refused(f"snapshot: denied name: {rel}")
            if stat.S_ISLNK(st.st_mode):
                raise Refused(f"snapshot: symlink not allowed: {rel}")
            if stat.S_ISDIR(st.st_mode):
                continue
            if not stat.S_ISREG(st.st_mode):
                raise Refused(f"snapshot: special file not allowed: {rel}")
            if st.st_nlink != 1:
                raise Refused(f"snapshot: hard-linked file not allowed: {rel}")
            if st.st_mode & (stat.S_ISUID | stat.S_ISGID):
                raise Refused(f"snapshot: setuid/setgid file not allowed: {rel}")
            found[rel] = st
    return found


def _excepted(rel: str, full: Path, st, exceptions: dict) -> bool:
    """A denied name passes only as an exact reviewed path with its exact hash
    (e.g. tracked `.npmrc` files). `.git` is never excepted."""
    if os.path.basename(rel) == ".git" or rel not in exceptions or not stat.S_ISREG(st.st_mode):
        return False
    return _sha256_file(full) == exceptions[rel]


def verify_snapshot(raw: str, exceptions: dict | None = None) -> Snapshot:
    """Accept a snapshot only if its tree equals its manifest byte for byte.
    Called before AND after the run with the same arguments."""
    root = real_dir(raw, "snapshot")
    doc, manifest_sha = _load_manifest(root)
    found = _walk(root, exceptions)
    found.pop(MANIFEST_NAME, None)
    listed = doc["files"]
    extra = sorted(set(found) - set(listed))
    missing = sorted(set(listed) - set(found))
    if extra or missing:
        raise Refused(f"snapshot: tree != manifest (extra={extra[:5]}, missing={missing[:5]})")
    for rel, entry in listed.items():
        if not isinstance(entry, dict) or not _HEX64.match(str(entry.get("sha256", ""))):
            raise Refused(f"snapshot: bad manifest entry: {rel}")
        if entry.get("size") != found[rel].st_size or _sha256_file(root / rel) != entry["sha256"]:
            raise Refused(f"snapshot: content mismatch: {rel}")
    if "scripts/run_tests.sh" not in listed:
        raise Refused("snapshot: scripts/run_tests.sh missing; cannot run the fixed runner")
    return Snapshot(root, manifest_sha, doc["base_commit"],
                    doc.get("candidate_patch_sha256") or "", len(listed))


def check_suite(snapshot: Snapshot, suite: str) -> tuple[str, ...]:
    if suite not in SUITES:
        raise Refused(f"suite: {suite!r} not in allowlist {sorted(SUITES)}")
    paths = SUITES[suite]
    manifest = json.loads((snapshot.root / MANIFEST_NAME).read_bytes())["files"]
    for p in paths:
        if not any(rel == p or rel.startswith(p) for rel in manifest):
            raise Refused(f"suite: path {p} absent from snapshot")
    return paths
