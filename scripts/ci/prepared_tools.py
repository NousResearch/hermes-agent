"""Unprivileged PM tree producer and default-branch-only artifact publisher.

The receipt is untrusted: the publisher derives the required package set and
source closure from a PR-head lockfile fetched as git DATA, never importing PR
code or executing archive members. It edits only that lockfile via git plumbing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import tempfile

from pm.artifact_mirror import github_repository, object_key
from pm.lock import Lockfile, SCHEMA
from pm.prepare import prepare, preparation_requirements, source_identity
from pm.registry import all_packages, get_package
from pm.store import ALL_TARGETS, extract, tree_digest
from scripts.ci.archive_inputs import GhCli, GitHubMirror, Mirror

_SHA = re.compile(r"[a-f0-9]{40}")
_NAME = re.compile(r"[a-z0-9][a-z0-9-]*")


def selected(lock: Lockfile, target: str) -> list[str]:
    if target not in ALL_TARGETS:
        raise ValueError(f"unknown target: {target}")
    return [name for name in all_packages()
            if not getattr(get_package(name), "pin_only", False)
            and get_package(name).missing_reason(target) is None
            and bool(lock.artifacts(name, target))]


def _filename(name: str, target: str) -> str:
    if not _NAME.fullmatch(name) or target not in ALL_TARGETS:
        raise ValueError("invalid prepared package or target")
    return f"{name}--{target}" + (".zip" if target.startswith("win32-") else ".tar.gz")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def produce(target: str, output: Path, *, lock: Lockfile | None = None) -> dict:
    """Produce ALL stageable tools, failing the entire target on any omission."""
    from pm import paths
    from pm.install import ensure

    lock = lock or Lockfile(paths.lockfile_path())
    names = selected(lock, target)
    if not names:
        raise ValueError(f"no stageable tools for {target}")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"output directory must be empty: {output}")
    rows = []
    for name in names:
        requirements = preparation_requirements(name, target)
        # Native npm uses PM's node; ARM Windows ripgrep needs PM's Python DLL.
        for requirement in requirements:
            if requirement.startswith("installed "):
                ensure(requirement.split()[1], explicit=True)
        filename = _filename(name, target)
        pin = prepare(name, target, output / filename)
        if pin["source"] != source_identity(lock, name, target):
            raise ValueError(f"source closure moved during preparation: {name}@{target}")
        if _sha256(output / filename) != pin["sha256"]:
            raise ValueError(f"archive changed during preparation: {name}@{target}")
        rows.append({"name": name, "file": filename, "pin": pin})
    receipt = {"schema": 1, "target": target, "packages": rows}
    (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return receipt


def _read_lock(path: Path) -> Lockfile:
    # Lockfile's ordinary reader intentionally tolerates invalid local state.
    # Publication must instead refuse corrupt/empty PR-supplied lock data.
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or data.get("schema") != SCHEMA or not isinstance(data.get("packages"), dict):
        raise ValueError("invalid PR-head PM lockfile")
    return Lockfile(path)


def validate(receipts: Path, lock: Lockfile, *, targets: tuple[str, ...] | None = None) -> list[tuple[str, str, Path, dict]]:
    """Require exact coverage, hashes, and extracted tree digests; never exec members."""
    targets = ALL_TARGETS if targets is None else tuple(targets)
    if not targets or len(set(targets)) != len(targets) or any(t not in ALL_TARGETS for t in targets):
        raise ValueError("invalid target set")
    expected_dirs = {f"prepared-{target}" for target in targets}
    if {p.name for p in receipts.iterdir()} != expected_dirs:
        raise ValueError("missing or unexpected target receipt directory")
    verified = []
    for target in targets:
        directory = receipts / f"prepared-{target}"
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f"missing receipt: {target}")
        receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8-sig"))
        if (not isinstance(receipt, dict) or set(receipt) != {"schema", "target", "packages"}
                or receipt["schema"] != 1 or receipt["target"] != target
                or not isinstance(receipt["packages"], list)):
            raise ValueError(f"invalid receipt: {target}")
        names = selected(lock, target)
        rows = receipt["packages"]
        if len(rows) != len(names) or not names:
            raise ValueError(f"incomplete prepared receipt: {target}")
        expected_files = {"receipt.json"}
        for name, row in zip(names, rows):
            filename = _filename(name, target)
            if (not isinstance(row, dict) or set(row) != {"name", "file", "pin"}
                    or row["name"] != name or row["file"] != filename):
                raise ValueError(f"wrong prepared entry: {name}@{target}")
            pin = row["pin"]
            if (not isinstance(pin, dict) or set(pin) != {"sha256", "digest", "source"}
                    or any(not isinstance(v, str) or not re.fullmatch(r"[a-f0-9]{64}", v)
                           for v in pin.values())
                    or pin["source"] != source_identity(lock, name, target)):
                raise ValueError(f"invalid source identity or pin: {name}@{target}")
            archive = directory / filename
            if not archive.is_file() or archive.is_symlink() or _sha256(archive) != pin["sha256"]:
                raise ValueError(f"missing or tampered archive: {name}@{target}")
            with tempfile.TemporaryDirectory(prefix="pm-prepared-verify-") as temporary:
                tree = Path(temporary) / "tree"
                extract(archive, tree)
                if (tree_digest(tree) != pin["digest"] or (tree / ".pm-stage-pin.json").exists()):
                    raise ValueError(f"wrong prepared tree digest: {name}@{target}")
            expected_files.add(filename)
            verified.append((name, target, archive, pin))
        if {p.name for p in directory.iterdir()} != expected_files:
            raise ValueError(f"extra or missing receipt files: {target}")
    return verified


def _run(*args: str, input: bytes | None = None, strip: bool = True) -> bytes:
    output = subprocess.run(args, input=input, stdout=subprocess.PIPE, check=True).stdout
    return output.strip() if strip else output


def _pr(repository: str, number: int, expected_sha: str) -> dict:
    data = json.loads(_run("gh", "api", f"repos/{repository}/pulls/{number}"))
    head = data["head"]
    if (data["state"] != "open" or head["repo"]["full_name"].lower() != repository.lower()
            or head["sha"] != expected_sha):
        raise ValueError("PR is not an open same-repo branch at the original head SHA")
    _run("git", "check-ref-format", f"refs/heads/{head['ref']}")
    return data


def _fork_owner_authorized(repository: str, author: str) -> bool:
    """The personal staging fork's owner explicitly admits her own PR heads."""
    return repository.lower() == "ethernet8023/hermes-agent" and author.lower() == "ethernet8023"


def _approved_head(repository: str, number: int, head_sha: str, author: str) -> bool:
    """A different repository writer must have approved these exact PR bytes."""
    pages = json.loads(_run("gh", "api", "--paginate", "--slurp",
                            f"repos/{repository}/pulls/{number}/reviews?per_page=100"))
    latest: dict[str, tuple[str, str]] = {}
    for page in pages:
        for review in page:
            user = review["user"]["login"]
            if review["state"] in ("APPROVED", "CHANGES_REQUESTED", "DISMISSED"):
                latest[user] = (review["state"], review.get("commit_id", ""))
    for user, (state, commit) in latest.items():
        if state != "APPROVED" or commit != head_sha or user == author:
            continue
        permission = json.loads(_run("gh", "api",
                                     f"repos/{repository}/collaborators/{user}/permission"))
        if permission["permission"] in ("admin", "maintain", "write"):
            return True
    return False


def _bootstrap_updates(head_sha: str, lock_path: Path, scratch: Path) -> dict[str, bytes]:
    """Generate standalone fragments with trusted code, preserving PR script bodies."""
    generator = Path(__file__).resolve().parents[1] / "gen-bootstrap-pins.py"
    if _run("git", "show", f"{head_sha}:scripts/gen-bootstrap-pins.py", strip=False) != generator.read_bytes():
        raise ValueError("bootstrap generator changed; land code before pinning")
    methods = runpy.run_path(str(generator))
    packages = json.loads(lock_path.read_text(encoding="utf-8-sig"))["packages"]
    fragments = {
        "scripts/install.sh": methods["_sh_fragment"](packages["uv"]),
        "scripts/install.ps1": methods["_ps1_fragment"](packages["uv"], packages["git"]),
    }
    result = {}
    for name, fragment in fragments.items():
        path = scratch / Path(name).name
        original = _run("git", "show", f"{head_sha}:{name}", strip=False)
        path.write_bytes(original)
        methods["_splice"](path, fragment, False)
        updated = path.read_bytes()
        if updated != original:
            result[name] = updated
    return result


def publish(receipts: Path, repository: str, number: int, head_sha: str,
            *, mirror: Mirror | None = None) -> str:
    """Use only default-branch code; build a PR-head child commit via git objects."""
    if repository.lower() != github_repository().lower() or not _SHA.fullmatch(head_sha) or number <= 0:
        raise ValueError("untrusted repository, PR number or commit")
    pr = _pr(repository, number, head_sha)
    branch = pr["head"]["ref"]
    remote_ref = f"refs/heads/{branch}"
    # Fetch a DATA object, never check out the PR tree or import its Python.
    _run("git", "fetch", "--no-tags", "origin", remote_ref)
    if _run("git", "rev-parse", "FETCH_HEAD").decode() != head_sha:
        raise ValueError("PR branch moved before publication")
    raw = _run("git", "show", f"{head_sha}:pm/lock.json", strip=False)
    with tempfile.TemporaryDirectory(prefix="pm-prepared-publish-") as temporary:
        scratch = Path(temporary)
        lock_path = scratch / "lock.json"
        lock_path.write_bytes(raw)
        lock = _read_lock(lock_path)
        verified = validate(receipts, lock)
        mirror = mirror or GitHubMirror(GhCli(repository))
        changed = any(lock.prepared(name, target) != pin for name, target, _, pin in verified)
        sizes = {}
        for _, _, archive, pin in verified:
            sha = pin["sha256"]
            object_key(sha)
            size = mirror.size(sha)
            if size is not None and size != archive.stat().st_size:
                raise ValueError(f"existing asset size differs: {sha}")
            sizes[sha] = size
        # The fork owner admitted her own PR head for this staging repo; all
        # other authors still need a writer's exact-head review. A bot-commit
        # rerun with unchanged pins and assets is read-only.
        author = pr["user"]["login"]
        admitted = _fork_owner_authorized(repository, author) or _approved_head(
            repository, number, head_sha, author)
        if (changed or any(size is None for size in sizes.values())) and not admitted:
            raise ValueError("a repository writer must approve the exact PR head SHA before publication")
        for name, target, _, pin in verified:
            lock.set_prepared(name, target, pin)
        lock.save()
        updates = _bootstrap_updates(head_sha, lock_path, scratch) if changed else {}
        # Asset transport is last: never write a release before every receipt
        # and generated bootstrap fragment is validated.
        for _, _, archive, pin in verified:
            sha = pin["sha256"]
            if sizes[sha] is None:
                mirror.put(sha, archive)
            mirror.read_back(sha, scratch / f"read-back-{sha}", archive.stat().st_size)
        # Compare the *exact* fetched object immediately before push, not only
        # the event payload; lease below closes the remaining branch-move race.
        _pr(repository, number, head_sha)
        if _run("git", "ls-remote", "origin", remote_ref).split()[:1] != [head_sha.encode()]:
            raise ValueError("PR branch moved before bot commit")
        if not changed:
            return head_sha  # App pushes trigger another run; no empty bot commit loop.
        blob = _run("git", "hash-object", "-w", "--stdin", input=lock_path.read_bytes()).decode()
        index = scratch / "index"
        env = {**os.environ, "GIT_INDEX_FILE": str(index)}
        subprocess.run(["git", "read-tree", head_sha], env=env, check=True)
        subprocess.run(["git", "update-index", "--add", "--cacheinfo", f"100644,{blob},pm/lock.json"],
                       env=env, check=True)
        for name, content in updates.items():
            fragment_blob = _run("git", "hash-object", "-w", "--stdin", input=content).decode()
            mode = _run("git", "ls-tree", head_sha, "--", name).decode().partition(" ")[0]
            if mode not in ("100644", "100755"):
                raise ValueError(f"invalid installer mode at PR head: {name}")
            subprocess.run(["git", "update-index", "--add", "--cacheinfo", f"{mode},{fragment_blob},{name}"],
                           env=env, check=True)
        tree = subprocess.run(["git", "write-tree"], env=env, check=True, capture_output=True).stdout.strip().decode()
        commit = _run("git", "-c", "user.name=github-actions[bot]",
                      "-c", "user.email=github-actions[bot]@users.noreply.github.com",
                      "commit-tree", tree, "-p", head_sha,
                      input=b"chore(pm): pin prepared tool trees\n").decode()
        _run("git", "push", f"--force-with-lease={remote_ref}:{head_sha}", "origin", f"{commit}:{remote_ref}")
        if _run("git", "ls-remote", "origin", remote_ref).split()[:1] != [commit.encode()]:
            raise ValueError("bot commit did not reach the PR branch")
        return commit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    producer = commands.add_parser("produce")
    producer.add_argument("--target", required=True, choices=ALL_TARGETS)
    producer.add_argument("--out", required=True, type=Path)
    publisher = commands.add_parser("publish")
    publisher.add_argument("--receipts", required=True, type=Path)
    publisher.add_argument("--repository", required=True)
    publisher.add_argument("--pr", required=True, type=int)
    publisher.add_argument("--head-sha", required=True)
    args = parser.parse_args(argv)
    if args.command == "produce":
        receipt = produce(args.target, args.out)
        print(f"Prepared {len(receipt['packages'])} tools for {args.target}")
    else:
        print(publish(args.receipts, args.repository, args.pr, args.head_sha))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
