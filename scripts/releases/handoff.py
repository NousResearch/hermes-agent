"""Transfer immutable release files through R2 without publishing a feed."""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath

from scripts.releases import r2
from scripts.releases.semver import is_valid_version


class MissingReceipt(ValueError):
    """A missing receipt is incomplete work, not a corrupt receipt."""


def validate_identity(tag: str, commit: str, name: str) -> None:
    if (not isinstance(tag, str) or not tag.startswith("v") or not is_valid_version(tag[1:])
            or not re.fullmatch(r"[a-f0-9]{40}", commit or "")
            or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name or "")):
        raise ValueError("Invalid release handoff identity")


def validate_commit_identity(commit: str, name: str) -> None:
    """Identity for commit-only handoffs: no tag, an exact full SHA."""
    if (not r2.is_full_sha(commit)
            or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name or "")):
        raise ValueError("Invalid commit handoff identity")


def validate_receipt_files(receipt: dict) -> list[dict]:
    """Shared receipt-body rules: non-empty unique paths, int sizes, digests."""
    files = receipt.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("No files in release handoff")
    seen = set()
    for row in files:
        if not isinstance(row, dict):
            raise ValueError("Invalid release file receipt")
        path = r2.relative_artifact_path(row.get("path"))
        if path.casefold() in seen:
            raise ValueError("Duplicate release artifact path")
        seen.add(path.casefold())
        if (type(row.get("size")) is not int or row["size"] < 0
                or not re.fullmatch(r"[a-f0-9]{64}", row.get("sha256", ""))):
            raise ValueError("Invalid release file size or digest")
    return files


def receipt_name(name: str) -> str:
    return f"handoff-{name}.json"


def validate_receipt(receipt: dict, tag: str, commit: str, name: str) -> list[dict]:
    validate_identity(tag, commit, name)
    if (not isinstance(receipt, dict) or receipt.get("schema") != 1
            or receipt.get("tag") != tag or receipt.get("commit") != commit or receipt.get("name") != name):
        raise ValueError("Release handoff identity mismatch")
    return validate_receipt_files(receipt)


def validate_commit_receipt(receipt: dict, commit: str, name: str) -> list[dict]:
    """Schema-2 receipts bind the files to a commit with no tag anywhere."""
    validate_commit_identity(commit, name)
    if (not isinstance(receipt, dict) or receipt.get("schema") != 2
            or receipt.get("commit") != commit or receipt.get("name") != name
            or "tag" in receipt):
        raise ValueError("Commit handoff identity mismatch")
    return validate_receipt_files(receipt)


def _select_files(root: Path, includes: list[str]) -> dict[str, Path]:
    """Shared stage selector: pattern matching, symlink/escape rejection."""
    root = root.resolve()
    selected = {}
    for pattern in includes:
        matched = [p for p in root.glob(pattern) if p.is_file()]
        if not matched:
            raise ValueError(f"No files match release handoff pattern: {pattern}")
        for file in matched:
            if file.is_symlink() or not file.resolve().is_relative_to(root):
                raise ValueError("Release artifact path escapes its build root")
            selected[file.relative_to(root).as_posix()] = file
    return selected


def _receipt_file_rows(selected: dict[str, Path]) -> list[dict]:
    return [{"path": r2.relative_artifact_path(path), "size": file.stat().st_size, "sha256": r2.file_sha256(file)}
            for path, file in sorted(selected.items())]


def _stage(receipt: dict, selected: dict[str, Path]) -> dict:
    """Publish completion only after every immutable artifact upload succeeds."""
    prefix = (r2.staging_key_for(receipt["tag"], "") if "tag" in receipt
              else r2.commit_prefix_for(receipt["commit"]))
    for row in receipt["files"]:
        r2.put(tag=receipt.get("tag", ""), key=prefix + row["path"],
               file=str(selected[row["path"]]), key_is_full=True, immutable=True)
    with tempfile.TemporaryDirectory() as directory:
        file = Path(directory) / receipt_name(receipt["name"])
        file.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        r2.put(tag=receipt.get("tag", ""), key=prefix + file.name,
               file=str(file), key_is_full=True, immutable=True)
    return receipt


def stage(tag: str, commit: str, name: str, root: Path, includes: list[str]) -> dict:
    validate_identity(tag, commit, name)
    selected = _select_files(root, includes)
    receipt = {"schema": 1, "tag": tag, "commit": commit, "name": name,
               "files": _receipt_file_rows(selected)}
    validate_receipt(receipt, tag, commit, name)
    return _stage(receipt, selected)


def stage_commit_build(commit: str, name: str, root: Path, includes: list[str]) -> dict:
    """Stage tagless artifacts. Identical retries pass; different bytes fail."""
    validate_commit_identity(commit, name)
    selected = _select_files(root, includes)
    receipt = {"schema": 2, "commit": commit, "name": name,
               "files": _receipt_file_rows(selected)}
    validate_commit_receipt(receipt, commit, name)
    return _stage(receipt, selected)


def read_receipt(tag: str, commit: str, name: str) -> dict:
    validate_identity(tag, commit, name)
    creds, base, bucket = r2.credentials()
    key = r2.staging_key_for(tag, receipt_name(name))
    text = r2.get_object(creds, base, bucket, key, r2.amz_timestamp())
    if text is None:
        raise ValueError(f"Missing release handoff: {name}")
    receipt = json.loads(text)
    validate_receipt(receipt, tag, commit, name)
    return receipt


def read_commit_receipt(commit: str, name: str) -> dict:
    validate_commit_identity(commit, name)
    creds, base, bucket = r2.credentials()
    key = r2.commit_key_for(commit, receipt_name(name))
    try:
        text = r2.get_object(creds, base, bucket, key, r2.amz_timestamp())
    except r2.R2RequestError as err:
        if err.status == 404:
            raise MissingReceipt(f"Missing commit handoff: {name}") from err
        raise
    if text is None:
        raise MissingReceipt(f"Missing commit handoff: {name}")
    receipt = json.loads(text)
    validate_commit_receipt(receipt, commit, name)
    return receipt


def fetch(tag: str, commit: str, names: list[str], root: Path,
          includes: list[str] | None = None) -> list[dict]:
    receipts = [read_receipt(tag, commit, name) for name in names]
    return _fetch_receipts(receipts, root, includes)


def fetch_commit_build(commit: str, names: list[str], root: Path,
                       includes: list[str] | None = None) -> list[dict]:
    """Fetch commit artifacts through the shared receipt-bound transport."""
    receipts = [read_commit_receipt(commit, name) for name in names]
    return _fetch_receipts(receipts, root, includes)


def _fetch_receipts(receipts: list[dict], root: Path,
                    includes: list[str] | None = None) -> list[dict]:
    """Download exact receipt-bound bytes before writing local receipt copies."""
    root = root.resolve()
    selected = {}
    for receipt in receipts:
        prefix = (r2.staging_key_for(receipt["tag"], "") if "tag" in receipt
                  else r2.commit_prefix_for(receipt["commit"]))
        for row in receipt["files"]:
            if includes and not any(fnmatch.fnmatchcase(row["path"], pattern) for pattern in includes):
                continue
            item = row, prefix + row["path"]
            prior = selected.get(row["path"].casefold())
            if prior is not None and prior != item:
                raise ValueError("Conflicting release file receipts")
            target = root / PurePosixPath(row["path"])
            if not target.resolve().is_relative_to(root):
                raise ValueError("Release artifact path escapes its download root")
            selected[row["path"].casefold()] = item
    if not selected:
        raise ValueError("No files selected from release handoff")
    root.mkdir(parents=True, exist_ok=True)
    creds, base, bucket = r2.credentials()
    for row, key in selected.values():
        r2.download_object(creds, base, bucket, key,
                           root / row["path"], r2.amz_timestamp(),
                           expected_size=row["size"], expected_sha256=row["sha256"])
    for receipt in receipts:
        (root / receipt_name(receipt["name"])).write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return receipts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["stage", "fetch"])
    parser.add_argument("--tag", default=os.environ.get("HERMES_PAYLOAD_TAG") or os.environ.get("RELEASE_TAG"), required=False)
    parser.add_argument("--commit-build", default=os.environ.get("HERMES_BUILD_COMMIT"),
                        help="Commit-only mode: stage/fetch under releases/commit/<sha>/ "
                             "with schema-2 receipts (no tag; exact full SHA required)")
    parser.add_argument("--commit", default=None,
                        help="Artifact commit. Must equal --commit-build when both are supplied.")
    parser.add_argument("--name", action="append", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--include", action="append")
    args = parser.parse_args(argv)
    if args.commit_build:
        if args.tag:
            parser.error("--commit-build and --tag are mutually exclusive")
        if args.commit and args.commit != args.commit_build:
            parser.error("--commit must equal --commit-build (they name the same commit)")
        commit = args.commit_build
        if args.command == "stage":
            if len(args.name) != 1 or not args.include:
                parser.error("stage needs one name and at least one include pattern")
            stage_commit_build(commit, args.name[0], args.root, args.include)
        else:
            fetch_commit_build(commit, args.name, args.root, args.include)
        return
    if args.command == "stage":
        if len(args.name) != 1 or not args.include:
            parser.error("stage needs one name and at least one include pattern")
        stage(args.tag, args.commit, args.name[0], args.root, args.include)
    else:
        fetch(args.tag, args.commit, args.name, args.root, args.include)


if __name__ == "__main__":
    main()
