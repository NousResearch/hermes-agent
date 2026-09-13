"""The wheelhouse manifest is the proof used to admit restored build output."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def build_identity(repo: Path, builder: str, platform_tag: str, python_abi: str) -> dict[str, str]:
    files = (
        "uv.lock", "pyproject.toml", "pm/lock.json",
        "scripts/termux/runtime_libs.json", "scripts/termux/build_config.sh",
        "scripts/termux/termux_build.sh", "scripts/termux/build_wheels.py",
        "scripts/termux/retag_wheel.py", "scripts/termux/python_linkage.py",
        "scripts/termux/wheelhouse_cache.py", "scripts/termux/build_environment.py",
        "pm/environment.py", "pm/build_operations.py", "pm/operations.py",
        "pm/pyproject.toml", "pm/uv.lock",
        "scripts/termux/termux-builder.Dockerfile",
    )
    return {
        "builder": builder,
        "platformTag": platform_tag,
        "pythonAbi": python_abi,
        **{name: _sha256(repo / name) for name in files},
    }


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_manifest(payload: Path, identity: dict[str, str], **metadata: str) -> None:
    wheels = sorted((payload / "wheelhouse").glob("*.whl"))
    if not wheels:
        raise ValueError("cannot cache an empty wheelhouse")
    index = {
        **metadata,
        "schemaVersion": 2,
        "inputs": identity,
        "resolvedSha256": _sha256(payload / ".work/resolved.txt"),
        "buildSetSha256": _sha256(payload / ".work/build_set.txt"),
        "wheels": [{"name": w.name, "sha256": _sha256(w)} for w in wheels],
    }
    (payload / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    (payload / "SHA256SUMS").write_text(
        "".join(f"{w['sha256']}  {w['name']}\n" for w in index["wheels"]),
        encoding="utf-8",
    )


def is_usable(payload: Path, identity: dict[str, str]) -> bool:
    try:
        index = json.loads((payload / "index.json").read_text(encoding="utf-8"))
        if index["schemaVersion"] != 2 or index["inputs"] != identity:
            return False
        if index["resolvedSha256"] != _sha256(payload / ".work/resolved.txt"):
            return False
        if index["buildSetSha256"] != _sha256(payload / ".work/build_set.txt"):
            return False
        wheels = index["wheels"]
        names = [w["name"] for w in wheels]
        actual = {w.name for w in (payload / "wheelhouse").glob("*.whl")}
        if not names or len(names) != len(set(names)) or set(names) != actual:
            return False
        for wheel in wheels:
            name = wheel["name"]
            if "/" in name or "\\" in name or name in (".", ".."):
                return False
            path = payload / "wheelhouse" / name
            if path.is_symlink() or _sha256(path) != wheel["sha256"]:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "write"))
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--builder", required=True)
    parser.add_argument("--platform-tag", required=True)
    parser.add_argument("--python-abi", required=True)
    provenance = parser.add_mutually_exclusive_group()
    provenance.add_argument("--tag", default="")
    provenance.add_argument("--commit")
    args = parser.parse_args()
    if args.commit is not None and not re.fullmatch(r"[a-f0-9]{40}", args.commit):
        parser.error("--commit requires an exact full SHA")
    identity = build_identity(args.repo, args.builder, args.platform_tag, args.python_abi)
    if args.action == "check":
        return 0 if is_usable(args.payload, identity) else 1
    write_manifest(
        args.payload, identity, **({"commit": args.commit} if args.commit else {"tag": args.tag}),
        platformTag=args.platform_tag, pythonAbi=args.python_abi,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
