#!/usr/bin/env python3
"""Comprehensive, non-disclosing scanner for a sealed candidate surface."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import tarfile
from pathlib import Path
from typing import Iterable


class SecretMaterialDetected(RuntimeError):
    pass


class UnscannedSurface(RuntimeError):
    pass


class ArchiveScanMismatch(RuntimeError):
    pass


SCHEMA = "AresContextGovernorSecretScanV2"
# Split the prohibited path so scanning this scanner's own source does not
# manufacture a false positive.  Candidate content is searched for the joined
# byte sequence.
LEGACY_PATH = b".hermes/context-governor/" + b"hmac.key"
MARKERS = (
    b"-----BEGIN " + b"PRIVATE KEY-----",
    b"-----BEGIN " + b"RSA PRIVATE KEY-----",
)
FIELD_RE = re.compile(
    rb'(?i)(api[_-]?key|secret|password|token|hmac)["\']?\s*[:=]\s*["\']?([^\s,"\'}]{12,})'
)


def fingerprint(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()[:16]


def entropy(value: bytes) -> float:
    if not value:
        return 0.0
    counts = {byte: value.count(byte) for byte in set(value)}
    return -sum(
        (count / len(value)) * math.log2(count / len(value))
        for count in counts.values()
    )


def surfaces(roots: Iterable[Path]) -> Iterable[tuple[str, bytes]]:
    for root in roots:
        if root.is_dir():
            for path in sorted(
                item
                for item in root.rglob("*")
                if item.is_file() and not item.is_symlink()
            ):
                yield str(path), path.read_bytes()
        elif root.is_file() and tarfile.is_tarfile(root):
            yield str(root), root.read_bytes()
            with tarfile.open(root, "r:") as archive:
                for member in archive.getmembers():
                    if member.isfile():
                        source = archive.extractfile(member)
                        assert source is not None
                        yield f"{root}!{member.name}", source.read()
                    else:
                        yield f"{root}!{member.name}", member.name.encode()
        elif root.is_file():
            yield str(root), root.read_bytes()
        else:
            raise UnscannedSurface(str(root))


def scan(
    roots: Iterable[Path], *, fixture_secrets: Iterable[bytes] = ()
) -> dict[str, object]:
    findings: list[dict[str, object]] = []
    scanned: list[str] = []
    probes = tuple(fixture_secrets)
    encoded = tuple(
        (secret, secret.hex().encode(), base64.b64encode(secret)) for secret in probes
    )
    for path, content in surfaces(roots):
        scanned.append(path)
        path_bytes = path.encode(errors="replace")

        def found(rule: str, value: bytes, offset: int) -> None:
            findings.append({
                "rule": rule,
                "surface": path,
                "offset": offset,
                "fingerprint": fingerprint(value),
            })

        if LEGACY_PATH in path_bytes or LEGACY_PATH in content:
            found(
                "prohibited_legacy_key_path",
                LEGACY_PATH,
                max(path_bytes.find(LEGACY_PATH), content.find(LEGACY_PATH)),
            )
        for marker in MARKERS:
            index = content.find(marker)
            if index >= 0:
                found("private_key_or_legacy_field", marker, index)
        # Python/Rust/TypeScript source necessarily contains secret-field
        # *names* and redaction fixtures.  Treat high-entropy values as secret
        # material only on structured configuration/evidence/log surfaces;
        # all source surfaces remain covered by the raw, marker, legacy-path,
        # and controlled-fixture rules below.
        surface_name = path.rsplit("!", 1)[-1].lower()
        structured = (
            surface_name.endswith((
                ".json",
                ".yaml",
                ".yml",
                ".toml",
                ".ini",
                ".env",
                ".log",
            ))
            or "config" in surface_name
            or "evidence" in surface_name
        )
        if structured:
            for match in FIELD_RE.finditer(content):
                value = match.group(2)
                if entropy(value) >= 3.5:
                    found("structured_high_entropy_secret_field", value, match.start(2))
        for secret, hex_value, b64_value in encoded:
            for rule, probe in (
                ("controlled_fixture_secret_raw", secret),
                ("controlled_fixture_secret_hex", hex_value),
                ("controlled_fixture_secret_base64", b64_value),
            ):
                index = content.find(probe)
                if index >= 0:
                    found(rule, probe, index)
    report = {
        "schema": SCHEMA,
        "surface_count": len(scanned),
        "surfaces": scanned,
        "findings": findings,
        "pass": not findings,
    }
    if findings:
        raise SecretMaterialDetected(json.dumps(report, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("surface", nargs="+", type=Path)
    parser.add_argument("--fixture-secret-hex", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = scan(
        args.surface,
        fixture_secrets=(bytes.fromhex(value) for value in args.fixture_secret_hex),
    )
    rendered = json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
