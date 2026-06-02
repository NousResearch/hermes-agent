#!/usr/bin/env python3
"""Canonicalize Chaquopy generated assets for reproducible Android APKs."""

from __future__ import annotations

import json
import marshal
import sys
import zipfile
from pathlib import Path


ZIP_TIMESTAMP = (1980, 2, 1, 0, 0, 0)
PYC_UNCHECKED_HASH_HEADER = (1).to_bytes(4, "little") + (b"\0" * 8)


def normalize_build_json(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    payload = json.dumps(data, indent=4, sort_keys=True) + "\n"
    path.write_bytes(payload.encode("utf-8"))


def normalize_pyc(payload: bytes) -> bytes:
    if len(payload) <= 16:
        return payload
    try:
        code = marshal.loads(payload[16:])
    except Exception:
        return payload
    return payload[:4] + PYC_UNCHECKED_HASH_HEADER + marshal.dumps(code, 2)


def normalize_requirements_imy(path: Path) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(path, "r") as source, zipfile.ZipFile(temp_path, "w") as target:
        infos = {info.filename: info for info in source.infolist()}
        for name in sorted(infos):
            if name.endswith(".dist-info/direct_url.json"):
                continue
            info = infos[name]
            payload = source.read(name)
            if name.endswith(".pyc"):
                payload = normalize_pyc(payload)
            elif name.endswith(".dist-info/METADATA"):
                payload = payload.replace(b"\r\n", b"\n")
            target_info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
            target_info.create_system = 3
            target_info.compress_type = zipfile.ZIP_STORED
            target_info.external_attr = info.external_attr
            target.writestr(target_info, payload)
    temp_path.replace(path)


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in {"build-json", "requirements-imy"}:
        print(
            "usage: normalize_chaquopy_assets.py {build-json|requirements-imy} PATH",
            file=sys.stderr,
        )
        return 2

    mode = sys.argv[1]
    path = Path(sys.argv[2])
    if mode == "build-json":
        normalize_build_json(path)
    else:
        normalize_requirements_imy(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
