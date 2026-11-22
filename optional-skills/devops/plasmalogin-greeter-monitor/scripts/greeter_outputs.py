#!/usr/bin/env python3
"""Inspect monitors and KWin output configs for the Plasma Login greeter.

Subcommands:
  scan                 Map DRM connectors to monitor make/model via EDID.
  check CONFIG         Show which connector each setup in a kwinoutputconfig.json
                       marks as primary (priority 1).
  stage CONFIG         Copy CONFIG to a private temp file and print its path
                       and sha256, ready for a root-owned install.

Stdlib only. Read-only except `stage`, which writes into the temp dir.
"""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

DRM_ROOT = Path("/sys/class/drm")


def parse_edid(raw: bytes) -> dict:
    """Return {'manufacturer', 'name'} from an EDID blob (empty if invalid)."""
    if len(raw) < 128 or raw[:8] != b"\x00\xff\xff\xff\xff\xff\xff\x00":
        return {}
    word = (raw[8] << 8) | raw[9]
    mfr = "".join(chr(((word >> s) & 0x1F) + 64) for s in (10, 5, 0))
    name = ""
    for off in (54, 72, 90, 108):
        block = raw[off : off + 18]
        if block[:3] == b"\x00\x00\x00" and block[3] == 0xFC:
            name = block[5:18].split(b"\x0a")[0].decode("ascii", "replace").strip()
    return {"manufacturer": mfr, "name": name}


def scan(drm_root: Path = DRM_ROOT) -> list[dict]:
    """List connectors with status and EDID identity."""
    rows = []
    for d in sorted(drm_root.glob("card*-*")):
        connector = d.name.split("-", 1)[1]
        status_f, edid_f = d / "status", d / "edid"
        status = status_f.read_text().strip() if status_f.exists() else "unknown"
        ident = parse_edid(edid_f.read_bytes()) if edid_f.exists() else {}
        rows.append({"connector": connector, "status": status, **ident})
    return rows


def primaries(config: list) -> list[dict]:
    """For each setup, return its connectors and the priority-1 connector."""
    sections = {e.get("name"): e.get("data", []) for e in config}
    outputs = sections.get("outputs", [])
    result = []
    for i, setup in enumerate(sections.get("setups", [])):
        names, primary = [], None
        for o in setup.get("outputs", []):
            idx = o.get("outputIndex")
            name = outputs[idx].get("connectorName") if 0 <= idx < len(outputs) else None
            if o.get("enabled", True):
                names.append(name)
            if o.get("priority") == 1:
                primary = name
        result.append({"setup": i, "connectors": names, "primary": primary})
    return result


def stage(src: Path) -> tuple[Path, str]:
    """Copy src into a fresh temp dir; return (path, sha256)."""
    dest = Path(tempfile.mkdtemp(prefix="plasmalogin-")) / "kwinoutputconfig.json"
    shutil.copyfile(src, dest)
    return dest, hashlib.sha256(dest.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Plasma Login greeter output helper")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scan")
    for name in ("check", "stage"):
        sub.add_parser(name).add_argument("config", type=Path)
    args = ap.parse_args(argv)

    if args.cmd == "scan":
        print(json.dumps(scan(), indent=2))
        return 0
    try:
        args.config.read_bytes()
    except PermissionError:
        print(f"error: cannot read {args.config} (root-owned greeter file?). "
              "Read it with: pkexec cat <file> > copy.json, then check the copy.",
              file=sys.stderr)
        return 2
    if args.cmd == "check":
        print(json.dumps(primaries(json.loads(args.config.read_text())), indent=2))
    else:
        json.loads(args.config.read_text())  # refuse to stage invalid JSON
        path, digest = stage(args.config)
        print(json.dumps({"staged": str(path), "sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
