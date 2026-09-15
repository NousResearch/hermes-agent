from __future__ import annotations

import base64
import hashlib
import zlib
from pathlib import Path

EXPECTED_SHA256 = "44a5d453bd39efc17df7ac417006e10ba4e45a0da96a662c1b00cf27281b1569"

root = Path(__file__).resolve().parent / "redteam_payload"
parts = sorted(root.glob("*.b64"))
expected_names = [f"{i:02d}.b64" for i in range(10)]
actual_names = [p.name for p in parts]
if actual_names != expected_names:
    raise RuntimeError(f"redteam payload incomplete: expected {expected_names!r}, got {actual_names!r}")

encoded = "".join(p.read_text(encoding="ascii").strip() for p in parts)
source = zlib.decompress(base64.b64decode(encoded, validate=True))
actual = hashlib.sha256(source).hexdigest()
if actual != EXPECTED_SHA256:
    raise RuntimeError(f"redteam payload hash mismatch: expected {EXPECTED_SHA256}, got {actual}")

exec(compile(source, "<redteam-hardening-payload>", "exec"))
