"""Shared service-definition normalization and safety helpers."""
from __future__ import annotations

import re
import tempfile
from pathlib import Path


def normalize_service_definition(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def temp_home_in_definition(definition: str) -> str | None:
    candidates = re.findall(r'HERMES_HOME=([^"\n]+)', definition)
    candidates += re.findall(
        r"<key>HERMES_HOME</key>\s*<string>(.*?)</string>",
        definition,
        flags=re.S,
    )
    temp_roots = {
        Path(tempfile.gettempdir()).resolve(),
        Path("/tmp"),
        Path("/var/tmp"),
        Path("/private/tmp"),
        Path("/private/var/tmp"),
    }
    for raw in candidates:
        try:
            resolved = Path(raw.strip().strip('"')).resolve()
        except (OSError, ValueError):
            continue
        if any(resolved == root or root in resolved.parents for root in temp_roots):
            return raw.strip()
    return None


def refuse_temp_home_write(definition: str, kind: str) -> bool:
    temp_home = temp_home_in_definition(definition)
    if temp_home is None:
        return False
    print(
        f"✗ Refusing to write the gateway {kind}: HERMES_HOME resolves "
        f"to a temporary directory ({temp_home})."
    )
    print(
        "  This usually means a test/E2E environment exported HERMES_HOME. "
        "Unset it (or run from a clean shell) and retry."
    )
    return True
