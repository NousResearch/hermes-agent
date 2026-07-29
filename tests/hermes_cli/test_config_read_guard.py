"""Lint guard: no new raw yaml.safe_load(config.yaml) reads outside owner modules.

The drift class this kills: scattered ``yaml.safe_load`` reads of the user's
``config.yaml`` silently miss the managed-scope overlay, ``${ENV_VAR}``
expansion, profile-aware pathing, and root-model normalization. Each new
config feature has historically required an N-site sweep (incident chain:
9cbcc0c9c8 → 732293cf87 → b0e47a98f9 → 1928aa0443).

Canonical owners:

  * ``hermes_cli/config.py`` — ``load_config()`` / ``load_config_readonly()``
    (merged + managed + env-expanded), ``read_raw_config()`` and
    ``read_user_config_raw()`` (the ONLY legal raw primitives: write-back
    round-trips + raw-file diagnostics).
  * ``gateway/config.py`` — the gateway's ``load_gateway_config`` owner.
  * ``gateway/run.py`` — ``_load_gateway_config()``'s monkeypatched-home
    fallback path (delegates to ``read_raw_config`` when paths agree).

Everything else must import one of those. If this test fails on your new
code, use ``load_config()``/``load_config_readonly()`` for behavioral reads,
or ``read_user_config_raw()`` for write-back round-trips — do not add your
file to the allowlist without a reason of the same class.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Files where a yaml.safe_load near a config.yaml reference is legal.
# Keep this list SHORT and justified:
ALLOWLIST = {
    # Canonical loader owners.
    "hermes_cli/config.py",
    "gateway/config.py",
    # _load_gateway_config()'s fallback path for tests that monkeypatch
    # gateway.run._hermes_home (delegates to read_raw_config otherwise).
    "gateway/run.py",
    # Reads the MANAGED-scope config.yaml (/etc/hermes/...), not the user's —
    # it IS the overlay source; the canonical loaders call into it.
    "hermes_cli/managed_scope.py",
    # Parse-health probe: intentionally answers "does the raw file parse?".
    "gateway/readiness.py",
}

# Directories that never count (tests may build fixture configs freely).
EXCLUDED_DIR_PARTS = {
    "tests", ".venv", ".git", ".worktrees", "node_modules", "website",
    "docs", "scripts", "examples", "apps",
}

# A safe_load within this many lines of a config.yaml reference is treated
# as a raw user-config read.
PROXIMITY = 6

SAFE_LOAD_RE = re.compile(r"\bsafe_load\s*\(")
CONFIG_YAML_RE = re.compile(r"""["']config\.yaml["']""")


def _iter_source_files():
    for path in REPO_ROOT.rglob("*.py"):
        rel = path.relative_to(REPO_ROOT)
        if any(part in EXCLUDED_DIR_PARTS for part in rel.parts):
            continue
        yield rel, path


def test_no_raw_config_yaml_reads_outside_owner_modules():
    offenders: list[str] = []
    for rel, path in _iter_source_files():
        rel_str = str(rel).replace("\\", "/")
        if rel_str in ALLOWLIST:
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        cfg_lines = [i for i, ln in enumerate(lines) if CONFIG_YAML_RE.search(ln)]
        if not cfg_lines:
            continue
        for i, ln in enumerate(lines):
            if not SAFE_LOAD_RE.search(ln):
                continue
            # Comment/docstring mentions don't count.
            stripped = ln.strip()
            if stripped.startswith("#"):
                continue
            if any(abs(i - j) <= PROXIMITY for j in cfg_lines):
                offenders.append(f"{rel_str}:{i + 1}: {stripped}")

    assert not offenders, (
        "Raw yaml.safe_load of config.yaml outside allowlisted owner modules.\n"
        "Behavioral reads must use hermes_cli.config.load_config()/"
        "load_config_readonly() (or gateway _load_gateway_config); write-back "
        "round-trips and raw-file diagnostics must use "
        "hermes_cli.config.read_user_config_raw().\nOffenders:\n  "
        + "\n  ".join(offenders)
    )


def test_read_user_config_raw_exists_and_documented():
    """The shared raw primitive must exist and carry its legality docstring."""
    from hermes_cli.config import read_user_config_raw

    doc = read_user_config_raw.__doc__ or ""
    assert "ONLY legal for write-back round-trips and raw-file diagnostics" in doc
    assert "load_config()" in doc
