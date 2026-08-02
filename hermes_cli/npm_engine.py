"""Recover from npm ``EBADENGINE`` failures by upgrading a managed npm.

The repo's ``.npmrc`` sets ``engine-strict=true`` and the root ``package.json``
pins an ``engines.npm`` range, so an npm outside that range aborts every
``npm ci`` / ``npm install`` we run inside the checkout::

    npm error code EBADENGINE
    npm error notsup Required: {"node":">=20.0.0","npm":">=11.17.0"}
    npm error notsup Actual:   {"npm":"10.9.8","node":"v22.23.1"}

Rather than predicting the failure (which would mean a semver range matcher and
an ``npm --version`` probe before work that usually succeeds), we react to it:
npm states the required range in the error, so the recovery reads the
constraint straight out of the output it just produced.

Scope of the repair is deliberately narrow. Hermes only upgrades an npm that
lives inside its **own** managed Node tree (``$HERMES_HOME/node``), installing
in place with ``--prefix`` so ``bin/npm`` keeps resolving to the upgraded
``lib/node_modules/npm``. A system / nvm / brew / Nix npm belongs to the user
and their other projects; Hermes never modifies those. When the failing npm is
one of those foreign installs, Hermes instead provisions its own managed Node
tree (the same tree a fresh install creates), upgrades *that* npm into range,
and hands the caller the managed npm to retry with — leaving the user's
toolchain untouched.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from hermes_constants import (
    bootstrap_hermes_managed_node,
    get_hermes_home,
    with_hermes_node_path,
)

__all__ = [
    "is_ebadengine",
    "required_npm_range",
    "managed_npm_prefix",
    "upgrade_managed_npm",
    "maybe_repair_npm_engine",
]

# npm prints `npm error notsup Required: {...}` on npm >= 10 and
# `npm ERR! notsup Required: {...}` on older releases.
_REQUIRED_RE = re.compile(r"Required:\s*(\{.*?\})")
_ACTUAL_RE = re.compile(r"Actual:\s*(\{.*?\})")

# Wall-clock cap for the self-upgrade. The measured in-place upgrade of a
# managed tree takes ~1s; this only has to cover a slow registry.
_UPGRADE_TIMEOUT = 300


def is_ebadengine(output: str) -> bool:
    """Return True when *output* is an npm engine-compatibility failure."""
    if not output:
        return False
    return "EBADENGINE" in output or "Unsupported engine" in output


def _iter_required_blocks(output: str) -> list[dict]:
    blocks: list[dict] = []
    for match in _REQUIRED_RE.finditer(output or ""):
        try:
            parsed = json.loads(match.group(1))
        except ValueError:
            continue
        if isinstance(parsed, dict):
            blocks.append(parsed)
    return blocks


def required_npm_range(output: str) -> str | None:
    """Return the ``engines.npm`` range npm demanded in *output*.

    Returns ``None`` when the output has no engine failure, or when the
    failure is about Node rather than npm — upgrading npm cannot fix a Node
    version mismatch, so the caller must not try.

    When several packages report conflicting npm ranges the repo's own root
    constraint is preferred (it is the one we control); otherwise the first
    range wins, since any of them is a strict improvement over an npm that
    satisfies none.
    """
    if not is_ebadengine(output):
        return None
    ranges = [
        str(block["npm"]).strip()
        for block in _iter_required_blocks(output)
        if block.get("npm")
    ]
    if not ranges:
        return None
    distinct = list(dict.fromkeys(ranges))
    if len(distinct) > 1:
        repo_range = _repo_npm_range()
        if repo_range in distinct:
            return repo_range
    return distinct[0]


def actual_npm_version(output: str) -> str | None:
    """Return the npm version npm reported as ``Actual`` in *output*."""
    for match in _ACTUAL_RE.finditer(output or ""):
        try:
            parsed = json.loads(match.group(1))
        except ValueError:
            continue
        if isinstance(parsed, dict) and parsed.get("npm"):
            return str(parsed["npm"]).strip()
    return None


def _repo_npm_range() -> str | None:
    """Return ``engines.npm`` from the checkout's root ``package.json``."""
    package_json = Path(__file__).resolve().parent.parent / "package.json"
    try:
        data = json.loads(package_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    engines = data.get("engines")
    if not isinstance(engines, dict):
        return None
    value = engines.get("npm")
    return str(value).strip() if value else None


def managed_npm_prefix(npm: str | os.PathLike[str] | None) -> Path | None:
    """Return the Hermes-managed Node root *npm* lives in, else ``None``.

    Symlinks are resolved first: an install links ``~/.local/bin/npm`` at
    ``$HERMES_HOME/node/bin/npm``, which itself links into
    ``lib/node_modules/npm/bin/npm-cli.js``. Every one of those spellings is
    the managed npm and must be recognised as such, or the repair silently
    declines to fix the very install it owns.
    """
    if not npm:
        return None
    prefix = get_hermes_home() / "node"
    try:
        resolved = Path(npm).resolve()
        prefix_resolved = prefix.resolve()
    except OSError:
        return None
    if resolved == prefix_resolved or prefix_resolved in resolved.parents:
        return prefix
    return None


def _upgrade_env() -> dict[str, str]:
    env = with_hermes_node_path()
    # The checkout's .npmrc sets `min-release-age`, which would gate the npm
    # release we are trying to install. The upgrade runs from a temp cwd so
    # that file is out of scope; this neutralises a user-level ~/.npmrc too.
    env["npm_config_min_release_age"] = "0"
    # `unicode-animations`-style postinstall animations no-op under CI=1.
    env["CI"] = "1"
    return env


def upgrade_managed_npm(
    npm: str,
    npm_range: str,
    *,
    prefix: Path,
    quiet: bool = False,
) -> bool:
    """Upgrade the managed npm at *npm* in place to satisfy *npm_range*.

    ``--prefix`` targets the managed tree explicitly: a managed install writes
    ``prefix=~/.local`` into ``$HERMES_HOME/node/etc/npmrc`` so that global
    installs land on PATH, and without the override the "upgrade" would install
    a second npm somewhere else while the managed one stayed stale.
    """
    if not quiet:
        print(
            f"→ Upgrading Hermes-managed npm to satisfy {npm_range}…",
            flush=True,
        )
    try:
        # A temp cwd keeps the checkout's .npmrc (engine-strict, min-release-age)
        # from applying to the upgrade itself.
        with tempfile.TemporaryDirectory(prefix="hermes-npm-upgrade-") as tmp:
            result = subprocess.run(
                [
                    npm,
                    "install",
                    "--global",
                    "--prefix",
                    str(prefix),
                    f"npm@{npm_range}",
                    "--no-fund",
                    "--no-audit",
                    "--progress=false",
                ],
                cwd=tmp,
                env=_upgrade_env(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=_UPGRADE_TIMEOUT,
                check=False,
            )
    except (OSError, subprocess.SubprocessError):
        if not quiet:
            print("  ✗ npm upgrade could not be started", file=sys.stderr)
        return False

    if result.returncode != 0:
        if not quiet:
            detail = (result.stderr or result.stdout or "").strip().splitlines()
            print("  ✗ npm upgrade failed", file=sys.stderr)
            for line in detail[-10:]:
                print(f"    {line}", file=sys.stderr)
        return False

    if not quiet:
        print(f"  ✓ npm upgraded to {_probe_version(npm) or npm_range}", flush=True)
    return True


def _probe_version(npm: str) -> str | None:
    try:
        result = subprocess.run(
            [npm, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env=with_hermes_node_path(),
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (result.stdout or "").strip() or None


def _print_manual_fix(npm: str, npm_range: str, actual: str | None) -> None:
    have = f"npm {actual} " if actual else "This npm "
    print(
        f"\n✗ {have}does not satisfy the range this project requires: {npm_range}\n"
        f"  Resolved npm: {npm}\n"
        "  Hermes could not provision its own Node.js runtime and never\n"
        "  modifies a system/nvm/brew/Nix npm. Upgrade yours yourself with:\n"
        f'      npm install -g npm@"{npm_range}"',
        file=sys.stderr,
    )


def _provision_managed_npm(npm_range: str | None, *, quiet: bool = False) -> str | None:
    """Provision a Hermes-managed Node tree and return a satisfying npm.

    Installs the managed tree under ``$HERMES_HOME/node`` (reusing a healthy
    one when present), then upgrades its bundled npm to *npm_range* — a fresh
    Node LTS bundles an npm that may itself be outside the repo's range, so
    without the upgrade the caller's single retry would fail the same way.
    Falls back to the checkout's own ``engines.npm`` when npm did not state a
    range (a Node-only mismatch), so the managed npm ends up in range either
    way. Returns the managed npm path, or ``None`` when provisioning failed.
    """
    if not quiet:
        print(
            "→ Provisioning a Hermes-managed Node.js runtime "
            "(the resolved npm belongs to your system and is left alone)…",
            flush=True,
        )
    managed_npm = bootstrap_hermes_managed_node()
    if not managed_npm:
        if not quiet:
            print("  ✗ Managed Node.js provisioning failed", file=sys.stderr)
        return None

    prefix = managed_npm_prefix(managed_npm)
    if prefix is None:  # pragma: no cover - bootstrap returned a foreign path
        return None

    target_range = npm_range or _repo_npm_range()
    if target_range and not upgrade_managed_npm(
        managed_npm, target_range, prefix=prefix, quiet=quiet
    ):
        return None
    return managed_npm


def maybe_repair_npm_engine(
    npm: str | None,
    output: str,
    *,
    quiet: bool = False,
) -> str | None:
    """Repair an ``EBADENGINE`` failure, never touching a foreign toolchain.

    *output* is the combined stdout/stderr of the npm command that just failed.
    Returns the npm executable the caller should retry its command with —
    the same *npm* after an in-place upgrade of a Hermes-managed install, or
    a freshly provisioned managed npm when the failing npm belongs to the
    user (system / nvm / brew / Nix installs are never modified). Returns
    ``None`` when no repair happened — not an engine failure, a Node mismatch
    a managed npm upgrade cannot fix, or a failed upgrade/bootstrap — leaving
    the original failure to stand.

    The returned value is truthy exactly when the caller should retry once,
    so ``if maybe_repair_npm_engine(...)`` call sites keep working; they just
    must run the retry with the returned path.
    """
    if not npm or not is_ebadengine(output):
        return None

    npm_range = required_npm_range(output)
    prefix = managed_npm_prefix(npm)

    if prefix is not None:
        # Hermes owns this npm — upgrade it in place. Only an npm-range
        # failure is fixable this way; a Node mismatch needs a Node upgrade.
        if not npm_range:
            return None
        if upgrade_managed_npm(npm, npm_range, prefix=prefix, quiet=quiet):
            return npm
        return None

    # Foreign npm (system / nvm / brew / Nix): provision our own runtime
    # instead. This also covers Node-version mismatches — the managed tree
    # ships a Node the repo supports.
    managed = _provision_managed_npm(npm_range, quiet=quiet)
    if managed:
        return managed

    if not quiet and npm_range:
        _print_manual_fix(npm, npm_range, actual_npm_version(output))
    return None
