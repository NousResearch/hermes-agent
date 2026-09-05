"""Recover from npm ``EBADENGINE`` failures by upgrading a managed npm.

We react to the failure rather than predict it: npm states the required range in the error, so the
recovery reads the constraint out of the output it just produced (no semver matcher, no probe).

Scope is deliberately narrow: Hermes only upgrades an npm inside its **own** managed Node tree
(``$HERMES_HOME/node``), installing in place with ``--prefix`` so ``bin/npm`` keeps resolving to
the upgraded ``lib/node_modules/npm``.
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
    managed_node_tree_in_use,
    with_hermes_node_path,
)

__all__ = [
    "is_ebadengine",
    "required_npm_range",
    "managed_npm_prefix",
    "upgrade_managed_npm",
    "maybe_repair_npm_engine",
]

# `npm error notsup Required: {...}` on npm >= 10, `npm ERR! notsup Required: {...}` on older.
_REQUIRED_RE = re.compile(r"Required:\s*(\{.*?\})")
_ACTUAL_RE = re.compile(r"Actual:\s*(\{.*?\})")

# Wall-clock cap for the self-upgrade (~1s measured; only has to cover a slow registry).
_UPGRADE_TIMEOUT = 300

_RUN_KW = dict(capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)


def is_ebadengine(output: str) -> bool:
    """Return True when *output* is an npm engine-compatibility failure."""
    return bool(output) and ("EBADENGINE" in output or "Unsupported engine" in output)


def _npm_fields(pattern: re.Pattern[str], output: str) -> list[str]:
    """``npm`` values of every well-formed JSON block matching *pattern*, in order."""
    values: list[str] = []
    for match in pattern.finditer(output or ""):
        try:
            parsed = json.loads(match.group(1))
        except ValueError:
            continue
        if isinstance(parsed, dict) and parsed.get("npm"):
            values.append(str(parsed["npm"]).strip())
    return values


def required_npm_range(output: str) -> str | None:
    """Return the ``engines.npm`` range npm demanded in *output*.

    ``None`` when there is no engine failure or the failure is about Node (upgrading npm cannot fix
    that, so the caller must not try). With conflicting ranges the repo's own root constraint wins
    (we control it); otherwise the first range, since any is a strict improvement.
    """
    if not is_ebadengine(output):
        return None
    distinct = list(dict.fromkeys(_npm_fields(_REQUIRED_RE, output)))
    if not distinct:
        return None
    if len(distinct) > 1:
        repo_range = _repo_npm_range()
        if repo_range in distinct:
            return repo_range
    return distinct[0]


def actual_npm_version(output: str) -> str | None:
    """Return the npm version npm reported as ``Actual`` in *output*."""
    return next(iter(_npm_fields(_ACTUAL_RE, output)), None)


def _repo_npm_range() -> str | None:
    """Return ``engines.npm`` from the checkout's root ``package.json``."""
    package_json = Path(__file__).resolve().parent.parent / "package.json"
    try:
        data = json.loads(package_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    engines = data.get("engines")
    value = engines.get("npm") if isinstance(engines, dict) else None
    return str(value).strip() if value else None


def managed_npm_prefix(npm: str | os.PathLike[str] | None) -> Path | None:
    """Return the Hermes-managed Node root *npm* lives in, else ``None``.

    Symlinks are resolved first: ``~/.local/bin/npm`` → ``$HERMES_HOME/node/bin/npm`` →
    ``lib/node_modules/npm/bin/npm-cli.js`` are all the managed npm, or the repair silently declines
    to fix the very install it owns.
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
    # The checkout's .npmrc `min-release-age` would gate the npm release we install; the upgrade
    # runs from a temp cwd so that file is out of scope, and this neutralises a user ~/.npmrc too.
    env["npm_config_min_release_age"] = "0"
    # `unicode-animations`-style postinstall animations no-op under CI=1.
    env["CI"] = "1"
    return env


def upgrade_managed_npm(npm: str, npm_range: str, *, prefix: Path, quiet: bool = False) -> bool:
    """Upgrade the managed npm at *npm* in place to satisfy *npm_range*.

    ``--prefix`` targets the managed tree explicitly: a managed install writes ``prefix=~/.local``
    into ``$HERMES_HOME/node/etc/npmrc`` (so global installs land on PATH), and without the override
    the "upgrade" would install a second npm elsewhere while the managed one stayed stale.
    """
    if not quiet:
        print(f"→ Upgrading Hermes-managed npm to satisfy {npm_range}…", flush=True)
    # The desktop app's Node processes execute from this tree; an in-place upgrade while in use
    # fails with PermissionError on npm.cmd. Defer — the upgrade re-triggers on the next resolution.
    # Defer instead of forcing the write — the upgrade re-triggers on the next resolution (e.g. the next
    # update once the app is closed). See #80926.
    if managed_node_tree_in_use():
        if not quiet:
            print(
                "  ⚠ deferred: the Hermes-managed Node.js tree is in use by a "
                "running app; the npm upgrade will apply on a later update "
                "once the app is closed.",
                file=sys.stderr,
            )
        return False
    try:
        # A temp cwd keeps the checkout's .npmrc (engine-strict, min-release-age) out of scope.
        with tempfile.TemporaryDirectory(prefix="hermes-npm-upgrade-") as tmp:
            result = subprocess.run(
                [
                    npm, "install", "--global", "--prefix", str(prefix), f"npm@{npm_range}",
                    "--no-fund", "--no-audit", "--progress=false",
                ],
                cwd=tmp,
                env=_upgrade_env(),
                timeout=_UPGRADE_TIMEOUT,
                **_RUN_KW,
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
        result = subprocess.run([npm, "--version"], timeout=30, env=with_hermes_node_path(), **_RUN_KW)
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
    """Provision (or reuse) the managed tree under ``$HERMES_HOME/node`` and return a satisfying npm.

    Its bundled npm is upgraded to *npm_range* (a fresh Node LTS may bundle an out-of-range npm, and
    the caller's single retry would fail the same way), falling back to the checkout's
    ``engines.npm`` when no range was stated. ``None`` on failure.
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
    if target_range and not upgrade_managed_npm(managed_npm, target_range, prefix=prefix, quiet=quiet):
        return None
    return managed_npm


def maybe_repair_npm_engine(npm: str | None, output: str, *, quiet: bool = False) -> str | None:
    """Repair an ``EBADENGINE`` failure, never touching a foreign toolchain.

    Truthy exactly when the caller should retry once (with the returned npm path).
    """
    if not npm or not is_ebadengine(output):
        return None

    npm_range = required_npm_range(output)
    prefix = managed_npm_prefix(npm)

    if prefix is not None:
        # Hermes owns this npm — upgrade in place. Only an npm-range failure is fixable this way.
        if npm_range and upgrade_managed_npm(npm, npm_range, prefix=prefix, quiet=quiet):
            return npm
        return None

    # Foreign npm (system / nvm / brew / Nix): provision our own runtime instead. This also covers
    # Node-version mismatches — the managed tree ships a Node the repo supports.
    managed = _provision_managed_npm(npm_range, quiet=quiet)
    if managed:
        return managed

    if not quiet and npm_range:
        _print_manual_fix(npm, npm_range, actual_npm_version(output))
    return None
