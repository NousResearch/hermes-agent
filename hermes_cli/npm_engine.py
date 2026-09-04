"""Recover from npm ``EBADENGINE`` failures with the pm-pinned npm.

We react to the failure rather than predict it: npm states the required range in the error, so the
recovery reads the constraint out of the output it just produced (no semver matcher, no probe).

Rather than predicting the failure (which would mean a semver range matcher and
an ``npm --version`` probe before work that usually succeeds), we react to it:
npm states the required range in the error, so the recovery reads the
constraint straight out of the output it just produced.

Scope of the repair is deliberately narrow. A system / nvm / brew / Nix npm
belongs to the user and their other projects; Hermes never modifies those.
When the failing npm is a foreign install, Hermes ensures its own pm-pinned
node/npm packages are installed and hands the caller pm's npm to retry with —
leaving the user's toolchain untouched. When the failing npm already *is*
pm's npm, the lockfile pin itself is out of range and no runtime action can
fix that; the caller gets the manual guidance instead.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

__all__ = [
    "is_ebadengine",
    "required_npm_range",
    "maybe_repair_npm_engine",
]

# `npm error notsup Required: {...}` on npm >= 10, `npm ERR! notsup Required: {...}` on older.
_REQUIRED_RE = re.compile(r"Required:\s*(\{.*?\})")
_ACTUAL_RE = re.compile(r"Actual:\s*(\{.*?\})")

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
        data = json.loads(package_json.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    engines = data.get("engines")
    value = engines.get("npm") if isinstance(engines, dict) else None
    return str(value).strip() if value else None


def _pm_npm(*, quiet: bool = False) -> str | None:
    """Install the pm-pinned node/npm packages and return pm's npm path."""
    if not quiet:
        print(
            "→ Provisioning the Hermes-pinned Node.js runtime "
            "(the resolved npm belongs to your system and is left alone)…",
            flush=True,
        )
    try:
        import pm

        pm.ensure("npm")
        from hermes_constants import _pm_node_executable

        managed = _pm_node_executable("npm")
    except Exception:
        managed = None
    if not managed and not quiet:
        print("  ✗ Managed Node.js provisioning failed", file=sys.stderr)
    return managed


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


def maybe_repair_npm_engine(
    npm: str | None,
    output: str,
    *,
    quiet: bool = False,
) -> str | None:
    """Repair an ``EBADENGINE`` failure, never touching a foreign toolchain.

    *output* is the combined stdout/stderr of the npm command that just failed.
    Returns the npm executable the caller should retry its command with — the
    pm-pinned npm, freshly ensured, when the failing npm was a foreign install
    (system / nvm / brew / Nix installs are never modified). Returns ``None``
    when no repair happened — not an engine failure, the failing npm already
    was pm's own (the lockfile pin is out of range; a runtime install cannot
    fix that), or the pm install failed — leaving the original failure to
    stand.

    The returned value is truthy exactly when the caller should retry once,
    so ``if maybe_repair_npm_engine(...)`` call sites keep working; they just
    must run the retry with the returned path.
    """
    if not npm or not is_ebadengine(output):
        return None

    managed = _pm_npm(quiet=quiet)
    if managed:
        try:
            already_managed = Path(managed).resolve() == Path(npm).resolve()
        except OSError:
            already_managed = False
        if not already_managed:
            return managed

    npm_range = required_npm_range(output)
    if not quiet and npm_range:
        _print_manual_fix(npm, npm_range, actual_npm_version(output))
    return None
