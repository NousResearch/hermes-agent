"""Optional packages the default install carries, and the user's opt-outs.

A ``default`` package (see ``Package.default``) joins the installers' PM stage,
a bare ``hermes pm install`` and ``hermes update`` on every target it builds
for. The user can decline one (``install.sh --skip-browser``,
``install.ps1 -SkipBrowser``, ``hermes pm install --without NAME``). The choice
is recorded per installation beside PM's other install state, so a later
update or bare install never re-adds it. An explicit
``hermes pm install NAME`` clears it.

Stdlib-only apart from PM's own modules: the installers run this before any
application dependency exists.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

DECLINED_FILENAME = "declined-packages.json"


def declined_path(project_root: Path | None = None) -> Path:
    from pm.environments import install_state_dir
    from pm.paths import repo_root

    return install_state_dir(repo_root() if project_root is None else Path(project_root)) / DECLINED_FILENAME


def declined(project_root: Path | None = None) -> frozenset[str]:
    try:
        data = json.loads(declined_path(project_root).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return frozenset()
    except (OSError, ValueError) as exc:
        # Silently treating a damaged record as "nothing declined" would
        # re-add a package the user explicitly refused.
        from pm.package import InstallError

        raise InstallError("defaults", f"cannot read {declined_path(project_root)}: {exc}",
                           "fix or delete the file, then retry") from exc
    names = data.get("declined") if isinstance(data, dict) else None
    return frozenset(name for name in names or () if isinstance(name, str))


def record_declined(*, add: Iterable[str] = (), remove: Iterable[str] = (),
                    project_root: Path | None = None) -> frozenset[str]:
    """Update the recorded opt-outs; returns the new set. Writes only on change."""
    from pm.filesystem import durable_write_bytes

    current = declined(project_root)
    updated = (current | set(add)) - set(remove)
    if updated != current:
        body = json.dumps({"schema": 1, "declined": sorted(updated)}, indent=2) + "\n"
        durable_write_bytes(declined_path(project_root), body.encode("utf-8"))
    return frozenset(updated)


def default_package_names() -> list[str]:
    """Every package that may be declined: the ``default`` ones, any target."""
    from pm.registry import all_packages, get_package

    return [name for name in all_packages() if get_package(name).default and not get_package(name).internal]


def default_packages(names: list[str], *, target: str | None = None,
                     declined_names: frozenset[str] | None = None) -> list[str]:
    """The optional defaults among ``names`` this install should carry.

    Excludes targets the package has no build for (its ``gaps``) and packages
    the user declined. ``names`` is the lockfile's package list.
    """
    from pm.registry import get_package
    from pm.store import current_target

    target = current_target() if target is None else target
    refused = declined() if declined_names is None else declined_names
    selected = []
    for name in names:
        try:
            package = get_package(name)
        except KeyError:
            continue  # Lockfile/definition skew mid-update; drift() skips these too.
        if package.default and not package.internal and name not in refused \
                and package.missing_reason(target) is None:
            selected.append(name)
    return selected
