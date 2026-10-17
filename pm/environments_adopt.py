"""Adopt a newly selected dependency generation inside a running process.

A plugin that installs a requirement at runtime wants it importable without a restart. That
is safe only when nothing this process already loaded would change underneath it; otherwise
the new generation waits for the next process, which boots straight into it.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import os
import re
import site
import sys
from pathlib import Path

from pm.environments import install_state_dir, selected_venv, site_packages, venv_bin_dir

logger = logging.getLogger(__name__)


def running_environment(project_root: Path) -> Path | None:
    """This install's generation, selected or not, whose site-packages this process imports from.

    A generation that is no longer selected still counts: an update landed under this process,
    and a request from it is recorded and built but only the next process can load the result.
    """
    entries = {Path(entry).resolve() for entry in sys.path if entry}
    try:
        selected = selected_venv(project_root)
    except (OSError, RuntimeError, ValueError):
        return None
    for environment in (selected, *sorted((install_state_dir(project_root) / "environments").glob("*/venv"))):
        if site_packages(environment).resolve() in entries:
            return environment
    return None


def _distributions(environment: Path) -> dict[str, importlib.metadata.Distribution]:
    return {re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower(): dist
            for dist in importlib.metadata.distributions(path=[str(site_packages(environment))])}


def _loaded_from_changed(previous: Path, selected: Path) -> list[str]:
    """Modules loaded from a distribution whose version changed or disappeared, read from RECORD."""
    new = _distributions(selected)
    changed = set()
    for name, dist in _distributions(previous).items():
        if name not in new or new[name].version != dist.version:
            changed.update(Path(str(dist.locate_file(file))).resolve() for file in dist.files or ())
    roots = (str(site_packages(previous)), str(site_packages(previous).resolve()))
    loaded = []
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if isinstance(file, str) and file.startswith(roots) and Path(file).resolve() in changed:
            loaded.append(name)
    return loaded


def _replace(value: str, old: Path, new: Path) -> str:
    return os.pathsep.join(str(new) if part and Path(part).resolve() == old.resolve() else part
                           for part in value.split(os.pathsep))


def _lost_extras(previous: Path, selected: Path) -> list[str]:
    """Extras whose anchors live in *previous*'s site-packages but not in *selected*'s.

    A selection that drops a feature the running environment had cannot satisfy the process
    being swapped onto it: after adoption the import fails, and dependents quietly degrade
    (aiohttp -> ``AIOHTTP_AVAILABLE`` False -> api_server never listens; 2026-10-09). The
    probe is a filesystem check per anchor, so it costs nothing on the hot path and needs no
    import of the trees involved.
    """
    from pm.extras import ANCHORS, _anchors, _installed_in

    old, new = site_packages(previous), site_packages(selected)
    lost = []
    for extra in sorted(ANCHORS):
        anchors = _anchors(extra)
        if all(_installed_in(old, anchor) for anchor in anchors) \
                and not all(_installed_in(new, anchor) for anchor in anchors):
            lost.append(extra)
    return lost


def adopt(previous: Path, selected: Path, running: Path) -> bool:
    """Swap this process from ``previous`` onto ``selected``; False means only a restart can load it."""
    from hermes_cli.runtime_state import lease_generation

    if running.resolve() != previous.resolve():
        return False  # An update landed under this process; its code no longer matches the selection.
    if selected.resolve() == previous.resolve():
        return True
    if _loaded_from_changed(previous, selected):
        return False
    lost = _lost_extras(previous, selected)
    if lost:
        # Refuse rather than silently run a downgraded environment: this process keeps the
        # features it was started with, and the warning names the generation to rebuild. A
        # restart still boots the selected generation, so whoever reads this must fix the
        # selection (rebuild it with the payload's extras) rather than restart and forget.
        logger.warning(
            "refusing to adopt dependency generation %s in this process: it is missing %s, "
            "which %s provides. Rebuild the generation with the payload's extras "
            "(`hermes pm install`) before restarting; a restart would boot it and lose those "
            "features.", selected.parent.name, ", ".join(lost), previous.parent.name,
        )
        return False
    lease_generation(selected)  # The old lease stays: loaded modules keep reading from it.
    old, new = site_packages(previous), site_packages(selected)
    sys.path[:] = [str(new) if entry and Path(entry).resolve() == old.resolve() else entry for entry in sys.path]
    known = {pth.name for pth in old.glob("*.pth")}
    for pth in sorted(new.glob("*.pth")):
        if pth.name not in known:
            site.addpackage(str(new), pth.name, None)
    if os.environ.get("PYTHONPATH"):
        os.environ["PYTHONPATH"] = _replace(os.environ["PYTHONPATH"], old, new)
    os.environ["PATH"] = _replace(os.environ.get("PATH", ""), venv_bin_dir(previous), venv_bin_dir(selected))
    importlib.invalidate_caches()
    return True


def _running_and_selected(project_root: Path) -> tuple[Path, Path] | None:
    running = running_environment(project_root)
    if running is None:
        return None
    try:
        return running, selected_venv(project_root)
    except (OSError, RuntimeError, ValueError):
        return None


def restart_needed(project_root: Path) -> str | None:
    """Why this process must restart to load the selected generation, or None when it runs it.

    Also None when the process does not run from one of this install's generations (a developer
    venv, Nix): it never loads a PM selection, so a restart would not change what it imports.
    The reason names the selected generation, so it changes with every new publication.
    """
    pair = _running_and_selected(project_root)
    if pair is None or pair[0].resolve() == pair[1].resolve():
        return None
    running, selected = pair
    return (f"dependency generation {selected.parent.name} was published after this process "
            f"loaded {running.parent.name}")


def adopt_selected(project_root: Path) -> bool:
    """Move this process onto the selected generation. True when it already runs it, adopted it,
    or runs from no generation of this install at all (there is nothing to adopt)."""
    pair = _running_and_selected(project_root)
    return pair is None or adopt(pair[0], pair[1], pair[0])
