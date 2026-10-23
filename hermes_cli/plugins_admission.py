"""Submit proposed plugin selections to the independent PM publisher."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


class AdmissionRefused(RuntimeError):
    """The candidate set was refused; config and environment untouched."""


class DependencyConflict(AdmissionRefused):
    """PM's resolver proved the candidate set cannot co-install with the pinned dependency set.

    The raw ``uv lock`` output names hashed workspace members, not the plugin the user asked for, so
    the message leads with the plugin and keeps the resolver's cause for the details.
    """

    def __init__(self, cause: str, *, plugin: Optional[str] = None):
        self.cause = cause
        self.plugin = plugin
        who = f"Plugin '{plugin}'" if plugin else "The plugin selection"
        super().__init__(
            f"{who} conflicts with the dependencies pinned by Hermes core or an enabled plugin, "
            f"so it was not admitted. Resolver: {cause}")


def candidate_member_dirs(
    candidate_enabled: Iterable[str],
    candidate_disabled: Iterable[str] = (),
    *,
    active_plugins_dir: Optional[Path] = None,
    extra_dirs: Iterable[Path] = (),
) -> list[Path]:
    """Shipped callers' member-list adapter; new discovery belongs to PM.

    Without an active plugins dir, preserve every home's recorded selection.
    """
    from pm.publication import candidate_members

    active = Path(active_plugins_dir) if active_plugins_dir else None
    return candidate_members(
        extra_dirs,
        proposed_home=active.parent if active else None,
        enabled=candidate_enabled,
        disabled=candidate_disabled,
    )


def admit_plugin_set_change(
    candidate_enabled: set,
    candidate_disabled: set,
    *,
    active_plugins_dir: Optional[Path] = None,
    extra_dirs: Iterable[Path] = (),
    expected_config: str | None = None,
    plugin: Optional[str] = None,
) -> None:
    """PM discovers and validates the proposed union under its install lock.

    No config or dependency selection is written by this application process. *plugin* names the
    plugin being admitted so a resolver conflict is reported against it (:class:`DependencyConflict`).
    """
    from hermes_constants import get_hermes_home
    from pm.client import sync_venv
    from pm.plugin_inputs import Selection

    home = Path(active_plugins_dir).parent if active_plugins_dir is not None else get_hermes_home()
    try:
        sync_venv(explicit=True, plugins=Selection({
            "home": str(home.resolve()), "enabled": sorted(candidate_enabled),
            "disabled": sorted(candidate_disabled), "extra_dirs": [str(Path(d).resolve()) for d in extra_dirs],
            **({"expected_config": expected_config} if expected_config is not None else {}),
        }))
    except Exception as exc:
        from pm.workspace import ResolutionConflict

        if isinstance(exc, ResolutionConflict):
            raise DependencyConflict(exc.cause, plugin=plugin) from exc
        raise AdmissionRefused(str(exc)) from exc
