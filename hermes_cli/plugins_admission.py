"""Submit proposed plugin selections to the independent PM publisher."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


class AdmissionRefused(RuntimeError):
    """The candidate set was refused; config and environment untouched."""


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
) -> None:
    """PM discovers and validates the proposed union under its install lock.

    No config or dependency selection is written by this application process.
    """
    from hermes_constants import get_hermes_home
    from pm.client import sync_venv

    home = Path(active_plugins_dir).parent if active_plugins_dir is not None else get_hermes_home()
    try:
        sync_venv(explicit=True, selection={
            "home": str(home.resolve()), "enabled": sorted(candidate_enabled),
            "disabled": sorted(candidate_disabled), "extra_dirs": [str(Path(d).resolve()) for d in extra_dirs],
            **({"expected_config": expected_config} if expected_config is not None else {}),
        })
    except Exception as exc:
        raise AdmissionRefused(str(exc)) from exc
