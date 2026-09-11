"""Validate a proposed plugin set, then publish config and runtime selection."""

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
    """Member dirs implied by the PROPOSED enabled sets: per plugins dir,
    its enabled names — the active dir's replaced by the candidate set
    (removals excluded), other homes unchanged — filtered to dirs that
    actually declare python deps. ``extra_dirs`` (the install target)
    join when they declare deps."""
    from pm.workspace import enabled_member_dirs, _is_member_candidate

    active = Path(active_plugins_dir) if active_plugins_dir else None
    members = enabled_member_dirs(
        proposed_home=active.parent if active else None,
        enabled=candidate_enabled, disabled=candidate_disabled,
    )
    for directory in extra_dirs:
        directory = Path(directory)
        if directory not in members and _is_member_candidate(directory):
            members.append(directory)
    return members


def _config_path() -> Path:
    from hermes_cli.config import get_hermes_home

    return get_hermes_home() / "config.yaml"


def _config_commit(candidate_enabled: set, candidate_disabled: set):
    """Journal both config versions before publishing either config or facts."""
    import shutil
    import tempfile

    from pm.paths import repo_root
    from hermes_cli.runtime_state import begin_publication, _atomic_bytes
    from hermes_cli.config import read_raw_config
    from utils import atomic_roundtrip_yaml_save

    path = _config_path()
    config = read_raw_config()
    plugins_cfg = config.setdefault("plugins", {})
    if not isinstance(plugins_cfg, dict):
        raise ValueError(f"plugins must be a mapping in {path}")
    plugins_cfg["enabled"] = sorted(candidate_enabled)
    plugins_cfg["disabled"] = sorted(candidate_disabled)
    with tempfile.TemporaryDirectory(prefix="hermes-admission-") as temporary:
        staged = Path(temporary) / "config.yaml"
        if path.is_file():
            shutil.copyfile(path, staged)
        atomic_roundtrip_yaml_save(staged, config)
        proposed = staged.read_bytes()
    publication = begin_publication(repo_root(), path, proposed)
    try:
        _atomic_bytes(path, proposed)
    except BaseException:
        publication()
        raise
    return publication


def admit_plugin_set_change(
    candidate_enabled: set,
    candidate_disabled: set,
    *,
    active_plugins_dir: Optional[Path] = None,
    extra_dirs: Iterable[Path] = (),
) -> None:
    """Validate the proposed sets against the active environment and
    commit — env selection and config move together, inside pm's single
    locked transaction.

    Raises :class:`AdmissionRefused` BEFORE anything is published when
    the candidate union fails to resolve (conflict, frozen feature set,
    …) or when the config write itself fails. The active environment and
    the previous config bytes are kept EXACTLY — no rollback re-resolve.
    """
    from pm.client import sync_venv

    extra_dirs = tuple(extra_dirs)
    try:
        sync_venv(
            explicit=True,
            plugin_dirs=lambda: candidate_member_dirs(
                candidate_enabled, candidate_disabled, active_plugins_dir=active_plugins_dir, extra_dirs=extra_dirs
            ),
            before_publish=lambda: _config_commit(candidate_enabled, candidate_disabled),
        )
    except Exception as exc:
        raise AdmissionRefused(str(exc)) from exc
