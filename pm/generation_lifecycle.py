"""Observe committed dependency changes without releasing a running reader's lease."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)


def collect_unused(project: Path) -> None:
    """Best-effort maintenance, retaining the collector's age, lease and selection rules."""
    from hermes_cli.runtime_state import collect_generations
    from pm.environments import install_state_dir
    from pm.runtime import collect_runtime_generations

    try:
        removed = collect_generations(
            project, wait=False
        ) + collect_runtime_generations(install_state_dir(project) / "pm-runtime")
    except (OSError, ValueError, RuntimeError):
        log.warning("dependency generation collection failed", exc_info=True)
        return
    if removed:
        log.info("collected %d unused dependency generations", len(removed))


class GenerationObserver:
    """A service's immutable boot selection and owning data root, captured before thread hops."""

    def __init__(self, project: Path):
        from pm.environments import _BOOT_ENVIRONMENTS, install_state_dir

        self.project = project.resolve()
        self.state = install_state_dir(project).resolve()
        self.home = self.state.parent.parent
        self.known = self.state in _BOOT_ENVIRONMENTS
        self.boot = _BOOT_ENVIRONMENTS.get(self.state)

    @contextmanager
    def scope(self):
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        token = set_hermes_home_override(self.home)
        try:
            yield
        finally:
            reset_hermes_home_override(token)

    def probe(self) -> str | None:
        """Changed reason, empty string when current, None when publication is indeterminate."""
        from hermes_cli.runtime_state import runtime_lock
        from hermes_cli.update_lock import update_in_progress
        from pm.environments import committed_venv

        if not self.known:
            return None
        try:
            with self.scope():
                if update_in_progress(self.project):
                    return None
                with runtime_lock(self.project, timeout=0) as held:
                    if not held or (self.state / "publication.json").exists():
                        return None
                    selected = committed_venv(self.project)
                    if selected is None:
                        return None
                    if selected.resolve() == self.boot:
                        return ""
                    return f"dependency generation {selected.parent.name} superseded this service's boot environment"
        except (OSError, ValueError, RuntimeError, TypeError):
            log.warning(
                "cannot establish committed dependency generation; retaining service",
                exc_info=True,
            )
            return None

    def collect(self) -> None:
        with self.scope():
            collect_unused(self.project)
