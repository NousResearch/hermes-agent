"""Update-target branch resolution shared by every update/check surface."""


def resolve_update_branch(explicit: str | None = None) -> str:
    """Resolve an explicit branch, then ``updates.branch``, then ``main``.

    A present but blank explicit value retains the historical ``main`` fallback instead of
    silently selecting a configured branch.
    """
    if explicit is not None:
        return str(explicit).strip() or "main"

    # This module is imported by the startup-sensitive banner path. Keep the heavy config
    # module lazy, and avoid loading it at all when an explicit --branch already won.
    from hermes_cli.config import load_config
    updates = (load_config() or {}).get("updates", {})
    configured = updates.get("branch", "main") if isinstance(updates, dict) else "main"
    return configured.strip() if isinstance(configured, str) and configured.strip() else "main"
