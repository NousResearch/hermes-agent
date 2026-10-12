"""Canonical identities used by config caches and last-known-good state."""

from pathlib import Path


def _canonical_config_path_key(path: Path) -> str:
    try:
        return str(path.resolve(strict=False))
    except (OSError, RuntimeError):
        return str(path.absolute())


def _combined_config_signature(user_sig, managed_sig) -> tuple | None:
    if user_sig is None and managed_sig is None:
        return None
    return (*(user_sig or (None,) * 4), *(managed_sig or (None,) * 4))


def _raw_config_cache_hit(
    cache: dict[str, tuple],
    path_key: str,
    cache_key: tuple,
):
    """Return cached raw config only when its content identity still matches."""
    cached = cache.get(path_key)
    if cached is not None and cached[: len(cache_key)] == cache_key:
        return cached[len(cache_key)]
    return None


def _load_config_cache_sig(config_path: Path) -> tuple[tuple | None, tuple | None]:
    """Return user and managed content identities for config cache lookups."""
    from hermes_cli import managed_scope
    from utils import file_content_signature, read_file_with_signature

    snapshot = read_file_with_signature(config_path, raise_errors=True)
    user_sig = snapshot[1] if snapshot is not None else None
    managed_dir = managed_scope.get_managed_dir()
    managed_sig = (
        file_content_signature(managed_dir / "config.yaml") if managed_dir else None
    )
    return user_sig, _combined_config_signature(user_sig, managed_sig)
