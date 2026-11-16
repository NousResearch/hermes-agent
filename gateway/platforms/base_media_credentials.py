"""HOME credential identities screened before media trust policy."""
import os
from pathlib import Path


def home_roots() -> list[Path]:
    """Resolve native, re-homed and profile HOME roots at delivery time."""
    from gateway.platforms import base
    from hermes_constants import _account_home

    real_home = base._or_default(base.get_real_home, "", exc=(OSError, RuntimeError, ValueError))
    account_home = base._or_default(_account_home, "", exc=(OSError, RuntimeError, ValueError))
    homes = dict.fromkeys(
        Path(home) for home in (os.path.expanduser("~"), os.environ.get("HOME"), real_home, account_home) if home
    )
    homes.update(dict.fromkeys(root / "home" for root in base._credential_home_roots()))
    return list(homes)


def credential_paths(session_key: str = "") -> list[Path]:
    """Include individual container HOME mounts, not only directory mounts."""
    from gateway.platforms import base
    from gateway.platforms.docker_media_paths import _docker_persistent_sandbox_roots

    paths = [home / name for home in home_roots() for name in base._MEDIA_DELIVERY_DENIED_HOME_FILES]
    mounts = [*base._parse_docker_volume_mounts(), *base._cache_dir_container_mounts()]
    if not any(container == Path("/root") for _, container in mounts):
        mounts.extend((root, Path("/root")) for root in _docker_persistent_sandbox_roots(session_key, "home"))
    for name in base._MEDIA_DELIVERY_DENIED_HOME_FILES:
        candidate = Path("/root") / name
        matched = [(host, container) for host, container in mounts
                   if candidate == container or base._path_is_within(candidate, container)]
        if matched:
            longest = max(len(container.as_posix()) for _, container in matched)
            # Metadata only: unlike media translation, stat may follow an outbound symlink.
            paths.extend(host / candidate.relative_to(container) for host, container in matched
                         if len(container.as_posix()) == longest)
    return paths


def matches_credential_file(candidate: Path, session_key: str = "") -> bool:
    """Protect exact credential identities even through links into trusted roots."""
    from gateway.platforms import base

    try:
        candidate_stat = candidate.stat()
    except OSError:
        return True
    for path in credential_paths(session_key):
        try:
            if os.path.samestat(candidate_stat, path.stat()):
                return True
        except OSError:
            continue
    return False


def blocks_remote_fallback(raw: str, session_key: str = "") -> bool:
    """Block only a known local credential identity after normal path resolution."""
    from gateway.platforms import base

    try:
        candidate = Path(os.path.expanduser(base._normalize_media_tag_path(raw)))
        if not candidate.is_absolute():
            return False
        resolved = (base._translate_docker_container_media_path(candidate, session_key=session_key, quiet=True)
                    or base._resolve_path(candidate, strict=True))
        return bool(resolved and resolved.is_file() and matches_credential_file(resolved, session_key))
    except (OSError, RuntimeError, ValueError):
        return False
