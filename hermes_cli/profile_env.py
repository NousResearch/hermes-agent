"""Profile-specific environment path resolution."""

from pathlib import Path


def profile_root_for_env_home(env_home: str, default_root: Path) -> Path:
    """Hermes root named by an exported ``HERMES_HOME`` (or *default_root* when unset)."""
    env_home = env_home.strip()
    if not env_home:
        return default_root
    env_path = Path(env_home)
    return env_path.parent.parent if env_path.parent.name == "profiles" else env_path
