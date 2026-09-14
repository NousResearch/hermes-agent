"""Backend-owned provenance for the canonical welcome guide, not a name-only exemption."""
from pathlib import Path

ONBOARDING_PROFILE = "hermes-setup"
_MARKER = ".onboarding-guide"


def mark_onboarding_profile(path: Path) -> None:
    from hermes_cli.profiles import get_profile_dir
    from hermes_constants import secure_parent_dir
    from utils import atomic_write_text
    if path.resolve() != Path(get_profile_dir(ONBOARDING_PROFILE)).resolve():
        raise ValueError("Only the canonical welcome profile can be marked as the guide")
    path.mkdir(parents=True, exist_ok=True)
    secure_parent_dir(path / _MARKER)
    atomic_write_text(path / _MARKER, "hermes-onboarding-v1\n", mode=0o600)


def is_onboarding_profile() -> bool:
    """Provenance only; admission additionally requires durable identity-wide grace."""
    from hermes_cli.profiles import get_profile_dir
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    return (home.resolve() == Path(get_profile_dir(ONBOARDING_PROFILE)).resolve()
            and (home / _MARKER).is_file())
