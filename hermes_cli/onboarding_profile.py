"""Backend-owned provenance for the canonical welcome guide, not a name-only exemption."""
from pathlib import Path

ONBOARDING_PROFILE = "hermes-setup"
_MARKER = ".onboarding-guide"


def mark_onboarding_profile(path: Path) -> None:
    from hermes_cli.auth import _write_private_file_atomic
    from hermes_cli.profiles import get_profile_dir
    if path.resolve() != Path(get_profile_dir(ONBOARDING_PROFILE)).resolve():
        raise ValueError("Only the canonical welcome profile can be marked as the guide")
    _write_private_file_atomic(path / _MARKER, "hermes-onboarding-v1\n")


def is_onboarding_profile() -> bool:
    from hermes_cli.profiles import get_profile_dir
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    return (home.resolve() == Path(get_profile_dir(ONBOARDING_PROFILE)).resolve()
            and (home / _MARKER).is_file())
