"""Post-import restoration of profile command wrappers."""

from pathlib import Path


def restore_profile_wrapper_scripts(hermes_root: Path) -> list[tuple[str, bool]]:
    """Restore aliases for imported profiles and report their availability."""
    profiles_dir = hermes_root / "profiles"
    restored_profiles: list[tuple[str, bool]] = []
    if profiles_dir.is_dir():
        try:
            from hermes_cli.profiles import (
                create_wrapper_script, check_alias_collision,
                _is_wrapper_dir_in_path, _get_wrapper_dir,
            )
            for entry in sorted(profiles_dir.iterdir()):
                if not entry.is_dir():
                    continue
                profile_name = entry.name
                # Only create wrappers for directories with config
                if not (entry / "config.yaml").exists() and not (entry / ".env").exists():
                    continue
                collision = check_alias_collision(profile_name)
                if collision:
                    print(f"  Skipped alias '{profile_name}': {collision}")
                    restored_profiles.append((profile_name, False))
                else:
                    wrapper = create_wrapper_script(profile_name)
                    restored_profiles.append((profile_name, wrapper is not None))

            if restored_profiles:
                created = [n for n, ok in restored_profiles if ok]
                skipped = [n for n, ok in restored_profiles if not ok]
                if created:
                    print(f"\n  Profile aliases restored: {', '.join(created)}")
                if skipped:
                    print(f"  Profile aliases skipped:  {', '.join(skipped)}")
                if not _is_wrapper_dir_in_path():
                    print(f"\n  Note: {_get_wrapper_dir()} is not in your PATH.")
                    print('  Add to your shell config (~/.bashrc or ~/.zshrc):')
                    print('    export PATH="$HOME/.local/bin:$PATH"')
        except ImportError:
            # hermes_cli.profiles might not be available (fresh install)
            if any(profiles_dir.iterdir()):
                print("\n  Profiles detected but aliases could not be created.")
                print("  Run: hermes profile list  (after installing hermes)")
    return restored_profiles
