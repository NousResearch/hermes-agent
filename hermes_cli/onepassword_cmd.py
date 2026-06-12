"""
1Password CLI command handler for hermes.

Provides commands to:
- Migrate .env files to 1Password references
- Set up 1Password integration
- Verify 1Password setup
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from hermes_cli.onepassword_resolver import (
    is_1password_available,
    get_1password_vaults,
    migrate_env_file_to_onepassword,
    create_1password_item,
)
from hermes_cli.config import get_env_path
from hermes_constants import get_hermes_home, display_hermes_home


def cmd_1password(args: argparse.Namespace) -> int:
    """Main 1password command handler."""
    if not hasattr(args, "onepassword_cmd"):
        _print_help()
        return 0
    
    if args.onepassword_cmd == "status":
        return _cmd_1password_status()
    elif args.onepassword_cmd == "migrate":
        return _cmd_1password_migrate(args)
    elif args.onepassword_cmd == "setup":
        return _cmd_1password_setup(args)
    else:
        _print_help()
        return 1


def _print_help():
    """Print 1password command help."""
    print("""
╔════════════════════════════════════════════════════════════════╗
║             1Password Integration for Hermes                   ║
╚════════════════════════════════════════════════════════════════╝

Secure credential management using 1Password instead of plaintext .env files.

Usage:
  hermes 1password status              Show 1Password CLI status
  hermes 1password setup               Setup 1Password integration
  hermes 1password migrate             Convert .env to 1Password references
  hermes 1password migrate --profile   Migrate specific profile
  hermes 1password migrate --all       Migrate all profiles

Requirements:
  - 1Password account (https://1password.com)
  - 1Password CLI installed (https://developer.1password.com/docs/cli/)
  - Signed in to 1Password (eval $(op signin) before running hermes)

Benefits:
  - Secrets never stored in plaintext on disk
  - 8x reduced blast radius (8 profiles = 8 copies of plaintext credentials)
  - Easy credential rotation (change in 1Password, not in 9 .env files)
  - Team credential sharing via 1Password vaults
""")


def _cmd_1password_status() -> int:
    """Show 1Password CLI status."""
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║            1Password CLI Status                             ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    
    if not is_1password_available():
        print("❌ 1Password CLI not available")
        print()
        print("   Install from: https://developer.1password.com/docs/cli/get-started/")
        print("   Then sign in:  eval $(op signin)")
        print()
        return 1
    
    print("✅ 1Password CLI available and signed in")
    print()
    
    # List available vaults
    vaults = get_1password_vaults()
    if vaults:
        print("   Available vaults:")
        for vault in vaults:
            marker = "👑" if vault == "Empire" else "  "
            print(f"   {marker} {vault}")
    else:
        print("   ⚠️  Could not list vaults")
    
    print()
    return 0


def _cmd_1password_setup(args: argparse.Namespace) -> int:
    """Setup 1Password integration."""
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║            1Password Setup                                   ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    
    # Check 1Password CLI status
    if not is_1password_available():
        print("❌ 1Password CLI not available or not signed in")
        print()
        print("   Steps:")
        print("   1. Install: https://developer.1password.com/docs/cli/get-started/")
        print("   2. Sign in: eval $(op signin)")
        print("   3. Then run: hermes 1password setup")
        print()
        return 1
    
    print("✅ 1Password CLI is available")
    print()
    
    vaults = get_1password_vaults()
    if not vaults:
        print("❌ Could not list vaults")
        return 1
    
    print("Available vaults:")
    for i, vault in enumerate(vaults, 1):
        print(f"  {i}. {vault}")
    
    # For now, recommend using "Empire" vault
    print()
    print("💡 Recommended: Use 'Empire' vault for Hermes credentials")
    print()
    print("Next step: Run 'hermes 1password migrate' to convert .env files")
    print()
    return 0


def _cmd_1password_migrate(args: argparse.Namespace) -> int:
    """Migrate .env files to 1Password references."""
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         1Password Migration Tool                             ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    
    # Check 1Password CLI
    if not is_1password_available():
        print("❌ 1Password CLI not available or not signed in")
        print()
        print("   Run: eval $(op signin)")
        print("   Then: hermes 1password migrate")
        print()
        return 1
    
    # Determine which .env file(s) to migrate
    env_files_to_migrate = []
    
    if args.all:
        # Migrate all profiles
        hermes_home = get_hermes_home()
        profiles_dir = hermes_home / "profiles"
        
        if profiles_dir.exists():
            for profile_dir in profiles_dir.iterdir():
                if profile_dir.is_dir():
                    env_file = profile_dir / ".env"
                    if env_file.exists():
                        env_files_to_migrate.append((env_file, profile_dir.name))
        
        # Also add default profile
        default_env = hermes_home / ".env"
        if default_env.exists():
            env_files_to_migrate.append((default_env, "default"))
    
    elif args.profile:
        # Migrate specific profile
        hermes_home = get_hermes_home()
        profile_dir = hermes_home / "profiles" / args.profile
        env_file = profile_dir / ".env"
        
        if not env_file.exists():
            print(f"❌ Profile .env not found: {env_file}")
            return 1
        
        env_files_to_migrate.append((env_file, args.profile))
    
    else:
        # Migrate default profile only
        env_file = get_env_path()
        if not env_file.exists():
            print(f"❌ .env file not found: {env_file}")
            return 1
        
        env_files_to_migrate.append((env_file, "default"))
    
    if not env_files_to_migrate:
        print("❌ No .env files found to migrate")
        return 1
    
    # Perform migration
    print(f"Migrating {len(env_files_to_migrate)} .env file(s)...")
    print()
    
    total_migrated = 0
    total_skipped = 0
    total_errors = 0
    
    for env_file, profile_name in env_files_to_migrate:
        print(f"📁 {profile_name}:")
        print(f"   File: {env_file}")
        
        vault_name = "Empire"
        result = migrate_env_file_to_onepassword(env_file, vault_name=vault_name)
        
        total_migrated += result["migrated"]
        total_skipped += result["skipped"]
        total_errors += len(result["errors"])
        
        if result["errors"]:
            print(f"   ❌ Errors:")
            for error in result["errors"]:
                print(f"      - {error}")
        
        if result["migrated"] > 0:
            print(f"   ✅ Migrated {result['migrated']} credential(s)")
        
        if result["skipped"] > 0:
            print(f"   ⏭️  Skipped {result['skipped']} (already op:// or placeholder)")
        
        if result["backup_path"]:
            print(f"   💾 Backup: {result['backup_path']}")
        
        if result["secrets_created"]:
            print(f"   🔐 Created in 1Password:")
            for secret in result["secrets_created"]:
                print(f"      - {secret['env_var']} → {secret['name']}")
        
        print()
    
    # Summary
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                    Migration Summary                         ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"Total migrated:  {total_migrated}")
    print(f"Total skipped:   {total_skipped}")
    print(f"Total errors:    {total_errors}")
    print()
    
    if total_errors == 0:
        print("✅ Migration completed successfully!")
        print()
        print("Next steps:")
        print("  1. Review the migrated .env files")
        print("  2. Verify with: hermes 1password status")
        print("  3. Test your configuration: hermes config")
        print()
        return 0
    else:
        print("⚠️  Migration completed with errors. Please review above.")
        print()
        return 1


def register_1password_parser(subparsers):
    """Register the 1password command with argument parser."""
    from hermes_cli.config import get_env_path
    
    onepassword_parser = subparsers.add_parser(
        "1password",
        help="Secure credential management with 1Password",
        description=(
            "Manage 1Password integration for secure credential storage. "
            "Replace plaintext .env files with 1Password references."
        ),
        aliases=["1p"],
    )
    
    onepassword_subparsers = onepassword_parser.add_subparsers(
        dest="onepassword_cmd",
        help="1Password subcommand"
    )
    
    # status command
    onepassword_subparsers.add_parser(
        "status",
        help="Check 1Password CLI status"
    )
    
    # setup command
    onepassword_subparsers.add_parser(
        "setup",
        help="Setup 1Password integration"
    )
    
    # migrate command
    migrate_parser = onepassword_subparsers.add_parser(
        "migrate",
        help="Migrate .env files to 1Password references"
    )
    migrate_parser.add_argument(
        "--all",
        action="store_true",
        help="Migrate all profiles (default: migrate only active profile)"
    )
    migrate_parser.add_argument(
        "--profile",
        type=str,
        help="Migrate specific profile"
    )
    migrate_parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not backup original .env files"
    )
    
    onepassword_parser.set_defaults(func=cmd_1password)
