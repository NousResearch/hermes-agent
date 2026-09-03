#!/usr/bin/env python3
"""
Bo2bot Credential Manager
Handles credential setup and validation for the Bo2bot skill.
Can be imported as a module or run standalone.
"""

import os
import sys
import json
from pathlib import Path
from getpass import getpass


class Bo2botCredentialManager:
    """Manages Bo2bot credentials with interactive setup."""
    
    CREDENTIALS_FILE = Path.home() / ".hermes" / "secrets" / "bo2bot.env"
    REQUIRED_FIELDS = [
        "BO2BOT_HANDLE",
        "BO2BOT_PUBLIC_ADDRESS",
        "BO2BOT_ACCOUNT_ID",
        "BO2BOT_AUTH_KEY"
    ]
    
    @staticmethod
    def load_credentials() -> dict:
        """Load credentials from .env file if it exists."""
        creds = {}
        if Bo2botCredentialManager.CREDENTIALS_FILE.exists():
            with open(Bo2botCredentialManager.CREDENTIALS_FILE, 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        creds[key] = value
        return creds
    
    @staticmethod
    def has_credentials() -> bool:
        """Check if all required credentials are present."""
        creds = Bo2botCredentialManager.load_credentials()
        return all(field in creds for field in Bo2botCredentialManager.REQUIRED_FIELDS)
    
    @staticmethod
    def prompt_for_credentials(interactive: bool = True) -> dict:
        """
        Prompt user for credentials interactively.
        
        Args:
            interactive: If False, raises error instead of prompting
            
        Returns:
            Dictionary with credential keys and values
        """
        if not interactive:
            raise EnvironmentError(
                "Bo2bot credentials not found. "
                "Run: bash ~/.hermes/skills/messaging/bo2bot-messaging/scripts/bo2bot-setup.sh"
            )
        
        existing = Bo2botCredentialManager.load_credentials()
        
        print("\n" + "=" * 60)
        print("BO2BOT CREDENTIAL SETUP")
        print("=" * 60 + "\n")
        
        creds = {}
        
        # Handle
        handle = input("Bot handle (e.g., @yourname): ").strip()
        if not handle:
            handle = existing.get('BO2BOT_HANDLE')
            if not handle:
                print("❌ Handle is required")
                sys.exit(1)
        creds['BO2BOT_HANDLE'] = handle
        
        # Address
        address = input("Public address (e.g., yourname@bo2bot.com): ").strip()
        if not address:
            address = existing.get('BO2BOT_PUBLIC_ADDRESS')
            if not address:
                print("❌ Public address is required")
                sys.exit(1)
        creds['BO2BOT_PUBLIC_ADDRESS'] = address
        
        # Account ID
        account_id = input("Account ID (acct_...): ").strip()
        if not account_id:
            account_id = existing.get('BO2BOT_ACCOUNT_ID')
            if not account_id:
                print("❌ Account ID is required")
                sys.exit(1)
        creds['BO2BOT_ACCOUNT_ID'] = account_id
        
        # Auth Key (hidden)
        auth_key = getpass("Auth key (bo2bot_...): ").strip()
        if not auth_key:
            auth_key = existing.get('BO2BOT_AUTH_KEY')
            if not auth_key:
                print("❌ Auth key is required")
                sys.exit(1)
        creds['BO2BOT_AUTH_KEY'] = auth_key
        
        return creds
    
    @staticmethod
    def save_credentials(creds: dict) -> None:
        """Save credentials to .env file with secure permissions."""
        Bo2botCredentialManager.CREDENTIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        with open(Bo2botCredentialManager.CREDENTIALS_FILE, 'w') as f:
            for key, value in creds.items():
                f.write(f"{key}={value}\n")
        
        # Set secure permissions (readable only by owner)
        os.chmod(Bo2botCredentialManager.CREDENTIALS_FILE, 0o600)
        
        print(f"\n✅ Credentials saved securely to: {Bo2botCredentialManager.CREDENTIALS_FILE}\n")
    
    @staticmethod
    def ensure_credentials(interactive: bool = True) -> dict:
        """
        Ensure credentials are available, prompting if necessary.
        
        Args:
            interactive: If True, prompt user if credentials missing.
                        If False, raise error if missing.
        
        Returns:
            Dictionary with all required credentials
        """
        creds = Bo2botCredentialManager.load_credentials()
        
        if Bo2botCredentialManager.has_credentials():
            return creds
        
        if not interactive:
            missing = [f for f in Bo2botCredentialManager.REQUIRED_FIELDS if f not in creds]
            raise EnvironmentError(
                f"Bo2bot credentials incomplete. Missing: {', '.join(missing)}\n"
                f"Run: bash ~/.hermes/skills/messaging/bo2bot-messaging/scripts/bo2bot-setup.sh"
            )
        
        new_creds = Bo2botCredentialManager.prompt_for_credentials(interactive=True)
        Bo2botCredentialManager.save_credentials(new_creds)
        return new_creds


def main():
    """Run credential setup interactively."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Bo2bot Credential Manager")
    parser.add_argument("--check", action="store_true", help="Check if credentials exist")
    parser.add_argument("--setup", action="store_true", help="Setup credentials interactively")
    parser.add_argument("--show", action="store_true", help="Show current credentials (masked)")
    
    args = parser.parse_args()
    
    if args.check:
        if Bo2botCredentialManager.has_credentials():
            print("✅ Credentials are configured")
            sys.exit(0)
        else:
            print("❌ Credentials are missing or incomplete")
            sys.exit(1)
    
    elif args.show:
        creds = Bo2botCredentialManager.load_credentials()
        if not creds:
            print("No credentials found")
            sys.exit(1)
        print("\nCurrent credentials:")
        for key, value in creds.items():
            if key == "BO2BOT_AUTH_KEY":
                masked = f"{value[:10]}***{value[-5:]}" if len(value) > 15 else "***"
                print(f"  {key}: {masked}")
            else:
                print(f"  {key}: {value}")
        print()
        sys.exit(0)
    
    elif args.setup:
        existing = Bo2botCredentialManager.load_credentials()

        if existing and Bo2botCredentialManager.has_credentials():
            print("Existing credentials found")
            confirm = input("Update credentials? (y/n): ").strip().lower()
            if confirm != 'y':
                print("No changes made")
                sys.exit(0)

        creds = Bo2botCredentialManager.prompt_for_credentials(interactive=True)
        Bo2botCredentialManager.save_credentials(creds)
        print("Next steps:")
        print("  1. Run validation: bash ~/.hermes/skills/messaging/bo2bot-messaging/scripts/bo2bot-validate.sh")
        print("  2. Or use the skill: hermes chat -s bo2bot-messaging")

    else:
        # Default: non-interactive check (safe for agents — never prompts)
        if Bo2botCredentialManager.has_credentials():
            print("✅ Credentials are configured")
            sys.exit(0)
        print("❌ Credentials are missing or incomplete")
        print(f"   Expected file: {Bo2botCredentialManager.CREDENTIALS_FILE}")
        print("   Human: fill that file per README Step 1, or run --setup in a terminal.")
        sys.exit(1)


if __name__ == "__main__":
    main()
