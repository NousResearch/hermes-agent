#!/usr/bin/env python3
"""
Bo2bot Skill Auto-Loader
Automatically checks and sets up credentials when the skill is first used.
Users don't have to run any bash commands - this handles it all.

Usage (in your Hermes prompt or Python code):
  from bo2bot_loader import ensure_bo2bot_ready
  ensure_bo2bot_ready()  # Handles credential setup automatically
  
Or use directly:
  python3 bo2bot_loader.py
"""

import sys
from pathlib import Path

# Import the credential manager
sys.path.insert(0, str(Path(__file__).parent))
from bo2bot_cred_manager import Bo2botCredentialManager


def ensure_bo2bot_ready(interactive=None):
    """
    Check if Bo2bot credentials are ready.

    Args:
        interactive: If True, prompt for setup when missing. If False, raise.
                     Default: only prompt when stdin/stdout are TTYs.

    Returns:
        dict: Loaded credentials

    Raises:
        EnvironmentError: If credentials missing and not interactive
    """
    if interactive is None:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()

    if Bo2botCredentialManager.has_credentials():
        creds = Bo2botCredentialManager.load_credentials()
        handle = creds.get("BO2BOT_HANDLE", "unknown")
        print(f"✅ Credentials loaded for {handle}")
        print(f"   File: {Bo2botCredentialManager.CREDENTIALS_FILE}\n")
        return creds

    if not interactive:
        raise EnvironmentError(
            "Bo2bot credentials not found at "
            f"{Bo2botCredentialManager.CREDENTIALS_FILE}. "
            "Human must fill that file (README Step 1). "
            "Do not ask for secrets in chat."
        )

    print("\n📋 No Bo2bot credentials found.\n")
    creds = Bo2botCredentialManager.prompt_for_credentials(interactive=True)
    Bo2botCredentialManager.save_credentials(creds)
    print(f"\n✅ Ready to use Bo2bot! Handle: {creds.get('BO2BOT_HANDLE')}\n")
    return creds


def main():
    """Run auto-setup when called directly."""
    try:
        ensure_bo2bot_ready()
        print("✅ Setup complete. You can now use the Bo2bot skill.\n")
    except EnvironmentError as e:
        print(f"\n{e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
