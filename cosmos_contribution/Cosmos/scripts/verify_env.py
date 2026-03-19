"""
Farnsworth Environment Verifier
-------------------------------
Checks system permissions and dependencies for advanced features.
"""

import sys
import os
import shutil
import subprocess
from pathlib import Path
from loguru import logger

def check_admin():
    """Check if we have admin rights (needed for Focus Mode)."""
    try:
        is_admin = os.getuid() == 0
    except AttributeError:
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
    
    if is_admin:
        logger.info("✅ Admin Privileges: DETECTED (Focus Mode will work)")
    else:
        logger.warning("⚠️  Admin Privileges: MISSING")
        logger.warning("   -> 'Focus Mode' (Cone of Silence) requires running as Administrator/Sudo.")

def check_playwright():
    """Check if Playwright browsers are installed (needed for Scraper)."""
    try:
        from playwright.sync_api import sync_playwright
        logger.info("✅ Playwright Library: INSTALLED")
        # Checking browsers is harder without running them, but this is a good first step
    except ImportError:
        logger.error("❌ Playwright Library: MISSING")
        logger.error("   -> Run 'pip install playwright && playwright install' to enable Scrapers.")

def check_tts_engine():
    """Check for TTS capability."""
    if sys.platform == "win32":
        # Check PowerShell availability
        if shutil.which("powershell"):
            logger.info("✅ TTS Engine: PowerShell Speech Available")
        else:
            logger.warning("⚠️  TTS Engine: PowerShell NOT found.")
    else:
        if shutil.which("espeak") or shutil.which("say"):
            logger.info("✅ TTS Engine: System TTS Available")
        else:
            logger.warning("⚠️  TTS Engine: 'espeak' not found (Linux) or 'say' not found (Mac).")

def check_python_ver():
    ver = sys.version_info
    if ver.major == 3 and ver.minor >= 10:
        logger.info(f"✅ Python Version: {ver.major}.{ver.minor} (Compatible)")
    else:
        logger.error(f"❌ Python Version: {ver.major}.{ver.minor} (Requires 3.10+)")

def main():
    print("\n🔍 FARNSWORTH SYSTEM DIAGNOSTICS\n" + "="*40)
    check_python_ver()
    check_admin()
    check_playwright()
    check_tts_engine()
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
