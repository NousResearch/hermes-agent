#!/usr/bin/env python3
"""
⚡ Facebook Access Skill — One-Click Setup for Hermes Agent

This script automates everything:
  1. Detects your Hermes profile
  2. Copies the skill files to the right location
  3. Helps you set up your Facebook Access Token
  4. Tests the connection

Usage:
  python3 setup.py
"""

import os
import sys
import shutil
import subprocess
import json

# ── Colors ──────────────────────────────────────────────────────────────
G = "\033[92m"  # green
Y = "\033[93m"  # yellow
R = "\033[91m"  # red
B = "\033[94m"  # blue
N = "\033[0m"   # reset
BOLD = "\033[1m"

SKILL_NAME = "facebook-access"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def print_banner():
    print(f"""
{B}╔══════════════════════════════════════════════════════╗
║  {N}{BOLD}Facebook Access Skill — Hermes Agent Setup{N}{B}     ║
║  {N}by Ali Ahsan (Aliahasan399){B}                         ║
╚══════════════════════════════════════════════════════╝{N}
    """)


def step(msg):
    print(f"\n{B}━━━ {msg} ━━━{N}")


def ok(msg):
    print(f"  {G}✅ {msg}{N}")


def warn(msg):
    print(f"  {Y}⚠️  {msg}{N}")


def fail(msg):
    print(f"  {R}❌ {msg}{N}")
    return False


def info(msg):
    print(f"     {msg}")


# ── Step 1: Find Hermes Profile ─────────────────────────────────────────

def detect_profile():
    """Find which Hermes profile is active."""
    step("Step 1: Detecting Hermes Profile")

    # Check common profile locations
    candidates = []

    # HERMES_HOME env var
    hermes_home = os.environ.get("HERMES_HOME", "")
    if hermes_home and os.path.isdir(hermes_home):
        candidates.append(("$HERMES_HOME", hermes_home))

    # Default profile
    default = os.path.expanduser("~/.hermes")
    if os.path.isdir(default):
        candidates.append(("default (~/.hermes)", default))

    # Hermes CLI config
    try:
        result = subprocess.run(
            ["hermes", "config", "env-path"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            env_path = result.stdout.strip()
            if env_path:
                # env_path is something like ~/.hermes/profiles/mybot/.env
                profile_dir = os.path.dirname(os.path.dirname(env_path))
                if os.path.isdir(profile_dir):
                    candidates.append(("hermes config", profile_dir))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if not candidates:
        warn("No Hermes profile detected automatically.")
        print("  Enter your Hermes profile path manually (e.g. ~/.hermes/profiles/mybot):")
        manual = input("  ➜ ").strip()
        if manual:
            candidates.append(("manual", os.path.expanduser(manual)))

    if not candidates:
        return fail("No profile found. Install Hermes Agent first: https://hermes-agent.nousresearch.com")

    # Pick the first valid one
    label, path = candidates[0]
    skills_dir = os.path.join(path, "skills")
    env_file = os.path.join(path, ".env")

    if not os.path.isdir(skills_dir):
        return fail(f"Skills directory not found at {skills_dir}")

    ok(f"Profile: {label}")
    info(f"  Skills dir: {skills_dir}")
    info(f"  Env file:   {env_file}")
    return skills_dir, env_file


# ── Step 2: Install Skill Files ─────────────────────────────────────────

def install_skill(skills_dir):
    """Copy skill files to Hermes profile."""
    step("Step 2: Installing Skill Files")

    dest = os.path.join(skills_dir, SKILL_NAME)
    if os.path.isdir(dest):
        warn(f"Skill already exists at {dest}")
        choice = input("  Overwrite? (y/N): ").strip().lower()
        if choice != "y":
            info("  Skipped — keeping existing files.")
            return True
        shutil.rmtree(dest)

    # Copy everything from this repo
    shutil.copytree(SCRIPT_DIR, dest, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
    os.chmod(os.path.join(dest, "facebook_graph.py"), 0o755)
    if os.path.isdir(os.path.join(dest, "scripts")):
        os.chmod(os.path.join(dest, "scripts", "facebook_graph.py"), 0o755)

    ok(f"Installed to {dest}")
    return True


# ── Step 3: Token Setup ─────────────────────────────────────────────────

def setup_token(env_file):
    """Check and help set up the Facebook Access Token."""
    step("Step 3: Facebook Access Token")

    # Already set?
    current_token = os.environ.get("FACEBOOK_ACCESS_TOKEN", "")

    # Also check .env file
    if not current_token and os.path.isfile(env_file):
        with open(env_file) as f:
            for line in f:
                if "FACEBOOK_ACCESS_TOKEN" in line and "=" in line:
                    val = line.split("=", 1)[1].strip().strip("'\"")
                    if val and val != "***":
                        current_token = val
                        break

    if current_token and current_token != "***":
        ok(f"Token already configured (starts with: {current_token[:15]}...)")
        return True

    warn("No Facebook Access Token found!")
    print()
    print("  You need a Facebook Access Token to use this skill.")
    print()
    print("  ── Quick Guide ─────────────────────────────────────")
    print("  1. Go to: https://developers.facebook.com/apps/creation/")
    print("  2. Select 'Manage everything on your Page'")
    print("  3. Create app → Add Pages API product")
    print("  4. Go to: https://developers.facebook.com/tools/explorer/")
    print("  5. Select your app → Get User Token")
    print("  6. Check ALL permissions & Generate")
    print("  7. Click 'Exchange' for 60-day token")
    print("  8. Switch app to Live mode (Settings → Basic → App Mode)")
    print("  ────────────────────────────────────────────────────")
    print()

    choice = input("  Enter your token now (or press Enter to skip): ").strip()
    if choice:
        token = choice
        # Write to .env
        with open(env_file, "a") as f:
            f.write(f"\n# Facebook Access Skill\nexport FACEBOOK_ACCESS_TOKEN='{token}'\n")
        ok(f"Token saved to {env_file}")
        info("  Restart your Hermes Agent session for it to take effect.")
        return True
    else:
        warn("Token setup skipped. You can add it later:")
        info(f"  echo \"export FACEBOOK_ACCESS_TOKEN='***'\" >> {env_file}")
        return True


# ── Step 4: Verify ──────────────────────────────────────────────────────

def verify():
    """Test the installation."""
    step("Step 4: Verify Installation")

    script_path = os.path.join(SCRIPT_DIR, "facebook_graph.py")
    if not os.path.isfile(script_path):
        script_path = os.path.join(SCRIPT_DIR, "scripts", "facebook_graph.py")

    if not os.path.isfile(script_path):
        return fail("facebook_graph.py not found")

    # Syntax check
    result = subprocess.run(
        [sys.executable, "-c", f"import py_compile; py_compile.compile('{script_path}', doraise=True)"],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode != 0:
        return fail(f"Syntax error in facebook_graph.py: {result.stderr}")

    ok("Script syntax is valid")

    # Test token
    token = os.environ.get("FACEBOOK_ACCESS_TOKEN", "")
    if token:
        result = subprocess.run(
            [sys.executable, script_path, "token-check"],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "FACEBOOK_ACCESS_TOKEN": token}
        )
        if result.returncode == 0:
            ok("API connection successful!")
            print(f"\n{result.stdout}")
        else:
            warn(f"Token check failed: {result.stderr[:200]}")
    else:
        warn("No token set — skipping API test")
        info("  Run this after setting your token:")
        info(f"  python3 {script_path} token-check")

    return True


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print_banner()

    result = detect_profile()
    if not result:
        sys.exit(1)
    skills_dir, env_file = result

    if not install_skill(skills_dir):
        sys.exit(1)

    if not setup_token(env_file):
        sys.exit(1)

    if not verify():
        sys.exit(1)

    print()
    print(f"{G}{'─' * 54}{N}")
    print(f"{G}✅  Setup complete! Here's how to use it:{N}")
    print()
    print(f"  {B}List your pages:{N}")
    print(f"    python3 {os.path.join(SCRIPT_DIR, 'facebook_graph.py')} list-pages")
    print()
    print(f"  {B}Create a post:{N}")
    print(f"    python3 {os.path.join(SCRIPT_DIR, 'facebook_graph.py')} create-post <page-id> \"Hello!\"")
    print()
    print(f"  {B}Via Hermes Agent:{N} just ask:")
    print(f'    "Post this on my Facebook page: Hello world!"')
    print(f'    "Show my Facebook page engagement"')
    print(f'    "Reply to comment ID 12345"')
    print()
    print(f"  {B}GitHub:{N} https://github.com/Aliahasan399/facebook-access")
    print(f"{G}{'─' * 54}{N}")
    print(f"  ⭐ Star the repo if you find this useful!")
    print(f"{G}{'─' * 54}{N}")


if __name__ == "__main__":
    main()
