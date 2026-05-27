#!/usr/bin/env python3
import json
import os
import subprocess
import time
from pathlib import Path

STATE_FILE = Path("~/.hermes/gateway_watchdog_state.json").expanduser()
HERMES_BIN = Path("/Users/alvin/.local/bin/hermes")
PYTHON_BIN = Path("/Users/alvin/personal/hermes-agent/venv/bin/python")
PROJECT_ROOT = Path("/Users/alvin/personal/hermes-agent")

def run_cmd(args):
    try:
        res = subprocess.run(args, capture_output=True, text=True, check=True)
        return res.stdout
    except subprocess.CalledProcessError as e:
        return e.stdout or e.stderr or ""

def get_profiles_status():
    stdout = run_cmd([str(HERMES_BIN), "profile", "list"])
    statuses = {}
    for line in stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        profile_name = parts[0].strip("◆")
        if profile_name in ["Profile", "───────────────"] or profile_name.startswith("──"):
            continue
        if len(parts) >= 3:
            status = parts[2]
            if status in ["running", "stopped"]:
                statuses[profile_name] = status
    return statuses

def start_gateway(profile):
    print(f"Starting gateway for profile: {profile}")
    # Try normal hermes gateway start first
    args = [str(HERMES_BIN)]
    if profile != "default":
        args += ["--profile", profile]
    args += ["gateway", "start"]
    
    try:
        subprocess.run(args, check=True, capture_output=True, timeout=15)
        print(f"Successfully started {profile} gateway via launchd")
        return True
    except Exception as e:
        print(f"Launchd start failed for {profile}: {e}. Falling back to background process.")
        
    # Fallback to manual background execution
    cmd_args = [str(PYTHON_BIN), "-m", "hermes_cli.main"]
    if profile != "default":
        cmd_args += ["--profile", profile]
    cmd_args += ["gateway", "run", "--replace"]
    
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(PROJECT_ROOT / "venv")
    env["PATH"] = f"{PROJECT_ROOT}/venv/bin:{env.get('PATH', '')}"
    if profile == "default":
        env["HERMES_HOME"] = "/Users/alvin/.hermes"
    else:
        env["HERMES_HOME"] = f"/Users/alvin/.hermes/profiles/{profile}"
        
    try:
        subprocess.Popen(
            cmd_args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(PROJECT_ROOT),
            env=env,
            start_new_session=True
        )
        print(f"Successfully launched {profile} gateway in background")
        return True
    except Exception as ex:
        print(f"Failed to launch {profile} gateway manually: {ex}")
        return False

def main():
    try:
        statuses = get_profiles_status()
    except Exception as e:
        print(f"Failed to get profile statuses: {e}")
        return

    if not statuses:
        print("No profiles found.")
        return

    # Load state
    state = {}
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text())
        except Exception:
            pass

    now = time.time()
    updated_state = {}

    for profile, status in statuses.items():
        if status == "stopped":
            first_seen = state.get(profile)
            if first_seen is not None:
                elapsed = now - first_seen
                print(f"Profile '{profile}' has been offline for {elapsed:.1f}s")
                if elapsed >= 300: # 5 minutes
                    start_gateway(profile)
                else:
                    updated_state[profile] = first_seen
            else:
                print(f"Profile '{profile}' detected offline. Recording timestamp.")
                updated_state[profile] = now
        else:
            # Running
            pass

    # Save state
    try:
        STATE_FILE.write_text(json.dumps(updated_state, indent=2))
    except Exception as e:
        print(f"Failed to write state file: {e}")

if __name__ == "__main__":
    main()
