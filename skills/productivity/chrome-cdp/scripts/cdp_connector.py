#!/usr/bin/env python3
"""
Chrome CDP Connector - Universal browser automation via Chrome DevTools Protocol.

Works on Linux, Mac, Windows natively, and WSL2 via --wsl-proxy mode.
No Selenium, no Playwright — raw CDP over HTTP + WebSocket.

Requirements: pip install websockets

WSL2 mode: When running in WSL2, Chrome runs on Windows side. WSL2 cannot reach
Windows 127.0.0.1 directly. Use --wsl-proxy to delegate commands through Windows py.exe.
"""

import argparse
import base64
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error

try:
    import websockets
    import asyncio
except ImportError:
    print("Error: 'websockets' library required. Install with: pip install websockets", file=sys.stderr)
    sys.exit(1)


# ─── Config ───────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.json")

DEFAULT_CONFIG = {
    "default_host": "127.0.0.1",
    "default_port": 9222,
    "chrome_profile": None,
    "chrome_binary": None,
    "auto_launch": False,
    "headless": False,
    "wsl_proxy": False,
    "wsl_windows_python": "/mnt/c/Windows/py.exe",
    "wsl_connector_path": None,
}


def load_config():
    """Load config from file, return defaults if not found."""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
        merged = DEFAULT_CONFIG.copy()
        merged.update(config)
        return merged
    return DEFAULT_CONFIG.copy()


def save_config(config):
    """Save config to file."""
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {CONFIG_PATH}")


# ─── Platform Detection ───────────────────────────────────────────────────────

def is_wsl():
    """Detect if running inside WSL."""
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except Exception:
        return False


def is_wsl2():
    """Detect WSL2 specifically (has network isolation)."""
    if not is_wsl():
        return False
    try:
        with open("/proc/version", "r") as f:
            content = f.read().lower()
            # WSL2 runs in a VM, WSL1 does not
            return "microsoft-standard" in content or "wsl2" in content
    except Exception:
        return True  # Assume WSL2 if can't tell


def get_wsl_host_ip():
    """Get the Windows host IP from inside WSL2."""
    try:
        result = subprocess.run(
            ["ip", "route", "show"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.split("\n"):
            if "default" in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == "via" and i + 1 < len(parts):
                        return parts[i + 1]
    except Exception:
        pass
    return None


def find_chrome_binary():
    """Find Chrome binary on this system."""
    system = platform.system()
    if system == "Windows":
        paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ]
        for p in paths:
            if os.path.exists(p):
                return p
    elif system == "Darwin":
        p = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        if os.path.exists(p):
            return p
    else:  # Linux
        for binary in ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]:
            try:
                result = subprocess.run(["which", binary], capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout.strip()
            except Exception:
                continue
    return None



def check_compatibility():
    """Check system compatibility for chrome-cdp skill."""
    import shutil
    issues = []
    warnings = []

    # Python version
    py = sys.version_info
    if py < (3, 8):
        issues.append(f"Python {py.major}.{py.minor} detected. Requires Python >= 3.8")
    else:
        print(f"[OK] Python {py.major}.{py.minor}.{py.micro}")

    # Platform
    system = platform.system()
    if system not in ("Linux", "Darwin", "Windows"):
        issues.append(f"Unsupported platform: {system}. Supports Linux, macOS, Windows.")
    else:
        print(f"[OK] Platform: {system} {platform.machine()}")

    # WSL detection
    if is_wsl():
        print("[OK] WSL detected — WSL proxy mode available")
        if is_wsl2():
            print("[OK] WSL2 detected — network isolation handled via proxy")

    # websockets library
    try:
        import websockets
        print(f"[OK] websockets library available (v{websockets.__version__})")
    except ImportError:
        issues.append("websockets not found. Install: pip install websockets")

    # Chrome/Chromium binary
    chrome = find_chrome_binary()
    if chrome:
        print(f"[OK] Chrome found: {chrome}")
    else:
        warnings.append("Chrome/Chromium not found. Install Chrome or run: apt install chromium-browser")
        warnings.append("You can still use the skill with remote Chrome via --host flag")

    return len(issues) == 0, issues, warnings




def get_default_profile_path():
    """Get default Chrome profile path for this platform."""
    system = platform.system()
    if system == "Windows":
        return os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "User Data")
    elif system == "Darwin":
        return os.path.expanduser("~/Library/Application Support/Google/Chrome")
    else:
        return os.path.expanduser("~/.config/google-chrome")


# ─── CDP HTTP + WebSocket ─────────────────────────────────────────────────────

def http_get(host, port, path, timeout=5):
    """Make HTTP GET to Chrome CDP."""
    url = f"http://{host}:{port}{path}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def get_tabs(host, port):
    """Get list of open tabs."""
    return http_get(host, port, "/json") or []


def get_first_tab_id(host, port):
    """Get first available page tab ID."""
    tabs = get_tabs(host, port)
    for tab in tabs:
        if tab.get("type") == "page":
            return tab.get("id")
    return tabs[0].get("id") if tabs else None


async def cdp_send(ws_url, method, params=None, timeout=10):
    """Send CDP command over WebSocket."""
    msg_id = int(time.time() * 1000) % 100000
    message = {"id": msg_id, "method": method}
    if params:
        message["params"] = params
    try:
        async with websockets.connect(ws_url, max_size=50 * 1024 * 1024) as ws:
            await ws.send(json.dumps(message))
            while True:
                response = await asyncio.wait_for(ws.recv(), timeout=timeout)
                data = json.loads(response)
                if data.get("id") == msg_id:
                    return data
    except Exception as e:
        return {"error": str(e)}


def run_cdp(host, port, tab_id, method, params=None, timeout=10):
    """Run CDP command on a specific tab."""
    ws_url = f"ws://{host}:{port}/devtools/page/{tab_id}"
    return asyncio.run(cdp_send(ws_url, method, params, timeout))


# ─── WSL Proxy Mode ──────────────────────────────────────────────────────────

def get_self_path():
    """Get this script's path in a way that works from both WSL and Windows."""
    return os.path.abspath(__file__)


def wsl_path_to_windows(wsl_path):
    """Convert WSL path to Windows path (e.g., /home/user/script.py -> \\\\wsl$\\Ubuntu\\home\\user\\script.py)."""
    # Use wslpath if available
    try:
        result = subprocess.run(
            ["wslpath", "-w", wsl_path],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    # Fallback: manual conversion
    if wsl_path.startswith("/home/"):
        return wsl_path.replace("/mnt/", "").replace("/", "\\")
    return wsl_path


def run_via_wsl_proxy(args, config):
    """Delegate command execution to Windows py.exe (for WSL2 -> Windows Chrome)."""
    py_exe = config.get("wsl_windows_python", "/mnt/c/Windows/py.exe")

    # Get connector path - either from config or auto-detect
    connector_path = config.get("wsl_connector_path")
    if not connector_path:
        # Convert our own path to Windows path
        connector_path = wsl_path_to_windows(get_self_path())

    # Build command args, removing --wsl-proxy to avoid infinite loop
    cmd = [py_exe, "-3", connector_path]

    # Pass all original args except --wsl-proxy
    skip_next = False
    for i, arg in enumerate(sys.argv[1:]):
        if skip_next:
            skip_next = False
            continue
        if arg == "--wsl-proxy":
            continue
        cmd.append(arg)

    # Build args: command first, then --host, then rest
    # Separate the command from its arguments
    raw_args = []
    skip_next = False
    for c in cmd[3:]:  # Skip py.exe -3 connector_path
        if c == "--host":
            skip_next = True
            continue
        if skip_next:
            skip_next = False
            continue
        raw_args.append(c)

    # raw_args[0] is the command (status, tabs, navigate, etc.)
    # raw_args[1:] are the rest of the args
    if raw_args:
        command = raw_args[0]
        rest = raw_args[1:]
        cmd = cmd[:3] + [command, "--host", "127.0.0.1"] + rest
    else:
        cmd = cmd[:3] + raw_args

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        return result.returncode
    except subprocess.TimeoutExpired:
        print("Error: WSL proxy command timed out", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: WSL proxy failed: {e}", file=sys.stderr)
        return 1


# ─── Commands ─────────────────────────────────────────────────────────────────

def cmd_setup(args):
    """Interactive setup wizard."""
    config = load_config()

    print("=" * 60)
    print("  Chrome CDP Connector - Setup Wizard")
    print("=" * 60)
    print()

    # Platform detection
    platform_name = platform.system()
    wsl = is_wsl2()

    print(f"Platform: {platform_name}", end="")
    if wsl:
        print(" (WSL2 detected)")
        print("Note: WSL2 cannot reach Windows localhost directly.")
    else:
        print()

    # Chrome detection
    chrome = config.get("chrome_binary") or find_chrome_binary()
    if chrome:
        print(f"[OK] Chrome found: {chrome}")
    else:
        print("[!] Chrome not found on this system.")
        if wsl:
            print("    Chrome must be installed on Windows (not in WSL).")
            print("    This is normal for WSL2 setups.")
        else:
            print("    Install Chrome or Chromium first.")
    print()

    # WSL2 proxy mode setup
    if wsl:
        print("WSL2 Mode Configuration")
        print("-" * 40)
        print("Since you're in WSL2, Chrome runs on Windows.")
        print("Options:")
        print("  1) Use --wsl-proxy mode (RECOMMENDED)")
        print("     Commands delegate to Windows py.exe automatically")
        print("  2) Use Windows host IP (advanced, may not work)")
        print()

        wsl_choice = input("Choose [1/2] (default: 1): ").strip() or "1"
        if wsl_choice == "1":
            config["wsl_proxy"] = True

            # Find Windows Python
            py_exe = "/mnt/c/Windows/py.exe"
            if os.path.exists(py_exe):
                print(f"[OK] Windows Python found: {py_exe}")
            else:
                print("[!] Windows py.exe not found at default path")
                py_exe = input("Enter path to Windows Python: ").strip()
            config["wsl_windows_python"] = py_exe

            # Convert connector path for Windows side
            windows_connector = wsl_path_to_windows(get_self_path())
            config["wsl_connector_path"] = windows_connector
            print(f"Connector path (Windows): {windows_connector}")

            # Test the proxy
            print("\nTesting WSL proxy...")
            test_cmd = [py_exe, "-3", windows_connector, "config"]
            try:
                result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print("[OK] WSL proxy working!")
                else:
                    print(f"[!] WSL proxy test failed: {result.stderr}")
            except Exception as e:
                print(f"[!] Could not test proxy: {e}")
        else:
            config["wsl_proxy"] = False
            host_ip = get_wsl_host_ip()
            if host_ip:
                config["default_host"] = host_ip
                print(f"Windows host IP: {host_ip}")
            else:
                config["default_host"] = input("Enter Windows host IP: ").strip()
    else:
        config["wsl_proxy"] = False
        config["default_host"] = "127.0.0.1"

    # Profile setup
    print()
    print("Chrome Profile Setup")
    print("-" * 40)
    print("1) Use existing Chrome profile (enter path)")
    print("2) Create fresh profile for CDP (recommended)")
    print("3) Delete existing CDP profile and start fresh")
    print()

    if config.get("chrome_profile"):
        print(f"Current profile: {config['chrome_profile']}")
    else:
        print("No profile configured yet.")
    print()

    if hasattr(args, "non_interactive") and args.non_interactive:
        if not config.get("chrome_profile"):
            fresh_path = os.path.join(tempfile.gettempdir(), "chrome_cdp_profile")
            config["chrome_profile"] = fresh_path
            os.makedirs(fresh_path, exist_ok=True)
            print(f"Auto-created fresh profile: {fresh_path}")
        else:
            print(f"Using existing profile: {config['chrome_profile']}")
    else:
        choice = input("Choose option [1/2/3] (default: 2): ").strip() or "2"

        if choice == "1":
            path = input("Enter Chrome profile path: ").strip()
            if not os.path.exists(path):
                print(f"Warning: Path does not exist: {path}")
                confirm = input("Save anyway? [y/N]: ").strip().lower()
                if confirm != "y":
                    print("Cancelled.")
                    return 1
            config["chrome_profile"] = path

        elif choice == "2":
            if wsl:
                # For WSL, suggest Windows temp path
                default_path = "D:\\Hermes\\chrome_cdp_profile"
                print(f"Note: For WSL, use a Windows path (accessible from both sides)")
                path = input(f"Profile path (Enter for default: {default_path}): ").strip()
                if not path:
                    path = default_path
            else:
                default_path = os.path.join(tempfile.gettempdir(), "chrome_cdp_profile")
                path = input(f"Profile path (Enter for default: {default_path}): ").strip()
                if not path:
                    path = default_path
                os.makedirs(path, exist_ok=True)
            config["chrome_profile"] = path
            print(f"Profile set to: {path}")

        elif choice == "3":
            if config.get("chrome_profile") and os.path.exists(config["chrome_profile"]):
                confirm = input(f"Delete {config['chrome_profile']}? [y/N]: ").strip().lower()
                if confirm == "y":
                    shutil.rmtree(config["chrome_profile"])
                    print("Profile deleted.")
            fresh_path = os.path.join(tempfile.gettempdir(), "chrome_cdp_profile")
            os.makedirs(fresh_path, exist_ok=True)
            config["chrome_profile"] = fresh_path
            print(f"Fresh profile: {fresh_path}")

    # Port
    print()
    port = input(f"Debugging port (default: {config['default_port']}): ").strip()
    if port:
        config["default_port"] = int(port)

    # Save connector path for WSL mode
    if config.get("wsl_proxy"):
        config["wsl_connector_path"] = wsl_path_to_windows(get_self_path())

    save_config(config)

    print()
    print("=" * 60)
    print("  Setup complete!")
    print(f"  Profile: {config['chrome_profile']}")
    print(f"  Host: {config['default_host']}")
    print(f"  Port: {config['default_port']}")
    if config.get("wsl_proxy"):
        print(f"  WSL Proxy: ON (via {config.get('wsl_windows_python')})")
    print("=" * 60)
    return 0


def cmd_config(args):
    """Show current configuration."""
    config = load_config()
    # Hide sensitive or internal fields for display
    display = config.copy()
    print(json.dumps(display, indent=2))
    return 0


def cmd_status(args):
    """Check if Chrome CDP is reachable."""
    config = load_config()

    # WSL proxy: delegate to Windows
    if config.get("wsl_proxy") and not getattr(args, "_internal", False):
        return run_via_wsl_proxy(args, config)

    host = getattr(args, "host", None) or config["default_host"]
    port = getattr(args, "port", None) or config["default_port"]

    # Try /json/version for more info
    version = http_get(host, port, "/json/version")
    tabs = get_tabs(host, port)

    if version or tabs is not None:
        result = {
            "connected": True,
            "host": host,
            "port": port,
            "platform": platform.system(),
        }
        if version:
            result["browser"] = version.get("Browser", "unknown")
            result["protocol_version"] = version.get("Protocol-Version", "unknown")
        result["tab_count"] = len(tabs) if tabs else 0
        print(json.dumps(result, indent=2))
        return 0
    else:
        print(json.dumps({"connected": False, "host": host, "port": port}))
        return 1


def cmd_tabs(args):
    """List open tabs."""
    config = load_config()
    if config.get("wsl_proxy") and not getattr(args, "_internal", False):
        return run_via_wsl_proxy(args, config)

    host = getattr(args, "host", None) or config["default_host"]
    port = getattr(args, "port", None) or config["default_port"]

    tabs = get_tabs(host, port)
    if tabs is None:
        print("Error: Cannot connect to Chrome CDP", file=sys.stderr)
        return 1
    print(json.dumps(tabs, indent=2))
    return 0


def cmd_launch(args):
    """Launch Chrome with remote debugging."""
    config = load_config()

    # WSL proxy: delegate to Windows
    if config.get("wsl_proxy") and not getattr(args, "_internal", False):
        return run_via_wsl_proxy(args, config)

    host = getattr(args, "host", None) or config["default_host"]
    port = getattr(args, "port", None) or config["default_port"]
    profile = getattr(args, "profile", None) or config.get("chrome_profile")
    chrome = getattr(args, "chrome", None) or config.get("chrome_binary") or find_chrome_binary()

    if not profile:
        profile = os.path.join(tempfile.gettempdir(), "chrome_cdp_profile")
        print(f"No profile configured. Using temporary: {profile}")

    if not chrome:
        print("Error: Chrome not found. Run: python3 cdp_connector.py setup", file=sys.stderr)
        return 1

    os.makedirs(profile, exist_ok=True)

    flags = [
        f"--remote-debugging-port={port}",
        "--no-first-run",
        "--disable-default-apps",
        f"--user-data-dir={profile}",
    ]

    if getattr(args, "headless", False) or config.get("headless"):
        flags.append("--headless=new")

    if getattr(args, "in_wsl", False) or is_wsl():
        flags.append("--remote-debugging-address=0.0.0.0")

    cmd = [chrome] + flags
    print(f"Launching Chrome on port {port}...")
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for i in range(10):
        time.sleep(1)
        if get_tabs(host, port) is not None:
            print(json.dumps({"launched": True, "port": port, "profile": profile}))
            return 0

    print("Error: Chrome launched but CDP not reachable", file=sys.stderr)
    return 1


def cmd_kill(args):
    """Kill all Chrome processes."""
    config = load_config()
    if config.get("wsl_proxy") and not getattr(args, "_internal", False):
        return run_via_wsl_proxy(args, config)

    system = platform.system()
    if system == "Windows":
        subprocess.run(["taskkill", "/F", "/IM", "chrome.exe"], capture_output=True)
    elif system == "Darwin":
        subprocess.run(["pkill", "-f", "Google Chrome"], capture_output=True)
    else:
        subprocess.run(["pkill", "-f", "chrome"], capture_output=True)

    print(json.dumps({"killed": True}))
    return 0


def cmd_navigate(args):
    """Navigate to a URL."""
    config = load_config()
    if config.get("wsl_proxy") and not getattr(args, "_internal", False):
        return run_via_wsl_proxy(args, config)

    host = getattr(args, "host", None) or config["default_host"]
    port = getattr(args, "port", None) or config["default_port"]

    tab_id = args.tab or get_first_tab_id(host, port)
    if not tab_id:
        print("Error: No tabs available", file=sys.stderr)
        return 1

    result = run_cdp(host, port, tab_id, "Page.navigate", {"url": args.url})
    if result and "error" not in result:
        if args.wait:
            time.sleep(args.wait)
        print(json.dumps({"navigated": True, "url": args.url, "tab": tab_id}))
        return 0
    else:
        print(json.dumps({"error": result.get("error", "Navigation failed")}))
        return 1


def cmd_open(args):
    """Open URL in a new tab."""
    config = load_config()
    if config.get("wsl_proxy") and not getattr(args, "_internal", False):
        return run_via_wsl_proxy(args, config)

    host = getattr(args, "host", None) or config["default_host"]
    port = getattr(args, "port", None) or config["default_port"]

    result = http_get(host, port, f"/json/new?{args.url}")
    if result:
        print(json.dumps(result, indent=2))
        return 0
    else:
        print("Error: Failed to open new tab", file=sys.stderr)
        return 1


def cmd_screenshot(args):
    """Take a screenshot."""
    config = load_config()
    if config.get("wsl_proxy") and not getattr(args, "_internal", False):
        return run_via_wsl_proxy(args, config)

    host = getattr(args, "host", None) or config["default_host"]
    port = getattr(args, "port", None) or config["default_port"]

    tab_id = args.tab or get_first_tab_id(host, port)
    if not tab_id:
        print("Error: No tabs available", file=sys.stderr)
        return 1

    if args.wait:
        time.sleep(args.wait)

    params = {"format": "png"}
    if args.full:
        metrics = run_cdp(host, port, tab_id, "Page.getLayoutMetrics")
        if metrics and "result" in metrics:
            content = metrics["result"].get("contentSize", {})
            params["clip"] = {
                "x": 0, "y": 0,
                "width": content.get("width", 1920),
                "height": content.get("height", 1080),
                "scale": 1,
            }
            params["captureBeyondViewport"] = True

    result = run_cdp(host, port, tab_id, "Page.captureScreenshot", params)
    if result and "result" in result:
        img_data = base64.b64decode(result["result"]["data"])
        output_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(img_data)
        print(json.dumps({"screenshot": True, "path": output_path, "size": len(img_data)}))
        return 0
    else:
        print(json.dumps({"error": result.get("error", "Screenshot failed")}))
        return 1


def cmd_click(args):
    """Click an element by CSS selector."""
    config = load_config()
    if config.get("wsl_proxy") and not getattr(args, "_internal", False):
        return run_via_wsl_proxy(args, config)

    host = getattr(args, "host", None) or config["default_host"]
    port = getattr(args, "port", None) or config["default_port"]

    tab_id = args.tab or get_first_tab_id(host, port)
    if not tab_id:
        print("Error: No tabs available", file=sys.stderr)
        return 1

    safe_selector = args.selector.replace("'", "\\'").replace('"', '\\"')
    js = f"""
    (() => {{
        const el = document.querySelector("{safe_selector}");
        if (!el) return {{error: 'Element not found'}};
        el.click();
        return {{clicked: true, tag: el.tagName}};
    }})()
    """
    result = run_cdp(host, port, tab_id, "Runtime.evaluate", {
        "expression": js,
        "returnByValue": True,
    })
    if result and "result" in result:
        print(json.dumps(result["result"].get("value", {}), indent=2))
        return 0
    else:
        print(json.dumps({"error": result.get("error", "Click failed")}))
        return 1


def cmd_type(args):
    """Type text into an element."""
    config = load_config()
    if config.get("wsl_proxy") and not getattr(args, "_internal", False):
        return run_via_wsl_proxy(args, config)

    host = getattr(args, "host", None) or config["default_host"]
    port = getattr(args, "port", None) or config["default_port"]

    tab_id = args.tab or get_first_tab_id(host, port)
    if not tab_id:
        print("Error: No tabs available", file=sys.stderr)
        return 1

    safe_selector = args.selector.replace("'", "\\'").replace('"', '\\"')
    clear_js = ".value = '';" if args.clear else ""

    js = f"""
    (() => {{
        const el = document.querySelector("{safe_selector}");
        if (!el) return {{error: 'Element not found'}};
        el.focus();
        {clear_js}
        return {{focused: true}};
    }})()
    """
    run_cdp(host, port, tab_id, "Runtime.evaluate", {"expression": js, "returnByValue": True})

    for char in args.text:
        run_cdp(host, port, tab_id, "Input.dispatchKeyEvent", {"type": "keyDown", "text": char, "key": char})
        run_cdp(host, port, tab_id, "Input.dispatchKeyEvent", {"type": "keyUp", "key": char})

    print(json.dumps({"typed": True, "selector": args.selector, "length": len(args.text)}))
    return 0


def cmd_press(args):
    """Press a keyboard key."""
    config = load_config()
    if config.get("wsl_proxy") and not getattr(args, "_internal", False):
        return run_via_wsl_proxy(args, config)

    host = getattr(args, "host", None) or config["default_host"]
    port = getattr(args, "port", None) or config["default_port"]

    tab_id = args.tab or get_first_tab_id(host, port)
    if not tab_id:
        print("Error: No tabs available", file=sys.stderr)
        return 1

    key_map = {
        "Enter": ("\r", "Enter"), "Tab": ("\t", "Tab"), "Escape": ("", "Escape"),
        "Backspace": ("", "Backspace"), "Delete": ("", "Delete"),
        "ArrowUp": ("", "ArrowUp"), "ArrowDown": ("", "ArrowDown"),
        "ArrowLeft": ("", "ArrowLeft"), "ArrowRight": ("", "ArrowRight"),
    }
    text, key = key_map.get(args.key, (args.key, args.key))

    params = {"type": "keyDown", "key": key}
    if text:
        params["text"] = text
    run_cdp(host, port, tab_id, "Input.dispatchKeyEvent", params)
    params["type"] = "keyUp"
    run_cdp(host, port, tab_id, "Input.dispatchKeyEvent", params)

    print(json.dumps({"pressed": args.key}))
    return 0


def cmd_eval(args):
    """Evaluate JavaScript."""
    config = load_config()
    if config.get("wsl_proxy") and not getattr(args, "_internal", False):
        return run_via_wsl_proxy(args, config)

    host = getattr(args, "host", None) or config["default_host"]
    port = getattr(args, "port", None) or config["default_port"]

    tab_id = args.tab or get_first_tab_id(host, port)
    if not tab_id:
        print("Error: No tabs available", file=sys.stderr)
        return 1

    result = run_cdp(host, port, tab_id, "Runtime.evaluate", {
        "expression": args.js_code,
        "returnByValue": True,
        "awaitPromise": True,
    })
    if result and "result" in result:
        value = result["result"].get("result", {}).get("value")
        print(json.dumps(value, indent=2) if value is not None else "undefined")
        return 0
    else:
        print(json.dumps({"error": result.get("error", "Eval failed")}))
        return 1


def cmd_cookies(args):
    """Get all cookies."""
    config = load_config()
    if config.get("wsl_proxy") and not getattr(args, "_internal", False):
        return run_via_wsl_proxy(args, config)

    host = getattr(args, "host", None) or config["default_host"]
    port = getattr(args, "port", None) or config["default_port"]

    tab_id = get_first_tab_id(host, port)
    if not tab_id:
        print("Error: No tabs available", file=sys.stderr)
        return 1

    result = run_cdp(host, port, tab_id, "Network.getAllCookies")
    if result and "result" in result:
        cookies = result["result"].get("cookies", [])
        print(json.dumps(cookies, indent=2))
        return 0
    else:
        print(json.dumps({"error": result.get("error", "Failed to get cookies")}))
        return 1


def cmd_set_cookie(args):
    """Set a cookie."""
    config = load_config()
    if config.get("wsl_proxy") and not getattr(args, "_internal", False):
        return run_via_wsl_proxy(args, config)

    host = getattr(args, "host", None) or config["default_host"]
    port = getattr(args, "port", None) or config["default_port"]

    tab_id = get_first_tab_id(host, port)
    if not tab_id:
        print("Error: No tabs available", file=sys.stderr)
        return 1

    params = {"name": args.name, "value": args.value}
    if args.domain:
        params["domain"] = args.domain
    if args.path:
        params["path"] = args.path

    result = run_cdp(host, port, tab_id, "Network.setCookie", params)
    if result and "result" in result:
        print(json.dumps(result["result"], indent=2))
        return 0
    else:
        print(json.dumps({"error": result.get("error", "Failed to set cookie")}))
        return 1


def cmd_close_tab(args):
    """Close a tab."""
    config = load_config()
    if config.get("wsl_proxy") and not getattr(args, "_internal", False):
        return run_via_wsl_proxy(args, config)

    host = getattr(args, "host", None) or config["default_host"]
    port = getattr(args, "port", None) or config["default_port"]

    result = http_get(host, port, f"/json/close/{args.tab_id}")
    if result:
        print(json.dumps({"closed": True, "tab": args.tab_id}))
        return 0
    else:
        print("Error: Failed to close tab", file=sys.stderr)
        return 1


# ─── Main ─────────────────────────────────────────────────────────────────────

def cmd_check(args):
    """Check system compatibility."""
    ok, issues, warnings = check_compatibility()
    result = {"compatible": ok, "issues": issues, "warnings": warnings}
    print(json.dumps(result, indent=2))
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(
        description="Chrome CDP Connector - Browser automation via Chrome DevTools Protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WSL2 Usage:
  When running in WSL2, Chrome is on the Windows side. Use --wsl-proxy
  to automatically delegate commands through Windows Python.

  First time: python3 cdp_connector.py setup
  Then just:  python3 cdp_connector.py navigate https://x.com --wsl-proxy
        """,
    )

    sub = parser.add_subparsers(dest="command", help="Command to run")

    # Common args for all commands
    def add_common_args(p):
        p.add_argument("--host", help="Chrome host (overrides config)")
        p.add_argument("--port", type=int, help="Chrome port (overrides config)")
        p.add_argument("--wsl-proxy", action="store_true", help="Delegate to Windows py.exe (WSL2 mode)")

    # setup
    p_setup = sub.add_parser("setup", help="Interactive setup wizard")
    p_setup.add_argument("--non-interactive", action="store_true", help="Use defaults")

    # config
    sub.add_parser("config", help="Show current configuration")

    # status
    p_status = sub.add_parser("status", help="Check CDP connection")
    add_common_args(p_status)

    # tabs
    p_tabs = sub.add_parser("tabs", help="List open tabs")
    add_common_args(p_tabs)

    # launch
    p_launch = sub.add_parser("launch", help="Launch Chrome with debugging")
    add_common_args(p_launch)
    p_launch.add_argument("--profile", help="Chrome profile path (overrides config)")
    p_launch.add_argument("--chrome", help="Chrome binary path (overrides config)")
    p_launch.add_argument("--in-wsl", action="store_true", help="WSL-friendly flags")
    p_launch.add_argument("--headless", action="store_true", help="Headless mode")

    # kill
    p_kill = sub.add_parser("kill", help="Kill all Chrome processes")
    add_common_args(p_kill)

    # navigate
    p_nav = sub.add_parser("navigate", help="Navigate to URL")
    add_common_args(p_nav)
    p_nav.add_argument("url", help="URL to navigate to")
    p_nav.add_argument("--tab", help="Tab ID (default: first tab)")
    p_nav.add_argument("--wait", type=float, default=5, help="Wait seconds after nav")

    # open
    p_open = sub.add_parser("open", help="Open URL in new tab")
    add_common_args(p_open)
    p_open.add_argument("url", help="URL to open")

    # screenshot
    p_ss = sub.add_parser("screenshot", help="Take screenshot")
    add_common_args(p_ss)
    p_ss.add_argument("output", help="Output file path")
    p_ss.add_argument("--tab", help="Tab ID")
    p_ss.add_argument("--full", action="store_true", help="Full page")
    p_ss.add_argument("--wait", type=float, help="Wait before capture")

    # click
    p_click = sub.add_parser("click", help="Click element by CSS selector")
    add_common_args(p_click)
    p_click.add_argument("selector", help="CSS selector")
    p_click.add_argument("--tab", help="Tab ID")

    # type
    p_type = sub.add_parser("type", help="Type text into element")
    add_common_args(p_type)
    p_type.add_argument("selector", help="CSS selector")
    p_type.add_argument("text", help="Text to type")
    p_type.add_argument("--tab", help="Tab ID")
    p_type.add_argument("--clear", action="store_true", help="Clear field first")

    # press
    p_press = sub.add_parser("press", help="Press keyboard key")
    add_common_args(p_press)
    p_press.add_argument("key", help="Key name")
    p_press.add_argument("--tab", help="Tab ID")

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate JavaScript")
    add_common_args(p_eval)
    p_eval.add_argument("js_code", help="JavaScript code")
    p_eval.add_argument("--tab", help="Tab ID")

    # cookies
    p_cookies = sub.add_parser("cookies", help="Get all cookies")
    add_common_args(p_cookies)

    # set-cookie
    p_set_cookie = sub.add_parser("set-cookie", help="Set a cookie")
    add_common_args(p_set_cookie)
    p_set_cookie.add_argument("name", help="Cookie name")
    p_set_cookie.add_argument("value", help="Cookie value")
    p_set_cookie.add_argument("--domain", help="Cookie domain")
    p_set_cookie.add_argument("--path", default="/", help="Cookie path")

    # check
    sub.add_parser("check", help="Check system compatibility")

    # close-tab
    p_close = sub.add_parser("close-tab", help="Close a tab")
    add_common_args(p_close)
    p_close.add_argument("tab_id", help="Tab ID to close")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        "setup": cmd_setup,
        "config": cmd_config,
        "check": cmd_check,
        "status": cmd_status,
        "tabs": cmd_tabs,
        "launch": cmd_launch,
        "kill": cmd_kill,
        "navigate": cmd_navigate,
        "open": cmd_open,
        "screenshot": cmd_screenshot,
        "click": cmd_click,
        "type": cmd_type,
        "press": cmd_press,
        "eval": cmd_eval,
        "cookies": cmd_cookies,
        "set-cookie": cmd_set_cookie,
        "close-tab": cmd_close_tab,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
