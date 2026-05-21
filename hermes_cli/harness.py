import os
import subprocess
import time
import logging
import platform
from pathlib import Path

from hermes_cli.config import load_config

logger = logging.getLogger(__name__)

_FALSE_ENV_VALUES = {"0", "false", "no", "off", "disabled"}


def _env_flag_disabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _FALSE_ENV_VALUES


def _auto_start_disabled_by_env() -> bool:
    return _env_flag_disabled("HYPURA_HARNESS_AUTO_START") or _env_flag_disabled(
        "HERMES_HARNESS_AUTO_START"
    )

def get_harness_url() -> str:
    """Get the local harness URL from config or env."""
    config = load_config()
    harness_cfg = config.get("harness", {})
    if not isinstance(harness_cfg, dict):
        harness_cfg = {}
    host = os.getenv("HYPURA_HARNESS_HOST") or harness_cfg.get("host", "127.0.0.1")
    port = os.getenv("HYPURA_HARNESS_PORT") or harness_cfg.get("port", 18794)
    return f"http://{host}:{port}"

def is_harness_running() -> bool:
    """Check if the harness is reachable."""
    from tools.openclaw.harness_client import is_harness_running as _probe

    return _probe()

def start_harness_daemon() -> bool:
    """Launch the Hypura Harness daemon in the background."""
    if is_harness_running():
        logger.info("Hypura Harness is already running.")
        return True

    project_root = Path(__file__).resolve().parents[1]
    harness_script = project_root / "vendor" / "openclaw-mirror" / "extensions" / "hypura-harness" / "scripts" / "harness_daemon.py"
    harness_cwd = harness_script.parent

    if not harness_script.exists():
        logger.error(f"Harness script not found at {harness_script}")
        return False

    config = load_config()
    port = os.getenv("HYPURA_HARNESS_PORT") or config.get("harness", {}).get("port", 18794)
    
    # Select python executor
    pyproject = harness_cwd / "pyproject.toml"
    if pyproject.exists():
        # Prefer uv run if pyproject.toml is present for dependency management
        cmd = ["uv", "run", "python", str(harness_script)]
    else:
        python_exe = "py" if platform.system() == "Windows" else "python3"
        cmd = [python_exe, "-3", str(harness_script)]
    
    # Ensure PYTHONPATH includes the scripts dir for local imports
    env = os.environ.copy()
    env["PYTHONPATH"] = str(harness_cwd) + os.pathsep + env.get("PYTHONPATH", "")
    
    try:
        # Launch as a detached process
        if platform.system() == "Windows":
            subprocess.Popen(
                cmd,
                cwd=str(harness_cwd),
                env=env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.Popen(
                cmd,
                cwd=str(harness_cwd),
                env=env,
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        
        # Wait for it to spin up
        for _ in range(30):
            time.sleep(1.0)
            if is_harness_running():
                logger.info(f"Hypura Harness started on port {port}")
                return True
        
        logger.warning("Harness daemon started but not responding yet.")
        return False
    except Exception as e:
        logger.error(f"Failed to launch harness: {e}")
        return False

def stop_harness_daemon() -> bool:
    """Stop the harness daemon by killing the process on its port."""
    url = get_harness_url()
    try:
        # Try graceful shutdown if implemented (it's not yet in harness_daemon.py)
        # For now, we use process killing
        import psutil
        config = load_config()
        port = int(os.getenv("HYPURA_HARNESS_PORT") or config.get("harness", {}).get("port", 18794))
        
        found = False
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                for conn in proc.connections(kind='inet'):
                    if conn.laddr.port == port:
                        logger.info(f"Stopping harness process {proc.pid}")
                        proc.terminate()
                        proc.wait(timeout=3)
                        found = True
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                continue
        
        return not is_harness_running()
    except Exception as e:
        logger.error(f"Error stopping harness: {e}")
        return False

def ensure_harness_running():
    """Higher-level check/start for auto-invocation."""
    if _auto_start_disabled_by_env():
        logger.debug("Hypura Harness auto-start disabled by environment.")
        return

    config = load_config()
    harness_cfg = config.get("harness", {})
    if not isinstance(harness_cfg, dict):
        harness_cfg = {}
    if not harness_cfg.get("enabled", True):
        return
    
    if harness_cfg.get("auto_start", True):
        if not is_harness_running():
            logger.info("Auto-starting Hypura Harness...")
            start_harness_daemon()
