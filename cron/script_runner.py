"""No-agent cron script execution helpers."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

from hermes_cli._subprocess_compat import windows_hide_flags


logger = logging.getLogger(__name__)

DEFAULT_SCRIPT_TIMEOUT = 3600  # seconds (1 hour)


def get_script_timeout(
    *,
    timeout_override: int | float | str | None = None,
    load_config_fn: Callable[[], dict] | None = None,
) -> int:
    """Resolve cron pre-run script timeout from override/env/config/default."""
    if timeout_override not in (None, DEFAULT_SCRIPT_TIMEOUT):
        try:
            timeout = int(float(timeout_override))
            if timeout > 0:
                return timeout
        except Exception:
            logger.warning("Invalid patched _SCRIPT_TIMEOUT=%r; using env/config/default", timeout_override)

    env_value = os.getenv("HERMES_CRON_SCRIPT_TIMEOUT", "").strip()
    if env_value:
        try:
            timeout = int(float(env_value))
            if timeout > 0:
                return timeout
        except Exception:
            logger.warning("Invalid HERMES_CRON_SCRIPT_TIMEOUT=%r; using config/default", env_value)

    if load_config_fn is not None:
        try:
            cfg = load_config_fn() or {}
            cron_cfg = cfg.get("cron", {}) if isinstance(cfg, dict) else {}
            configured = cron_cfg.get("script_timeout_seconds")
            if configured is not None:
                timeout = int(float(configured))
                if timeout > 0:
                    return timeout
        except Exception as exc:
            logger.debug("Failed to load cron script timeout from config: %s", exc)

    return DEFAULT_SCRIPT_TIMEOUT


def run_job_script(
    script_path: str,
    *,
    hermes_home: Path,
    timeout_override: int | float | str | None = None,
    load_config_fn: Callable[[], dict] | None = None,
) -> tuple[bool, str]:
    """Execute a cron job script inside HERMES_HOME/scripts and capture output."""
    scripts_dir = hermes_home / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir_resolved = scripts_dir.resolve()

    raw = Path(script_path).expanduser()
    if raw.is_absolute():
        path = raw.resolve()
    else:
        path = (scripts_dir / raw).resolve()

    try:
        path.relative_to(scripts_dir_resolved)
    except ValueError:
        return False, (
            f"Blocked: script path resolves outside the scripts directory "
            f"({scripts_dir_resolved}): {script_path!r}"
        )

    if not path.exists():
        return False, f"Script not found: {path}"
    if not path.is_file():
        return False, f"Script path is not a file: {path}"

    script_timeout = get_script_timeout(
        timeout_override=timeout_override,
        load_config_fn=load_config_fn,
    )

    suffix = path.suffix.lower()
    if suffix in {".sh", ".bash"}:
        bash = shutil.which("bash") or ("/bin/bash" if os.path.isfile("/bin/bash") else None)
        if bash is None:
            return False, (
                f"Cannot run .sh/.bash script {path.name!r}: bash not found on PATH. "
                "On Windows, install Git for Windows (which ships Git Bash) "
                "or rewrite the script as Python (.py)."
            )
        argv = [bash, str(path)]
    else:
        argv = [sys.executable, str(path)]

    try:
        from tools.environments.local import _sanitize_subprocess_env

        popen_kwargs = {"creationflags": windows_hide_flags()} if sys.platform == "win32" else {}
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=script_timeout,
            cwd=str(path.parent),
            env=_sanitize_subprocess_env(os.environ.copy()),
            **popen_kwargs,
        )
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        try:
            from agent.redact import redact_sensitive_text

            stdout = redact_sensitive_text(stdout)
            stderr = redact_sensitive_text(stderr)
        except Exception as exc:
            logger.warning("Failed to redact sensitive text from output: %s", exc)
            stdout = "[REDACTED - redaction failed]"
            stderr = "[REDACTED - redaction failed]"

        if result.returncode != 0:
            parts = [f"Script exited with code {result.returncode}"]
            if stderr:
                parts.append(f"stderr:\n{stderr}")
            if stdout:
                parts.append(f"stdout:\n{stdout}")
            return False, "\n".join(parts)

        return True, stdout

    except subprocess.TimeoutExpired:
        return False, f"Script timed out after {script_timeout}s: {path}"
    except Exception as exc:
        return False, f"Script execution failed: {exc}"
