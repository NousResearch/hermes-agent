"""Docker backend diagnostics for ``hermes doctor``.

The terminal Docker backend fails early when the Docker CLI is installed but the
Docker daemon is unavailable. These helpers make that failure visible in doctor
without making doctor itself depend on Docker.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _read_yaml_config(hermes_home: Path) -> dict[str, Any]:
    """Best-effort read of ``config.yaml``/``config.yml`` from HERMES_HOME."""
    for name in ("config.yaml", "config.yml"):
        path = hermes_home / name
        if not path.exists():
            continue
        try:
            import yaml

            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _configured_terminal_backend(*, hermes_home: Path | None = None) -> str:
    """Infer the terminal backend from env/config, defaulting to local."""
    for env_name in ("TERMINAL_ENV", "HERMES_TERMINAL_BACKEND"):
        value = os.getenv(env_name)
        if value:
            return value.strip().lower()

    config = _read_yaml_config(hermes_home or Path.home() / ".hermes")
    terminal = config.get("terminal")
    if isinstance(terminal, dict):
        value = terminal.get("backend") or terminal.get("env")
        if value:
            return str(value).strip().lower()
    value = config.get("terminal_backend") or config.get("terminal_env")
    return str(value).strip().lower() if value else "local"


def _docker_backend_diagnostic(doctor_module: Any) -> dict[str, Any] | None:
    """Return a diagnostic when Docker backend is configured but unhealthy.

    The Docker daemon probe is intentionally skipped unless the configured
    terminal backend is Docker *and* the Docker CLI is present.
    """
    hermes_home = Path(getattr(doctor_module, "HERMES_HOME", Path.home() / ".hermes"))
    backend = _configured_terminal_backend(hermes_home=hermes_home)
    if backend != "docker":
        return None

    docker_path = doctor_module.shutil.which("docker")
    if not docker_path:
        return None

    try:
        result = doctor_module.subprocess.run(
            ["docker", "version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        return {
            "ok": False,
            "summary": "Docker backend is configured but Docker daemon is not responding",
            "detail": str(exc),
            "remediation": [
                "Start Docker Desktop",
                "On Linux, run: sudo systemctl start docker",
            ],
        }

    if getattr(result, "returncode", 1) == 0:
        return None

    detail = "\n".join(
        part.strip()
        for part in (getattr(result, "stderr", ""), getattr(result, "stdout", ""))
        if part and part.strip()
    )
    return {
        "ok": False,
        "summary": "Docker backend is configured but Docker daemon is not responding",
        "detail": detail or "docker version returned a non-zero exit status",
        "remediation": [
            "Start Docker Desktop",
            "On Linux, run: sudo systemctl start docker",
        ],
    }


def _emit_docker_backend_diagnostic(doctor_module: Any, diagnostic: dict[str, Any]) -> None:
    """Print a Docker backend diagnostic using doctor helpers when available."""
    summary = diagnostic["summary"]
    try:
        doctor_module.check_error("Docker backend", summary)
    except Exception:
        print(f"✗ Docker backend: {summary}")

    detail = diagnostic.get("detail")
    if detail:
        for line in str(detail).splitlines():
            if line.strip():
                try:
                    doctor_module.check_info("Docker daemon", line.strip())
                except Exception:
                    print(f"  {line.strip()}")

    for item in diagnostic.get("remediation", []):
        try:
            doctor_module.check_info("Docker remediation", item)
        except Exception:
            print(f"  {item}")


def patch_doctor_module(doctor_module: Any) -> None:
    """Install Docker backend diagnostics into ``hermes_cli.doctor``."""
    if getattr(doctor_module, "_docker_backend_diagnostics_patched", False):
        return

    def diagnostic_wrapper() -> dict[str, Any] | None:
        return _docker_backend_diagnostic(doctor_module)

    original_run_doctor = doctor_module.run_doctor

    def run_doctor_with_docker_backend_diagnostics(*args, **kwargs):
        diagnostic = diagnostic_wrapper()
        if diagnostic:
            _emit_docker_backend_diagnostic(doctor_module, diagnostic)
        return original_run_doctor(*args, **kwargs)

    doctor_module._docker_backend_diagnostic = diagnostic_wrapper
    doctor_module.run_doctor = run_doctor_with_docker_backend_diagnostics
    doctor_module._docker_backend_diagnostics_patched = True
