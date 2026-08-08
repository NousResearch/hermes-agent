"""Memory-provider setup helpers (extracted verbatim from web_server.py).

Cluster c16 (memory_provider_setup) of the s2 shard plan: provider
discovery/load, manifest parsing, dependency checks, and the pip/external
setup command pipeline.  Bodies are byte-identical to their previous
in-web_server form.

Cross-module helpers that tests monkeypatch on ``web_server`` are reached
through the late-binding seam in :mod:`hermes_cli.web_deps`, so
``monkeypatch.setattr(web_server, ...)`` keeps working (see web_deps).
"""

import logging
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import HTTPException

from hermes_cli.web_deps import late

# Late-bound web_server helpers (resolved at call time; cycle-safe,
# monkeypatch-transparent) - same seam as hermes_cli/web_routers/cron.py.
_discover_memory_provider_statuses = late("_discover_memory_provider_statuses")

_log = logging.getLogger("hermes_cli.web_server")


def _memory_provider_label(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").title()


def _normalize_memory_provider_name(name: Any) -> str:
    provider = str(name or "").strip()
    if provider.lower() in {"built-in", "builtin", "none"}:
        return ""
    return provider


def _load_memory_provider(name: str):
    try:
        from plugins.memory import load_memory_provider

        return load_memory_provider(name)
    except Exception:
        _log.debug("Failed to load memory provider %s", name, exc_info=True)
        return None


def _memory_provider_manifest(name: str) -> Dict[str, Any]:
    try:
        from plugins.memory import find_provider_dir

        provider_dir = find_provider_dir(name)
        if provider_dir is None:
            return {}
        manifest_path = provider_dir / "plugin.yaml"
        if not manifest_path.exists():
            return {}
        with manifest_path.open(encoding="utf-8-sig") as handle:
            manifest = yaml.safe_load(handle) or {}
        return manifest if isinstance(manifest, dict) else {}
    except Exception:
        _log.debug("Failed to read memory provider manifest for %s", name, exc_info=True)
        return {}


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _memory_provider_setup_manifest(name: str) -> Dict[str, Any]:
    manifest = _memory_provider_manifest(name)
    external_dependencies: List[Dict[str, str]] = []
    for raw in manifest.get("external_dependencies") or []:
        if not isinstance(raw, dict):
            continue
        dep = {
            "name": str(raw.get("name") or "").strip(),
            "install": str(raw.get("install") or "").strip(),
            "check": str(raw.get("check") or "").strip(),
        }
        if dep["name"] or dep["install"] or dep["check"]:
            external_dependencies.append(dep)

    return {
        "pip_dependencies": _string_list(manifest.get("pip_dependencies")),
        "external_dependencies": external_dependencies,
        "required_env": _string_list(manifest.get("requires_env")),
    }


def _memory_provider_setup_info(name: str) -> Dict[str, Any]:
    setup = _memory_provider_setup_manifest(name)
    setup["dependencies_installed"] = _memory_provider_dependencies_installed(setup)
    return setup


_MEMORY_PROVIDER_IMPORT_NAMES = {
    "honcho-ai": "honcho",
    "mem0ai": "mem0",
    "hindsight-client": "hindsight_client",
    "hindsight-all": "hindsight",
}


def _memory_provider_dependency_package(dep: str) -> str:
    return re.split(r"[\[<>=!~;]", dep, maxsplit=1)[0].strip()


def _memory_provider_import_name(dep: str) -> str:
    package = _memory_provider_dependency_package(dep)
    return _MEMORY_PROVIDER_IMPORT_NAMES.get(package, package.replace("-", "_"))


def _dependency_importable(dep: str) -> bool:
    import_name = _memory_provider_import_name(dep)
    if not import_name:
        return False
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def _trim_setup_output(value: Optional[str], limit: int = 4000) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... truncated ..."


def _memory_provider_setup_env() -> Dict[str, str]:
    # External package-manager child (npm/uv/pip): exact env preservation —
    # scrubbing or HOME rewriting could break user tool auth/config.
    from tools.environments.local import build_subprocess_env
    env = build_subprocess_env(scrub_secrets=False, inherit_profile_home=False)
    home = Path.home()
    extra_bins = [
        home / ".brv-cli" / "bin",
        home / ".local" / "bin",
        home / ".npm-global" / "bin",
        Path("/usr/local/bin"),
    ]
    existing_path = env.get("PATH", "")
    prefix = os.pathsep.join(str(path) for path in extra_bins if path.exists())
    if prefix:
        env["PATH"] = prefix + os.pathsep + existing_path
    return env


def _command_result(
    *,
    kind: str,
    name: str,
    status: str,
    command: str = "",
    completed: Optional[subprocess.CompletedProcess] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "kind": kind,
        "name": name,
        "status": status,
        "command": command,
        "returncode": None if completed is None else completed.returncode,
        "stdout": "" if completed is None else _trim_setup_output(completed.stdout),
        "stderr": _trim_setup_output(error or ("" if completed is None else completed.stderr)),
    }


def _run_setup_command(
    command: Any,
    *,
    display: str,
    shell: bool = False,
    timeout: int = 180,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        shell=shell,
        executable="/bin/bash" if shell else None,
        env=_memory_provider_setup_env(),
        capture_output=True,
        text=True,
        # Lossy UTF-8 decode — setup tools emit UTF-8; never let a
        # locale-mismatched byte raise in the reader thread (#52649).
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _memory_provider_dependencies_installed(setup: Dict[str, Any]) -> bool:
    pip_dependencies = _string_list(setup.get("pip_dependencies"))
    external_dependencies = setup.get("external_dependencies") or []

    pip_ok = all(_dependency_importable(dep) for dep in pip_dependencies)
    external_ok = True
    for dep in external_dependencies:
        if not isinstance(dep, dict):
            continue
        check_cmd = str(dep.get("check") or "").strip()
        install_cmd = str(dep.get("install") or "").strip()
        if not check_cmd:
            if install_cmd:
                external_ok = False
            continue
        try:
            completed = _run_setup_command(
                shlex.split(check_cmd),
                display=check_cmd,
                timeout=20,
            )
        except Exception:
            external_ok = False
            continue
        if completed.returncode != 0:
            external_ok = False

    return pip_ok and external_ok


def _install_memory_provider_pip_dependencies(dependencies: List[str]) -> List[Dict[str, Any]]:
    missing = [dep for dep in dependencies if not _dependency_importable(dep)]
    if not dependencies:
        return []
    if not missing:
        return [
            _command_result(kind="pip", name=", ".join(dependencies), status="already_installed")
        ]

    # Route through the lazy-install pipeline (tools.lazy_deps.install_specs)
    # instead of shelling out to pip against sys.executable directly. That
    # pipeline is environment-aware: on hosted/immutable images the agent venv
    # under /opt/hermes is sealed read-only, and installs must be redirected
    # to the writable durable target on the data volume
    # (HERMES_LAZY_INSTALL_TARGET, e.g. /opt/data/lazy-packages) — the same
    # path every lazy backend already uses. A direct `pip install --python
    # sys.executable` on those images fails with a permission error (NS-605).
    # install_specs also activates the target on sys.path post-install so the
    # availability recheck below sees the new packages without a restart.
    try:
        from tools.lazy_deps import install_specs

        outcome = install_specs(missing, timeout=240)
    except Exception as exc:
        return [
            _command_result(
                kind="pip",
                name=", ".join(missing),
                status="failed",
                error=str(exc),
            )
        ]

    if outcome.blocked:
        return [
            _command_result(
                kind="pip",
                name=", ".join(missing),
                status="failed",
                command=outcome.command,
                error=outcome.reason,
            )
        ]

    return [
        _command_result(
            kind="pip",
            name=", ".join(missing),
            status="installed" if outcome.ok else "failed",
            command=outcome.command,
            completed=subprocess.CompletedProcess(
                args=outcome.command,
                returncode=0 if outcome.ok else 1,
                stdout=outcome.stdout,
                stderr=outcome.stderr,
            ),
        )
    ]


def _install_memory_provider_external_dependencies(
    dependencies: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for dep in dependencies:
        name = dep.get("name") or "dependency"
        check_cmd = dep.get("check") or ""
        install_cmd = dep.get("install") or ""

        if check_cmd:
            try:
                check = _run_setup_command(
                    shlex.split(check_cmd),
                    display=check_cmd,
                    timeout=20,
                )
            except Exception as exc:
                results.append(
                    _command_result(
                        kind="external_check",
                        name=name,
                        status="missing" if install_cmd else "failed",
                        command=check_cmd,
                        error=str(exc),
                    )
                )
            else:
                if check.returncode == 0:
                    results.append(
                        _command_result(
                            kind="external_check",
                            name=name,
                            status="already_installed",
                            command=check_cmd,
                            completed=check,
                        )
                    )
                    continue
                results.append(
                    _command_result(
                        kind="external_check",
                        name=name,
                        status="missing" if install_cmd else "failed",
                        command=check_cmd,
                        completed=check,
                    )
                )

            if not install_cmd:
                continue

        if install_cmd:
            try:
                install = _run_setup_command(
                    install_cmd,
                    display=install_cmd,
                    shell=True,
                    timeout=300,
                )
            except Exception as exc:
                results.append(
                    _command_result(
                        kind="external_install",
                        name=name,
                        status="failed",
                        command=install_cmd,
                        error=str(exc),
                    )
                )
                continue

            results.append(
                _command_result(
                    kind="external_install",
                    name=name,
                    status="installed" if install.returncode == 0 else "failed",
                    command=install_cmd,
                    completed=install,
                )
            )

            if check_cmd and install.returncode == 0:
                try:
                    post_check = _run_setup_command(
                        shlex.split(check_cmd),
                        display=check_cmd,
                        timeout=20,
                    )
                    results.append(
                        _command_result(
                            kind="external_check",
                            name=name,
                            status="verified" if post_check.returncode == 0 else "failed",
                            command=check_cmd,
                            completed=post_check,
                        )
                    )
                except Exception as exc:
                    results.append(
                        _command_result(
                            kind="external_check",
                            name=name,
                            status="failed",
                            command=check_cmd,
                            error=str(exc),
                        )
                    )

    return results


def _install_memory_provider_setup(name: str) -> Dict[str, Any]:
    provider = _load_memory_provider(name)
    manifest = _memory_provider_manifest(name)
    if provider is None and not manifest:
        raise HTTPException(status_code=404, detail=f"Unknown memory provider: {name}")

    setup = _memory_provider_setup_manifest(name)
    results = []
    results.extend(_install_memory_provider_pip_dependencies(setup["pip_dependencies"]))
    results.extend(
        _install_memory_provider_external_dependencies(setup["external_dependencies"])
    )

    if not results:
        results.append(
            _command_result(
                kind="setup",
                name=name,
                status="no_declared_steps",
            )
        )

    ok = all(result["status"] not in {"failed"} for result in results)
    statuses = {row["name"]: row for row in _discover_memory_provider_statuses()}
    return {
        "ok": ok,
        "provider": name,
        "results": results,
        "status": statuses.get(name),
    }


# --- Monkeypatch-transparent rebinding ------------------------------------
# ``tests/hermes_cli/test_web_server.py`` patches ``web_server._dependency_importable``
# and expects in-module callers (pip/external dependency install paths) to see
# the patch, exactly as they did in the pre-extraction single-module layout.
# The original def is kept for web_server's legacy re-export; the public name
# is rebound to a late proxy so every in-module call re-reads the live
# attribute on web_server.
_dependency_importable_impl = _dependency_importable
_dependency_importable = late("_dependency_importable")
