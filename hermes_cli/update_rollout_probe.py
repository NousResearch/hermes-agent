"""Read-only observation probe for the update rollout protocol.

The probe deliberately reads the raw installation identity instead of calling
``get_install_id``: a health check must never mint identity, create a profile,
or repair a marker.  ``collect_observation`` is the shared observation shape
for the CLI preflight and the downstream TypeScript health adapter.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

from hermes_constants import get_default_hermes_root
from hermes_cli.update_inventory import RuntimeRecord

PROTOCOL_VERSION = 1
_INSTALL_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_CODE_ROOT = Path(__file__).resolve().parent.parent
_MODES = ("inventory", "health")


def _read_raw_install_id(root: Path) -> str | None:
    """Return a valid stored id, or ``None``; never create or rewrite the file."""
    try:
        value = (root / "install_id").read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError, UnicodeError):
        return None
    return value if _INSTALL_ID_RE.fullmatch(value) else None


def _run_git(root: Path, *args: str) -> str | None:
    """Run one local, read-only git query. Network and prompts are disabled."""
    try:
        result = subprocess.run(
            ["git", *args], cwd=str(root), stdin=subprocess.DEVNULL,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=3, check=False,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _read_git_metadata(root: Path) -> dict[str, Any]:
    """Read checkout identity and local remote/tracking metadata without fetching."""
    code_root = _run_git(root, "rev-parse", "--show-toplevel")
    sha = _run_git(root, "rev-parse", "HEAD")
    origin = _run_git(root, "remote", "get-url", "origin")
    tracking = _run_git(root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    status = _run_git(root, "status", "--porcelain")
    checkout_state = "unknown" if status is None else ("clean" if not status else "dirty")
    divergence = "unknown"
    ahead = behind = None
    counts = _run_git(root, "rev-list", "--left-right", "--count", "HEAD...@{u}")
    if counts:
        parts = counts.split()
        if len(parts) == 2 and all(part.isdigit() for part in parts):
            ahead, behind = (int(parts[0]), int(parts[1]))
            divergence = "diverged" if ahead and behind else "ahead" if ahead else "behind" if behind else "in-sync"
    return {
        "codeRoot": code_root,
        "origin": origin,
        "trackingBranch": tracking,
        "checkoutSha": sha if sha and re.fullmatch(r"[0-9a-fA-F]{40}", sha) else None,
        "checkoutState": checkout_state,
        "divergence": divergence,
        "ahead": ahead,
        "behind": behind,
    }


_POSITIVE_INT_RE = re.compile(r"^[1-9][0-9]*$")
_FLOAT_RE = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")
_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def _marker_state(path: Path, *, fleet: bool = False) -> dict[str, Any]:
    """Read and validate a recovery marker emitted by the update producers."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"state": "absent", "path": str(path)}
    except (OSError, UnicodeError) as exc:
        return {"state": "unavailable", "path": str(path), "reason": type(exc).__name__}

    fields_by_name: dict[str, str] = {}
    for line in text.splitlines():
        if not line or "=" not in line:
            return {"state": "malformed", "path": str(path)}
        name, value = line.split("=", 1)
        if name in fields_by_name or not name or not value:
            return {"state": "malformed", "path": str(path)}
        fields_by_name[name] = value
    if set(fields_by_name) < {"started", "pid"}:
        return {"state": "malformed", "path": str(path)}
    if not _FLOAT_RE.fullmatch(fields_by_name["started"]):
        return {"state": "malformed", "path": str(path)}
    if not _POSITIVE_INT_RE.fullmatch(fields_by_name["pid"]):
        return {"state": "malformed", "path": str(path)}
    allowed_fields = {"started", "pid"} | ({"expected_sha", "inventory"} if fleet else set())
    if not set(fields_by_name) <= allowed_fields:
        return {"state": "malformed", "path": str(path)}

    if "expected_sha" in fields_by_name and not _SHA_RE.fullmatch(fields_by_name["expected_sha"]):
        return {"state": "malformed", "path": str(path)}
    if "inventory" in fields_by_name:
        try:
            inventory = json.loads(fields_by_name["inventory"])
        except (ValueError, UnicodeError):
            return {"state": "malformed", "path": str(path)}
        if (
            not isinstance(inventory, dict)
            or set(inventory) != {"version", "runtimes"}
            or type(inventory.get("version")) is not int
            or inventory["version"] != 1
        ):
            return {"state": "malformed", "path": str(path)}
        runtimes = inventory.get("runtimes")
        if not isinstance(runtimes, list) or any(not isinstance(runtime, dict) for runtime in runtimes):
            return {"state": "malformed", "path": str(path)}
        runtime_fields = {field.name for field in fields(RuntimeRecord)}
        if any(set(runtime) != runtime_fields for runtime in runtimes):
            return {"state": "malformed", "path": str(path)}
        for runtime in runtimes:
            if (
                not isinstance(runtime["kind"], str)
                or not isinstance(runtime["profile"], str)
                or (runtime["pid"] is not None and (type(runtime["pid"]) is not int or runtime["pid"] <= 0))
                or not isinstance(runtime["supervisor"], str)
                or (runtime["code_sha"] is not None and not isinstance(runtime["code_sha"], str))
                or (runtime["code_version"] is not None and not isinstance(runtime["code_version"], str))
                or not isinstance(runtime["restart_via"], str)
                or not isinstance(runtime["detail"], dict)
            ):
                return {"state": "malformed", "path": str(path)}

    return {"state": "live", "path": str(path)}


def _read_image_marker() -> dict[str, Any]:
    """Adapt the existing immutable-deployment marker without flattening absence."""
    try:
        from hermes_cli.image_provenance import read_image_provenance
        marker = read_image_provenance()
    except Exception as exc:
        return {"state": "unavailable", "reason": type(exc).__name__}
    if marker is None:
        return {"state": "absent"}
    if not marker.valid:
        return {"state": "malformed", "path": marker.marker_path, "reason": marker.error}
    return {
        "state": "live", "path": marker.marker_path, "deploymentKind": marker.deployment_kind,
        "manager": marker.manager, "version": marker.version, "revision": marker.revision,
    }


def _marker_path(owner: str) -> Path:
    if owner == "updateIncomplete":
        from hermes_cli.main_install_repair import _update_marker_path
        return _update_marker_path()
    if owner == "lazyRefreshIncomplete":
        from hermes_cli.main_install_repair import _lazy_refresh_marker_path
        return _lazy_refresh_marker_path()
    from hermes_cli.update_cmd_fleet import _fleet_restart_pending_marker_path
    return _fleet_restart_pending_marker_path()


def _read_recovery() -> dict[str, Any]:
    markers = {
        "updateIncomplete": _marker_state(_marker_path("updateIncomplete")),
        "lazyRefreshIncomplete": _marker_state(_marker_path("lazyRefreshIncomplete")),
        "fleetRestartPending": _marker_state(_marker_path("fleetRestartPending"), fleet=True),
    }
    live = [name for name, value in markers.items() if value["state"] == "live"]
    malformed = [name for name, value in markers.items() if value["state"] == "malformed"]
    unavailable = [name for name, value in markers.items() if value["state"] == "unavailable"]
    return {
        "state": "live" if live else "malformed" if malformed else "unavailable" if unavailable else "clear",
        "markers": markers,
    }


def _deployment_eligible(image: dict[str, Any]) -> bool:
    return image.get("state") == "absent"


def _read_runtime_evidence() -> dict[str, Any]:
    """Read runtime status snapshots; absence is unknown, not an empty requirement set."""
    root = Path(get_default_hermes_root())
    candidates = [root / "gateway_state.json"]
    profiles = root / "profiles"
    try:
        if profiles.is_dir():
            candidates.extend(path / "gateway_state.json" for path in profiles.iterdir() if path.is_dir())
    except OSError:
        pass
    processes: list[dict[str, Any]] = []
    generations: set[Any] = set()
    required_scopes: set[str] = set()
    required_scopes_known = False
    try:
        from gateway.status import read_runtime_status
    except Exception:
        read_runtime_status = None
    for path in candidates:
        try:
            payload = (read_runtime_status(path) if read_runtime_status else
                       json.loads(path.read_text(encoding="utf-8")))
        except (FileNotFoundError, OSError, ValueError, UnicodeError):
            continue
        if not isinstance(payload, dict):
            continue
        generation = payload.get("generation", payload.get("process_generation"))
        if generation is not None:
            generations.add(generation)
        scopes = payload.get("requiredScopes", payload.get("required_scopes"))
        if scopes is not None:
            required_scopes_known = isinstance(scopes, list)
            if required_scopes_known:
                required_scopes.update(str(scope) for scope in scopes)
        processes.append({
            "pid": payload.get("pid"), "generation": generation,
            "startTime": payload.get("start_time", payload.get("process_start_time")),
            "codeSha": payload.get("code_sha"), "path": str(path),
        })
    generation: Any = next(iter(generations)) if len(generations) == 1 else None
    try:
        from hermes_cli.update_inventory import collect_runtime_inventory
        inventory = collect_runtime_inventory().to_dict()
    except Exception:
        inventory = {"state": "unavailable"}
    return {
        "generation": generation,
        "requiredScopes": sorted(required_scopes) if required_scopes_known else None,
        "processes": processes,
        "inventory": inventory,
        "processGeneration": {
            "state": "matched" if generation is not None and len(generations) == 1 else "unknown",
            "observed": generation,
        },
    }


def _dependency_evidence() -> dict[str, Any]:
    """Bounded dependency evidence; this probe never installs or repairs anything."""
    executable = Path(sys.executable)
    return {
        "state": "ready" if executable.is_file() else "unknown",
        "python": str(executable),
        "checked": ["python-executable"],
        "missing": [],
    }


def collect_observation(*, mode: str = "inventory", correlation_id: str | None = None) -> dict[str, Any]:
    """Collect one bounded, JSON-compatible observation without mutation."""
    if mode not in _MODES:
        raise ValueError(f"unsupported mode: {mode}")
    root = Path(get_default_hermes_root())
    git = _read_git_metadata(_CODE_ROOT)
    image = _read_image_marker()
    recovery = _read_recovery()
    runtime = _read_runtime_evidence()
    dependencies = _dependency_evidence()
    deployment_kind = image.get("deploymentKind") or "git"
    deployment_eligible = _deployment_eligible(image)
    deployment_reason = None if deployment_eligible else f"{image.get('state', 'unknown')}-image-marker"
    checkout_state = git.get("checkoutState", "unknown")
    recovery_markers = recovery.get("markers", {})
    if not isinstance(recovery_markers, dict):
        recovery_markers = {"recovery": recovery_markers}
    process_generation = runtime.get("processGeneration", {
        "state": "matched" if runtime.get("generation") is not None else "unknown",
        "observed": runtime.get("generation"),
    })
    observation = {
        "installId": _read_raw_install_id(root),
        "codeRoot": git.get("codeRoot") or str(_CODE_ROOT.resolve()),
        "repository": {"origin": git.get("origin"), "trackingBranch": git.get("trackingBranch")},
        "checkoutSha": git.get("checkoutSha"),
        "checkout": {
            "state": checkout_state, "divergence": git.get("divergence", "unknown"),
            "ahead": git.get("ahead"), "behind": git.get("behind"),
        },
        "protocolCapability": {
            "version": PROTOCOL_VERSION, "readOnly": True, "modes": list(_MODES),
        },
        "deployment": {
            "kind": deployment_kind, "eligible": deployment_eligible,
            "reason": deployment_reason,
        },
        "markers": {"imageProvenance": image, **recovery_markers},
        "recovery": recovery,
        "runtime": runtime,
        "dependencies": dependencies,
        "readiness": {
            "state": "ready" if deployment_eligible and checkout_state == "clean" and dependencies["state"] == "ready" else "blocked",
            "deploymentEligible": deployment_eligible,
            "dependencies": dependencies["state"],
            "processGeneration": process_generation,
        },
    }
    request: dict[str, Any] = {"mode": mode}
    if correlation_id is not None:
        request["correlationId"] = correlation_id
    return {"metadata": {"protocol": PROTOCOL_VERSION}, "request": request, "observation": observation}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only Hermes update rollout probe")
    parser.add_argument("--mode", choices=_MODES, default="inventory")
    parser.add_argument("--correlation-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Print exactly one JSON object to stdout; diagnostics are stderr-only."""
    try:
        args = _parser().parse_args(argv)
        payload = collect_observation(mode=args.mode, correlation_id=args.correlation_id)
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        print(f"update rollout probe: {type(exc).__name__}: {exc}", file=sys.stderr)
        fallback = {
            "metadata": {"protocol": PROTOCOL_VERSION},
            "request": {"mode": "unknown"},
            "observation": {"installId": None, "codeRoot": None, "checkoutSha": None,
                            "protocolCapability": {"version": PROTOCOL_VERSION, "readOnly": True}},
        }
        print(json.dumps(fallback, sort_keys=True, separators=(",", ":")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
