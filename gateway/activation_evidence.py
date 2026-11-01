"""Candidate v0.22: project already resolved gateway state; never load configuration.

Installed target (only after approval): gateway/activation_evidence.py.
The control handler executes on the gateway loop, without awaits in the projection.
This is activation intent, not a claim that a listener or a business workflow works.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import CancelledError
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import tempfile


def project_activation(runner, identity, install_directory):
    """Strict whitelist: no config serialization, tokens, environment or file reads."""
    if not getattr(runner, "_running", False):
        raise ValueError("ACTIVATION_UNVERIFIED")
    lock = getattr(runner, "_profile_reconcile_lock", None)
    if lock is not None and lock.locked():
        raise ValueError("ACTIVATION_RECONFIGURING")
    homes = getattr(runner, "_served_profile_homes", None)
    configs = getattr(runner, "_profile_configs", None)
    signatures = getattr(runner, "_served_profile_signatures", None)
    primary = getattr(runner, "_primary_profile_name", None) or "default"
    if not isinstance(homes, dict) or not homes or not isinstance(configs, dict):
        raise ValueError("ACTIVATION_UNVERIFIED")
    profiles = {}
    for name in sorted(homes):
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name):
            raise ValueError("ACTIVATION_UNVERIFIED")
        if name == primary:
            config = runner.config
        else:
            if not isinstance(signatures, dict) or name not in signatures or name not in configs:
                raise ValueError("ACTIVATION_UNVERIFIED")
            config = configs[name]
        # Same resolved PlatformConfig.enabled value that the adapter loop consumes.
        # No webhook entry means it cannot be selected by that loop.
        platforms = config.platforms
        matches = [value for key, value in platforms.items() if key.value == "webhook"]
        if len(matches) > 1:
            raise ValueError("ACTIVATION_UNVERIFIED")
        enabled = matches[0].enabled if matches else False
        if type(enabled) is not bool:
            raise ValueError("ACTIVATION_UNVERIFIED")
        profiles[name] = {"webhookEnabled": enabled}
    result = {
        "schemaVersion": "hermes.gateway-activation-evidence/v1",
        "installDirectory": str(install_directory),
        "runtimeCommit": identity.get("code_sha"),
        "runtimePid": identity.get("pid"),
        "runtimeStartTime": identity.get("start_time"),
        "observedAt": datetime.now(UTC).isoformat(),
        "profiles": profiles,
    }
    return validate_evidence(result, install_directory)


def validate_evidence(value, install_directory):
    """Reject unexpected fields as well as incomplete or foreign payloads."""
    expected = {"schemaVersion", "installDirectory", "runtimeCommit", "runtimePid",
                "runtimeStartTime", "observedAt", "profiles"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("ACTIVATION_UNVERIFIED")
    if (value["schemaVersion"] != "hermes.gateway-activation-evidence/v1"
            or not isinstance(value["runtimeCommit"], str)
            or not re.fullmatch(r"[a-f0-9]{40}", value["runtimeCommit"])
            or type(value["runtimePid"]) is not int or value["runtimePid"] < 1
            or type(value["runtimeStartTime"]) is not int or value["runtimeStartTime"] < 1
            or not isinstance(value["installDirectory"], str)
            or not Path(value["installDirectory"]).is_absolute()
            or Path(value["installDirectory"]).resolve() != Path(install_directory).resolve()):
        raise ValueError("ACTIVATION_UNVERIFIED")
    try:
        observed = datetime.fromisoformat(value["observedAt"])
        age = (datetime.now(UTC) - observed).total_seconds()
    except (ValueError, TypeError):
        raise ValueError("ACTIVATION_UNVERIFIED") from None
    if not 0 <= age <= 300:
        raise ValueError("ACTIVATION_UNVERIFIED")
    profiles = value["profiles"]
    if not isinstance(profiles, dict) or not profiles or len(profiles) > 256:
        raise ValueError("ACTIVATION_UNVERIFIED")
    for name, entry in profiles.items():
        if (not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", name)
                or not isinstance(entry, dict) or set(entry) != {"webhookEnabled"}
                or type(entry["webhookEnabled"]) is not bool):
            raise ValueError("ACTIVATION_UNVERIFIED")
    return value


def activation_evidence_handlers(runner, loop):
    """No new control surface without a strict, explicit configuration opt-in."""
    enabled = runner.config.activation_evidence_export
    if type(enabled) is not bool:
        raise ValueError("activation_evidence_export must be a boolean")
    return {"activation-evidence": activation_evidence_verb(runner, loop)} if enabled else {}


def activation_evidence_verb(runner, loop):
    """Read-only control verb. Serialize on the same loop as profile reconciliation."""
    def snapshot():
        try:
            from gateway.status import _build_pid_record, _get_code_identity_fields
            identity = {**_build_pid_record(), **_get_code_identity_fields()}
            return project_activation(runner, identity, Path(__file__).resolve().parents[1])
        except Exception:  # health: allow BLE001 -- RPC boundary must sanitize arbitrary plugin exceptions to prevent credential disclosure.
            # Never expose exception text from a config/plugin object on this surface.
            raise ValueError("ACTIVATION_UNVERIFIED") from None

    def handle():
        # Windows pipe dispatch is already on the loop; waiting on a submitted
        # coroutine there would deadlock. POSIX dispatch runs on an executor.
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is loop:
            return snapshot()
        async def on_loop():
            return snapshot()
        future = asyncio.run_coroutine_threadsafe(on_loop(), loop)
        try:
            return future.result(timeout=2.0)
        except (ValueError, RuntimeError, TimeoutError, CancelledError):
            future.cancel()
            raise ValueError("ACTIVATION_UNVERIFIED") from None
    return handle


def export_evidence(home, output, install_directory, query):
    """One explicit read-only request; atomic output, no change to the gateway/home."""
    output = Path(output)
    # Non-sensitive artifact only. Refuse links and existing destinations to preserve data.
    if output.name != "activation-evidence.json" or output.exists() or output.is_symlink():
        raise ValueError("EXPORT_DESTINATION_REFUSED")
    resolved = output.parent.resolve(strict=True)
    home = Path(home).resolve(strict=True)
    install = Path(install_directory).resolve(strict=True)
    if resolved == home or home in resolved.parents or resolved == install or install in resolved.parents:
        raise ValueError("EXPORT_DESTINATION_REFUSED")
    value = validate_evidence(query(home, "activation-evidence", timeout=3.0), install_directory)
    text = json.dumps(value, ensure_ascii=True, sort_keys=True) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=resolved,
                                         prefix=".activation-", delete=False) as stream:
            temporary = Path(stream.name)
            os.chmod(temporary, 0o600)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication refuses a destination created concurrently (no overwrite).
        os.link(temporary, resolved / output.name)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"exported": True, "profiles": len(value["profiles"])}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export non-sensitive effective activation once")
    parser.add_argument("--home", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        from gateway.control_socket import query_gateway_control
        result = export_evidence(args.home, args.output, Path(__file__).resolve().parents[1],
                                 query_gateway_control)
    except (OSError, ValueError, TypeError, ImportError):
        print("ACTIVATION_EXPORT_REFUSED")
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
