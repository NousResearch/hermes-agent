"""Explicit, non-auto-starting Hermes-owned adapter runtime."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import stat
import sys
import tempfile
import re
from dataclasses import dataclass
from pathlib import Path

from .adapter import BuilderDispatchAdapter
from .attestation import GovernanceSnapshot, HermesProfileResolver
from .auth import HMACAuthenticator, PrincipalKey, darwin_peer_credentials
from .errors import AdapterError
from .gitops import GitVerifier
from .native import NativeKanbanBackend
from .schemas import SchemaRegistry
from .service import BuilderAdapterService, serve_until
from .store import DispatchStore
from .validation import ValidationRunner


SCHEMA_PATHS = {
    "dispatch_request": "contracts/schemas/hermes-builder-dispatch-request-v1.json",
    "dispatch_result": "contracts/schemas/hermes-builder-dispatch-result-v1.json",
    "completion_evidence": "contracts/schemas/hermes-builder-completion-evidence-v1.json",
    "allowed_manifest": "contracts/schemas/allowed-path-manifest-v1.json",
}


@dataclass(frozen=True)
class RuntimeSettings:
    socket_path: Path
    state_path: Path
    auth_file: Path
    governance_repo: Path
    governance_commit: str
    repository_allowlist: dict[str, str]
    validation_profile_id: str
    board: str
    cycle_registry: dict[str, dict]
    validation_docker_binary: str | None
    validation_docker_host: str | None
    validation_image_id: str | None

    @classmethod
    def from_file(cls, path: str | Path) -> "RuntimeSettings":
        value = _read_owner_json(Path(path), exact_mode=None)
        return cls(
            socket_path=Path(value["socket_path"]),
            state_path=Path(value["state_path"]),
            auth_file=Path(value["auth_file"]),
            governance_repo=Path(value["governance_repo"]),
            governance_commit=value["governance_commit"],
            repository_allowlist=dict(value["repository_allowlist"]),
            validation_profile_id=value["validation_profile_id"],
            board=value.get("board", "governed-builder"),
            cycle_registry=dict(value["cycle_registry"]),
            validation_docker_binary=value.get("validation_docker_binary"),
            validation_docker_host=value.get("validation_docker_host"),
            validation_image_id=value.get("validation_image_id"),
        )


def _read_owner_json(path: Path, *, exact_mode: int | None) -> dict:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        info = os.fstat(descriptor)
        if (
            info.st_uid != os.geteuid()
            or not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_size > 1_000_000
            or (
                exact_mode is None
                and stat.S_IMODE(info.st_mode) & 0o077
            )
            or (
                exact_mode is not None
                and stat.S_IMODE(info.st_mode) != exact_mode
            )
        ):
            raise AdapterError(
                "AUTHORIZATION_FAILED", "owner-only configuration file required"
            )
        chunks = []
        remaining = 1_000_001
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
    except OSError as exc:
        raise AdapterError(
            "AUTHORIZATION_FAILED", "configuration file could not be opened safely"
        ) from exc
    finally:
        if "descriptor" in locals():
            os.close(descriptor)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdapterError("INVALID_REQUEST", "invalid configuration JSON") from exc
    if not isinstance(value, dict):
        raise AdapterError("INVALID_REQUEST", "configuration must be an object")
    return value


def _load_keys(path: Path) -> list[PrincipalKey]:
    value = _read_owner_json(path, exact_mode=0o600)
    keys = []
    for item in value.get("keys", []):
        secret_env = item["secret_env"]
        if not re.fullmatch(r"HERMES_BUILDER_ADAPTER_SECRET_[A-Z0-9_]+", secret_env):
            raise AdapterError(
                "AUTHORIZATION_FAILED", "unapproved secret source identifier"
            )
        secret = os.environ.get(secret_env)
        if not secret:
            raise AdapterError(
                "AUTHENTICATION_FAILED",
                f"approved secret source did not supply {secret_env}",
            )
        keys.append(
            PrincipalKey(
                principal=item["principal"],
                key_id=item["key_id"],
                secret=secret.encode(),
                allowed_uid=int(item["allowed_uid"]),
                allowed_gid=(
                    int(item["allowed_gid"])
                    if item.get("allowed_gid") is not None
                    else None
                ),
                active=bool(item.get("active", True)),
            )
        )
    if not keys:
        raise AdapterError("AUTHENTICATION_FAILED", "no authorized principals")
    return keys


def build_runtime(settings: RuntimeSettings):
    snapshot = GovernanceSnapshot(settings.governance_repo, settings.governance_commit)
    validation_profile = snapshot.value("validation_profile")
    if validation_profile.get("profile_id") != settings.validation_profile_id:
        raise AdapterError(
            "MANIFEST_MISMATCH", "runtime validation profile ID is not registered"
        )
    git = GitVerifier(settings.repository_allowlist)
    schema_temp = tempfile.TemporaryDirectory(prefix="hermes-builder-schemas-")
    schema_root = Path(schema_temp.name)
    schema_files = {}
    schema_artifacts = {
        "dispatch_request": "dispatch_request_schema",
        "dispatch_result": "dispatch_result_schema",
        "completion_evidence": "completion_evidence_schema",
        "allowed_manifest": "allowed_path_manifest_schema",
    }
    for name, artifact_id in schema_artifacts.items():
        destination = schema_root / f"{name}.json"
        destination.write_bytes(snapshot.raw(artifact_id))
        schema_files[name] = destination
    schemas = SchemaRegistry(schema_files)
    store = DispatchStore(settings.state_path)
    adapter = BuilderDispatchAdapter(
        store=store,
        schemas=schemas,
        git=git,
        kanban=NativeKanbanBackend(board=settings.board),
        validation=ValidationRunner(
            {settings.validation_profile_id: validation_profile},
            python=sys.executable,
            docker=settings.validation_docker_binary,
            docker_host=settings.validation_docker_host,
            image_id=settings.validation_image_id,
        ),
        governance_repo=settings.governance_repo,
        governance_attestor=snapshot,
        profile_resolver=HermesProfileResolver(),
        cycle_registry=settings.cycle_registry,
    )
    auth = HMACAuthenticator(_load_keys(settings.auth_file), store)
    service = BuilderAdapterService(
        adapter, auth, peer_resolver=darwin_peer_credentials
    )
    return service.application(), schema_temp


async def serve(settings: RuntimeSettings) -> None:
    app, schema_temp = build_runtime(settings)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed_signals = _install_shutdown_handlers(loop, stop)
    try:
        await serve_until(app, settings.socket_path, stop)
    finally:
        for signum in installed_signals:
            loop.remove_signal_handler(signum)
        schema_temp.cleanup()


def _install_shutdown_handlers(loop, stop: asyncio.Event) -> tuple[signal.Signals, ...]:
    """Turn supervisor termination into an orderly UDS cleanup."""
    installed = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop.set)
        except (NotImplementedError, RuntimeError):
            continue
        installed.append(signum)
    return tuple(installed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m plugins.builder_adapter")
    parser.add_argument("command", choices=["serve"])
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    settings = RuntimeSettings.from_file(args.config)
    asyncio.run(serve(settings))
    return 0
