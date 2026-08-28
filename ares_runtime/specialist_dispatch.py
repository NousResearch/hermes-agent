"""Explicit, Electron-admitted specialist dispatch.

This module owns strict request validation, descriptor/binding checks, the
profile-isolated one-shot runner, and durable terminal receipts.  It never
computes capacity or launches a worker when Electron has not admitted the
request.  The Desktop owns the process group and four-slot capacity budget.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

from .collaboration import (
    ContractError,
    SpecialistDescriptorV1,
    digest,
    specialist_descriptor_ref,
    validate_specialist_descriptor_set,
)


REQUEST_SCHEMA = "AresExplicitSpecialistDispatchRequestV1"
ENVELOPE_SCHEMA = "AresDesktopSpecialistDispatchEnvelopeV1"
ENDPOINT_SCHEMA = "AresDesktopSpecialistDispatchEndpointV1"
RECEIPT_SCHEMA = "AresExplicitSpecialistDispatchReceiptV1"
MAX_REQUEST_BYTES = 64 * 1024
MAX_BRIEF_BYTES = 12 * 1024
MAX_RESPONSE_BYTES = 16 * 1024
MAX_PROFILES = 4
RUN_ID_RE = re.compile(r"^specialist-run-[a-z0-9][a-z0-9-]{7,63}$")
PROFILE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
CAPABILITY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")
QUIESCE_LEASE_RE = re.compile(r"^specialist-quiesce-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
REQUEST_FIELDS = {
    "schema",
    "run_id",
    "profile_ids",
    "requested_capabilities",
    "workspace",
    "brief",
    "request_digest",
}


class DispatchError(ValueError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}{(': ' + detail) if detail else ''}")


@dataclass(frozen=True)
class ExplicitDispatchRequest:
    payload: Mapping[str, object]

    @property
    def brief(self) -> str:
        return str(self.payload["brief"])

    @property
    def profile_ids(self) -> tuple[str, ...]:
        return tuple(self.payload["profile_ids"])  # type: ignore[arg-type]

    @property
    def run_id(self) -> str:
        return str(self.payload["run_id"])

    @property
    def request_digest(self) -> str:
        return str(self.payload["request_digest"])

    @property
    def requested_capabilities(self) -> Mapping[str, str]:
        return dict(self.payload["requested_capabilities"])  # type: ignore[arg-type]

    @property
    def workspace(self) -> Path:
        return Path(str(self.payload["workspace"]))

    def public_dict(self) -> dict[str, object]:
        return {key: value for key, value in self.payload.items() if key != "brief"}


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DispatchError("DUPLICATE_JSON_KEY", key)
        result[key] = value
    return result


def _strict_json(raw: bytes) -> dict[str, Any]:
    if len(raw) > MAX_REQUEST_BYTES:
        raise DispatchError("REQUEST_TOO_LARGE")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicates)
    except UnicodeDecodeError as exc:
        raise DispatchError("INVALID_JSON", "utf8") from exc
    except json.JSONDecodeError as exc:
        raise DispatchError("INVALID_JSON") from exc
    if not isinstance(value, dict):
        raise DispatchError("INVALID_REQUEST", "root")
    return value


def parse_dispatch_request(raw: bytes) -> ExplicitDispatchRequest:
    value = _strict_json(raw)
    if set(value) != REQUEST_FIELDS or value.get("schema") != REQUEST_SCHEMA:
        unknown = sorted(set(value) - REQUEST_FIELDS)
        if unknown:
            raise DispatchError("UNKNOWN_FIELD", unknown[0])
        raise DispatchError("INVALID_REQUEST", "schema_or_fields")
    run_id = value.get("run_id")
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
        raise DispatchError("INVALID_RUN_ID")
    profile_ids = value.get("profile_ids")
    if (
        not isinstance(profile_ids, list)
        or not (1 <= len(profile_ids) <= MAX_PROFILES)
        or any(not isinstance(profile, str) or not PROFILE_RE.fullmatch(profile) for profile in profile_ids)
        or profile_ids != sorted(profile_ids)
        or len(set(profile_ids)) != len(profile_ids)
    ):
        raise DispatchError("INVALID_PROFILE_SET")
    requested_capabilities = value.get("requested_capabilities")
    if (
        not isinstance(requested_capabilities, dict)
        or set(requested_capabilities) != set(profile_ids)
        or any(
            not isinstance(profile_id, str)
            or not PROFILE_RE.fullmatch(profile_id)
            or not isinstance(capability, str)
            or not CAPABILITY_RE.fullmatch(capability)
            for profile_id, capability in requested_capabilities.items()
        )
    ):
        raise DispatchError("INVALID_CAPABILITY_SET")
    workspace = value.get("workspace")
    if not isinstance(workspace, str) or len(workspace) > 4096:
        raise DispatchError("INVALID_WORKSPACE")
    workspace_path = Path(workspace)
    if not workspace_path.is_absolute() or not workspace_path.is_dir():
        raise DispatchError("INVALID_WORKSPACE")
    brief = value.get("brief")
    if not isinstance(brief, str) or not brief.strip() or "\x00" in brief or len(brief.encode("utf-8")) > MAX_BRIEF_BYTES:
        raise DispatchError("INVALID_BRIEF")
    supplied = value.get("request_digest")
    unsigned = {key: item for key, item in value.items() if key != "request_digest"}
    if not isinstance(supplied, str) or not SHA256_RE.fullmatch(supplied) or supplied != digest(unsigned):
        raise DispatchError("REQUEST_DIGEST_MISMATCH")
    return ExplicitDispatchRequest(dict(value))


def explicit_dispatch_decision(
    *,
    request: ExplicitDispatchRequest,
    candidates: Mapping[str, SpecialistDescriptorV1],
    profile_binding_refs: Mapping[str, str],
) -> dict[str, object]:
    """Validate explicit profiles without selecting, dialling, or reserving them."""
    rejections: list[dict[str, str]] = []
    selected: list[str] = []
    for profile_id in request.profile_ids:
        candidate = candidates.get(profile_id)
        if candidate is None:
            rejections.append({"profile_id": profile_id, "reason_code": "DESCRIPTOR_NOT_FOUND"})
            continue
        payload = candidate.to_dict()
        if profile_binding_refs.get(profile_id) != specialist_descriptor_ref(candidate):
            rejections.append({"profile_id": profile_id, "reason_code": "PROFILE_BINDING_MISMATCH"})
        elif payload.get("enabled") is not True:
            rejections.append({"profile_id": profile_id, "reason_code": "DESCRIPTOR_DISABLED"})
        elif request.requested_capabilities[profile_id] not in payload.get("capability_classes", []):
            rejections.append({"profile_id": profile_id, "reason_code": "CAPABILITY_NOT_MATCHED"})
        else:
            selected.append(profile_id)
    base: dict[str, object] = {
        "schema": "AresExplicitSpecialistDispatchDecisionV1",
        "request_digest": request.request_digest,
        "requested_capabilities": dict(request.requested_capabilities),
        "requested_profile_ids": list(request.profile_ids),
        "selected_profile_ids": selected if not rejections else [],
        "capacity_authority": "electron_required",
        "automatic_selection": False,
        "generic_fallback": "forbidden",
        "candidate_rejections": rejections,
    }
    base["outcome"] = "eligible" if not rejections else "blocked"
    base["decision_digest"] = digest(base)
    return base


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_explicit_dispatch(
    *,
    request: ExplicitDispatchRequest,
    candidates: Mapping[str, SpecialistDescriptorV1],
    profile_binding_refs: Mapping[str, str],
    receipt_root: Path,
    worker: Callable[[str, ExplicitDispatchRequest], Mapping[str, object]],
) -> dict[str, object]:
    """Run already-Electron-admitted profiles and write a durable terminal receipt."""
    receipt_dir = receipt_root / request.run_id
    if receipt_dir.exists():
        raise DispatchError("RUN_ID_ALREADY_EXISTS")
    receipt_dir.mkdir(parents=True, mode=0o700)
    decision = explicit_dispatch_decision(
        request=request, candidates=candidates, profile_binding_refs=profile_binding_refs
    )
    receipt: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "run_id": request.run_id,
        "request": request.public_dict(),
        "decision": decision,
        "profiles": [],
        "terminal_state": "running",
    }
    if decision["outcome"] != "eligible":
        receipt["terminal_state"] = "rejected"
        receipt["receipt_digest"] = digest(receipt)
        _atomic_json(receipt_dir / "receipt.json", receipt)
        return receipt

    profiles = list(request.profile_ids)
    results: dict[str, Mapping[str, object]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(profiles)) as executor:
        futures = {executor.submit(worker, profile, request): profile for profile in profiles}
        for future in concurrent.futures.as_completed(futures):
            profile = futures[future]
            try:
                results[profile] = future.result()
            except Exception as exc:
                results[profile] = {"outcome": "runner_failed", "exit_code": None, "error_type": type(exc).__name__}
    normalized: list[dict[str, object]] = []
    for profile in profiles:
        result = results[profile]
        normalized.append(
            {
                "profile_id": profile,
                "outcome": result.get("outcome", "runner_failed"),
                "exit_code": result.get("exit_code"),
            }
        )
    receipt["profiles"] = normalized
    receipt["terminal_state"] = "released" if all(item["outcome"] == "returned" and item["exit_code"] == 0 for item in normalized) else "runner_failed"
    receipt["receipt_digest"] = digest(receipt)
    _atomic_json(receipt_dir / "receipt.json", receipt)
    return receipt


def _role_artifacts(registry: Mapping[str, Any]) -> dict[str, list[str]]:
    roles = registry.get("roles")
    if not isinstance(roles, list):
        raise DispatchError("SEMANTIC_REGISTRY_INVALID")
    result: dict[str, list[str]] = {}
    for role in roles:
        if not isinstance(role, dict) or not isinstance(role.get("role_id"), str):
            raise DispatchError("SEMANTIC_REGISTRY_INVALID")
        artifacts = role.get("required_artifacts")
        if not isinstance(artifacts, list):
            raise DispatchError("SEMANTIC_REGISTRY_INVALID")
        values = [item.get("artifact_id") for item in artifacts if isinstance(item, dict)]
        if len(values) != len(artifacts) or not all(isinstance(item, str) and item for item in values):
            raise DispatchError("SEMANTIC_REGISTRY_INVALID")
        result[role["role_id"]] = [str(item) for item in values]
    return result


def load_active_candidates(source_root: Path, profile_ids: Sequence[str]) -> dict[str, SpecialistDescriptorV1]:
    descriptor_dir = source_root / "docs" / "specialist-descriptors" / "v1"
    registry_path = source_root / "docs" / "role-contracts" / "role-contracts.json"
    manifest_path = descriptor_dir / "manifest.json"
    try:
        registry_raw = registry_path.read_bytes()
        registry = _strict_json(registry_raw)
        manifest = _strict_json(manifest_path.read_bytes())
        raw_by_profile = {
            path.stem: _strict_json(path.read_bytes())
            for path in descriptor_dir.glob("*.json")
            if path.name != "manifest.json"
        }
    except OSError as exc:
        raise DispatchError("DESCRIPTOR_SOURCE_UNAVAILABLE") from exc
    registry_digest = "sha256:" + hashlib.sha256(registry_raw).hexdigest()
    if (
        manifest.get("schema") != "AresSpecialistDescriptorManifestV1"
        or manifest.get("manifest_digest") != digest({key: value for key, value in manifest.items() if key != "manifest_digest"})
    ):
        raise DispatchError("DESCRIPTOR_MANIFEST_INVALID")
    descriptor_rows: list[dict[str, str]] = []
    for filename, raw in raw_by_profile.items():
        profile_id = raw.get("profile_id")
        provenance = raw.get("provenance")
        descriptor_digest = raw.get("descriptor_digest")
        if (
            not isinstance(profile_id, str)
            or not isinstance(descriptor_digest, str)
            or not isinstance(provenance, dict)
            or provenance.get("semantic_registry_ref") != "docs:role-contracts/role-contracts.json"
            or provenance.get("semantic_registry_digest") != registry_digest
        ):
            raise DispatchError("DESCRIPTOR_PROVENANCE_INVALID", filename)
        descriptor_rows.append(
            {"profile_id": profile_id, "path": f"{filename}.json", "descriptor_digest": descriptor_digest}
        )
    descriptor_rows.sort(key=lambda item: item["profile_id"])
    if (
        manifest.get("profile_ids") != [item["profile_id"] for item in descriptor_rows]
        or manifest.get("descriptors") != descriptor_rows
    ):
        raise DispatchError("DESCRIPTOR_MANIFEST_INVALID")
    role_artifacts = _role_artifacts(registry)
    raw_descriptors = list(raw_by_profile.values())
    errors = validate_specialist_descriptor_set(
        raw_descriptors,
        profile_ids=sorted(raw_by_profile),
        semantic_role_artifacts=role_artifacts,
        require_disabled=False,
    )
    if errors:
        raise DispatchError("DESCRIPTOR_SET_INVALID", errors[0])
    selected: dict[str, SpecialistDescriptorV1] = {}
    for profile_id in profile_ids:
        raw = raw_by_profile.get(profile_id)
        if raw is None:
            continue
        try:
            selected[profile_id] = SpecialistDescriptorV1.parse(
                raw,
                profile_exists=lambda candidate: candidate in raw_by_profile,
                semantic_role_artifacts=role_artifacts,
            )
        except ContractError as exc:
            raise DispatchError("DESCRIPTOR_SET_INVALID", exc.code) from exc
    return selected


def _profile_bindings(profile_ids: Sequence[str]) -> dict[str, str]:
    from hermes_cli import profiles

    return {
        profile_id: value
        for profile_id in profile_ids
        if isinstance((value := profiles.get_specialist_descriptor_ref(profile_id)), str)
    }


def _redact_text(value: str) -> str:
    value = re.sub(r"(?i)\b(Bearer)\s+[A-Za-z0-9._~+/=-]{8,}", r"\1 [REDACTED]", value)
    value = re.sub(r"(?i)\b(?:sk-[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9_]{8,}|github_pat_[A-Za-z0-9_]{8,})\b", "[REDACTED]", value)
    return value


def _run_profile_worker(profile_id: str, request: ExplicitDispatchRequest, receipt_dir: Path, root: Path) -> Mapping[str, object]:
    profile_home = root / "profiles" / profile_id
    if not profile_home.is_dir():
        return {"outcome": "runner_failed", "exit_code": None, "error_type": "PROFILE_HOME_MISSING"}
    environment = os.environ.copy()
    environment.update({"HERMES_HOME": str(profile_home), "ARES_MANAGED_RUNTIME": "1", "HERMES_SESSION_SOURCE": "cli"})
    command = [sys.executable, "-m", "hermes_cli.main", "--in", str(request.workspace), "-z", request.brief]
    process = subprocess.Popen(command, cwd=request.workspace, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout_raw, stderr_raw = process.communicate()
    profile_dir = receipt_dir / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    stdout = _redact_text((stdout_raw or b"").decode("utf-8", errors="replace"))
    stderr = _redact_text((stderr_raw or b"").decode("utf-8", errors="replace"))
    for name, text in (("stdout", stdout), ("stderr", stderr)):
        payload = text.encode("utf-8")[:512 * 1024]
        (profile_dir / f"{profile_id}.{name}.txt").write_bytes(payload)
    return {
        "outcome": "returned" if process.returncode == 0 else "runner_failed",
        "exit_code": process.returncode,
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest(),
    }


def runner_main() -> int:
    raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    try:
        request = parse_dispatch_request(raw)
        root = Path(os.environ.get("HERMES_HOME", "")).expanduser().resolve()
        if not root.is_dir():
            raise DispatchError("HERMES_HOME_UNAVAILABLE")
        source_root = Path(__file__).resolve().parents[1]
        candidates = load_active_candidates(source_root, request.profile_ids)
        bindings = _profile_bindings(request.profile_ids)
        receipt = run_explicit_dispatch(
            request=request,
            candidates=candidates,
            profile_binding_refs=bindings,
            receipt_root=root / "specialist-dispatch-runs",
            worker=lambda profile, parsed: _run_profile_worker(
                profile, parsed, root / "specialist-dispatch-runs" / parsed.run_id, root
            ),
        )
        print(json.dumps({"run_id": request.run_id, "terminal_state": receipt["terminal_state"]}, sort_keys=True))
        return 0 if receipt["terminal_state"] == "released" else 1
    except DispatchError as exc:
        print(json.dumps({"schema": RECEIPT_SCHEMA, "terminal_state": "rejected", "reason_code": exc.code}, sort_keys=True))
        return 2


def _read_endpoint(root: Path) -> dict[str, object]:
    try:
        state = _strict_json((root / "specialist-dispatch.json").read_bytes())
    except OSError as exc:
        raise DispatchError("DESKTOP_DISPATCH_UNAVAILABLE") from exc
    if set(state) != {"schema", "host", "port", "token"} or state.get("schema") != ENDPOINT_SCHEMA:
        raise DispatchError("DESKTOP_DISPATCH_UNAVAILABLE")
    if state.get("host") != "127.0.0.1" or not isinstance(state.get("port"), int) or not (1 <= state["port"] <= 65535):
        raise DispatchError("DESKTOP_DISPATCH_UNAVAILABLE")
    if not isinstance(state.get("token"), str) or len(state["token"]) < 32:
        raise DispatchError("DESKTOP_DISPATCH_UNAVAILABLE")
    return state


def _desktop_request(root: Path, envelope: Mapping[str, object]) -> dict[str, object]:
    endpoint = _read_endpoint(root)
    payload = json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8") + b"\n"
    if len(payload) > MAX_REQUEST_BYTES:
        raise DispatchError("REQUEST_TOO_LARGE")
    try:
        host = endpoint["host"]
        port = endpoint["port"]
        assert isinstance(host, str) and isinstance(port, int)
        with socket.create_connection((host, port), timeout=10) as connection:
            connection.sendall(payload)
            response = connection.recv(MAX_RESPONSE_BYTES + 1)
    except OSError as exc:
        raise DispatchError("DESKTOP_DISPATCH_UNAVAILABLE") from exc
    if len(response) > MAX_RESPONSE_BYTES:
        raise DispatchError("DESKTOP_RESPONSE_INVALID")
    result = _strict_json(response.rstrip(b"\n"))
    if "token" in result:
        raise DispatchError("DESKTOP_RESPONSE_INVALID")
    return result


def _parse_capability_bindings(
    bindings: Sequence[str], profile_ids: Sequence[str]
) -> dict[str, str]:
    """Require one explicit ``profile_id:capability`` binding per profile."""
    result: dict[str, str] = {}
    for binding in bindings:
        profile_id, separator, capability = binding.partition(":")
        if (
            not separator
            or not PROFILE_RE.fullmatch(profile_id)
            or not CAPABILITY_RE.fullmatch(capability)
            or profile_id in result
        ):
            raise DispatchError("INVALID_CAPABILITY_SET")
        result[profile_id] = capability
    if set(result) != set(profile_ids):
        raise DispatchError("INVALID_CAPABILITY_SET")
    return {profile_id: result[profile_id] for profile_id in sorted(result)}


def client_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Submit an explicit specialist dispatch to the running Ares Desktop.")
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run")
    run.add_argument("--profile", action="append", required=True)
    run.add_argument(
        "--capability",
        action="append",
        required=True,
        metavar="PROFILE_ID:CAPABILITY",
        help="One exact capability binding for each explicit --profile.",
    )
    run.add_argument("--brief-file", required=True, type=Path)
    run.add_argument("--workspace", required=True, type=Path)
    quiesce = sub.add_parser("quiesce")
    quiesce.add_argument("--profile", action="append", required=True)
    unquiesce = sub.add_parser("unquiesce")
    unquiesce.add_argument("--lease-id", required=True)
    status = sub.add_parser("status")
    status.add_argument("--run-id", required=True)
    cancel = sub.add_parser("cancel")
    cancel.add_argument("--run-id", required=True)
    args = parser.parse_args(list(argv))
    root = Path(os.environ.get("HERMES_HOME", Path.home() / ".ares")).expanduser().resolve()
    endpoint = _read_endpoint(root)
    if args.action == "run":
        profile_ids = sorted(args.profile)
        if (
            len(profile_ids) != len(set(profile_ids))
            or not (1 <= len(profile_ids) <= MAX_PROFILES)
            or any(not PROFILE_RE.fullmatch(profile_id) for profile_id in profile_ids)
        ):
            parser.error("--profile values must be sorted-valid unique profile IDs (1-4 total)")
        try:
            requested_capabilities = _parse_capability_bindings(args.capability, profile_ids)
        except DispatchError:
            parser.error("--capability requires exactly one PROFILE_ID:CAPABILITY binding per --profile")
        brief = args.brief_file.read_text(encoding="utf-8")
        request: dict[str, object] = {
            "schema": REQUEST_SCHEMA,
            "run_id": f"specialist-run-{hashlib.sha256(os.urandom(32)).hexdigest()[:16]}",
            "profile_ids": profile_ids,
            "requested_capabilities": requested_capabilities,
            "workspace": str(args.workspace.expanduser().resolve()),
            "brief": brief,
        }
        request["request_digest"] = digest(request)
        response = _desktop_request(root, {"schema": ENVELOPE_SCHEMA, "operation": "submit", "token": endpoint["token"], "request": request})
    elif args.action == "quiesce":
        profile_ids = sorted(args.profile)
        if (
            len(profile_ids) != len(set(profile_ids))
            or not (1 <= len(profile_ids) <= MAX_PROFILES)
            or any(not PROFILE_RE.fullmatch(profile_id) for profile_id in profile_ids)
        ):
            parser.error("--profile values must be sorted-valid unique profile IDs (1-4 total)")
        response = _desktop_request(
            root,
            {"schema": ENVELOPE_SCHEMA, "operation": "quiesce", "token": endpoint["token"], "profile_ids": profile_ids},
        )
    elif args.action == "unquiesce":
        if not QUIESCE_LEASE_RE.fullmatch(args.lease_id):
            parser.error("--lease-id must be an exact specialist quiesce lease ID")
        response = _desktop_request(
            root,
            {"schema": ENVELOPE_SCHEMA, "operation": "unquiesce", "token": endpoint["token"], "lease_id": args.lease_id},
        )
    else:
        response = _desktop_request(root, {"schema": ENVELOPE_SCHEMA, "operation": args.action, "token": endpoint["token"], "run_id": args.run_id})
    print(json.dumps(response, sort_keys=True))
    return 0 if response.get("outcome") in {"admitted", "released", "status", "quiesced", "unquiesced"} else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("mode", choices=("runner", "client"))
    args, remaining = parser.parse_known_args(list(argv) if argv is not None else sys.argv[1:])
    return runner_main() if args.mode == "runner" else client_main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
