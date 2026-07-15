#!/usr/bin/env python3
"""Credential-free CLI for explicit Cloud Muncho operational operations."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from gateway.operational_edge_catalog import (
    build_operation_argv,
    catalog_public_contract,
    operation_catalog,
)
from gateway.operational_edge_client import (
    AttestedMainPidFileProvider,
    OperationalEdgeClient,
    OperationalEdgeClientConfig,
    OperationalEdgeClientError,
    parse_operational_edge_client_configs,
)
from gateway.operational_edge_protocol import (
    OperationalAccess,
    OperationalIntent,
    operational_command_sha256,
    sha256_json,
)


DEFAULT_CLIENT_CONFIG = Path("/etc/muncho/operational-edge-client.json")


def _stable_json(path: Path, *, maximum: int, allowed_modes: set[int]) -> Mapping[str, Any]:
    try:
        metadata = os.lstat(path)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_gid != 0
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) not in allowed_modes
            or not 0 < metadata.st_size <= maximum
        ):
            raise ValueError
        raw = path.read_bytes()
        def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in items:
                if key in result:
                    raise ValueError("duplicate_key")
                result[key] = item
            return result
        value = json.loads(
            raw.decode("ascii", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (OSError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("operational_edge_client_file_invalid") from exc
    if (
        not isinstance(value, Mapping)
        or raw
        != json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    ):
        raise ValueError("operational_edge_client_file_invalid")
    return value


def _config(path: Path, domain: str) -> OperationalEdgeClientConfig:
    value = _stable_json(path, maximum=256 * 1024, allowed_modes={0o400, 0o440, 0o444})
    try:
        configs = parse_operational_edge_client_configs(value)
        config = configs[domain]
    except (KeyError, OperationalEdgeClientError, TypeError, ValueError) as exc:
        raise ValueError("operational_edge_client_config_invalid") from exc
    return config


def _arguments(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("operational_edge_arguments_invalid") from exc
    if not isinstance(parsed, dict):
        raise ValueError("operational_edge_arguments_invalid")
    return parsed


def _consume_approved_capability(intent: OperationalIntent) -> Mapping[str, Any]:
    """Mechanically consume the exact hash from an existing approved plan."""

    from gateway.canonical_writer_boundary import canonical_writer_call
    from gateway.canonical_writer_protocol import CanonicalWriterOperation

    command_sha256 = operational_command_sha256(intent)
    result = canonical_writer_call(
        CanonicalWriterOperation.CAPABILITY_CONSUME.value,
        {
            "command_sha256": command_sha256,
            "idempotency_key": (
                "operational-edge-consume:" + command_sha256
            ),
            "operational_edge_intent": intent.to_mapping(),
        },
        idempotency_key="operational-edge-consume:" + command_sha256,
    )
    capability = result.get("operational_edge_capability")
    if (
        result.get("authorized") is not True
        or not isinstance(capability, Mapping)
    ):
        raise ValueError("operational_edge_approved_capability_unavailable")
    return capability


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="muncho-ops")
    parser.add_argument("--config", type=Path, default=DEFAULT_CLIENT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("catalog")
    schema = sub.add_parser("schema")
    schema.add_argument("--operation", required=True)
    authorize = sub.add_parser("authorization-hash")
    authorize.add_argument("--operation", required=True)
    authorize.add_argument("--arguments-json", required=True)
    authorize.add_argument("--idempotency-key", required=True)
    invoke = sub.add_parser("invoke")
    invoke.add_argument("--operation", required=True)
    invoke.add_argument("--arguments-json", required=True)
    invoke.add_argument("--idempotency-key", required=True)
    invoke.add_argument("--capability-file", type=Path)
    args = parser.parse_args(argv)

    catalog = operation_catalog()
    if args.command == "catalog":
        print(json.dumps(catalog_public_contract(), ensure_ascii=False, sort_keys=True))
        return 0
    operation = catalog.get(args.operation)
    if operation is None:
        raise SystemExit("unknown operational edge operation")
    if args.command == "schema":
        row = next(
            item
            for item in catalog_public_contract()["operations"]
            if item["operation_id"] == args.operation
        )
        print(json.dumps(row, ensure_ascii=False, sort_keys=True))
        return 0
    if not operation.available:
        raise SystemExit(operation.blocker_code)
    arguments = _arguments(args.arguments_json)
    # Validate the typed argv contract before any socket is contacted.
    build_operation_argv(operation, arguments)
    if args.command == "authorization-hash":
        if operation.access is not OperationalAccess.MUTATION:
            raise SystemExit("authorization hash is only valid for mutation operations")
        intent = OperationalIntent(
            operation_id=operation.operation_id,
            arguments=arguments,
            arguments_sha256=sha256_json(arguments),
            idempotency_key=args.idempotency_key,
        )
        # Round-trip through the exact protocol validator before exposing the
        # object used by Canonical Writer capability.consume.
        intent = OperationalIntent.from_mapping(intent.to_mapping())
        print(
            json.dumps(
                {
                    "command_sha256": operational_command_sha256(intent),
                    "operational_edge_intent": intent.to_mapping(),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    capability = None
    if args.capability_file is not None:
        if operation.access is not OperationalAccess.MUTATION:
            raise SystemExit(
                "capability files are only valid for mutation operations"
            )
        capability = _stable_json(
            args.capability_file, maximum=128 * 1024, allowed_modes={0o400, 0o440, 0o444, 0o600, 0o640}
        )
    if operation.access is OperationalAccess.MUTATION and capability is None:
        intent = OperationalIntent.from_mapping(
            {
                "operation_id": operation.operation_id,
                "arguments": arguments,
                "arguments_sha256": sha256_json(arguments),
                "idempotency_key": args.idempotency_key,
            }
        )
        capability = _consume_approved_capability(intent)
    config = _config(args.config, operation.domain)
    provider = AttestedMainPidFileProvider(
        Path("/run/muncho-operational-edge") / operation.domain / "mainpid.json",
        domain=operation.domain,
    )
    client = OperationalEdgeClient(config, main_pid_provider=provider)
    receipt = client.invoke(
        operation.operation_id,
        arguments,
        idempotency_key=args.idempotency_key,
        capability=capability,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0 if receipt.get("outcome") == "succeeded" else 2


if __name__ == "__main__":
    raise SystemExit(main())
