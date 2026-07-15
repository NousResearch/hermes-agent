#!/usr/bin/env python3
"""Stage the digest-bound production cron continuity package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_RELEASE_ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
if str(_RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(_RELEASE_ROOT))

from gateway.production_cron_continuity_package import (  # noqa: E402
    ProductionCronContinuityPackageError,
    derive_packaged_continuity_from_host,
    stage_packaged_continuity_from_host,
)
from gateway import canonical_writer_production_cutover as cutover  # noqa: E402
from gateway.production_cron_migration import DEFAULT_JOBS_PATH  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path(
    "/var/lib/muncho-production-legacy-cutover/staged/cron-continuity"
)
LEGACY_NOOP_SCHEMA = "muncho-production-cron-continuity-stage-noop.v1"


def _json(path: Path) -> Mapping[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError("production_cron_stage_input_unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= 8 * 1024 * 1024
        ):
            raise RuntimeError("production_cron_stage_input_invalid")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_uid,
        item.st_gid,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if len(raw) != before.st_size or identity(before) != identity(after):
        raise RuntimeError("production_cron_stage_input_changed")
    value = json.loads(raw.decode("utf-8", errors="strict"))
    if not isinstance(value, Mapping):
        raise RuntimeError("production_cron_stage_input_invalid")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage production cron continuity without installing it"
    )
    parser.add_argument("action", choices=("derive", "stage"))
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--mechanical-job-package",
        type=Path,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if os.geteuid() != 0:  # windows-footgun: ok — Linux production/canary boundary
        raise RuntimeError("production_cron_stage_root_required")
    if args.action == "derive":
        if args.mechanical_job_package is None:
            raise RuntimeError("production_cron_derive_package_required")
        mechanical = _json(args.mechanical_job_package)
        derived = derive_packaged_continuity_from_host(
            revision=args.revision,
            mechanical_job_package=mechanical,
            source_jobs_path=DEFAULT_JOBS_PATH,
        )
        unsigned = {
            "schema": "muncho-production-cron-continuity-derivation.v1",
            "continuity_plan": derived.build.plan,
            "inventory": derived.inventory,
            "replacement_bundle_sha256": derived.build.replacement_bundle[
                "bundle_sha256"
            ],
            "filesystem_mutation_performed": False,
            "prompt_or_script_content_recorded": False,
        }
        encoded = json.dumps(
            unsigned,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        result = {
            **unsigned,
            "derivation_sha256": hashlib.sha256(encoded).hexdigest(),
        }
    else:
        if args.mechanical_job_package is not None:
            raise RuntimeError("production_cron_stage_package_unexpected")
        freeze_object = cutover.FreezePlan.from_mapping(
            _json(cutover.STAGED_FREEZE_PLAN_PATH)
        )
        freeze = freeze_object.value
        if freeze["release_revision"] != args.revision:
            raise RuntimeError("production_cron_stage_revision_drifted")
        authority = freeze["cutover_authority"]
        continuity_plan = authority["cron_continuity_plan"]
        if continuity_plan.get("schema") != (
            "muncho-production-cron-packaged-continuity-plan.v3"
        ):
            unsigned = {
                "schema": LEGACY_NOOP_SCHEMA,
                "release_revision": args.revision,
                "freeze_plan_sha256": freeze_object.sha256,
                "continuity_plan_sha256": continuity_plan[
                    "owner_approved_plan_sha256"
                ],
                "legacy_noop": True,
                "artifacts_staged": False,
                "units_installed": False,
                "timers_enabled": False,
                "timers_started": False,
                "jobs_store_mutated": False,
                "secret_material_recorded": False,
            }
            result = {
                **unsigned,
                "receipt_sha256": hashlib.sha256(
                    json.dumps(
                        unsigned,
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("ascii")
                ).hexdigest(),
            }
        else:
            result = stage_packaged_continuity_from_host(
                revision=args.revision,
                mechanical_job_package=authority[
                    "mechanical_job_package"
                ],
                source_jobs_path=DEFAULT_JOBS_PATH,
                output_root=DEFAULT_OUTPUT_ROOT,
                expected_continuity_plan=continuity_plan,
            )
    sys.stdout.buffer.write(
        json.dumps(
            result,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        RuntimeError,
        ProductionCronContinuityPackageError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None
