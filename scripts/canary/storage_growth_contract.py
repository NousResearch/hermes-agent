#!/usr/bin/env python3
"""Pure canonical contract for the one reviewed canary storage-growth plan.

This module deliberately imports no approval, journal, launcher, provider, or
service code.  Both the historical host state machine and the passkey-v2
executor consume this exact value so there is one semantic source of truth.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


STORAGE_GROWTH_PLAN_SCHEMA = "muncho-isolated-canary-storage-growth-plan.v1"
PROJECT = "adventico-ai-platform"
PROJECT_NUMBER = "39589465056"
ZONE = "europe-west3-a"
OWNER_ACCOUNT = "lomliev@adventico.com"
VM_NAME = "muncho-canary-v2-01"
VM_INSTANCE_ID = "9153645328899914617"
DISK_NAME = "muncho-canary-v2-01"
DISK_ID = "4195397669213846393"
BOOT_DEVICE_NAME = "persistent-disk-0"
SOURCE_SIZE_GB = 40
TARGET_SIZE_GB = 80
DISK_TYPE = "pd-balanced"
SOURCE_IMAGE_PROJECT = "debian-cloud"
SOURCE_IMAGE = "debian-12-bookworm-v20260609"
ROOT_SOURCE = "/dev/sda1"
ROOT_FILESYSTEM = "ext4"
ROOT_MOUNTPOINT = "/"
MINIMUM_TARGET_ROOT_FILESYSTEM_BYTES = 84_000_000_000
MINIMUM_FREE_BYTES = 32 * 1024 * 1024 * 1024
AUTHORITATIVE_JOURNAL_ROOT = "/var/lib/muncho-owner-gate/executor"
AUTHORITATIVE_EXECUTOR_UID = 29103
AUTHORITATIVE_EXECUTOR_GID = 29103
TRANSACTION_AUTHORIZATION_MAX_AGE_SECONDS = 60 * 60
PREFLIGHT_MAX_AGE_SECONDS = 900
STORAGE_GROWTH_PREFLIGHT_SCHEMA = (
    "muncho-isolated-canary-storage-growth-preflight.v1"
)
MINIMUM_SOURCE_ROOT_FILESYSTEM_BYTES = 39_000_000_000
MAXIMUM_SOURCE_ROOT_FILESYSTEM_BYTES = SOURCE_SIZE_GB * 1024**3
MINIMUM_SOURCE_FREE_BYTES = 2 * 1024 * 1024 * 1024
MAXIMUM_TARGET_ROOT_FILESYSTEM_BYTES = TARGET_SIZE_GB * 1024**3

CURRENT_STOPPED_RELEASE_SHA = "bc37d4252c46f6780e10552580fedb5147157bee"
CURRENT_HOST_RECEIPT_FILE_SHA256 = (
    "ecb53958439984bb317578f8495358c04db01669df06dc7e0f3af8c7eb982f55"
)
CURRENT_HOST_RECEIPT_SHA256 = (
    "4b6a6716c27a52659f204fb8a796657aeb370426c80fe21c5b470bfa763f74c7"
)
CURRENT_STOPPED_RELEASE_RECEIPT_FILE_SHA256 = (
    "180aee0ee954d114e26b509bf8af78dab5e8896da05d00f5e275200b94b4f2ed"
)
CURRENT_STOPPED_RELEASE_RECEIPT_SHA256 = (
    "47c79dbc36d2c13009af82572885ce8481c82079f7a2694ccf5ec209ee30541f"
)
EXTERNAL_IAM_POLICY_SHA256 = (
    "236924140942a99e6162ae6492261ddd8b3a3f61013691a44c9b5e79bfcddb16"
)
RUNTIME_SERVICE_ACCOUNT = (
    "muncho-canary-v2-runtime@adventico-ai-platform.iam.gserviceaccount.com"
)
RUNTIME_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)
RUNTIME_ROLES = (
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "projects/adventico-ai-platform/roles/munchoCanaryCloudSqlReadinessV1",
)
RUNTIME_PERMISSIONS = (
    "cloudsql.instances.get",
    "logging.logEntries.create",
    "logging.logEntries.route",
    "monitoring.metricDescriptors.create",
    "monitoring.metricDescriptors.get",
    "monitoring.metricDescriptors.list",
    "monitoring.monitoredResourceDescriptors.get",
    "monitoring.monitoredResourceDescriptors.list",
    "monitoring.timeSeries.create",
)

RESIZE_STEP = "resize_canary_boot_disk_from_40gb_to_80gb"
STOP_STEP = "stop_identity_pinned_canary_for_80gb_filesystem_expansion"
START_STEP = "start_identity_pinned_canary_for_80gb_filesystem_expansion"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_plan_payload() -> Mapping[str, Any]:
    """Return a fresh exact plan payload safe for caller-owned serialization."""

    common = [
        f"--project={PROJECT}",
        f"--zone={ZONE}",
        f"--account={OWNER_ACCOUNT}",
        "--quiet",
    ]
    return {
        "schema": STORAGE_GROWTH_PLAN_SCHEMA,
        "spec": {
            "project": PROJECT,
            "zone": ZONE,
            "owner_account": OWNER_ACCOUNT,
            "vm_name": VM_NAME,
            "vm_instance_id": VM_INSTANCE_ID,
            "disk_name": DISK_NAME,
            "disk_id": DISK_ID,
            "boot_device_name": BOOT_DEVICE_NAME,
            "source_size_gb": SOURCE_SIZE_GB,
            "target_size_gb": TARGET_SIZE_GB,
            "disk_type": DISK_TYPE,
            "source_image_project": SOURCE_IMAGE_PROJECT,
            "source_image": SOURCE_IMAGE,
            "root_source": ROOT_SOURCE,
            "root_filesystem": ROOT_FILESYSTEM,
            "root_mountpoint": ROOT_MOUNTPOINT,
            "minimum_target_root_filesystem_bytes": (
                MINIMUM_TARGET_ROOT_FILESYSTEM_BYTES
            ),
            "minimum_free_bytes": MINIMUM_FREE_BYTES,
        },
        "architecture": {
            "phase": "isolated_canary_host_storage_growth",
            "historical_20_to_40_contract_unchanged": True,
            "source_size_gb": SOURCE_SIZE_GB,
            "target_size_gb": TARGET_SIZE_GB,
            "exact_resize_command_count": 1,
            "conditional_journaled_stop_start": True,
            "requires_complete_twelve_runtime_unit_superset_stopped": True,
            "requires_fresh_read_only_preflight": True,
            "requires_fresh_read_only_postflight": True,
            "transaction_authorization_max_age_seconds": (
                TRANSACTION_AUTHORIZATION_MAX_AGE_SECONDS
            ),
            "authoritative_journal_backend": (
                "muncho-owner-gate-sqlite-full-begin-immediate.v1"
            ),
            "authoritative_journal_owner": "muncho-storage-executor",
            "authoritative_journal_root": AUTHORITATIVE_JOURNAL_ROOT,
            "authoritative_executor_uid": AUTHORITATIVE_EXECUTOR_UID,
            "authoritative_executor_gid": AUTHORITATIVE_EXECUTOR_GID,
            "local_journal_authority": False,
            "terminal_replay_without_mutation": True,
            "guest_command_authority": False,
            "shell_authority": False,
            "cleanup_authority": False,
            "delete_authority": False,
            "snapshot_authority": False,
            "service_start_authority": False,
            "opens_runtime_gate": False,
        },
        # These argv values are display-only.  The privileged executor uses
        # fixed Compute REST endpoints and never executes these strings.
        "steps": [
            {
                "name": RESIZE_STEP,
                "argv": [
                    "gcloud",
                    "compute",
                    "disks",
                    "resize",
                    DISK_NAME,
                    *common[:-1],
                    f"--size={TARGET_SIZE_GB}GB",
                    common[-1],
                ],
            },
            {
                "name": STOP_STEP,
                "argv": [
                    "gcloud",
                    "compute",
                    "instances",
                    "stop",
                    VM_NAME,
                    *common,
                ],
            },
            {
                "name": START_STEP,
                "argv": [
                    "gcloud",
                    "compute",
                    "instances",
                    "start",
                    VM_NAME,
                    *common,
                ],
            },
        ],
    }


def canonical_plan_sha256() -> str:
    return hashlib.sha256(_canonical_json_bytes(canonical_plan_payload())).hexdigest()


def canonical_plan_report() -> Mapping[str, Any]:
    payload = canonical_plan_payload()
    return {**payload, "plan_sha256": canonical_plan_sha256()}
