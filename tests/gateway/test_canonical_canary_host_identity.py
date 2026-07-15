from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from gateway import canonical_canary_host_identity as host_identity


REPO_ROOT = Path(__file__).resolve().parents[2]


def _metadata() -> dict[str, str]:
    return {
        host_identity._GCE_METADATA_PATHS["project_id"]: (
            host_identity.DEDICATED_CANARY_PROJECT_ID
        ),
        host_identity._GCE_METADATA_PATHS["project_number"]: (
            host_identity.DEDICATED_CANARY_PROJECT_NUMBER
        ),
        host_identity._GCE_METADATA_PATHS["zone"]: (
            "projects/39589465056/zones/europe-west3-a"
        ),
        host_identity._GCE_METADATA_PATHS["instance_name"]: (
            host_identity.DEDICATED_CANARY_INSTANCE_NAME
        ),
        host_identity._GCE_METADATA_PATHS["instance_id"]: (
            host_identity.DEDICATED_CANARY_INSTANCE_ID
        ),
        host_identity._GCE_METADATA_PATHS["service_account_email"]: (
            host_identity.DEDICATED_CANARY_SERVICE_ACCOUNT
        ),
    }


def _local() -> dict[str, str]:
    return {
        "machine_id": "1" * 32,
        "hostname": "muncho-canary-v2-01",
        "boot_id": "22222222-2222-4222-8222-222222222222",
    }


def test_stdlib_host_module_preserves_exact_runtime_receipt_contract():
    metadata = _metadata()
    local = _local()

    receipt = host_identity.collect_dedicated_canary_host_identity_receipt(
        metadata_reader=lambda path: metadata[path],
        local_identity_reader=lambda name: local[name],
        observed_at_unix=1_700_000_000,
    )

    assert receipt["schema"] == host_identity.FULL_CANARY_HOST_IDENTITY_SCHEMA
    assert receipt["collector_authority"] == ("trusted_root_read_only_host_collector")
    assert receipt["instance_id"] == host_identity.DEDICATED_CANARY_INSTANCE_ID
    assert receipt["observed_at_unix"] == 1_700_000_000
    unsigned = {
        name: value for name, value in receipt.items() if name != "receipt_sha256"
    }
    assert receipt["receipt_sha256"] == host_identity._sha256_json(unsigned)


def test_bare_system_python_import_and_collection_need_only_stdlib():
    interpreter = Path("/usr/bin/python3")
    if not interpreter.is_file():  # pragma: no cover - supported hosts provide it.
        return
    probe = r"""
import builtins
import json
import sys

real_import = builtins.__import__
forbidden_roots = {
    "cryptography", "psycopg", "pydantic", "requests", "yaml"
}
forbidden_project = {
    "gateway.canonical_full_canary_runtime",
    "gateway.canonical_writer_activation",
    "gateway.canonical_writer_bootstrap",
}

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".", 1)[0] in forbidden_roots or name in forbidden_project:
        raise AssertionError("forbidden import: " + name)
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
from gateway import canonical_canary_host_identity as host
from scripts.canary import writer_release

metadata = {
    host._GCE_METADATA_PATHS["project_id"]: host.DEDICATED_CANARY_PROJECT_ID,
    host._GCE_METADATA_PATHS["project_number"]: host.DEDICATED_CANARY_PROJECT_NUMBER,
    host._GCE_METADATA_PATHS["zone"]: "projects/39589465056/zones/europe-west3-a",
    host._GCE_METADATA_PATHS["instance_name"]: host.DEDICATED_CANARY_INSTANCE_NAME,
    host._GCE_METADATA_PATHS["instance_id"]: host.DEDICATED_CANARY_INSTANCE_ID,
    host._GCE_METADATA_PATHS["service_account_email"]: host.DEDICATED_CANARY_SERVICE_ACCOUNT,
}
local = {
    "machine_id": "1" * 32,
    "hostname": "muncho-canary-v2-01",
    "boot_id": "22222222-2222-4222-8222-222222222222",
}
receipt = host.collect_dedicated_canary_host_identity_receipt(
    metadata_reader=lambda path: metadata[path],
    local_identity_reader=lambda name: local[name],
    observed_at_unix=1700000000,
)
host._observe_dedicated_canary_host = lambda: {"stdlib_only": "yes"}
assert writer_release._default_host_observer() == {"stdlib_only": "yes"}
assert "gateway.canonical_full_canary_runtime" not in sys.modules
print(json.dumps({"ok": True, "schema": receipt["schema"]}, sort_keys=True))
"""
    completed = subprocess.run(
        [str(interpreter), "-B", "-E", "-s", "-c", probe],
        cwd=REPO_ROOT,
        env={
            "HOME": "/nonexistent",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "ok": True,
        "schema": host_identity.FULL_CANARY_HOST_IDENTITY_SCHEMA,
    }
    assert completed.stderr == ""
