from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from tests.gateway.test_canonical_writer_production_cutover import (
    _runtime_attestation,
)


ROOT = Path(__file__).parents[3]
DEPLOY_HELPER = ROOT / "ops/muncho/runtime/muncho-auto-deploy-release"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _hashed(value: dict[str, object], field: str) -> dict[str, object]:
    unsigned = {name: item for name, item in value.items() if name != field}
    return {**unsigned, field: hashlib.sha256(_canonical(unsigned)).hexdigest()}


def _identity(user: str, group: str, uid: int, gid: int) -> dict[str, object]:
    return {"user": user, "group": group, "uid": uid, "gid": gid}


def _operational_receipt_key_ids() -> dict[str, str]:
    domains = (
        "adventico_email", "bitrix", "canonical", "github",
        "infrastructure", "skyvision_db", "skyvision_email",
        "skyvision_gitlab", "skyvision_panel",
    )
    return {
        domain: f"{index:x}" * 64
        for index, domain in enumerate(domains, start=1)
    }


def _unit_input_authority(revision: str) -> tuple[dict, dict, dict]:
    domains = sorted(_operational_receipt_key_ids())
    payload = {
        "schema": "muncho-production-cutover-unit-input-payload.v3",
        "database_ip": "10.20.30.40",
        "target": {
            "project": "adventico-ai-platform",
            "zone": "europe-west3-a",
            "vm": "ai-platform-runtime-01",
            "database": "ai_platform_brain",
            "sql_instance": "production-pg18",
            "sql_host": "10.20.30.40",
            "tls_server_name": "production.example.internal",
            "port": 5432,
            "writer_login": "muncho_production_writer_login",
        },
        "gateway": _identity(
            "ai-platform-brain", "ai-platform-brain", 1000, 1000
        ),
        "writer": _identity(
            "muncho-canonical-writer", "muncho-canonical-writer", 2000, 2000
        ),
        "projector": _identity("muncho-projector", "muncho-projector", 2004, 2004),
        "routeback": _identity(
            "muncho-discord-egress", "muncho-discord-egress", 2002, 2002
        ),
        "connector": _identity(
            "muncho-discord-connector", "muncho-discord-connector", 2001, 2001
        ),
        "mac_ops": _identity(
            "muncho-mac-ops-edge", "muncho-mac-ops-edge", 2003, 2003
        ),
        "browser": _identity(
            "muncho-capability-browser",
            "muncho-capability-browser",
            2006,
            2006,
        ),
        "worker": _identity("muncho-worker", "muncho-worker", 2007, 2007),
        "writer_client_group": {"group": "muncho-writer-client", "gid": 2005},
        "worker_client_group": {"group": "muncho-worker-clients", "gid": 2008},
        "operational_edge_identities": {
            domain: _identity(
                f"muncho-edge-{domain}",
                f"muncho-edge-{domain}",
                2100 + index,
                2100 + index,
            )
            for index, domain in enumerate(domains)
        },
        "operational_edge_socket_groups": {
            domain: {
                "group": f"muncho-edge-{domain}-c",
                "gid": 2200 + index,
            }
            for index, domain in enumerate(domains)
        },
        "writer_capability_public_key_id": "c" * 64,
        "discord_edge_receipt_public_key_id": "a" * 64,
        "operational_edge_key_foundation_sha256": "d" * 64,
        "operational_edge_receipt_public_key_ids": (
            _operational_receipt_key_ids()
        ),
        "discord_reconciliation_intent": {
            "schema": "muncho-production-discord-reconciliation-intent.v1",
            "purpose": "production_discord_policy_reconciliation",
            "release_revision": revision,
            "legacy_public_policy_sha256": "1" * 64,
            "target_public_policy_sha256": "2" * 64,
            "reviewed_reconciliation": True,
            "secret_material_recorded": False,
            "secret_digest_recorded": False,
        },
        "release_owner_uid": 1000,
        "release_owner_gid": 1000,
        "bwrap_sha256": "6" * 64,
        "shell_sha256": "7" * 64,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    public = "8" * 64
    plan = _hashed(
        {
            "schema": "muncho-production-cutover-unit-input-plan.v3",
            "release_revision": revision,
            "unit_inputs": payload,
            "owner_subject_sha256": "9" * 64,
            "owner_public_key_ed25519_hex": public,
            "owner_key_id": hashlib.sha256(bytes.fromhex(public)).hexdigest(),
            "owner_runtime_attestation": _runtime_attestation(revision),
            "created_at_unix": 1_800_000_000,
            "secret_material_recorded": False,
            "secret_digest_recorded": False,
        },
        "plan_sha256",
    )
    approval = _hashed(
        {
            "schema": "muncho-production-cutover-unit-input-approval.v3",
            "purpose": "production_cutover_unit_inputs",
            "plan_sha256": plan["plan_sha256"],
            "release_revision": revision,
            "owner_subject_sha256": plan["owner_subject_sha256"],
            "owner_public_key_ed25519_hex": public,
            "owner_key_id": plan["owner_key_id"],
            "nonce_sha256": "a" * 64,
            "issued_at_unix": 1_800_000_000,
            "expires_at_unix": 1_800_000_600,
            "approved": True,
            "signature_ed25519_hex": "b" * 128,
        },
        "approval_sha256",
    )
    unit_inputs = {
        "schema": "muncho-production-cutover-unit-inputs.v3",
        "release_revision": revision,
        "authority_plan_sha256": plan["plan_sha256"],
        "authority_approval_sha256": approval["approval_sha256"],
        **{name: item for name, item in payload.items() if name != "schema"},
    }
    return plan, approval, unit_inputs


def _run_shell(body: str, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f'source "$DEPLOY_HELPER"\n{body}'],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "DEPLOY_HELPER": str(DEPLOY_HELPER), **environment},
        timeout=20,
    )


def test_target_blob_bootstrap_does_not_depend_on_active_old_command(
    tmp_path: Path,
) -> None:
    source = (tmp_path / "target").resolve()
    bootstrap = source / "ops/muncho/cutover/production_unit_input_bootstrap.py"
    bootstrap.parent.mkdir(parents=True)
    bootstrap.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "from pathlib import Path\n"
        "path = Path(os.environ['TEST_UNIT_INPUT_OUTPUT'])\n"
        "path.chmod(0o444)\n"
        "print('{\"schema\":\"target-blob-bootstrap-test.v1\"}')\n",
        encoding="utf-8",
    )
    git = shutil.which("git") or "/usr/bin/git"
    subprocess.run([git, "init", "-q", str(source)], check=True)
    subprocess.run(
        [git, "-C", str(source), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        [git, "-C", str(source), "config", "user.name", "Test"],
        check=True,
    )
    subprocess.run([git, "-C", str(source), "add", "."], check=True)
    subprocess.run([git, "-C", str(source), "commit", "-qm", "target"], check=True)
    revision = subprocess.check_output(
        [git, "-C", str(source), "rev-parse", "HEAD"],
        text=True,
    ).strip()

    staged = (tmp_path / "staged").resolve()
    staged.mkdir(mode=0o700)
    plan_path = staged / "unit-input-plan.json"
    approval_path = staged / "unit-input-approval.json"
    output_path = staged / "production-unit-inputs.json"
    plan, approval, unit_inputs = _unit_input_authority(revision)
    plan_path.write_bytes(_canonical(plan))
    approval_path.write_bytes(_canonical(approval))
    output_path.write_bytes(_canonical(unit_inputs) + b"\n")
    plan_path.chmod(0o400)
    approval_path.chmod(0o400)
    output_path.chmod(0o600)
    for path in (staged, plan_path, approval_path, output_path):
        os.chown(path, os.geteuid(), os.getegid())

    fake_bin = (tmp_path / "bin").resolve()
    fake_bin.mkdir()
    sudo = fake_bin / "sudo"
    sudo.write_text(
        "#!/bin/sh\n"
        "[ \"$1\" = -n ] && shift\n"
        "[ \"$1\" = -u ] && shift 2\n"
        "exec \"$@\"\n",
        encoding="utf-8",
    )
    sudo.chmod(0o755)
    chown = fake_bin / "chown"
    chown.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    chown.chmod(0o755)
    runtime = (tmp_path / "run").resolve()
    runtime.mkdir()
    owner = subprocess.check_output(["id", "-un"], text=True).strip()
    body = f'''
OWNER={json.dumps(owner)}
SYSTEM_GIT={json.dumps(git)}
SYSTEM_PYTHON={json.dumps(sys.executable)}
CUTOVER_BOOTSTRAP_RUNTIME_DIR={json.dumps(str(runtime))}
CUTOVER_UNIT_INPUT_PLAN_PATH={json.dumps(str(plan_path))}
CUTOVER_UNIT_INPUT_APPROVAL_PATH={json.dumps(str(approval_path))}
CUTOVER_UNIT_INPUTS_PATH={json.dumps(str(output_path))}
CUTOVER_STAGED_TRUSTED_UID={os.getuid()}
CUTOVER_STAGED_TRUSTED_GID={os.getgid()}
bootstrap_cutover_unit_inputs_from_target {json.dumps(str(source))} {revision}
'''

    completed = _run_shell(
        body,
        {
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "TEST_UNIT_INPUT_OUTPUT": str(output_path),
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["schema"] == "target-blob-bootstrap-test.v1"
    assert output_path.stat().st_mode & 0o777 == 0o444
    # There is deliberately no active release or active-side bootstrap command
    # in this fixture; success proves the reviewed target Git blob is sufficient.
    assert not (tmp_path / "active").exists()


def test_root_config_seal_rejects_owner_payload_drift_before_chown(
    tmp_path: Path,
) -> None:
    revision = "c" * 40
    releases = (tmp_path / "releases").resolve()
    release = releases / f".hermes-agent-{revision[:12]}.tmp.123"
    config = release / "ops/muncho/runtime/dependencies/agent-browser.json"
    config.parent.mkdir(parents=True)
    config.write_bytes(b'{"proxy":"forbidden"}\n')
    config.chmod(0o444)
    identity_before = config.stat()
    owner = subprocess.check_output(["id", "-un"], text=True).strip()
    id_path = shutil.which("id") or "/usr/bin/id"
    body = f'''
OWNER={json.dumps(owner)}
SYSTEM_ID={json.dumps(id_path)}
SYSTEM_PYTHON={json.dumps(sys.executable)}
RELEASES={json.dumps(str(releases))}
RUNTIME_CONFIG_ROOT_UID={os.getuid()}
RUNTIME_CONFIG_ROOT_GID={os.getgid()}
seal_agent_browser_config {json.dumps(str(release))} {revision}
'''

    completed = _run_shell(body, {})

    assert completed.returncode != 0
    assert "runtime_config_prepared_identity_invalid" in completed.stderr
    identity_after = config.stat()
    assert (identity_after.st_uid, identity_after.st_gid) == (
        identity_before.st_uid,
        identity_before.st_gid,
    )
    assert config.read_bytes() == b'{"proxy":"forbidden"}\n'


def test_owner_config_is_mechanically_sealed_before_manifest_build(
    tmp_path: Path,
) -> None:
    revision = "d" * 40
    releases = (tmp_path / "releases").resolve()
    release = releases / f".hermes-agent-{revision[:12]}.tmp.456"
    config = release / "ops/muncho/runtime/dependencies/agent-browser.json"
    config.parent.mkdir(parents=True)
    config.write_bytes(b"{}\n")
    config.chmod(0o444)
    for path in (
        releases,
        release,
        release / "ops",
        release / "ops/muncho",
        release / "ops/muncho/runtime",
        release / "ops/muncho/runtime/dependencies",
        config,
    ):
        os.chown(path, os.geteuid(), os.getegid())
    owner = subprocess.check_output(["id", "-un"], text=True).strip()
    id_path = shutil.which("id") or "/usr/bin/id"
    body = f'''
OWNER={json.dumps(owner)}
SYSTEM_ID={json.dumps(id_path)}
SYSTEM_PYTHON={json.dumps(sys.executable)}
RELEASES={json.dumps(str(releases))}
RUNTIME_CONFIG_ROOT_UID={os.getuid()}
RUNTIME_CONFIG_ROOT_GID={os.getgid()}
seal_agent_browser_config {json.dumps(str(release))} {revision}
'''

    completed = _run_shell(body, {})

    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(completed.stdout)
    assert receipt["schema"] == "muncho-production-runtime-config-root-seal.v1"
    assert receipt["sha256"] == hashlib.sha256(b"{}\n").hexdigest()
    assert receipt["mode"] == "0444"
    assert config.read_bytes() == b"{}\n"
