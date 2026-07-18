from __future__ import annotations

import subprocess
import sys
import traceback
from pathlib import Path
from typing import Callable

import pytest

from scripts.canary import direct_iam_identity_author as direct_iam
from scripts.canary import owner_gate_firewall_readiness as firewall
from scripts.canary import owner_gate_foundation_apply as foundation_apply
from scripts.canary import owner_gate_host_identity as host_identity
from scripts.canary import owner_gate_outer_stage0 as outer_stage0
from scripts.canary import owner_gate_owner_reauth as owner_reauth
from scripts.canary import owner_gate_pre_foundation as pre_foundation
from scripts.canary import owner_gate_project_ancestry as ancestry
from scripts.canary import passkey_v2_service as passkey_service
from scripts.canary import source_artifact_publication as source_publication
from scripts.canary import storage_growth_trusted_collector as trusted_collector
from scripts.canary import trusted_signer_provisioning as signer_provisioning
from scripts.canary import trusted_signer_stage0 as signer_stage0


@pytest.mark.parametrize(
    ("author", "error_type"),
    [
        (direct_iam._error, direct_iam.DirectIamIdentityAuthorError),
        (host_identity._error, host_identity.OwnerGateHostIdentityError),
        (ancestry._error, ancestry.OwnerGateProjectAncestryError),
        (outer_stage0._error, outer_stage0.OwnerGateOuterStage0Error),
        (owner_reauth._error, owner_reauth.OwnerGateOwnerReauthError),
        (
            pre_foundation._error,
            pre_foundation.OwnerGatePreFoundationError,
        ),
        (
            foundation_apply._error,
            foundation_apply.OwnerGateFoundationApplyError,
        ),
        (
            signer_provisioning._stable_error,
            signer_provisioning.TrustedSignerProvisioningError,
        ),
        (signer_stage0._error, signer_stage0.TrustedSignerStage0Error),
        (
            source_publication._error,
            source_publication._SourceArtifactPublicationError,
        ),
    ],
)
def test_public_boundary_errors_render_only_stable_codes(
    author,
    error_type,
) -> None:
    sensitive_detail = "https://token.invalid/private?credential=do-not-render"
    nested = RuntimeError(sensitive_detail)

    with pytest.raises(error_type, match="^stable_boundary_failure$") as captured:
        author("stable_boundary_failure", nested)

    rendered = "".join(
        traceback.format_exception(
            captured.type,
            captured.value,
            captured.tb,
        )
    )
    assert rendered.rstrip().endswith(": stable_boundary_failure")
    assert sensitive_detail not in rendered
    assert captured.value.__cause__ is None
    assert captured.value.__suppress_context__ is True


def test_direct_firewall_boundary_suppresses_decoder_context() -> None:
    sensitive_detail = "https://token.invalid/private?credential=do-not-render"
    poisoned = b"\xff" + sensitive_detail.encode("ascii")

    with pytest.raises(
        firewall.OwnerGateFirewallError,
        match="^owner_gate_firewall_ruleset_invalid$",
    ) as captured:
        firewall.validate_live_ruleset(poisoned)

    rendered = "".join(
        traceback.format_exception(
            captured.type,
            captured.value,
            captured.tb,
        )
    )
    assert rendered.rstrip().endswith(": owner_gate_firewall_ruleset_invalid")
    assert sensitive_detail not in rendered
    assert captured.value.__cause__ is None
    assert captured.value.__suppress_context__ is True


@pytest.mark.parametrize(
    ("reader", "error_type", "stable_code"),
    [
        (
            passkey_service._read_regular_file,
            passkey_service.PasskeyV2ServiceError,
            "passkey_v2_service_file_unavailable",
        ),
        (
            trusted_collector._read_regular_file,
            trusted_collector.TrustedObservationError,
            "trusted_observation_file_unavailable",
        ),
    ],
)
def test_file_boundaries_suppress_sensitive_os_error_context(
    reader: Callable[..., bytes | tuple[bytes, object]],
    error_type: type[Exception],
    stable_code: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"fixed")
    sensitive_detail = "https://token.invalid/private?credential=do-not-render"

    def fail_open(*_args, **_kwargs) -> int:
        raise OSError(sensitive_detail)

    with monkeypatch.context() as scoped:
        scoped.setattr(passkey_service.os, "open", fail_open)
        with pytest.raises(error_type, match=f"^{stable_code}$") as captured:
            reader(source, maximum=1024)

    rendered = "".join(
        traceback.format_exception(
            captured.type,
            captured.value,
            captured.tb,
        )
    )
    assert rendered.rstrip().endswith(f": {stable_code}")
    assert sensitive_detail not in rendered
    assert captured.value.__cause__ is None
    assert captured.value.__suppress_context__ is True


@pytest.mark.parametrize(
    ("script", "arguments", "error_name", "stable_prefix"),
    [
        (
            "owner_gate_outer_stage0.py",
            (
                "seal",
                "--incoming",
                "{sensitive}",
                "--expected-manifest-sha256",
                "0" * 64,
            ),
            "OwnerGateOuterStage0Error",
            "owner_gate_outer_stage0_",
        ),
        (
            "owner_gate_firewall_readiness.py",
            (
                "receipt",
                "--rules",
                "{sensitive}",
                "--receipt",
                "/tmp/unused-owner-gate-firewall-receipt.json",
            ),
            "OwnerGateFirewallError",
            "owner_gate_firewall_",
        ),
    ],
)
def test_public_entrypoints_do_not_render_sensitive_arguments(
    script: str,
    arguments: tuple[str, ...],
    error_name: str,
    stable_prefix: str,
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[3]
    sensitive = tmp_path / "credential=do-not-render"
    rendered_arguments = tuple(
        str(sensitive) if value == "{sensitive}" else value for value in arguments
    )

    completed = subprocess.run(
        (
            sys.executable,
            str(repository / "scripts/canary" / script),
            *rendered_arguments,
        ),
        cwd=repository,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15,
        check=False,
    )

    rendered = completed.stdout + completed.stderr
    assert completed.returncode != 0
    assert str(sensitive) not in rendered
    assert f"{error_name}: {stable_prefix}" in completed.stderr
