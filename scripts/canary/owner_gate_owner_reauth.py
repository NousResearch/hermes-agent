#!/usr/bin/env python3
"""Produce and validate one secret-free owner gcloud reauthentication receipt.

The producer accepts only Hermes' fully pinned gcloud runtime and configuration
protocols.  It runs one fixed ``gcloud auth login ACCOUNT --force`` flow and a
fixed read-only project probe through the sealed SDK/Python command prefix.  A
closed environment excludes ambient proxies, Python injection, custom CA
paths, and token logging.  No access-token command is present and no
credential output is captured or stored.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Protocol, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from scripts.canary import full_canary_owner_launcher as launcher
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_trust as release_trust


RECEIPT_SCHEMA = "muncho-owner-gate-owner-reauthentication-receipt.v1"
RECEIPT_PURPOSE = "muncho_owner_gate_fresh_interactive_gcloud_owner_reauth"
SIGNATURE_DOMAIN = b"muncho-owner-gate/owner-reauthentication/v1\x00"
OWNER_ACCOUNT = "lomliev@adventico.com"
PROJECT = foundation.PROJECT
ZONE = foundation.ZONE
GCLOUD_CONFIGURATION = "adventico-ai-platform-admin"
MAX_RECEIPT_TTL_SECONDS = 900
MAX_INTERACTIVE_DURATION_SECONDS = 900
INTERACTIVE_TIMEOUT_SECONDS = 930.0
CAPTURE_TIMEOUT_SECONDS = 30.0
MAX_CAPTURE_BYTES = 1024 * 1024

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NUMERIC_ID = re.compile(r"^[1-9][0-9]{5,30}$")
_B64URL = re.compile(r"^[A-Za-z0-9_-]{86}$")
_SDK_ROOT = re.compile(r"^/[^\x00\r\n]{1,1023}/google-cloud-sdk$")
_BODY_FIELDS = frozenset({
    "schema",
    "purpose",
    "trusted_runtime_identity",
    "interactive_reauthentication",
    "authenticated_probe",
    "issued_at_unix",
    "expires_at_unix",
    "signer_key_id",
})
_FIELDS = _BODY_FIELDS | frozenset({
    "owner_reauthentication_receipt_sha256",
    "signature_ed25519_b64url",
})
_RUNTIME_FIELDS = frozenset({
    "release_revision",
    "sealed_runtime_identity_sha256",
    "command_prefix_sha256",
    "python_executable_sha256",
    "gcloud_module_sha256",
    "sdk_root",
    "sdk_python_config_identity_sha256",
    "closed_environment_sha256",
    "configuration",
    "account",
    "project",
    "zone",
})
_SEALED_RUNTIME_FIELDS = frozenset({
    "schema",
    "release_sha",
    "command_prefix_sha256",
    "sdk_tree_entries",
    "sdk_tree_bytes",
    "sdk_tree_sha256",
    "sdk_publication_tree_entries",
    "sdk_publication_tree_bytes",
    "sdk_publication_tree_sha256",
    "sdk_publication_intent_sha256",
    "python_version",
    "python_tree_entries",
    "python_tree_bytes",
    "python_tree_sha256",
    "owner_support_tree_entries",
    "owner_support_tree_bytes",
    "owner_support_tree_sha256",
    "owner_support_manifest_sha256",
    "owner_support_source_tree_oid",
    "bootstrap_receipt_file_sha256",
    "identity_sha256",
})
_REAUTH_FIELDS = frozenset({
    "method",
    "started_at_unix",
    "completed_at_unix",
    "command_sha256",
    "interactive_tty_verified",
    "access_token_requested",
    "credential_material_captured",
})
_PROBE_FIELDS = frozenset({
    "command_sha256",
    "output_sha256",
    "project_id",
    "project_number",
})


class OwnerGateOwnerReauthError(RuntimeError):
    """Stable, secret-free owner reauthentication failure."""


def _error(code: str, exc: BaseException | None = None) -> NoReturn:
    del exc
    raise OwnerGateOwnerReauthError(code) from None


def _canonical(value: Any) -> bytes:
    try:
        return foundation.canonical_json_bytes(value)
    except foundation.OwnerGateFoundationError as exc:
        _error("owner_gate_owner_reauth_json_invalid", exc)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256(_canonical(value))


def _strict_mapping(
    value: Any,
    fields: frozenset[str],
    *,
    code: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != fields:
        _error(code)
    return value


def _public_key_id(public_key: Ed25519PublicKey) -> str:
    if not isinstance(public_key, Ed25519PublicKey):
        _error("owner_gate_owner_reauth_signer_invalid")
    return _sha256(public_key.public_bytes_raw())


def _require_pinned_key(public_key: Ed25519PublicKey) -> str:
    key_id = _public_key_id(public_key)
    if (
        _SHA256.fullmatch(
            release_trust.PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256 or ""
        )
        is None
        or key_id != release_trust.PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256
    ):
        _error("owner_gate_owner_reauth_signer_not_pinned")
    return key_id


def _decode_signature(value: Any) -> bytes:
    if not isinstance(value, str) or _B64URL.fullmatch(value) is None:
        _error("owner_gate_owner_reauth_signature_invalid")
    try:
        raw = base64.b64decode(
            value + "=" * (-len(value) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (TypeError, ValueError) as exc:
        _error("owner_gate_owner_reauth_signature_invalid", exc)
    if (
        len(raw) != 64
        or base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
        != value
    ):
        _error("owner_gate_owner_reauth_signature_invalid")
    return raw


def _pinned_file_sha256(path: str, *, executable: bool) -> str:
    if not os.path.isabs(path) or os.path.realpath(path) != path:
        _error("owner_gate_owner_reauth_runtime_invalid")
    descriptor: int | None = None
    try:
        before = os.lstat(path)
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or opened.st_uid not in {0, os.getuid()}  # windows-footgun: ok — POSIX owner boundary
            or opened.st_mode & 0o022
            or (executable and not opened.st_mode & 0o100)
            or opened.st_size <= 0
            or opened.st_size > 32 * 1024 * 1024
        ):
            _error("owner_gate_owner_reauth_runtime_invalid")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > opened.st_size:
                _error("owner_gate_owner_reauth_runtime_changed")
            digest.update(chunk)
        after = os.fstat(descriptor)
        if total != opened.st_size or (
            opened.st_mode,
            opened.st_uid,
            opened.st_gid,
            opened.st_dev,
            opened.st_ino,
            opened.st_nlink,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) != (
            after.st_mode,
            after.st_uid,
            after.st_gid,
            after.st_dev,
            after.st_ino,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            _error("owner_gate_owner_reauth_runtime_changed")
        return digest.hexdigest()
    except OSError as exc:
        _error("owner_gate_owner_reauth_runtime_invalid", exc)
    finally:
        if descriptor is not None:
            os.close(descriptor)


@dataclass(frozen=True)
class CapturedCommand:
    returncode: int
    stdout: bytes
    stderr: bytes


class OwnerReauthRunner(Protocol):
    def interactive_tty_verified(self) -> bool: ...

    def run_interactive(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        timeout_seconds: float,
    ) -> int: ...

    def run_capture(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        timeout_seconds: float,
    ) -> CapturedCommand: ...


class SubprocessOwnerReauthRunner:
    """Bounded runner used only by the fixed producer."""

    def interactive_tty_verified(self) -> bool:
        return sys.stdin.isatty() and sys.stdout.isatty()

    def run_interactive(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        timeout_seconds: float,
    ) -> int:
        try:
            completed = subprocess.run(
                tuple(argv),
                check=False,
                env=dict(env),
                timeout=timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            _error("owner_gate_owner_reauth_interactive_failed", exc)
        return completed.returncode

    def run_capture(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        timeout_seconds: float,
    ) -> CapturedCommand:
        try:
            completed = subprocess.run(
                tuple(argv),
                check=False,
                capture_output=True,
                env=dict(env),
                timeout=timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            _error("owner_gate_owner_reauth_probe_failed", exc)
        return CapturedCommand(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


def _trusted_snapshot(
    executable: launcher.StableExecutable,
    configuration: launcher.StableGcloudConfiguration,
) -> tuple[Mapping[str, Any], tuple[str, ...], Mapping[str, str]]:
    try:
        configuration.assert_stable()
        prefix = tuple(executable.trusted_command_prefix())
        account = configuration.account
        isolation = tuple(launcher._GCLOUD_PYTHON_ISOLATION_ARGS)
        if (
            account != OWNER_ACCOUNT
            or len(prefix) != len(isolation) + 2
            or prefix[1:-1] != isolation
            or not os.path.isabs(prefix[0])
            or not os.path.isabs(prefix[-1])
            or os.path.basename(prefix[-1]) != "gcloud.py"
        ):
            _error("owner_gate_owner_reauth_runtime_invalid")
        python_sha256 = _pinned_file_sha256(prefix[0], executable=True)
        module_sha256 = _pinned_file_sha256(prefix[-1], executable=False)
        sdk_root = str(Path(prefix[-1]).parent.parent)
        if _SDK_ROOT.fullmatch(sdk_root) is None:
            _error("owner_gate_owner_reauth_runtime_invalid")
        environment = dict(
            launcher._owner_gcloud_environment(configuration, prefix[0])
        )
        environment["CLOUDSDK_CORE_DISABLE_PROMPTS"] = "0"
        forbidden = {
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "no_proxy",
            "PYTHONPATH",
            "PYTHONHOME",
            "REQUESTS_CA_BUNDLE",
            "SSL_CERT_FILE",
        }
        if forbidden & set(environment):
            _error("owner_gate_owner_reauth_environment_invalid")
        config_values = dict(configuration.environment_values())
        config_identity = {
            "account": account,
            "project": PROJECT,
            "zone": ZONE,
            "configuration": GCLOUD_CONFIGURATION,
            "environment": config_values,
        }
        identity = {
            "command_prefix_sha256": _sha256_json(list(prefix)),
            "python_executable_sha256": python_sha256,
            "gcloud_module_sha256": module_sha256,
            "sdk_root": sdk_root,
            "sdk_python_config_identity_sha256": _sha256_json(
                {
                    "prefix": list(prefix),
                    "python_sha256": python_sha256,
                    "gcloud_module_sha256": module_sha256,
                    "config": config_identity,
                }
            ),
            "closed_environment_sha256": _sha256_json(environment),
            "configuration": GCLOUD_CONFIGURATION,
            "account": account,
            "project": PROJECT,
            "zone": ZONE,
        }
        configuration.assert_stable()
        if tuple(executable.trusted_command_prefix()) != prefix:
            _error("owner_gate_owner_reauth_runtime_changed")
        return identity, prefix, environment
    except launcher.OwnerLauncherError as exc:
        _error("owner_gate_owner_reauth_runtime_invalid", exc)


def _commands(prefix: Sequence[str]) -> Mapping[str, tuple[str, ...]]:
    return {
        "interactive_reauthentication": (
            *prefix,
            "auth",
            "login",
            OWNER_ACCOUNT,
            "--force",
            "--brief",
            f"--configuration={GCLOUD_CONFIGURATION}",
        ),
        "authenticated_probe": (
            *prefix,
            "projects",
            "describe",
            PROJECT,
            f"--account={OWNER_ACCOUNT}",
            f"--configuration={GCLOUD_CONFIGURATION}",
            "--format=json",
        ),
    }


def _validate_sealed_runtime_identity(
    value: Any,
    *,
    expected_release_revision: str,
    prefix: Sequence[str],
) -> Mapping[str, Any]:
    identity = _strict_mapping(
        value,
        _SEALED_RUNTIME_FIELDS,
        code="owner_gate_owner_reauth_runtime_invalid",
    )
    unsigned = dict(identity)
    digest = unsigned.pop("identity_sha256", None)
    count_fields = (
        "sdk_tree_entries",
        "sdk_tree_bytes",
        "sdk_publication_tree_entries",
        "sdk_publication_tree_bytes",
        "python_tree_entries",
        "python_tree_bytes",
        "owner_support_tree_entries",
        "owner_support_tree_bytes",
    )
    sha_fields = (
        "sdk_tree_sha256",
        "sdk_publication_tree_sha256",
        "sdk_publication_intent_sha256",
        "python_tree_sha256",
        "owner_support_tree_sha256",
        "owner_support_manifest_sha256",
        "bootstrap_receipt_file_sha256",
    )
    source_tree_oid = identity.get("owner_support_source_tree_oid")
    if re.fullmatch(r"[0-9a-f]{40}", expected_release_revision or "") is None:
        _error("owner_gate_owner_reauth_runtime_invalid")
    if (
        identity.get("schema")
        != "muncho-owner-sealed-gcloud-runtime-identity.v1"
        or identity.get("release_sha") != expected_release_revision
        or identity.get("command_prefix_sha256")
        != _sha256_json(list(prefix))
        or any(type(identity.get(field)) is not int or identity[field] <= 0 for field in count_fields)
        or any(
            type(identity.get(field)) is not str
            or _SHA256.fullmatch(identity[field]) is None
            for field in sha_fields
        )
        or not isinstance(source_tree_oid, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_tree_oid) is None
        or not isinstance(identity.get("python_version"), str)
        or not identity["python_version"]
        or digest != _sha256_json(unsigned)
    ):
        _error("owner_gate_owner_reauth_runtime_invalid")
    return dict(identity)


def _capture_json(
    runner: OwnerReauthRunner,
    argv: Sequence[str],
    *,
    env: Mapping[str, str],
) -> tuple[Mapping[str, Any], bytes]:
    result = runner.run_capture(
        argv,
        env=env,
        timeout_seconds=CAPTURE_TIMEOUT_SECONDS,
    )
    if (
        type(result.returncode) is not int
        or result.returncode != 0
        or type(result.stdout) is not bytes
        or type(result.stderr) is not bytes
        or not result.stdout
        or len(result.stdout) > MAX_CAPTURE_BYTES
        or len(result.stderr) > MAX_CAPTURE_BYTES
    ):
        _error("owner_gate_owner_reauth_probe_failed")
    try:
        value = json.loads(result.stdout.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        _error("owner_gate_owner_reauth_probe_failed", exc)
    if not isinstance(value, Mapping):
        _error("owner_gate_owner_reauth_probe_failed")
    return value, result.stdout


def _produce_owner_reauth_receipt_with_runtime(
    *,
    runner: OwnerReauthRunner,
    private_key: Ed25519PrivateKey,
    now_unix: Callable[[], int],
    gcloud_executable: launcher.StableExecutable,
    gcloud_configuration: launcher.StableGcloudConfiguration,
    expected_release_revision: str,
    sealed_runtime_snapshot: Callable[[], Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Private test seam; the public boundary admits exact pinned classes."""

    if not isinstance(private_key, Ed25519PrivateKey):
        _error("owner_gate_owner_reauth_signer_invalid")
    signer_key_id = _require_pinned_key(private_key.public_key())
    before, prefix, environment = _trusted_snapshot(
        gcloud_executable,
        gcloud_configuration,
    )
    sealed_before = _validate_sealed_runtime_identity(
        sealed_runtime_snapshot(),
        expected_release_revision=expected_release_revision,
        prefix=prefix,
    )
    before = {
        **before,
        "release_revision": expected_release_revision,
        "sealed_runtime_identity_sha256": sealed_before["identity_sha256"],
    }
    commands = _commands(prefix)
    if not runner.interactive_tty_verified():
        _error("owner_gate_owner_reauth_interactive_tty_required")
    started = now_unix()
    if type(started) is not int or started <= 0:
        _error("owner_gate_owner_reauth_time_invalid")
    returncode = runner.run_interactive(
        commands["interactive_reauthentication"],
        env=environment,
        timeout_seconds=INTERACTIVE_TIMEOUT_SECONDS,
    )
    completed = now_unix()
    if (
        returncode != 0
        or type(completed) is not int
        or completed < started
        or completed - started > MAX_INTERACTIVE_DURATION_SECONDS
    ):
        _error("owner_gate_owner_reauth_interactive_failed")
    probe, probe_raw = _capture_json(
        runner,
        commands["authenticated_probe"],
        env=environment,
    )
    after, after_prefix, after_environment = _trusted_snapshot(
        gcloud_executable,
        gcloud_configuration,
    )
    sealed_after = _validate_sealed_runtime_identity(
        sealed_runtime_snapshot(),
        expected_release_revision=expected_release_revision,
        prefix=after_prefix,
    )
    after = {
        **after,
        "release_revision": expected_release_revision,
        "sealed_runtime_identity_sha256": sealed_after["identity_sha256"],
    }
    issued = now_unix()
    project_number = str(probe.get("projectNumber", ""))
    if (
        before != after
        or sealed_before != sealed_after
        or prefix != after_prefix
        or environment != after_environment
        or type(issued) is not int
        or issued < completed
        or probe.get("projectId") != PROJECT
        or _NUMERIC_ID.fullmatch(project_number) is None
    ):
        _error("owner_gate_owner_reauth_runtime_changed")
    body = {
        "schema": RECEIPT_SCHEMA,
        "purpose": RECEIPT_PURPOSE,
        "trusted_runtime_identity": before,
        "interactive_reauthentication": {
            "method": "gcloud_auth_login_force_interactive",
            "started_at_unix": started,
            "completed_at_unix": completed,
            "command_sha256": _sha256_json(
                list(commands["interactive_reauthentication"])
            ),
            "interactive_tty_verified": True,
            "access_token_requested": False,
            "credential_material_captured": False,
        },
        "authenticated_probe": {
            "command_sha256": _sha256_json(
                list(commands["authenticated_probe"])
            ),
            "output_sha256": _sha256(probe_raw),
            "project_id": PROJECT,
            "project_number": project_number,
        },
        "issued_at_unix": issued,
        "expires_at_unix": issued + MAX_RECEIPT_TTL_SECONDS,
        "signer_key_id": signer_key_id,
    }
    return _sign_owner_reauth_receipt(body, private_key=private_key)


def produce_owner_reauth_receipt(
    *,
    runner: OwnerReauthRunner,
    private_key: Ed25519PrivateKey,
    now_unix: Callable[[], int],
    gcloud_executable: launcher.TrustedGcloudExecutable,
    gcloud_configuration: launcher.PinnedGcloudConfiguration,
    expected_release_revision: str,
) -> Mapping[str, Any]:
    """Run fresh owner reauth only through the exact production pin classes."""

    if (
        type(gcloud_executable) is not launcher.TrustedGcloudExecutable
        or type(gcloud_configuration) is not launcher.PinnedGcloudConfiguration
        or re.fullmatch(r"[0-9a-f]{40}", expected_release_revision or "")
        is None
    ):
        _error("owner_gate_owner_reauth_runtime_invalid")

    def sealed_snapshot() -> Mapping[str, Any]:
        try:
            return gcloud_executable.sealed_runtime_identity(
                expected_release_sha=expected_release_revision,
            )
        except launcher.OwnerLauncherError as exc:
            _error("owner_gate_owner_reauth_runtime_invalid", exc)

    return _produce_owner_reauth_receipt_with_runtime(
        runner=runner,
        private_key=private_key,
        now_unix=now_unix,
        gcloud_executable=gcloud_executable,
        gcloud_configuration=gcloud_configuration,
        expected_release_revision=expected_release_revision,
        sealed_runtime_snapshot=sealed_snapshot,
    )


def _validate_body(value: Any, *, now_unix: int | None) -> Mapping[str, Any]:
    body = _strict_mapping(
        value,
        _BODY_FIELDS,
        code="owner_gate_owner_reauth_receipt_invalid",
    )
    runtime = _strict_mapping(
        body.get("trusted_runtime_identity"),
        _RUNTIME_FIELDS,
        code="owner_gate_owner_reauth_receipt_invalid",
    )
    reauth = _strict_mapping(
        body.get("interactive_reauthentication"),
        _REAUTH_FIELDS,
        code="owner_gate_owner_reauth_receipt_invalid",
    )
    probe = _strict_mapping(
        body.get("authenticated_probe"),
        _PROBE_FIELDS,
        code="owner_gate_owner_reauth_receipt_invalid",
    )
    issued = body.get("issued_at_unix")
    expires = body.get("expires_at_unix")
    started = reauth.get("started_at_unix")
    completed = reauth.get("completed_at_unix")
    if (
        body.get("schema") != RECEIPT_SCHEMA
        or body.get("purpose") != RECEIPT_PURPOSE
        or any(
            _SHA256.fullmatch(str(runtime.get(field, ""))) is None
            for field in (
                "command_prefix_sha256",
                "python_executable_sha256",
                "gcloud_module_sha256",
                "sdk_python_config_identity_sha256",
                "closed_environment_sha256",
            )
        )
        or re.fullmatch(
            r"[0-9a-f]{40}",
            str(runtime.get("release_revision", "")),
        )
        is None
        or _SHA256.fullmatch(
            str(runtime.get("sealed_runtime_identity_sha256", ""))
        )
        is None
        or _SDK_ROOT.fullmatch(str(runtime.get("sdk_root", ""))) is None
        or runtime.get("configuration") != GCLOUD_CONFIGURATION
        or runtime.get("account") != OWNER_ACCOUNT
        or runtime.get("project") != PROJECT
        or runtime.get("zone") != ZONE
        or reauth.get("method") != "gcloud_auth_login_force_interactive"
        or type(started) is not int
        or type(completed) is not int
        or started <= 0
        or completed < started
        or completed - started > MAX_INTERACTIVE_DURATION_SECONDS
        or _SHA256.fullmatch(str(reauth.get("command_sha256", ""))) is None
        or reauth.get("interactive_tty_verified") is not True
        or reauth.get("access_token_requested") is not False
        or reauth.get("credential_material_captured") is not False
        or _SHA256.fullmatch(str(probe.get("command_sha256", ""))) is None
        or _SHA256.fullmatch(str(probe.get("output_sha256", ""))) is None
        or probe.get("project_id") != PROJECT
        or _NUMERIC_ID.fullmatch(str(probe.get("project_number", "")))
        is None
        or type(issued) is not int
        or type(expires) is not int
        or issued < completed
        or expires <= issued
        or expires - issued > MAX_RECEIPT_TTL_SECONDS
        or _SHA256.fullmatch(str(body.get("signer_key_id", ""))) is None
    ):
        _error("owner_gate_owner_reauth_receipt_invalid")
    if now_unix is not None and (
        type(now_unix) is not int
        or now_unix < issued
        or now_unix > expires
    ):
        _error("owner_gate_owner_reauth_receipt_expired")
    return dict(body)


def _sign_owner_reauth_receipt(
    body: Mapping[str, Any],
    *,
    private_key: Ed25519PrivateKey,
) -> Mapping[str, Any]:
    checked = _validate_body(body, now_unix=body.get("issued_at_unix"))
    if not isinstance(private_key, Ed25519PrivateKey):
        _error("owner_gate_owner_reauth_signer_invalid")
    key_id = _require_pinned_key(private_key.public_key())
    if checked["signer_key_id"] != key_id:
        _error("owner_gate_owner_reauth_signer_invalid")
    digest = _sha256_json(checked)
    signed_payload = {
        **checked,
        "owner_reauthentication_receipt_sha256": digest,
    }
    signature = private_key.sign(SIGNATURE_DOMAIN + _canonical(signed_payload))
    if len(signature) != 64:
        _error("owner_gate_owner_reauth_signature_invalid")
    return {
        **signed_payload,
        "signature_ed25519_b64url": base64.urlsafe_b64encode(signature)
        .rstrip(b"=")
        .decode("ascii"),
    }


def validate_owner_reauth_receipt(
    value: Any,
    *,
    public_key: Ed25519PublicKey,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    receipt = _strict_mapping(
        value,
        _FIELDS,
        code="owner_gate_owner_reauth_receipt_invalid",
    )
    body = {
        key: item
        for key, item in receipt.items()
        if key
        not in {
            "owner_reauthentication_receipt_sha256",
            "signature_ed25519_b64url",
        }
    }
    checked = _validate_body(body, now_unix=now_unix)
    key_id = _require_pinned_key(public_key)
    signed_payload = {
        **checked,
        "owner_reauthentication_receipt_sha256": receipt.get(
            "owner_reauthentication_receipt_sha256"
        ),
    }
    if (
        checked["signer_key_id"] != key_id
        or receipt.get("owner_reauthentication_receipt_sha256")
        != _sha256_json(checked)
    ):
        _error("owner_gate_owner_reauth_receipt_invalid")
    try:
        public_key.verify(
            _decode_signature(receipt.get("signature_ed25519_b64url")),
            SIGNATURE_DOMAIN + _canonical(signed_payload),
        )
    except InvalidSignature as exc:
        _error("owner_gate_owner_reauth_signature_invalid", exc)
    return dict(receipt)


def decode_canonical_owner_reauth_receipt(
    raw: bytes,
    **validation: Any,
) -> Mapping[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > MAX_CAPTURE_BYTES:
        _error("owner_gate_owner_reauth_receipt_invalid")
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        _error("owner_gate_owner_reauth_receipt_invalid", exc)
    if not isinstance(value, Mapping) or _canonical(value) != raw:
        _error("owner_gate_owner_reauth_receipt_invalid")
    return validate_owner_reauth_receipt(value, **validation)


__all__ = [
    "CapturedCommand",
    "GCLOUD_CONFIGURATION",
    "MAX_RECEIPT_TTL_SECONDS",
    "OWNER_ACCOUNT",
    "OwnerGateOwnerReauthError",
    "OwnerReauthRunner",
    "RECEIPT_SCHEMA",
    "SubprocessOwnerReauthRunner",
    "decode_canonical_owner_reauth_receipt",
    "produce_owner_reauth_receipt",
    "validate_owner_reauth_receipt",
]
