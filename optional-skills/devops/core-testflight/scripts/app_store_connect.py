"""Read-only App Store Connect verification through Core Secrets.

The module deliberately has no browser dependency.  Browser authentication is only
appropriate for operations that this read-only API surface cannot perform.
"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import os
import ssl
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

CORE_SECRETS_REFERENCE = "apple/app-store-connect"
APPLE_API_BASE = "https://api.appstoreconnect.apple.com/v1"
ELIGIBLE_TESTER_STATES = frozenset(
    {"ACCEPTED", "ENABLED", "INSTALLED", "INVITED", "REDEEMED"}
)
TESTABLE_BETA_STATES = frozenset(
    {"IN_BETA_TESTING", "READY_FOR_BETA_TESTING", "BETA_REVIEW_IN_PROGRESS"}
)


class AppStoreConnectError(RuntimeError):
    """A redacted, user-safe App Store Connect verification error."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class ApiClient:
    """Small authenticated GET client with no credential-bearing diagnostics."""

    def __init__(self, token: str, *, timeout: float = 30.0) -> None:
        self._token = token
        self._timeout = timeout
        self._context = ssl.create_default_context()

    def get(self, path: str) -> dict[str, Any]:
        url = f"{APPLE_API_BASE}{path}"
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._token}",
                "User-Agent": "Hermes-Core-TestFlight/3",
            },
        )
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(
                    request, context=self._context, timeout=self._timeout
                ) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                raise _apple_http_error(exc) from None
            except (
                urllib.error.URLError,
                http.client.RemoteDisconnected,
                TimeoutError,
                ConnectionError,
            ) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(attempt + 1)
        assert last_error is not None
        raise AppStoreConnectError(
            f"App Store Connect transport failure: {type(last_error).__name__}"
        ) from None


def _apple_http_error(exc: urllib.error.HTTPError) -> AppStoreConnectError:
    """Read only Apple's title/detail; never include a request or token."""

    try:
        payload = json.load(exc)
    except (ValueError, TypeError):
        return AppStoreConnectError(
            f"App Store Connect HTTP failure ({exc.code})", status=exc.code
        )
    errors = payload.get("errors") or []
    if not errors:
        return AppStoreConnectError(
            f"App Store Connect HTTP failure ({exc.code})", status=exc.code
        )
    first = errors[0]
    title = str(first.get("title") or "unknown error")
    detail = str(first.get("detail") or "no detail")
    return AppStoreConnectError(
        f"App Store Connect HTTP failure ({exc.code}): {title}: {detail}",
        status=exc.code,
    )


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _raw_ecdsa_signature(der: bytes) -> bytes:
    """Convert OpenSSL's DER ECDSA signature to JWT's raw r||s form."""

    if len(der) < 2 or der[0] != 0x30:
        raise AppStoreConnectError("Core Secrets App Store Connect key produced an invalid signature")
    index = 2
    if der[1] & 0x80:
        index += der[1] & 0x7F

    def read_integer(position: int) -> tuple[int, int]:
        if position >= len(der) or der[position] != 0x02:
            raise AppStoreConnectError("Core Secrets App Store Connect key produced an invalid signature")
        length = der[position + 1]
        position += 2
        if length & 0x80:
            count = length & 0x7F
            length = int.from_bytes(der[position : position + count], "big")
            position += count
        end = position + length
        return int.from_bytes(der[position:end], "big"), end

    r, index = read_integer(index)
    s, _ = read_integer(index)
    try:
        return r.to_bytes(32, "big") + s.to_bytes(32, "big")
    except OverflowError as exc:
        raise AppStoreConnectError("Core Secrets App Store Connect key produced an invalid signature") from exc


def make_jwt(fields: Mapping[str, str], *, now: int | None = None) -> str:
    """Create the short-lived Apple JWT entirely in memory."""

    required = {"key_id", "issuer_id", "private_key"}
    missing = sorted(required - set(fields))
    if missing:
        raise AppStoreConnectError(
            "Core Secrets App Store Connect reference is missing fields: "
            + ", ".join(missing)
        )
    header = _b64(
        json.dumps(
            {"alg": "ES256", "kid": fields["key_id"], "typ": "JWT"},
            separators=(",", ":"),
        ).encode()
    )
    issued_at = int(time.time()) if now is None else now
    payload = _b64(
        json.dumps(
            {
                "iss": fields["issuer_id"],
                "iat": issued_at - 30,
                "exp": issued_at + 900,
                "aud": "appstoreconnect-v1",
            },
            separators=(",", ":"),
        ).encode()
    )
    unsigned = f"{header}.{payload}".encode()
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, fields["private_key"].encode())
        os.close(write_fd)
        write_fd = -1
        signed = subprocess.run(
            ["/usr/bin/openssl", "dgst", "-sha256", "-sign", f"/dev/fd/{read_fd}"],
            input=unsigned,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=(read_fd,),
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AppStoreConnectError(
            "Core Secrets App Store Connect private key could not sign the API request"
        ) from exc
    finally:
        if write_fd != -1:
            os.close(write_fd)
        os.close(read_fd)
    return f"{header}.{payload}.{_b64(_raw_ecdsa_signature(signed))}"


def _one(items: list[dict[str, Any]], label: str) -> dict[str, Any]:
    if len(items) != 1:
        raise AppStoreConnectError(f"{label} match count was {len(items)}; expected exactly one")
    return items[0]


def _relationship_id(resource: Mapping[str, Any], name: str) -> str | None:
    relationship = resource.get("relationships", {}).get(name, {})
    data = relationship.get("data")
    if isinstance(data, dict):
        return data.get("id")
    return None


def verify_build(
    get: Callable[[str], dict[str, Any]],
    *,
    bundle_id: str,
    build_number: str,
    beta_group: str,
) -> dict[str, Any]:
    """Verify one exact build and return only non-sensitive evidence."""

    encoded_bundle = urllib.parse.quote(bundle_id, safe="")
    apps = get(f"/apps?filter%5BbundleId%5D={encoded_bundle}&limit=10").get("data", [])
    app = _one(apps, "app")
    app_id = app["id"]
    builds = get(
        f"/builds?filter%5Bapp%5D={urllib.parse.quote(app_id, safe='')}&limit=200"
    ).get("data", [])
    matching_builds = [
        item for item in builds if item.get("attributes", {}).get("version") == build_number
    ]
    build = _one(matching_builds, f"build {build_number}")
    build_id = build["id"]
    detail = get(
        f"/builds/{urllib.parse.quote(build_id, safe='')}?include=preReleaseVersion,buildBetaDetail"
    ).get("data", build)
    pre_release_id = _relationship_id(detail, "preReleaseVersion")
    if not pre_release_id:
        raise AppStoreConnectError(
            f"build {build_number} has no linked marketing version"
        )
    pre_release = get(
        f"/preReleaseVersions/{urllib.parse.quote(pre_release_id, safe='')}"
    ).get("data", {})
    marketing_version = pre_release.get("attributes", {}).get("version")
    if not marketing_version:
        raise AppStoreConnectError(
            f"build {build_number} has no readable marketing version"
        )
    attributes = detail.get("attributes", {})
    processing_state = attributes.get("processingState")
    if processing_state != "VALID":
        raise AppStoreConnectError(
            f"build {build_number} is not ready: processingState={processing_state}"
        )
    if attributes.get("expired") is True:
        raise AppStoreConnectError(f"build {build_number} is expired")
    compliance_complete = attributes.get("usesNonExemptEncryption") is False
    if not compliance_complete:
        raise AppStoreConnectError(f"build {build_number} has incomplete export compliance")

    beta_details = get(
        f"/buildBetaDetails?filter%5Bbuild%5D={urllib.parse.quote(build_id, safe='')}&limit=10"
    ).get("data", [])
    beta_detail = _one(beta_details, f"build {build_number} beta detail")
    beta_attributes = beta_detail.get("attributes", {})
    beta_state = (
        beta_attributes.get("betaState")
        or beta_attributes.get("internalBetaState")
        or beta_attributes.get("internalBuildState")
        or beta_attributes.get("externalBuildState")
    )
    if beta_state not in TESTABLE_BETA_STATES:
        raise AppStoreConnectError(
            f"build {build_number} is not testable: betaState={beta_state}"
        )

    groups = get(
        f"/betaGroups?filter%5Bapp%5D={urllib.parse.quote(app_id, safe='')}&limit=200"
    ).get("data", [])
    group = _one(
        [item for item in groups if item.get("attributes", {}).get("name") == beta_group],
        f"beta group {beta_group}",
    )
    group_id = group["id"]
    group_builds = get(
        f"/betaGroups/{urllib.parse.quote(group_id, safe='')}/builds?limit=200"
    ).get("data", [])
    group_member = any(item.get("id") == build_id for item in group_builds)
    if not group_member:
        raise AppStoreConnectError(
            f"build {build_number} is not assigned to beta group {beta_group}"
        )
    testers = get(
        f"/betaGroups/{urllib.parse.quote(group_id, safe='')}/betaTesters?limit=200"
    ).get("data", [])
    eligible_testers = sum(
        1
        for tester in testers
        if tester.get("attributes", {}).get("state") in ELIGIBLE_TESTER_STATES
    )
    if eligible_testers == 0:
        raise AppStoreConnectError(f"beta group {beta_group} has no eligible tester")

    return {
        "app_bundle_id": bundle_id,
        "app_record_present": True,
        "build_number": build_number,
        "marketing_version": marketing_version,
        "build_id": build_id,
        "processing_state": processing_state,
        "expired": False,
        "compliance_complete": compliance_complete,
        "beta_state": beta_state,
        "beta_group": beta_group,
        "group_member": True,
        "eligible_tester_count": eligible_testers,
        "verification": "API",
    }


def load_api_client(*, workstation: str = "WKST-01") -> ApiClient:
    """Resolve the established App Store Connect record through Core Secrets."""

    try:
        from accessible_core_secrets.client import SecretsServiceClient
        from core_secrets_keychain_client import (
            credential_file_read,
            principal_for_workstation,
        )
    except ImportError as exc:
        raise AppStoreConnectError(
            "Core Secrets helper is unavailable; browser login is not a substitute"
        ) from exc
    endpoint = os.environ.get(
        "CORE_SECRETS_URL", "http://ah-nas-01.taila0be49.ts.net:8787"
    )
    principal = principal_for_workstation(workstation)
    client = SecretsServiceClient(
        endpoint, credential_file_read(principal), timeout=15
    )
    try:
        fields = client.resolve(CORE_SECRETS_REFERENCE)
    except Exception as exc:  # SDK errors may contain provider-specific details.
        raise AppStoreConnectError(
            "Core Secrets could not resolve apple/app-store-connect"
        ) from exc
    return ApiClient(make_jwt(fields))


def verify_existing_build(
    *,
    bundle_id: str,
    build_number: str,
    beta_group: str,
) -> dict[str, Any]:
    """Perform the supported verification without consulting browser state."""

    api = load_api_client()
    return verify_build(
        api.get,
        bundle_id=bundle_id,
        build_number=build_number,
        beta_group=beta_group,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--build", required=True, dest="build_number")
    parser.add_argument("--beta-group", required=True)
    args = parser.parse_args(argv)
    try:
        result = verify_existing_build(
            bundle_id=args.bundle_id,
            build_number=args.build_number,
            beta_group=args.beta_group,
        )
    except AppStoreConnectError as exc:
        print(f"ERROR={exc}")
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
