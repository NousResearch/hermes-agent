"""Pre-build App Store Connect credential check for macOS notarization.

Runs ``xcrun notarytool history`` right after writing ``apple-api-key.p8`` so
revoked keys, missing legal agreements (HTTP 401/403), or malformed PEM files
fail the macOS build in seconds instead of after building and signing (~10 min).
Transient Apple 5xx / network / timeout blips emit a warning and exit 0 so
``Build and package`` keeps its full retry path and Gatekeeper verification.
"""
from __future__ import annotations

import argparse
import base64
import binascii
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path
import re
import subprocess
import sys

_PEM_BEGIN = "-----BEGIN " + "PRIVATE KEY-----"
_PEM_END = "-----END " + "PRIVATE KEY-----"
_PEM_BLOCK_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]+-----.*?-----END [A-Z0-9 ]+-----",
    re.DOTALL,
)
_HTTP_STATUS_RE = re.compile(
    r'\b(?:HTTP\s+status\s+code|"?statusCode"?)\s*:\s*(?:Optional\()?(\d{3})\b',
    re.IGNORECASE,
)
_TRANSIENT_TRANSPORT_RE = re.compile(
    r"\b(?:"
    r"NSURLErrorDomain"
    r"|kCFStreamErrorDomain"
    r"|The request timed out"
    r"|network connection was lost"
    r"|cannot find host"
    r"|cannot connect to host"
    r"|Internet connection appears to be offline"
    r")\b",
    re.IGNORECASE,
)
_DETERMINISTIC_LOCAL_AUTH_RE = re.compile(
    r"(?:"
    r"\bThe value\b[^\n]*\bis invalid for\s+'--(?:key|key-id|issuer)\b"
    r"|\bInvalid option:"
    r"|\b(?:private key|invalid PEM|ASN\.?1|CryptoKit)\b"
    r"|\bThe file\b[^\n]*couldn['\u2019]t be opened\b"
    r"|\b(?:Unauthenticated|Invalid credentials|required agreement is missing or has expired"
    r"|signed the necessary legal agreements)\b"
    r")",
    re.IGNORECASE,
)
_DEFAULT_TIMEOUT_SECONDS = 30.0

CommandRunner = Callable[[list[str], float], subprocess.CompletedProcess[str]]


class PreflightStatus(Enum):
    OK = "ok"
    AUTH_FAILED = "auth_failed"
    TRANSIENT = "transient"


@dataclass(frozen=True)
class PreflightResult:
    status: PreflightStatus
    message: str
    detail: str = ""


def _der_element(der: bytes, pos: int) -> tuple[int, int, int]:
    """Return the tag, content start and end of the definite-length DER element at *pos*."""
    if pos + 2 > len(der):
        raise ValueError("truncated DER header")
    tag, length, pos = der[pos], der[pos + 1], pos + 2
    if length & 0x80:
        count = length & 0x7F
        length, pos = int.from_bytes(der[pos : pos + count], "big"), pos + count
    if pos + length > len(der):
        raise ValueError("truncated DER element")
    return tag, pos, pos + length


def _is_pkcs8_private_key_info(der: bytes) -> bool:
    """SEQUENCE { INTEGER version, SEQUENCE algorithm, OCTET STRING privateKey, ... } filling *der* exactly."""
    try:
        tag, pos, end = _der_element(der, 0)
        if tag != 0x30 or end != len(der):
            return False
        tags = []
        while pos < end and len(tags) < 3:
            child, _start, pos = _der_element(der, pos)
            tags.append(child)
    except ValueError:
        return False
    return tags == [0x02, 0x30, 0x04]


def validate_p8_pem(pem_text: str) -> str | None:
    """Return an error message if *pem_text* is not a valid PKCS#8 PEM private key."""
    lines = [line.strip() for line in pem_text.splitlines() if line.strip()]
    if not lines:
        return "App Store Connect .p8 key file is empty"
    if lines.count(_PEM_BEGIN) != 1 or lines.count(_PEM_END) != 1:
        return (
            "App Store Connect .p8 key file must contain exactly one PKCS#8 PEM block "
            f"({_PEM_BEGIN} / {_PEM_END})"
        )
    begin_idx = lines.index(_PEM_BEGIN)
    end_idx = lines.index(_PEM_END)
    if end_idx <= begin_idx + 1:
        return "App Store Connect .p8 key file has an empty or inverted PEM block"
    body_lines = lines[begin_idx + 1 : end_idx]
    if any(line.startswith("-----") for line in body_lines):
        return "App Store Connect .p8 key file contains nested PEM markers"
    try:
        der = base64.b64decode("".join(body_lines), validate=True)
    except (binascii.Error, ValueError):
        return "App Store Connect .p8 key file contains invalid base64 PEM data"
    if not _is_pkcs8_private_key_info(der):
        return "App Store Connect .p8 key file does not contain a complete DER-encoded PKCS#8 key"
    return None


def redact_secrets(text: str, secrets: Iterable[str]) -> str:
    """Remove secret tokens and PEM blocks before logging diagnostic output."""
    cleaned = _PEM_BLOCK_RE.sub("[REDACTED PEM]", text)
    for secret in secrets:
        if secret:
            cleaned = cleaned.replace(secret, "***")
    return cleaned.strip()


def classify_notarytool_failure(output: str) -> tuple[PreflightStatus, str]:
    """Classify non-zero ``notarytool history`` output as auth failure vs transient."""
    status_match = _HTTP_STATUS_RE.search(output)
    if status_match is not None:
        code = int(status_match.group(1))
        if code == 401:
            return (
                PreflightStatus.AUTH_FAILED,
                "Apple rejected App Store Connect credentials (HTTP 401 Unauthenticated) — "
                "verify the Team API key in release-signing is active and matches "
                "APPLE_API_KEY_ID and APPLE_API_ISSUER",
            )
        if code == 403:
            return (
                PreflightStatus.AUTH_FAILED,
                "Apple rejected App Store Connect credentials (HTTP 403 Forbidden) — "
                "check the API key role (Developer/App Manager/Admin) and any pending "
                "Apple Developer Program legal agreements",
            )
        return (
            PreflightStatus.TRANSIENT,
            f"notarytool credential preflight received transient HTTP {code} from Apple; "
            "continuing to Build and package",
        )
    if _TRANSIENT_TRANSPORT_RE.search(output):
        return (
            PreflightStatus.TRANSIENT,
            "notarytool credential preflight hit a transient Apple network error; "
            "continuing to Build and package",
        )
    if _DETERMINISTIC_LOCAL_AUTH_RE.search(output):
        return (
            PreflightStatus.AUTH_FAILED,
            "notarytool rejected local App Store Connect credentials or key file — "
            "verify APPLE_API_KEY_P8, APPLE_API_KEY_ID, and APPLE_API_ISSUER in release-signing",
        )
    return (
        PreflightStatus.TRANSIENT,
        "notarytool credential preflight exited non-zero without a deterministic auth error; "
        "continuing to Build and package",
    )


def _default_runner(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
        timeout=timeout,
        check=False,
    )


def _success_message(stdout: str) -> str:
    try:
        payload = json.loads(stdout)
    except ValueError:
        return "App Store Connect credentials verified via notarytool history."
    history = payload.get("history") if isinstance(payload, dict) else None
    if isinstance(history, list):
        return (
            f"App Store Connect credentials verified via notarytool history "
            f"({len(history)} submissions visible)."
        )
    return "App Store Connect credentials verified via notarytool history."


def check_notary_auth(
    key_path: str,
    key_id: str,
    issuer: str,
    *,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    runner: CommandRunner | None = None,
) -> PreflightResult:
    """Validate the .p8 file locally, then probe Apple with ``xcrun notarytool history``."""
    clean_path = key_path.strip()
    clean_key_id = key_id.strip()
    clean_issuer = issuer.strip()

    missing = [
        name
        for name, value in (
            ("APPLE_API_KEY", clean_path),
            ("APPLE_API_KEY_ID", clean_key_id),
            ("APPLE_API_ISSUER", clean_issuer),
        )
        if not value
    ]
    if missing:
        return PreflightResult(
            PreflightStatus.AUTH_FAILED,
            f"App Store Connect notarization credential inputs are unset: {', '.join(missing)}",
        )

    path = Path(clean_path)
    if not path.is_file():
        return PreflightResult(
            PreflightStatus.AUTH_FAILED,
            f"App Store Connect .p8 key file does not exist: {clean_path}",
        )
    try:
        pem_text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        return PreflightResult(
            PreflightStatus.AUTH_FAILED,
            f"App Store Connect .p8 key file could not be read ({type(exc).__name__}): {clean_path}",
        )

    pem_error = validate_p8_pem(pem_text)
    if pem_error is not None:
        return PreflightResult(PreflightStatus.AUTH_FAILED, pem_error)

    argv = [
        "xcrun",
        "notarytool",
        "history",
        "--key",
        clean_path,
        "--key-id",
        clean_key_id,
        "--issuer",
        clean_issuer,
        "--output-format",
        "json",
    ]
    execute = runner or _default_runner
    try:
        proc = execute(argv, timeout)
    except subprocess.TimeoutExpired:
        return PreflightResult(
            PreflightStatus.TRANSIENT,
            f"notarytool credential preflight timed out after {timeout:g}s; "
            "continuing to Build and package",
        )
    except OSError as exc:
        return PreflightResult(
            PreflightStatus.TRANSIENT,
            f"notarytool credential preflight could not invoke xcrun ({type(exc).__name__}); "
            "continuing to Build and package",
        )

    if proc.returncode == 0:
        return PreflightResult(PreflightStatus.OK, _success_message(proc.stdout or ""))

    raw_output = "\n".join(part for part in (proc.stderr or "", proc.stdout or "") if part.strip())
    # Key material echoed without its PEM markers would escape the block pattern.
    body = [line.strip() for line in pem_text.splitlines() if line.strip() and not line.strip().startswith("-----")]
    redacted = redact_secrets(raw_output, (*body, clean_key_id, clean_issuer))
    status, message = classify_notarytool_failure(redacted)
    return PreflightResult(status, message, detail=redacted)


def main(argv: list[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    source_env = os.environ if env is None else env
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", default=source_env.get("APPLE_API_KEY", ""))
    parser.add_argument("--key-id", default=source_env.get("APPLE_API_KEY_ID", ""))
    parser.add_argument("--issuer", default=source_env.get("APPLE_API_ISSUER", ""))
    parser.add_argument("--timeout", type=float, default=_DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)

    result = check_notary_auth(args.key, args.key_id, args.issuer, timeout=args.timeout)
    if result.status is PreflightStatus.OK:
        print(f"[notary-preflight] {result.message}")
        return 0
    if result.status is PreflightStatus.TRANSIENT:
        print(f"::warning::{result.message}")
        if result.detail:
            print(result.detail)
        return 0
    print(f"::error::{result.message}")
    if result.detail:
        print(result.detail, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
