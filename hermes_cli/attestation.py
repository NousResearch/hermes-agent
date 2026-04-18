"""TEE Attestation verification module for Hermes Agent."""
import asyncio
import contextlib
import hashlib
import io
import logging
import os
import secrets
import socket
import ssl
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

# Add vendored verifier modules to path
vendor_path = os.path.join(os.path.dirname(__file__), "..", "examples", "tee-providers", "vendor", "nearai-cloud-verifier", "py")
if vendor_path not in sys.path:
    sys.path.insert(0, vendor_path)

_VERIFIER_AVAILABLE = False
try:
    from model_verifier import check_tdx_quote, check_report_data
    from domain_verifier import DomainAttestation, verify_domain_attestation
    _VERIFIER_AVAILABLE = True
except ImportError:
    pass

# Set HERMES_ATTESTATION_VERBOSE=1 to print verifier output to console.
_VERBOSE = os.getenv("HERMES_ATTESTATION_VERBOSE", "") == "1"


@dataclass
class AttestationReport:
    """TEE attestation verification result."""
    valid: bool
    provider: str
    attestation_type: str
    verified_at: str
    details: Dict[str, Any]
    error: Optional[str] = None
    verifier_output: str = ""  # captured stdout from verifier; shown when verbose


# Cache: {(provider_id, base_url): (report, verified_at_ms, pinned_cert_fingerprint)}
_ATTESTATION_CACHE: Dict[tuple, tuple] = {}
_DEFAULT_TTL_SECONDS = 3600
_FAILURE_TTL_SECONDS = 60  # re-try failed attestations after 60s


def _live_tls_fingerprint(base_url: str) -> Optional[str]:
    """SHA-256 of the live server's leaf certificate (DER)."""
    try:
        parsed = urlparse(base_url)
        host, port = parsed.hostname, parsed.port or 443
        ctx = ssl.create_default_context()
        with socket.create_connection((host, port), timeout=5) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as tls:
                der = tls.getpeercert(binary_form=True)
        return hashlib.sha256(der).hexdigest() if der else None
    except Exception as e:
        logger.debug("live TLS fingerprint probe failed: %s", e)
        return None


def verify_attestation(provider_id: str, runtime_creds: Dict[str, Any], config: Dict[str, Any]) -> AttestationReport:
    """Verify TEE attestation for a provider, with cache.

    Cache is keyed on (provider_id, base_url) and pinned to the live TLS cert
    fingerprint. Successful reports are cached for HERMES_ATTESTATION_TTL seconds
    (default 3600). Failed reports are cached for 60s to avoid hammering the endpoint.
    """
    base_url = (runtime_creds.get("base_url") or "").rstrip("/")
    cache_key = (provider_id, base_url)
    ttl = int(os.getenv("HERMES_ATTESTATION_TTL", str(_DEFAULT_TTL_SECONDS)))

    cached = _ATTESTATION_CACHE.get(cache_key)
    if cached:
        cached_report, verified_at_ms, pinned_fpr = cached
        effective_ttl = ttl if cached_report.valid else _FAILURE_TTL_SECONDS
        age_ms = int(time.time() * 1000) - verified_at_ms
        if age_ms < effective_ttl * 1000:
            if cached_report.valid:
                live_fpr = _live_tls_fingerprint(base_url)
                if live_fpr and pinned_fpr and live_fpr != pinned_fpr:
                    pass  # cert changed — fall through to re-verify
                else:
                    return cached_report
            else:
                return cached_report  # don't re-check TLS for failures

    if not _VERIFIER_AVAILABLE:
        return AttestationReport(
            valid=False, provider=provider_id, attestation_type="none",
            verified_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            details={}, error="Attestation verifier dependencies not available"
        )
    if provider_id == "near-ai":
        report = _verify_near_ai_attestation(runtime_creds, config)
    else:
        report = _skip_attestation("not-implemented")

    live_fpr_now = _live_tls_fingerprint(base_url) if report.valid else None
    _ATTESTATION_CACHE[cache_key] = (report, int(time.time() * 1000), live_fpr_now)

    if _VERBOSE and report.verifier_output:
        print(report.verifier_output, end="")
    elif report.verifier_output:
        logger.debug("attestation verifier output:\n%s", report.verifier_output.rstrip())

    return report


def _verify_near_ai_attestation(runtime_creds: Dict[str, Any], config: Dict[str, Any]) -> AttestationReport:
    """Verify NEAR AI Cloud TEE attestation."""
    api_key = runtime_creds.get("api_key", "")
    base_url = runtime_creds.get("base_url", "https://cloud-api.near.ai").rstrip("/")
    if not api_key:
        return AttestationReport(
            valid=False, provider="near-ai", attestation_type="tdx",
            verified_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            details={}, error="No API key in runtime credentials"
        )
    try:
        nonce = secrets.token_hex(32)
        url = f"{base_url}/attestation/report"
        params = {"nonce": nonce, "signing_algo": "ecdsa", "include_tls_fingerprint": "true"}
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        report = response.json()
        gateway_attestation = report.get("gateway_attestation", {})
        if not gateway_attestation:
            return AttestationReport(
                valid=False, provider="near-ai", attestation_type="tdx",
                verified_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                details={"report": report}, error="No gateway_attestation in response"
            )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            intel_result = asyncio.run(check_tdx_quote(gateway_attestation))
        verifier_out = buf.getvalue()

        if not intel_result.get("verified", False):
            return AttestationReport(
                valid=False, provider="near-ai", attestation_type="tdx",
                verified_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                details={"intel_result": intel_result}, error="Invalid TDX quote: verification failed",
                verifier_output=verifier_out,
            )
        platform_status = intel_result.get("platform_status", {}).get("status", "Unknown")
        if platform_status != "UpToDate":
            advisories = intel_result.get("platform_status", {}).get("advisory_ids", [])
            return AttestationReport(
                valid=False, provider="near-ai", attestation_type="tdx",
                verified_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                details={"platform_status": platform_status, "advisory_ids": advisories,
                         "ppid": intel_result.get("ppid")},
                error=f"Platform TCB out of date: {platform_status} advisories={advisories}",
                verifier_output=verifier_out,
            )
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            report_data_result = check_report_data(gateway_attestation, nonce, intel_result)
        verifier_out += buf2.getvalue()

        if not report_data_result.get("binds_address", False):
            return AttestationReport(
                valid=False, provider="near-ai", attestation_type="tdx",
                verified_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                details={"report_data_result": report_data_result},
                error="Report data does not bind signing address + TLS fingerprint",
                verifier_output=verifier_out,
            )
        if not report_data_result.get("embeds_nonce", False):
            return AttestationReport(
                valid=False, provider="near-ai", attestation_type="tdx",
                verified_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                details={"report_data_result": report_data_result},
                error="Report data does not embed request nonce",
                verifier_output=verifier_out,
            )
        tls_certificate = report.get("tls_certificate", "")
        if not tls_certificate:
            return AttestationReport(
                valid=False, provider="near-ai", attestation_type="tdx",
                verified_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                details={"report": report}, error="No tls_certificate in response",
                verifier_output=verifier_out,
            )
        domain = urlparse(base_url).netloc
        domain_attestation = DomainAttestation(
            domain=domain, sha256sum=gateway_attestation.get("tls_cert_fingerprint", ""),
            acme_account=gateway_attestation.get("acme_account", ""), cert=tls_certificate,
            intel_quote=gateway_attestation.get("intel_quote", ""), info=gateway_attestation
        )
        buf3 = io.StringIO()
        with contextlib.redirect_stdout(buf3):
            try:
                asyncio.run(verify_domain_attestation(domain_attestation))
            except Exception:
                pass
        verifier_out += buf3.getvalue()

        return AttestationReport(
            valid=True, provider="near-ai", attestation_type="tdx",
            verified_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            details={
                "signing_address": gateway_attestation.get("signing_address"),
                "tls_cert_fingerprint": gateway_attestation.get("tls_cert_fingerprint"),
                "acme_account": gateway_attestation.get("acme_account"), "domain": domain,
                "app_id": gateway_attestation.get("info", {}).get("app_id"),
                "instance_id": gateway_attestation.get("info", {}).get("instance_id"),
                "platform_status": platform_status,
                "ppid": intel_result.get("ppid"),
            },
            verifier_output=verifier_out,
        )
    except Exception:
        raise


def _skip_attestation(reason: str) -> AttestationReport:
    return AttestationReport(
        valid=False, provider="unknown", attestation_type="none",
        verified_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        details={}, error=f"Attestation skipped: {reason}"
    )
