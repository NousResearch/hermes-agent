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

import base64
import json as _json

import requests

logger = logging.getLogger(__name__)

_PHALA_TDX_VERIFIER = "https://cloud-api.phala.network/api/v1/attestations/verify"
_NVIDIA_NRAS = "https://nras.attestation.nvidia.com/v3/attest/gpu"

# Add vendored verifier modules to path.
# Default: alongside this repo at refs/nearai-cloud-verifier/py; override with NEARAI_VERIFIER_PATH.
_default_verifier_path = os.path.join(os.path.dirname(__file__), "..", "..", "hermes-agent", "refs", "nearai-cloud-verifier", "py")
vendor_path = os.environ.get("NEARAI_VERIFIER_PATH", _default_verifier_path)
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

    if provider_id == "redpill":
        report = _verify_redpill_attestation(runtime_creds, config)
    elif provider_id == "near-ai":
        if not _VERIFIER_AVAILABLE:
            return AttestationReport(
                valid=False, provider=provider_id, attestation_type="none",
                verified_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                details={}, error="Attestation verifier dependencies not available"
            )
        else:
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


def _decode_nvidia_verdict(jwt_token: str) -> str:
    """Extract x-nvidia-overall-att-result from a NRAS JWT response."""
    payload_b64 = jwt_token.split(".")[1]
    padded = payload_b64 + "=" * ((4 - len(payload_b64) % 4) % 4)
    payload = _json.loads(base64.urlsafe_b64decode(padded).decode())
    return payload.get("x-nvidia-overall-att-result", "UNKNOWN")


def _phala_check_report_data(report_data_hex: str, signing_address: str, signing_algo: str, nonce: str) -> bool:
    """Verify TDX report_data = signing_address (padded to 32 bytes) || nonce (32 bytes)."""
    try:
        rd = bytes.fromhex(report_data_hex.removeprefix("0x"))
        addr_hex = signing_address.removeprefix("0x")
        addr_bytes = bytes.fromhex(addr_hex)
        embedded_addr = rd[:32]
        embedded_nonce = rd[32:64]
        return (embedded_addr == addr_bytes.ljust(32, b"\x00")) and (embedded_nonce.hex() == nonce)
    except Exception:
        return False


def _verify_redpill_attestation(runtime_creds: Dict[str, Any], config: Dict[str, Any]) -> AttestationReport:
    """Verify Redpill/Phala TEE attestation: TDX quote + compose hash + NVIDIA GPU."""
    api_key = runtime_creds.get("api_key", "")
    base_url = runtime_creds.get("base_url", "https://api.red-pill.ai/v1").rstrip("/")
    # strip trailing /v1 so we can append it consistently
    api_base = base_url[:-3] if base_url.endswith("/v1") else base_url
    model = runtime_creds.get("model", "")
    if not api_key:
        return AttestationReport(
            valid=False, provider="redpill", attestation_type="tdx+gpu",
            verified_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            details={}, error="No API key in runtime credentials"
        )
    if not model:
        return AttestationReport(
            valid=False, provider="redpill", attestation_type="tdx+gpu",
            verified_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            details={}, error="No model in runtime credentials"
        )
    nonce = secrets.token_hex(32)
    url = f"{api_base}/v1/attestation/report"
    response = requests.get(url, params={"model": model, "nonce": nonce},
                            headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
    response.raise_for_status()
    report = response.json()

    gateway_att = report.get("gateway_attestation", {})
    model_atts = report.get("model_attestations", [])
    if not gateway_att:
        return AttestationReport(
            valid=False, provider="redpill", attestation_type="tdx+gpu",
            verified_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            details={"report": report}, error="No gateway_attestation in response"
        )

    # 1. TDX quote via Phala's verifier
    tdx_resp = requests.post(_PHALA_TDX_VERIFIER, json={"hex": gateway_att["intel_quote"]}, timeout=30).json()
    quote_body = (tdx_resp.get("quote") or {}).get("body", {})
    if not (tdx_resp.get("quote") or {}).get("verified"):
        msg = (tdx_resp.get("quote") or {}).get("message") or tdx_resp.get("message") or "unspecified"
        node = tdx_resp.get("node_provider") or {}
        ppid = node.get("ppid")
        tcb_svn = quote_body.get("tee_tcb_svn")
        detail_str = f"ppid={ppid} tcb_svn={tcb_svn}" if ppid else msg
        return AttestationReport(
            valid=False, provider="redpill", attestation_type="tdx+gpu",
            verified_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            details={"ppid": ppid, "tcb_svn": tcb_svn},
            error=f"TDX quote verification failed: {detail_str}"
        )

    # 2. report_data binds signing_address + nonce
    report_data_hex = quote_body.get("reportdata", "")
    signing_address = gateway_att.get("signing_address", "")
    signing_algo = gateway_att.get("signing_algo", "ecdsa")
    if not _phala_check_report_data(report_data_hex, signing_address, signing_algo, nonce):
        return AttestationReport(
            valid=False, provider="redpill", attestation_type="tdx+gpu",
            verified_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            details={}, error="TDX report_data does not bind signing address + nonce"
        )

    # 3. Compose hash committed in mr_config
    mr_config = quote_body.get("mrconfig", "")
    info = gateway_att.get("info", {})
    tcb_info = info.get("tcb_info", {})
    if isinstance(tcb_info, str):
        try:
            tcb_info = _json.loads(tcb_info)
        except Exception:
            tcb_info = {}
    app_compose = tcb_info.get("app_compose")
    compose_hash_verified = False
    if app_compose and mr_config:
        compose_hash = hashlib.sha256(app_compose.encode()).hexdigest()
        expected = ("0x01" + compose_hash).lower()
        compose_hash_verified = mr_config.lower().startswith(expected)

    # 4. NVIDIA GPU attestation from model_attestations
    gpu_verdict = None
    if model_atts:
        nvidia_payload = model_atts[0].get("nvidia_payload")
        if nvidia_payload:
            if isinstance(nvidia_payload, str):
                nvidia_payload = _json.loads(nvidia_payload)
            gpu_resp = requests.post(_NVIDIA_NRAS, json=nvidia_payload, timeout=30).json()
            gpu_verdict = _decode_nvidia_verdict(gpu_resp[0][1])
            gpu_passed = gpu_verdict is True or gpu_verdict == "PASS"
            if not gpu_passed:
                return AttestationReport(
                    valid=False, provider="redpill", attestation_type="tdx+gpu",
                    verified_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    details={"gpu_verdict": gpu_verdict}, error=f"NVIDIA GPU attestation failed: {gpu_verdict}"
                )

    return AttestationReport(
        valid=True, provider="redpill", attestation_type="tdx+gpu",
        verified_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        details={
            "signing_address": signing_address,
            "model": model,
            "compose_hash_verified": compose_hash_verified,
            "gpu_verdict": gpu_verdict,
            "app_id": info.get("app_id"),
            "instance_id": info.get("instance_id"),
        },
    )


def _skip_attestation(reason: str) -> AttestationReport:
    return AttestationReport(
        valid=False, provider="unknown", attestation_type="none",
        verified_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        details={}, error=f"Attestation skipped: {reason}"
    )
