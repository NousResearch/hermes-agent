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
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import base64
import json as _json

import requests

from hermes_cli.anchors import expected_for_model, load_nearai_anchor

logger = logging.getLogger(__name__)

_NEARAI_ANCHOR = load_nearai_anchor()

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
    from model_verifier import check_tdx_quote, check_report_data, check_gpu
    from domain_verifier import DomainAttestation, verify_domain_attestation
    _VERIFIER_AVAILABLE = True
except ImportError:
    pass

# Set HERMES_ATTESTATION_VERBOSE=1 to print verifier output to console.
_VERBOSE = os.getenv("HERMES_ATTESTATION_VERBOSE", "") == "1"

# contextlib.redirect_stdout swaps sys.stdout globally; NEAR AI verifier calls
# that print must be serialized across threads so parallel probes don't leak
# verifier output onto the user's terminal.
_STDOUT_CAPTURE_LOCK = threading.Lock()


def _sync_run(coro):
    """Run an async coroutine synchronously. Safe inside a running event loop
    (e.g. Hermes CLI) by offloading to a worker thread with its own loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    from concurrent.futures import ThreadPoolExecutor
    def _worker():
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_worker).result()


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
    signing_public_key: Optional[str] = None
    signing_algo: str = "ecdsa"


# Cache: {(provider_id, base_url): (report, verified_at_ms, pinned_cert_fingerprint)}
_ATTESTATION_CACHE: Dict[tuple, tuple] = {}
_DEFAULT_TTL_SECONDS = 3600
_FAILURE_TTL_SECONDS = 60  # re-try failed attestations after 60s

# Per-model cache: {(provider_id, model_id): report} — populated lazily as models are used
_MODEL_ATTESTATION_CACHE: Dict[tuple, "AttestationReport"] = {}


def get_model_attestation_status(provider_id: str, model_id: str) -> "Optional[AttestationReport]":
    return _MODEL_ATTESTATION_CACHE.get((provider_id, model_id))


def probe_models_for_provider(
    provider_id: str,
    api_key: str,
    base_url: str,
    models: "list[str]",
    config: Optional[Dict[str, Any]] = None,
    max_workers: int = 6,
) -> "list[str]":
    """Probe attestation for each model in parallel; return only models with valid reports."""
    from concurrent.futures import ThreadPoolExecutor

    cfg = config or {}

    def _probe(model_id: str) -> "tuple[str, bool]":
        creds = {"api_key": api_key, "base_url": base_url, "model": model_id}
        try:
            report = verify_attestation(provider_id, creds, cfg)
        except Exception as exc:
            logger.debug("probe failed for %s/%s: %s", provider_id, model_id, exc)
            return model_id, False
        return model_id, report.valid

    if not models:
        return []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(models))) as ex:
        results = list(ex.map(_probe, models))
    return [m for m, ok in results if ok]


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

    Cache is keyed on (provider_id, base_url, model) and pinned to the live TLS cert
    fingerprint. Successful reports are cached for HERMES_ATTESTATION_TTL seconds
    (default 3600). Failed reports are cached for 60s to avoid hammering the endpoint.
    """
    base_url = (runtime_creds.get("base_url") or "").rstrip("/")
    model_id = runtime_creds.get("model", "") or ""
    cache_key = (provider_id, base_url, model_id)
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
    elif provider_id == "venice":
        report = _verify_venice_attestation(runtime_creds, config)
    elif provider_id == "near-ai":
        if not _VERIFIER_AVAILABLE:
            report = AttestationReport(
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

    if model_id:
        _MODEL_ATTESTATION_CACHE[(provider_id, model_id)] = report

    if _VERBOSE and report.verifier_output:
        print(report.verifier_output, end="")
    elif report.verifier_output:
        logger.debug("attestation verifier output:\n%s", report.verifier_output.rstrip())

    return report


def _verify_near_ai_attestation(runtime_creds: Dict[str, Any], config: Dict[str, Any]) -> AttestationReport:
    """Verify NEAR AI Cloud TEE attestation: gateway + model TDX quotes, GPU, E2EE key binding."""
    api_key = runtime_creds.get("api_key", "")
    base_url = runtime_creds.get("base_url", "https://cloud-api.near.ai").rstrip("/")
    # base_url may include /v1 (the OpenAI inference path); strip it for the attestation endpoint
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    if not api_key:
        return AttestationReport(
            valid=False, provider="near-ai", attestation_type="tdx",
            verified_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            details={}, error="No API key in runtime credentials"
        )

    model = runtime_creds.get("model", "openai/gpt-oss-120b")
    nonce = secrets.token_hex(32)
    url = f"{base_url}/v1/attestation/report"
    params = {"model": model, "nonce": nonce, "signing_algo": "ecdsa", "include_tls_fingerprint": "true"}
    response = requests.get(url, params=params, headers={"Authorization": f"Bearer {api_key}"}, timeout=60)
    response.raise_for_status()
    report = response.json()
    verifier_out = ""
    now = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _fail(error, details=None):
        return AttestationReport(
            valid=False, provider="near-ai", attestation_type="tdx",
            verified_at=now(), details=details or {}, error=error,
            verifier_output=verifier_out,
        )

    # ── Gateway attestation ──────────────────────────────────────────────────
    gateway = report.get("gateway_attestation", {})
    if not gateway:
        return _fail("No gateway_attestation in response")

    buf = io.StringIO()
    with _STDOUT_CAPTURE_LOCK, contextlib.redirect_stdout(buf):
        gw_intel = _sync_run(check_tdx_quote(gateway))
    verifier_out += buf.getvalue()

    if not gw_intel or not gw_intel.get("verified", False):
        return _fail("Gateway TDX quote verification failed", {"intel_result": gw_intel})

    buf2 = io.StringIO()
    with _STDOUT_CAPTURE_LOCK, contextlib.redirect_stdout(buf2):
        gw_rd = check_report_data(gateway, nonce, gw_intel)
    verifier_out += buf2.getvalue()

    if not gw_rd.get("binds_address"):
        return _fail("Gateway report_data does not bind signing address + TLS fingerprint")
    if not gw_rd.get("embeds_nonce"):
        return _fail("Gateway report_data does not embed request nonce")

    tls_certificate = report.get("tls_certificate", "")
    if not tls_certificate:
        return _fail("No tls_certificate in response")

    domain = urlparse(base_url).netloc
    buf3 = io.StringIO()
    with _STDOUT_CAPTURE_LOCK, contextlib.redirect_stdout(buf3):
        try:
            _sync_run(verify_domain_attestation(DomainAttestation(
                domain=domain, sha256sum=gateway.get("tls_cert_fingerprint", ""),
                acme_account=gateway.get("acme_account", ""), cert=tls_certificate,
                intel_quote=gateway.get("intel_quote", ""), info=gateway
            )))
        except Exception:
            pass
    verifier_out += buf3.getvalue()

    gw_status = gw_intel.get("status", "Unknown")
    gw_advisories = gw_intel.get("advisory_ids", [])
    if gw_status == "OutOfDate":
        logger.warning("Gateway platform TCB is OutOfDate (advisories: %s) — quote is valid but firmware is unpatched", ", ".join(gw_advisories) or "none")

    # ── Model attestations ───────────────────────────────────────────────────
    model_atts = report.get("model_attestations") or []
    if not model_atts:
        return _fail("No model_attestations in response — cannot verify E2EE key")

    model_signing_key = None
    model_details = []

    for i, model_att in enumerate(model_atts):
        buf_m = io.StringIO()
        with _STDOUT_CAPTURE_LOCK, contextlib.redirect_stdout(buf_m):
            m_intel = _sync_run(check_tdx_quote(model_att))
        verifier_out += buf_m.getvalue()

        if not m_intel or not m_intel.get("verified", False):
            return _fail(f"Model attestation #{i+1} TDX quote verification failed")

        buf_m2 = io.StringIO()
        with _STDOUT_CAPTURE_LOCK, contextlib.redirect_stdout(buf_m2):
            m_rd = check_report_data(model_att, nonce, m_intel)
        verifier_out += buf_m2.getvalue()

        backend_id = (
            f"signing_address={model_att.get('signing_address','?')[:10]}… "
            f"app_id={model_att.get('info',{}).get('app_id','?')[:10]}…"
        )

        if not m_rd.get("binds_address"):
            return _fail(f"Model attestation #{i+1} ({backend_id}) report_data does not bind signing address + nonce")
        if not m_rd.get("embeds_nonce"):
            return _fail(f"Model attestation #{i+1} ({backend_id}) report_data does not embed request nonce")

        if not model_att.get("nvidia_payload"):
            return _fail(f"Model attestation #{i+1} ({backend_id}) missing nvidia_payload — GPU attestation required")
        buf_m3 = io.StringIO()
        with _STDOUT_CAPTURE_LOCK, contextlib.redirect_stdout(buf_m3):
            gpu_result = check_gpu(model_att, nonce)
        verifier_out += buf_m3.getvalue()
        if gpu_result.get("verdict") not in ("PASS", True):
            nras_detail = _capture_nras_failure_detail(model_att.get("nvidia_payload", ""))
            return _fail(
                f"Model attestation #{i+1} ({backend_id}) GPU NRAS verdict={gpu_result.get('verdict')!r} "
                f"(nonce_matches={gpu_result.get('nonce_matches')}); "
                f"NRAS claims: {_json.dumps(nras_detail, default=str)}. "
                f"Try `model: openai/gpt-oss-120b` or `zai-org/GLM-5.1-FP8` to isolate fleet vs service."
            )
        if not gpu_result.get("nonce_matches"):
            return _fail(f"Model attestation #{i+1} ({backend_id}) GPU nonce mismatch — replay or stale evidence")

        # compose hash for model CVM
        info = model_att.get("info", {})
        tcb_info = info.get("tcb_info", {})
        if isinstance(tcb_info, str):
            try:
                tcb_info = _json.loads(tcb_info)
            except Exception:
                tcb_info = {}
        app_compose = tcb_info.get("app_compose") if tcb_info else None
        m_mr_config = m_intel.get("quote", {}).get("body", {}).get("mrconfig", "")
        compose_verified = False
        compose_hash = ""
        if app_compose and m_mr_config:
            compose_hash = hashlib.sha256(app_compose.encode()).hexdigest()
            compose_verified = m_mr_config.lower().startswith(("01" + compose_hash).lower())

        # Block B-static: enforce pinned anchor (app_id, compose_hash, os_image_hash, kms pubkey)
        # Until on-chain Base RPC reads land, hermes_cli/anchors/nearai_mainnet.json is the
        # authority for "is this CVM permitted to serve model M". Strict mode → _fail() raises.
        expected = expected_for_model(_NEARAI_ANCHOR, model)
        if expected is None:
            return _fail(
                f"Model {model!r} has no entry in hermes_cli/anchors/nearai_mainnet.json — "
                f"add expected app_id/compose_hashes (refusing in strict mode)"
            )
        attested_app_id = (info.get("app_id") or "").lower().removeprefix("0x")
        expected_app_id = expected["app_id"].lower().removeprefix("0x")
        if attested_app_id != expected_app_id:
            return _fail(
                f"Model #{i+1} ({backend_id}) app_id 0x{attested_app_id} != "
                f"anchored 0x{expected_app_id} for {model!r}"
            )
        if not compose_verified:
            return _fail(
                f"Model #{i+1} ({backend_id}) compose_hash unverifiable: "
                f"in-quote app_compose sha256 does not match mr_config — quote vs evidence inconsistency"
            )
        allowed_composes = {c.lower().removeprefix("0x") for c in expected.get("compose_hashes", [])}
        if compose_hash.lower() not in allowed_composes:
            return _fail(
                f"Model #{i+1} ({backend_id}) compose_hash 0x{compose_hash} not in anchored "
                f"compose_hashes for {model!r} (allowed: {sorted(allowed_composes)})"
            )
        attested_os = (info.get("os_image_hash") or "").lower().removeprefix("0x")
        allowed_os = {h.lower().removeprefix("0x") for h in _NEARAI_ANCHOR.get("os_image_hashes", [])}
        if allowed_os and attested_os not in allowed_os:
            return _fail(
                f"Model #{i+1} ({backend_id}) os_image_hash 0x{attested_os} not in anchored "
                f"os_image_hashes (allowed: {sorted(allowed_os)})"
            )
        kpi_raw = info.get("key_provider_info") or "{}"
        kpi = _json.loads(kpi_raw) if isinstance(kpi_raw, str) else kpi_raw
        attested_kpi_id = (kpi.get("id") or "").lower().removeprefix("0x")
        expected_kpi_id = _NEARAI_ANCHOR.get("kms_provider_info_id", "").lower().removeprefix("0x")
        if expected_kpi_id and attested_kpi_id != expected_kpi_id:
            return _fail(
                f"Model #{i+1} ({backend_id}) key_provider_info.id != anchored KMS pubkey "
                f"(KMS substitution? expected sha256={hashlib.sha256(bytes.fromhex(expected_kpi_id)).hexdigest()[:16]}…, "
                f"got sha256={hashlib.sha256(bytes.fromhex(attested_kpi_id)).hexdigest()[:16] if attested_kpi_id else '<none>'}…)"
            )

        # verify signing_public_key derives to signing_address (so we know the E2EE key is hardware-bound)
        spk = model_att.get("signing_public_key")
        signing_addr = model_att.get("signing_address", "")
        if spk and signing_addr:
            from eth_keys.datatypes import PublicKey as _EthPubKey
            pub_bytes = bytes.fromhex(spk)
            if len(pub_bytes) == 65 and pub_bytes[0] == 0x04:
                pub_bytes = pub_bytes[1:]
            derived = "0x" + _EthPubKey(pub_bytes).to_canonical_address().hex()
            if derived.lower() != signing_addr.lower():
                return _fail(
                    f"Model attestation #{i+1} signing_public_key does not derive to signing_address",
                    {"derived": derived, "claimed": signing_addr},
                )
            if i == 0:
                model_signing_key = spk

        m_status = m_intel.get("status", "Unknown")
        m_advisories = m_intel.get("advisory_ids", [])
        if m_status == "OutOfDate":
            logger.warning("Model attestation #%d platform TCB is OutOfDate (advisories: %s) — quote is valid but firmware is unpatched", i + 1, ", ".join(m_advisories) or "none")

        model_details.append({
            "signing_address": signing_addr,
            "app_id": info.get("app_id"),
            "compose_hash": compose_hash,
            "os_image_hash": info.get("os_image_hash"),
            "status": m_status,
            "advisory_ids": m_advisories,
            "compose_hash_verified": compose_verified,
            "anchor_matched": True,
            "gpu_verdict": gpu_result.get("verdict"),
        })

    return AttestationReport(
        valid=True, provider="near-ai", attestation_type="tdx+gpu",
        verified_at=now(),
        details={
            "gateway": {
                "signing_address": gateway.get("signing_address"),
                "tls_cert_fingerprint": gateway.get("tls_cert_fingerprint"),
                "domain": domain,
                "app_id": gateway.get("info", {}).get("app_id"),
                "status": gw_status,
                "advisory_ids": gw_advisories,
            },
            "models": model_details,
        },
        signing_public_key=model_signing_key,
        signing_algo="ecdsa",
        verifier_output=verifier_out,
    )


def _decode_nvidia_jwt_claims(jwt_token: str) -> dict:
    """Decode a NRAS JWT and return its full claims dict."""
    payload_b64 = jwt_token.split(".")[1]
    padded = payload_b64 + "=" * ((4 - len(payload_b64) % 4) % 4)
    return _json.loads(base64.urlsafe_b64decode(padded).decode())


def _decode_nvidia_verdict(jwt_token: str) -> str:
    """Extract x-nvidia-overall-att-result from a NRAS JWT response."""
    return _decode_nvidia_jwt_claims(jwt_token).get("x-nvidia-overall-att-result", "UNKNOWN")


def _capture_nras_failure_detail(nvidia_payload_str: str) -> dict:
    """Replay the NRAS POST and return full JWT claims so the caller can surface
    the specific reason NVIDIA rejected the GPU evidence (driver mismatch, stale
    measurement, signature, etc.). Returns {} if replay itself fails."""
    try:
        payload = _json.loads(nvidia_payload_str)
        resp = requests.post(_NVIDIA_NRAS, json=payload, timeout=30).json()
        jwt_token = resp[0][1]
        claims = _decode_nvidia_jwt_claims(jwt_token)
        detail_keys = {
            k: v for k, v in claims.items()
            if k.startswith("x-nvidia-") and k != "x-nvidia-eat-ver"
        }
        return detail_keys
    except Exception as e:
        return {"_replay_error": str(e)}


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


def _verify_redpill_chutes(report: Dict[str, Any], model: str, nonce: str) -> AttestationReport:
    """Verify Chutes-routed redpill models (attestation_type="chutes")."""
    now = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _fail(err: str, details: Optional[Dict[str, Any]] = None) -> AttestationReport:
        return AttestationReport(
            valid=False, provider="redpill", attestation_type="chutes+tdx",
            verified_at=now(), details=details or {}, error=err,
        )

    atts = report.get("all_attestations") or []
    if not atts:
        return _fail("Chutes response missing all_attestations")

    verified_instances = []
    for i, att in enumerate(atts):
        quote_b64 = att.get("intel_quote") or ""
        if not quote_b64:
            return _fail(f"Chutes instance #{i+1} missing intel_quote")
        e2e_pubkey = att.get("e2e_pubkey") or ""
        inst_nonce = att.get("nonce") or nonce
        if not e2e_pubkey:
            return _fail(f"Chutes instance #{i+1} missing e2e_pubkey (anti-tamper binding requires it)")

        # Phala TDX verifier accepts hex; Chutes sends base64. Decode and re-encode.
        try:
            quote_bytes = base64.b64decode(quote_b64)
        except Exception as exc:
            return _fail(f"Chutes instance #{i+1} intel_quote base64 decode failed: {exc}")
        quote_hex = quote_bytes.hex()

        tdx_resp = requests.post(_PHALA_TDX_VERIFIER, json={"hex": quote_hex}, timeout=30).json()
        quote_body = (tdx_resp.get("quote") or {}).get("body", {})
        if not (tdx_resp.get("quote") or {}).get("verified"):
            msg = (tdx_resp.get("quote") or {}).get("message") or tdx_resp.get("message") or "unspecified"
            return _fail(f"Chutes instance #{i+1} TDX quote verification failed: {msg}",
                         {"tcb_svn": quote_body.get("tee_tcb_svn")})

        # Debug mode = td_attributes bit 0; must be OFF
        td_attr = quote_body.get("td_attributes", "") or quote_body.get("tdAttributes", "")
        if td_attr:
            try:
                if int(td_attr, 16) & 1:
                    return _fail(f"Chutes instance #{i+1} TDX running in Debug mode")
            except ValueError:
                pass

        # Anti-tamper binding: SHA256(nonce || e2e_pubkey) == report_data[0:32]
        report_data_hex = (quote_body.get("reportdata") or "").removeprefix("0x").lower()
        expected = hashlib.sha256((inst_nonce + e2e_pubkey).encode()).hexdigest().lower()
        if report_data_hex[:64] != expected:
            return _fail(
                f"Chutes instance #{i+1} anti-tamper binding failed: "
                f"SHA256(nonce||e2e_pubkey) != report_data[0:32]",
                {"expected": expected, "got": report_data_hex[:64]},
            )

        verified_instances.append({
            "instance_id": att.get("instance_id"),
            "status": quote_body.get("status") or (tdx_resp.get("quote") or {}).get("status"),
        })

    return AttestationReport(
        valid=True, provider="redpill", attestation_type="chutes+tdx",
        verified_at=now(),
        details={"model": model, "instances": verified_instances, "instance_count": len(verified_instances)},
        signing_public_key=None,
        signing_algo="ecdsa",
    )


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
    # Chutes-routed models can take 60s+ to return the attestation bundle
    # (5 instances × per-instance TDX quote).
    response = requests.get(url, params={"model": model, "nonce": nonce},
                            headers={"Authorization": f"Bearer {api_key}"}, timeout=90)
    response.raise_for_status()
    report = response.json()

    # Redpill returns four formats, matching the four backends in the verifier docs:
    # - Phala simple: top-level intel_quote/nvidia_payload (phala/gpt-oss-20b, qwen-2.5-7b, ...)
    # - NEAR AI:      gateway_attestation + model_attestations (phala/gpt-oss-120b, deepseek-chat-v3.1, ...)
    # - Chutes:       attestation_type="chutes" + all_attestations[] with e2e_pubkey binding
    # - Tinfoil:      hw policy + sigstore golden values (not yet routed through our curated list)
    if report.get("attestation_type") == "chutes":
        return _verify_redpill_chutes(report, model, nonce)
    if "gateway_attestation" in report:
        gateway_att = report["gateway_attestation"]
        model_atts = report.get("model_attestations", [])
    elif "intel_quote" in report:
        gateway_att = report
        model_atts = []
    else:
        return AttestationReport(
            valid=False, provider="redpill", attestation_type="tdx+gpu",
            verified_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            details={"shape_keys": sorted(list(report.keys()))[:10]},
            error="Unrecognized attestation response format",
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

    # For E2EE: use model attestation's key/algo when present; fall back to gateway's
    if model_atts and model_atts[0].get("signing_public_key"):
        model_signing_key = model_atts[0]["signing_public_key"]
        signing_algo = model_atts[0].get("signing_algo", signing_algo)
    else:
        model_signing_key = gateway_att.get("signing_public_key")
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
        signing_public_key=model_signing_key,
        signing_algo=signing_algo,
    )


def _verify_venice_attestation(runtime_creds: Dict[str, Any], config: Dict[str, Any]) -> AttestationReport:
    """Verify Venice AI TEE attestation.

    Venice exposes an undocumented ``GET /tee/attestation?model=X&nonce=Y`` endpoint
    (base = https://api.venice.ai/api/v1). Response is a Phala-shape bundle with
    ``intel_quote`` (hex), optional ``nvidia_payload``, ``signing_address``,
    ``signing_public_key``, and ``nonce_source``. We re-run TDX via Phala's public
    verifier and NVIDIA NRAS ourselves rather than trusting Venice's own
    ``server_verification`` block.
    """
    api_key = runtime_creds.get("api_key", "")
    base_url = (runtime_creds.get("base_url") or "https://api.venice.ai/api/v1").rstrip("/")
    model = runtime_creds.get("model", "")
    now = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _fail(err: str, details: Optional[Dict[str, Any]] = None) -> AttestationReport:
        return AttestationReport(
            valid=False, provider="venice", attestation_type="tdx+gpu",
            verified_at=now(), details=details or {}, error=err,
        )

    if not api_key:
        return _fail("No API key in runtime credentials")
    if not model:
        return _fail("No model in runtime credentials")

    nonce = secrets.token_hex(32)
    url = f"{base_url}/tee/attestation"
    response = requests.get(
        url, params={"model": model, "nonce": nonce},
        headers={"Authorization": f"Bearer {api_key}"}, timeout=60,
    )
    if response.status_code == 404:
        return _fail("no TEE attestation available for this model (404)")
    response.raise_for_status()
    att = response.json()

    quote_hex = att.get("intel_quote") or att.get("quote") or ""
    if isinstance(quote_hex, dict):
        quote_hex = quote_hex.get("hex") or quote_hex.get("intel_quote") or ""
    if not quote_hex:
        return _fail("no TDX quote in response", {"keys": sorted(att.keys())[:20]})

    nonce_bound = att.get("nonce_source") == "client"

    tdx_resp = requests.post(_PHALA_TDX_VERIFIER, json={"hex": quote_hex}, timeout=30).json()
    quote_body = (tdx_resp.get("quote") or {}).get("body", {})
    if not (tdx_resp.get("quote") or {}).get("verified"):
        msg = (tdx_resp.get("quote") or {}).get("message") or tdx_resp.get("message") or "unspecified"
        return _fail(f"TDX quote verification failed: {msg}", {"tcb_svn": quote_body.get("tee_tcb_svn")})

    signing_address = att.get("signing_address", "")
    signing_public_key = att.get("signing_public_key", "")
    signing_algo = att.get("signing_algo", "ecdsa")

    report_data_hex = quote_body.get("reportdata", "")
    if not _phala_check_report_data(report_data_hex, signing_address, signing_algo, nonce):
        return _fail("TDX report_data does not bind signing_address + nonce")

    if signing_public_key and signing_address:
        from eth_keys.datatypes import PublicKey as _EthPubKey
        pub_bytes = bytes.fromhex(signing_public_key)
        if len(pub_bytes) == 65 and pub_bytes[0] == 0x04:
            pub_bytes = pub_bytes[1:]
        derived = "0x" + _EthPubKey(pub_bytes).to_canonical_address().hex()
        if derived.lower() != signing_address.lower():
            return _fail("signing_public_key does not derive to signing_address",
                         {"derived": derived, "claimed": signing_address})

    mr_config = quote_body.get("mrconfig", "")
    tcb_info = att.get("tcb_info") or (att.get("info") or {}).get("tcb_info") or {}
    if isinstance(tcb_info, str):
        try:
            tcb_info = _json.loads(tcb_info)
        except Exception:
            tcb_info = {}
    app_compose = tcb_info.get("app_compose")
    compose_hash_verified = False
    if app_compose and mr_config:
        compose_hash = hashlib.sha256(app_compose.encode()).hexdigest()
        compose_hash_verified = mr_config.lower().startswith(("0x01" + compose_hash).lower())

    gpu_verdict = None
    nvidia_payload = att.get("nvidia_payload")
    if nvidia_payload:
        if isinstance(nvidia_payload, str):
            nvidia_payload = _json.loads(nvidia_payload)
        gpu_resp = requests.post(_NVIDIA_NRAS, json=nvidia_payload, timeout=30).json()
        gpu_verdict = _decode_nvidia_verdict(gpu_resp[0][1])
        gpu_passed = gpu_verdict is True or gpu_verdict == "PASS"
        if not gpu_passed:
            return _fail(f"NVIDIA GPU attestation failed: {gpu_verdict}",
                         {"gpu_verdict": gpu_verdict})

    return AttestationReport(
        valid=True, provider="venice", attestation_type="tdx+gpu",
        verified_at=now(),
        details={
            "signing_address": signing_address,
            "model": model,
            "nonce_bound": nonce_bound,
            "compose_hash_verified": compose_hash_verified,
            "gpu_verdict": gpu_verdict,
            "tee_provider": att.get("tee_provider"),
            "tee_hardware": att.get("tee_hardware"),
        },
        signing_public_key=signing_public_key or None,
        signing_algo=signing_algo,
    )


def _skip_attestation(reason: str) -> AttestationReport:
    return AttestationReport(
        valid=False, provider="unknown", attestation_type="none",
        verified_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        details={}, error=f"Attestation skipped: {reason}"
    )
