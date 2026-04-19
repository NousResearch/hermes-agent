---
name: hermes-attestation-inspection
description: Inspect Hermes Agent TEE attestation state — read live verification cache, explain what was verified for the current model, and point at the ground-truth verifier source.
version: 2.0.0
author: amiller (Hermes Agent)
license: MIT
metadata:
  hermes:
    tags: [attestation, tdx, gpu, nras, phala, redpill, near-ai, e2ee]
    related_skills: []
---

# Overview

Hermes supports two TEE-attested inference providers: `near-ai` and `redpill`.
Both verify **Intel TDX** (CPU confidentiality) plus **NVIDIA GPU attestation
via NRAS** (not SGX). Verification happens per-model at first use and is cached
in-process. This skill lets you answer "is attestation working for this model
right now?" from the live cache, rather than guessing.

Ground-truth verifier: `hermes_cli/attestation.py` (import as `hermes_cli.attestation`).

# What gets verified

For **near-ai** (`_verify_near_ai_attestation`):
1. `GET {base_url}/v1/attestation/report?model=...&nonce=...` returns a gateway
   TDX quote + one or more per-model TDX quotes + NVIDIA payloads.
2. Gateway quote verified via the vendored `model_verifier.check_tdx_quote`.
3. `check_report_data` asserts the quote's `report_data` binds the gateway
   signing address + TLS fingerprint and embeds our nonce.
4. Each model attestation: TDX quote, nonce-binding, NRAS GPU attestation
   (`check_gpu`), and that its `signing_public_key` derives to `signing_address`
   (so the E2EE key is hardware-bound).

For **redpill** (`_verify_redpill_attestation`): dispatches on response shape.
Four backends, four shapes:

| Shape                                     | Verifier path              | Example models                          |
|-------------------------------------------|----------------------------|------------------------------------------|
| top-level `intel_quote` (phala-simple)    | same-process Phala + NRAS  | `phala/gpt-oss-20b`, `phala/qwen-2.5-7b` |
| `gateway_attestation` + `model_attestations[]` | NEAR AI–backed        | `phala/gpt-oss-120b`, `phala/glm-4.7`    |
| `attestation_type: "chutes"` + `all_attestations[]` | `_verify_redpill_chutes` | `phala/deepseek-v3.2`, `phala/kimi-k2.5` |
| Tinfoil hw-policy (not yet exercised)     | not implemented            | —                                        |

Chutes has an extra anti-tamper check: `SHA256(nonce ‖ e2e_pubkey) == report_data[0:32]`.

Phala TDX verifier endpoint (used by both paths):
`POST https://cloud-api.phala.network/api/v1/attestations/verify` body `{"hex": <quote_hex>}`.

NRAS endpoint: `POST https://nras.attestation.nvidia.com/v3/attest/gpu`.

# Steps

1. **Check whether attestation is enabled/strict**
   ```bash
   grep -A3 attestation ~/.hermes-near-test/config.yaml
   ```
   `enabled: true` turns on signing; `strict: true` makes unverified models fail
   loudly instead of surfacing a warning.

2. **Query the live cache from inside hermes**
   The verdict for a currently-used model lives in `_MODEL_ATTESTATION_CACHE`:
   ```python
   from hermes_cli.attestation import get_model_attestation_status
   r = get_model_attestation_status("near-ai", "zai-org/GLM-5.1-FP8")
   print(r.valid, r.attestation_type, r.error)
   print(r.details)  # gateway + per-model: signing_address, app_id, TCB status, GPU verdict
   ```
   Or dump everything the process has verified:
   ```python
   from hermes_cli.attestation import _ATTESTATION_CACHE
   for (provider, base, model), (rep, ts, fpr) in _ATTESTATION_CACHE.items():
       print(f"{provider}/{model}: valid={rep.valid} err={rep.error}")
   ```

3. **Re-run a verification on demand**
   ```python
   from hermes_cli.attestation import verify_attestation
   rep = verify_attestation("near-ai",
       {"api_key": "...", "base_url": "https://cloud-api.near.ai", "model": "..."},
       {})
   ```
   Set `HERMES_ATTESTATION_VERBOSE=1` before starting hermes to have the
   underlying verifier's stdout printed; otherwise it's captured into
   `report.verifier_output` (logged at DEBUG).

4. **Interpret common failure modes**
   - `TDX quote verification failed: ppid=<...> tcb_svn=<...>` — firmware on that
     host is unpatched (Intel SA advisory). Fleet issue, not a client bug.
   - `NVIDIA GPU attestation failed: False` — NRAS returned `False` for that
     GPU. Persistent on several NEAR AI shared-pool GPUs.
   - `Unrecognized attestation response format` — redpill routed to a backend
     we don't yet dispatch (e.g. Tinfoil); check `details.shape_keys`.
   - `OutOfDate` status is logged as a warning but does **not** fail the quote —
     the quote is cryptographically valid, just not on the latest TCB.

5. **Cache semantics**
   - Key: `(provider_id, base_url, model_id)`.
   - Success TTL: `HERMES_ATTESTATION_TTL` env (default 3600s).
   - Failure TTL: 60s — failed attestations re-try quickly so a flapping model
     recovers without a restart.
   - TLS cert pinning: cached success is invalidated if the live leaf cert's
     SHA-256 differs from the fingerprint pinned at verification time.

6. **Checking attestation status in practice**
   The caches (`_ATTESTATION_CACHE`, `_MODEL_ATTESTATION_CACHE`) live in the
   Hermes process's memory. You **cannot** meaningfully query them from a
   spawned subprocess — it gets empty dicts plus no API keys. Instead, use the
   log file as the ground truth for what the running process has verified:
   ```bash
   grep -i 'attest' ~/.hermes-near-test/logs/agent.log | tail -20
   ```
   This shows: valid TDX quotes, OutOfDate TCB warnings, NRAS GPU retries,
   and full verification outcomes. Use this as the primary method when you
   need to know "did attestation succeed for the current session?"

# What this does NOT give you

- **No persistent DB.** Attestation evidence is not stored in `state.db`. It
  lives only in the `_ATTESTATION_CACHE` / `_MODEL_ATTESTATION_CACHE` dicts of
  the running process. A hermes restart means a re-verify on next use.
- **No manifest-derived app_id.** The `app_id` is whatever the CVM's `tcb_info`
  reports (read from the TDX quote); it's not computed client-side from a
  manifest file.
- **SGX is not used anywhere** in this code path. If a message mentions SGX,
  it's wrong.

# Verification

After a chat round-trip against a strict-attested model, `get_model_attestation_status`
should return a report with `valid=True` and a populated `signing_public_key`.
If the CLI's E2EE proxy is in use, that key is what the local proxy encrypts
outgoing messages to.
