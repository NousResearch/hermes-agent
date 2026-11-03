# Neon Genie — Privacy Contract

**Contract version:** `1.0.0`  
**Skill target:** 3.24.0+  
**Status:** Binding product contract for repository-owned behavior

This document is the human-readable privacy contract for Neon Genie (Hermes skill + packaging CLI). A hub-safe mirror ships as [`references/PRIVACY.md`](references/PRIVACY.md). Compact machine summary: [`references/privacy-contract.md`](references/privacy-contract.md). Runtime engine: [`scripts/privacy_runtime.py`](scripts/privacy_runtime.py). Provenance appears on run receipts and envelopes (nested `privacy` context + top-level `privacy_mode`).

---

## 1. Summary

Neon Genie is **privacy-by-construction** in the limited sense that this repository:

- Declares what *it* does with data (packaging, receipts, doctrine, gates).
- Keeps **repository-owned telemetry off** (no collector, no silent usage pipeline).
- Records egress attempts and privacy mode on sealed artifacts when the spine is in use.
- Blocks high-confidence secrets from *packaging* egress preflight where that tool is invoked.

It does **not** claim that prompts, files, or model traffic “never leave your device.” Host runtimes (Hermes, model providers, search/MCP tools, OS, cloud IDEs) have their own retention and logging. Unsupported privacy state is recorded as `NOT_COMPUTABLE` and/or `privacy_warnings[]`—never as a warm-blanket guarantee.

**Wayfinder** is an **optional** handoff consumer. It is not part of Neon’s privacy boundary and is never required for doctor, privacy checks, or recipes.

---

## 2. Data-flow boundary (Neon vs host/provider)

Neon can **guarantee** only repository-owned behavior. Everything else is **disclosed**, not certified.

| Guaranteed by Neon | Not guaranteed (disclose only) |
|--------------------|--------------------------------|
| Files it writes under operator-selected output paths | Hermes host retention / logging |
| Its own telemetry flag (`disabled`; no collector in-repo) | Model provider training / logs |
| Doctrine + CLEAR gates + packaging validation | Search / web / MCP vendor policies |
| What it records on the receipt / envelope privacy summary | OS, browser, cloud IDE, or third-party side channels |
| Deterministic secret preflight on strings/paths it is asked to check | Memory of an already-running remote model session |

### Layers (design §3.2)

```text
Operator (prompts, files, workspace, flags)
        │
        ▼
Hermes prose runtime (Tier 0)
  SKILL.md + core + privacy
  OPEN → ALIGN → ASCEND → CLEAR → SEAL
  RUNE.PRIVACY_EGRESS_CHECK before host research/tool
        │                              │
        │ local only                   │ ALLOW / REDACT_THEN_ALLOW
        ▼                              ▼
Operator out/ path              Host tools (optional)
packets, receipt,               web / search / fetch / MCP
envelope, ledger*               = external boundary

* learning ledger: explicit operator `do learn` only
```

**Dual enforcement (honest):**

| Path | What enforces privacy |
|------|------------------------|
| Live Hermes run | Doctrine in SKILL + `profiles/privacy.md` + gates; agent logs `external_actions` / redaction on receipt |
| Packaging CLI | `privacy_runtime` context into receipts/envelopes; preflight; doctor / `do privacy --json`; fixtures |
| Neither | May control a frontier model’s own logging — disclosed as host boundary |

Neon does **not** proxy host research in Python. The packaging CLI makes no network research calls.

---

## 3. Data categories

| Category | Meaning |
|----------|---------|
| `operator_input` | Prompts, pasted text, attached files the operator supplies |
| `workspace_private` | Local repo/files Hermes can read under operator control |
| `public_web` | Fetched or search results from the public web |
| `provider_model` | Content sent to a model host (when the host path is used) |
| Neon artifacts | Packets, receipt, envelope, optional learn ledger under operator `out/…` |

Receipt field `data_sources_used` lists the subset actually used for a run. Packaging defaults typically include `operator_input`.

---

## 4. Privacy modes

Runtime / receipt vocabulary (canonical, lowercase):

| Mode | Meaning |
|------|---------|
| `local_only` | Default packaging mode. No Neon-initiated external research **sends**; `egress.allowed=false`; `external_actions` empty or only blocked/denied records |
| `external_research_allowed` | Host research may run under purpose-bound rules; every successful egress should be logged |
| `custom` | Explicit operator configuration (still no global privacy-disable) |

Doctrine docs may also use uppercase aliases (`LOCAL_ONLY`, `EXTERNAL_RESEARCH_ALLOWED`, `UNKNOWN_HOST_BOUNDARY`). **Packaging receipts follow the runtime schema** (`local_only`, …).

`local_only` means **Neon did not record a successful external research send**. It does **not** mean the model weights, host session, or provider logs are local. Absolute “never leaves your device” claims without a fully local stack are forbidden (Gate **W** / `NOT_COMPUTABLE`).

---

## 5. Telemetry

- **Status:** `telemetry_status` is **`disabled`** for this contract version (W1).
- There is **no** usage/analytics collector in this repository.
- Validation fails if telemetry is anything other than `disabled` (Gate **X**).
- Explicit operator `do learn` (learning ledger) is **not** telemetry; it is a separate, path-local optional artifact.

---

## 6. Training

- **This repository does not train models** on operator data.
- **Host / model providers may train or log** according to *their* policies. Neon does not control or certify that surface.
- Do not state “your data is never used for training” unless the entire stack (host + provider + tools) is proven to forbid it—and then document that proof outside Neon guarantees.

---

## 7. Offline / research

- Operator flags map into receipt `research_policy` (`enabled`, `offline`) and influence `privacy_mode`.
- Under `local_only` / research disabled: packaging and doctrine require **zero** successful external sends (Gate **T**; `NG-PRIV-003`).
- Offline requested but host cannot enforce it → prefer `UNKNOWN_HOST_BOUNDARY` + warnings rather than false offline claims.

---

## 8. External research disclosure

When research is allowed:

1. Classify and minimize payload (`RUNE.PRIVACY_CLASSIFY` / `RUNE.PRIVACY_MINIMIZE`).
2. Run `RUNE.PRIVACY_EGRESS_CHECK` before host web/search/fetch/MCP use.
3. Outcomes: `ALLOW` | `REDACT_THEN_ALLOW` | `REQUEST_CONSENT` | `BLOCK`.
4. Log each attempt in `external_actions[]` (destination known when `sent: true`—Gate **S**).
5. Credentials/secret-like material must not be sent (Gate **U**); private/operator material needs consent (Gate **V**).

Successful external research should appear in both evidence-spine research attempts and `external_actions`. Blocked preflight may appear only in `external_actions` + `redaction.events`.

Remote model sends (when disclosed) use destination style `model:<provider>` and category `provider_model`. Unknown provider → do not claim local-only privacy.

---

## 9. Artifacts, retention, deletion

**Where Neon writes:** operator-selected output roots (commonly under `out/…`): packets, `run-receipt.json`, `run-envelope.json`, optional notices, optional learn ledger.

**Retention:** Neon retains only what remains on those local paths. Host/provider retention is **unknown unless the operator knows their stack**—never claimed zero by Neon.

**Deletion (operator steps):**

1. Delete the run output directory listed in receipt `artifact_paths` / `deletion_instructions`.
2. If a learning ledger path was used, delete that file explicitly.
3. Clear host chat history / provider logs / tool caches according to **host** documentation (outside Neon).
4. Revoke any credentials that may have been pasted into prompts (ops hygiene; preflight may have blocked egress but not host memory).

Receipt fields: `retention_statement`, `deletion_instructions`, `artifact_paths`.

---

## 10. Secret / PII preflight scope

Module: `scripts/privacy_preflight.py` (stdlib).

| Scope | Behavior |
|-------|----------|
| High-precision | API keys, PEMs/private keys, Bearer tokens, common cloud key shapes, password-in-URL / password assignments, crude PAN patterns → **block** egress (`safe_for_egress: false`) |
| Best-effort / documented | Broader PII (government IDs, contact lists, precise geo, special category) — not a complete detector |
| Not in scope | Whole-disk scanning; scanning paths not explicitly passed; guaranteeing host session redaction |

Local packet drafting remains allowed when the operator task requires private material **on disk**; external queries must not carry blocked categories. Packaging preflight does not replace host-side policy.

---

## 11. Gates S–Y pointer

Privacy CLEAR gates (after authority and evidence P–R). Full registration: [`references/gates.yaml`](references/gates.yaml) and [`references/anti-overclaim-patterns.md`](references/anti-overclaim-patterns.md).

| Gate | Name | Blocking when |
|------|------|---------------|
| **S** | `egress_destination_known` | `sent: true` and destination unknown/empty |
| **T** | `offline_no_external_send` | `local_only` / research disabled but successful external send recorded |
| **U** | `secret_no_egress` | Credential/secret-like payload would be or was sent |
| **V** | `private_egress_needs_consent` | Private/operator egress without `consent_ref` |
| **W** | `privacy_claim_must_be_supported` | Absolute privacy claim without matching mode + evidence |
| **X** | `telemetry_off_unless_opt_in` | `telemetry_status != disabled` (W1) |
| **Y** | `sealed_receipt_privacy_complete` | SEAL without required privacy provenance |

---

## 12. What we cannot guarantee

- That prompts never reach a remote model or provider log.
- That “offline mode” is enforced by Hermes/OS without host cooperation.
- Complete PII detection or redaction of every sensitive string.
- Search, MCP, browser, or IDE vendor behavior.
- Wayfinder (or any remote consumer) handling of handoff packets after the operator pushes them.
- Legal compliance certifications (GDPR/CCPA/etc. as a product claim).
- Absolute “never leaves your device” without proof of a fully local stack.

When evidence is missing: use `NOT_COMPUTABLE`, `privacy_warnings[]`, and/or `UNKNOWN_HOST_BOUNDARY`—not invented privacy.

---

## 13. Contact / issues

Questions, corrections, and privacy incidents related to this skill:

- **Issues:** https://github.com/scrimshawlife-ctrl/NeonGenie/issues  
- Prefer labeling privacy contract gaps clearly so `privacy_contract_version` can bump when semantics change.

Related docs: ADR [0006 — Privacy by construction](docs/adr/0006-privacy-by-construction.md), ADR [0004 — Wayfinder boundary](docs/adr/0004-neon-genie-wayfinder-boundary.md), design spec `docs/superpowers/specs/2026-08-06-neon-genie-privacy-spine-design.md`.

## 14. Runtime engine & purpose-bound consent (integrated #17)

Packaging runs use `scripts/privacy_runtime.py` with fail-closed defaults (`local_only`,
telemetry `disabled`). There is **no** global privacy-disable switch; overrides are
**purpose-bound consent records** only.

Briefs/recipes may carry a `privacy:` section that flows into receipts, envelopes,
and learning-ledger disclosures. Diagnostics:

```bash
python scripts/neon_genie.py do privacy --json
python scripts/test_privacy_runtime.py
```

Mode vocabulary in the runtime/context schema uses lowercase (`local_only`,
`external_research_allowed`, `custom`). Doctrine docs may also use uppercase
aliases (`LOCAL_ONLY`, …); packaging receipts follow the runtime schema.

See also: `references/privacy-contract.md`, `schemas/privacy-context.schema.json`.

