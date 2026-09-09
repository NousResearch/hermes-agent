---
name: nvidia-aiq-signal-discovery
description: Run governed NVIDIA signal research for LunaBot.
version: 0.1.0
author: Mike DeMott (mrkillbob), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [NVIDIA, AI-Q, Quantitative Research, LunaBot, Governance]
    category: mlops
    config:
      - key: AIQ_SERVER_URL
        description: Trusted NVIDIA AI-Q service endpoint.
        prompt: Enter the trusted NVIDIA AI-Q service URL.
---

# NVIDIA AI-Q Signal Discovery Skill

Use this skill to coordinate NVIDIA AI-Q, the Quantitative Signal Discovery
blueprint, and optional NemoClaw isolation for LunaBot research. It produces
research receipts only; it does not place orders, alter live signal gates, or
promote a generated signal.

## When to Use

- The user asks Hermes to discover, optimize, or compare quantitative signals.
- A result from NVIDIA's Quantitative Signal Discovery Agent must be attached
  to LunaBot with explicit provenance.
- A research run needs a durable report, bounded iteration count, or sandbox
  boundary.

Do not use this skill for live-cycle diagnosis, paper-order submission, or
broker/data-authority changes.

## Prerequisites

- A trusted AI-Q server URL declared as skill config key `AIQ_SERVER_URL` and
  injected by Hermes when this skill loads.
- NVIDIA credentials configured in the AI-Q service, not copied into Hermes
  prompts or receipts.
- A LunaBot checkout resolved from the operator's configured project root.
- The skill-relative client `scripts/aiq_client.py` and a JSON request file.
- If using NemoClaw, a supported NVIDIA OpenShell host and an explicit user
  decision to install or run its preview tooling.

Before sending private research data, use `read_file` or `terminal` to confirm
that `AIQ_SERVER_URL` points to the intended trusted service. Never paste API
keys into a request, `SKILL.md`, or generated receipt.

## How to Run

1. Use `terminal` to record the LunaBot root, branch, HEAD, dirty state, and
   data source before starting.
2. Submit a JSON request, poll the returned job, and download its result with
   the bundled client. Preserve its job ID, profile, report ID, and artifact
   paths. The client uses this concrete HTTP contract:

   ```text
   POST <AIQ_SERVER_URL>/v1/jobs       request JSON; response JSON includes job_id
   GET  <AIQ_SERVER_URL>/v1/jobs/<id>  job status JSON
   GET  <AIQ_SERVER_URL>/v1/jobs/<id>/result  completed result JSON
   ```

   ```text
   python3 scripts/aiq_client.py --server "<configured AIQ_SERVER_URL>" submit request.json
   python3 scripts/aiq_client.py --server "<configured AIQ_SERVER_URL>" status <job-id>
   python3 scripts/aiq_client.py --server "<configured AIQ_SERVER_URL>" download <job-id> result.json
   ```
   Override endpoint paths with `--path` or `--path-template` when the
   trusted deployment exposes the same contract under a gateway prefix.
3. Run the NVIDIA Quantitative Signal Discovery workflow in its own external
   checkout with a bounded iteration count and the no-telemetry profile.
4. Import only its structured JSON result through the LunaBot bridge:

   ```text
   python3 <lunabot-root>/scripts/bridge_nvidia_signal_result.py \
     --input <nvidia-result.json> \
     --output <lunabot-artifacts>/nvidia-signal-receipt.json \
     --data-source <explicit-dataset-label>
   ```

5. Use `read_file` to inspect the receipt and verify that its authority fields
   remain false. Treat `accepted` as “accepted by the external research
   metric,” never as LunaBot acceptance.
6. Report the result as a candidate for independent LunaBot backtesting. A
   candidate must be re-evaluated on LunaBot-authoritative data before any
   engineering review.

## Quick Reference

| Surface | Allowed outcome |
| --- | --- |
| NVIDIA workflow | Signal ideas, code, metrics, feedback |
| AI-Q | Research plan, report, durable artifacts |
| LunaBot bridge | Hash-bound research receipt |
| Live or paper execution | No access |
| Promotion | No access |

## Procedure

### 1. Establish the evidence boundary

Use `search_files` and `read_file` to locate the active LunaBot root and its
research-only contract. Record the exact checkout identity. Do not run the
workflow from a dirty live/paper checkout when a clean research worktree is
available.

### 2. Check the data source

The upstream blueprint defaults to a generic stock-data loader. That is useful
for discovery but is not Schwab/Binance authority. Pass a descriptive source
label to the bridge and, when available, the SHA-256 of the evaluated dataset.
Missing dataset digest means the receipt remains explicitly unverified
external research.

### 3. Preserve the AI-Q/NemoClaw boundary

AI-Q reports and artifacts are external research evidence. NemoClaw/OpenShell
is an optional execution boundary for the agent, not a LunaBot runtime
dependency. If OpenShell is unavailable, record `environment_blocked` and keep
the research receipt; do not silently run an unrestricted replacement.

### 4. Review the result

Check request, thresholds, metrics, selected signal, result-file digest, data
source, and the receipt hash. Use `patch` only for a requested follow-up
change, never to rewrite a failed metric or authority flag.

## Pitfalls

- NVIDIA “signal accepted” is not LunaBot signal acceptance.
- A generated Python signal is untrusted research code; do not copy it into
  `tradingbot/` runtime or execution modules.
- A yfinance or hosted-NIM result does not establish Schwab/Binance authority.
- AI-Q endpoint health does not prove the research report is complete.
- NemoClaw alpha/preview status is not production isolation evidence.

## Verification

Run the LunaBot bridge tests and validate the emitted receipt:

```text
python3 -m pytest <lunabot-root>/tests/test_nvidia_signal_discovery.py -q
python3 <lunabot-root>/scripts/bridge_nvidia_signal_result.py --help
```

The required final classification is one of: `research_only`,
`environment_blocked`, or `failed`. This skill never produces a live or paper
submission authorization.
