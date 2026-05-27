# Secrets Agent Operations

This document is for Hermes/Codex-style agents that work in the repo without
prior local context.

## Mental model

- Bitwarden Secrets Manager is the vault.
- `secrets-manifest.yaml` is the BWS SSOT ledger.
- `.env` is a runtime projection generated from that ledger.

Do not treat a source integration as the SSOT. The policy gate is the manifest
plus the projection sidecar and preflight checks.

## Commands

Safe read/check commands:

```bash
hermes secrets check
hermes secrets preflight
hermes secrets source bitwarden status
```

Source refresh:

```bash
hermes secrets source bitwarden refresh
```

This fetches from Bitwarden and reports what would enter the process. It is not
the manifest-to-`.env` projection sync.

SSOT projection sync:

```bash
hermes secrets sync --target <target> --apply
```

This updates the runtime `.env` projection from the BWS manifest. It requires
current user approval because it mutates secret-bearing runtime state.

## Agent rules

- Never print secret values, `.env` contents, tokens, or raw BWS payloads.
- Prefer `check`, `preflight`, and `status` when diagnosing.
- Ask for explicit approval before `sync --apply`, `source bitwarden setup`,
  `source bitwarden refresh --apply`, token changes, project changes, or any
  `.env` write.
- If unsure whether a command mutates secret-bearing state, treat it as
  approval-required.
