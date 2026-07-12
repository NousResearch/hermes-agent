# Lanyard Brain RPC (host handlers)

Hermes host-side handlers for **Brain RPC** MVP methods. Portal invokes
read-oriented operations on the customer brain host (vault, settings, projects)
over the **existing egress-only relay WebSocket** — no public inbound host port.

## Authority

Cross-repo contract (normative schemas, auth, errors):

- Deploy repo: [`company-brain-deploy/references/brain-rpc-contract.md`](https://github.com/LanyardBrain/company-brain-deploy/blob/main/references/brain-rpc-contract.md)
- Related RBAC: `references/hermes-profile-rbac.md` (same deploy repo)
- Transport substrate (chat relay, not method set): [`docs/relay-connector-contract.md`](./relay-connector-contract.md)

This document is a **Hermes implementation note** only. Do not treat it as a
substitute for the deploy contract. G3 overall readiness is owned by the deploy
readiness matrix; G3.3 (Portal vertical slice) is separate.

## Wire

| Direction | Frame `type` |
|-----------|----------------|
| Portal/relay → Hermes | `brain_rpc_request` |
| Hermes → Portal/relay | `brain_rpc_result` |

Frames are newline-delimited JSON on the **authenticated** relay session the
gateway already dials (`GATEWAY_RELAY_URL` + instance secret). The transport
dispatches in `gateway/relay/ws_transport.py` when `type == brain_rpc_request`.

## MVP methods

| Method | Purpose |
|--------|---------|
| `brain.ping` | Liveness / echo |
| `brain.health` | Structured readiness checks |
| `vault.list` | Directory metadata (capped) |
| `vault.stat` | Single-path metadata |
| `vault.read` | Single-file content (size-capped) |
| `settings.snapshot` | Redacted Hermes settings subset |
| `projects.list` | Hermes `projects.db` rows (empty OK if missing) |

## Auth (fail-closed)

Host re-checks stamped claims (contract §3):

1. **Auth present** — `tenant_id`, `instance_id`, `subject`, `expires_at`
2. **Expiry** — expired → `unauthenticated`
3. **Instance pin** — when `GATEWAY_RELAY_INSTANCE_ID` / `GATEWAY_RELAY_ID` is set, must match
4. **Tenant pin** — when `BRAIN_TENANT_ID` / `GATEWAY_RELAY_TENANT_ID` is set, must match
5. **Host profile** — capabilities from `$HERMES_HOME/profiles/<name>.json` (or builtin admin/contributor seeds)
6. **Path ACL** — vault paths must fall under `subject.path_prefixes` (admin `vault_full` + empty prefixes = full vault)

Channel secret verification (WS upgrade) remains the relay connector’s job.

## Host configuration

| Env | Meaning |
|-----|---------|
| `BRAIN_RPC_ENABLED` | Default `1`. Set `0`/`false` to refuse RPC with `unavailable` |
| `VAULT_ROOT` | Vault filesystem root (preferred). Fallback: `COMPANY_BRAIN_VAULT_ROOT`, then `$HERMES_HOME/vault` |
| `GATEWAY_RELAY_INSTANCE_ID` / `GATEWAY_RELAY_ID` | Local instance pin for L1 |
| `BRAIN_TENANT_ID` / `GATEWAY_RELAY_TENANT_ID` | Optional local tenant pin |
| `BRAIN_PROFILES_DIR` | Override for profile seed JSON directory |

**Logging:** request id, method, tenant/instance/user, duration, error codes.
Vault **file bodies are never logged**.

## Code map

```
gateway/brain_rpc/
  dispatcher.py     # envelope + concurrency bound
  auth.py           # L1/L3/L4/L5
  vault_ops.py      # list / stat / read
  settings_snapshot.py
  projects_handler.py
  handlers.py       # method table
gateway/relay/ws_transport.py   # frame hook → brain_rpc_result
```

## Tests

```bash
scripts/run_tests.sh tests/gateway/brain_rpc/ -q
```

## Non-goals (this track)

- Portal BFF / browser API surface (G3.3)
- Terraform/deploy modules
- Vault write / settings mutate
- Claiming G3 production-ready
