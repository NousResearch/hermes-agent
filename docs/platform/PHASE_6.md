# Phase 6 — The deployment configuration seam

Where a customer's model endpoints, credentials and per-agent provider settings live, and how
they reach a worker without NOVA ever holding a secret. **Core patches: still 1.**

The audit that produced this design is [`DEPLOYMENT_CONFIG_AUDIT.md`](DEPLOYMENT_CONFIG_AUDIT.md).
The gap it closes was found by [`LIVE_RUN.md`](LIVE_RUN.md): provider configuration had
nowhere to live, because a worker reads only its own profile's `config.yaml` and NOVA
rewrites that file on every apply.

---

## The one rule

> **NOVA writes the *name* of a credential and never its value.**

Everything below exists to make that rule survivable in a real deployment rather than merely
true in principle. A design that is secure only until the first operator needs to get a key
to a worker is not secure; it is about to be worked around.

---

## Ownership

| Layer | Path | Owner | Contains |
|---|---|---|---|
| Agent config | `<profile>/config.yaml` | **NOVA**, rewritten every apply | Policy, limits, plugins, provider *shape* |
| Credentials | `<profile>/.env` | **Operator** — on `NEVER_WRITE` | The actual secrets |
| Reference | `key_env` in config | NOVA writes the **name** | Never a value |

The split is not new machinery. `<profile>/.env` is loaded by
`hermes_cli/env_loader.py::load_hermes_dotenv` because a worker's `HERMES_HOME` *is* its
profile, and `.env` was already on `materialize.NEVER_WRITE`. The audit's finding was that
the right ownership line already existed and nothing was using it.

### Why `key_env` rather than `${VAR}`

Both work. `key_env` is better, and the difference is not cosmetic:

- `api_key: ${ACME_LLM_KEY}` expands at config load, so the secret sits in the loaded config
  object for the life of the process.
- `key_env: ACME_LLM_KEY` is resolved at the point of use by
  `auxiliary_client.py::_custom_provider_credential` → `secret_scope.get_secret`, which under
  multiplexing **refuses to borrow another profile's value**.

So `key_env` is both less exposed and per-agent scoped. NOVA emits it and never `api_key`.

---

## Vocabulary

`custom_providers`, `base_url` and `key_env` are Hermes words. An `AgentSpec` containing them
would make a second adapter a rewrite. So NOVA declares in its own terms:

```yaml
# deployment.yaml — operator-owned
provider:
  provider: ${ACME_LLM_PROVIDER:-acme-gateway}
  model: ${ACME_LLM_MODEL:-acme-default}
  endpoint: ${ACME_LLM_URL}
  api_key_env: ACME_LLM_KEY      # the NAME, never the value
  context_window: 200000
```

and `nova/runtime/hermes/provider.py` translates. One file knows `custom_providers` exists.

**Tenant defaults, per-agent overrides, merged field by field.** An agent that needs a cheaper
model sets `model:` alone and keeps the tenant's endpoint and credential. Wholesale
replacement would force it to restate the entire block, and restated config is where
deployments drift.

`endpoint` and `region` are deliberately **not expanded at load**: a `${VAR}` must reach the
worker as a reference and resolve in the worker's environment, which is a different machine
from whoever ran `nova apply`.

---

## Refusing a secret at the only moment it helps

A credential pasted into `deployment.yaml` is in version control, in every clone, and in the
history after it is removed. Refusing it at parse time is the only point where saying no is
still worth anything.

Two checks, because the obvious one is insufficient:

1. **Shape** — `api_key_env` must match `[A-Za-z_][A-Za-z0-9_]*`.
2. **Content** — a recognised credential prefix (`AKIA`, `sk-`, `ghp_`, `AIza`, `xox`, …), or
   a 20+ character run with no underscore and a digit in it.

The second exists because `AKIAIOSFODNN7EXAMPLE` — an AWS access key id — passes the shape
check cleanly. A test asserts both that those are refused and that ordinary names like
`OPENAI_API_KEY` and `K` are not: a check that rejects legitimate names is one people work
around.

The same content check runs over the passthrough block, where NOVA cannot otherwise tell a
credential from configuration.

---

## The passthrough, and what it may not touch

A customer with a private CA or a provider-specific body parameter cannot wait for NOVA to
model it, so `runtime_config` is written verbatim into the agent's runtime configuration.

It **may not set** `plugins`, `approvals`, `agent`, `delegation`, `kanban`, `nova`, `model` or
`custom_providers`. An `approvals` block that re-permits a denied tool, or a `plugins` block
that disables the enforcement plugin, would undo a governance control while every NOVA-side
check still reported it enforced — the failure would be invisible in exactly the review meant
to catch it.

Guarded twice: refused at parse time, and NOVA's own compilation is applied last regardless.

---

## Readiness: reported, never fixed

Because NOVA writes names and not values, "materialized" and "able to run" are different
states. `nova doctor` closes that gap the only way NOVA honestly can:

```
  [NOT READY] customer-support
           provider: acme-gateway   model: scripted-1
           ACME_LLM_KEY: MISSING
           -> add the missing name(s) to <profile>/.env
```

It exits non-zero, so a deployment pipeline fails there rather than at the first task. `apply`
emits the same warnings inline.

Two details worth their code:

- **A defaulted reference is not required.** `${AWS_REGION:-eu-west-1}` resolves with or
  without the variable, so demanding it would send an operator hunting for something never
  needed — and a readiness report that cries wolf is one nobody reads the third time.
- **A native provider's `api_key_env` is not demanded.** `bedrock` resolves its own
  credential and never reads the declared name; the report says so instead of requiring it.

Readiness also distinguishes a variable found in `<profile>/.env` from one found in the
process environment. The second is host-wide — it stops being per-agent the moment a second
tenant lands on the host.

---

## Proven on a fresh deployment

From an empty directory, no manual configuration at any point:

| Step | Result |
|---|---|
| `nova apply` with no credentials | Agents created, **warned** they cannot run |
| `nova doctor` | exit **1**, naming both variables and the file |
| Operator writes two `.env` files | — |
| `nova doctor` | exit **0**, each variable shown with where it resolved |
| `nova knowledge ingest` + `objective submit` | 4 work items |
| Real dispatcher | 4/4 done; parallel steps spawned together |
| `nova objective status` | `done (4/4 steps done)` |
| Re-`apply` | `unchanged=2`, `.env` byte-identical (md5 unchanged) |

The live worker ran with **no `--provider` and no `-m` flag** — the provider resolved entirely
from what NOVA wrote plus what the operator provisioned. It used the knowledge tool and cited
`refunds.md:1-20 (Refund Policy)`.

**Secret containment, verified by grep:** `acme-gateway-secret-value` appears in the two
operator-owned `.env` files and nowhere else — not in `config.yaml`, `SOUL.md`, the audit log,
the compiled policy, or the knowledge grant.

---

## Known limitations

- **`nova doctor` checks presence, not validity.** A wrong key is indistinguishable from a
  right one until a request is made. Detecting that means making a billable call at apply
  time, which is the wrong trade.
- **Provider names are runtime vocabulary.** `NATIVE_PROVIDERS` is a Hermes list; another
  adapter brings its own. Carried opaquely by the spec, interpreted only by the adapter.
- **One `.env` per agent, written by hand.** NOVA deliberately has no `nova secrets set` —
  that would put a credential through NOVA's own process and logs. Real deployments should
  render `.env` from their existing secret manager (`op run`, Vault agent, SOPS), and the
  runtime's own external secret sources remain available underneath.
- **The managed scope is not used.** `/etc/hermes` offers an IT-pushed immutable layer, but it
  is host-global and needs root, and a standing constraint is that NOVA must not require
  permanent root from a customer. Documented as an option, not a mechanism.
