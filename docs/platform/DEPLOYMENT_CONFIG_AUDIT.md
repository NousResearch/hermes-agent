# Provider and deployment configuration — audit

How customer-owned model endpoints, environment variables, secrets and per-agent provider
settings actually reach a Hermes worker. Every claim below was verified at its call site or
proven empirically; nothing here is inferred from naming.

---

## 1. A worker's config is exactly one file

`hermes_cli/config.py::_load_config_impl` reads `get_config_path()` — a single file — merged
over `DEFAULT_CONFIG`, then env-expanded, then overlaid with managed scope. **There is no
profile→root inheritance.**

The dispatcher sets `HERMES_HOME` to the profile directory
(`kanban_db_dispatch.py::_default_spawn` → `resolve_profile_env`), so a worker's config is
`<profile>/config.yaml` and nothing else. Provider settings in the runtime home's
`config.yaml` are invisible to it.

That is the whole cause of the gap: NOVA owns and rewrites `<profile>/config.yaml` on every
apply, so an operator has nowhere to put an endpoint that survives.

## 2. `<profile>/.env` is already a per-agent secret layer

`hermes_cli/env_loader.py::load_hermes_dotenv` loads `<HERMES_HOME>/.env` with
`override=True`. With `HERMES_HOME` pointing at the profile, that is a **per-agent env file
the runtime already reads**.

It is also already on NOVA's `materialize.NEVER_WRITE`, so NOVA refuses to write it today —
which turns out to be exactly the right ownership line rather than an accident.

Proven:

```
dotenv files loaded: ['/tmp/probe/profiles/p1/.env']
NOVA_PROBE_SECRET in env: from-profile-dotenv
```

## 3. Config can reference env without containing it

`_load_config_impl` calls `_expand_env_vars`, so `${VAR}` in a profile's `config.yaml`
resolves against the environment that `.env` just populated. Proven:

```yaml
base_url: ${SCRIPTED_BASE_URL}      ->  http://127.0.0.1:8904/v1
api_key:  ${NOVA_PROBE_SECRET}      ->  from-profile-dotenv
```

## 4. `key_env` is better than `${VAR}` for credentials

`custom_providers[]` accepts `key_env` (`_VALID_CUSTOM_PROVIDER_FIELDS`), resolved in
`auxiliary_client.py::_custom_provider_credential` in the order
`api_key → key_env → key_cmd → credential pool`.

`key_env` reads through `_scoped_key_env` → `agent/secret_scope.py::get_secret`, which under
multiplexing **refuses to borrow another profile's value** rather than falling back to
`os.environ`. So the credential is per-agent scoped at read time.

This is strictly better than `api_key: ${VAR}`: with `${VAR}` the secret is expanded into the
in-memory config object at load; with `key_env` only the *name* ever appears in config, and
the value is fetched at the point of use through the scope.

**Design consequence: NOVA emits `key_env`, never `api_key`.**

## 5. The managed-scope layer exists but is the wrong shape

`hermes_cli/managed_scope.py` provides an IT-pushed, user-immutable config and env layer at
`/etc/hermes` (or `$HERMES_MANAGED_DIR`), merged last so it wins at the leaf.

Genuinely useful, and genuinely not the answer here:

- **Host-global**, not per-profile — it cannot express "this agent uses the cheap model".
- **Needs root** to write `/etc/hermes`. A standing constraint is that NOVA must not require
  permanent root/admin credentials from the customer.
- **Overrides rather than fills gaps**, so it would silently win over a NOVA-compiled key and
  make a governance control untrue without changing anything NOVA can see.

Worth documenting as a deployment option for a customer who already manages hosts that way;
wrong as NOVA's mechanism.

## 6. What the dispatcher does to the worker's environment

`_default_spawn` builds the child env with
`tools/environments/local.py::build_subprocess_env(scrub_secrets=is_multiplex_active(), inherit_profile_home=True)`,
then pops every key in `gateway/session_context._VAR_MAP` so a detached worker cannot inherit
routing from a previous gateway turn, then injects `HERMES_HOME`, `HERMES_TENANT`,
`HERMES_KANBAN_TASK` and the workspace.

So process-env injection (systemd, `op run`) reaches a worker, but is **host-wide** — every
agent on the box gets it. Per-agent separation requires `<profile>/.env`.

---

## Ownership, as it must end up

| Layer | Path | Owner | Contains |
|---|---|---|---|
| Agent config | `<profile>/config.yaml` | **NOVA** — rewritten every apply | Compiled policy, limits, plugins, provider *shape* |
| Secrets | `<profile>/.env` | **Operator** — NOVA never writes | API keys, tokens |
| Credential reference | `config.yaml` `key_env` | NOVA writes the **name** | Never a value |
| Host env | process environment | Operator / init system | Host-wide only |
| Immutable policy | `/etc/hermes` | IT, root | Optional, host-global |

---

## Four requirements this produces

1. **NOVA must be able to write provider shape** — endpoint, model, context window, and the
   *name* of a credential variable — into the profile config it already owns.
2. **NOVA must never write a secret value**, and must refuse one that is handed to it rather
   than trusting the author to know better.
3. **The declaration must stay runtime-agnostic.** `custom_providers` is Hermes vocabulary;
   it may not appear in an `AgentSpec`. NOVA declares in its own terms and the adapter
   translates — the same rule that keeps `profile`, `kanban` and `soul` out of the contract.
4. **Apply must report readiness**, not guess it: which variables each agent needs, which are
   missing, and which file the operator has to create. An agent materialized against a
   credential nobody has provisioned should be visible before it fails at 3am, and NOVA
   cannot fix it by writing the secret itself.
