# Hermes-Core Product Pivot Spec

## Purpose

This document captures the current product pivot direction for the `hermes-core` fork.

The fork is no longer treated as just a lightly customized runtime dependency for a separate MYNAH product. Instead, the target is a supplier-curated `hermes-core` distribution that can be installed, configured, and operated as a real local multi-user product.

In this model:

- the primary codebase is the Hermes fork
- end users do not see MYNAH as the product brand by default
- visible branding comes from setup-time product configuration
- MYNAH remains the maintainer and integration layer around the fork

## Product Direction

The product should be installed through a CLI flow similar to installing Hermes itself.

After installation, a guided setup CLI should configure:

- visible product name and branding shown in the web app
- agent personality seed content, derived from configurable markdown content
- enabled user-facing tools
- execution placement for tools that are allowed to run either inside or outside the runtime sandbox
- initial global model or API route
- optional Tailscale exposure
- local identity-provider integration
- the first admin account

After setup, the product should start:

- a lightweight authenticated local web app
- optional tailnet exposure when enabled
- isolated per-user Hermes runtimes

The setup CLI should write a single product-owned configuration file at:

- `HERMES_HOME/product.yaml`

That product config should be the deployment source of truth for:

- branding and product name
- agent personality seed content
- enabled tool profiles
- placement for `selectable` tools
- initial model or API route
- optional Tailscale exposure
- identity-provider integration settings
- first-admin bootstrap metadata

The first implementation should distinguish:

- `network.bind_host` for local service binding
- `network.public_host` for generated local URLs and OIDC-facing origins

For the `Kanidm`-backed product flow:

- `network.public_host` must be a hostname or domain, not a raw IP address
- the default local value should be `localhost`
- setup must ask for the user-facing hostname separately from the bind address

The web app and runtime launcher should derive their behavior from this product config rather than becoming independent configuration authorities.

The first implementation should keep Hermes' existing `config.yaml` for generic Hermes behavior and use `product.yaml` only for setup-owned product deployment behavior.

## Authentication Direction

The preferred identity system for this pivot is `Kanidm`.

Reasons:

- fully local onboarding is required
- SMTP-backed signup and recovery should not be required
- the product needs local-first user creation, activation, and recovery flows
- the supplier must be able to bootstrap the first admin during setup
- later admins must be able to create additional users from the web UI

The intended identity model is:

- `Kanidm` provides identity, login, credential enrollment, and recovery flows
- the product remains an OIDC client and still owns its own application session and authorization model
- bundled `Kanidm` should run as a local Docker-managed service in the first implementation
- the supplier creates the first admin during setup
- later users are created by an admin in the web UI
- user recovery should use `Kanidm` native recovery actions rather than a product-owned password reset flow

The bundled `Kanidm` service files should be generated under:

- `HERMES_HOME/product/services/kanidm`

The first implementation should generate:

- a Docker Compose file for the bundled `Kanidm` service
- a local `server.toml` for the service
- a local OIDC client secret reference stored outside `product.yaml`
- local TLS material generated before service startup

The first implementation should also treat these startup details as part of the contract:

- `Kanidm` certificate generation must happen before `compose up`
- the generated service should run `kanidmd` directly, not through a shell wrapper
- the bundled container should run with the host uid/gid when available so the bind-mounted service directory remains accessible without loosening permissions

The intended user-lifecycle flow is:

1. supplier setup creates the first admin account
2. admin signs into the authenticated local web app through `Kanidm`
3. admin creates additional users in the web UI
4. the product service provisions those users into `Kanidm`
5. admin issues a native `Kanidm` recovery or reset action for first access
6. the user completes credential enrollment through `Kanidm`
7. the product continues to treat `Kanidm` as the login authority and OIDC provider

The first setup flow should start the stack automatically after config generation and first-admin bootstrap.

The setup CLI should allow an auth-mode choice:

- `passkey`
- `password`

The default should be:

- `passkey`

Hybrid auth mode should not be part of the first design.

`Pocket ID` remains a possible later alternative if passkey-first simplicity becomes more valuable than the added depth of `Kanidm`.

## Runtime and Tool Model

Hermes should remain the runtime core.

MYNAH-specific behavior should stay concentrated near the runtime edge rather than spreading through Hermes core.

Runtime identity should continue to come from `SOUL.md`.

The product service should configure and launch isolated per-user Hermes runtimes.

In the first implementation:

- the product app should read runtime defaults from `HERMES_HOME/product.yaml`
- explicit runtime env vars should still override product-config defaults when set
- per-user runtimes should remain separate isolated processes or containers managed by the product app

The product should treat tools as product-managed capabilities with execution metadata.

Each tool should eventually declare an execution policy such as:

- `inside_only`
- `outside_only`
- `selectable`

The setup CLI should only expose placement choices for tools marked `selectable`.

The preferred execution split is:

- file and workspace tools run inside the isolated user runtime
- admin, auth, routing, and infrastructure tools run outside the runtime
- selected product tools may be configurable at setup if they are safe to support in both placements

The first product policy should treat these categories as fixed:

- `inside_only`:
  - workspace file read, write, patch, and search
  - runtime-local reasoning helpers
- `outside_only`:
  - user provisioning and recovery
  - model routing and provider orchestration
  - Tailscale and network control
  - audit-aware control-plane operations
- `selectable`:
  - web search
  - browser automation
  - image generation
  - code-execution-style helper tools if they can be exposed safely in either placement

The browser admin UI should remain narrow in the first version and only manage:

- users
- activation and reset actions
- runtime visibility and status

Branding, auth mode, tool placement, and model route settings should remain setup-managed product config rather than browser-managed settings.

## User Workspace Model

The product should support a per-user workspace exposed through the web UI.

Users should be able to upload files into that workspace for their own agent to use.

The preferred model is:

- user files are live-mounted into the user runtime workspace
- the mounted workspace is user-scoped, not a general machine filesystem
- the product makes it clear that files handed to the agent can be inspected and edited by that agent
- runtime policy still prevents access outside the allowed workspace
- agent edits are direct live edits in the mounted workspace in the first version

This favors real agent usefulness over a purely indirect file-access model.

The product should therefore guarantee:

- the mounted workspace is the only writable user file area exposed to the runtime
- the runtime cannot browse the general host filesystem
- uploads through the web UI are an intentional handoff of files to the agent
- the UI clearly communicates that uploaded files may be modified by the agent
- deletion and overwrite behavior are treated as normal direct edits rather than delayed sync actions in the first version

The canonical storage root for the first implementation should live under `HERMES_HOME`, with per-user workspace and runtime state derived from that root.

## Upstream Compatibility Rule

Changes to upstream Hermes must stay minimal.

This fork should preserve upstream compatibility as much as possible by preferring:

- setup-driven product configuration
- runtime-edge wrappers
- MYNAH-specific toolsets and env-controlled behavior
- narrow product services outside the Hermes core

This fork should avoid deep product-specific changes in generic Hermes internals unless there is no reasonable configuration or wrapper-based alternative.

If a required product behavior can be expressed through:

- `SOUL.md`
- runtime env/config
- setup-generated tool policies
- wrapper services around Hermes

then it should not be implemented as a deep patch to Hermes core.

## Open Design Work

This pivot still requires concrete design work in these areas:

- exact `Kanidm` integration contract
- exact tool execution metadata and broker model
- exact live-mounted workspace lifecycle and isolation rules
- exact separation between Hermes core, runtime wrapper, and product service
- exact upstream-sync strategy once the product glue becomes part of the Hermes fork
