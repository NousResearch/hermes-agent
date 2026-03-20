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

The product-owned setup flow should be exposed separately from generic upstream Hermes setup.

The intended command shape is:

- `hermes product setup` for product-layer installation and configuration
- `hermes setup` remains generic Hermes setup and should stay upstream-compatible

After installation, a guided setup CLI should configure:

- visible product name and branding shown in the web app
- agent personality seed content, derived from configurable markdown content
- enabled user-facing tools
- execution placement for tools that are allowed to run either inside or outside the runtime sandbox
- initial global model or API route
- optional Tailscale exposure
- local identity-provider integration
- the first admin account
- the global Hermes runtime toolsets that should be enabled for all product chat sessions

After setup, the product should start:

- a lightweight authenticated local web app
- optional tailnet exposure when enabled
- isolated per-user Hermes runtimes

The first authenticated web app should stay intentionally minimal.

The intended primary surfaces are:

- for signed-out users:
  - one lightweight landing/auth card
- for signed-in normal users:
  - chat
  - later the shared or user workspace directory
- for signed-in admins:
  - the same user experience
  - plus a compact user-management section for creating users, issuing signup links, and deactivating users

The old development-style multi-panel control plane should not be the model for this forked product UI.

The product should reuse the existing MYNAH visual language where it helps:

- the styling system
- the dark-mode light-mode sun and moon toggle
- the card-based single-page layout

But it should not carry over cluttered control-plane sections that are not part of the user-facing product.

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

For the product auth flow:

- `network.public_host` must still be treated as the user-facing host used in generated URLs
- the default local value should be `localhost`
- setup must ask for the user-facing hostname separately from the bind address

The web app and runtime launcher should derive their behavior from this product config rather than becoming independent configuration authorities.

The first implementation should keep Hermes' existing `config.yaml` for generic Hermes behavior and use `product.yaml` only for setup-owned product deployment behavior.

If the product setup flow reuses generic Hermes setup helpers for model/provider or tool selection, that reuse should happen from the product-owned command path and the resulting product runtime settings should still be synchronized back into `product.yaml`.

The product should not maintain a second MYNAH-specific tier vocabulary for runtime tools. Product setup should persist Hermes toolset names directly, with an initial safe default of:

- `memory`
- `session_search`

## Authentication Direction

The preferred identity system for this pivot is `Pocket ID`.

Reasons:

- fully local onboarding is required
- SMTP-backed signup and recovery should not be required
- the product needs local-first user creation, activation, and recovery flows
- the supplier must be able to bootstrap the first admin during setup
- later admins must be able to create additional users from the web UI
- passkey-first authentication is acceptable and now preferred if it materially simplifies local product auth

The intended identity model is:

- `Pocket ID` provides identity, login, passkey enrollment, signup-token/login-code onboarding, and OIDC
- the product remains an OIDC client and still owns its own application session and authorization model
- bundled `Pocket ID` should run as a local Docker-managed service in the first implementation
- the supplier bootstraps the first product admin during setup
- later users are created by an admin in the web UI
- user onboarding and recovery should use native `Pocket ID` login-code/signup-token flows rather than a product-owned password reset flow
- the first product admin should be able to enroll directly during setup without requiring an SMTP service or a separate identity admin account

The bundled `Pocket ID` service files should be generated under:

- `HERMES_HOME/product/services/pocket-id`

The first implementation should generate:

- a Docker Compose file for the bundled `Pocket ID` service
- a local `.env` file for the bundled service with `APP_URL`, `ENCRYPTION_KEY`, and `STATIC_API_KEY`
- persistent storage bindings under `HERMES_HOME/product/services/pocket-id/data`
- a local OIDC client secret reference stored outside `product.yaml`

The first implementation should also treat these startup details as part of the contract:

- service startup should use a reproducible Docker path and should recreate changed container definitions cleanly
- bundled auth bootstrap should use the supported `Pocket ID` admin/API surface rather than shelling into containers for identity mutations
- setup should bootstrap the product's OIDC client through the `STATIC_API_KEY` admin surface
- setup should surface the native `Pocket ID` `/setup` flow for first-admin enrollment instead of inventing a product-owned password bootstrap
- localhost development should be a first-class supported path, not a degraded special case
- the product's own login flow should consume provider metadata through a provider-neutral OIDC helper layer with PKCE rather than Pocket-ID-specific redirect logic in the app
- the first product app auth surface should provide:
  - login start
  - callback
  - authenticated session inspection
  - logout

The intended user-lifecycle flow is:

1. supplier setup bootstraps the bundled `Pocket ID` service and product OIDC client
2. supplier setup surfaces the native `Pocket ID` setup URL for first-admin enrollment
3. the first product admin signs into the authenticated local web app through `Pocket ID`
4. admin creates additional users in the web UI
5. the product service can issue native `Pocket ID` signup tokens for self-service onboarding
6. each user completes passkey enrollment through `Pocket ID`
7. the product continues to treat `Pocket ID` as the login authority and OIDC provider

The first browser admin surface should implement:

- listing visible product users from `Pocket ID`
- creating regular non-admin users
- issuing one-time signup links through native `Pocket ID` signup tokens
- deactivating existing users rather than hard-deleting them

The first browser admin surface should not implement:

- browser-side promotion to product admin
- destructive identity deletion as the default user-removal path
- browser-side editing of broader product setup or provider configuration

Pocket ID currently requires an email address for admin-created users.

To preserve the intended product UX where email is optional for passkey-first local onboarding:

- the browser admin form may leave email blank
- the product backend should synthesize a deterministic `.invalid` placeholder email for provider compatibility
- the product UI should treat those placeholder emails as effectively unset rather than showing them as meaningful user contact data

The first setup flow should start the stack automatically after config generation and first-admin bootstrap.

The setup CLI should allow an auth-mode choice:

- `passkey`
- `password`

The default should be:

- `passkey`

Hybrid auth mode should not be part of the first design.

Password-first auth is no longer the preferred default. If a password mode ever returns, it should be justified as a separate product policy, not kept as silent compatibility baggage.

## Runtime and Tool Model

Hermes should remain the runtime core.

MYNAH-specific behavior should stay concentrated near the runtime edge rather than spreading through Hermes core.

Runtime identity should continue to come from `SOUL.md`.

The product service should configure and launch isolated per-user Hermes runtimes.

In the first implementation:

- the product app should read runtime defaults from `HERMES_HOME/product.yaml`
- runtime launch settings should be derived on the host from `product.yaml` and injected into each runtime as env rather than requiring the runtime to read the host product config directly
- explicit runtime env vars should still override product-config defaults when set
- per-user runtimes should remain separate isolated containers managed by the product app
- each runtime should get its own `HERMES_HOME` under the product storage root
- each runtime should keep its own Hermes session DB and file-backed memory state
- runtime tools should come directly from Hermes toolset names selected in product setup rather than from a separate MYNAH tier or profile layer
- browser chat APIs should proxy to runtime-local HTTP endpoints rather than calling `AIAgent` in-process
- runtime control endpoints may be published only to host loopback for product-app proxying and must never be exposed on the LAN directly
- if a configured model route points at host loopback such as `127.0.0.1` or `localhost`, the runtime launcher must rewrite that URL to a container-reachable host alias before injecting it into the runtime env
- the default local host alias for containerized runtimes is `host.docker.internal`, and Docker launch should add an explicit `host-gateway` mapping for that alias

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

The browser UI should also preserve the live-chat feel of the MYNAH prototype where it adds real value:

- reasoning should stream live
- answer text should stream live
- streamed reasoning should visually fade or soften compared with the final answer

This behavior is part of the intended product UX and should be preserved when the authenticated chat surface is rebuilt in the fork.

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

The expected first-version per-user layout is:

- `HERMES_HOME/product/users/<user_id>/workspace`
- `HERMES_HOME/product/users/<user_id>/runtime/hermes`

The runtime `SOUL.md` should be rendered into:

- `HERMES_HOME/product/users/<user_id>/runtime/hermes/SOUL.md`

The content for that file should come from setup-owned product config:

- `product.agent.soul_template_path`

If no custom template path is configured, the product should render a bundled default identity template.

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

- exact tool execution metadata and broker model
- exact live-mounted workspace lifecycle and isolation rules
- exact separation between Hermes core, runtime wrapper, and product service
- exact upstream-sync strategy once the product glue becomes part of the Hermes fork
