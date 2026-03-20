# Hermes-Core Product Development

This is a maintainer working note for the product direction of this fork.

## Development stance

- No backwards compatibility for provisional product-layer experiments.
- Prefer the cleanest correct implementation over preserving provisional code.
- Keep Hermes changes as small as possible, but do not protect a bad product design just to avoid edits.
- Remove old experiments and test artifacts once they have served their purpose.

## Required workflow

For every prompt and implementation task, create a TODO.

A TODO is not complete until all of the following are true:

- the change is implemented
- the change is tested
- old code and test artifacts are cleaned up
- docs are updated
- the security impact is checked
- the changes are committed

Do not mark work done early.

## Execution order

1. Understand the relevant local code and docs first.
2. Define the smallest correct change that fits the current architecture.
3. Implement it.
4. Test it.
5. Remove obsolete code, experiments, and temporary artifacts.
6. Update docs.
7. Check security implications.
8. Commit the change.

## Technical rules

- Production target is Linux.
- Windows development is fine, but runtime validation must follow Linux semantics.
- Ubuntu WSL with its own Linux Docker daemon is the preferred local validation path on Windows.
- Default to `gVisor` for per-user runtime isolation.
- Treat plain containers as fallback, not the primary design.
- Preserve Hermes memory semantics where practical, but always isolate them per user namespace.
- Keep product-specific changes near the runtime edge and setup/config layer whenever possible.
- Preserve upstream compatibility by preferring config, wrappers, and narrow extension points over deep Hermes-core patches.

## Current product-direction rules

- The fork is treated as a supplier-curated `hermes-core` distribution.
- End-user branding should come from setup-owned product config, not hardcoded MYNAH branding.
- `Pocket ID` is the preferred auth direction for the product layer.
- The preferred auth experience is passkey-first local onboarding without SMTP.
- The setup CLI should be the source of truth for product-wide settings.
- Product setup should live in a separate product-owned command path, not inside generic `hermes setup`.
- Generic `hermes setup` should remain upstream-compatible and should not bootstrap Pocket ID or other product-only services.
- The browser admin UI should remain narrow and should not become a general product-config console in the first version.

## Development environment

- The local product app may run on Windows for development, but intended production behavior must still be designed against Linux.
- Local development assumptions must be written down when they matter to implementation or verification.
- Linux and WSL validation remain the reference environment for runtime isolation and deployment behavior.
- If a local workflow depends on hostname-specific auth or container-network assumptions, document the reliable localhost path and all prerequisites clearly.
- SSH validation on the separate Linux laptop is part of the normal closed-loop workflow when auth, Docker stack generation, runtime isolation, or filesystem-mount behavior changes.
- Temporary SSH validation workspaces, helper scripts, containers, and logs must be cleaned up after testing.

## Pocket ID-specific rules

- `network.public_host` must be a hostname or domain, not a raw IP address.
- Default local `public_host` is `localhost`.
- Treat `bind_host` and `public_host` as separate concerns in setup and implementation.
- Bundle `Pocket ID` as a product-managed local auth service rather than requiring external SMTP-backed identity infrastructure.
- Prefer native `Pocket ID` onboarding primitives such as signup tokens or login codes over custom product-owned invite/reset flows.
- Keep the product auth flow passkey-first by default. Password-first remains a supported setup choice only if the underlying `Pocket ID` flow supports it cleanly without product-side auth hacks.
- Keep the product app as a standard OIDC client. Avoid provider-specific product logic unless the provider workflow genuinely requires it.
- Start or restart the bundled auth stack with `docker compose up -d --wait --force-recreate` so changed container definitions are actually applied.
- The bundled Pocket ID service contract currently relies on:
  - `APP_URL`
  - `ENCRYPTION_KEY`
  - `STATIC_API_KEY`
- The setup flow should use `STATIC_API_KEY` for setup-time admin API work such as OIDC client bootstrap. Do not shell into containers for provider mutations.
- The first-admin setup contract is the native Pocket ID `/setup` flow. Do not recreate the old temporary-password bootstrap pattern in product code.
- If the first-admin setup state is persisted locally, it should only contain non-secret enrollment metadata such as username, display name, email, setup URL, and client id.
- Keep product login logic provider-neutral. Discovery, PKCE, authorize URL generation, and code exchange should live in a reusable OIDC helper layer rather than being hardcoded inside future app routes.
- The first product auth app surface should stay minimal: login, callback, session, and logout only. Do not grow browser-side auth features beyond that until the real product app exists.
- If product setup reuses generic Hermes setup helpers for model or tool selection, that reuse should happen from the product-owned command path. Do not modify generic `hermes setup` semantics to fit the product.

## Local process hygiene

- Before testing or handing off local behavior, verify that there are no stale old processes still serving outdated code.
- If a task involves restarting a local service, stop previous matching processes first and confirm the new process is the one answering requests.
- Do not treat a browser result as valid verification if an older local process may still be running.
- Cleaning up stale local server processes is part of test cleanup, not optional follow-up work.

## Product safety rules

- No shell access for end-user agents unless it is an explicitly designed product capability.
- No arbitrary code execution for end-user agents unless it is intentionally included in the allowed runtime tool profile.
- No broad filesystem access outside the user-scoped workspace.
- No network tools unless explicitly designed, justified, and policy-controlled.
- No feature should be exposed only because upstream Hermes supports it.

## Cleanliness rules

- Do not accumulate dead architecture notes in side folders once they have been consolidated.
- Keep the repo root authoritative for active product guidance.
- Prefer deleting superseded documents over letting multiple conflicting specs coexist.

## Done criteria

Work is only done when implementation, testing, cleanup, documentation, security review, and commit are all finished.
