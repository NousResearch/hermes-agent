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
- `Kanidm` is the preferred auth direction for the product layer.
- The setup CLI should be the source of truth for product-wide settings.
- The browser admin UI should remain narrow and should not become a general product-config console in the first version.

## Development environment

- The local product app may run on Windows for development, but intended production behavior must still be designed against Linux.
- Local development assumptions must be written down when they matter to implementation or verification.
- Linux and WSL validation remain the reference environment for runtime isolation and deployment behavior.
- If a local workflow depends on hostname-specific auth or container-network assumptions, document the reliable localhost path and all prerequisites clearly.
- SSH validation on the separate Linux laptop is part of the normal closed-loop workflow when auth, Docker stack generation, runtime isolation, or filesystem-mount behavior changes.
- Temporary SSH validation workspaces, helper scripts, containers, and logs must be cleaned up after testing.

## Kanidm-specific rules

- `network.public_host` must be a hostname or domain, not a raw IP address.
- Default local `public_host` is `localhost`.
- Treat `bind_host` and `public_host` as separate concerns in setup and implementation.
- For the bundled `Kanidm` stack, generate certificates before starting the compose service.
- The generated `Kanidm` service should run `kanidmd` directly, not through a shell entrypoint assumption.
- When bind-mounting the service data directory, run the container with the host uid/gid when available so secure local permissions still work on Linux.

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
