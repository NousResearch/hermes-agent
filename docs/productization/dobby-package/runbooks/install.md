# Install Runbook

Reader: operator performing a fresh install. Next action: create a staging
runtime, configure only V1 surfaces, and stop before live use until verification
passes.

## Inputs

- Release artifact or repository checkout for the Dobby/Hermes package.
- Customer-owned Discord app and bot token.
- BYO model endpoint URL and API key.
- Fresh staging `HERMES_HOME`.
- Optional webhook HMAC secret.

## Guardrails

- Do not copy `~/.hermes`, logs, sessions, or personal memory.
- Do not paste real secrets into docs, tickets, demos, or screenshots.
- Do not run commands against a live remote host during install verification.
- Do not enable non-Discord messaging surfaces in the default package profile.

## Steps

1. Create a fresh staging home.

   ```bash
   export HERMES_HOME="$HOME/.hermes-dobby-staging"
   mkdir -p "$HERMES_HOME"
   chmod 700 "$HERMES_HOME"
   ```

2. Install the package following the signed release instructions.

3. Copy the package env example into the staging home.

   ```bash
   cp <PACKAGE_ENV_EXAMPLE> "$HERMES_HOME/.env"
   chmod 600 "$HERMES_HOME/.env"
   ```

4. Replace angle-bracket placeholders on the staging host only.

5. Configure Discord using `guides/discord-setup.md`.

6. Configure `SOUL.md` and memory consent using `guides/memory-soul.md`.

7. Run package preflight. Stop if any check fails.

8. Start the gateway with Discord and signed webhooks only.

9. Run `runbooks/verify.md`.

10. Run `demo/demo-script.md` with synthetic data.

## Acceptance Criteria

- `HERMES_HOME` is fresh and package-owned.
- Env examples contain placeholders only; real secrets exist only on the host.
- Discord app is customer-owned and allowlisted.
- Model endpoint is configured and low-risk for staging.
- Preflight and verification pass.
- Rollback steps are known before live promotion.
