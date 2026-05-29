# Rollback Runbook

Reader: operator responding to a failed install, unsafe behavior, or bad live
promotion. Next action: stop Dobby, preserve redacted evidence, restore the
previous package state, and verify quiet behavior.

## Rollback Principles

- Stop new Discord and webhook intake first.
- Preserve data by default.
- Do not delete memory, sessions, or logs unless privacy cleanup is explicitly
  requested.
- Rotate exposed secrets instead of trying to recover them.
- Keep rollback local to the package-owned `HERMES_HOME`.

## Immediate Rollback

1. Announce maintenance in the operator channel.
2. Stop the gateway using the package service command.
3. Disable inbound webhook routes or remove the public route from the sender.
4. Confirm the bot is no longer responding in Discord.
5. Save redacted diagnostics under the package incident folder.

## Restore Previous Config

1. Copy the last known-good package config into place.
2. Keep the current data directory intact.
3. Re-run preflight.
4. Start staging first.
5. Run the status, allowlist, webhook rejection, and attachment metadata checks.
6. Promote back to live only after staging passes.

## Secret Exposure

If a Discord token, model key, or webhook secret may have leaked:

1. Stop the gateway.
2. Revoke or rotate the affected credential at its owner.
3. Replace the value in the host secret store or env file.
4. Clear terminal scrollback and logs only according to company policy.
5. Run verification before restart.

Do not commit rotated values or paste them into the incident report.

## Memory Or Privacy Rollback

If memory captured data without consent:

1. Stop intake.
2. Export redacted evidence for the incident owner.
3. Use targeted forget for specific entries when possible.
4. Use delete-all only after explicit confirmation.
5. Verify `session_search` no longer returns deleted package-owned content.

## Rollback Complete

Rollback is complete when:

- The bot is quiet outside allowed channels.
- Signed webhooks reject invalid payloads.
- Status output is redacted.
- Previous config is restored or a new safe config is verified.
- Remaining data cleanup is documented as done or intentionally deferred.
