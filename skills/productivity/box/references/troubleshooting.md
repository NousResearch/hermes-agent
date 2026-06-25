# Troubleshooting

Capture before changing approach: acting auth context, exact command/endpoint, object type + ID, status code, error body.

## Debugging checklist

1. `box users:get me --json --fields id,name,login` — who is the actor?
2. Can the actor see the folder? (collaboration / workspace model)
3. Correct object ID (file vs folder)?
4. Required app scopes enabled and app re-authorized after scope changes?
5. For CLI: only one `box` process at a time?

Run `box <command> --help` to confirm subcommand exists in installed CLI version.

## 401 / 403

- Invalid or expired credentials
- App not authorized for enterprise (`unauthorized_client`)
- Missing scope for the operation
- Acting user lacks permission on the object

Fix: re-check CCG env, app authorization, scopes; confirm service account is collaborated into target folder.

## 404

- Wrong ID
- Object exists but **not visible to current actor** (most common with service accounts)
- Shared link points to different object than expected

Fix: verify actor with `users/me`; add collaboration or use `--ccg-user` for user-owned content.

## 409

- Duplicate name on create/upload
- Collaboration already exists
- Metadata template conflict

Fix: list parent folder for existing item; reuse ID or rename.

## 429

- Rate limit — read `Retry-After`, wait, retry
- Parallel CLI invocations — run serially
- Bulk batches too fast — add 200–500ms pause between ops

See `references/bulk-operations.md`.

## CLI auth problems

- CCG environment not set current: `box configure:environments:set-current hermes`
- Wrong actor: missing `--as-user` or wrong `--ccg-user` environment
- `-t` token override conflicts with configured environment
- Secrets wrong in config JSON — match `~/.hermes/.env` values

Safe check: `box users:get me --json`. Avoid `box configure:environments:get --current` in routine checks.

## Search / AI empty results

- Wrong actor (service account cannot see user content)
- Over-broad query — add type or ancestor filters
- Box AI not enabled for account or file type unsupported — try PDF/DOCX sample

## Network blocked in sandbox

If `box users:get me` works but API calls fail with DNS/connection errors, the runtime may block outbound network. Test the same command outside the sandbox.

## Webhook verification failures

- Wrong signing secret
- Body mutated before verification
- Missing timestamp/replay checks

## Docs

- Permissions and scopes: https://developer.box.com/guides/api-calls/permissions-and-errors/scopes
- 404 causes: https://support.box.com/hc/en-us/articles/360043693734
