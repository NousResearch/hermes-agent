# OpenMail Error Reference

## API error format

All errors return a consistent JSON structure:

```json
{
  "error": "error_code",
  "message": "Human-readable description"
}
```

## Error codes

| Status | Code | Description | Resolution |
|--------|------|-------------|------------|
| 400 | `invalid_mailbox_name` | Mailbox name format invalid | Use lowercase letters, numbers, and hyphens only |
| 401 | `unauthorized` | Invalid or missing API key | Check `OPENMAIL_API_KEY` is set and starts with `om_` |
| 404 | `not_found` | Resource does not exist | Verify the inbox ID or thread ID |
| 409 | `address_taken` | Mailbox name already in use | Choose a different `--mailbox-name` |
| 422 | `recipient_suppressed` | Recipient on suppression list | Do not contact them |
| 429 | `rate_limit_exceeded` | Too many requests | Wait for the `Retry-After` header duration |
| 429 | `cold_outreach_limit` | Cold send limit for new inbox | New inboxes have a warm-up period |

## CLI errors

### `missing API key`

No key found in `--api-key`, `OPENMAIL_API_KEY`, or `./.env`. Follow
[setup.md](setup.md).

### `missing inbox id`

No default inbox saved. Run:

```bash
openmail init --mailbox-name "agent" --display-name "Agent"
```

### Unexpected errors — report them

A `500 internal_error` is a problem on OpenMail's side. Report it:

```bash
openmail feedback --type bug --message "<what you did and what you got>" \
  --endpoint "<path>" --error-code "<error code>"
```

Reporting never blocks your work. Do not retry-loop a 500 more than a few
times, and do not report the same problem more than once per session.
