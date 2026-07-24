# JMAP cheatsheet

## Capabilities (`using`)

- `urn:ietf:params:jmap:core`
- `urn:ietf:params:jmap:mail`
- `urn:ietf:params:jmap:submission`
- `urn:ietf:params:jmap:blob`

## Placeholders

- `$ACCOUNT_ID`, `$INBOX`, `$INBOX_MAILBOX_ID`, `$UPLOAD_URL`, `$DOWNLOAD_URL`
- Any other `$NAME` must come from `vars` / `--vars`.

## Notes

- Bare methodCalls arrays default to core+mail only.
- For submission/blob methods use a full envelope with `using`.
- `inMailbox` expects mailbox id (`$INBOX_MAILBOX_ID`), not email address.
