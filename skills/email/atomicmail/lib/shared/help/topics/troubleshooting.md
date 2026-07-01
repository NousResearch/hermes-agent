# Troubleshooting

- Missing API key: run `register` or provide valid credentials.
- Wrong `inMailbox`: use `$INBOX_MAILBOX_ID` (mailbox id), not inbox email.
- Blob/upload shape errors: `data` must be an array of datasource objects.
- Missing variables (`$TO`, etc.): provide them via `vars` / `--vars`.
