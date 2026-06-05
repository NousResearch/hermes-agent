# Proton Email Gateway

Hermes supports Proton Mail as a gateway-native email provider. Use this when
Proton cannot be exposed through ordinary IMAP/SMTP.

## Configuration

```bash
EMAIL_PROVIDER=proton
EMAIL_ADDRESS=agent@thomas.md
EMAIL_ALLOWED_USERS=thomas@lfglabs.dev,thomas.marchand@lfglabs.dev
PROTON_CLIENT_FACTORY=your_module:create_client
EMAIL_PROTON_SEEN_PATH=/var/lib/hermes-assistant/state/proton-email-seen.json
```

The Proton factory must return an object with:

```python
class Client:
    def event_polling(self): ...
    def get_message(self, email_id: str): ...
```

Outbound email replies require one of:

```python
send_reply(message_id=..., thread_id=..., to=..., subject=..., body=...)
reply_message(message_id=..., thread_id=..., to=..., subject=..., body=...)
send_email(to=..., subject=..., body=...)
send_message(to=..., subject=..., body=...)
```

If none of those methods exists, Hermes can ingest Proton messages but email
send attempts fail explicitly instead of pretending to send.

Outbound Proton sends are policy-gated. `thomas@lfglabs.dev` is allowed by
default; add more owners with `EMAIL_ALLOWED_RECIPIENTS`, `EMAIL_ALLOWED_USERS`,
or `EMAIL_ALLOWED_DOMAINS`. Unknown recipients are rejected unless
`EMAIL_ALLOW_UNKNOWN_RECIPIENTS=true` is set deliberately. Attachments passed
through `MEDIA:/path` are sent only from approved roots and are blocked for
secret-like filenames, unsafe extensions/MIME types, oversized totals, or paths
outside the allowed roots.

## Behavior

- Trusted senders listed in `EMAIL_ALLOWED_USERS` create real Hermes
  `MessageEvent` inputs.
- Proton summaries with empty `Sender: {}` are hydrated from the full message
  before authorization.
- Non-trusted senders do not trigger autonomous agent execution.
- Optional passive Telegram notifications can be configured with
  `EMAIL_PASSIVE_TELEGRAM_BOT_TOKEN` and `EMAIL_PASSIVE_TELEGRAM_CHAT_ID`.
- Message ids are persisted in `EMAIL_PROTON_SEEN_PATH` to prevent duplicate
  delivery across restarts.
- Transient Proton polling failures reconnect with exponential backoff.

## Operations

Run the gateway-native integration under the normal Hermes service:

```bash
systemctl restart hermes-assistant.service
systemctl status hermes-assistant.service
journalctl -u hermes-assistant.service
```

When `EMAIL_PROVIDER=proton` is enabled in the gateway, disable any legacy
`hermes-proton-daemon.service` sidecar so exactly one process owns Proton
polling.

Never start Proton polling with shell backgrounding such as `&`, `nohup`, or
`disown`.
