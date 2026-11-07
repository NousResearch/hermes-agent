# Gmail Routing Reference

> **Safety:** All write operations (POST, PUT, PATCH, DELETE) require explicit user confirmation before execution. Verify the target resource and intended effect with the user first. See the main [SKILL.md](../SKILL.md#security--permissions) for full security policy.

**App name:** `google-mail`
**Base URL proxied:** `gmail.googleapis.com`

## API Path Pattern

```
/google-mail/gmail/v1/users/me/{endpoint}
```

## Common Endpoints

### List Messages
```bash
GET /google-mail/gmail/v1/users/me/messages?maxResults=10
```

Example:

```bash
maton google-mail message list -L 10
```

With query filter:
```bash
GET /google-mail/gmail/v1/users/me/messages?q=is:unread&maxResults=10
```

Example:

```bash
maton google-mail message list --query 'is:unread' -L 10
```

### Get Message
```bash
GET /google-mail/gmail/v1/users/me/messages/{messageId}
```

Example:

```bash
maton google-mail message view {messageId} --headers
```

With metadata only:
```bash
GET /google-mail/gmail/v1/users/me/messages/{messageId}?format=metadata&metadataHeaders=From&metadataHeaders=Subject&metadataHeaders=Date
```

Example:

```bash
maton google-mail message view {messageId} --fetch-format metadata --metadata-header From,Subject,Date
```

### Send Message
```bash
POST /google-mail/gmail/v1/users/me/messages/send
Content-Type: application/json

{
  "raw": "BASE64_ENCODED_EMAIL"
}
```

Example:

```bash
maton google-mail message send --to alice@example.com --subject 'Hello' --body 'Hi there!'
```

### Reply to Message

Example:

```bash
maton google-mail message reply {messageId} --body 'Thanks!'
```

### Forward Message

Example:

```bash
maton google-mail message forward {messageId} --to dave@example.com --body 'FYI'
```

### List Labels
```bash
GET /google-mail/gmail/v1/users/me/labels
```

Example:

```bash
maton google-mail label list
```

### List Threads
```bash
GET /google-mail/gmail/v1/users/me/threads?maxResults=10
```

Example:

```bash
maton google-mail thread list -L 10
```

### Get Thread
```bash
GET /google-mail/gmail/v1/users/me/threads/{threadId}
```

Example:

```bash
maton google-mail thread view {threadId}
```

### Modify Message Labels
```bash
POST /google-mail/gmail/v1/users/me/messages/{messageId}/modify
Content-Type: application/json

{
  "addLabelIds": ["STARRED"],
  "removeLabelIds": ["UNREAD"]
}
```

Example:

```bash
maton google-mail message modify {messageId} --add-label STARRED --remove-label UNREAD
```

### Trash Message
```bash
POST /google-mail/gmail/v1/users/me/messages/{messageId}/trash
```

Example:

```bash
maton google-mail message trash {messageId}
```

### Create Draft
```bash
POST /google-mail/gmail/v1/users/me/drafts
Content-Type: application/json

{
  "message": {
    "raw": "BASE64URL_ENCODED_EMAIL"
  }
}
```

Example:

```bash
maton google-mail draft create --to alice@example.com --subject 'Hello' --body 'Draft content here'
```

### Update Draft
```bash
PUT /google-mail/gmail/v1/users/me/drafts/{draftId}
Content-Type: application/json

{
  "message": {
    "raw": "BASE64URL_ENCODED_EMAIL"
  }
}
```

### Send Draft
```bash
POST /google-mail/gmail/v1/users/me/drafts/send
Content-Type: application/json

{
  "id": "{draftId}"
}
```

Example:

```bash
maton google-mail draft send {draftId}
```

### Get Profile
```bash
GET /google-mail/gmail/v1/users/me/profile
```

## Query Operators

Use in the `q` parameter:
- `is:unread` - Unread messages
- `is:starred` - Starred messages
- `from:email@example.com` - From specific sender
- `to:email@example.com` - To specific recipient
- `subject:keyword` - Subject contains keyword
- `after:2024/01/01` - After date
- `before:2024/12/31` - Before date
- `has:attachment` - Has attachments
- `newer_than:7d` - Within last 7 days
- `older_than:1y` - Older than 1 year
- `label:LABEL_NAME` - Has specific label

## Pagination

Gmail uses `pageToken`-based pagination. The CLI handles this automatically with `--paginate`:

```bash
maton google-mail message list --query 'newer_than:7d' --paginate
```

For raw HTTP requests, pass the `nextPageToken` from the previous response as the `pageToken` query parameter.

## Notes

- Authentication is automatic - the router injects the OAuth token
- Use `me` as userId for the authenticated user
- Message body is base64url encoded in the `raw` field (RFC 2822 format)
- Common labels: `INBOX`, `SENT`, `DRAFT`, `STARRED`, `UNREAD`, `TRASH`, `SPAM`, `IMPORTANT`
- Rate limit: ~10 requests/sec per account
- Use `format=metadata` with `metadataHeaders` to fetch only headers and avoid downloading full message bodies

## Resources

- [API Overview](https://developers.google.com/gmail/api/reference/rest)
- [List Messages](https://developers.google.com/gmail/api/reference/rest/v1/users.messages/list)
- [Get Message](https://developers.google.com/gmail/api/reference/rest/v1/users.messages/get)
- [Send Message](https://developers.google.com/gmail/api/reference/rest/v1/users.messages/send)
- [Modify Message Labels](https://developers.google.com/gmail/api/reference/rest/v1/users.messages/modify)
- [Trash Message](https://developers.google.com/gmail/api/reference/rest/v1/users.messages/trash)
- [List Threads](https://developers.google.com/gmail/api/reference/rest/v1/users.threads/list)
- [Get Thread](https://developers.google.com/gmail/api/reference/rest/v1/users.threads/get)
- [List Labels](https://developers.google.com/gmail/api/reference/rest/v1/users.labels/list)
- [Create Draft](https://developers.google.com/gmail/api/reference/rest/v1/users.drafts/create)
- [Update Draft](https://developers.google.com/gmail/api/reference/rest/v1/users.drafts/update)
- [Send Draft](https://developers.google.com/gmail/api/reference/rest/v1/users.drafts/send)
- [Get Profile](https://developers.google.com/gmail/api/reference/rest/v1/users/getProfile)
- [Search Operators](https://support.google.com/mail/answer/7190)
- [Maton CLI Manual](https://cli.maton.ai/manual)