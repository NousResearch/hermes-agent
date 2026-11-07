# Gmail Trigger Reference

## Event Types

- [`email.received`](#emailreceived)

---

## `email.received`

### Parameters

- `labels` (string[], optional, default `["INBOX"]`): Label IDs that must all be present for the trigger to fire.

System label IDs:

| Label ID | Meaning |
|----------|---------|
| `INBOX` | In the inbox |
| `DRAFT` | Unsent draft |
| `UNREAD` | Not yet read |
| `STARRED` | Starred |
| `IMPORTANT` | Marked important |
| `SPAM` | In spam |

Other system labels may exist; list the account's labels to see all label IDs (system and custom).

### Sample Payload

```json
{
  "threadId": "19ee22f009a785f5",
  "snippet": "Delivered in as little as 25 minutes.",
  "labelIds": [
    "CATEGORY_PROMOTIONS",
    "UNREAD",
    "INBOX"
  ],
  "payload": {
    "headers": [
      {
        "name": "Delivered-To",
        "value": "recipient@example.com"
      }
    ],
    "mimeType": "text/html"
  },
  "historyId": "14215811",
  "id": "19ee22f009a785f5",
  "sizeEstimate": 131819,
  "internalDate": "1781911191000"
}
```