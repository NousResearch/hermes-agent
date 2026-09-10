# JMAP (RFC 8620 / RFC 8621) Technical Reference

A quick reference for the core JMAP email protocol methods used by Hermes Agent.

---

## 1. Capabilities
- `urn:ietf:params:jmap:core`: Standard batching, error objects, and discovery.
- `urn:ietf:params:jmap:mail`: Mailbox, Thread, and Email management.
- `urn:ietf:params:jmap:submission`: Outbox and email delivery submission.

---

## 2. Core Method Calls

### `Email/query` (Search & Filter)
```json
[
  "Email/query",
  {
    "accountId": "acc-123",
    "filter": {
      "inMailbox": "mbx-inbox",
      "notKeyword": "$seen",
      "text": "invoice"
    },
    "sort": [{ "property": "receivedAt", "isAscending": false }],
    "limit": 10
  },
  "call-1"
]
```

### `Email/get` (Fetch Content)
```json
[
  "Email/get",
  {
    "accountId": "acc-123",
    "ids": ["M18f3a9b2c"],
    "properties": ["id", "subject", "from", "to", "receivedAt", "bodyValues", "textBody", "htmlBody"],
    "fetchTextBodyValues": true,
    "fetchHTMLBodyValues": true
  },
  "call-2"
]
```

### `Email/set` + `EmailSubmission/set` (Send Message)
JMAP decouples creating an email object from submitting it for SMTP delivery:
1. `Email/set`: Creates the email record in Drafts.
2. `EmailSubmission/set`: Uses result reference `#onSuccessCreateEmail` to deliver the newly created draft.
