# Notion Trigger Reference

## Event Types

- [`page.created`](#pagecreated)
- [`page.content_updated`](#pagecontent_updated)
- [`comment.created`](#commentcreated)

---

## `page.created`

### Parameters

- `parent_id` (string, optional): Database or page ID to scope to pages created inside it. Leave blank to match any parent.

### Sample Payload

```json
{
  "integration_id": "00000000-0000-0000-0000-0000000000c1",
  "data": {
    "parent": {
      "type": "page",
      "id": "00000000-0000-0000-0000-0000000000d2"
    }
  },
  "api_version": "2026-03-11",
  "type": "page.created",
  "workspace_id": "00000000-0000-0000-0000-0000000000c2",
  "subscription_id": "00000000-0000-0000-0000-0000000000c3",
  "accessible_by": [
    {
      "type": "person",
      "id": "00000000-0000-0000-0000-0000000000c4"
    },
    {
      "type": "bot",
      "id": "00000000-0000-0000-0000-0000000000c5"
    }
  ],
  "attempt_number": 1,
  "id": "00000000-0000-0000-0000-0000000000d3",
  "workspace_name": "Acme",
  "entity": {
    "type": "page",
    "id": "00000000-0000-0000-0000-0000000000d4"
  },
  "timestamp": "2026-06-24T21:44:44.441Z",
  "authors": [
    {
      "type": "person",
      "id": "00000000-0000-0000-0000-0000000000c4"
    }
  ]
}
```

---

## `page.content_updated`

### Parameters

- `page_id` (string, optional): Page ID to match only that page for block changes. Leave blank to match all pages.

### Sample Payload

```json
{
  "integration_id": "00000000-0000-0000-0000-0000000000c1",
  "data": {
    "parent": {
      "type": "space",
      "id": "00000000-0000-0000-0000-0000000000c2"
    },
    "updated_blocks": [
      {
        "type": "block",
        "id": "00000000-0000-0000-0000-0000000000e1"
      },
      {
        "type": "block",
        "id": "00000000-0000-0000-0000-0000000000e2"
      },
      {
        "type": "block",
        "id": "00000000-0000-0000-0000-0000000000e3"
      },
      {
        "type": "block",
        "id": "00000000-0000-0000-0000-0000000000e4"
      }
    ]
  },
  "api_version": "2026-03-11",
  "type": "page.content_updated",
  "workspace_id": "00000000-0000-0000-0000-0000000000c2",
  "subscription_id": "00000000-0000-0000-0000-0000000000c3",
  "accessible_by": [
    {
      "type": "person",
      "id": "00000000-0000-0000-0000-0000000000c4"
    },
    {
      "type": "bot",
      "id": "00000000-0000-0000-0000-0000000000c5"
    }
  ],
  "attempt_number": 1,
  "id": "00000000-0000-0000-0000-0000000000d5",
  "workspace_name": "Acme",
  "entity": {
    "type": "page",
    "id": "00000000-0000-0000-0000-0000000000d1"
  },
  "timestamp": "2026-06-24T21:48:13.627Z",
  "authors": [
    {
      "type": "person",
      "id": "00000000-0000-0000-0000-0000000000c4"
    }
  ]
}
```

---

## `comment.created`

### Parameters

- `parent_id` (string, optional): Page ID to receive comment events only from that page. Leave blank to match any page.

### Sample Payload

```json
{
  "integration_id": "00000000-0000-0000-0000-0000000000c1",
  "data": {
    "page_id": "00000000-0000-0000-0000-0000000000d1",
    "parent": {
      "type": "page",
      "id": "00000000-0000-0000-0000-0000000000d1"
    }
  },
  "api_version": "2026-03-11",
  "type": "comment.created",
  "workspace_id": "00000000-0000-0000-0000-0000000000c2",
  "subscription_id": "00000000-0000-0000-0000-0000000000c3",
  "accessible_by": [
    {
      "type": "person",
      "id": "00000000-0000-0000-0000-0000000000c4"
    },
    {
      "type": "bot",
      "id": "00000000-0000-0000-0000-0000000000c5"
    }
  ],
  "attempt_number": 1,
  "id": "00000000-0000-0000-0000-0000000000d6",
  "workspace_name": "Acme",
  "entity": {
    "type": "comment",
    "id": "00000000-0000-0000-0000-0000000000d7"
  },
  "timestamp": "2026-06-20T00:21:37.946Z",
  "authors": [
    {
      "type": "person",
      "id": "00000000-0000-0000-0000-0000000000c4"
    }
  ]
}
```