# Webhooks API Reference

Base URL: `https://api.tally.so`

## Endpoints

### List Webhooks
```
GET /webhooks
```

Returns all webhooks for the authenticated user.

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| formId | string | No | Filter webhooks by form ID |

**Response (200):**
```json
{
  "items": [
    {
      "id": "webhook-id",
      "formId": "form-id",
      "url": "https://example.com/webhook",
      "isActive": true,
      "eventTypes": ["FORM_RESPONSE"],
      "createdAt": "2024-01-01T00:00:00Z",
      "updatedAt": "2024-01-02T00:00:00Z"
    }
  ]
}
```

---

### Create Webhook
```
POST /webhooks
```

Creates a new webhook subscription.

**Request Body:**
```json
{
  "formId": "form-id",
  "url": "https://example.com/webhook",
  "eventTypes": ["FORM_RESPONSE"]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| formId | string | Yes | Form ID to subscribe to |
| url | string | Yes | HTTPS URL to receive webhook events |
| eventTypes | array | Yes | Event types to subscribe to |

**Response (201):**
Returns the created webhook object.

---

### Update Webhook
```
PATCH /webhooks/{webhookId}
```

Updates a webhook's URL, status, or event types.

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| webhookId | string | Yes | The webhook ID |

**Request Body:**
```json
{
  "url": "https://example.com/new-webhook",
  "isActive": true,
  "eventTypes": ["FORM_RESPONSE"]
}
```

**Response (200):**
Returns the updated webhook object.

---

### Delete Webhook
```
DELETE /webhooks/{webhookId}
```

Deletes a webhook subscription.

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| webhookId | string | Yes | The webhook ID |

**Response (204):**
No content on success.

---

### List Webhook Events
```
GET /webhooks/{webhookId}/events
```

Returns delivery history for a webhook.

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| webhookId | string | Yes | The webhook ID |

**Response (200):**
```json
{
  "items": [
    {
      "id": "event-id",
      "webhookId": "webhook-id",
      "eventType": "FORM_RESPONSE",
      "status": "DELIVERED",
      "statusCode": 200,
      "payload": { ... },
      "createdAt": "2024-01-15T10:30:00Z",
      "deliveredAt": "2024-01-15T10:30:01Z"
    }
  ]
}
```

---

### Retry Webhook Event
```
POST /webhooks/{webhookId}/events/{eventId}/retry
```

Retries delivery of a failed webhook event.

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| webhookId | string | Yes | The webhook ID |
| eventId | string | Yes | The event ID to retry |

**Response (200):**
Returns the retried event object.

---

## Event Types

| Event Type | Description |
|------------|-------------|
| FORM_RESPONSE | Triggered when a form receives a new submission |

## Webhook Payload (FORM_RESPONSE)

When a submission is received, the webhook sends a POST request with:

```json
{
  "eventId": "event-id",
  "eventType": "FORM_RESPONSE",
  "createdAt": "2024-01-15T10:30:00Z",
  "data": {
    "formId": "form-id",
    "formName": "Contact Us",
    "submissionId": "submission-id",
    "respondentId": "respondent-id",
    "createdAt": "2024-01-15T10:30:00Z",
    "fields": [
      {
        "key": "question_abc12345",
        "label": "Your Name",
        "type": "INPUT_TEXT",
        "value": "Jane Doe"
      }
    ]
  }
}
```

## Event Delivery Status

| Status | Description |
|--------|-------------|
| DELIVERED | Successfully delivered (2xx response) |
| FAILED | Delivery failed (non-2xx or timeout) |
| PENDING | Queued for delivery |

## Best Practices

- Use HTTPS URLs only for webhook endpoints
- Respond with a 2xx status code within 30 seconds
- Implement idempotency — events may be delivered more than once
- Use webhook events instead of polling submissions for real-time updates
