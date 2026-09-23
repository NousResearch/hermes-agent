# Forms API Reference

Base URL: `https://api.tally.so`

## Endpoints

### List Forms
```
GET /forms
```

Returns a paginated array of form objects.

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| page | number | No | Page number (default: 1) |
| limit | number | No | Items per page (default: 50, max: 500) |
| workspaceIds | array | No | Filter by workspace IDs |

**Response (200):**
```json
{
  "items": [
    {
      "id": "form-id",
      "name": "Form Name",
      "workspaceId": "workspace-id",
      "status": "PUBLISHED",
      "numberOfSubmissions": 42,
      "isClosed": false,
      "createdAt": "2024-01-01T00:00:00Z",
      "updatedAt": "2024-01-02T00:00:00Z"
    }
  ],
  "page": 1,
  "limit": 50,
  "total": 100,
  "hasMore": true
}
```

---

### Get Form
```
GET /forms/{formId}
```

Returns a single form by its ID with all its blocks and settings.

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| formId | string | Yes | The ID of the form |

**Response (200):**
```json
{
  "id": "form-id",
  "name": "Form Name",
  "workspaceId": "workspace-id",
  "status": "PUBLISHED",
  "numberOfSubmissions": 42,
  "isClosed": false,
  "settings": {
    "language": "en",
    "isClosed": false,
    "hasProgressBar": false,
    "hasPartialSubmissions": false
  },
  "blocks": [
    {
      "uuid": "block-uuid",
      "type": "FORM_TITLE",
      "groupUuid": "group-uuid",
      "groupType": "FORM_TITLE",
      "payload": {
        "html": "<h1>Form Title</h1>"
      }
    }
  ],
  "createdAt": "2024-01-01T00:00:00Z",
  "updatedAt": "2024-01-02T00:00:00Z"
}
```

---

### Create Form
```
POST /forms
```

Creates a new form, optionally based on a template or within a specific workspace.

**Request Body:**
```json
{
  "workspaceId": "workspace-id",
  "templateId": "template-id",
  "status": "DRAFT",
  "blocks": [
    {
      "uuid": "uuid-v4",
      "type": "FORM_TITLE",
      "groupUuid": "uuid-v4",
      "groupType": "FORM_TITLE",
      "payload": {
        "html": "<h1>Form Title</h1>"
      }
    }
  ],
  "settings": {
    "language": "en",
    "isClosed": false
  }
}
```

**Fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| status | string | Yes | BLANK, DRAFT, or PUBLISHED |
| blocks | array | Yes | Array of block objects |
| workspaceId | string | No | Target workspace ID |
| templateId | string | No | Template to base form on |
| settings | object | No | Form settings object |

**Response (201):**
Returns the created form object.

---

### Update Form
```
PATCH /forms/{formId}
```

Updates a form's settings, blocks, or status.

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| formId | string | Yes | The ID of the form |

**Request Body:**
```json
{
  "name": "New Form Name",
  "status": "PUBLISHED",
  "blocks": [...],
  "settings": {...}
}
```

**Response (200):**
Returns the updated form object.

---

### Delete Form
```
DELETE /forms/{formId}
```

Deletes a form by its ID and moves it to the trash.

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| formId | string | Yes | The ID of the form |

**Response (204):**
No content on success.

---

## Form Status Values

| Status | Description |
|--------|-------------|
| BLANK | Empty form, just created |
| DRAFT | Has content but not published |
| PUBLISHED | Live and accepting responses |
| DELETED | Moved to trash |

## Form Settings Object

| Field | Type | Description |
|-------|------|-------------|
| language | string | Form language (e.g., "en") |
| isClosed | boolean | Whether form is closed |
| closeMessageTitle | string | Message when form is closed |
| closeMessageDescription | string | Description when closed |
| submissionsLimit | integer | Max number of submissions |
| redirectOnCompletion | string | URL to redirect after submit |
| hasProgressBar | boolean | Show progress bar |
| hasPartialSubmissions | boolean | Allow partial submissions |
| saveForLater | boolean | Enable save for later |

## Example: Complete Form Object

```json
{
  "id": "mKz8pQ",
  "name": "Contact Us",
  "workspaceId": "wAb3xY",
  "status": "PUBLISHED",
  "numberOfSubmissions": 156,
  "isClosed": false,
  "settings": {
    "language": "en",
    "isClosed": false,
    "closeMessageTitle": "Form Closed",
    "closeMessageDescription": "Thank you for your interest!",
    "submissionsLimit": null,
    "redirectOnCompletion": null,
    "hasProgressBar": true,
    "hasPartialSubmissions": false,
    "saveForLater": true
  },
  "blocks": [
    {
      "uuid": "550e8400-e29b-41d4-a716-446655440000",
      "type": "FORM_TITLE",
      "groupUuid": "550e8400-e29b-41d4-a716-446655440000",
      "groupType": "FORM_TITLE",
      "payload": {
        "html": "<h1>Contact Us</h1>",
        "logo": null,
        "cover": null
      }
    }
  ],
  "createdAt": "2024-01-01T00:00:00.000Z",
  "updatedAt": "2024-01-15T12:30:00.000Z"
}
```
