# Submissions API Reference

Base URL: `https://api.tally.so`

## Endpoints

### List Submissions
```
GET /forms/{formId}/submissions
```

Returns a paginated list of submissions for a form.

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| formId | string | Yes | The ID of the form |

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| page | number | No | Page number (default: 1) |
| limit | number | No | Items per page (default: 50, max: 500) |
| filter | string | No | Filter expression for field values |
| startDate | string | No | ISO 8601 date — only submissions after this date |
| endDate | string | No | ISO 8601 date — only submissions before this date |
| afterId | string | No | Return submissions after this submission ID (cursor pagination) |

**Response (200):**
```json
{
  "items": [
    {
      "id": "submission-id",
      "formId": "form-id",
      "respondentId": "respondent-id",
      "createdAt": "2024-01-15T10:30:00Z",
      "updatedAt": "2024-01-15T10:30:00Z",
      "isCompleted": true,
      "fields": [
        {
          "key": "question_abc12345",
          "label": "Your Name",
          "type": "INPUT_TEXT",
          "value": "Jane Doe"
        },
        {
          "key": "email_def67890",
          "label": "Email",
          "type": "INPUT_EMAIL",
          "value": "jane@example.com"
        }
      ]
    }
  ],
  "page": 1,
  "limit": 50,
  "total": 156,
  "hasMore": true
}
```

**Filter Examples:**

Filter submissions where a field equals a value:
```
GET /forms/{formId}/submissions?filter=question_abc12345:eq:Jane
```

---

### Get Submission
```
GET /forms/{formId}/submissions/{submissionId}
```

Returns a single submission by ID.

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| formId | string | Yes | The ID of the form |
| submissionId | string | Yes | The ID of the submission |

**Response (200):**
```json
{
  "id": "submission-id",
  "formId": "form-id",
  "respondentId": "respondent-id",
  "createdAt": "2024-01-15T10:30:00Z",
  "updatedAt": "2024-01-15T10:30:00Z",
  "isCompleted": true,
  "fields": [
    {
      "key": "question_abc12345",
      "label": "Your Name",
      "type": "INPUT_TEXT",
      "value": "Jane Doe"
    }
  ]
}
```

---

### Delete Submission
```
DELETE /forms/{formId}/submissions/{submissionId}
```

Permanently deletes a submission.

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| formId | string | Yes | The ID of the form |
| submissionId | string | Yes | The ID of the submission |

**Response (204):**
No content on success.

---

## Submission Fields Object

Each field in the `fields` array contains:

| Field | Type | Description |
|-------|------|-------------|
| key | string | The unique field identifier (matches block name) |
| label | string | Human-readable field label |
| type | string | Block type (INPUT_TEXT, INPUT_EMAIL, MULTIPLE_CHOICE, etc.) |
| value | any | The submitted value — string, number, array, or object |

### Value Types by Field

| Block Type | Value Type | Example |
|------------|-----------|---------|
| INPUT_TEXT / TEXTAREA | string | `"Hello world"` |
| INPUT_EMAIL | string | `"user@example.com"` |
| INPUT_NUMBER | number | `42` |
| INPUT_DATE | string | `"2024-01-15"` |
| INPUT_TIME | string | `"14:30"` |
| INPUT_PHONE_NUMBER | string | `"+1234567890"` |
| INPUT_LINK | string | `"https://example.com"` |
| MULTIPLE_CHOICE | string | `"Option A"` |
| CHECKBOXES | array | `["Option A", "Option B"]` |
| DROPDOWN | string | `"Option A"` |
| MULTI_SELECT | array | `["Tag1", "Tag2"]` |
| RATING | number | `4` |
| LINEAR_SCALE | number | `7` |
| FILE_UPLOAD | array | `[{"name": "file.pdf", "url": "https://..."}]` |
| SIGNATURE | string | Base64 encoded image data |
| MATRIX | object | `{"Row 1": "Column A", "Row 2": "Column B"}` |
| RANKING | array | `["First", "Second", "Third"]` |
| PAYMENT | object | `{"amount": 1000, "currency": "USD", "status": "paid"}` |

## Example: Pagination with Cursor

```python
submissions = []
after_id = None

while True:
    params = {"limit": 100}
    if after_id:
        params["afterId"] = after_id
    
    response = GET /forms/{formId}/submissions?{params}
    
    submissions.extend(response["items"])
    
    if not response["hasMore"]:
        break
    
    after_id = response["items"][-1]["id"]
```

## Example: Date Filtering

```
GET /forms/{formId}/submissions?startDate=2024-01-01T00:00:00Z&endDate=2024-01-31T23:59:59Z
```
