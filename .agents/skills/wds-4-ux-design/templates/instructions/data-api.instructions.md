# Data & API Requirements

**Include when:** Page requires data from APIs or external sources

---

## Data Sources

| Data Element | Source | Type | Required | Notes |
|--------------|--------|------|----------|-------|
| `{data-field}` | {API / static / localStorage} | {string / number / array} | {yes/no} | {notes} |

---

## API Endpoints

### {Endpoint Name}

| Property | Value |
|----------|-------|
| Method | {GET / POST / PUT / DELETE} |
| Path | `/api/{path}` |
| Purpose | {What this endpoint does} |
| Auth | {Required / Optional / None} |

**Request:**
```json
{
  "field": "value"
}
```

**Response (Success):**
```json
{
  "data": {}
}
```

**Response (Error):**
```json
{
  "error": "message",
  "code": "ERR_XXX"
}
```

**Error Codes:**
| Code | Meaning | User Message |
|------|---------|--------------|
| `{code}` | {technical meaning} | {user-friendly message} |

---

## Loading States

| State | Duration | UI |
|-------|----------|-----|
| Initial load | {expected ms} | {skeleton / spinner / etc.} |
| Refresh | {expected ms} | {indicator type} |
| Background | {expected ms} | {silent / toast} |

---

## Caching Strategy

| Data | Cache Duration | Invalidation |
|------|----------------|--------------|
| {data type} | {duration} | {when to refresh} |
