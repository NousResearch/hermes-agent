# Users API Reference

Base URL: `https://api.tally.so`

## Endpoints

### Get Current User
```
GET /users/me
```

Returns the authenticated user's profile information.

**Response (200):**
```json
{
  "id": "user-id",
  "email": "user@example.com",
  "name": "Jane Doe",
  "organizationId": "org-id",
  "createdAt": "2024-01-01T00:00:00Z",
  "updatedAt": "2024-01-02T00:00:00Z"
}
```

---

## User Object

| Field | Type | Description |
|-------|------|-------------|
| id | string | Unique user identifier |
| email | string | User's email address |
| name | string | User's display name |
| organizationId | string | User's organization ID |
| createdAt | string | ISO 8601 creation timestamp |
| updatedAt | string | ISO 8601 last update timestamp |

## Usage

This endpoint is useful for:
- Verifying API key validity
- Getting the user's organization ID (needed for organization endpoints)
- Displaying the authenticated user's info
