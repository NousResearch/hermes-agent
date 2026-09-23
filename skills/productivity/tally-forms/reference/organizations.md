# Organizations API Reference

Base URL: `https://api.tally.so`

## Users

### List Organization Users
```
GET /organizations/{organizationId}/users
```

Returns all users in an organization.

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| organizationId | string | Yes | The organization ID |

**Response (200):**
```json
{
  "items": [
    {
      "id": "user-id",
      "email": "user@example.com",
      "name": "Jane Doe",
      "role": "ADMIN",
      "joinedAt": "2024-01-01T00:00:00Z"
    }
  ]
}
```

---

### Remove Organization User
```
DELETE /organizations/{organizationId}/users/{userId}
```

Removes a user from the organization.

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| organizationId | string | Yes | The organization ID |
| userId | string | Yes | The user ID to remove |

**Response (204):**
No content on success.

---

## Invites

### List Organization Invites
```
GET /organizations/{organizationId}/invites
```

Returns all pending invites for an organization.

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| organizationId | string | Yes | The organization ID |

**Response (200):**
```json
{
  "items": [
    {
      "id": "invite-id",
      "email": "invitee@example.com",
      "role": "MEMBER",
      "status": "PENDING",
      "createdAt": "2024-01-10T00:00:00Z",
      "expiresAt": "2024-01-17T00:00:00Z"
    }
  ]
}
```

---

### Create Invite
```
POST /organizations/{organizationId}/invites
```

Sends an invitation to join the organization.

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| organizationId | string | Yes | The organization ID |

**Request Body:**
```json
{
  "email": "newuser@example.com",
  "role": "MEMBER"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| email | string | Yes | Email address to invite |
| role | string | No | Role to assign (ADMIN or MEMBER, default: MEMBER) |

**Response (201):**
Returns the created invite object.

---

### Cancel Invite
```
DELETE /organizations/{organizationId}/invites/{inviteId}
```

Cancels a pending invitation.

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| organizationId | string | Yes | The organization ID |
| inviteId | string | Yes | The invite ID to cancel |

**Response (204):**
No content on success.

---

## Role Values

| Role | Description |
|------|-------------|
| ADMIN | Full access, can manage users and billing |
| MEMBER | Can create and manage forms within assigned workspaces |

## Invite Status Values

| Status | Description |
|--------|-------------|
| PENDING | Invite sent, awaiting acceptance |
| ACCEPTED | User accepted the invite |
| EXPIRED | Invite expired (7 days) |
| CANCELLED | Invite was cancelled |
