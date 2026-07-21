# Workspaces API Reference

Base URL: `https://api.tally.so`

## Endpoints

### List Workspaces
```
GET /workspaces
```

Returns all workspaces accessible to the authenticated user.

**Response (200):**
```json
{
  "items": [
    {
      "id": "workspace-id",
      "name": "My Workspace",
      "organizationId": "org-id",
      "createdAt": "2024-01-01T00:00:00Z",
      "updatedAt": "2024-01-02T00:00:00Z"
    }
  ]
}
```

---

### Get Workspace
```
GET /workspaces/{workspaceId}
```

Returns a single workspace by ID.

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| workspaceId | string | Yes | The ID of the workspace |

**Response (200):**
```json
{
  "id": "workspace-id",
  "name": "My Workspace",
  "organizationId": "org-id",
  "createdAt": "2024-01-01T00:00:00Z",
  "updatedAt": "2024-01-02T00:00:00Z"
}
```

---

### Create Workspace
```
POST /workspaces
```

Creates a new workspace.

**Request Body:**
```json
{
  "name": "New Workspace"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| name | string | Yes | Workspace name |

**Response (201):**
Returns the created workspace object.

---

### Update Workspace
```
PATCH /workspaces/{workspaceId}
```

Updates a workspace's name.

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| workspaceId | string | Yes | The ID of the workspace |

**Request Body:**
```json
{
  "name": "Renamed Workspace"
}
```

**Response (200):**
Returns the updated workspace object.

---

### Delete Workspace
```
DELETE /workspaces/{workspaceId}
```

Deletes a workspace. All forms within the workspace will be moved to the default workspace.

**Path Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| workspaceId | string | Yes | The ID of the workspace |

**Response (204):**
No content on success.

---

## Workspace Object

| Field | Type | Description |
|-------|------|-------------|
| id | string | Unique workspace identifier |
| name | string | Workspace name |
| organizationId | string | Parent organization ID |
| createdAt | string | ISO 8601 creation timestamp |
| updatedAt | string | ISO 8601 last update timestamp |
