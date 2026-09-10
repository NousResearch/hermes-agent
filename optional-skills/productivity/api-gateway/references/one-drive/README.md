# OneDrive Routing Reference

> **Safety:** All write operations (POST, PUT, PATCH, DELETE) require explicit user confirmation before execution. Verify the target resource and intended effect with the user first. See the main [SKILL.md](../SKILL.md#security--permissions) for full security policy.

**App name:** `one-drive`
**Base URL proxied:** `graph.microsoft.com`

## API Path Pattern

```
/one-drive/v1.0/me/drive/{resource}
```

## Common Endpoints

### Get User's Drive
```bash
GET /one-drive/v1.0/me/drive
```

Example:

```bash
maton one-drive whoami
```

### List Drives
```bash
GET /one-drive/v1.0/me/drives
```

Example:

```bash
maton one-drive drive list
```

### Get Drive Root
```bash
GET /one-drive/v1.0/me/drive/root
```

Example:

```bash
maton one-drive item view root
```

### List Root Children
```bash
GET /one-drive/v1.0/me/drive/root/children
```

Example:

```bash
maton one-drive item list
```

### Get Item by ID
```bash
GET /one-drive/v1.0/me/drive/items/{item-id}
```

Example:

```bash
maton one-drive item view {item-id}
```

### Get Item by Path
```bash
GET /one-drive/v1.0/me/drive/root:/Documents/file.txt
```

Example:

```bash
maton one-drive item view-by-path Documents/file.txt
```

### List Folder Children by Path
```bash
GET /one-drive/v1.0/me/drive/root:/Documents:/children
```

Example:

```bash
maton one-drive item list Documents
```

### Create Folder
```bash
POST /one-drive/v1.0/me/drive/root/children
Content-Type: application/json

{
  "name": "New Folder",
  "folder": {}
}
```

Example:

```bash
maton one-drive item create-folder 'New Folder'
```

### Upload File (Simple - up to 4MB)
```bash
PUT /one-drive/v1.0/me/drive/root:/filename.txt:/content
Content-Type: text/plain

{file content}
```

Example:

```bash
maton one-drive item upload ./filename.txt --path filename.txt
```

Files larger than 4 MiB automatically use a resumable upload session.

### Delete Item
```bash
DELETE /one-drive/v1.0/me/drive/items/{item-id}
```

Example:

```bash
maton one-drive item delete {item-id}
```

### Create Sharing Link
```bash
POST /one-drive/v1.0/me/drive/items/{item-id}/createLink
Content-Type: application/json

{
  "type": "view",
  "scope": "anonymous"
}
```

Example:

```bash
maton one-drive item share {item-id} --type view --scope anonymous
```

### Search Files
```bash
GET /one-drive/v1.0/me/drive/root/search(q='query')
```

Example:

```bash
maton one-drive drive search 'query'
```

### Special Folders
```bash
GET /one-drive/v1.0/me/drive/special/documents
GET /one-drive/v1.0/me/drive/special/photos
```

Example:

```bash
maton one-drive item view --special documents
```

### Recent Files
```bash
GET /one-drive/v1.0/me/drive/recent
```

Example:

```bash
maton one-drive drive recent
```

### Shared With Me
```bash
GET /one-drive/v1.0/me/drive/sharedWithMe
```

Example:

```bash
maton one-drive drive shared
```

## Pagination

OneDrive uses cursor-based pagination. The CLI handles this automatically with `--paginate`:

```bash
maton one-drive item list --paginate
```

For raw HTTP requests, follow the `@odata.nextLink` URL returned in the response.

## Notes

- Authentication is automatic - the router injects the OAuth token
- Uses Microsoft Graph API (`graph.microsoft.com`)
- Use colon (`:`) syntax for path-based addressing
- Files less than or equal to 4MB upload via a single PUT; larger files automatically use a resumable upload session
- Download URLs in `@microsoft.graph.downloadUrl` are pre-authenticated and temporary
- Supports OData query parameters: `$select`, `$expand`, `$filter`, `$orderby`, `$top`
- Conflict behavior options: `fail`, `replace`, `rename`
- On personal OneDrive accounts, only the user's own drive ID (returned by `whoami`) is directly addressable. The additional `b!...`-prefixed IDs that appear in `drive list` return HTTP 400 from Microsoft Graph when fetched this way. Use `me/drive` instead.

## Resources

- [OneDrive Developer Documentation](https://learn.microsoft.com/en-us/onedrive/developer/)
- [Microsoft Graph API Reference](https://learn.microsoft.com/en-us/graph/api/overview)
- [DriveItem Resource](https://learn.microsoft.com/en-us/graph/api/resources/driveitem)
- [Maton CLI Manual](https://cli.maton.ai/manual)