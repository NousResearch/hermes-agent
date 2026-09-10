# Twitter Bookmark Import to Anytype - Session Notes (May 2026)

## Context
User exported 500 Twitter bookmarks via `xurl bookmarks --export` and wanted to import them into Anytype collections.

## What Worked
- Created 421/500 bookmarks using individual subprocess calls
- Each bookmark created with proper frontmatter, markdown content, and tags
- Used `API-create-object` with `type_key: "bookmark"`

## What Failed
### 1. Persistent MCP Process Approach
- **Problem:** Process crashes after 5-10 requests with "Connection closed" errors
- **Attempted:** Keep one MCP server process open and send multiple requests
- **Result:** Unreliable for bulk operations

### 2. Adding Bookmarks to Collections
- **Problem:** `API-update-object` with `links` parameter returns "failed to retrieve object"
- **Attempted:**
  ```python
  payload = {
      "name": "API-update-object",
      "arguments": {
          "space_id": "<SPACE_ID>",
          "object_id": "<COLLECTION_ID>",
          "links": [{"objectId": "<BOOKMARK_ID>", "relationKey": "..." }]
      }
  }
  ```
- **Result:** Anytype MCP API doesn't support this operation

## Bulk Operation Pattern That Worked
```python
# Set OPENAPI_MCP_HEADERS env var, then call npx @anyproto/anytype-mcp with JSON-RPC
```

## Key Learnings
1. **Always use individual subprocess calls** for bulk operations - persistent processes crash
2. **Anytype MCP API has limitations** - some operations (like adding to collections) aren't supported
3. **Rate limiting is real** - 421/500 succeeded before hitting limits
4. **Manual collection management required** - use Anytype desktop app to organize bookmarks

## Recommendations for Future Bulk Imports
1. Process in smaller batches (50-100 items)
2. Use individual subprocess calls, not persistent connections
3. Plan for manual organization in collections afterward
4. Consider using Anytype's built-in import features if available
