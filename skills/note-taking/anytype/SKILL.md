---
name: anytype
description: "Expert skill for interacting with Anytype via the official MCP server. Uses the user's verified configuration."
version: 2.0.0
author: Hermes Agent Community
license: MIT
metadata:
  hermes:
    tags: [anytype, mcp, knowledge-management, production]
    related_skills: [native-mcp]
---

# Anytype - Master Skill

**The definitive way to interact with Anytype via the MCP server.**

## When to Use

Use this skill for any task involving Anytype — the local-first, end-to-end encrypted knowledge management app. Triggers on requests like "create a note in Anytype", "search my Anytype space", "add a task to Anytype", or any mention of Anytype objects, spaces, or collections.

## Configuration (Source of Truth)

Use this exact configuration for all calls. Do NOT attempt to use other headers or versions.

**MCP Server Command:**
```
npx -y @anyproto/anytype-mcp
```

**Required Environment Variable:**
```
OPENAPI_MCP_HEADERS='{"Authorization":"Bearer <YOUR_TOKEN>","Anytype-Version":"<VERSION>"}'
```

> **IMPORTANT:** Replace `<YOUR_TOKEN>` and `<VERSION>` with your actual credentials. The token is stored in your Anytype app settings → API Keys.

## How to Execute Commands

To call any Anytype tool, use a shell command that injects the headers.

**Pattern (via Terminal/Execute Code):**
```bash
OPENAPI_MCP_HEADERS='{"Authorization":"Bearer <YOUR_TOKEN>","Anytype-Version":"<VERSION>"}' \
npx -y @anyproto/anytype-mcp
```

> **NOTE:** The MCP server runs on stdio and expects JSON-RPC payloads. Use Python subprocess for reliable execution.

## Formatting Standards

All generated pages should follow a consistent layout:

1. **Frontmatter**: YAML block with `title`, `type`, `tags`, `created`, `updated`, `status`
2. **Header**: H1 — do NOT put emoji in the title text; use the dedicated `icon` field instead
3. **Structure**: Use ╔═══ boxes for important info/summaries

### Icons (MANDATORY)

**Every created object MUST have an icon.** This is a common user expectation and should always be included.

- Set the icon via `API-update-object` AFTER creation (create doesn't support it inline reliably)
- Pass `icon` as a **JSON object/dict**, NOT a string:
  ```python
  "icon": {"format": "emoji", "emoji": "🧪"}  # correct — dict
  "icon": '{"format":"emoji","emoji":"🧪"}'   # wrong — string → 400 bad_request
  ```
- Pick an emoji that matches the object's purpose (task → ✅, page → 📄, meeting → 🤝, etc.)
- Never skip this step - objects without icons look unfinished and inconsistent.

Example post-create icon update:
```python
payload = json.dumps({
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "API-update-object",
        "arguments": {
            "space_id": "<SPACE_ID>",
            "object_id": "<OBJECT_ID>",
            "icon": {"format": "emoji", "emoji": "📄"}
        }
    }
}) + "\n"
```

## Core Methods

| Method | What | Notes |
|--------|------|-------|
| `API-search-space` | Find objects by query | Use `query` parameter |
| `API-create-object` | Create notes, tasks, bookmarks, etc. | `type_key` can be `note`, `task`, `bookmark`, `collection`, `set`. **Critical**: Must include `space_id` in arguments for ALL operations (including create) |
| `API-update-object` | Modify existing content or metadata | Requires `object_id` and `space_id` in arguments |
| `API-delete-object` | Delete objects | Requires `object_id` and `space_id` in arguments |
| `API-get-object` | Retrieve object details | Requires `object_id` and `space_id` in arguments |
| `API-search-global` | Search across all spaces | Returns nested structure - parse `content[0].text` |

**Important:** Ensure `space_id` is correct for your workspace.

## Setup Instructions

### Step 1: Get Your API Token

1. Open Anytype app
2. Go to Settings → API Keys
3. Create a new token or copy existing one
4. Note the version number from the same settings page

### Step 2: Configure Environment Variable

Set OPENAPI_MCP_HEADERS with your token and version before running any Anytype MCP commands.

### Step 3: Verify MCP Server

Test the connection by listing available tools:
```bash
OPENAPI_MCP_HEADERS='{"Authorization":"Bearer YOUR_TOKEN_HERE","Anytype-Version":"YOUR_VERSION"}' \
npx -y @anyproto/anytype-mcp
```

## Pitfalls and Troubleshooting

### Common Issues

1. **Method Not Found**
   - Ensure you are calling `API-<method_name>`
   - Check the MCP server is running: `npx -y @anyproto/anytype-mcp`

2. **JSON Parsing Error**
   - Always ensure your shell arguments are properly escaped
   - Especially for the JSON headers
   - Use single quotes around the JSON string

3. **Space ID Issues (CRITICAL!)**
   - **space_id is REQUIRED in arguments for ALL operations**, including `API-get-object`, `API-update-object`, and `API-delete-object`
   - Without it: you'll get 500 Internal Server Error with "failed to retrieve object" OR the operation silently succeeds but doesn't actually execute (e.g. delete returns exit 0 but object stays unarchived)
   - Find your space_id in Anytype Settings → Space → API Keys

4. **MCP Response Parsing**
   - Responses are double-encoded JSON: outer result → content[0].text → inner JSON string
   - Pattern: `json.loads(out)["result"]["content"][0]["text"]` then parse that string again
   - The actual object data lives inside the nested structure

5. **MCP Wrapper Failure**
   - The `@anyproto/anytype-mcp` executable does NOT support a `call` subcommand
   - Use direct JSON-RPC injection via standard input to the MCP server process
   - **Fallback Pattern (via Python/Terminal):**
     ```python
     # Use subprocess with env var for each call
     # See references/jsonrpc-pattern.md for detailed examples
     ```

6. **Search Result Parsing**
   - `API-search-global` via `tools/call` returns a nested structure
   - The actual data is inside `content[0].text` as a JSON string
   - You must parse this string to access the `data` array containing object information

7. **Persistent MCP Process Instability (CRITICAL FOR BULK OPERATIONS)**
   - **Problem:** Persistent MCP server processes crash after 5-10 requests with "Connection closed" errors
   - **Solution:** Always use individual subprocess calls for bulk operations (one npx call per request)
   - **Impact:** Slower due to npx startup time, but reliable
   - **Example:** For large batch operations, use separate subprocess calls, NOT one persistent connection

8. **API-update-object with `links` Parameter (KNOWN LIMITATION)**
   - **Problem:** Attempting to add objects to collections via `API-update-object` with `links` parameter returns "failed to retrieve object" errors
   - **Root Cause:** Anytype MCP API doesn't support adding objects to collections via this method
   - **Workaround:** Use Anytype desktop app to manually add objects to collections, or wait for API support

9. **Token Expiration**
   - Anytype tokens may expire
   - Regenerate token in Settings → API Keys if you get authentication errors

10. **Icon Format — JSON Object, NOT String**
    - **Problem:** Passing `icon` as a string returns `400 bad_request: json: cannot unmarshal string into Go struct field UpdateObjectRequest.icon`
    - **Solution:** Pass `icon` as a native Python dict / JSON object
    - `json.dumps()` will serialize it correctly — just don't pre-stringify the icon value

### Verification Steps

After any operation, verify:

1. **Create**: Check the object exists in Anytype app
2. **Update**: Verify changes are reflected
3. **Delete**: Confirm object is removed
4. **Search**: Verify search returns expected results

## Quick Reference Examples

### Create a Task
```python
payload = json.dumps({
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "API-create-object",
        "arguments": {
            "space_id": "<SPACE_ID>",
            "type_key": "task",
            "name": "My new task",
            "body": [{"type": "paragraph", "content": [{"text": {"text": "Task description here"}}]}]
        }
    }
}) + "\n"
# Then update with icon: API-update-object with icon={"format": "emoji", "emoji": "✅"}
```

### Search for Notes
```python
payload = json.dumps({
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "API-search-space",
        "arguments": {
            "space_id": "<SPACE_ID>",
            "query": "search term"
        }
    }
}) + "\n"
# Parse response: json.loads(out)["result"]["content"][0]["text"] → then parse again for data array
```

### Update an Object (with Markdown Content)
```python
payload = json.dumps({
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "API-update-object",
        "arguments": {
            "space_id": "<SPACE_ID>",
            "object_id": "<OBJECT_ID>",
            "body": [{"type": "heading", "content": [{"text": {"text": "Updated Title"}}]}]
        }
    }
}) + "\n"
```

### Get an Object (Correct Parameters)
```python
payload = json.dumps({
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "API-get-object",
        "arguments": {
            "space_id": "<SPACE_ID>",
            "object_id": "<OBJECT_ID>"
        }
    }
}) + "\n"
# IMPORTANT: space_id is required even for get operations!
```

### Delete an Object
```python
payload = json.dumps({
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "API-delete-object",
        "arguments": {
            "space_id": "<SPACE_ID>",
            "object_id": "<OBJECT_ID>"
        }
    }
}) + "\n"
```

## Debugging Tips

1. **Check MCP Server Logs**
   - Look for errors in terminal output
   - Verify the server is responding to JSON-RPC requests

2. **Test Connection**
   - Use simple commands first (search-space)
   - Gradually increase complexity

3. **Validate Parameters**
   - Ensure all required parameters are provided
   - Check parameter types match expectations

4. **Use Python for Complex Operations**
   - For multi-step workflows, use Python with subprocess
   - Easier to handle complex JSON and error handling

## Additional Resources

- Anytype Official Docs: https://anytype.io/docs
- MCP Protocol: https://modelcontextprotocol.io
- Anytype Community: https://community.anytype.io

---

> **Remember:** Always keep your API token secure. Never commit it to version control.
