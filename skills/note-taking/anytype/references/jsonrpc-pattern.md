# Anytype MCP JSON-RPC Pattern

## Working Pattern

The `@anyproto/anytype-mcp` executable does NOT support a `call` subcommand. Instead, use direct JSON-RPC injection via standard input.

### Basic Structure

```python
# Set OPENAPI_MCP_HEADERS env var, then call npx @anyproto/anytype-mcp with JSON-RPC
```

## Method Names

| Method | Description |
|--------|-------------|
| `API-create-object` | Create new objects (notes, tasks, pages, etc.) |
| `API-update-object` | Modify existing objects |
| `API-delete-object` | Delete objects |
| `API-search-space` | Search within a space |
| `API-search-global` | Search across all spaces |

## Parameter Examples

### Create Object
```python
"arguments": {
    "space_id": "<YOUR_SPACE_ID>",
    "name": "My Note",
    "type_key": "note",
    "body": "Content here"
}
```

### Update Object
```python
"arguments": {
    "object_id": "<YOUR_OBJECT_ID>",
    "name": "Updated Title",
    "body": "Updated content"
}
```

### Delete Object
```python
"arguments": {
    "object_id": "<YOUR_OBJECT_ID>"
}
```

### Search Space
```python
"arguments": {
    "space_id": "<YOUR_SPACE_ID>",
    "query": "search term"
}
```

## Error Handling

```python
if result.returncode != 0:
    print(f"Error: {result.stderr}")
    return None

try:
    response = json.loads(result.stdout)
    if "error" in response:
        print(f"API Error: {response['error']}")
        return None
    return response.get("result")
except json.JSONDecodeError:
    print(f"Failed to parse response: {result.stdout}")
    return None
```

## Tips

1. **Always set `env`**: The environment variable must be passed to the subprocess
2. **Use `npx -y`**: Automatically installs if not present
3. **Add newline**: The payload must end with `\n` for proper JSON-RPC framing
4. **Check return code**: Non-zero return code indicates an error
5. **Parse stdout**: The server response is in stdout, errors in stderr
