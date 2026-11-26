# Anytype MCP Authentication

## Authorization Header Format

The `OPENAPI_MCP_HEADERS` environment variable must be set with the exact format:

```bash
OPENAPI_MCP_HEADERS='{"Authorization":"Bearer <YOUR_TOKEN>","Anytype-Version":"<VERSION>"}'
```

### Required Fields

1. **Authorization**: Bearer token from Anytype app settings
2. **Anytype-Version**: Version number from Anytype app settings

### Where to Find Credentials

1. Open Anytype app
2. Go to Settings → API Keys
3. Copy your token and note the version number

## Testing Authentication

```python
# Set OPENAPI_MCP_HEADERS env var, then call npx @anyproto/anytype-mcp with JSON-RPC
```

## Common Errors

### Unauthorized (401)
- Token expired or incorrect
- Version mismatch
- Token not found in Anytype settings

### Method Not Found
- Wrong method name (should be `API-<method_name>`)
- Server not responding to JSON-RPC

### JSON Parsing Error
- Shell escaping issues with JSON string
- Use single quotes around JSON payload
