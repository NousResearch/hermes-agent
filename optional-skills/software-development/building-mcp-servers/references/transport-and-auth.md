# Transport, Authentication, and Migration

## Streamable HTTP Transport

**Single endpoint** replaces dual SSE endpoints. Supports stateful sessions or stateless (serverless) mode.

### Protocol Flow

```text
Client                              Server
  |------ POST /mcp ---------------->|  (JSON-RPC messages)
  |<----- JSON or SSE response ------|
  |------ GET /mcp ----------------->|  (Optional: server-initiated)
  |<----- SSE stream ----------------|
```

### Required Headers

```http
POST /mcp HTTP/1.1
Content-Type: application/json
Accept: application/json, text/event-stream
Mcp-Session-Id: <session-id>  # After initialization
```

## Authentication (OAuth 2.1)

MCP servers are **OAuth Resource Servers** (not Authorization Servers). Key requirements:

- **PKCE mandatory** (S256 method)
- **Resource Indicators** (RFC 8707) required
- **Bearer tokens** in Authorization header
- **Audience validation** on every request

### Discovery Endpoints

```http
GET /.well-known/oauth-protected-resource  # Server metadata
GET /.well-known/oauth-authorization-server  # Auth server metadata
```

### Server Response on 401

```http
HTTP/1.1 401 Unauthorized
WWW-Authenticate: Bearer realm="mcp",
  resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource"
```

### Token Validation (Python)

```python
import jwt
from jwt import PyJWKClient

class TokenValidator:
    def __init__(self, jwks_uri: str, audience: str):
        self.jwks = PyJWKClient(jwks_uri)
        self.audience = audience

    def validate(self, token: str) -> dict:
        key = self.jwks.get_signing_key_from_jwt(token)
        return jwt.decode(
            token, key.key,
            algorithms=["RS256"],
            audience=self.audience,
            options={"require": ["exp", "aud", "iss"]}
        )
```

### Protected Resource Metadata Response

```json
{
  "resource": "https://mcp.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["read", "write", "admin"],
  "bearer_methods_supported": ["header"]
}
```

### OAuth Middleware Integration (Python/Starlette)

```python
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from mcp.server.fastmcp import FastMCP

class BearerAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, validator: TokenValidator):
        super().__init__(app)
        self.validator = validator

    async def dispatch(self, request, call_next):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return JSONResponse(
                {"error": "unauthorized"},
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="mcp"'}
            )
        try:
            token = auth.replace("Bearer ", "")
            request.state.auth = self.validator.validate(token)
        except Exception:
            return JSONResponse({"error": "invalid_token"}, status_code=401)
        return await call_next(request)

# Setup
mcp = FastMCP("secure-server")
validator = TokenValidator(
    jwks_uri="https://auth.example.com/.well-known/jwks.json",
    audience="https://mcp.example.com"
)

app = Starlette(
    routes=[Mount("/mcp", app=mcp.streamable_http_app())],
    middleware=[Middleware(BearerAuthMiddleware, validator=validator)]
)
```

### OAuth Middleware Integration (TypeScript/Express)

```typescript
import jwt from "jsonwebtoken";
import jwksClient from "jwks-rsa";

const client = jwksClient({ jwksUri: "https://auth.example.com/.well-known/jwks.json" });

const authMiddleware = async (req, res, next) => {
  const auth = req.headers.authorization;
  if (!auth?.startsWith("Bearer ")) {
    return res.status(401).json({ error: "unauthorized" });
  }
  try {
    const token = auth.slice(7);
    const decoded = jwt.decode(token, { complete: true });
    const key = await client.getSigningKey(decoded.header.kid);
    req.auth = jwt.verify(token, key.getPublicKey(), {
      audience: "https://mcp.example.com",
      algorithms: ["RS256"]
    });
    next();
  } catch (err) {
    res.status(401).json({ error: "invalid_token" });
  }
};

app.use("/mcp", authMiddleware);
```

### Authenticated Client Request

```typescript
const transport = new StreamableHTTPClientTransport(
  new URL("https://mcp.example.com/mcp"),
  {
    requestInit: {
      headers: { Authorization: `Bearer ${accessToken}` }
    }
  }
);
```

### Security Anti-Patterns

| Anti-Pattern | Fix |
| -------------- | ----- |
| Token passthrough to upstream APIs | Use separate tokens for upstream calls |
| Missing audience validation | Always validate `aud` claim |
| Tokens in URLs | Use Authorization header only |

### Scope Enforcement in Tools

Check scopes before executing sensitive operations:

**TypeScript:**

```typescript
const authMiddleware = async (req, res, next) => {
  // ... token validation ...
  req.auth = { sub: decoded.sub, scopes: decoded.scope?.split(" ") || [] };
  next();
};

server.registerTool("delete_user", { /* ... */ }, async ({ userId }, { meta }) => {
  const scopes = meta?.auth?.scopes || [];
  if (!scopes.includes("admin:write")) {
    return { isError: true, content: [{ type: "text", text: "Insufficient scope" }] };
  }
  // ... perform deletion ...
});
```

**Python:** Use HTTP middleware to validate, then check in tools:

```python
# With Starlette middleware (see OAuth Middleware Integration above)
# Store validated claims in request.state, then check in tool:

@mcp.tool()
def delete_user(user_id: str) -> str:
    """Delete user - requires admin:write scope."""
    # Scope enforcement happens at HTTP layer via middleware
    # Tool assumes request already passed auth checks
    return f"Deleted user {user_id}"
```

> **Note:** For advanced middleware with per-tool auth context, use `fastmcp` package (`pip install fastmcp`) which provides `Middleware` class and `Context.get_state()`. The official `mcp` package provides FastMCP but with simpler middleware options.

### Passing Auth Context to Tool Handlers

**TypeScript:** Store auth on transport or use a request-scoped context:

```typescript
// In middleware: attach to request
req.auth = { sub: decoded.sub, email: decoded.email };

// Pass to tool via server context or closure
const sessions = new Map();
app.all("/mcp", async (req, res) => {
  const transport = new StreamableHTTPServerTransport({ /* ... */ });
  sessions.set(transport.sessionId, { auth: req.auth });
  // Tools access via sessions.get(sessionId)
});
```

**Python (with `fastmcp` package):** Use middleware and context state:

```python
# pip install fastmcp
from fastmcp import FastMCP, Context
from fastmcp.server.middleware import Middleware, MiddlewareContext

mcp = FastMCP("my-server")

class AuthMiddleware(Middleware):
    async def on_call_tool(self, context: MiddlewareContext, call_next):
        # Extract and validate token, then store in context
        context.fastmcp_context.set_state("user_id", "user_123")
        context.fastmcp_context.set_state("scopes", ["read", "write"])
        return await call_next()

mcp.add_middleware(AuthMiddleware())

@mcp.tool
async def get_my_profile(ctx: Context) -> dict:
    user_id = ctx.get_state("user_id")  # Set by middleware
    return {"user_id": user_id, "profile": "..."}
```

### Token Refresh Handling

Access tokens are short-lived (typically 1 hour). Strategies:

1. **Client-side refresh:** Clients refresh tokens before expiration and reconnect
2. **Proactive server refresh:** Background task refreshes tokens expiring soon
3. **On-demand refresh:** Return 401, client refreshes and retries

**Recommended pattern:** Use short-lived access tokens (5-60 min) with refresh tokens. On 401:

```typescript
// Client retry logic
async function callWithRefresh(tool: string, args: object) {
  try {
    return await client.callTool(tool, args);
  } catch (err) {
    if (err.status === 401) {
      accessToken = await refreshAccessToken(refreshToken);
      transport.updateHeaders({ Authorization: `Bearer ${accessToken}` });
      return await client.callTool(tool, args);
    }
    throw err;
  }
}
```

## Migration: SSE to Streamable HTTP

### Server Changes

```typescript
// OLD (SSE) - Two endpoints
app.get("/sse", async (req, res) => {
  const transport = new SSEServerTransport("/sse/messages", res);
  await server.connect(transport);
});
app.post("/sse/messages", async (req, res) => { /* ... */ });

// NEW (Streamable HTTP) - Single endpoint
app.all("/mcp", async (req, res) => {
  const transport = new StreamableHTTPServerTransport();
  await server.connect(transport);
  await transport.handleRequest(req, res);
});
```

### Client Changes

```typescript
// OLD
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";
const transport = new SSEClientTransport(new URL("http://localhost:3000/sse"));

// NEW
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
const transport = new StreamableHTTPClientTransport(new URL("http://localhost:3000/mcp"));
```

### Backward-Compatible Server

```typescript
// Support both transports during migration
app.all("/mcp", /* new Streamable HTTP handler */);
app.get("/sse", /* legacy SSE handler */);
app.post("/sse/messages", /* legacy message handler */);
```

## Client Configuration

### Claude Desktop / Claude Code (stdio)

```json
{
  "mcpServers": {
    "my-server": {
      "command": "node",
      "args": ["/path/to/server.js"],
      "env": { "API_KEY": "secret" }
    }
  }
}
```

### Remote Server (Streamable HTTP)

```json
{
  "mcpServers": {
    "remote": {
      "type": "streamable-http",
      "url": "https://mcp.example.com/mcp"
    }
  }
}
```

## Error Handling

Return tool errors (not protocol errors) for model self-correction:

**TypeScript:**

```typescript
server.registerTool("query", { /* ... */ }, async ({ sql }) => {
  if (sql.includes("DROP")) {
    return {
      isError: true,
      content: [{ type: "text", text: "Destructive queries not allowed" }]
    };
  }
  // ...
});
```

**Python:** Raise exceptions in tools - they're caught and returned as errors:

```python
@mcp.tool()
def query(sql: str) -> dict:
    if "DROP" in sql.upper():
        raise ValueError("Destructive queries not allowed")
    return {"result": "data"}
```

## Common Mistakes

| Mistake | Fix |
| --------- | ----- |
| Using SSE for new servers | Use Streamable HTTP |
| Writing to stdout in stdio servers | Use stderr for logs |
| Missing session ID after init | Always include `Mcp-Session-Id` header |
| Not validating OAuth audience | Validate token `aud` matches your server |
| Forwarding client tokens upstream | Use separate credentials for upstream APIs |
