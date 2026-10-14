# Quick Start: Language Implementations

Inline implementations for TypeScript, Python, and C# MCP servers. For complete guides, see `node_mcp_server.md`, `python_mcp_server.md`, and `dotnet_mcp_server.md`.

## TypeScript Implementation

### Installation

```bash
npm install @modelcontextprotocol/sdk zod express
npm install -D @types/express
```

### Basic Server

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { z } from "zod";
import express from "express";

const app = express();
app.use(express.json());

const server = new McpServer({
  name: "my-server",
  version: "1.0.0"
});

// Tool
server.registerTool("add", {
  description: "Add two numbers",
  inputSchema: { a: z.number(), b: z.number() }
}, async ({ a, b }) => ({
  content: [{ type: "text", text: String(a + b) }]
}));

// Resource
server.registerResource(
  "config",
  "config://app",
  { description: "App configuration" },
  async (uri) => ({
    contents: [{ uri: uri.href, text: JSON.stringify({ env: "prod" }) }]
  })
);

// Prompt
server.registerPrompt("review", {
  description: "Code review",
  argsSchema: { code: z.string() }
}, ({ code }) => ({
  messages: [{ role: "user", content: { type: "text", text: `Review:\n${code}` } }]
}));

// Transport
const transport = new StreamableHTTPServerTransport({
  sessionIdGenerator: () => crypto.randomUUID()
});

await server.connect(transport);

app.all("/mcp", async (req, res) => {
  await transport.handleRequest(req, res);
});

app.listen(3000);
```

> **Note:** Top-level `await` requires Node.js with ES modules (`"type": "module"` in package.json or `.mjs` extension).

### Stateless Mode (Serverless)

```typescript
const transport = new StreamableHTTPServerTransport({
  sessionIdGenerator: undefined  // Disables sessions
});
```

## Python Implementation

### Installation

```bash
pip install "mcp[cli]"
```

### Basic Server (FastMCP)

```python
from mcp.server.fastmcp import FastMCP, Context
from typing import List

mcp = FastMCP("my-server")

# Tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

# Async tool with progress
@mcp.tool()
async def process_files(files: List[str], ctx: Context) -> str:
    """Process files with progress."""
    for i, f in enumerate(files):
        await ctx.report_progress(i + 1, len(files))
    return f"Processed {len(files)} files"

# Resource
@mcp.resource("config://app")
def get_config() -> dict:
    """App configuration."""
    return {"env": "prod", "version": "1.0.0"}

# Dynamic resource
@mcp.resource("users://{user_id}/profile")
def get_user(user_id: str) -> dict:
    """Get user profile."""
    return {"id": user_id, "name": f"User {user_id}"}

# Prompt
@mcp.prompt()
def code_review(code: str, language: str = "python") -> str:
    """Code review prompt."""
    return f"Review this {language} code:\n\n```{language}\n{code}\n```"

if __name__ == "__main__":
    # Streamable HTTP
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000, path="/mcp")
    # Or stdio: mcp.run()
```

### FastAPI Integration

```python
from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP

api = FastAPI()
mcp = FastMCP("api-tools")

@mcp.tool()
def query(sql: str) -> dict:
    return {"result": "data"}

api.mount("/mcp", mcp.streamable_http_app())
# Run: uvicorn app:api --port 8000
```

## C# Implementation

### Installation

```bash
# HTTP / remote server
dotnet add package ModelContextProtocol.AspNetCore

# stdio / local server (lighter — no ASP.NET Core dependency)
# dotnet add package ModelContextProtocol
```

### Basic Server

```csharp
using System.ComponentModel;
using Microsoft.Extensions.DependencyInjection;
using ModelContextProtocol.Server;

var builder = WebApplication.CreateBuilder(args);

builder.Services
    .AddMcpServer()
    .WithHttpTransport()
    .WithToolsFromAssembly()
    .WithPromptsFromAssembly()
    .WithResourcesFromAssembly();

var app = builder.Build();
app.MapMcp();  // Maps the Streamable HTTP MCP endpoint at the root path
app.Run();

// Tools
[McpServerToolType]
public static class MyTools
{
    [McpServerTool, Description("Add two numbers")]
    public static int Add(
        [Description("First number")] int a,
        [Description("Second number")] int b) => a + b;

    [McpServerTool, Description("Get weather")]
    public static async Task<string> GetWeather(
        HttpClient http,  // Injected from DI
        [Description("City name")] string city,
        CancellationToken ct)
    {
        var data = await http.GetStringAsync($"https://api.weather.example/{city}", ct);
        return data;
    }
}

// Prompts
[McpServerPromptType]
public static class MyPrompts
{
    [McpServerPrompt, Description("Code review prompt")]
    public static ChatMessage CodeReview(
        [Description("Code to review")] string code) =>
        new(ChatRole.User, $"Review this code:\n\n{code}");
}

// Resources
[McpServerResourceType]
public static class MyResources
{
    [McpServerResource(Name = "config://app"), Description("App config")]
    public static string Config() => """{"env": "production"}""";

    [McpServerResource(UriTemplate = "docs://{topic}")]
    public static string GetDoc([Description("Topic")] string topic) =>
        $"Documentation for {topic}";
}
```

### Stdio Transport (Local)

```csharp
var builder = Host.CreateApplicationBuilder(args);

builder.Logging.AddConsole(o => o.LogToStandardErrorThreshold = LogLevel.Trace);

builder.Services
    .AddMcpServer()
    .WithStdioServerTransport()
    .WithToolsFromAssembly();

await builder.Build().RunAsync();
```

> **Package note:** the stdio host above only needs the **`ModelContextProtocol`** package (plus `Microsoft.Extensions.Hosting`). `ModelContextProtocol.AspNetCore` is only required for the HTTP transport.

### Stateless HTTP (Serverless)

In SDK v2, HTTP servers run **stateless by default** — set `SessionMode` explicitly to pin behavior. Only opt into sessions for unsolicited notifications, resource subscriptions, per-client isolation, or server-to-client requests to pre-MRTR clients:

```csharp
builder.Services
    .AddMcpServer()
    .WithHttpTransport(options =>
        options.SessionMode = HttpServerSessionMode.Stateless) // v2 default
    .WithToolsFromAssembly();
```
