# MCP C# SDK 2.0 — What's New (spec revision 2026-07-28)

Companion to `dotnet_mcp_server.md`. Covers everything introduced by the **2.0** release of the
official MCP C# SDK (July 2026), which implements the `2026-07-28` revision of the MCP
specification — the largest protocol revision since launch.

Sources:
- Announcement: <https://devblogs.microsoft.com/dotnet/announcing-v20-of-the-official-mcp-csharp-sdk/>
- Conceptual docs: <https://csharp.sdk.modelcontextprotocol.io/v2/concepts/index.html>

---

## Big picture

The 2026-07-28 revision rethinks MCP over HTTP:

1. **Stateless by default** — the `initialize`/`initialized` handshake and the `Mcp-Session-Id`
   header are gone from the wire format (SEP-2575, SEP-2567). Every request is self-contained.
2. **Plain HTTP surface** — standardized `Mcp-Method`, `Mcp-Name`, and `Mcp-Param-*` headers let
   load balancers, proxies, gateways, and WAFs route MCP traffic without parsing the JSON-RPC body
   (SEP-2243).
3. **Multi Round-Trip Requests (MRTR)** — interactive tools (elicitation, sampling, roots) work
   without a long-lived session; continuity travels in the request payload (SEP-2322).
4. **Extensions as first-class packages** — Tasks and MCP Apps ship as separate opt-in NuGet
   packages instead of being baked into the base SDK.

**Backward compatible by design.** Stable, non-deprecated 1.x APIs keep compiling and running in
2.0. A v2 server accepts the legacy `initialize` handshake from down-level clients, and a v2 client
falls back to `initialize` when the server doesn't speak `2026-07-28`. The only wire-incompatible
feature is the experimental **Tasks** extension (redesigned per SEP-2663).

---

## Packages

| Package | Purpose |
| --- | --- |
| `ModelContextProtocol.Core` | Client and low-level server APIs; minimal dependencies. Bundles Roslyn analyzers. |
| `ModelContextProtocol` | stdio server, hosting/DI, attribute-based discovery. **Start here.** |
| `ModelContextProtocol.AspNetCore` | Streamable HTTP server (ASP.NET Core). |
| `ModelContextProtocol.Extensions.Tasks` | Long-running tools with client polling + pluggable persistence (opt-in). |
| `ModelContextProtocol.Extensions.Apps` | Interactive server-delivered UI (experimental; `MCPEXP003`). |

```bash
# Most servers:
dotnet add package ModelContextProtocol

# HTTP servers:
dotnet add package ModelContextProtocol.AspNetCore

# Just a client, or the low-level API:
dotnet add package ModelContextProtocol.Core
```

Target frameworks: **net8.0, net9.0, net10.0, and netstandard2.0** (the last for .NET Framework use).

---

## Stateless by default (and `SessionMode`)

In v2, `HttpServerTransportOptions.Stateless` defaults to `true` — HTTP servers run statelessly out
of the box. The single most important setting is now **`SessionMode`**:

```csharp
builder.Services.AddMcpServer()
    .WithHttpTransport(options =>
    {
        options.SessionMode = HttpServerSessionMode.Stateless; // default; set it explicitly
    })
    .WithToolsFromAssembly();
```

| `HttpServerSessionMode` | Use when |
| --- | --- |
| `Stateless` (default) | Almost always. Forward-compatible with `2026-07-28`; no `Mcp-Session-Id`; horizontal scaling, serverless, and edge deployments just work. Down-level clients still get the legacy handshake for that single POST. |
| `Stateful` | You need unsolicited notifications, resource subscriptions, per-client isolation, or server-to-client requests against clients that don't support MRTR. Refuses `2026-07-28` requests with `UnsupportedProtocolVersion` so dual-path clients downgrade to the handshake. |
| `StatefulForInitializeClients` | **Hybrid / migration mode.** `initialize`-handshake clients get full sessions while `2026-07-28` clients are served statelessly on the same endpoint. |

**Recommendations:**

- Always set `SessionMode` explicitly so behavior doesn't silently change on a future default bump.
- Prefer `Stateless`. Treat "stateless protocol" as different from "stateless application": when a
  tool needs continuity across calls, mint an explicit handle (`basketId`, `browserId`) from one
  tool and have the model pass it back as an ordinary argument — this composes better than hidden
  session state.
- Use `Stateful` only for unsolicited notifications, resource subscriptions, per-client isolation,
  or server-to-client requests to pre-MRTR clients.
- Use `StatefulForInitializeClients` when you must serve both session-dependent legacy clients and
  modern `2026-07-28` clients during migration.
- The bool `Stateless` property remains a shorthand (`true` → `Stateless`, `false` → `Stateful`);
  use `SessionMode` when you need the hybrid mode.

### Stateful-only options are obsolete

Because the 2026-07-28 wire format removes sessions, the stateful-only knobs on
`HttpServerTransportOptions` — `IdleTimeout`, `MaxIdleSessionCount`, `EventStreamStore`,
`SessionMigrationHandler`, `PerSessionExecutionContext` — are `[Obsolete]` under diagnostic
`MCP9006` (warnings, not removals; they still govern the back-compat handshake path).

### Legacy SSE disabled by default

The legacy `/sse` + `/message` endpoints are off by default; `EnableLegacySse` is `[Obsolete]`
(`MCP9004`). SSE returns `202 Accepted` before the handler runs — no HTTP-level backpressure — so
only enable it for clients that truly can't speak Streamable HTTP, and always with
`SessionMode = Stateful`.

---

## Standardized HTTP headers and `[McpHeader]`

On `2026-07-28`, every request carries the method and tool/resource name as headers
(`Mcp-Method: tools/call`, `Mcp-Name: get_order_status`) alongside the JSON-RPC body, so
intermediaries can route without body inspection. Promote any primitive tool parameter into an
`Mcp-Param-*` header with one attribute:

```csharp
[McpServerTool(Name = "get_order_status"),
 Description("Gets order status from the regional orders service")]
public static async Task<string> GetOrderStatus(
    OrdersServiceClient orders,
    [McpHeader("Region"), Description("Orders service region")] string region,
    [Description("The order to look up")] string orderId)
{
    // Client mirrors `region` into a request header:
    //   Mcp-Param-Region: eastus2
    return await orders.GetStatusAsync(region, orderId);
}
```

Rules:

- Emits an `x-mcp-header` keyword into the tool's input schema; clients lift the argument into a
  header on the wire.
- The **JSON-RPC body stays authoritative** — a header that disagrees with the body is rejected
  with a `HeaderMismatch` error.
- Primitive types only (`string`, numerics, `bool`); header names must be unique
  (case-insensitive) within the schema and ASCII-only; non-ASCII values are Base64-wrapped
  (`=?base64?{value}?=`).
- Enforced only on `2026-07-28` and later — additive and non-breaking for older clients.
- On the client, `Mcp-Param-*` headers are sent automatically for tools discovered via
  `ListToolsAsync`. Pre-load schemas out-of-band with `client.AddKnownTools([...])` (validated
  all-or-nothing) and remove with `RemoveKnownTools` / `ClearKnownTools`.

---

## Multi Round-Trip Requests (MRTR)

MRTR replaces server-initiated requests for elicitation, sampling, and roots. A tool returns an
incomplete result saying "I need something from you first"; the client resolves it and retries the
**same `tools/call`** with `inputResponses` plus an opaque `requestState`. Works with **no session
at all** — continuity travels in the payload.

### Server side

Throw `InputRequiredException` with input requests built via
`InputRequest.ForElicitation(...)`, `InputRequest.ForSampling(...)`, or
`InputRequest.ForRootsList(...)`. Check `server.IsMrtrSupported` first (true for `2026-07-28`
clients, and for stateful `2025-11-25` sessions where the SDK bridges to legacy elicitation).
On retry, read `context.Params.InputResponses` and `context.Params.RequestState`.

```csharp
[McpServerTool, Description("Closes a support ticket, recording why it was closed.")]
public static string CloseSupportTicket(
    McpServer server,
    RequestContext<CallToolRequestParams> context,
    [Description("The ID of the ticket to close")] long ticketId,
    [Description("Why the ticket is being closed")] string? closeReason = null)
{
    string defaultCloseReason = "completed";
    var confirmedReason = closeReason;

    // Retry path: the client answered via MRTR (or bridged legacy elicitation)
    if (string.IsNullOrWhiteSpace(confirmedReason) &&
        context.Params?.InputResponses?.TryGetValue("closeReason", out var reasonResponse) is true)
    {
        var reasonResult = reasonResponse.Deserialize(InputResponse.ElicitResultJsonTypeInfo);
        if (reasonResult?.IsAccepted is not true) return "Ticket close cancelled";

        confirmedReason = reasonResult.Content?.TryGetValue("closeReason", out var v) is true
            ? v.GetString() : null;
        confirmedReason = string.IsNullOrWhiteSpace(confirmedReason)
            ? defaultCloseReason : confirmedReason;
    }

    if (!string.IsNullOrWhiteSpace(confirmedReason))
        return $"Closed ticket {ticketId}: {confirmedReason}";

    if (server.IsMrtrSupported)
    {
        throw new InputRequiredException(
            inputRequests: new Dictionary<string, InputRequest>
            {
                ["closeReason"] = InputRequest.ForElicitation(new ElicitRequestParams
                {
                    Message = $"Close ticket '{ticketId}'? Accept the default reason or provide your own.",
                    RequestedSchema = new()
                    {
                        Properties =
                        {
                            ["closeReason"] = new ElicitRequestParams.StringSchema
                            {
                                Title = "Close reason",
                                Description = "The reason for closing the ticket",
                                Default = defaultCloseReason,
                            },
                        },
                    },
                })
            },
            requestState: ticketId.ToString()); // opaque; echoed back on the retry
    }

    // Session-less down-level client: can't prompt. Fall back to a guidance message.
    return "Closing a ticket requires a reason. Resend with `closeReason`.";
}
```

Deserializing `InputResponses` entries (no on-the-wire discriminator — use the type matching the
request's method):

| Input request | Deserialize call |
| --- | --- |
| Elicitation | `response.Deserialize(InputResponse.ElicitResultJsonTypeInfo)` |
| Sampling | `response.Deserialize(InputResponse.CreateMessageResultJsonTypeInfo)` |
| Roots list | `response.Deserialize(InputResponse.ListRootsResultJsonTypeInfo)` |

Other patterns:

- **Load shedding**: throw `InputRequiredException` with only `requestState` (no input requests) —
  the client retries automatically, echoing the state. Useful to break long work across requests.
- **Multi-round**: throw `InputRequiredException` repeatedly; track the round inside `requestState`.
- **URL-mode elicitation** (secure out-of-band consent, e.g. third-party OAuth): throw
  `UrlElicitationRequiredException` — the client presents a server-hosted URL, gathers consent out
  of band, and retries.

### Client side

The high-level `McpClient` resolves MRTR **automatically** — register handlers and `CallToolAsync`
returns the final result:

```csharp
var client = await McpClient.CreateAsync(
    clientTransport,
    clientOptions: new()
    {
        Handlers = new McpClientHandlers
        {
            ElicitationHandler = (requestParams, ct) =>
                ValueTask.FromResult(new ElicitResult { Action = "accept" }),
        }
    });

var result = await client.CallToolAsync(
    "close_support_ticket",
    new Dictionary<string, object?> { ["ticketId"] = 1234L },
    cancellationToken: CancellationToken.None);
```

### MRTR compatibility matrix

| Negotiated protocol | Session mode | MRTR behavior |
| --- | --- | --- |
| `2026-07-28` | Stateless | **Native** — no server-side handler state needed |
| `2026-07-28` | Stateful | **Native** — `InputRequiredResult` on the wire |
| `2025-11-25` and earlier | Stateful | **Back-compat resolver** — SDK bridges to legacy server-initiated requests |
| `2025-11-25` and earlier | Stateless | **Not supported** — surfaces as `McpException` |

### Deprecations replaced by MRTR

Under `2026-07-28` the spec removes `elicitation/create`, `sampling/createMessage`, and
`roots/list` as server-to-client methods:

- `ElicitAsync` → use `InputRequiredException` + `InputRequest.ForElicitation`. Still works on
  stateful sessions; throws `InvalidOperationException` in stateless mode.
- `SampleAsync` / `RequestRootsAsync` → deprecated (`MCP9005`, per SEP-2577); still available on
  stateful sessions, throw in stateless mode.
- MCP Logging → deprecated (`MCP9005`) in favor of stderr + OpenTelemetry.

---

## Extensions: Tasks and MCP Apps

The 2026-07-28 revision makes extensions first-class, capability-negotiated concepts — and the SDK
ships them as **separate opt-in packages** ("pay as you go": the base packages stay lean).

### Tasks (`ModelContextProtocol.Extensions.Tasks`)

Long-running tool execution with client-side polling and pluggable persistence (SEP-2663). Task
lifecycle: `working → input_required → completed | cancelled | failed`.

```csharp
using ModelContextProtocol.Extensions.Tasks;

builder.Services.AddMcpServer()
    .WithTools<MyTools>()
    .WithTasks(new InMemoryMcpTaskStore());
```

`WithTasks(store)` automatically wires `tasks/get`, `tasks/update`, `tasks/cancel`; advertises the
`io.modelcontextprotocol/tasks` extension; offloads opted-in tool calls to background tasks;
surfaces `ElicitAsync`/`SampleAsync`/`RequestRootsAsync` calls inside the tool as task
`inputRequests` (MRTR flows through the task store); and plumbs cooperative cancellation from
`tasks/cancel`.

- `InMemoryMcpTaskStore` is for **development and testing** — implement `IMcpTaskStore` over durable
  shared storage for tasks that must survive restarts or span instances.
- Client auto-polling: `client.CallToolWithPollingAsync(...)` drives the whole lifecycle and
  dispatches input requests to your `SamplingHandler`/`ElicitationHandler`.
- Client manual control: `CallToolAsTaskAsync` → `GetTaskAsync` / `UpdateTaskAsync` /
  `CancelTaskAsync`.
- Stuck-task detector: `CallToolWithPollingAsync` gives up after `maxConsecutiveStuckPolls`
  (default 60) polls stuck in `InputRequired` with no new keys, then cancels and throws.

> **Migration note:** v2 Tasks are **not** wire-compatible with the experimental v1 Tasks
> (`McpServerOptions.TaskStore` / `McpClientOptions.TaskStore`). Move to the
> `ModelContextProtocol.Extensions.Tasks` package and the APIs above.

### MCP Apps (`ModelContextProtocol.Extensions.Apps`)

Interactive, server-delivered UIs inside supporting clients (SEP-1865). Enable with
`.WithMcpApps()` and annotate tools that provide UI resources. **Experimental** — APIs require
suppressing diagnostic `MCPEXP003`.

---

## Identity, roles, and authorization

The SDK propagates the caller's identity from the transport into handlers:

```
HTTP request (auth token)
  → ASP.NET Core auth middleware (HttpContext.User)
  → MCP transport copies User into JsonRpcMessage.Context.User
  → filters → handler / tool method
```

**Recommended: `ClaimsPrincipal` parameter injection.** The SDK excludes it from the input schema
and resolves it per-request (null on stdio unless set via a message filter):

```csharp
[McpServerTool, Description("Returns a personalized greeting.")]
public static string Greet(ClaimsPrincipal? user, string message)
    => $"{user?.Identity?.Name ?? "anonymous"}: {message}";
```

**Declarative authorization** with standard ASP.NET Core attributes — requires
`AddAuthorizationFilters()`:

```csharp
services.AddMcpServer()
    .WithHttpTransport()
    .AddAuthorizationFilters()
    .WithTools<RoleProtectedTools>();

[McpServerToolType]
public class RoleProtectedTools
{
    [McpServerTool, Authorize]
    public string GetData(string query) => $"Data for: {query}";

    [McpServerTool, Authorize(Roles = "Admin")]
    public string AdminOperation(string action) => $"Admin action: {action}";

    [McpServerTool, AllowAnonymous]
    public string PublicInfo() => "public";
}
```

On failure: list operations silently drop unauthorized items; individual calls return a JSON-RPC
forbidden error.

**Filters** (`WithRequestFilters` / `WithMessageFilters`) expose `context.User` for auditing,
logging, or custom checks — e.g. synthesize a `ClaimsPrincipal` on stdio from process-level
context. `IHttpContextAccessor` works too but is HTTP-only (and can return stale claims on legacy
SSE); prefer `ClaimsPrincipal` for transport-agnostic code.

Cross-application SSO is supported via the **Identity Assertion Authorization Grant (ID-JAG)**
flow: `IdentityAssertionGrantProvider` performs the RFC 8693 token exchange at the enterprise IdP
and the RFC 7523 JWT bearer grant at the MCP authorization server, caching the access token until
expiry (`InvalidateCache()` to force re-auth).

---

## Security hardening (HTTP)

- **Host validation**: Kestrel doesn't validate `Host` headers — never use `AllowedHosts: "*"` on a
  local server (DNS-rebinding protection). Use `localhost;127.0.0.1;[::1]` in dev and the exact
  public hostnames in prod (validate at the proxy if it forwards `Host`).
- **CORS**: only enable for intentional browser access, with the most restrictive policy possible.
  A stateless browser client typically needs only `Content-Type`, `Authorization`, and
  `MCP-Protocol-Version` headers; add/expose `Mcp-Session-Id` and `Last-Event-ID` only if sessions
  are enabled.

```csharp
app.MapMcp("/mcp").RequireCors("McpBrowserClient");
```

- **stdio client env vars**: `StdioClientTransportOptions` inherits **all** parent env vars by
  default — including `AWS_SECRET_ACCESS_KEY`, `GITHUB_TOKEN`, `OPENAI_API_KEY`. For third-party or
  untrusted servers set `InheritEnvironmentVariables = false` and pass
  `StdioClientTransportOptions.GetDefaultEnvironmentVariables()` (curated PATH/HOME/etc.) plus any
  server-specific keys.

---

## Error handling model (v2)

- Tool exceptions → `CallToolResult` with `IsError = true` (the LLM sees and can recover).
  - Exceptions derived from `McpException` (except `McpProtocolException`) → message included.
  - All other exceptions → generic `"An error occurred invoking '<tool>'."` to avoid leaking
    internals.
- `McpProtocolException` → propagates as a JSON-RPC error (e.g. `McpErrorCode.InvalidParams`).
- `OperationCanceledException` re-throws when the token fired.
- Client side: check `result.IsError` after `CallToolAsync`.

---

## Diagnostics quick reference

| Diagnostic | Trigger |
| --- | --- |
| `MCP9004` | Legacy SSE usage (`EnableLegacySse` — off by default) |
| `MCP9005` | Deprecated server-initiated requests (`SampleAsync`, `RequestRootsAsync`, legacy elicitation path, MCP Logging) |
| `MCP9006` | Stateful-only transport options (`IdleTimeout`, `MaxIdleSessionCount`, `EventStreamStore`, `SessionMigrationHandler`, `PerSessionExecutionContext`) |
| `MCPEXP003` | Experimental MCP Apps APIs |

All are **warnings, not removals** — migrate at your own pace.

## v1 → v2 migration summary

| v1 | v2 |
| --- | --- |
| HTTP transport defaults to stateful | Accept the stateless default, or set `SessionMode = Stateful`/`StatefulForInitializeClients` explicitly |
| Experimental Tasks in Core via `McpServerOptions.TaskStore` | `ModelContextProtocol.Extensions.Tasks`; `.WithTasks(store)` on the server, `CallToolWithPollingAsync`/`CallToolAsTaskAsync` on the client |
| Clients start with `initialize` handshake | v2 prefers `2026-07-28` and falls back automatically; pin `McpClientOptions.ProtocolVersion` only for strict behavior |
| `ElicitAsync` / `SampleAsync` / `RequestRootsAsync` inside tools | `InputRequiredException` + `InputRequest.For*` (MRTR); legacy calls throw in stateless mode |
| Sessions required for interactive tools | MRTR works statelessly; sessions only for unsolicited notifications / subscriptions / per-client isolation |

## Quality checklist additions (v2)

- [ ] `SessionMode` set explicitly (usually `Stateless`)
- [ ] No reliance on `Mcp-Session-Id` or the `initialize` handshake for new HTTP clients
- [ ] Interactive tools use `InputRequiredException` (MRTR) with an up-front argument fallback for session-less down-level clients
- [ ] `[McpHeader]` on parameters intermediaries must route on (region, tenant, shard)
- [ ] Long-running tools evaluated for the Tasks extension; `IMcpTaskStore` durable in production
- [ ] `AllowedHosts` locked down; CORS minimal (or absent)
- [ ] `ClaimsPrincipal` injection + `AddAuthorizationFilters()` for authenticated servers
- [ ] Build clean of `MCP9004`/`MCP9005`/`MCP9006` warnings (or suppressions are deliberate)
- [ ] stdio clients disable env inheritance for untrusted servers

## Links

- Conceptual docs hub: <https://csharp.sdk.modelcontextprotocol.io/v2/concepts/index.html>
- Stateless vs stateful: <https://csharp.sdk.modelcontextprotocol.io/v2/concepts/stateless/stateless.html>
- MRTR: <https://csharp.sdk.modelcontextprotocol.io/v2/concepts/mrtr/mrtr.html>
- Tools: <https://csharp.sdk.modelcontextprotocol.io/v2/concepts/tools/tools.html>
- Transports: <https://csharp.sdk.modelcontextprotocol.io/v2/concepts/transports/transports.html>
- Tasks: <https://csharp.sdk.modelcontextprotocol.io/v2/concepts/tasks/tasks.html>
- Identity and roles: <https://csharp.sdk.modelcontextprotocol.io/v2/concepts/identity/identity.html>
- Samples: <https://github.com/modelcontextprotocol/csharp-sdk/tree/main/samples>
