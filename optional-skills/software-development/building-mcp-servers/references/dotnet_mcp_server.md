# .NET/C# MCP Server Implementation Guide

## Overview

This document provides .NET/C#-specific best practices and examples for implementing MCP servers using the ModelContextProtocol.AspNetCore SDK. It covers project structure, server setup, tool registration patterns, input validation, error handling, and complete working examples.

> **SDK 2.0:** The MCP C# SDK is at **v2.0** (implements spec revision `2026-07-28`). HTTP servers are **stateless by default**, interactive tools use **MRTR** (`InputRequiredException`), and Tasks/MCP Apps ship as opt-in extension packages. See [🆕 MCP C# SDK 2.0 — What's New](./dotnet_mcp_v2.md) for the full v2 guide; the patterns below remain valid on 2.0.

---

## Quick Reference

### Key Imports

```csharp
using ModelContextProtocol.Server;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.AI;        // ChatMessage / ChatRole (prompts)
using System.ComponentModel;         // [Description]
using System.Text.Json;              // JsonSerializer
using System.Net.Http.Json;          // ReadFromJsonAsync / PostAsJsonAsync
```

### Server Initialization

```csharp
var builder = WebApplication.CreateBuilder(args);

builder.Services
    .AddMcpServer()
    .WithHttpTransport()
    .WithToolsFromAssembly()
    .WithPromptsFromAssembly()
    .WithResourcesFromAssembly();

var app = builder.Build();
app.MapMcp();
app.Run();
```

### Tool Registration Pattern

```csharp
[McpServerToolType]
public static class MyTools
{
    [McpServerTool, Description("Tool description")]
    public static async Task<string> ToolName(
        [Description("Parameter description")] string param)
    {
        return $"Processed: {param}";
    }
}
```

---

## MCP .NET SDK

The official MCP C# SDK is **stable** and at **v2.0** (implements spec revision `2026-07-28`). It ships as **three base NuGet packages** plus opt-in extension packages. Pick the smallest one that covers your scenario:

| Package | Use it for | References |
| --- | --- | --- |
| **`ModelContextProtocol.Core`** | Client-only apps or low-level server APIs with minimum dependencies | — |
| **`ModelContextProtocol`** | stdio / local servers — adds hosting + DI extensions | `ModelContextProtocol.Core` |
| **`ModelContextProtocol.AspNetCore`** | HTTP / remote servers (Streamable HTTP) | `ModelContextProtocol` |
| **`ModelContextProtocol.Extensions.Tasks`** | Long-running tools with client polling (opt-in) | `ModelContextProtocol.Core` |
| **`ModelContextProtocol.Extensions.Apps`** | Interactive server-delivered UI (experimental, `MCPEXP003`) | `ModelContextProtocol.Core` |

Packages target **net8.0, net9.0, net10.0, and netstandard2.0** (the last for .NET Framework).

### Choosing a package + transport

| Scenario | Transport | Package |
| --- | --- | --- |
| Local dev tool, single user, same machine as the IDE | stdio | `ModelContextProtocol` |
| Distributed via NuGet, run on demand by `dnx` | stdio | `ModelContextProtocol` + `.mcp/server.json` |
| Internal team service behind organizational auth | Streamable HTTP | `ModelContextProtocol.AspNetCore` |
| Public SaaS exposing tools to many users | Streamable HTTP + OAuth 2.1 | `ModelContextProtocol.AspNetCore` |
| Library that only calls remote MCP servers | n/a (client only) | `ModelContextProtocol.Core` |

All server packages provide:

- `AddMcpServer()` extension method for DI registration
- `WithHttpTransport()` for Streamable HTTP transport (AspNetCore package)
- `WithStdioServerTransport()` for local stdio transport
- Attribute-based tool registration with `[McpServerTool]`
- Dependency injection support for tools
- Automatic schema generation from method signatures

**IMPORTANT - Use Modern APIs Only:**

- **DO use**: `[McpServerTool]` attribute, `WithToolsFromAssembly()`, `WithHttpTransport()`
- **DO use** the stable `2.x` packages — the `--prerelease` flag is no longer required
- **DO use** `SessionMode` (usually `HttpServerSessionMode.Stateless`, the v2 default) instead of relying on sessions
- **DO use** `InputRequiredException` (MRTR) for elicitation/sampling/roots on new servers — `ElicitAsync`/`SampleAsync`/`RequestRootsAsync` throw in stateless mode
- **DO NOT use**: Manual handler registration, deprecated SSE-only transports (`EnableLegacySse`, `MCP9004`), stateful-only options (`MCP9006`), or `0.x` preview packages
- The attribute-based approach provides automatic schema generation and DI integration
- See [🆕 MCP C# SDK 2.0](./dotnet_mcp_v2.md) for `SessionMode`, MRTR, `[McpHeader]`, Tasks/Apps, and the v1→v2 migration table

### Fastest start: project template

The `Microsoft.McpServer.ProjectTemplates` template scaffolds a ready-to-run stdio server (Program.cs, a sample tool, and `.mcp/server.json` for publishing):

```bash
dotnet new install Microsoft.McpServer.ProjectTemplates
dotnet new mcpserver -n MyService.McpServer
cd MyService.McpServer
dotnet run
```

See the MCP SDK documentation in the references for complete details.

## Server Naming Convention

.NET/C# MCP servers must follow this naming pattern:

- **Format**: `{Service}.McpServer` (PascalCase with namespace)
- **Project name**: `{service}-mcp-server` (lowercase with hyphens for package/CLI)
- **Examples**: `GitHub.McpServer`, `Jira.McpServer`, `Stripe.McpServer`

The name should be:

- General (not tied to specific features)
- Descriptive of the service/API being integrated
- Easy to infer from the task description
- Without version numbers or dates

## Project Structure

Create the following structure for .NET/C# MCP servers:

```text
{service}-mcp-server/
├── {Service}.McpServer.csproj
├── README.md
├── Program.cs                 # Main entry point with server initialization
├── Tools/                     # Tool implementations (one file per domain)
│   ├── UserTools.cs
│   ├── ProjectTools.cs
│   └── DataTools.cs
├── Services/                  # API clients and shared utilities
│   ├── ApiService.cs
│   └── HttpClientFactory.cs
├── Models/                    # Data models and DTOs
│   ├── User.cs
│   └── Project.cs
└── Constants.cs               # Shared constants (API_URL, CHARACTER_LIMIT, etc.)
```

## Tool Implementation

### Tool Naming

Use PascalCase for tool names (e.g., "SearchUsers", "CreateProject", "GetChannelInfo") with clear, action-oriented names.

**Avoid Naming Conflicts**: Include the service context to prevent overlaps:

- Use "SlackSendMessage" instead of just "SendMessage"
- Use "GitHubCreateIssue" instead of just "CreateIssue"
- Use "AsanaListTasks" instead of just "ListTasks"

### Tool Structure

Tools are registered using the `[McpServerTool]` attribute with the following requirements:

- Mark the class with `[McpServerToolType]`
- Mark each tool method with `[McpServerTool]`
- Use `[Description]` attribute for tool and parameter descriptions
- Use dependency injection for services (HttpClient, etc.)
- Type all parameters and return values explicitly
- Use async/await for I/O operations

> **Parameter ordering (compile-time rule):** C# requires that every parameter **with a default value comes after all parameters without one** (otherwise you get error **CS1737**). Because the SDK resolves DI services (e.g. `HttpClient`, custom services) from the method signature, the safest convention is **DI services first (no default), then required tool params, then optional tool params, with `CancellationToken ct = default` last**. Do not give injected services a `= null` default just to satisfy ordering — reorder instead.

```csharp
using System.ComponentModel;
using ModelContextProtocol.Server;

[McpServerToolType]
public static class UserTools
{
    [McpServerTool, Description("Search for users in the Example system by name, email, or team.\n\n" +
        "This tool searches across all user profiles in the Example platform, supporting partial matches and various search filters. " +
        "It does NOT create or modify users, only searches existing ones.\n\n" +
        "Args:\n" +
        "  - query (string): Search string to match against names/emails\n" +
        "  - limit (number): Maximum results to return, between 1-100 (default: 20)\n" +
        "  - offset (number): Number of results to skip for pagination (default: 0)\n\n" +
        "Returns:\n" +
        "  Structured data with schema:\n" +
        "  {\n" +
        "    \"total\": number,           // Total number of matches found\n" +
        "    \"count\": number,           // Number of results in this response\n" +
        "    \"offset\": number,          // Current pagination offset\n" +
        "    \"users\": [\n" +
        "      {\n" +
        "        \"id\": string,          // User ID (e.g., \"U123456789\")\n" +
        "        \"name\": string,        // Full name (e.g., \"John Doe\")\n" +
        "        \"email\": string,       // Email address\n" +
        "        \"team\": string,        // Team name (optional)\n" +
        "        \"active\": boolean      // Whether user is active\n" +
        "      }\n" +
        "    ],\n" +
        "    \"has_more\": boolean,       // Whether more results are available\n" +
        "    \"next_offset\": number      // Offset for next page (if has_more is true)\n" +
        "  }\n\n" +
        "Examples:\n" +
        "  - Use when: \"Find all marketing team members\" -> params with query=\"team:marketing\"\n" +
        "  - Use when: \"Search for John's account\" -> params with query=\"john\"\n" +
        "  - Don't use when: You need to create a user (use CreateUser instead)\n\n" +
        "Error Handling:\n" +
        "  - Returns \"Error: Rate limit exceeded\" if too many requests (429 status)\n" +
        "  - Returns \"No users found matching '<query>'\" if search returns empty")]
    public static async Task<string> SearchUsers(
        HttpClient http,  // Injected from DI (services first)
        [Description("Search string to match against names/emails")] string query,
        [Description("Maximum results to return (1-100)")] int limit = 20,
        [Description("Number of results to skip for pagination")] int offset = 0,
        CancellationToken ct = default)
    {
        try
        {
            // Make API request using injected HttpClient
            var response = await http.GetAsync(
                $"https://api.example.com/v1/users/search?q={Uri.EscapeDataString(query)}&limit={limit}&offset={offset}",
                ct);

            response.EnsureSuccessStatusCode();
            var data = await response.Content.ReadFromJsonAsync<UserSearchResponse>(ct);

            if (data?.Users == null || !data.Users.Any())
            {
                return $"No users found matching '{query}'";
            }

            // Format structured output
            var output = new UserSearchOutput
            {
                Total = data.Total,
                Count = data.Users.Count,
                Offset = offset,
                Users = data.Users.Select(u => new UserOutput
                {
                    Id = u.Id,
                    Name = u.Name,
                    Email = u.Email,
                    Team = u.Team,
                    Active = u.Active
                }).ToList(),
                HasMore = data.Total > offset + data.Users.Count,
                NextOffset = data.Total > offset + data.Users.Count ? offset + data.Users.Count : null
            };

            return JsonSerializer.Serialize(output, new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            });
        }
        catch (HttpRequestException ex) when (ex.StatusCode == System.Net.HttpStatusCode.TooManyRequests)
        {
            return "Error: Rate limit exceeded. Please wait before making more requests.";
        }
        catch (HttpRequestException ex) when (ex.StatusCode == System.Net.HttpStatusCode.NotFound)
        {
            return "Error: Resource not found. Please check the API endpoint.";
        }
        catch (Exception ex)
        {
            return $"Error: {ex.Message}";
        }
    }
}
```

## Input Validation

.NET uses parameter validation through attributes and runtime checks:

```csharp
using System.ComponentModel.DataAnnotations;

[McpServerToolType]
public static class UserTools
{
    [McpServerTool]
    public static async Task<string> CreateUser(
        [Description("User name (1-100 characters)")] 
        [StringLength(100, MinimumLength = 1, ErrorMessage = "Name must be between 1-100 characters")]
        string name,
        
        [Description("User email address")]
        [EmailAddress(ErrorMessage = "Invalid email format")]
        string email,
        
        [Description("User age")]
        [Range(0, 150, ErrorMessage = "Age must be between 0-150")]
        int age)
    {
        // Validation is handled by Data Annotations
        // Additional validation logic can be added here
        return $"User created: {name}";
    }
}
```

For complex validation, use manual checks:

```csharp
[McpServerTool]
public static async Task<string> SearchUsers(
    [Description("Search query (2-200 characters)")] string query,
    [Description("Maximum results (1-100)")] int limit = 20)
{
    // Manual validation
    if (string.IsNullOrWhiteSpace(query) || query.Length < 2)
    {
        return "Error: Query must be at least 2 characters";
    }
    
    if (query.Length > 200)
    {
        return "Error: Query must not exceed 200 characters";
    }
    
    if (limit < 1 || limit > 100)
    {
        return "Error: Limit must be between 1-100";
    }
    
    // Proceed with implementation
    return $"Searching for: {query}";
}
```

## Response Format Options

Support multiple output formats for flexibility:

```csharp
public enum ResponseFormat
{
    Markdown,
    Json
}

[McpServerTool]
public static async Task<string> SearchUsers(
    [Description("Search query")] string query,
    [Description("Output format")] ResponseFormat format = ResponseFormat.Markdown)
{
    var users = await GetUsersAsync(query);
    
    if (format == ResponseFormat.Json)
    {
        return JsonSerializer.Serialize(users, new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        });
    }
    
    // Markdown format
    var sb = new StringBuilder();
    sb.AppendLine("# User Search Results");
    sb.AppendLine($"Found {users.Count} users");
    sb.AppendLine();
    
    foreach (var user in users)
    {
        sb.AppendLine($"## {user.Name} ({user.Id})");
        sb.AppendLine($"- **Email**: {user.Email}");
        if (!string.IsNullOrEmpty(user.Team))
        {
            sb.AppendLine($"- **Team**: {user.Team}");
        }
        sb.AppendLine();
    }
    
    return sb.ToString();
}
```

**Markdown format**:

- Use headers, lists, and formatting for clarity
- Convert timestamps to human-readable format
- Show display names with IDs in parentheses
- Omit verbose metadata
- Group related information logically

**JSON format**:

- Return complete, structured data suitable for programmatic processing
- Include all available fields and metadata
- Use consistent field names and types (camelCase)

## Pagination Implementation

For tools that list resources:

```csharp
public class PaginationResult<T>
{
    public int Total { get; set; }
    public int Count { get; set; }
    public int Offset { get; set; }
    public List<T> Items { get; set; }
    public bool HasMore { get; set; }
    public int? NextOffset { get; set; }
}

[McpServerTool]
public static async Task<string> ListUsers(
    HttpClient http,
    [Description("Maximum results (1-100)")] int limit = 20,
    [Description("Number of results to skip")] int offset = 0,
    CancellationToken ct = default)
{
    var response = await http.GetAsync(
        $"https://api.example.com/v1/users?limit={limit}&offset={offset}",
        ct);

    var data = await response.Content.ReadFromJsonAsync<UserListResponse>(ct);
    
    var result = new PaginationResult<UserOutput>
    {
        Total = data.Total,
        Count = data.Users.Count,
        Offset = offset,
        Items = data.Users,
        HasMore = data.Total > offset + data.Users.Count,
        NextOffset = data.Total > offset + data.Users.Count ? offset + data.Users.Count : null
    };
    
    return JsonSerializer.Serialize(result, new JsonSerializerOptions
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    });
}
```

## Character Limits and Truncation

Add a CHARACTER_LIMIT constant to prevent overwhelming responses:

```csharp
// In Constants.cs
public static class Constants
{
    public const int CharacterLimit = 25000;
}

[McpServerTool]
public static async Task<string> SearchUsers(
    HttpClient http,
    [Description("Search query")] string query,
    CancellationToken ct = default)
{
    var response = await http.GetAsync($"https://api.example.com/v1/users?q={query}", ct);
    var data = await response.Content.ReadFromJsonAsync<UserSearchResponse>(ct);
    
    var result = JsonSerializer.Serialize(data, new JsonSerializerOptions
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    });
    
    // Check character limit and truncate if needed
    if (result.Length > Constants.CharacterLimit)
    {
        var truncatedData = data.Users.Take(data.Users.Count / 2).ToList();
        data.Users = truncatedData;
        
        var truncatedResult = JsonSerializer.Serialize(data, new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        });
        
        return $"{truncatedResult}\n\n" +
               $"Response truncated from {data.Users.Count} to {truncatedData.Count} items. " +
               $"Use 'offset' parameter or add filters to see more results.";
    }
    
    return result;
}
```

## Error Handling

Provide clear, actionable error messages:

```csharp
[McpServerTool]
public static async Task<string> GetUser(
    HttpClient http,
    [Description("User ID")] string userId,
    CancellationToken ct = default)
{
    try
    {
        var response = await http.GetAsync($"https://api.example.com/v1/users/{userId}", ct);
        
        response.EnsureSuccessStatusCode();
        
        var user = await response.Content.ReadFromJsonAsync<User>(ct);
        return JsonSerializer.Serialize(user, new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        });
    }
    catch (HttpRequestException ex) when (ex.StatusCode == System.Net.HttpStatusCode.NotFound)
    {
        return "Error: User not found. Please check the user ID is correct.";
    }
    catch (HttpRequestException ex) when (ex.StatusCode == System.Net.HttpStatusCode.Forbidden)
    {
        return "Error: Permission denied. You don't have access to this user.";
    }
    catch (HttpRequestException ex) when (ex.StatusCode == System.Net.HttpStatusCode.TooManyRequests)
    {
        return "Error: Rate limit exceeded. Please wait before making more requests.";
    }
    catch (TaskCanceledException)
    {
        return "Error: Request timed out. Please try again.";
    }
    catch (Exception ex)
    {
        return $"Error: Unexpected error occurred: {ex.Message}";
    }
}
```

## Shared Utilities

Extract common functionality into reusable services:

```csharp
// Services/ApiService.cs
public class ApiService
{
    private readonly HttpClient _http;
    private readonly string _baseUrl;
    
    public ApiService(HttpClient http, IConfiguration config)
    {
        _http = http;
        _baseUrl = config["ApiBaseUrl"];
    }
    
    public async Task<T> GetAsync<T>(string endpoint, CancellationToken ct = default)
    {
        var response = await _http.GetAsync($"{_baseUrl}/{endpoint}", ct);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<T>(ct);
    }
    
    public async Task<T> PostAsync<T>(string endpoint, object data, CancellationToken ct = default)
    {
        var response = await _http.PostAsJsonAsync($"{_baseUrl}/{endpoint}", data, ct);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<T>(ct);
    }
}

// Register in DI
builder.Services.AddScoped<ApiService>();
```

## Dependency Injection

Use .NET's built-in DI for services:

```csharp
// Program.cs
var builder = WebApplication.CreateBuilder(args);

// Register HttpClient
builder.Services.AddHttpClient<ApiService>(client =>
{
    client.BaseAddress = new Uri(builder.Configuration["ApiBaseUrl"]);
    client.Timeout = TimeSpan.FromSeconds(30);
});

// Register custom services
builder.Services.AddScoped<UserService>();
builder.Services.AddScoped<ProjectService>();

// Register MCP server
builder.Services
    .AddMcpServer()
    .WithHttpTransport()
    .WithToolsFromAssembly()
    .WithPromptsFromAssembly()
    .WithResourcesFromAssembly();

var app = builder.Build();
app.MapMcp();
app.Run();
```

Inject services in tools:

```csharp
[McpServerToolType]
public static class UserTools
{
    [McpServerTool]
    public static async Task<string> GetUser(
        UserService userService,
        [Description("User ID")] string userId,
        CancellationToken ct = default)
    {
        var user = await userService.GetUserAsync(userId, ct);
        return JsonSerializer.Serialize(user);
    }
}
```

## Async/Await Best Practices

Always use async/await for network requests and I/O operations:

```csharp
// Good: Async with proper cancellation
[McpServerTool]
public static async Task<string> FetchData(
    HttpClient http,
    [Description("Resource ID")] string resourceId,
    CancellationToken ct = default)
{
    var response = await http.GetAsync($"https://api.example.com/v1/resource/{resourceId}", ct);
    var data = await response.Content.ReadAsStringAsync(ct);
    return data;
}

// Bad: Synchronous blocking
[McpServerTool]
public static string FetchDataBad(
    [Description("Resource ID")] string resourceId,
    HttpClient http = null)
{
    var response = http.GetAsync($"https://api.example.com/v1/resource/{resourceId}").Result; // Blocks thread
    var data = response.Content.ReadAsStringAsync().Result; // Blocks thread
    return data;
}
```

## .NET Best Practices

1. **Use Async All The Way**: Always use async/await for I/O operations
2. **Pass CancellationToken**: Support cancellation for long-running operations
3. **Use HttpClient via DI**: Don't create HttpClient instances manually
4. **Proper Exception Handling**: Use specific exception types and pattern matching
5. **Null Safety**: Use null-conditional operators and null-coalescing
6. **Immutable Types**: Use records for data transfer objects
7. **JSON Serialization**: Use System.Text.Json with proper options

```csharp
// Good: Async with cancellation, proper error handling
[McpServerTool]
public static async Task<string> GetUser(
    UserService userService,
    [Description("User ID")] string userId,
    CancellationToken ct = default)
{
    try
    {
        var user = await userService.GetUserAsync(userId, ct);
        if (user == null)
        {
            return "Error: User not found";
        }
        
        return JsonSerializer.Serialize(user, new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        });
    }
    catch (OperationCanceledException)
    {
        return "Error: Operation was cancelled";
    }
    catch (Exception ex)
    {
        return $"Error: {ex.Message}";
    }
}

// Bad: No cancellation, generic exception handling
[McpServerTool]
public static string GetUserBad(
    [Description("User ID")] string userId,
    UserService userService)
{
    try
    {
        var user = userService.GetUser(userId).Result; // Blocks thread
        return JsonSerializer.Serialize(user);
    }
    catch (Exception ex)
    {
        return $"Error: {ex.Message}"; // Too generic
    }
}
```

## Project Configuration

### .csproj

For an **HTTP / remote** server, use the `Microsoft.NET.Sdk.Web` SDK and the `ModelContextProtocol.AspNetCore` package:

```xml
<Project Sdk="Microsoft.NET.Sdk.Web">

  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
    <RootNamespace>{Service}.McpServer</RootNamespace>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="ModelContextProtocol.AspNetCore" Version="2.0.0" />
    <PackageReference Include="Microsoft.Extensions.Http" Version="8.0.0" />
  </ItemGroup>

</Project>
```

For a **stdio / local** server, use the plain `Microsoft.NET.Sdk` and the lighter `ModelContextProtocol` package:

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
    <RootNamespace>{Service}.McpServer</RootNamespace>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="ModelContextProtocol" Version="2.0.0" />
    <PackageReference Include="Microsoft.Extensions.Hosting" Version="8.0.0" />
  </ItemGroup>

</Project>
```


### appsettings.json

```json
{
  "ApiBaseUrl": "https://api.example.com/v1",
  "Logging": {
    "LogLevel": {
      "Default": "Information",
      "Microsoft.AspNetCore": "Warning"
    }
  },
  "AllowedHosts": "*"
}
```

## Complete Example

```csharp
// Program.cs
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using ModelContextProtocol.Server;

var builder = WebApplication.CreateBuilder(args);

// Add services
builder.Services.AddHttpClient<UserService>(client =>
{
    client.BaseAddress = new Uri(builder.Configuration["ApiBaseUrl"]);
    client.Timeout = TimeSpan.FromSeconds(30);
});

builder.Services.AddScoped<UserService>();
builder.Services.AddScoped<ProjectService>();

// Add MCP server
builder.Services
    .AddMcpServer()
    .WithHttpTransport()
    .WithToolsFromAssembly()
    .WithPromptsFromAssembly()
    .WithResourcesFromAssembly();

var app = builder.Build();
app.MapMcp();
app.Run();

// Tools/UserTools.cs
using System.ComponentModel;
using ModelContextProtocol.Server;
using System.Text.Json;

[McpServerToolType]
public static class UserTools
{
    [McpServerTool, Description("Search for users by name or email")]
    public static async Task<string> SearchUsers(
        UserService userService,  // Injected from DI (services first)
        [Description("Search query")] string query,
        [Description("Maximum results")] int limit = 20,
        CancellationToken ct = default)
    {
        var users = await userService.SearchUsersAsync(query, limit, ct);
        
        return JsonSerializer.Serialize(users, new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        });
    }
}

// Services/UserService.cs
public class UserService
{
    private readonly HttpClient _http;
    
    public UserService(HttpClient http)
    {
        _http = http;
    }
    
    public async Task<List<User>> SearchUsersAsync(string query, int limit, CancellationToken ct = default)
    {
        var response = await _http.GetAsync($"users/search?q={Uri.EscapeDataString(query)}&limit={limit}", ct);
        response.EnsureSuccessStatusCode();
        
        var data = await response.Content.ReadFromJsonAsync<UserSearchResponse>(ct);
        return data?.Users ?? new List<User>();
    }
    
    public async Task<User?> GetUserAsync(string userId, CancellationToken ct = default)
    {
        var response = await _http.GetAsync($"users/{userId}", ct);
        
        if (response.StatusCode == System.Net.HttpStatusCode.NotFound)
        {
            return null;
        }
        
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<User>(ct);
    }
}

// Models/User.cs
public record User(
    string Id,
    string Name,
    string Email,
    string? Team = null,
    bool Active = true
);

public record UserSearchResponse(
    int Total,
    List<User> Users
);
```

---

## Advanced MCP Features

### Resource Registration

Expose data as resources for efficient, URI-based access:

```csharp
[McpServerResourceType]
public static class MyResources
{
    [McpServerResource(Name = "config://app"), Description("Application configuration")]
    public static string GetConfig() => 
        JsonSerializer.Serialize(new { env = "production", version = "1.0.0" }, new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        });
    
    [McpServerResource(UriTemplate = "docs://{topic}"), Description("Documentation by topic")]
    public static string GetDocumentation([Description("Topic name")] string topic) =>
        $"Documentation for {topic}";
}
```

**When to use Resources vs Tools:**

- **Resources**: For data access with simple URI-based parameters
- **Tools**: For complex operations requiring validation and business logic
- **Resources**: When data is relatively static or template-based
- **Tools**: When operations have side effects or complex workflows

### Prompt Registration

Define prompts for LLMs:

```csharp
[McpServerPromptType]
public static class MyPrompts
{
    [McpServerPrompt, Description("Code review prompt")]
    public static ChatMessage CodeReview(
        [Description("Code to review")] string code,
        [Description("Programming language")] string language = "csharp") =>
        new(ChatRole.User, $"Review this {language} code:\n\n```{language}\n{code}\n```");
    
    [McpServerPrompt, Description("Generate documentation")]
    public static ChatMessage GenerateDocs(
        [Description("Code to document")] string code) =>
        new(ChatRole.User, $"Generate documentation for this code:\n\n{code}");
}
```

### Transport Options

The .NET SDK supports two main transport mechanisms:

#### Streamable HTTP (Recommended for Remote Servers)

```csharp
var builder = WebApplication.CreateBuilder(args);

builder.Services
    .AddMcpServer()
    .WithHttpTransport()
    .WithToolsFromAssembly();

var app = builder.Build();
app.MapMcp();  // Maps the Streamable HTTP MCP endpoint at the root path
app.Run();
```

In SDK v2, HTTP servers run **stateless by default** (`SessionMode = HttpServerSessionMode.Stateless`), matching the `2026-07-28` wire format — ideal for serverless and horizontally-scaled deployments. Set it explicitly and only opt into sessions when you need unsolicited notifications, resource subscriptions, per-client isolation, or server-to-client requests to pre-MRTR clients:

```csharp
builder.Services
    .AddMcpServer()
    .WithHttpTransport(options =>
    {
        // Stateless is the v2 default; set explicitly to pin behavior.
        options.SessionMode = HttpServerSessionMode.Stateless;
        // Or: HttpServerSessionMode.Stateful (sessions) /
        //     HttpServerSessionMode.StatefulForInitializeClients (hybrid migration)
    })
    .WithToolsFromAssembly();
```

#### Stdio (For Local Integrations)

```csharp
var builder = Host.CreateApplicationBuilder(args);

builder.Logging.AddConsole(o => o.LogToStandardErrorThreshold = LogLevel.Trace);

builder.Services
    .AddMcpServer()
    .WithStdioServerTransport()
    .WithToolsFromAssembly();

await builder.Build().RunAsync();
```

**Transport selection:**

- **Streamable HTTP**: Web services, remote access, multiple clients
- **stdio**: Command-line tools, local development, subprocess integration

---

## Code Best Practices

### Code Composability and Reusability

Your implementation MUST prioritize composability and code reuse:

1. **Extract Common Functionality**:
   - Create reusable services for operations used across multiple tools
   - Build shared API clients for HTTP requests instead of duplicating code
   - Centralize error handling logic in utility functions
   - Extract business logic into dedicated services that can be composed
   - Extract shared JSON or markdown formatting functionality

2. **Avoid Duplication**:
   - NEVER copy-paste similar code between tools
   - If you find yourself writing similar logic twice, extract it into a service
   - Common operations like pagination, filtering, and formatting should be shared
   - Authentication/authorization logic should be centralized

## Building and Running

Build and run your .NET MCP server:

```bash
# Build the project
dotnet build

# Run the server
dotnet run

# Run in development mode with hot reload
dotnet watch run

# Publish for deployment
dotnet publish -c Release -o ./publish
```

Always ensure `dotnet build` completes successfully before considering the implementation complete.

## Packaging and Distribution

Local servers can be published to the **NuGet MCP registry** so any MCP host can run them on demand with `dnx` (the .NET tool runner). Add a `.mcp/server.json` manifest describing the package and its inputs:

```json
{
  "$schema": "https://static.modelcontextprotocol.io/schemas/2025-10-17/server.schema.json",
  "description": "<server description>",
  "name": "io.github.<github-user>/<repo>",
  "version": "1.0.0",
  "packages": [
    {
      "registryType": "nuget",
      "registryBaseUrl": "https://api.nuget.org",
      "identifier": "<your.package.id>",
      "version": "1.0.0",
      "transport": { "type": "stdio" }
    }
  ]
}
```

Pack and push the NuGet package, then reference it from an MCP host config using `dnx`:

```bash
dotnet pack -c Release
dotnet nuget push bin/Release/*.nupkg --api-key <key> --source https://api.nuget.org/v3/index.json
```

```json
{
  "servers": {
    "MyService.McpServer": {
      "type": "stdio",
      "command": "dnx",
      "args": ["<your.package.id>@1.0.0", "--yes"]
    }
  }
}
```


## VS Code / MCP Host Configuration

Register the server with an MCP host (e.g. VS Code `.vscode/mcp.json`):

```json
{
  "servers": {
    "MyService.McpServer": {
      "type": "stdio",
      "command": "dotnet",
      "args": ["run", "--project", "<relative-path-to-csproj>"]
    }
  }
}
```

For an HTTP server, point the host at the running URL instead:

```json
{
  "servers": {
    "MyService.McpServer": {
      "type": "http",
      "url": "http://localhost:3001"
    }
  }
}
```

## Quality Checklist

Before finalizing your .NET/C# MCP server implementation, ensure:

### Strategic Design

- [ ] Tools enable complete workflows, not just API endpoint wrappers
- [ ] Tool names reflect natural task subdivisions
- [ ] Response formats optimize for agent context efficiency
- [ ] Human-readable identifiers used where appropriate
- [ ] Error messages guide agents toward correct usage

### Implementation Quality

- [ ] FOCUSED IMPLEMENTATION: Most important and valuable tools implemented
- [ ] All tools registered using `[McpServerTool]` attribute
- [ ] All tools have `[Description]` attributes for tool and parameters
- [ ] Dependency injection used for services (HttpClient, custom services)
- [ ] All async methods use proper async/await pattern
- [ ] CancellationToken passed to all async operations
- [ ] All tools have comprehensive descriptions with input/output types
- [ ] Error messages are clear, actionable, and educational

### .NET Quality

- [ ] Nullable reference types enabled in .csproj
- [ ] Records used for immutable data transfer objects
- [ ] HttpClient registered via DI, not instantiated manually
- [ ] System.Text.Json used with proper naming policy (camelCase)
- [ ] Proper exception handling with specific exception types
- [ ] Async methods return Task&lt;T&gt; explicitly

### Advanced Features (where applicable)

- [ ] Resources registered for appropriate data endpoints
- [ ] Prompts registered for common LLM tasks
- [ ] Appropriate transport configured (stdio or streamable HTTP)
- [ ] Type-safe with SDK interfaces

### SDK v2 (see dotnet_mcp_v2.md)

- [ ] `SessionMode` set explicitly (usually `HttpServerSessionMode.Stateless`)
- [ ] Interactive tools use `InputRequiredException` (MRTR) instead of `ElicitAsync`/`SampleAsync`/`RequestRootsAsync`
- [ ] `[McpHeader]` on parameters that load balancers/gateways must route on
- [ ] Long-running tools evaluated for `ModelContextProtocol.Extensions.Tasks` (`WithTasks` + durable `IMcpTaskStore` in prod)
- [ ] `ClaimsPrincipal` parameter injection / `AddAuthorizationFilters()` for authenticated servers
- [ ] `AllowedHosts` locked down (no `"*"` on local servers); CORS minimal or absent
- [ ] Build clean of `MCP9004`/`MCP9005`/`MCP9006` warnings

### Project Setup

- [ ] .csproj includes ModelContextProtocol.AspNetCore package
- [ ] appsettings.json configured with API base URL
- [ ] Server name follows format: `{Service}.McpServer`
- [ ] Project name follows format: `{service}-mcp-server`

### Code Quality

- [ ] Pagination is properly implemented where applicable
- [ ] Large responses check CHARACTER_LIMIT constant and truncate with clear messages
- [ ] Filtering options are provided for potentially large result sets
- [ ] All network operations handle timeouts and connection errors gracefully
- [ ] Common functionality is extracted into reusable services
- [ ] Return types are consistent across similar operations

### Testing and Build

- [ ] `dotnet build` completes successfully without errors
- [ ] Server runs: `dotnet run`
- [ ] All dependencies resolve correctly
- [ ] Sample tool calls work as expected
