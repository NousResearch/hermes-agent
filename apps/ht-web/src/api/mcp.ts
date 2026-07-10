// Typed wrappers over the MCP admin REST API. Shapes mirror web/src/lib/api.ts
// (the source of truth); only the subset the McpPage consumes is declared.
import { apiGet, apiPost, apiPut, apiDelete } from "./client";

export interface McpServer {
  name: string;
  transport: "http" | "stdio" | "unknown";
  url: string | null;
  command: string | null;
  args: string[];
  env: Record<string, string>;
  auth: string | null;
  enabled: boolean;
  tools: string[] | null;
}

export interface McpServerCreate {
  name: string;
  url?: string;
  command?: string;
  args?: string[];
  env?: Record<string, string>;
  auth?: string;
}

export interface McpTestResult {
  ok: boolean;
  error?: string;
  tools: Array<{ name: string; description: string }>;
}

export interface McpCatalogRequiredEnv {
  name: string;
  prompt: string;
  required: boolean;
}

export interface McpCatalogEntry {
  name: string;
  description: string;
  source: string;
  transport: "http" | "stdio";
  auth_type: "api_key" | "oauth" | "none";
  required_env: McpCatalogRequiredEnv[];
  command: string | null;
  args: string[];
  url: string | null;
  needs_install: boolean;
  installed: boolean;
  enabled: boolean;
}

export interface McpCatalogDiagnostic {
  name: string;
  kind: string;
  message: string;
}

export interface McpCatalogResponse {
  entries: McpCatalogEntry[];
  diagnostics: McpCatalogDiagnostic[];
}

export interface McpInstallResult {
  ok: boolean;
  name: string;
  background: boolean;
  action?: string;
}

/** GET /api/mcp/servers */
export const getMcpServers = () => apiGet<{ servers: McpServer[] }>("/api/mcp/servers");

/** POST /api/mcp/servers */
export const addMcpServer = (body: McpServerCreate) => apiPost<McpServer>("/api/mcp/servers", body);

/** DELETE /api/mcp/servers/{name} */
export const removeMcpServer = (name: string) =>
  apiDelete<{ ok: boolean }>(`/api/mcp/servers/${encodeURIComponent(name)}`);

/** POST /api/mcp/servers/{name}/test */
export const testMcpServer = (name: string) =>
  apiPost<McpTestResult>(`/api/mcp/servers/${encodeURIComponent(name)}/test`);

/** PUT /api/mcp/servers/{name}/enabled */
export const setMcpServerEnabled = (name: string, enabled: boolean) =>
  apiPut<{ ok: boolean; name: string; enabled: boolean }>(
    `/api/mcp/servers/${encodeURIComponent(name)}/enabled`,
    { enabled },
  );

/** GET /api/mcp/catalog */
export const getMcpCatalog = () => apiGet<McpCatalogResponse>("/api/mcp/catalog");

/** POST /api/mcp/catalog/install */
export const installMcpCatalogEntry = (
  name: string,
  env: Record<string, string> = {},
  enable = true,
) => apiPost<McpInstallResult>("/api/mcp/catalog/install", { name, env, enable });
