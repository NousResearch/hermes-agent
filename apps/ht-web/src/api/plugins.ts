// Typed wrappers for the dashboard/agent plugin REST endpoints. Shapes mirror
// web/src/lib/api.ts (the source of truth); only the subset the ported Plugins
// page consumes is declared here.
import { apiGet, apiPost, apiDelete } from "./client";

// ── Dashboard plugin manifest ───────────────────────────────────────
export interface PluginManifestResponse {
  name: string;
  label: string;
  description: string;
  icon: string;
  version: string;
  tab: {
    path: string;
    position?: string;
    override?: string;
    hidden?: boolean;
  };
  slots?: string[];
  entry: string;
  css?: string | null;
  has_api: boolean;
  source: string;
}

// ── Hub view of installed agent plugins ─────────────────────────────
export interface HubAgentPluginRow {
  name: string;
  version: string;
  description: string;
  source: string;
  runtime_status: "disabled" | "enabled" | "inactive";
  has_dashboard_manifest: boolean;
  dashboard_manifest: PluginManifestResponse | null;
  path: string;
  can_remove: boolean;
  can_update_git: boolean;
  auth_required: boolean;
  auth_command: string;
  user_hidden: boolean;
}

export interface PluginsHubProviders {
  memory_provider: string;
  memory_options: Array<{ name: string; description?: string }>;
  context_engine: string;
  context_options: Array<{ name: string; description: string }>;
}

export interface PluginsHubResponse {
  plugins: HubAgentPluginRow[];
  orphan_dashboard_plugins: PluginManifestResponse[];
  providers: PluginsHubProviders;
}

// ── Install / update ────────────────────────────────────────────────
export interface AgentPluginInstallRequest {
  identifier: string;
  force?: boolean;
  enable?: boolean;
}

export interface AgentPluginInstallResponse {
  ok: boolean;
  plugin_name?: string;
  warnings?: string[];
  missing_env?: string[];
  after_install_path?: string | null;
  enabled?: boolean;
  /** Present when the install runs as a background action to be polled. */
  name?: string;
  pid?: number | null;
  error?: string;
}

export interface AgentPluginUpdateResponse {
  ok: boolean;
  name?: string;
  output?: string;
  unchanged?: boolean;
  error?: string;
}

// ── Endpoint wrappers ───────────────────────────────────────────────
export const getPlugins = () =>
  apiGet<PluginManifestResponse[]>("/api/dashboard/plugins");

export const getPluginsHub = () =>
  apiGet<PluginsHubResponse>("/api/dashboard/plugins/hub");

export const rescanPlugins = () =>
  apiPost<{ ok: boolean; count: number }>("/api/dashboard/plugins/rescan");

export const installAgentPlugin = (body: AgentPluginInstallRequest) =>
  apiPost<AgentPluginInstallResponse>("/api/dashboard/agent-plugins/install", body);

/** URL-encode a plugin key while preserving `/` segment separators. */
function pluginPath(name: string): string {
  return name.split("/").map(encodeURIComponent).join("/");
}

export const enableAgentPlugin = (name: string) =>
  apiPost<{ ok: boolean; name: string; unchanged?: boolean }>(
    `/api/dashboard/agent-plugins/${pluginPath(name)}/enable`,
  );

export const disableAgentPlugin = (name: string) =>
  apiPost<{ ok: boolean; name: string; unchanged?: boolean }>(
    `/api/dashboard/agent-plugins/${pluginPath(name)}/disable`,
  );

export const updateAgentPlugin = (name: string) =>
  apiPost<AgentPluginUpdateResponse>(
    `/api/dashboard/agent-plugins/${pluginPath(name)}/update`,
  );

export const removeAgentPlugin = (name: string) =>
  apiDelete<{ ok: boolean; name: string }>(
    `/api/dashboard/agent-plugins/${pluginPath(name)}`,
  );
