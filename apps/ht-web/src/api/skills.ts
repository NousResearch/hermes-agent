// Typed wrappers for the Skills, Toolsets, and Skills-hub REST endpoints.
// Shapes mirror web/src/lib/api.ts (the source of truth); only the subset the
// ported Skills page consumes is declared here.
import { apiGet, apiPost, apiPut } from "./client";

// ── Installed skills ────────────────────────────────────────────────
export interface SkillInfo {
  name: string;
  description: string;
  category: string;
  enabled: boolean;
}

export interface SkillContent {
  name: string;
  content: string;
  path: string;
}

// ── Installed toolsets ──────────────────────────────────────────────
export interface ToolsetInfo {
  name: string;
  label: string;
  description: string;
  enabled: boolean;
  configured: boolean;
  tools: string[];
}

// ── Skills hub (search / install) ───────────────────────────────────
export interface SkillHubResult {
  name: string;
  description: string;
  source: string;
  identifier: string;
  trust_level: string;
  repo: string | null;
  tags: string[];
}

/** Lock-entry summary for an already-installed hub skill (keyed by identifier). */
export interface SkillHubInstalledEntry {
  name: string | null;
  trust_level: string | null;
  scan_verdict: string | null;
}

export interface SkillHubSearchResponse {
  results: SkillHubResult[];
  /** source_id -> number of results returned by that source. */
  source_counts: Record<string, number>;
  /** source ids that didn't return within the parallel-search timeout. */
  timed_out: string[];
  /** identifier -> installed lock entry (for "already installed" badges). */
  installed: Record<string, SkillHubInstalledEntry>;
}

export interface SkillHubSource {
  id: string;
  label: string;
  rate_limited?: boolean;
  available?: boolean;
}

export interface SkillHubSourcesResponse {
  sources: SkillHubSource[];
  index_available: boolean;
  featured: SkillHubResult[];
  installed: Record<string, SkillHubInstalledEntry>;
}

/** Async-action envelope returned by the hub install/uninstall/update POSTs. */
export interface ActionResponse {
  name: string;
  ok: boolean;
  pid: number | null;
  error?: string;
  message?: string;
}

// ── Endpoint wrappers ───────────────────────────────────────────────
export const getSkills = () => apiGet<SkillInfo[]>("/api/skills");

export const toggleSkill = (name: string, enabled: boolean) =>
  apiPut<{ ok: boolean }>("/api/skills/toggle", { name, enabled });

export const getSkillContent = (name: string) =>
  apiGet<SkillContent>(`/api/skills/content?name=${encodeURIComponent(name)}`);

export const getToolsets = () => apiGet<ToolsetInfo[]>("/api/tools/toolsets");

export const toggleToolset = (name: string, enabled: boolean) =>
  apiPut<{ ok: boolean; name: string; enabled: boolean }>(
    `/api/tools/toolsets/${encodeURIComponent(name)}`,
    { enabled },
  );

export const searchSkillsHub = (q: string, source = "all", limit = 20) =>
  apiGet<SkillHubSearchResponse>(
    `/api/skills/hub/search?q=${encodeURIComponent(q)}&source=${encodeURIComponent(source)}&limit=${limit}`,
  );

export const getSkillHubSources = () =>
  apiGet<SkillHubSourcesResponse>("/api/skills/hub/sources");

export const installSkillFromHub = (identifier: string) =>
  apiPost<ActionResponse>("/api/skills/hub/install", { identifier });

export const uninstallSkillFromHub = (name: string) =>
  apiPost<ActionResponse>("/api/skills/hub/uninstall", { name });

export const updateSkillsFromHub = () =>
  apiPost<ActionResponse>("/api/skills/hub/update", {});
