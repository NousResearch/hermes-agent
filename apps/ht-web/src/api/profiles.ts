// Typed wrappers over the profiles admin REST API. Shapes mirror
// web/src/lib/api.ts (the source of truth); only the subset ProfilesPage
// consumes is declared here. Profile endpoints are machine-global (not
// profile-scoped), so no ?profile= query param is appended.
import { apiGet, apiPost, apiPut, apiDelete, apiFetch } from "./client";

export interface ProfileInfo {
  name: string;
  path: string;
  is_default: boolean;
  model: string | null;
  provider: string | null;
  has_env: boolean;
  skill_count: number;
  gateway_running: boolean;
  description: string;
  description_auto: boolean;
  distribution_name: string | null;
  distribution_version: string | null;
  distribution_source: string | null;
  has_alias: boolean;
}

export interface ActiveProfileInfo {
  active: string;
  current: string;
}

/** Subset of the create-profile body ProfilesPage sends (name + optional
 * description). The full endpoint accepts many clone/model/mcp options; the
 * MVP form only surfaces these two. */
export interface ProfileCreate {
  name: string;
  description?: string;
}

export interface ProfileCreateResult {
  ok: boolean;
  name: string;
  path: string;
}

export interface ProfileSoul {
  content: string;
  exists: boolean;
}

/** GET /api/profiles → { profiles } */
export const getProfiles = () => apiGet<{ profiles: ProfileInfo[] }>("/api/profiles");

/** GET /api/profiles/active */
export const getActiveProfile = () => apiGet<ActiveProfileInfo>("/api/profiles/active");

/** POST /api/profiles/active  body { name } */
export const setActiveProfile = (name: string) =>
  apiPost<{ ok: boolean; active: string }>("/api/profiles/active", { name });

/** POST /api/profiles */
export const createProfile = (body: ProfileCreate) =>
  apiPost<ProfileCreateResult>("/api/profiles", body);

/** PATCH /api/profiles/{name}  body { new_name } */
export const renameProfile = (name: string, newName: string) =>
  apiFetch<{ ok: boolean; name: string; path: string }>(
    `/api/profiles/${encodeURIComponent(name)}`,
    { method: "PATCH", body: JSON.stringify({ new_name: newName }) },
  );

/** DELETE /api/profiles/{name} */
export const deleteProfile = (name: string) =>
  apiDelete<{ ok: boolean }>(`/api/profiles/${encodeURIComponent(name)}`);

/** GET /api/profiles/{name}/soul → { content, exists } */
export const getProfileSoul = (name: string) =>
  apiGet<ProfileSoul>(`/api/profiles/${encodeURIComponent(name)}/soul`);

/** PUT /api/profiles/{name}/soul  body { content } */
export const updateProfileSoul = (name: string, content: string) =>
  apiPut<{ ok: boolean }>(`/api/profiles/${encodeURIComponent(name)}/soul`, { content });
