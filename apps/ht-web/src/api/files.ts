// Typed wrappers for the managed-files API (/api/files/*). Shapes mirror
// web/src/lib/api.ts (the source of truth); only the subset the Files browser
// consumes is declared here.
import { apiGet, apiPost, apiDelete, authedFetch, ApiError } from "./client";

export interface ManagedFileEntry {
  name: string;
  path: string;
  is_directory: boolean;
  size: number | null;
  mtime: number;
  mime_type: string | null;
}

export interface ManagedFilesResponse {
  root: string | null;
  path: string;
  parent: string | null;
  locked_root: string | null;
  can_change_path: boolean;
  entries: ManagedFileEntry[];
}

export interface ManagedFileReadResponse {
  name: string;
  path: string;
  size: number;
  mime_type: string;
  data_url: string;
  root: string | null;
  locked_root: string | null;
  can_change_path: boolean;
}

export interface ManagedFileWriteResponse {
  ok: boolean;
  path: string;
  entry: ManagedFileEntry;
  root: string | null;
  locked_root: string | null;
  can_change_path: boolean;
}

export function listFiles(path?: string): Promise<ManagedFilesResponse> {
  const query = path ? `?path=${encodeURIComponent(path)}` : "";
  return apiGet<ManagedFilesResponse>(`/api/files${query}`);
}

export const readFile = (path: string) =>
  apiGet<ManagedFileReadResponse>(`/api/files/read?path=${encodeURIComponent(path)}`);

export const createDirectory = (path: string) =>
  apiPost<ManagedFileWriteResponse>("/api/files/mkdir", { path });

export const deleteFile = (path: string, recursive = false) =>
  apiDelete<{ ok: boolean; path: string }>("/api/files", { path, recursive });

/**
 * Upload a file via streamed multipart/form-data. Uses authedFetch (raw
 * Response, no throw / no 401 redirect) because the body is FormData: the
 * browser must set the multipart Content-Type with its own boundary, so we
 * never set it manually. We inspect res.ok and surface failures as ApiError to
 * match the rest of the client.
 */
export async function uploadFile(
  path: string,
  file: File,
  overwrite = true,
): Promise<ManagedFileWriteResponse> {
  const form = new FormData();
  form.append("path", path);
  form.append("overwrite", String(overwrite));
  form.append("file", file, file.name);

  const res = await authedFetch("/api/files/upload-stream", {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    let body: unknown;
    try {
      body = await res.clone().json();
      const err = (body as { error?: string; message?: string }) ?? {};
      message = err.message || err.error || message;
    } catch {
      /* non-JSON error body — keep the status line */
    }
    throw new ApiError(message, res.status, body);
  }

  return (await res.json()) as ManagedFileWriteResponse;
}
