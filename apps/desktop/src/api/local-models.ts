import type { LocalCatalogModel, LocalHardware, LocalModelsStatus, LocalRuntimeJob } from '@/types/rabbit'

import { rabbitApi, profileScoped } from './client'

export interface LocalModelsScope {
  connectionId: string | null
  profile: string
}

// The desktop surface of the managed llama.cpp runtime: status/catalog
// reads, download/install/activate jobs, and server control.

export function getLocalModelsStatus(scope?: LocalModelsScope): Promise<LocalModelsStatus> {
  return rabbitApi<LocalModelsStatus>({
    ...(scope ?? profileScoped()),
    path: '/api/local-models/status'
  })
}

export function getLocalHardware(scope?: LocalModelsScope): Promise<LocalHardware> {
  return rabbitApi<LocalHardware>({
    ...(scope ?? profileScoped()),
    path: '/api/local-models/hardware'
  })
}

export function getLocalCatalog(scope?: LocalModelsScope): Promise<{ models: LocalCatalogModel[] }> {
  return rabbitApi<{ models: LocalCatalogModel[] }>({
    ...(scope ?? profileScoped()),
    path: '/api/local-models/catalog'
  })
}

export function installLocalRuntime(
  backend?: string,
  scope?: LocalModelsScope
): Promise<{ backend: string; job_id: string; tag: string }> {
  return rabbitApi<{ backend: string; job_id: string; tag: string }>({
    ...(scope ?? profileScoped()),
    body: { backend: backend ?? null },
    method: 'POST',
    path: '/api/local-models/runtime/install'
  })
}

export interface QuickstartResponse {
  display_name: string
  download_bytes: number
  job_id: string
  model_id: string
  needs_download: boolean
  needs_runtime: boolean
}

export function quickstartLocalModels(modelId?: string, scope?: LocalModelsScope): Promise<QuickstartResponse> {
  return rabbitApi<QuickstartResponse>({
    ...(scope ?? profileScoped()),
    body: { model_id: modelId ?? null },
    method: 'POST',
    path: '/api/local-models/quickstart'
  })
}

export function downloadLocalModel(
  modelId: string,
  scope?: LocalModelsScope
): Promise<{ already_downloaded?: boolean; job_id: null | string }> {
  return rabbitApi<{ already_downloaded?: boolean; job_id: null | string }>({
    ...(scope ?? profileScoped()),
    body: { model_id: modelId },
    method: 'POST',
    path: '/api/local-models/download'
  })
}

export function deleteLocalModel(modelId: string, scope?: LocalModelsScope): Promise<{ ok: boolean }> {
  return rabbitApi<{ ok: boolean }>({
    ...(scope ?? profileScoped()),
    method: 'DELETE',
    path: `/api/local-models/models/${encodeURIComponent(modelId)}`
  })
}

export function getLocalRuntimeJob(jobId: string, scope?: LocalModelsScope): Promise<LocalRuntimeJob> {
  return rabbitApi<LocalRuntimeJob>({
    ...(scope ?? profileScoped()),
    path: `/api/local-models/jobs/${encodeURIComponent(jobId)}`
  })
}

export function getLocalModelsJobs(scope?: LocalModelsScope): Promise<{ jobs: LocalRuntimeJob[] }> {
  return rabbitApi<{ jobs: LocalRuntimeJob[] }>({
    ...(scope ?? profileScoped()),
    path: '/api/local-models/jobs'
  })
}

// Pause/resume a download-phase job (catalog model, quickstart, runtime
// install/update, HF-browsed). The backend answers {ok, paused} /
// {ok, resumed} — a false flag (no live download handle, e.g. a
// quickstart engine leg) is reported to the caller, not treated as success.
export function pauseLocalDownload(jobId: string, scope?: LocalModelsScope): Promise<{ ok: boolean; paused: boolean }> {
  return rabbitApi<{ ok: boolean; paused: boolean }>({
    ...(scope ?? profileScoped()),
    body: { job_id: jobId },
    method: 'POST',
    path: '/api/local-models/download/pause'
  })
}

export function resumeLocalDownload(
  jobId: string,
  scope?: LocalModelsScope
): Promise<{ ok: boolean; resumed: boolean }> {
  return rabbitApi<{ ok: boolean; resumed: boolean }>({
    ...(scope ?? profileScoped()),
    body: { job_id: jobId },
    method: 'POST',
    path: '/api/local-models/download/resume'
  })
}

export function activateLocalModel(modelId: string, scope?: LocalModelsScope): Promise<{ job_id: string }> {
  return rabbitApi<{ job_id: string }>({
    ...(scope ?? profileScoped()),
    body: { model_id: modelId },
    method: 'POST',
    path: '/api/local-models/activate'
  })
}

export function ejectLocalModel(modelId: string, scope?: LocalModelsScope): Promise<{ ok: boolean }> {
  return rabbitApi<{ ok: boolean }>({
    ...(scope ?? profileScoped()),
    body: { model_id: modelId },
    method: 'POST',
    path: '/api/local-models/eject'
  })
}

export function setLocalServer(action: 'start' | 'stop', scope?: LocalModelsScope): Promise<{ ok: boolean }> {
  return rabbitApi<{ ok: boolean }>({
    ...(scope ?? profileScoped()),
    body: { action },
    method: 'POST',
    path: '/api/local-models/server'
  })
}

// ── Hugging Face browser + sideload ─────────────────────────────

export interface HFSearchHit {
  repo: string
  downloads: number
  likes: number
  updated: string
  gated: boolean
}

export interface HFFileGroup {
  label: string
  paths: string[]
  total_bytes: number
  fit: 'fits-gpu' | 'needs-ram' | 'too-big' | 'unknown'
}

export function searchHFModels(
  q: string,
  limit: number = 20,
  scope?: LocalModelsScope
): Promise<{ hits: HFSearchHit[] }> {
  return rabbitApi<{ hits: HFSearchHit[] }>({
    ...(scope ?? profileScoped()),
    path: `/api/local-models/search?q=${encodeURIComponent(q)}&limit=${limit}`
  })
}

export function listHFRepoFiles(repo: string, scope?: LocalModelsScope): Promise<{ files: HFFileGroup[] }> {
  return rabbitApi<{ files: HFFileGroup[] }>({
    ...(scope ?? profileScoped()),
    path: `/api/local-models/search/files?repo=${encodeURIComponent(repo)}`
  })
}

export function downloadBrowsedModel(
  repo: string,
  paths: string[],
  scope?: LocalModelsScope
): Promise<{ already_downloaded?: boolean; job_id: null | string; model_id: string }> {
  return rabbitApi<{ already_downloaded?: boolean; job_id: null | string; model_id: string }>({
    ...(scope ?? profileScoped()),
    body: { paths, repo },
    method: 'POST',
    path: '/api/local-models/download-browsed'
  })
}

export function sideloadLocalModel(
  path: string,
  scope?: LocalModelsScope
): Promise<{ already_present?: boolean; model_id: string; ok: boolean }> {
  return rabbitApi<{ already_present?: boolean; model_id: string; ok: boolean }>({
    ...(scope ?? profileScoped()),
    body: { path },
    method: 'POST',
    path: '/api/local-models/sideload'
  })
}
