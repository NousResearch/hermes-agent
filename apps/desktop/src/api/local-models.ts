import type {
  LocalCatalogModel,
  LocalHardware,
  LocalModelsStatus,
  LocalRuntimeJob
} from '@/types/hermes'

import { hermesApi, profileScoped } from './client'

// The desktop surface of the managed llama.cpp runtime: status/catalog
// reads, download/install/activate jobs, and server control.

export function getLocalModelsStatus(): Promise<LocalModelsStatus> {
  return hermesApi<LocalModelsStatus>({
    ...profileScoped(),
    path: '/api/local-models/status'
  })
}

export function getLocalHardware(): Promise<LocalHardware> {
  return hermesApi<LocalHardware>({
    ...profileScoped(),
    path: '/api/local-models/hardware'
  })
}

export function getLocalCatalog(): Promise<{ models: LocalCatalogModel[] }> {
  return hermesApi<{ models: LocalCatalogModel[] }>({
    ...profileScoped(),
    path: '/api/local-models/catalog'
  })
}

export function installLocalRuntime(backend?: string): Promise<{ backend: string; job_id: string; tag: string }> {
  return hermesApi<{ backend: string; job_id: string; tag: string }>({
    ...profileScoped(),
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

export function quickstartLocalModels(modelId?: string): Promise<QuickstartResponse> {
  return hermesApi<QuickstartResponse>({
    ...profileScoped(),
    body: { model_id: modelId ?? null },
    method: 'POST',
    path: '/api/local-models/quickstart'
  })
}

export function downloadLocalModel(modelId: string): Promise<{ already_downloaded?: boolean; job_id: null | string }> {
  return hermesApi<{ already_downloaded?: boolean; job_id: null | string }>({
    ...profileScoped(),
    body: { model_id: modelId },
    method: 'POST',
    path: '/api/local-models/download'
  })
}

export function deleteLocalModel(modelId: string): Promise<{ ok: boolean }> {
  return hermesApi<{ ok: boolean }>({
    ...profileScoped(),
    method: 'DELETE',
    path: `/api/local-models/models/${encodeURIComponent(modelId)}`
  })
}

export function getLocalRuntimeJob(jobId: string): Promise<LocalRuntimeJob> {
  return hermesApi<LocalRuntimeJob>({
    ...profileScoped(),
    path: `/api/local-models/jobs/${encodeURIComponent(jobId)}`
  })
}

export function getLocalModelsJobs(): Promise<{ jobs: LocalRuntimeJob[] }> {
  return hermesApi<{ jobs: LocalRuntimeJob[] }>({
    ...profileScoped(),
    path: '/api/local-models/jobs'
  })
}

export function activateLocalModel(modelId: string): Promise<{ job_id: string }> {
  return hermesApi<{ job_id: string }>({
    ...profileScoped(),
    body: { model_id: modelId },
    method: 'POST',
    path: '/api/local-models/activate'
  })
}

export function ejectLocalModel(modelId: string): Promise<{ ok: boolean }> {
  return hermesApi<{ ok: boolean }>({
    ...profileScoped(),
    body: { model_id: modelId },
    method: 'POST',
    path: '/api/local-models/eject'
  })
}

export function setLocalServer(action: 'start' | 'stop'): Promise<{ ok: boolean }> {
  return hermesApi<{ ok: boolean }>({
    ...profileScoped(),
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

export function searchHFModels(q: string, limit = 20): Promise<{ hits: HFSearchHit[] }> {
  return hermesApi<{ hits: HFSearchHit[] }>({
    ...profileScoped(),
    path: `/api/local-models/search?q=${encodeURIComponent(q)}&limit=${limit}`
  })
}

export function listHFRepoFiles(repo: string): Promise<{ files: HFFileGroup[] }> {
  return hermesApi<{ files: HFFileGroup[] }>({
    ...profileScoped(),
    path: `/api/local-models/search/files?repo=${encodeURIComponent(repo)}`
  })
}

export function downloadBrowsedModel(repo: string, paths: string[]): Promise<{ already_downloaded?: boolean; job_id: null | string; model_id: string }> {
  return hermesApi<{ already_downloaded?: boolean; job_id: null | string; model_id: string }>({
    ...profileScoped(),
    body: { paths, repo },
    method: 'POST',
    path: '/api/local-models/download-browsed'
  })
}

export function sideloadLocalModel(path: string): Promise<{ already_present?: boolean; model_id: string; ok: boolean }> {
  return hermesApi<{ already_present?: boolean; model_id: string; ok: boolean }>({
    ...profileScoped(),
    body: { path },
    method: 'POST',
    path: '/api/local-models/sideload'
  })
}
