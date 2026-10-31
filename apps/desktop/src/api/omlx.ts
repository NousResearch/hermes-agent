import { hermesApi } from './client'
import type { LocalModelsScope } from './local-models'

export interface OmlxStatus {
  supported: boolean
  installed: boolean
  running: boolean
  selected: boolean
  model_dir: string
  base_url: string
  version: string | null
  models: { id: string; active: boolean }[]
  error?: string | null
}

export function getOmlxStatus(scope: LocalModelsScope): Promise<OmlxStatus> {
  return hermesApi<OmlxStatus>({ ...scope, path: '/api/local-models/omlx/status' })
}

export function selectLocalEngine(engine: 'llamacpp' | 'omlx', scope: LocalModelsScope): Promise<{ ok: boolean }> {
  return hermesApi<{ ok: boolean }>({
    ...scope,
    body: { engine },
    method: 'POST',
    path: '/api/local-models/omlx/select'
  })
}

export function installOmlx(scope: LocalModelsScope): Promise<{ job_id: string }> {
  return hermesApi<{ job_id: string }>({ ...scope, body: {}, method: 'POST', path: '/api/local-models/omlx/install' })
}

export function setOmlxServer(action: 'start' | 'stop', scope: LocalModelsScope): Promise<{ ok: boolean }> {
  return hermesApi<{ ok: boolean }>({
    ...scope,
    body: { action },
    method: 'POST',
    path: '/api/local-models/omlx/server'
  })
}

export function activateOmlxModel(modelId: string, scope: LocalModelsScope): Promise<{ ok: boolean }> {
  return hermesApi<{ ok: boolean }>({
    ...scope,
    body: { model_id: modelId },
    method: 'POST',
    path: '/api/local-models/omlx/activate'
  })
}

export function downloadOmlxModel(repo: string, scope: LocalModelsScope): Promise<{ job_id: string }> {
  return hermesApi<{ job_id: string }>({
    ...scope,
    body: { repo },
    method: 'POST',
    path: '/api/local-models/omlx/download'
  })
}
