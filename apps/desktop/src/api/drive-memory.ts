import { capabilityScoped, type ProfileScope } from './client'

export interface DriveMemoryStatus {
  connected: boolean
  folders: { id: string; name: string }[]
  indexed: number
  vault: { exists: boolean; path: string }
}

export interface DriveSyncStep extends DriveMemoryStatus {
  failed: number
  processed: number
  remaining: number
}

const call = <T>(profile: ProfileScope | undefined, path: string, body?: unknown) =>
  window.hermesDesktop.api<T>({
    ...capabilityScoped(profile),
    body,
    method: body === undefined ? undefined : 'POST',
    path,
    timeoutMs: 180_000
  })

export const getDriveMemory = (profile?: ProfileScope) => call<DriveMemoryStatus>(profile, '/api/drive-memory/status')

export const addDriveFolder = (link: string, profile?: ProfileScope) =>
  call<DriveMemoryStatus>(profile, '/api/drive-memory/folders', { link })

export const removeDriveFolder = (id: string, profile?: ProfileScope) =>
  call<DriveMemoryStatus>(profile, '/api/drive-memory/folders/remove', { id })

export const syncDriveMemory = (profile?: ProfileScope) => call<DriveSyncStep>(profile, '/api/drive-memory/sync', {})
