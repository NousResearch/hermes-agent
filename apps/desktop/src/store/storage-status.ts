import { atom } from 'nanostores'

export type StorageStatus = 'degraded' | 'ok'

// The backend latches this state for its lifetime after confirmed corruption.
// Keep the renderer's copy outside a particular sidebar/chat surface so the
// warning remains visible while users navigate the app.
export const $storageStatus = atom<StorageStatus>('ok')

export function setStorageStatus(status: StorageStatus | undefined): void {
  $storageStatus.set(status === 'degraded' ? 'degraded' : 'ok')
}
