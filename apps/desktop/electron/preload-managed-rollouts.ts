import type { IpcRenderer } from 'electron'

export function createManagedRolloutsBridge(ipcRenderer: Pick<IpcRenderer, 'invoke'>) {
  return {
    capabilities: () => ipcRenderer.invoke('hermes:managed-rollouts:capabilities'),
    inventory: () => ipcRenderer.invoke('hermes:managed-rollouts:inventory'),
    resolveTarget: (payload: unknown) => ipcRenderer.invoke('hermes:managed-rollouts:resolveTarget', payload),
    preflight: (draft: unknown) => ipcRenderer.invoke('hermes:managed-rollouts:preflight', { draft }),
    start: (payload: unknown) => ipcRenderer.invoke('hermes:managed-rollouts:start', payload),
    activeRevision: () => ipcRenderer.invoke('hermes:managed-rollouts:activeRevision'),
    read: (sinceRevision: number | null) => ipcRenderer.invoke('hermes:managed-rollouts:read', { sinceRevision }),
    get: (id: string) => ipcRenderer.invoke('hermes:managed-rollouts:get', { id }),
    command: (payload: unknown) => ipcRenderer.invoke('hermes:managed-rollouts:command', payload),
    history: (page: unknown) => ipcRenderer.invoke('hermes:managed-rollouts:history', page),
    events: (page: unknown) => ipcRenderer.invoke('hermes:managed-rollouts:events', page)
  }
}
