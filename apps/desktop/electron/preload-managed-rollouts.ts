import type { IpcRenderer } from 'electron'

async function invokeManagedRollout<T>(
  ipcRenderer: Pick<IpcRenderer, 'invoke'>,
  channel: string,
  payload?: unknown
): Promise<T> {
  const response: unknown = await ipcRenderer.invoke(channel, payload)

  if (!response || typeof response !== 'object' || Array.isArray(response)) {
    throw new Error('Managed rollout IPC returned an invalid response.')
  }

  const result = response as Record<string, unknown>

  if (result.ok === true && Object.prototype.hasOwnProperty.call(result, 'value')) {
    return result.value as T
  }

  if (result.ok === false && typeof result.code === 'string' && typeof result.message === 'string') {
    const error = new Error(result.message) as Error & { code: string }
    error.code = result.code
    throw error
  }

  throw new Error('Managed rollout IPC returned an invalid response.')
}

export function createManagedRolloutsBridge(ipcRenderer: Pick<IpcRenderer, 'invoke'>) {
  return {
    capabilities: () => invokeManagedRollout(ipcRenderer, 'hermes:managed-rollouts:capabilities'),
    inventory: () => invokeManagedRollout(ipcRenderer, 'hermes:managed-rollouts:inventory'),
    resolveTarget: (payload: unknown) => invokeManagedRollout(ipcRenderer, 'hermes:managed-rollouts:resolveTarget', payload),
    preflight: (draft: unknown) => invokeManagedRollout(ipcRenderer, 'hermes:managed-rollouts:preflight', { draft }),
    start: (payload: unknown) => invokeManagedRollout(ipcRenderer, 'hermes:managed-rollouts:start', payload),
    activeRevision: () => invokeManagedRollout(ipcRenderer, 'hermes:managed-rollouts:activeRevision'),
    read: (sinceRevision: number | null) => invokeManagedRollout(ipcRenderer, 'hermes:managed-rollouts:read', { sinceRevision }),
    get: (id: string) => invokeManagedRollout(ipcRenderer, 'hermes:managed-rollouts:get', { id }),
    command: (payload: unknown) => invokeManagedRollout(ipcRenderer, 'hermes:managed-rollouts:command', payload),
    history: (page: unknown) => invokeManagedRollout(ipcRenderer, 'hermes:managed-rollouts:history', page),
    events: (page: unknown) => invokeManagedRollout(ipcRenderer, 'hermes:managed-rollouts:events', page)
  }
}
