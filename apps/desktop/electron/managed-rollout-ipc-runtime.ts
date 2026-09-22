import type { IpcMain } from 'electron'

import {
  createManagedRolloutIpcHandler,
  type ManagedRolloutIpcAdapter,
  type ManagedRolloutIpcMethod
} from './managed-rollout-ipc'

const UNAVAILABLE_MESSAGE = 'Managed rollout service is unavailable.'

const METHODS: readonly ManagedRolloutIpcMethod[] = [
  'capabilities',
  'inventory',
  'resolveTarget',
  'preflight',
  'start',
  'activeRevision',
  'get',
  'command',
  'history',
  'events'
]

/**
 * The default adapter is deliberately inert. A future trusted provider may be
 * injected at this seam, but renderer IPC never gains an accidental legacy
 * updater fallback while that provider is absent.
 */
export function createUnavailableManagedRolloutAdapter(): ManagedRolloutIpcAdapter {
  const unavailable = async (): Promise<never> => {
    throw new Error(UNAVAILABLE_MESSAGE)
  }

  return {
    capabilities: async () => ({
      protocol: 1,
      available: false,
      reason: 'trusted-assurance-provider-unavailable',
      maxConcurrency: 0,
      maxInstallations: 0
    }),
    activeRevision: async () => null,
    get: unavailable,
    command: unavailable,
    history: unavailable,
    events: unavailable
  }
}

export function registerManagedRolloutIpc(
  ipcMain: Pick<IpcMain, 'handle'>,
  isTrustedSender: (sender: unknown) => boolean,
  adapter: ManagedRolloutIpcAdapter = createUnavailableManagedRolloutAdapter()
): void {
  const handler = createManagedRolloutIpcHandler(adapter, isTrustedSender)
  for (const method of METHODS) {
    ipcMain.handle(`hermes:managed-rollouts:${method}`, (event, payload) =>
      handler({ sender: event.sender }, method, payload)
    )
  }
}

export { UNAVAILABLE_MESSAGE }
