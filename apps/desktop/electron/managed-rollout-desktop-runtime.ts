import path from 'node:path'

import type { App, BrowserWindow, IpcMain } from 'electron'

import { backendScopePrefix } from './connection-registry'
import type { createDesktopConnectionAssembly } from './desktop-connection-assembly'
import { readVerifiedHostKeyFingerprint } from './managed-rollout-host-key'
import { registerManagedRolloutIpc } from './managed-rollout-ipc-runtime'
import { createManagedRolloutMainIntegration } from './managed-rollout-main-integration'
import { createManagedRolloutProvider } from './managed-rollout-provider'
import type { createManagedSshLifecycleRuntime } from './managed-ssh-lifecycle-runtime'
import { managedSshRecoveryScopes } from './managed-ssh-update'

interface DesktopRuntimeDeps {
  app: Pick<App, 'getPath'>
  ipcMain: Pick<IpcMain, 'handle'>
  connections: ReturnType<typeof createDesktopConnectionAssembly>
  lifecycle: ReturnType<typeof createManagedSshLifecycleRuntime>
  getMainWindow: () => BrowserWindow | null
  processOwner: () => boolean
}

/** Register the trusted Main-owned rollout provider only after the app lock. */
export function registerManagedRolloutDesktopRuntime(deps: DesktopRuntimeDeps) {
  if (!deps.processOwner()) {throw new Error('managed-rollout-owner-unavailable')}

  const { connections, lifecycle } = deps
  const service = lifecycle.managedSshUpdateService

  if (
    service.gate !== connections.managedConnectionUpdateGate ||
    service.activeUpdates !== connections.managedConnectionUpdates ||
    service.activeRecoveries !== connections.managedConnectionRecoveries
  ) {
    throw new Error('managed-rollout-shared-update-admission-unavailable')
  }

  const root = path.join(deps.app.getPath('userData'), 'managed-rollouts')
  const sources = () => connections.readDesktopConnectionsRegistry().connections

  const getSource = (connectionId: string) =>
    sources().find((source: { id: string }) => source.id === connectionId) || null

  const assertOwner = () => {
    if (!deps.processOwner()) {throw new Error('managed-rollout-owner-unavailable')}
  }

  const integration = createManagedRolloutMainIntegration({
    nowMono: () => Number(process.hrtime.bigint() / 1_000_000n),
    processOwner: deps.processOwner,
    listSources: sources,
    getSource,
    managedSshConfig: lifecycle.managedSshConfig,
    openTransport: lifecycle.openManagedSshUpdateTransport,
    captureScopes: lifecycle.captureManagedSshScopes,
    readHostKeyFingerprint: readVerifiedHostKeyFingerprint,
    effectiveConfigFingerprint: connections.effectiveSshConfigFingerprint,
    reviewManifestPath: path.join(root, 'review.json'),
    assuranceRoot: path.join(root, 'assurance'),
    journalRoot: path.join(root, 'journal'),
    recoverManagedSsh: record => {
      assertOwner()

      return lifecycle.managedSshUpdateService.recover({ ...record, scopes: [...record.scopes] })
    },
    recoveryScopes: (source, scopes) =>
      managedSshRecoveryScopes(scopes, backendScopePrefix(source.id))
  })

  const provider = createManagedRolloutProvider({
    ...integration.adapters,
    journal: integration.journal,
    managedSshUpdateService: {
      issueLaunchCapability: (...args: Parameters<typeof service.issueLaunchCapability>) => {
        assertOwner()

        return service.issueLaunchCapability(...args)
      },
      request: (...args: Parameters<typeof service.request>) => {
        assertOwner()

        return service.request(...args)
      }
    },
    observe: integration.observe,
    evidence: integration.evidence,
    processGeneration: integration.processGeneration,
    ready: integration.adapters.ready
  })

  registerManagedRolloutIpc(
    deps.ipcMain,
    sender => {
      const window = deps.getMainWindow()

      return Boolean(deps.processOwner() && window && !window.isDestroyed() && sender === window.webContents)
    },
    provider
  )

  return provider
}
