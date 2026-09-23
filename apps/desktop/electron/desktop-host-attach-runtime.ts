import fs from 'node:fs'
import path from 'node:path'

import { resolveServedDashboardToken } from './dashboard-token'
import { probeGatewayWebSocket } from './gateway-ws-probe'
import {
  type AttachedBackend,
  attachOrReserveSpawn,
  spawnLedgerPath,
  type SpawnReservation
} from './host-backend-attach'

interface DesktopHostAttachRuntimeDeps {
  HERMES_HOME: string
  ISOLATED_BACKEND: boolean
  rememberLog: (message: string) => void
  waitForHermes: (...args: any[]) => Promise<any>
  invalidatePrimaryConnection: () => void
  scheduleUnexpectedPrimaryRecovery: (cause: any) => void
}

export function createDesktopHostAttachRuntime(deps: DesktopHostAttachRuntimeDeps) {
  const {
    HERMES_HOME,
    ISOLATED_BACKEND,
    rememberLog,
    waitForHermes,
    invalidatePrimaryConnection,
    scheduleUnexpectedPrimaryRecovery
  } = deps

  const ATTACHED_LIVENESS_POLL_MS = 15_000
  let attachedBackendMonitor: NodeJS.Timeout | null = null
  let hostSpawnReservation: SpawnReservation | null = null

  function stopAttachedBackendMonitor() {
    if (attachedBackendMonitor) {
      clearInterval(attachedBackendMonitor)
      attachedBackendMonitor = null
    }
  }

  /**
   * An attached backend has no child process, so `child.exit` can never drive
   * recovery. Poll its readiness instead; a backend that dies under us
   * invalidates the connection and hands the respawn to the same supervisor path
   * a dead child would (which re-runs discovery and spawns, since the host now
   * has no backend).
   */
  function startAttachedBackendMonitor(attached: AttachedBackend) {
    stopAttachedBackendMonitor()

    attachedBackendMonitor = setInterval(() => {
      void waitForHermes(attached.baseUrl, attached.token, undefined, 'token', {}).catch(() => {
        stopAttachedBackendMonitor()
        rememberLog(`[attach] attached backend on ${attached.baseUrl} (pid ${attached.pid}) is gone; recovering`)
        invalidatePrimaryConnection()
        scheduleUnexpectedPrimaryRecovery({ error: 'The Hermes backend this app attached to exited.', ready: true })
      })
    }, ATTACHED_LIVENESS_POLL_MS)

    attachedBackendMonitor.unref?.()
  }

  /** Discover and attach to the host's running backend; null means "spawn one". */
  function attachToRunningHostBackend(): Promise<AttachedBackend | null> {
    const options = { isolated: ISOLATED_BACKEND, ledgerPath: spawnLedgerPath(HERMES_HOME, path.join) }

    return attachOrReserveSpawn(options, hostBackendAttachDeps(), hostSpawnGateDeps())
      .then(outcome => {
        if ('attached' in outcome) {
          releaseHostSpawnReservation()

          return outcome.attached
        }

        hostSpawnReservation = outcome.reservation

        return null
      })
      .catch(error => {
        // Discovery must never be able to block boot: fall through to spawning.
        rememberLog(`[attach] host backend discovery failed (${error.message}); spawning our own`)

        return null
      })
  }

  function hostBackendAttachDeps() {
    return {
      log: rememberLog,
      readLedger: (target: string) => {
        try {
          return fs.readFileSync(target, 'utf8')
        } catch {
          return null
        }
      },
      probeWebSocket: (wsUrl: string) => probeGatewayWebSocket(wsUrl, { WebSocketImpl: globalThis.WebSocket }),
      resolveServedToken: (baseUrl: string) => resolveServedDashboardToken(baseUrl, ''),
      waitForReady: (baseUrl: string, token: string) => waitForHermes(baseUrl, token, undefined, 'token', {})
    }
  }

  function hostSpawnGatePath() {
    return path.join(HERMES_HOME, 'desktop-backend-spawn.json')
  }

  function hostSpawnGateDeps() {
    return {
      now: () => Date.now(),
      read: () => {
        try {
          const record = JSON.parse(fs.readFileSync(hostSpawnGatePath(), 'utf8'))
          const owner = Number(record?.pid)

          if (!Number.isInteger(owner) || owner <= 0) {
            return null
          }

          // A gate whose owner is gone is no gate at all.
          try {
            process.kill(owner, 0)
          } catch {
            return null
          }

          return { ownerAlive: true, startedAt: Number(record?.startedAt) || 0 }
        } catch {
          return null
        }
      },
      take: () => {
        const gatePath = hostSpawnGatePath()

        try {
          fs.writeFileSync(gatePath, JSON.stringify({ pid: process.pid, startedAt: Date.now() }), { mode: 0o600 })
        } catch {
          // A gate we cannot write is a race we cannot win; spawning anyway is
          // exactly today's behaviour, so never fail boot over it.
        }

        return () => {
          try {
            fs.unlinkSync(gatePath)
          } catch {
            // Already gone / never written.
          }
        }
      },
      sleep: (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms))
    }
  }

  function releaseHostSpawnReservation() {
    hostSpawnReservation?.release()
    hostSpawnReservation = null
  }

  return {
    stopAttachedBackendMonitor,
    startAttachedBackendMonitor,
    attachToRunningHostBackend,
    releaseHostSpawnReservation,
    hostSpawnGateDeps
  }
}
