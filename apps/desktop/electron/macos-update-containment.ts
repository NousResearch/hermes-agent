import { execFileSync } from 'node:child_process'

import type { UpdateScriptHandoff } from './updater-process'

export const MACOS_SAFE_PUBLISHER_CAPABILITY = 'hermes-desktop-macos-publisher'
export const MACOS_SAFE_PUBLISHER_MIN_VERSION = 1
export const SAFE_PUBLISHER_UNAVAILABLE_MESSAGE =
  'In-app updates are temporarily disabled on macOS because this build does not include the safe Desktop publisher. Run `hermes update` in a terminal, then install Hermes through the trusted replacement package.'

export interface MacosSafePublisherCapability {
  capability: typeof MACOS_SAFE_PUBLISHER_CAPABILITY
  version: number
}

export function parseMacosSafePublisherCapability(raw: string): MacosSafePublisherCapability | null {
  try {
    const parsed = JSON.parse(raw)

    if (
      parsed?.capability !== MACOS_SAFE_PUBLISHER_CAPABILITY ||
      !Number.isInteger(parsed?.version) ||
      parsed.version < MACOS_SAFE_PUBLISHER_MIN_VERSION
    ) {
      return null
    }

    return { capability: MACOS_SAFE_PUBLISHER_CAPABILITY, version: parsed.version }
  } catch {
    return null
  }
}

/** Ask the repo-owned coordinator whether it delegates macOS publication to a
 * compatible safe publisher. Unsupported, stale, malformed, or failed probes
 * all fail closed; executing the probe must not mutate the checkout or app. */
export function probeMacosSafePublisherCapability(
  handoff: UpdateScriptHandoff,
  run: typeof execFileSync = execFileSync
): MacosSafePublisherCapability | null {
  try {
    const raw = run(handoff.command, [...handoff.args, '--publisher-capability'], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
      timeout: 5_000
    })

    return parseMacosSafePublisherCapability(String(raw).trim())
  } catch {
    return null
  }
}

interface UpdateIpcHandlersDeps {
  applyUpdates: (payload: any) => Promise<any>
  checkUpdates: () => Promise<any>
  isMac: boolean
  publisherCapability: () => MacosSafePublisherCapability | null
  checkErrorBranch?: () => string
  now?: () => number
}

export interface DesktopUpdateIpcHandlers {
  check: () => Promise<any>
  apply: (payload?: any) => Promise<any>
}

/** Authoritative Electron-main boundary for desktop update IPC. The publisher
 * capability is probed for every macOS mutation request, so direct IPC and a
 * renderer that was loaded before containment cannot reuse stale permission. */
export function createDesktopUpdateIpcHandlers(deps: UpdateIpcHandlersDeps): DesktopUpdateIpcHandlers {
  return {
    check: async () =>
      deps.checkUpdates().catch(error => ({
        branch: deps.checkErrorBranch?.() ?? 'main',
        error: 'check-failed',
        fetchedAt: (deps.now ?? Date.now)(),
        message: error?.message || String(error),
        supported: true
      })),
    apply: async (payload?: any) => {
      if (deps.isMac && !deps.publisherCapability()) {
        return {
          error: 'safe-publisher-unavailable',
          message: SAFE_PUBLISHER_UNAVAILABLE_MESSAGE,
          ok: false
        }
      }

      return deps.applyUpdates(payload || {}).catch(error => ({
        error: 'apply-failed',
        message: error?.message || String(error),
        ok: false
      }))
    }
  }
}

interface IpcMainHandleBoundary {
  handle: (channel: string, handler: (...args: any[]) => Promise<any>) => unknown
}

/** Register both updater channels from one shared main-process owner. */
export function registerDesktopUpdateIpc(ipcMain: IpcMainHandleBoundary, handlers: DesktopUpdateIpcHandlers): void {
  ipcMain.handle('hermes:updates:check', async () => handlers.check())
  ipcMain.handle('hermes:updates:apply', async (_event, payload) => handlers.apply(payload))
}
