import type {
  PreviewUblockController,
  PreviewUblockExtension,
  PreviewUblockSession,
  PreviewUblockState
} from './preview-ublock'
import { createPreviewUblockController } from './preview-ublock'
import {
  createPreviewUblockBlockCounter,
  type PreviewUblockBlockCounter,
  type PreviewUblockBlockCounterSession
} from './preview-ublock-block-counter'
import { createPreviewUblockInstaller, type PreviewUblockInstaller } from './preview-ublock-installer'
import {
  createPreviewUblockPopupManager,
  type PreviewUblockPopupManager,
  type PreviewUblockPopupParent,
  type PreviewUblockPopupWindow,
  type PreviewUblockPopupWindowOptions
} from './preview-ublock-popup'
import {
  broadcastPreviewUblockStateToWindows,
  type PreviewUblockStateWindow
} from './preview-ublock-state-broadcaster'
import {
  createPreviewUblockRequestBlocker,
  type PreviewUblockRequestBlocker,
  type PreviewUblockRequestBlockerSession
} from './preview-ublock-request-blocker'
import { readPreviewUblockSettings, writePreviewUblockSettings } from './preview-ublock-settings'

export interface PreviewUblockBootstrapWindow {
  destroy(): void
  executeJavaScript(code: string): Promise<unknown>
  loadURL(url: string): Promise<void>
}

export interface PreviewUblockOwnerWebContents {
  isDestroyed(): boolean
  send(channel: string, payload: unknown): void
}

export interface PreviewUblockRuntimeSession extends PreviewUblockSession {
  webRequest: unknown
}

export interface PreviewUblockRuntime {
  dispose(): Promise<void>
  getBlockedRequestCount(webContentsId: number): number
  getRequestBlockerBlockedRequestCount(): number
  getState(): PreviewUblockState
  initialize(): Promise<PreviewUblockState>
  loadRules(extensionPath: string): boolean
  openPopup(parent: PreviewUblockPopupParent): Promise<void>
  registerGuest(webContentsId: number, ownerWebContentsId: number, guest: PreviewUblockGuest): boolean
  recordBlockedRequest(webContentsId: number, requestId?: number): void
  setEnabled(enabled: boolean): Promise<PreviewUblockState>
  unregisterGuest(webContentsId: number): void
}

export interface PreviewUblockGuest {
  on(event: 'destroyed' | 'did-navigate', listener: () => void): unknown
  removeListener?(event: 'destroyed' | 'did-navigate', listener: () => void): unknown
  reload?(): unknown
}

interface PreviewUblockRuntimeDependencies {
  createInstaller?: (userDataPath: string) => Pick<PreviewUblockInstaller, 'resolve'>
  createBootstrapWindow: () => PreviewUblockBootstrapWindow
  createPopupWindow: (options: PreviewUblockPopupWindowOptions) => PreviewUblockPopupWindow
  getPopupWorkArea?: (bounds: { height: number; width: number; x: number; y: number }) => {
    height: number
    width: number
    x: number
    y: number
  }
  getOwnerWebContents: (ownerWebContentsId: number) => PreviewUblockOwnerWebContents | null
  getStateWindows: () => readonly PreviewUblockStateWindow[]
  log: (message: string) => void
  openExternal: (url: string) => boolean
  previewSession: PreviewUblockRuntimeSession
  settingsPath: string
  userDataPath: string
}

export function createPreviewUblockRuntime({
  createBootstrapWindow,
  createPopupWindow,
  createInstaller,
  getPopupWorkArea,
  getOwnerWebContents,
  getStateWindows,
  log,
  openExternal,
  previewSession,
  settingsPath,
  userDataPath
}: PreviewUblockRuntimeDependencies): PreviewUblockRuntime {
  let controller: PreviewUblockController | null = null
  let popupManager: PreviewUblockPopupManager | null = null
  let blockCounter: PreviewUblockBlockCounter | null = null
  let requestBlocker: PreviewUblockRequestBlocker | null = null
  let unsubscribe: (() => void) | null = null
  let disposed = false

  const requireController = (): PreviewUblockController => {
    if (!controller) {
      throw new Error('uBlock Origin Lite is not ready')
    }

    return controller
  }

  const publishState = (state: PreviewUblockState): void => {
    broadcastPreviewUblockStateToWindows(getStateWindows(), state, log)
  }

  const createController = (enabled: boolean): PreviewUblockController => {
    const installer = createInstaller?.(userDataPath) ?? createPreviewUblockInstaller({ userDataPath })

    blockCounter = createPreviewUblockBlockCounter({
      onUpdate: update => {
        const ownerWindow = getOwnerWebContents(update.ownerWebContentsId)

        if (!ownerWindow || ownerWindow.isDestroyed()) {
          return
        }

        ownerWindow.send('hermes:preview-ublock:blocked-request-count', {
          blockedRequestCount: update.blockedRequestCount,
          webContentsId: update.webContentsId
        })
      },
      session: previewSession as PreviewUblockBlockCounterSession
    })

    requestBlocker = createPreviewUblockRequestBlocker({
      onBlocked: details => {
        if (typeof details.webContentsId === 'number') {
          blockCounter?.recordBlockedRequest(details.webContentsId, details.id)
        }
      },
      session: previewSession as PreviewUblockRequestBlockerSession
    })

    const nextController = createPreviewUblockController({
      bootstrap: async (extension: PreviewUblockExtension) => {
        if (!requestBlocker?.loadRules(extension.path)) {
          return false
        }

        const bootstrapWindow = createBootstrapWindow()

        try {
          await bootstrapWindow.loadURL(`${extension.url}/dashboard.html`)

          return Boolean(
            await bootstrapWindow.executeJavaScript(
              `(async () => {
              const configured = await chrome.runtime.sendMessage({ what: 'getEnabledRulesets' })
              const enabled = await chrome.declarativeNetRequest.getEnabledRulesets()
              const staticIds = Array.isArray(configured)
                ? configured.filter(id => typeof id === 'string' && id.includes('://') === false)
                : []
              const missing = staticIds.filter(id => enabled.includes(id) === false)
              if (missing.length !== 0) {
                await chrome.declarativeNetRequest.updateEnabledRulesets({ enableRulesetIds: missing })
              }
              const after = await chrome.declarativeNetRequest.getEnabledRulesets()
              if (!staticIds.every(id => after.includes(id))) {
                return false
              }

              return true
            })()`
            )
          )
        } catch {
          return false
        } finally {
          bootstrapWindow.destroy()
        }
      },
      enabled,
      installer,
      session: previewSession
    })

    popupManager = createPreviewUblockPopupManager({
      controller: nextController,
      createWindow: createPopupWindow,
      getWorkArea: getPopupWorkArea,
      log,
      openExternal,
      session: previewSession
    })

    unsubscribe = nextController.subscribe(state => {
      const active = state.enabled && state.available && state.rulesetsReady
      requestBlocker?.setActive(active)
      blockCounter?.setActive(active)
      publishState(state)
    })

    return nextController
  }

  return {
    async dispose() {
      if (disposed) {
        return
      }

      disposed = true
      unsubscribe?.()
      unsubscribe = null
      popupManager?.dispose()
      popupManager = null
      blockCounter?.dispose()
      blockCounter = null
      requestBlocker?.dispose()
      requestBlocker = null

      if (controller) {
        await controller.dispose()
        controller = null
      }
    },
    getBlockedRequestCount(webContentsId) {
      return blockCounter?.getCount(webContentsId) ?? 0
    },
    getRequestBlockerBlockedRequestCount() {
      return requestBlocker?.getBlockedRequestCount() ?? 0
    },
    getState() {
      return requireController().getState()
    },
    async initialize() {
      if (disposed) {
        throw new Error('uBlock Origin Lite runtime is disposed')
      }

      const persistedEnabled = readPreviewUblockSettings(settingsPath).enabled

      if (!controller) {
        controller = createController(persistedEnabled)
      }

      const state = await controller.initialize()

      if (persistedEnabled && !state.enabled) {
        try {
          writePreviewUblockSettings(settingsPath, { enabled: false })
        } catch (error) {
          log(`[preview] uBlock preference reset failed: ${error instanceof Error ? error.message : String(error)}`)
        }

        log('[preview] uBlock Origin Lite cache unavailable; content blocking remains disabled')
      }

      log(
        state.available
          ? `[preview] loaded uBlock Origin Lite ${state.version ?? 'unknown'} (${state.extensionId}); rulesetsReady=${state.rulesetsReady}`
          : '[preview] uBlock Origin Lite unavailable'
      )

      return state
    },
    loadRules(extensionPath) {
      return requestBlocker?.loadRules(extensionPath) ?? false
    },
    openPopup(parent) {
      if (!popupManager) {
        throw new Error('Preview uBlock popup could not be opened')
      }

      return popupManager.open(parent)
    },
    recordBlockedRequest(webContentsId, requestId) {
      blockCounter?.recordBlockedRequest(webContentsId, requestId)
    },
    registerGuest(webContentsId, ownerWebContentsId, guest) {
      if (!blockCounter || !requestBlocker) {
        return false
      }

      const registered = blockCounter.registerGuest(webContentsId, ownerWebContentsId, guest)

      if (registered) {
        requestBlocker.registerGuest(webContentsId, ownerWebContentsId)
      }

      return registered
    },
    async setEnabled(enabled) {
      const requestedEnabled = enabled === true
      const nextState = await requireController().setEnabled(requestedEnabled)

      if (!requestedEnabled || nextState.enabled) {
        try {
          writePreviewUblockSettings(settingsPath, { enabled: nextState.enabled })
        } catch (error) {
          log(`[preview] uBlock preference write failed: ${error instanceof Error ? error.message : String(error)}`)
        }
      }

      return nextState
    },
    unregisterGuest(webContentsId) {
      blockCounter?.unregisterGuest(webContentsId)
      requestBlocker?.unregisterGuest(webContentsId)
    }
  }
}
