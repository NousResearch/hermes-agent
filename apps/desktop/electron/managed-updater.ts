type UpdaterEvent =
  | 'checking-for-update'
  | 'download-progress'
  | 'error'
  | 'update-available'
  | 'update-downloaded'
  | 'update-not-available'

interface AppUpdaterLike {
  allowDowngrade: boolean
  autoDownload: boolean
  autoInstallOnAppQuit: boolean
  checkForUpdates: () => Promise<unknown>
  on: (event: UpdaterEvent, listener: (...args: any[]) => void) => unknown
  removeListener: (event: UpdaterEvent, listener: (...args: any[]) => void) => unknown
}

export type ManagedUpdateStage = 'available' | 'checking' | 'disabled' | 'downloaded' | 'downloading' | 'error' | 'idle'

export interface ManagedUpdateSnapshot {
  checkedAt?: number
  error?: string
  percent: number | null
  stage: ManagedUpdateStage
  version?: string
}

interface ManagedUpdaterOptions {
  enabled: boolean
  now?: () => number
  shouldAcceptVersion?: (version: string) => boolean
  updater: AppUpdaterLike
}

interface ManagedUpdateRuntime {
  isPackaged: boolean
  platform: string
  updateConfigExists: boolean
}

type SnapshotListener = (snapshot: ManagedUpdateSnapshot) => void

export function shouldEnableManagedUpdates({
  isPackaged,
  platform,
  updateConfigExists
}: ManagedUpdateRuntime): boolean {
  return isPackaged && platform === 'win32' && updateConfigExists
}

function versionFrom(value: unknown): string | undefined {
  if (!value || typeof value !== 'object' || !('version' in value)) {
    return undefined
  }

  const version = (value as { version?: unknown }).version

  return typeof version === 'string' && version.trim() ? version.trim() : undefined
}

function normalizedPercent(value: unknown): number | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return null
  }

  return Math.round(Math.min(100, Math.max(0, value)) * 10) / 10
}

function errorMessage(value: unknown): string {
  const raw = value instanceof Error ? value.message : String(value || 'Update failed')

  // Updater/network errors can include authenticated feed URLs. Keep the useful
  // diagnostic while ensuring renderer-visible state never exposes credentials.
  return raw.replace(/([?&](?:access_token|api[_-]?key|auth|key|signature|token)=)[^&\s]+/giu, '$1[REDACTED]')
}

export function createManagedUpdater({
  enabled,
  now = Date.now,
  shouldAcceptVersion = () => true,
  updater
}: ManagedUpdaterOptions) {
  let snapshot: ManagedUpdateSnapshot = {
    percent: null,
    stage: enabled ? 'idle' : 'disabled'
  }

  let started = false
  let checkPromise: Promise<void> | null = null
  const subscribers = new Set<SnapshotListener>()

  const publish = (next: ManagedUpdateSnapshot) => {
    snapshot = Object.freeze({ ...next })

    for (const subscriber of subscribers) {
      subscriber(snapshot)
    }
  }

  const handlers: Record<UpdaterEvent, (...args: any[]) => void> = {
    'checking-for-update': () => {
      publish({ checkedAt: now(), percent: null, stage: 'checking' })
    },
    'update-available': info => {
      const version = versionFrom(info)

      if (version && !shouldAcceptVersion(version)) {
        updater.autoDownload = false
        publish({ checkedAt: now(), percent: null, stage: 'idle', version })

        return
      }

      updater.autoDownload = true
      publish({ checkedAt: now(), percent: 0, stage: 'available', version })
    },
    'download-progress': progress => {
      const value = progress && typeof progress === 'object' ? (progress as { percent?: unknown }).percent : null

      publish({
        checkedAt: now(),
        percent: normalizedPercent(value),
        stage: 'downloading',
        version: snapshot.version
      })
    },
    'update-downloaded': info => {
      publish({
        checkedAt: now(),
        percent: 100,
        stage: 'downloaded',
        version: versionFrom(info) ?? snapshot.version
      })
    },
    'update-not-available': info => {
      publish({ checkedAt: now(), percent: null, stage: 'idle', version: versionFrom(info) })
    },
    error: error => {
      publish({ checkedAt: now(), error: errorMessage(error), percent: null, stage: 'error' })
    }
  }

  const bind = () => {
    for (const [event, handler] of Object.entries(handlers) as Array<[UpdaterEvent, (...args: any[]) => void]>) {
      updater.on(event, handler)
    }
  }

  const unbind = () => {
    for (const [event, handler] of Object.entries(handlers) as Array<[UpdaterEvent, (...args: any[]) => void]>) {
      updater.removeListener(event, handler)
    }
  }

  const check = (): Promise<void> => {
    if (!enabled) {
      return Promise.resolve()
    }

    if (checkPromise) {
      return checkPromise
    }

    publish({ checkedAt: now(), percent: null, stage: 'checking' })

    try {
      checkPromise = Promise.resolve(updater.checkForUpdates())
        .then(() => undefined)
        .catch(error => {
          handlers.error(error)
        })
        .finally(() => {
          checkPromise = null
        })
    } catch (error) {
      handlers.error(error)
      checkPromise = Promise.resolve()
    }

    return checkPromise
  }

  return {
    check,
    dispose() {
      if (started) {
        unbind()
      }

      subscribers.clear()
      started = false
    },
    getSnapshot: () => snapshot,
    async start() {
      if (!enabled || started) {
        return
      }

      updater.autoDownload = true
      updater.autoInstallOnAppQuit = true
      updater.allowDowngrade = false
      bind()
      started = true
      await check()
    },
    subscribe(listener: SnapshotListener) {
      subscribers.add(listener)
      listener(snapshot)

      return () => subscribers.delete(listener)
    }
  }
}
