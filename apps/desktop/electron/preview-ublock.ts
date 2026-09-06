import { randomUUID } from 'node:crypto'

import type {
  InstalledUblock,
  PreviewUblockFailureCode,
  PreviewUblockInstallCandidate,
  PreviewUblockInstaller,
  UblockInstallIntent
} from './preview-ublock-installer'

const UBLOCK_DASHBOARD_PATH = 'dashboard.html'

export type PreviewUblockPhase =
  | 'checking-cache'
  | 'downloading'
  | 'extracting'
  | 'failed'
  | 'idle'
  | 'loading'
  | 'preparing'
  | 'ready'
  | 'validating'
  | 'verifying'

export interface PreviewUblockFailure {
  code: PreviewUblockFailureCode | 'activation'
  message: string
}

export interface PreviewUblockOperation {
  failure: PreviewUblockFailure | null
  operationId: string | null
  phase: PreviewUblockPhase
  receivedBytes: number
  totalBytes: number | null
}

export interface PreviewUblockExtension {
  id: string
  manifest: { name?: string; version?: string }
  path: string
  url: string
}

export interface PreviewUblockSession {
  extensions: {
    getAllExtensions(): PreviewUblockExtension[]
    getExtension(extensionId: string): PreviewUblockExtension | null
    loadExtension(path: string, options?: { allowFileAccess?: boolean }): Promise<PreviewUblockExtension>
    removeExtension(extensionId: string): void
  }
}

export interface PreviewUblockState {
  available: boolean
  dashboardUrl: string | null
  enabled: boolean
  extensionId: string | null
  operation: PreviewUblockOperation
  popupUrl: string | null
  rulesetsReady: boolean
  version: string | null
}

export interface PreviewUblockController {
  dispose(): Promise<void>
  getState(): PreviewUblockState
  initialize(): Promise<PreviewUblockState>
  setEnabled(enabled: boolean): Promise<PreviewUblockState>
  subscribe(listener: (state: PreviewUblockState) => void): () => void
}

interface PreviewUblockControllerOptions {
  bootstrap?: (extension: PreviewUblockExtension) => Promise<boolean>
  enabled?: boolean
  installer: Pick<PreviewUblockInstaller, 'resolve'>
  session: PreviewUblockSession
}

function isInstallCandidate(
  value: InstalledUblock | PreviewUblockInstallCandidate
): value is PreviewUblockInstallCandidate {
  return 'stagedPath' in value
}

const idleOperation = (): PreviewUblockOperation => ({
  failure: null,
  operationId: null,
  phase: 'idle',
  receivedBytes: 0,
  totalBytes: null
})

function initialState(enabled: boolean): PreviewUblockState {
  return {
    available: false,
    dashboardUrl: null,
    enabled,
    extensionId: null,
    operation: idleOperation(),
    popupUrl: null,
    rulesetsReady: false,
    version: null
  }
}

function extensionResourceUrl(extensionUrl: string, relativePath: string): string {
  const encodedPath = relativePath
    .split('/')
    .map(segment => encodeURIComponent(segment))
    .join('/')

  return new URL(encodedPath, `${extensionUrl.replace(/\/$/, '')}/`).toString()
}

function failureFor(error: unknown): PreviewUblockFailure {
  const candidate = error as { code?: unknown } | null
  const code = candidate?.code
  const validCodes: Array<PreviewUblockFailure['code']> = [
    'activation',
    'compatibility',
    'integrity',
    'network',
    'storage',
    'timeout'
  ]

  return {
    code:
      typeof code === 'string' && validCodes.includes(code as PreviewUblockFailure['code'])
        ? (code as PreviewUblockFailure['code'])
        : 'activation',
    message:
      error instanceof Error ? error.message.replace(/^uBlock Origin Lite could not be installed: /, '') : String(error)
  }
}

export function createPreviewUblockController({
  bootstrap,
  enabled: initiallyEnabled = false,
  installer,
  session
}: PreviewUblockControllerOptions): PreviewUblockController {
  let desiredEnabled = initiallyEnabled
  let extension: PreviewUblockExtension | null = null
  let rulesetsReady = false
  let state = initialState(false)
  let initializePromise: Promise<PreviewUblockState> | null = null
  let enablePromise: Promise<PreviewUblockState> | null = null
  let lifecycle = Promise.resolve()
  const listeners = new Set<(nextState: PreviewUblockState) => void>()

  const publish = (nextState: PreviewUblockState): PreviewUblockState => {
    state = nextState

    for (const listener of listeners) {
      try {
        listener(state)
      } catch {
        // State presentation is deliberately isolated from the install
        // lifecycle. A window can disappear while an operation is completing,
        // and one broken subscriber must not turn a committed install into a
        // reported failure for every other subscriber.
      }
    }

    return state
  }

  const enqueue = <T>(operation: () => Promise<T>): Promise<T> => {
    const next = lifecycle.then(operation, operation)
    lifecycle = next.then(
      () => undefined,
      () => undefined
    )

    return next
  }

  const setOperation = (operation: PreviewUblockOperation): void => {
    publish({ ...state, operation })
  }

  const beginOperation = (): string => {
    const operationId = randomUUID()
    setOperation({ ...idleOperation(), operationId, phase: 'checking-cache' })

    return operationId
  }

  const removeLoadedExtension = (): void => {
    if (extension && session.extensions.getExtension(extension.id)) {
      session.extensions.removeExtension(extension.id)
    }
    extension = null
    rulesetsReady = false
  }

  const loadAndValidate = async (
    resolved: { path: string; popupPath?: string; version: string },
    publishReady = true
  ): Promise<PreviewUblockState> => {
    setOperation({ ...state.operation, phase: 'loading' })
    const loaded = session.extensions.getAllExtensions().find(item => item.path === resolved.path)
    const candidate = loaded ?? (await session.extensions.loadExtension(resolved.path, { allowFileAccess: false }))

    try {
      setOperation({ ...state.operation, phase: 'validating' })
      const ready = bootstrap ? await bootstrap(candidate) : true

      if (!ready) {
        throw Object.assign(new Error('uBlock Origin Lite failed its functional validation'), {
          code: 'activation' as const
        })
      }
      extension = candidate
      rulesetsReady = true

      const nextState: PreviewUblockState = {
        available: true,
        dashboardUrl: `${candidate.url}/${UBLOCK_DASHBOARD_PATH}`,
        enabled: true,
        extensionId: candidate.id,
        operation: { ...state.operation, failure: null, phase: publishReady ? 'ready' : 'validating' },
        popupUrl: resolved.popupPath ? extensionResourceUrl(candidate.url, resolved.popupPath) : null,
        rulesetsReady: true,
        version: candidate.manifest.version ?? resolved.version
      }

      return publishReady ? publish(nextState) : nextState
    } catch (error) {
      if (session.extensions.getExtension(candidate.id)) {
        session.extensions.removeExtension(candidate.id)
      }
      throw error
    }
  }

  const resolveAndLoad = async (intent: UblockInstallIntent): Promise<PreviewUblockState> => {
    const resolved = await installer.resolve(intent, {
      onPhase: phase => setOperation({ ...state.operation, phase: phase as PreviewUblockPhase }),
      onProgress: progress => {
        const receivedBytes = Math.max(state.operation.receivedBytes, progress.receivedBytes)
        setOperation({ ...state.operation, receivedBytes, totalBytes: progress.totalBytes })
      }
    })

    if (!resolved) {
      throw Object.assign(new Error('uBlock Origin Lite is not installed'), { code: 'storage' as const })
    }

    if (isInstallCandidate(resolved)) {
      try {
        // The staged tree is loaded first. This keeps the active pointer and
        // the previous immutable release untouched while compatibility is
        // checked against the candidate.
        await loadAndValidate(
          { path: resolved.stagedPath, popupPath: resolved.popupPath, version: resolved.version },
          false
        )
        removeLoadedExtension()

        resolved.installFinal()

        // Loading the final immutable path catches rename/filesystem issues
        // before the candidate can become the active release.
        const finalState = await loadAndValidate(
          { path: resolved.finalPath, popupPath: resolved.popupPath, version: resolved.version },
          false
        )
        resolved.commitActive()

        return publish({
          ...finalState,
          operation: { ...finalState.operation, failure: null, phase: 'ready' }
        })
      } catch (error) {
        resolved.discard()
        throw error
      }
    }

    return loadAndValidate(resolved)
  }

  const runEnable = async (startup: boolean): Promise<PreviewUblockState> => {
    const operationId = beginOperation()

    try {
      return await resolveAndLoad(startup ? 'cached' : 'pinned')
    } catch (error) {
      removeLoadedExtension()

      return publish({
        ...initialState(false),
        operation: {
          ...state.operation,
          failure: failureFor(error),
          operationId,
          phase: 'failed'
        }
      })
    }
  }

  const disable = async (): Promise<PreviewUblockState> => {
    removeLoadedExtension()

    return publish({ ...initialState(false), operation: idleOperation() })
  }

  const controller: PreviewUblockController = {
    async dispose() {
      desiredEnabled = false
      initializePromise = null
      enablePromise = null
      await enqueue(disable)
    },
    getState() {
      return state
    },
    initialize() {
      if (!initializePromise) {
        initializePromise = enqueue(() => (desiredEnabled ? runEnable(true) : disable()))
      }

      return initializePromise
    },
    setEnabled(enabled) {
      if (enabled && enablePromise) {
        return enablePromise
      }
      desiredEnabled = enabled

      if (enabled) {
        enablePromise = enqueue(() => runEnable(false))
        void enablePromise.finally(() => {
          enablePromise = null
        })

        return enablePromise
      }

      return enqueue(disable)
    },
    subscribe(listener) {
      listeners.add(listener)
      try {
        listener(state)
      } catch {
        // Match publish() above: an initial state notification is also
        // presentation-only and must not make subscribing observable as an
        // install failure.
      }

      return () => listeners.delete(listener)
    }
  }

  return controller
}

export { UBLOCK_DASHBOARD_PATH }
