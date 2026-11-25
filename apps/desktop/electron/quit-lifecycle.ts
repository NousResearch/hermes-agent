import { shouldQuitOnWindowAllClosed } from './main-window-lifecycle'
import { waitForManagedUpdateOperations } from './managed-ssh-update'
import { createQuitFinalization, shouldArmQuitFinalization } from './quit-finalization'
import { type ActiveWork, mergeActiveWork, type QuitPrompt, quitPromptFor } from './quit-guard'
import {
  type BackendQuitActivity,
  backendQuitNeedsWait,
  createQuitTeardownCoordinator,
  type QuitTeardownTask
} from './quit-teardown'

export interface PreventableEvent {
  preventDefault: () => void
}

export interface QuitLifecycleDeps {
  platform: NodeJS.Platform | string
  skipQuitConfirm: boolean
  isQuittingForHandoff: () => boolean
  hasBackendShutdownStarted: () => boolean
  getActiveWorkReports: () => Iterable<ActiveWork>
  quitStopsBackendWork: () => boolean
  restoreTrayBeforePrompt: () => void
  showQuitPrompt: (prompt: QuitPrompt) => Promise<boolean> | null
  setAppQuitting: (quitting: boolean) => void
  onQuitAdmitted: () => void
  getManagedUpdateOperations: () => Promise<unknown>[]
  sealSshBootstrap: () => void
  getBackendActivity: () => BackendQuitActivity
  hasSshTeardownWork: () => boolean
  runBackendShutdown: () => Promise<void>
  runSshTeardown: () => Promise<void>
  onBeforeQuitCleanup: () => void
  runWillQuitCleanup: () => void
  log: (message: string) => void
  flushLogsSync: () => void
  requestQuit: () => void
  hardExit: (code: number) => void
  finalizationTimeoutMs?: number
  scheduleFinalization?: (callback: () => void, timeoutMs: number) => unknown
  cancelFinalization?: (handle: unknown) => void
}

export interface QuitLifecycle {
  isQuitInProgress: () => boolean
  isQuitUnderway: () => boolean
  heldQuitForActiveWork: (event: PreventableEvent) => boolean
  handleBeforeQuit: (event: PreventableEvent) => void
  handleWillQuit: () => void
  handleQuit: () => void
  handleWindowAllClosed: () => void
  handleActivateWhileQuitting: () => boolean
}

export function createQuitLifecycle(deps: QuitLifecycleDeps): QuitLifecycle {
  let quitInProgress = false
  let quitPromptOpen = false
  let quitConfirmedWithActiveWork = false
  let managedUpdateQuitWait: Promise<unknown> | null = null
  let managedUpdateQuitWaitDone = false
  let willQuitCleanupRan = false

  const isMac = deps.platform === 'darwin'

  function isQuitUnderway(): boolean {
    return quitInProgress || deps.isQuittingForHandoff() || deps.hasBackendShutdownStarted()
  }

  function runWillQuitCleanupOnce(): void {
    if (willQuitCleanupRan) {
      return
    }

    willQuitCleanupRan = true
    deps.runWillQuitCleanup()
  }

  const quitFinalization = createQuitFinalization({
    enabled: shouldArmQuitFinalization(deps.platform),
    timeoutMs: deps.finalizationTimeoutMs,
    schedule: deps.scheduleFinalization,
    cancel: deps.cancelFinalization,
    hardExit: code => {
      runWillQuitCleanupOnce()
      deps.log(`[quit] forcing ${deps.platform} process exit after Electron quit finalization stalled`)
      deps.flushLogsSync()
      deps.hardExit(code)
    }
  })

  function armQuitFinalization(): void {
    // Packaged macOS Squirrel updates (`quitAndInstall()`) set
    // `isQuittingForHandoff` and may pause for an admin password prompt before
    // emitting `quit`; never hard-exit out from under Squirrel handoff.
    if (isMac && deps.isQuittingForHandoff()) {
      return
    }

    quitFinalization.arm()
  }

  const quitTeardown = createQuitTeardownCoordinator(() => {
    armQuitFinalization()
    deps.requestQuit()
  })

  function heldQuitForActiveWork(event: PreventableEvent): boolean {
    if (deps.skipQuitConfirm || quitConfirmedWithActiveWork || isQuitUnderway()) {
      return false
    }

    if (quitPromptOpen) {
      event.preventDefault()

      return true
    }

    const work = mergeActiveWork(deps.getActiveWorkReports())
    const prompt = quitPromptFor(work, deps.isQuittingForHandoff(), deps.quitStopsBackendWork())

    if (!prompt) {
      return false
    }

    deps.restoreTrayBeforePrompt()
    const pending = deps.showQuitPrompt(prompt)

    if (!pending) {
      return false
    }

    event.preventDefault()
    quitPromptOpen = true

    void pending
      .then(confirmed => {
        quitPromptOpen = false

        if (confirmed) {
          quitConfirmedWithActiveWork = true
          deps.requestQuit()
        }
      })
      .catch(() => {
        // A dialog we can't show must not become a quit we can't perform.
        quitPromptOpen = false
        quitConfirmedWithActiveWork = true
        deps.requestQuit()
      })

    return true
  }

  function handleBeforeQuit(event: PreventableEvent): void {
    if (heldQuitForActiveWork(event)) {
      deps.setAppQuitting(false)
      quitInProgress = false

      return
    }

    deps.setAppQuitting(true)
    quitInProgress = true
    deps.onQuitAdmitted()

    if (
      !managedUpdateQuitWaitDone &&
      (managedUpdateQuitWait || deps.getManagedUpdateOperations().length > 0)
    ) {
      event.preventDefault()

      if (!managedUpdateQuitWait) {
        managedUpdateQuitWait = waitForManagedUpdateOperations(deps.getManagedUpdateOperations).finally(() => {
          managedUpdateQuitWaitDone = true
          setTimeout(() => deps.requestQuit(), 0)
        })
      }

      return
    }

    // Latch past the managed-update join once we enter teardown so a late IPC
    // update cannot re-defer before-quit under an armed quitFinalization timer.
    managedUpdateQuitWaitDone = true

    deps.sealSshBootstrap()

    const backendNeedsWait = backendQuitNeedsWait(deps.getBackendActivity())

    const teardownTasks: QuitTeardownTask[] = [
      { run: deps.runBackendShutdown, waitForCompletion: backendNeedsWait }
    ]

    if (deps.hasSshTeardownWork()) {
      teardownTasks.push({ run: deps.runSshTeardown, waitForCompletion: true })
    }

    if (quitTeardown.begin(teardownTasks)) {
      event.preventDefault()
    } else {
      armQuitFinalization()
    }

    deps.onBeforeQuitCleanup()
  }

  function handleWillQuit(): void {
    runWillQuitCleanupOnce()
    armQuitFinalization()
  }

  function handleQuit(): void {
    quitFinalization.cancel()
  }

  function handleWindowAllClosed(): void {
    if (
      shouldQuitOnWindowAllClosed({
        platform: deps.platform,
        isQuittingForHandoff: deps.isQuittingForHandoff(),
        quitInProgress
      })
    ) {
      deps.requestQuit()
    }
  }

  function handleActivateWhileQuitting(): boolean {
    if (!isQuitUnderway()) {
      return false
    }

    if (quitTeardown.isFinished()) {
      armQuitFinalization()
      setTimeout(() => deps.requestQuit(), 0)
    }

    return true
  }

  return {
    isQuitInProgress: () => quitInProgress,
    isQuitUnderway,
    heldQuitForActiveWork,
    handleBeforeQuit,
    handleWillQuit,
    handleQuit,
    handleWindowAllClosed,
    handleActivateWhileQuitting
  }
}
