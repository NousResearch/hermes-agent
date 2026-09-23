import type { BrowserWindow } from 'electron'

import { shouldHoldBootProgressForReauth } from './backend-start-failure'
import { createFirstRunSetupGate } from './first-run-setup-gate'

interface FirstRunBootRuntimeOptions {
  activeRoot: string
  fakeMode: boolean
  fakeStepMs: number
  getMainWindow: () => BrowserWindow | null
  getRemoteReauthFailure: () => string | null
  log: (message: string) => void
  platform: string
}

export function createFirstRunBootRuntime({
  activeRoot: ACTIVE_HERMES_ROOT,
  fakeMode: BOOT_FAKE_MODE,
  fakeStepMs: BOOT_FAKE_STEP_MS,
  getMainWindow,
  getRemoteReauthFailure,
  log: rememberLog,
  platform
}: FirstRunBootRuntimeOptions) {
  let bootProgressState = {
    error: null,
    fakeMode: BOOT_FAKE_MODE,
    isCloudBackendDown: false,
    message: 'Waiting to start Hermes backend',
    phase: 'idle',
    progress: 0,
    retryable: false,
    running: false,
    statusCode: null,
    timestamp: Date.now()
  }

  function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms))
  }

  function clampBootProgress(value) {
    const numeric = Number(value)

    if (!Number.isFinite(numeric)) {
      return 0
    }

    return Math.max(0, Math.min(100, Math.round(numeric)))
  }

  function broadcastBootProgress() {
    const mainWindow = getMainWindow()

    if (!mainWindow || mainWindow.isDestroyed()) {
      return
    }

    const { webContents } = mainWindow

    if (!webContents || webContents.isDestroyed()) {
      return
    }

    webContents.send('hermes:boot-progress', bootProgressState)
  }

  // Bootstrap-event broadcast channel + state. The bootstrap runner emits a
  // stream of events (manifest, stage, log, complete, failed) that the renderer
  // install overlay subscribes to. We also keep a running snapshot:
  //   - manifest: the stage list (rendered as a checklist in the overlay)
  //   - stages:   per-stage state ('pending' | 'running' | 'succeeded' |
  //               'skipped' | 'failed') keyed by stage name
  //   - active:   true while a bootstrap is in flight; false otherwise
  //   - error:    last 'failed' event's error message
  //   - log:      bounded ring buffer of the last 200 log lines for the
  //               "Show details" affordance in the overlay
  //
  // The snapshot is queryable via the hermes:bootstrap:get IPC handler so a
  // reloaded renderer (e.g. devtools reload during dev) recovers state.
  // Bootstrap log ring: bounded buffer so a long install (npm + playwright
  // downloads can emit thousands of lines) doesn't grow unbounded in memory
  // AND so the renderer's getBootstrapState() reply stays a reasonable size.
  // We keep enough to cover an entire failed stage's transcript so the
  // 'Copy output' button gives the user actually-actionable context, not
  // just the last few lines.
  const BOOTSTRAP_LOG_RING_MAX = 500

  let bootstrapState = {
    active: false,
    manifest: null,
    stages: {},
    error: null,
    log: [],
    startedAt: null,
    completedAt: null,
    setupChoice: null,
    unsupportedPlatform: null
  }

  let firstRunSetupGate = null

  function broadcastBootstrapEvent(ev) {
    if (ev.type === 'manifest') {
      bootstrapState.manifest = ev
      bootstrapState.active = true
      bootstrapState.setupChoice = null
      bootstrapState.startedAt = bootstrapState.startedAt || Date.now()
      bootstrapState.stages = {}

      for (const stage of ev.stages || []) {
        bootstrapState.stages[stage.name] = { state: 'pending', json: null, durationMs: null, error: null }
      }
    } else if (ev.type === 'stage') {
      bootstrapState.stages[ev.name] = {
        state: ev.state,
        durationMs: ev.durationMs ?? null,
        json: ev.json ?? null,
        error: ev.error ?? null
      }
    } else if (ev.type === 'log') {
      bootstrapState.log.push({ ts: Date.now(), stage: ev.stage || null, line: ev.line, stream: ev.stream || 'stdout' })

      if (bootstrapState.log.length > BOOTSTRAP_LOG_RING_MAX) {
        bootstrapState.log.splice(0, bootstrapState.log.length - BOOTSTRAP_LOG_RING_MAX)
      }
    } else if (ev.type === 'complete') {
      bootstrapState.active = false
      bootstrapState.completedAt = Date.now()
      bootstrapState.error = null
      bootstrapState.unsupportedPlatform = null
    } else if (ev.type === 'failed') {
      bootstrapState.active = false
      bootstrapState.error = ev.error || 'unknown error'
      bootstrapState.setupChoice = null
    } else if (ev.type === 'unsupported-platform') {
      bootstrapState.active = false
      bootstrapState.setupChoice = null
      bootstrapState.unsupportedPlatform = {
        platform: ev.platform,
        activeRoot: ev.activeRoot,
        installCommand: ev.installCommand,
        docsUrl: ev.docsUrl
      }
    } else if (ev.type === 'setup-choice') {
      bootstrapState.active = false
      bootstrapState.error = null
      bootstrapState.manifest = null
      bootstrapState.stages = {}
      bootstrapState.setupChoice = ev.active
        ? {
            platform: ev.platform,
            activeRoot: ev.activeRoot
          }
        : null
      bootstrapState.unsupportedPlatform = null
    } else if (ev.type === 'dismissed') {
      resetBootstrapSnapshot()
    }

    const mainWindow = getMainWindow()

    if (!mainWindow || mainWindow.isDestroyed()) {
      return
    }

    const { webContents } = mainWindow

    if (!webContents || webContents.isDestroyed()) {
      return
    }

    webContents.send('hermes:bootstrap:event', ev)
  }

  function getBootstrapState() {
    return bootstrapState
  }

  function resetBootstrapSnapshot() {
    bootstrapState = {
      active: false,
      manifest: null,
      stages: {},
      error: null,
      log: [],
      startedAt: null,
      completedAt: null,
      setupChoice: null,
      unsupportedPlatform: null
    }
  }

  function promptFirstRunSetupChoice(backend) {
    broadcastBootstrapEvent({
      type: 'setup-choice',
      active: true,
      platform: backend.platform || platform,
      activeRoot: backend.activeRoot || ACTIVE_HERMES_ROOT
    })
  }

  function hideFirstRunSetupChoice() {
    if (bootstrapState.setupChoice) {
      broadcastBootstrapEvent({ type: 'setup-choice', active: false })
    }
  }

  function getFirstRunSetupGate() {
    if (!firstRunSetupGate) {
      firstRunSetupGate = createFirstRunSetupGate({
        hideChoice: hideFirstRunSetupChoice,
        log: rememberLog,
        onStuck: (_backend, stuckAfterMs) => {
          updateBootProgress(
            {
              error: null,
              message: `Still waiting for first-run setup choice after ${Math.round(stuckAfterMs / 1000)} seconds`,
              phase: 'bootstrap.choice',
              progress: 12,
              running: true
            },
            { allowDecrease: true }
          )
        },
        promptChoice: promptFirstRunSetupChoice
      })
    }

    return firstRunSetupGate
  }

  async function waitForFirstRunSetupChoice(backend) {
    const gate = getFirstRunSetupGate()

    if (!gate.shouldGate(backend)) {
      return 'continue-local'
    }

    updateBootProgress(
      {
        error: null,
        message: 'Waiting for first-run setup choice',
        phase: 'bootstrap.choice',
        progress: 12,
        running: true
      },
      { allowDecrease: true }
    )

    return gate.wait(backend)
  }

  function continueFirstRunLocalBootstrap() {
    getFirstRunSetupGate().continueLocal()
  }

  function abandonFirstRunSetupChoiceForRemoteApply() {
    const gate = getFirstRunSetupGate()

    if (!gate.hasWaiter()) {
      return false
    }

    const resumedGatedConnection = gate.abandonForRemoteApply()

    if (resumedGatedConnection) {
      broadcastBootstrapEvent({ type: 'dismissed' })
    }

    return resumedGatedConnection
  }

  function updateBootProgress(update, options: { allowDecrease?: boolean } = {}) {
    // A latched reauth rejection owns the boot surface until a recovery path
    // clears it; see shouldHoldBootProgressForReauth (#95701).
    if (shouldHoldBootProgressForReauth(getRemoteReauthFailure(), update)) {
      return
    }

    const nextProgressRaw =
      typeof update.progress === 'number' ? clampBootProgress(update.progress) : bootProgressState.progress

    const nextProgress = options.allowDecrease ? nextProgressRaw : Math.max(bootProgressState.progress, nextProgressRaw)

    bootProgressState = {
      ...bootProgressState,
      ...update,
      error: update.error === undefined ? bootProgressState.error : update.error,
      fakeMode: BOOT_FAKE_MODE || Boolean(update.fakeMode),
      progress: nextProgress,
      // `retryable` rides with `error`: it survives updates that preserve the
      // error and resets alongside a new/cleared error unless explicitly set.
      retryable:
        update.retryable === undefined
          ? update.error === undefined && Boolean(bootProgressState.retryable)
          : Boolean(update.retryable),
      timestamp: Date.now()
    }

    if (update.message) {
      rememberLog(`[boot] ${update.message}`)
    }

    broadcastBootProgress()
  }

  async function advanceBootProgress(phase, message, progress) {
    updateBootProgress({
      phase,
      message,
      progress,
      running: true,
      error: null
    })

    if (BOOT_FAKE_MODE) {
      await sleep(BOOT_FAKE_STEP_MS)
    }
  }

  return {
    abandonFirstRunSetupChoiceForRemoteApply,
    advanceBootProgress,
    broadcastBootProgress,
    broadcastBootstrapEvent,
    continueFirstRunLocalBootstrap,
    getBootProgressState: () => bootProgressState,
    getBootstrapState,
    getFirstRunSetupGate,
    resetBootstrapSnapshot,
    resetExistingSetupGateForRetry: () => firstRunSetupGate?.resetForRetry(),
    updateBootProgress,
    waitForFirstRunSetupChoice
  }
}
