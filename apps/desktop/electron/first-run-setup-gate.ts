interface FirstRunSetupBackend {
  activeRoot?: string
  kind?: string
  platform?: string
  /** What the local setup card represents: 'none' = installer offer, the rest = use existing. */
  local?: 'none' | 'installed' | 'bundled' | 'bundled-damaged'
}

interface FirstRunSetupGateOptions {
  hideChoice?: () => void
  log?: (message: string) => void
  onStuck?: (backend: FirstRunSetupBackend, stuckAfterMs: number) => void
  promptChoice?: (backend: FirstRunSetupBackend) => void
  stuckAfterMs?: number
}

export type FirstRunSetupDecision = 'continue-local' | 'remote-applied' | 'reset'

export function createFirstRunSetupGate({
  hideChoice,
  log,
  onStuck,
  promptChoice,
  stuckAfterMs = 120000
}: FirstRunSetupGateOptions = {}) {
  let localBootstrapConfirmed = false

  let waiter: {
    promise: Promise<FirstRunSetupDecision>
    resolve: (decision: FirstRunSetupDecision) => void
  } | null = null

  let stuckTimer: ReturnType<typeof setTimeout> | null = null

  const clearStuckTimer = () => {
    if (stuckTimer) {
      clearTimeout(stuckTimer)
      stuckTimer = null
    }
  }

  const armStuckTimer = (backend: FirstRunSetupBackend) => {
    clearStuckTimer()

    if (!Number.isFinite(stuckAfterMs) || stuckAfterMs <= 0 || typeof log !== 'function') {
      return
    }

    stuckTimer = setTimeout(() => {
      onStuck?.(backend, stuckAfterMs)
      log(
        `[bootstrap] still waiting for first-run setup choice after ${Math.round(stuckAfterMs / 1000)}s ` +
          `(platform=${backend?.platform || 'unknown'})`
      )
    }, stuckAfterMs)

    if (typeof stuckTimer.unref === 'function') {
      stuckTimer.unref()
    }
  }

  // Both sentinels need a user decision: 'bootstrap-needed' offers install-or-
  // connect; 'bundled-unusable' (a bundled install whose payload is damaged)
  // offers connect-or-reinstall — never install.
  const shouldGate = (backend?: FirstRunSetupBackend | null) =>
    Boolean(
      backend &&
        (backend.kind === 'bootstrap-needed' || backend.kind === 'bundled-unusable') &&
        !localBootstrapConfirmed
    )

  const wait = async (backend?: FirstRunSetupBackend | null) => {
    if (!shouldGate(backend)) {
      return 'continue-local' as const
    }

    if (waiter) {
      return waiter.promise
    }

    promptChoice?.(backend)
    armStuckTimer(backend)

    let resolveWaiter: (decision: FirstRunSetupDecision) => void = () => {}

    const promise = new Promise<FirstRunSetupDecision>(resolve => {
      resolveWaiter = resolve
    })

    waiter = { promise, resolve: resolveWaiter }

    return promise
  }

  const settleWaiter = (decision: FirstRunSetupDecision) => {
    clearStuckTimer()

    if (!waiter) {
      return false
    }

    const activeWaiter = waiter
    waiter = null
    activeWaiter.resolve(decision)

    return true
  }

  const continueLocal = () => {
    localBootstrapConfirmed = true
    settleWaiter('continue-local')
  }

  const resetForRetry = () => {
    // Reset paths are followed by a renderer reload / fresh startHermes() call.
    // Settle the old boot explicitly so it cannot fall through into local
    // bootstrap and cannot leak a forever-pending connection promise.
    settleWaiter('reset')
  }

  const resetForRepair = () => {
    resetForRetry()
    localBootstrapConfirmed = false
  }

  const abandonForRemoteApply = () => {
    // Resume the gated startHermes() with an explicit remote decision. The
    // caller re-resolves the newly-persisted remote config instead of falling
    // through into local bootstrap or leaking the original connection promise.
    const resumedWaiter = settleWaiter('remote-applied')

    if (!resumedWaiter) {
      return false
    }

    localBootstrapConfirmed = false
    hideChoice?.()

    return true
  }

  const isLocalBootstrapConfirmed = () => localBootstrapConfirmed
  const hasWaiter = () => Boolean(waiter)

  return {
    abandonForRemoteApply,
    continueLocal,
    hasWaiter,
    isLocalBootstrapConfirmed,
    resetForRepair,
    resetForRetry,
    shouldGate,
    wait
  }
}
