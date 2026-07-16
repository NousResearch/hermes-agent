import { useCallback, useEffect, useRef } from 'react'

import { getGlobalModelInfo } from '@/hermes'
import {
  $activeSessionId,
  $currentModel,
  $currentModelExplicitlySet,
  $currentProvider,
  setCurrentModel,
  setCurrentProvider
} from '@/store/session'

// Same cadence as the cron-jobs polling block in desktop-controller.tsx —
// 30 s while the app is open + visible. Cheap (single GET against a backend
// that already has the config in memory), and rare enough that a Dashboard
// edit / `hermes model` / `hermes config set` round-trip shows up within a
// polling interval without flooding the backend.
const MODEL_POLL_INTERVAL_MS = 30_000

export interface ProfileDefaultSnapshot {
  model: string
  provider: string
}

interface UseModelProfileSyncOptions {
  /** True only after the gateway is open (web_server is up + serving). */
  gatewayOpen: boolean
}

/**
 * Run one sync tick against the server. Exported for tests so the timing
 * machinery (setInterval + visibilitychange) doesn't have to be driven
 * through fake timers — call this directly with the previous baseline.
 *
 * Returns the *next* baseline snapshot — caller writes it back to its ref.
 *
 * Contract:
 *  - Returns immediately if a live session is active.
 *  - On the first call (lastSeenDefault empty), records the server value
 *    and writes nothing to the composer — first-run seed is
 *    `refreshCurrentModel`'s job.
 *  - On subsequent calls, if the server value matches the previous baseline
 *    OR the user explicitly picked via the picker (`$currentModelExplicitlySet`)
 *    OR the composer diverges from the previous baseline, returns the new
 *    server value as the baseline without writing. The user pick (explicit or
 *    divergent composer) is sacred.
 *  - Otherwise (server drifted AND composer was still showing the previous
 *    baseline AND the user didn't explicitly pick), writes the new value to
 *    the composer AND returns it as the new baseline.
 */
export async function syncProfileDefaultTick(
  lastSeenDefault: ProfileDefaultSnapshot
): Promise<ProfileDefaultSnapshot> {
  if ($activeSessionId.get()) {
    return lastSeenDefault
  }

  let result: { model?: unknown; provider?: unknown }

  try {
    result = await getGlobalModelInfo()
  } catch {
    return lastSeenDefault
  }

  const serverModel = typeof result.model === 'string' ? result.model : ''
  const serverProvider = typeof result.provider === 'string' ? result.provider : ''

  // No baseline yet → record and stop.
  if (!lastSeenDefault.model) {
    return { model: serverModel, provider: serverProvider }
  }

  // Server value unchanged → no-op.
  if (lastSeenDefault.model === serverModel && lastSeenDefault.provider === serverProvider) {
    return lastSeenDefault
  }

  // Server drifted. Skip if the user explicitly picked — a picker selection
  // equal to the previous default is still an explicit pick (selectModel
  // persists it), and we must not overwrite it on the next external change.
  if ($currentModelExplicitlySet.get()) {
    return { model: serverModel, provider: serverProvider }
  }

  // Did the composer follow the previous baseline?
  const composerModel = $currentModel.get()
  const composerProvider = $currentProvider.get()

  const composerFollowedBaseline =
    composerModel === lastSeenDefault.model && composerProvider === lastSeenDefault.provider

  // Only re-seed when the composer is still showing the old default.
  if (composerFollowedBaseline && serverModel) {
    setCurrentModel(serverModel)
    // Always write provider — even empty — to match refreshCurrentModel's seed
    // behaviour (use-model-controls.ts:69). An empty provider is the server's
    // answer; leaving a stale provider produces a model/provider mismatch.
    setCurrentProvider(serverProvider)
  }

  return { model: serverModel, provider: serverProvider }
}

/**
 * Keep the composer ($currentModel / $currentProvider) in sync with the
 * profile's `model.default` after the initial seed. The empty-composer
 * first-run path lives in `useModelControls.refreshCurrentModel`; this hook
 * picks up the case where the user *did* seed from the default but never
 * made an explicit pick, so the composer is just holding the last-seen
 * default. If that default changes externally — Dashboard Models page,
 * `hermes model`, `hermes config set`, another Hermes client on the same
 * profile — the composer should follow.
 *
 * The invariant the existing test contract pins: **a user pick is sacred**.
 * We never overwrite a pick. The mechanism is to compare the server's
 * current value against the server's last-seen value, not against the
 * composer's current value. If they match, the composer is already showing
 * what the server thinks; if they diverge, the user (or something else)
 * picked, and we stay out of the way.
 *
 * Skipped while a live session is active — the in-flight footer owns the
 * model label for that session.
 */
export function useModelProfileSync({ gatewayOpen }: UseModelProfileSyncOptions): void {
  const lastSeenDefault = useRef<ProfileDefaultSnapshot>({ model: '', provider: '' })

  const tick = useCallback(async () => {
    lastSeenDefault.current = await syncProfileDefaultTick(lastSeenDefault.current)
  }, [])

  useEffect(() => {
    if (!gatewayOpen) {
      // Reset on disconnect so a reconnect takes a fresh baseline (the
      // server may have been replaced with a different profile's backend).
      lastSeenDefault.current = { model: '', provider: '' }

      return
    }

    let cancelled = false

    const onVisible = () => {
      if (document.visibilityState === 'visible' && !cancelled) {
        void tick()
      }
    }

    const intervalId = window.setInterval(() => {
      if (document.visibilityState === 'visible' && !cancelled) {
        void tick()
      }
    }, MODEL_POLL_INTERVAL_MS)

    document.addEventListener('visibilitychange', onVisible)
    // Take an initial baseline immediately so the next real tick (or a
    // Dashboard-driven change) compares against something current rather
    // than waiting 30 s.
    void tick()

    return () => {
      cancelled = true
      window.clearInterval(intervalId)
      document.removeEventListener('visibilitychange', onVisible)
    }
  }, [gatewayOpen, tick])
}
