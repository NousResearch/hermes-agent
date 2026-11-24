import { hermesApi } from '@/api/client'
import { $profileRemoteOverrides } from '@/store/profile-remote-override'

import {
  backendTransitionFor,
  type BackendTransitionKind,
  beginBackendTransition,
  endBackendTransition,
  transitionProfileKey
} from './backend-transition'

// A profile a plugin runs (host.setProfileBackend) sits behind the plugin's own
// process on this computer, which keeps answering while the backend behind it
// restarts. When the connection to such a profile drops, the app asks the
// plugin what is going on before calling it a failure, through the profile's
// route:
//
//   GET /api/plugins/<plugin>/status  ->  { state: string }
//
// `state` "starting", "restarting" or "stopping" is a planned transition;
// "running" means the backend is back; anything else, or no answer, is not
// planned and the normal offline path shows. (desktop-plugin-sdk.md)

export interface LauncherStatus {
  state?: string
}

export const LAUNCHER_POLL_MS = 1_000
/** After the launcher reports the backend running, how long the planned state
 *  waits for the socket to open before it ends anyway (a real failure then
 *  shows). */
export const LAUNCHER_REOPEN_GRACE_MS = 10_000
const LAUNCHER_STATUS_TIMEOUT_MS = 4_000
// Consecutive unanswered polls before a watched transition counts as failed.
const LAUNCHER_MISSES_ALLOWED = 5

/** The plugin whose launcher runs this profile's backend, or null. */
export function launcherPluginFor(profile: null | string | undefined): null | string {
  return $profileRemoteOverrides.get()[transitionProfileKey(profile)]?.launcher?.plugin ?? null
}

export async function readLauncherStatus(profile: string, plugin: string): Promise<LauncherStatus | null> {
  try {
    const status = await hermesApi<LauncherStatus>({
      path: `/api/plugins/${encodeURIComponent(plugin)}/status`,
      profile,
      timeoutMs: LAUNCHER_STATUS_TIMEOUT_MS
    })

    return status && typeof status === 'object' ? status : null
  } catch {
    return null
  }
}

export type LauncherReading = { kind: BackendTransitionKind; planned: true } | { planned: false; ready: boolean }

const PLANNED_STATES: readonly string[] = ['restarting', 'starting', 'stopping']

export function launcherReading(status: LauncherStatus | null): LauncherReading {
  const state = status?.state ?? ''

  if (PLANNED_STATES.includes(state)) {
    return { kind: state as BackendTransitionKind, planned: true }
  }

  return { planned: false, ready: state === 'running' }
}

const watches = new Map<string, { stop: () => void }>()
// Profiles whose backend is back while their socket is still re-opening: the
// planned state stays (useLauncherTransitions ends it when the socket opens),
// so the reconnect's own offline/connecting states never show, and no new poll
// starts for it meanwhile.
const reopening = new Map<string, ReturnType<typeof setTimeout>>()

function endReopening(key: string): void {
  const timer = reopening.get(key)

  if (timer !== undefined) {
    clearTimeout(timer)
    reopening.delete(key)
  }
}

/** The launcher reported the backend running: reconnect now and hold the
 *  planned state until the socket opens, at most LAUNCHER_REOPEN_GRACE_MS. */
function holdUntilReopened(key: string, onReady?: () => void): void {
  const since = backendTransitionFor(key)?.since

  endReopening(key)
  reopening.set(
    key,
    setTimeout(() => {
      reopening.delete(key)

      if (backendTransitionFor(key)?.since === since) {
        endBackendTransition(key, { ok: true, source: 'launcher' })
      }
    }, LAUNCHER_REOPEN_GRACE_MS)
  )
  onReady?.()
}

/** The socket of a launched profile opened: its planned state is over. */
export function endLauncherTransitionOnOpen(profile: null | string | undefined): void {
  const key = transitionProfileKey(profile)

  endReopening(key)
  endBackendTransition(key, { source: 'launcher' })
}

/**
 * Ask the launcher once. A planned transition is published (and watched until
 * it ends); returns whether the profile is in one now.
 */
export async function probeLauncherTransition(
  profile: null | string | undefined,
  onReady?: () => void
): Promise<boolean> {
  const key = transitionProfileKey(profile)
  const plugin = launcherPluginFor(key)

  if (!plugin) {
    return false
  }

  const current = backendTransitionFor(key)

  // A switch in progress owns its state and its reconnect; only a launcher
  // transition is watched here (not while its socket is already re-opening).
  if (current) {
    if (current.source === 'launcher' && !reopening.has(key)) {
      watchLauncherTransition(key, onReady)
    }

    return true
  }

  const reading = launcherReading(await readLauncherStatus(key, plugin))
  // The answer describes the moment it was asked. A transition that began meanwhile (a switch, another
  // probe) keeps its state, and a profile that left this launcher gets none from it.
  const begun = backendTransitionFor(key)

  if (begun || launcherPluginFor(key) !== plugin) {
    return begun !== null
  }

  if (!reading.planned) {
    return false
  }

  beginBackendTransition(key, reading.kind, { source: 'launcher' })
  watchLauncherTransition(key, onReady)

  return true
}

/**
 * Poll the launcher until the profile's planned transition ends: "running"
 * ends it as planned (then `onReady`, e.g. reconnect now instead of waiting
 * out the backoff); a state that is neither planned nor running, or several
 * unanswered polls, end it as failed. One watch per profile.
 */
export function watchLauncherTransition(profile: string, onReady?: () => void): () => void {
  const key = transitionProfileKey(profile)
  const existing = watches.get(key)

  if (existing) {
    return existing.stop
  }

  let stopped = false
  let misses = 0
  let timer: ReturnType<typeof setTimeout> | null = null

  const stop = () => {
    stopped = true

    if (timer !== null) {
      clearTimeout(timer)
    }

    if (watches.get(key)?.stop === stop) {
      watches.delete(key)
    }
  }

  const tick = async () => {
    timer = null
    const plugin = launcherPluginFor(key)
    const watched = backendTransitionFor(key)

    if (stopped || watched?.source !== 'launcher') {
      return stop()
    }

    if (!plugin) {
      endBackendTransition(key, { ok: false, source: 'launcher' })

      return stop()
    }

    const status = await readLauncherStatus(key, plugin)

    if (stopped) {
      return
    }

    // The answer is for the transition that was asked about. One that ended or was replaced meanwhile
    // (the socket opened, its deadline, a switch) is not this poll's to change or to bring back.
    if (backendTransitionFor(key) !== watched) {
      return stop()
    }

    const reading = launcherReading(status)

    if (reading.planned) {
      misses = 0
      beginBackendTransition(key, reading.kind, { source: 'launcher' })
    } else if (reading.ready) {
      stop()
      holdUntilReopened(key, onReady)

      return
    } else if (status || ++misses >= LAUNCHER_MISSES_ALLOWED) {
      endBackendTransition(key, { ok: false, source: 'launcher' })

      return stop()
    }

    timer = setTimeout(() => void tick(), LAUNCHER_POLL_MS)
  }

  watches.set(key, { stop })
  timer = setTimeout(() => void tick(), LAUNCHER_POLL_MS)

  return stop
}

/** Test seam. */
export function stopLauncherWatchesForTests(): void {
  for (const { stop } of [...watches.values()]) {
    stop()
  }

  for (const key of [...reopening.keys()]) {
    endReopening(key)
  }
}
