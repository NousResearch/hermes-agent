import { $workspaceMode, $workspaceOwnerKey } from '@/components/pane-shell/workspace-scope'
import { $activeGatewayProfile, $hydrationSyncProfile, normalizeProfileKey } from '@/store/profile'
import { $activeSessionId, $connection, $messages, $selectedStoredSessionId } from '@/store/session'
import { $focusedStoredSessionId } from '@/store/session-focus'
import { $focusedRuntimeId, $focusedSessionState, $sessionStates, $sessionTiles } from '@/store/session-states'

export const HYDRATION_SYNC_BADGE_TIMEOUT_MS = 30_000

let backgroundSyncCleanup: (() => void) | undefined

export function clearHydrationBackgroundSync(): void {
  backgroundSyncCleanup?.()
  $hydrationSyncProfile.set(null)
}

function beginHydrationBackgroundSync(profile: string, signal?: AbortSignal): void {
  clearHydrationBackgroundSync()
  signal?.throwIfAborted()

  const cleanup = () => {
    window.clearTimeout(timer)
    unlisten()
    signal?.removeEventListener('abort', cleanup)

    if (backgroundSyncCleanup === cleanup) {
      backgroundSyncCleanup = undefined
      $hydrationSyncProfile.set(null)
    }
  }

  const unlisten = $activeGatewayProfile.listen(next => {
    if (normalizeProfileKey(next) === profile) {
      cleanup()
    }
  })

  const timer = window.setTimeout(cleanup, HYDRATION_SYNC_BADGE_TIMEOUT_MS)
  backgroundSyncCleanup = cleanup
  signal?.addEventListener('abort', cleanup, { once: true })
  $hydrationSyncProfile.set(profile)
}

/** A cancelled caller stops waiting; shared socket/IPC warmup may still finish.
 * Subscribe to both outcomes so an abandoned rejection remains observed. */
export function awaitSessionOpenWork<T>(work: () => Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) {
    return work()
  }

  signal.throwIfAborted()

  return new Promise<T>((resolve, reject) => {
    let settled = false

    const finish = (publish: () => void) => {
      if (settled) {
        return
      }
      settled = true
      signal.removeEventListener('abort', abort)
      publish()
    }

    const abort = () => finish(() => reject(signal.reason))
    signal.addEventListener('abort', abort, { once: true })
    void Promise.resolve()
      .then(() => {
        signal.throwIfAborted()

        return work()
      })
      .then(
        value => finish(() => resolve(value)),
        error => finish(() => reject(error))
      )
  })
}

export function assertSessionOpenCurrent(isCurrent: () => boolean, signal?: AbortSignal): void {
  signal?.throwIfAborted()

  if (!isCurrent()) {
    throw new Error('Session open was superseded by a newer selection.')
  }
}

export function waitForFocusedSessionHydration({
  expectHistory,
  isCurrent,
  profile,
  requireActiveProfile,
  signal,
  storedSessionId,
  timeoutMs
}: {
  expectHistory: boolean
  isCurrent: () => boolean
  profile: string
  requireActiveProfile: boolean
  signal?: AbortSignal
  storedSessionId: string
  timeoutMs: number
}): Promise<void> {
  signal?.throwIfAborted()

  return new Promise((resolve, reject) => {
    let settled = false
    const unbinds: Array<() => void> = []
    let timer: number | undefined

    const finish = (error?: unknown) => {
      if (settled) {
        return
      }
      settled = true

      if (timer !== undefined) {
        window.clearTimeout(timer)
      }

      for (const unbind of unbinds) {
        unbind()
      }

      if (error !== undefined) {
        reject(error)
      } else {
        resolve()
      }
    }

    const abort = () => finish(signal?.reason)

    const check = () => {
      if (!isCurrent()) {
        finish(new Error('Session open was superseded by a newer selection.'))

        return
      }

      const profileMatches = !requireActiveProfile || normalizeProfileKey($activeGatewayProfile.get()) === profile
      const mainMatches = $selectedStoredSessionId.get() === storedSessionId
      const storedTile = $sessionTiles.get().find(tile => tile.storedSessionId === storedSessionId)
      const tileMatches = $focusedStoredSessionId.get() === storedSessionId || Boolean(storedTile)
      const focusedTileMatches = $focusedStoredSessionId.get() === storedSessionId
      const tileRuntimeId = focusedTileMatches ? $focusedRuntimeId.get() : (storedTile?.runtimeId ?? null)

      const tileState = focusedTileMatches
        ? $focusedSessionState.get()
        : tileRuntimeId
          ? $sessionStates.get()[tileRuntimeId]
          : undefined

      const runtimeReady = mainMatches ? Boolean($activeSessionId.get()) : tileMatches ? Boolean(tileRuntimeId) : false
      const historyPainted = mainMatches
        ? Boolean($messages.get().length)
        : tileMatches
          ? Boolean(tileState?.messages.length)
          : false
      // Persisted history is enough for paint-first opens; expected-empty chats
      // still require a runtime. A shared-remote profile gate may never match.
      const hydrated = expectHistory ? historyPainted : runtimeReady

      if ((mainMatches || tileMatches) && hydrated) {
        if (profileMatches) {
          finish()

          return
        }

        if (expectHistory && historyPainted) {
          beginHydrationBackgroundSync(profile, signal)
          finish()
        }
      }
    }

    unbinds.push($activeGatewayProfile.listen(check))
    unbinds.push($connection.listen(check))
    unbinds.push($selectedStoredSessionId.listen(check))
    unbinds.push($activeSessionId.listen(check))
    unbinds.push($messages.listen(check))
    unbinds.push($focusedStoredSessionId.listen(check))
    unbinds.push($focusedRuntimeId.listen(check))
    unbinds.push($focusedSessionState.listen(check))
    unbinds.push($sessionTiles.listen(check))
    unbinds.push($sessionStates.listen(check))
    unbinds.push($workspaceMode.listen(check))
    unbinds.push($workspaceOwnerKey.listen(check))
    signal?.addEventListener('abort', abort, { once: true })
    unbinds.push(() => signal?.removeEventListener('abort', abort))
    timer = window.setTimeout(() => {
      finish(new Error(`Timed out loading ${profile}'s session history.`))
    }, timeoutMs)
    check()
  })
}

/** Activation and hydration deliberately get separate full budgets. A timed-out
 * or aborted warmup may finish, but cannot authorize its caller to navigate. */
export async function awaitProfileActivation(
  dial: () => Promise<void>,
  targetProfile: string,
  timeoutMs: number,
  signal?: AbortSignal
): Promise<void> {
  let timer: number | undefined

  try {
    await awaitSessionOpenWork(
      () =>
        Promise.race([
          dial(),
          new Promise<never>((_resolve, reject) => {
            timer = window.setTimeout(
              () => reject(new Error(`Timed out loading ${targetProfile}'s session history.`)),
              timeoutMs
            )
          })
        ]),
      signal
    )
  } finally {
    if (timer !== undefined) {
      window.clearTimeout(timer)
    }
  }
}
