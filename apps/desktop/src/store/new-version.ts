/**
 * A Webapp tab that was open when a new build was deployed (W1). It never reloads mid-task: the new
 * build loads at the next new chat when nothing is running (lib/deployed-build.ts has the rule), on
 * the next visit, or when the person picks "Refresh now".
 *
 * Detection reads the host's no-store index and compares its entry module with the one this page
 * booted from, only while the tab is visible. The desktop app updates through its own flow
 * (store/updates.ts) and never runs this.
 */

import { atom, computed } from 'nanostores'

import { type ApplyConditions, entryModuleOf, isNewerBuild, mayApplyOnNewChat } from '@/lib/deployed-build'

import { $workingSessionIds } from './session-states'
import { $voiceStreamPaintOwners } from './voice-stream-paint'

export interface DeployedBuildProbe {
  /** The entry module this page booted from. */
  running: null | string
  /** The entry module the host serves now; null when the index could not be read. */
  served: () => Promise<null | string>
}

/** Becomes true once a newer build is seen and stays so until the page reloads into it. */
export const $newVersionReady = atom(false)

/** "Later" put the chat's notice away for this page; the tray keeps the build waiting. */
const $newVersionNoticeAway = atom(false)

/** The chat says a newer build is waiting (E4), until Later. */
export const $newVersionNoticeShown = computed(
  [$newVersionReady, $newVersionNoticeAway],
  (ready, away) => ready && !away
)

export function putNewVersionNoticeAway(): void {
  $newVersionNoticeAway.set(true)
}

export const $applyConditions = computed(
  [$workingSessionIds, $voiceStreamPaintOwners],
  (working, voice): ApplyConditions => ({ workingChats: working.length, voiceActive: voice > 0 })
)

export async function checkForNewVersion(probe: DeployedBuildProbe): Promise<void> {
  if ($newVersionReady.get()) {
    return
  }

  if (isNewerBuild(probe.running, await probe.served())) {
    $newVersionReady.set(true)
  }
}

/** The page's own index, as the host serves it now. */
export function browserBuildProbe(): DeployedBuildProbe {
  return {
    running: document.querySelector('script[type="module"][src]')?.getAttribute('src') ?? null,
    served: async () => {
      try {
        const response = await fetch(new URL(window.location.pathname, window.location.origin), {
          cache: 'no-store',
          credentials: 'same-origin'
        })

        // A redirect (a sign-in wall) or an error page is not this app's index.
        return response.ok && !response.redirected ? entryModuleOf(await response.text()) : null
      } catch {
        // Offline or the host restarting: the next check asks again.
        return null
      }
    }
  }
}

const CHECK_EVERY_MS = 10 * 60_000
// Coming back to the tab checks at once, but tab-flicking does not refetch the index each time.
const MIN_GAP_MS = 60_000

/** Checks while the tab is visible: on an interval and on coming back to it. Returns the stop. */
export function watchForNewVersion(probe: DeployedBuildProbe = browserBuildProbe()): () => void {
  let last = 0

  const check = () => {
    if (document.visibilityState !== 'visible' || $newVersionReady.get() || Date.now() - last < MIN_GAP_MS) {
      return
    }

    last = Date.now()
    void checkForNewVersion(probe)
  }

  const timer = window.setInterval(check, CHECK_EVERY_MS)
  document.addEventListener('visibilitychange', check)
  check()

  return () => {
    window.clearInterval(timer)
    document.removeEventListener('visibilitychange', check)
  }
}

/** A new chat was opened: load the waiting build if that interrupts nothing. True when it reloads. */
export function applyNewVersionOnNewChat(reload: () => void = () => window.location.reload()): boolean {
  if (!$newVersionReady.get() || !mayApplyOnNewChat($applyConditions.get())) {
    return false
  }

  reload()

  return true
}
