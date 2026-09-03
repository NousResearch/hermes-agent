/**
 * Mirror of "which chats are mid-turn" to the main process.
 *
 * The renderer is the only side that knows a turn is in flight, and the main
 * process is the only side that can intercept a quit. This module bridges the
 * two: it publishes a small summary on every membership change, and
 * `electron/quit-guard.ts` turns that into the confirmation dialog.
 *
 * Imported for its side effect from `main.tsx`, alongside `store/translucency`.
 */

import { computed } from 'nanostores'

import type { HermesActiveWork } from '@/global'
import { $sessions } from '@/store/session'
import { $sessionStates, $workingSessionIds } from '@/store/session-states'

const $activeWork = computed(
  [$workingSessionIds, $sessions, $sessionStates],
  (workingIds, sessions, states): HermesActiveWork => {
    // `busy` sessions are actively streaming output. But a turn also counts as
    // "in flight" from the moment the request is sent until the first assistant
    // token lands — i.e. `awaitingResponse && !busy` ("thinking", waiting on the
    // backend). Closing the window during that window would kill a turn the user
    // can see is still working, so include it. See #80241.
    const inFlight = new Set(workingIds)

    for (const [id, s] of Object.entries(states)) {
      if (s.awaitingResponse && !s.busy) {
        inFlight.add(id)
      }
    }

    const titleById = new Map(sessions.map(session => [session.id, session.title?.trim() ?? '']))

    return {
      count: inFlight.size,
      titles: [...inFlight].map(id => titleById.get(id) ?? '').filter(Boolean)
    }
  }
)

if (typeof window !== 'undefined') {
  // Expose the latest summary on window so the main process can read it
  // synchronously on quit (via webContents.executeJavaScript) — the async IPC
  // publish below can be missed if the bridge isn't ready at first emit or the
  // value doesn't change again before a quit.
  const publish = (work: HermesActiveWork) => {
    ;(window as unknown as { __hermesActiveWork?: HermesActiveWork }).__hermesActiveWork = work
  }

  publish($activeWork.get())

  // `$sessions` republishes on unrelated churn (previews, heartbeats), so only
  // send when the summary itself moved — this crosses a process boundary.
  let lastSent = ''

  $activeWork.subscribe(work => {
    publish(work)

    const next = JSON.stringify(work)

    if (next === lastSent) {
      return
    }

    lastSent = next
    window.hermesDesktop?.setActiveWork?.(work)
  })
}
