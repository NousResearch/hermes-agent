import { isPaneVisible } from '@/components/pane-shell/tree/store'
import type { StatusItemState } from '@/store/composer-status'

import { $activeTerminalId, $terminals, closeAgentTerminalByProc } from './terminals'

// Read-only agent tabs (one per `terminal(background=true)` process) close
// themselves a while after their process is SEEN finishing. The status stack
// clears in seconds; the tab keeps the full output a little longer so it can
// still be read, and a failure lingers longest. Closing drops only the view
// (closeAgentTerminalByProc): the process is never killed, and the status-stack
// row can reopen the tab.
export const AGENT_TAB_SUCCESS_GRACE_MS = 60_000
export const AGENT_TAB_FAILURE_GRACE_MS = 5 * 60_000
// A tab the user is looking at never vanishes under them: re-check this often
// until it is no longer the active tab of a visible pane.
export const AGENT_TAB_ON_SCREEN_RECHECK_MS = 10_000

// procId → last state actually observed in a process snapshot. Kept after the
// row leaves the snapshot: a process last seen running that simply disappears
// (its session stopped polling) is unknown, not finished, so it never closes.
const lastSeen = new Map<string, StatusItemState>()
const timers = new Map<string, ReturnType<typeof setTimeout>>()

function onScreen(procId: string): boolean {
  const activeId = $activeTerminalId.get()
  const active = $terminals.get().find(term => term.id === activeId)

  return active?.kind === 'agent' && active.procId === procId && isPaneVisible('terminal')
}

function cancel(procId: string): void {
  clearTimeout(timers.get(procId))
  timers.delete(procId)
}

function agentTabId(procId: string): string | undefined {
  return $terminals.get().find(term => term.kind === 'agent' && term.procId === procId)?.id
}

// The close is bound to the tab that was open when it was armed. Dismissing
// that tab and reopening it from the status row creates a NEW terminal id: the
// user asked for it back, so the old timer must not close the replacement.
function schedule(procId: string, delayMs: number, tabId: string | undefined = agentTabId(procId)): void {
  cancel(procId)
  timers.set(
    procId,
    setTimeout(() => {
      timers.delete(procId)

      if (lastSeen.get(procId) === 'running') {
        return
      }

      if (tabId !== undefined && agentTabId(procId) !== tabId) {
        return
      }

      if (onScreen(procId)) {
        schedule(procId, AGENT_TAB_ON_SCREEN_RECHECK_MS, tabId)

        return
      }

      closeAgentTerminalByProc(procId)
    }, delayMs)
  )
}

/** Feed one observed background-process state. Arms the close on the
 *  transition into a finished state (once per finish: a tab the user reopens
 *  afterwards stays until they close it); a running state cancels it. */
export function observeAgentProcessState(procId: string, state: StatusItemState): void {
  const previous = lastSeen.get(procId)
  lastSeen.set(procId, state)

  if (state === 'running') {
    cancel(procId)

    return
  }

  if (previous !== state) {
    schedule(procId, state === 'failed' ? AGENT_TAB_FAILURE_GRACE_MS : AGENT_TAB_SUCCESS_GRACE_MS)
  }
}
