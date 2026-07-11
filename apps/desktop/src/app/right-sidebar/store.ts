import { atom, computed } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'
import { $utilityPreviewTabs, closeRightRailTab, openUtilityPreviewTab } from '@/store/preview'

const TAKEOVER_KEY = 'hermes.desktop.terminalTakeover'
const TERMINAL_TAB_ID = 'utility:terminal' as const

if (storedBoolean(TAKEOVER_KEY, false)) {
  openUtilityPreviewTab('terminal')
}

/** Compatibility name retained for terminal callers. "Takeover" now means the
 * peer Terminal utility tab is open; it no longer replaces another pane. */
export const $terminalTakeover = computed($utilityPreviewTabs, tabs => tabs.some(tab => tab.id === TERMINAL_TAB_ID))

$terminalTakeover.subscribe(active => persistBoolean(TAKEOVER_KEY, active))

export const setTerminalTakeover = (active: boolean) => {
  if (active) {
    openUtilityPreviewTab('terminal')
  } else {
    closeRightRailTab(TERMINAL_TAB_ID)
  }
}

/** A command queued to run in the embedded terminal. The terminal pane flushes
 *  (and clears) it once its session is live, so a value set before the pane
 *  mounts still runs. Cleared after flush so a later remount can't replay it. */
export const $terminalInjection = atom<null | string>(null)

/** Open the terminal pane and run a command in it. Used to disconnect external
 *  (CLI-managed) providers, which Hermes can't clear via the API — the user
 *  sees exactly what runs instead of Hermes silently deleting their creds. */
export const runInTerminal = (command: string) => {
  const trimmed = command.trim()

  if (!trimmed) {
    return
  }

  setTerminalTakeover(true)
  $terminalInjection.set(trimmed)
}
