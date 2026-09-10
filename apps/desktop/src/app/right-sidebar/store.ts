import { atom } from 'nanostores'

import { persistBoolean, storedBoolean } from '@/lib/storage'

const TAKEOVER_KEY = 'hermes.desktop.terminalTakeover'

export const $terminalTakeover = atom(storedBoolean(TAKEOVER_KEY, false))

$terminalTakeover.subscribe(active => persistBoolean(TAKEOVER_KEY, active))

export const setTerminalTakeover = (active: boolean) => $terminalTakeover.set(active)

/** A command queued to run in the embedded terminal. The terminal pane flushes
 *  (and clears) it once its session is live, so a value set before the pane
 *  mounts still runs. Cleared after flush so a later remount can't replay it. */
export const $terminalInjection = atom<null | string>(null)

export interface ChatTerminalRunRequest {
  command: string
  terminalId: string
}

/** One-shot, user-approved chat command bound to one freshly-created user terminal.
 * Unlike $terminalInjection this must never float to whichever shell happens to be active. */
export const $chatTerminalRunRequest = atom<ChatTerminalRunRequest | null>(null)

export function takeChatTerminalRunRequest(terminalId: string): string | null {
  const pending = $chatTerminalRunRequest.get()

  if (!pending || pending.terminalId !== terminalId) {
    return null
  }

  // Clear before PTY write: a failed write is safer as a dropped command than a stale replay.
  $chatTerminalRunRequest.set(null)

  return pending.command
}

export function cancelChatTerminalRunRequest(terminalId: string): void {
  if ($chatTerminalRunRequest.get()?.terminalId === terminalId) {
    $chatTerminalRunRequest.set(null)
  }
}

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
