import { atom } from 'nanostores'

// The `agents.new` shortcut belongs to the Agents page: it opens that page's New agent dialog and
// declines everywhere else, so the same chord stays "new chat" outside it. The page's button hosts
// the request; with several mounted (panel row and narrow header) only the newest answers.

let hosts: symbol[] = []

/** Bumped with the host that should answer it. */
export const $newAgentRequest = atom<null | { host: symbol; seq: number }>(null)

/** Mount a New agent button as a host for the shortcut; returns the unmount. */
export function hostNewAgentRequests(host: symbol): () => void {
  hosts = [...hosts, host]

  return () => {
    hosts = hosts.filter(item => item !== host)
  }
}

/** The shortcut's handler: `false` hands the chord on when no New agent button is showing. */
export function requestNewAgent(): boolean {
  const host = hosts.at(-1)

  if (!host) {
    return false
  }

  $newAgentRequest.set({ host, seq: ($newAgentRequest.get()?.seq ?? 0) + 1 })

  return true
}
