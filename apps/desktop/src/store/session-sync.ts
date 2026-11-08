// Cross-window session-list sync. Each desktop window is its own renderer
// process with its own gateway socket and session store, so a mutation in one
// (e.g. a new chat started in the compact pop-out) never reaches another
// window. This bus pings every window to re-pull the shared session list; the
// data already lives in the backend, the other window just doesn't know to look.
const CHANNEL = 'hermes:sessions'

const channel = typeof BroadcastChannel === 'undefined' ? null : new BroadcastChannel(CHANNEL)

interface SessionRestore {
  type: 'restored'
  sessionId: string
  connectionId: string | null
  profile: string
}

const restoreVersions = new Map<string, number>()
const restoreKey = (connectionId: string | null, profile: string, id: string) => JSON.stringify([connectionId, profile, id])

function receiveRestore(restore: SessionRestore): void {
  // Legacy profile-only owners share no exact connection identity. Exact
  // owners observe only their own connection's restore version.
  for (const connection of new Set([restore.connectionId, null])) {
    const key = restoreKey(connection, restore.profile, restore.sessionId)
    restoreVersions.set(key, (restoreVersions.get(key) ?? 0) + 1)
  }
}

channel?.addEventListener('message', event => {
  const restore = event.data as Partial<SessionRestore> | null

  if (restore?.type === 'restored' && typeof restore.sessionId === 'string' &&
    typeof restore.profile === 'string' && (restore.connectionId === null || typeof restore.connectionId === 'string')) {
    receiveRestore(restore as SessionRestore)
  }
})

/** Fence an asynchronous closeout against an explicit restore in any window. */
export function captureSessionRestoreFence(connectionId: string | null, profile: string, ids: string[]) {
  const versions = ids.map(id => {
    const key = restoreKey(connectionId, profile, id)

    return { key, version: restoreVersions.get(key) ?? 0 }
  })

  const isCurrent = () => versions.every(({ key, version }) => (restoreVersions.get(key) ?? 0) === version)

  return {
    isCurrent,
    assertCurrent: () => {
      if (!isCurrent()) {
        throw new Error('Conversation was restored; start a new closeout action to preserve later work')
      }
    }
  }
}

/** The shared restore mutation calls this before sending and after acknowledgement. */
export function broadcastSessionRestored(sessionId: string, connectionId: string | null, profile: string): void {
  const restore: SessionRestore = { type: 'restored', sessionId, connectionId, profile }
  receiveRestore(restore)
  channel?.postMessage(restore)
}

// A window that mutated the session list (created / titled a chat) tells the
// others to refresh. A BroadcastChannel never delivers to its own poster, so the
// caller refreshes locally as it already does.
export function broadcastSessionsChanged(): void {
  channel?.postMessage(1)
}

export function onSessionsChanged(handler: () => void): () => void {
  if (!channel) {
    return () => {}
  }

  channel.addEventListener('message', handler)

  return () => channel.removeEventListener('message', handler)
}
