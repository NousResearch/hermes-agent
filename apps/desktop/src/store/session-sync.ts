// Cross-window session-list sync. Each desktop window is its own renderer
// process with its own gateway socket and session store, so a mutation in one
// (e.g. a new chat started in the compact pop-out) never reaches another
// window. This bus pings every window to re-pull the shared session list; the
// data already lives in the backend, the other window just doesn't know to look.
const CHANNEL = 'hermes:sessions'

const channel = typeof BroadcastChannel === 'undefined' ? null : new BroadcastChannel(CHANNEL)

interface SessionUnreadChangedMessage {
  sessionId: string
  type: 'session-unread-changed'
  unread: boolean
}

interface SessionsChangedMessage {
  type: 'sessions-changed'
}

type SessionSyncMessage = SessionUnreadChangedMessage | SessionsChangedMessage

function isSessionSyncMessage(value: unknown): value is SessionSyncMessage {
  if (!value || typeof value !== 'object' || !('type' in value)) {
    return false
  }

  const candidate = value as Record<string, unknown>

  if (candidate.type === 'sessions-changed') {
    return true
  }

  return (
    candidate.type === 'session-unread-changed' &&
    typeof candidate.sessionId === 'string' &&
    typeof candidate.unread === 'boolean'
  )
}

// A window that mutated the session list (created / titled a chat) tells the
// others to refresh. A BroadcastChannel never delivers to its own poster, so the
// caller refreshes locally as it already does.
export function broadcastSessionsChanged(): void {
  channel?.postMessage({ type: 'sessions-changed' } satisfies SessionsChangedMessage)
}

export function onSessionsChanged(handler: () => void): () => void {
  if (!channel) {
    return () => {}
  }

  const listener = (event: MessageEvent<unknown>) => {
    // Keep accepting the original numeric ping while older Desktop windows are
    // still open during an app update.
    if (event.data === 1 || (isSessionSyncMessage(event.data) && event.data.type === 'sessions-changed')) {
      handler()
    }
  }

  channel.addEventListener('message', listener)

  return () => channel.removeEventListener('message', listener)
}

export function broadcastSessionUnreadChanged(sessionId: string, unread: boolean): void {
  channel?.postMessage({ sessionId, type: 'session-unread-changed', unread } satisfies SessionUnreadChangedMessage)
}

export function onSessionUnreadChanged(handler: (sessionId: string, unread: boolean) => void): () => void {
  if (!channel) {
    return () => {}
  }

  const listener = (event: MessageEvent<unknown>) => {
    if (isSessionSyncMessage(event.data) && event.data.type === 'session-unread-changed') {
      handler(event.data.sessionId, event.data.unread)
    }
  }

  channel.addEventListener('message', listener)

  return () => channel.removeEventListener('message', listener)
}
