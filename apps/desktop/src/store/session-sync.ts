// Cross-window session-list sync. Each desktop window is its own renderer
// process with its own gateway socket and session store, so a mutation in one
// never reaches another unless it is mirrored over this bus.
const CHANNEL = 'hermes:sessions'

const channel = typeof BroadcastChannel === 'undefined' ? null : new BroadcastChannel(CHANNEL)

export interface SessionUnreadVersion {
  revision: number
  source: string
}

export interface SessionCompletionToken {
  epoch: SessionUnreadVersion
  generation: number
  id: string
}

export interface SessionUnreadEntry {
  acknowledged: boolean
  completion: SessionCompletionToken
  profile?: string
  sessionId: string
}

interface SessionUnreadChangedMessage {
  entry: SessionUnreadEntry
  type: 'session-unread-changed'
}

interface LegacySessionUnreadChangedMessage {
  revision?: number
  sessionId: string
  source?: string
  type: 'session-unread-changed'
  unread: boolean
}

interface SessionUnreadResetMessage extends SessionUnreadVersion {
  type: 'session-unread-reset'
}

interface SessionUnreadSnapshotRequestMessage {
  source: string
  type: 'session-unread-snapshot-request'
}

interface SessionUnreadSnapshotMessage {
  entries: SessionUnreadEntry[]
  reset: null | SessionUnreadVersion
  target: string
  type: 'session-unread-snapshot'
}

interface SessionsChangedMessage {
  type: 'sessions-changed'
}

type SessionSyncMessage =
  | LegacySessionUnreadChangedMessage
  | SessionUnreadChangedMessage
  | SessionUnreadResetMessage
  | SessionUnreadSnapshotMessage
  | SessionUnreadSnapshotRequestMessage
  | SessionsChangedMessage

const sourceId =
  typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`

const wallClockRevision = () =>
  Math.floor((typeof performance === 'undefined' ? Date.now() : performance.timeOrigin + performance.now()) * 1_000)

let logicalClock = wallClockRevision()

function isSessionUnreadVersion(value: unknown): value is SessionUnreadVersion {
  if (!value || typeof value !== 'object') {
    return false
  }

  const candidate = value as Record<string, unknown>

  return (
    typeof candidate.revision === 'number' &&
    Number.isFinite(candidate.revision) &&
    typeof candidate.source === 'string'
  )
}

function isSessionUnreadEntry(value: unknown): value is SessionUnreadEntry {
  if (!value || typeof value !== 'object') {
    return false
  }

  const candidate = value as Record<string, unknown>

  return (
    typeof candidate.sessionId === 'string' &&
    (candidate.profile === undefined || typeof candidate.profile === 'string') &&
    typeof candidate.acknowledged === 'boolean' &&
    isSessionCompletionToken(candidate.completion)
  )
}

function isSessionCompletionToken(value: unknown): value is SessionCompletionToken {
  if (!value || typeof value !== 'object') {
    return false
  }

  const candidate = value as Record<string, unknown>

  return (
    typeof candidate.id === 'string' &&
    typeof candidate.generation === 'number' &&
    Number.isFinite(candidate.generation) &&
    isSessionUnreadVersion(candidate.epoch)
  )
}

function isSessionSyncMessage(value: unknown): value is SessionSyncMessage {
  if (!value || typeof value !== 'object' || !('type' in value)) {
    return false
  }

  const candidate = value as Record<string, unknown>

  if (candidate.type === 'sessions-changed') {
    return true
  }

  if (candidate.type === 'session-unread-changed') {
    return (
      isSessionUnreadEntry(candidate.entry) ||
      (typeof candidate.sessionId === 'string' && typeof candidate.unread === 'boolean')
    )
  }

  if (candidate.type === 'session-unread-reset') {
    return isSessionUnreadVersion(candidate)
  }

  if (candidate.type === 'session-unread-snapshot-request') {
    return typeof candidate.source === 'string'
  }

  return (
    candidate.type === 'session-unread-snapshot' &&
    typeof candidate.target === 'string' &&
    Array.isArray(candidate.entries) &&
    candidate.entries.every(isSessionUnreadEntry) &&
    (candidate.reset === null || isSessionUnreadVersion(candidate.reset))
  )
}

function observeRevision(revision: number): void {
  logicalClock = Math.max(logicalClock, revision)
}

export function nextSessionUnreadVersion(): SessionUnreadVersion {
  logicalClock = Math.max(logicalClock + 1, wallClockRevision())

  return { revision: logicalClock, source: sourceId }
}

export function compareSessionUnreadVersions(left: SessionUnreadVersion, right: SessionUnreadVersion): number {
  if (left.revision !== right.revision) {
    return left.revision - right.revision
  }

  return left.source.localeCompare(right.source)
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

export function broadcastSessionUnreadChanged(entry: SessionUnreadEntry): void {
  channel?.postMessage({ entry, type: 'session-unread-changed' } satisfies SessionUnreadChangedMessage)
}

export function broadcastSessionUnreadReset(version: SessionUnreadVersion): void {
  channel?.postMessage({ ...version, type: 'session-unread-reset' } satisfies SessionUnreadResetMessage)
}

export function broadcastSessionUnreadSnapshot(
  target: string,
  entries: SessionUnreadEntry[],
  reset: null | SessionUnreadVersion
): void {
  channel?.postMessage({
    entries,
    reset,
    target,
    type: 'session-unread-snapshot'
  } satisfies SessionUnreadSnapshotMessage)
}

export function requestSessionUnreadSnapshot(): void {
  channel?.postMessage({
    source: sourceId,
    type: 'session-unread-snapshot-request'
  } satisfies SessionUnreadSnapshotRequestMessage)
}

interface SessionUnreadSyncHandlers {
  onChange: (entry: SessionUnreadEntry) => void
  onLegacyChange: (sessionId: string, unread: boolean, version: SessionUnreadVersion) => void
  onReset: (version: SessionUnreadVersion) => void
  onSnapshot: (entries: SessionUnreadEntry[], reset: null | SessionUnreadVersion) => void
  onSnapshotRequest: (source: string) => void
}

export function onSessionUnreadSync(handlers: SessionUnreadSyncHandlers): () => void {
  if (!channel) {
    return () => {}
  }

  const listener = (event: MessageEvent<unknown>) => {
    if (!isSessionSyncMessage(event.data)) {
      return
    }

    const message = event.data

    if (message.type === 'session-unread-changed') {
      if ('entry' in message && isSessionUnreadEntry(message.entry)) {
        observeRevision(message.entry.completion.epoch.revision)
        handlers.onChange(message.entry)
      } else if ('sessionId' in message) {
        const version = isSessionUnreadVersion(message) ? message : nextSessionUnreadVersion()

        observeRevision(version.revision)
        handlers.onLegacyChange(message.sessionId, message.unread, version)
      }
    } else if (message.type === 'session-unread-reset') {
      observeRevision(message.revision)
      handlers.onReset(message)
    } else if (message.type === 'session-unread-snapshot-request') {
      handlers.onSnapshotRequest(message.source)
    } else if (message.type === 'session-unread-snapshot' && message.target === sourceId) {
      message.entries.forEach(entry => observeRevision(entry.completion.epoch.revision))

      if (message.reset) {
        observeRevision(message.reset.revision)
      }

      handlers.onSnapshot(message.entries, message.reset)
    }
  }

  channel.addEventListener('message', listener)

  return () => channel.removeEventListener('message', listener)
}
