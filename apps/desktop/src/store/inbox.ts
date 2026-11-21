/**
 * Inbox: finished scheduled-job (cron) runs, newest first, with a backend read
 * state. The gateway is the authority (`inbox.list` / `inbox.read`, profile
 * scoped); these atoms are a cache of the ACTIVE profile's inbox on the active
 * connection.
 *
 * Read state is the backend session-unread watermark, so opening the run
 * anywhere (sidebar, another surface) clears it too; the next refresh shows
 * that. `markInboxItemRead` is the explicit write from the Inbox view:
 * optimistic, rolled back on failure, and a list page that lands mid-write
 * cannot repaint the old value over it.
 *
 * New-result notifications fire only for runs that FINISH after the first list
 * in a scope (the baseline): a run first seen while still running notifies
 * when it completes; nothing replays on launch, reconnect or profile switch.
 */
import { atom } from 'nanostores'

import { INBOX_ROUTE } from '@/app/routes'
import { translateNow } from '@/i18n'
import { isMissingRpcMethod } from '@/lib/gateway-rpc'
import { $activeConnectionId } from '@/store/connections'
import { requestGatewayForAgent } from '@/store/gateway'
import { $changeEventsAvailable, $sessionsChangeTick } from '@/store/live-sync'
import { dispatchNativeNotification } from '@/store/native-notifications'
import { notify } from '@/store/notifications'
import { $activeGatewayProfile } from '@/store/profile'
import { $gatewayState } from '@/store/session'

export interface InboxItem {
  session_id: string
  job_id: string
  job_name: string
  /** Unix seconds. */
  delivered_at: number
  preview: string
  unread: boolean
  running: boolean
}

interface InboxListResponse {
  items: InboxItem[]
  unread_count: number
}

interface InboxReadResponse {
  session_id: string
  unread: boolean
}

// The whole server window: `inbox.list` reads only the newest 200
// runs (`LIST_MAX`, gateway/session_inbox.py) and counts `unread_count` over that same
// window, so asking for all of it keeps every counted unread run listed.
const INBOX_LIST_LIMIT = 200
// A cron run's finish moves state.db several times in a row; one list read
// after the burst settles is enough.
const TICK_COALESCE_MS = 1_500
// Older backends broadcast no change events: poll slowly instead.
const BACKSTOP_POLL_MS = 60_000

export const $inboxItems = atom<InboxItem[]>([])
export const $inboxUnreadCount = atom(0)
/** False until a backend answers `inbox.list`; false again on -32601 (older
 *  backend). Hides the sidebar entry rather than showing a dead page. */
export const $inboxAvailable = atom(false)
/** True once the current scope's first list landed (the page's loading gate). */
export const $inboxLoaded = atom(false)
/** True when the current scope's first list failed (nothing to show yet), so
 *  the page offers Retry instead of loading forever. Cleared by a landed list. */
export const $inboxLoadError = atom(false)

interface ScopeState {
  scope: string
  /** Runs already seen FINISHED in this scope. Null = no baseline yet. */
  settled: null | Set<string>
  /** session_id -> unread value we wrote and have not seen confirmed. */
  pendingWrites: Map<string, boolean>
}

let state: ScopeState = { scope: '', settled: null, pendingWrites: new Map() }
// Only one window per app raises new-result alerts; peers just mirror the list.
let announces = true
let inFlight: null | Promise<void> = null
let rerunRequested = false

interface InboxScope {
  connectionId: null | string
  profile: string
  scope: string
}

function currentScope(): InboxScope {
  const connectionId = $activeConnectionId.get() ?? null
  const profile = $activeGatewayProfile.get()

  return { connectionId, profile, scope: `${connectionId ?? ''}\u0000${profile}` }
}

/** A reply belongs on screen only while its scope is both the cached one and
 *  the one the window is on NOW (a switch may not have re-entered yet). */
function isLive(scope: string): boolean {
  return state.scope === scope && currentScope().scope === scope
}

function enterScope(scope: string): void {
  if (state.scope === scope) {
    return
  }

  state = { scope, settled: null, pendingWrites: new Map() }
  $inboxItems.set([])
  $inboxUnreadCount.set(0)
  $inboxLoaded.set(false)
  $inboxLoadError.set(false)
}

function announce(fresh: InboxItem[]): void {
  if (!announces || !fresh.length) {
    return
  }

  const [first] = fresh
  const single = fresh.length === 1
  const title = single ? first.job_name : translateNow('inbox.newResultsTitle', fresh.length)
  const body = single ? first.preview || translateNow('inbox.noPreview') : translateNow('inbox.newResultsBody')

  notify({ icon: 'inbox', kind: 'info', message: body, title })
  // Global: a cron result belongs to no open chat. Fires only while Hermes is
  // in the background; a click lands on the Inbox page via the activate path.
  dispatchNativeNotification({
    activate: INBOX_ROUTE,
    body,
    global: true,
    kind: 'inbox',
    tag: single ? `inbox:${first.session_id}` : 'inbox',
    title
  })
}

function commit(response: InboxListResponse): void {
  const items = (response.items ?? []).map(item => {
    const written = state.pendingWrites.get(item.session_id)

    if (written === undefined) {
      return item
    }

    if (written === item.unread) {
      state.pendingWrites.delete(item.session_id)
    }

    return { ...item, unread: written }
  })

  let unreadCount = response.unread_count ?? items.filter(item => item.unread).length

  for (const [id, written] of state.pendingWrites) {
    const listed = response.items.find(item => item.session_id === id)

    if (listed && listed.unread !== written) {
      unreadCount += written ? 1 : -1
    }
  }

  const finished = items.filter(item => !item.running)

  if (state.settled) {
    const settled = state.settled

    announce(finished.filter(item => item.unread && !settled.has(item.session_id)))
  }

  state.settled = new Set([...(state.settled ?? []), ...finished.map(item => item.session_id)])
  $inboxItems.set(items)
  $inboxUnreadCount.set(Math.max(0, unreadCount))
  $inboxAvailable.set(true)
  $inboxLoaded.set(true)
  $inboxLoadError.set(false)
}

async function fetchOnce(): Promise<void> {
  const { connectionId, profile, scope } = currentScope()

  enterScope(scope)

  try {
    const response = await requestGatewayForAgent<InboxListResponse>(connectionId, profile, 'inbox.list', {
      limit: INBOX_LIST_LIMIT
    })

    if (isLive(scope)) {
      commit(response)
    }
  } catch (error) {
    if (!isLive(scope)) {
      return
    }

    if (isMissingRpcMethod(error)) {
      $inboxAvailable.set(false)
      $inboxItems.set([])
      $inboxUnreadCount.set(0)
    } else if (!$inboxLoaded.get()) {
      $inboxLoadError.set(true)
    }
    // After a landed list a failure is transient: keep it; the next tick retries.
  }
}

/** Fetch the active scope's inbox. Coalesced: calls during an in-flight fetch
 *  collapse into ONE follow-up fetch, so a burst of change ticks costs two
 *  requests at most. */
export function refreshInbox(): Promise<void> {
  if (inFlight) {
    rerunRequested = true

    return inFlight
  }

  inFlight = (async () => {
    try {
      do {
        rerunRequested = false
        await fetchOnce()
      } while (rerunRequested)
    } finally {
      inFlight = null
    }
  })()

  return inFlight
}

function paintUnread(sessionId: string, unread: boolean): boolean {
  const items = $inboxItems.get()
  const index = items.findIndex(item => item.session_id === sessionId)

  if (index === -1 || items[index].unread === unread) {
    return false
  }

  $inboxItems.set(items.map((item, i) => (i === index ? { ...item, unread } : item)))
  $inboxUnreadCount.set(Math.max(0, $inboxUnreadCount.get() + (unread ? 1 : -1)))

  return true
}

/** Mark one run read (or unread): paint now, write `inbox.read`, roll back if
 *  the backend refuses. No-op when the row is unknown or already in that state. */
export async function markInboxItemRead(sessionId: string, read = true): Promise<void> {
  const { connectionId, profile, scope } = currentScope()

  if (state.scope !== scope || !paintUnread(sessionId, !read)) {
    return
  }

  state.pendingWrites.set(sessionId, !read)

  try {
    await requestGatewayForAgent<InboxReadResponse>(connectionId, profile, 'inbox.read', {
      read,
      session_id: sessionId
    })
  } catch (error) {
    if (!isLive(scope)) {
      return
    }

    state.pendingWrites.delete(sessionId)
    paintUnread(sessionId, read)
    throw error
  }
}

/** Keep the inbox live for the window's life: refetch on gateway open and on
 *  scope change, on each (coalesced) `sessions.changed` tick, and on a slow
 *  poll only while the backend cannot push change events. Call once at boot
 *  from each full-shell window; `announce` only in the one that owns alerts. */
export function watchInbox({ announce: raiseAlerts = true }: { announce?: boolean } = {}): () => void {
  announces = raiseAlerts
  let tickTimer: null | ReturnType<typeof setTimeout> = null
  let pollTimer: null | ReturnType<typeof setInterval> = null

  const open = () => $gatewayState.get() === 'open'

  const refreshIfOpen = () => {
    if (open()) {
      void refreshInbox()
    }
  }

  const syncPoll = () => {
    if (pollTimer !== null) {
      clearInterval(pollTimer)
      pollTimer = null
    }

    if (open() && !$changeEventsAvailable.get()) {
      pollTimer = setInterval(refreshIfOpen, BACKSTOP_POLL_MS)
    }
  }

  const onTick = () => {
    if (tickTimer === null) {
      tickTimer = setTimeout(() => {
        tickTimer = null
        refreshIfOpen()
      }, TICK_COALESCE_MS)
    }
  }

  const onScopeOrState = () => {
    // Drop the old scope's cache now, socket open or not: an in-flight reply
    // for it then fails the scope check instead of painting or announcing.
    enterScope(currentScope().scope)
    syncPoll()
    refreshIfOpen()
  }

  const unsubscribers = [
    $gatewayState.listen(onScopeOrState),
    $activeGatewayProfile.listen(onScopeOrState),
    $activeConnectionId.listen(onScopeOrState),
    $changeEventsAvailable.listen(syncPoll),
    $sessionsChangeTick.listen(onTick)
  ]

  onScopeOrState()

  return () => {
    unsubscribers.forEach(unsubscribe => unsubscribe())

    if (tickTimer !== null) {
      clearTimeout(tickTimer)
    }

    if (pollTimer !== null) {
      clearInterval(pollTimer)
    }
  }
}

/** @internal Tests: forget scope, baseline and in-flight bookkeeping. */
export function _resetInboxForTests(): void {
  state = { scope: '', settled: null, pendingWrites: new Map() }
  inFlight = null
  rerunRequested = false
  announces = true
  $inboxItems.set([])
  $inboxUnreadCount.set(0)
  $inboxAvailable.set(false)
  $inboxLoaded.set(false)
  $inboxLoadError.set(false)
}
