import { atom } from 'nanostores'

import type { ContextSuggestion } from '@/app/types'
import type { HermesConnection } from '@/global'
import type { ChatMessage } from '@/lib/chat-messages'
import { persistString, storedString } from '@/lib/storage'
import type { SessionInfo, UsageStats } from '@/types/hermes'

type Updater<T> = T | ((current: T) => T)

const WORKSPACE_CWD_KEY = 'hermes.desktop.workspace-cwd'

export const getRememberedWorkspaceCwd = (): string => storedString(WORKSPACE_CWD_KEY)?.trim() || ''

interface AppAtom<T> {
  get: () => T
  set: (value: T) => void
}

function updateAtom<T>(store: AppAtom<T>, next: Updater<T>) {
  store.set(typeof next === 'function' ? (next as (current: T) => T)(store.get()) : next)
}

/** Durable id for pinning. Auto-compression rotates a conversation's session
 *  id (root -> continuation tip), so pins keyed on the live id evaporate. The
 *  lineage root is stable across every compression, so we pin on that. */
export const sessionPinId = (session: Pick<SessionInfo, '_lineage_root_id' | 'id'>): string =>
  session._lineage_root_id ?? session.id

/** Merge a fresh server session page into the in-memory list, keeping any
 *  row the server omitted that we still want visible — both still-"working"
 *  sessions and pinned sessions.
 *
 *  Two reasons the server drops a row we must keep:
 *
 *  1. A brand-new session's first user message isn't flushed to the SessionDB
 *     until its turn is persisted, so `listSessions(min_messages=1)` skips
 *     sessions that are mid-first-response. Because every `message.complete`
 *     triggers a full refresh, a hard replace makes concurrent new chats vanish
 *     the instant any one of them finishes.
 *  2. The sidebar lists only the most-recent page (`SIDEBAR_SESSIONS_PAGE_SIZE`)
 *     ordered by activity. A pinned conversation that hasn't been touched in a
 *     while falls off that page, so a hard replace silently evicts it from the
 *     in-memory list — and because the Pinned section resolves pins against
 *     that list, the pin "disappears until you refresh".
 *
 *  `keepIds` carries both the working set and the pinned set. Pins are stored
 *  on the durable lineage-root id (see {@link sessionPinId}), while the loaded
 *  row surfaces under its live compression tip, so we match a survivor by
 *  either its live `id` or its `_lineage_root_id`. Optimistic deletes/archives
 *  drop the row from `previous` (and unpin it), so a removed session can't be
 *  resurrected here. */
export function mergeSessionPage(
  previous: SessionInfo[],
  incoming: SessionInfo[],
  keepIds: Iterable<string>
): SessionInfo[] {
  const keep = keepIds instanceof Set ? keepIds : new Set(keepIds)

  if (keep.size === 0) {
    return incoming
  }

  const incomingIds = new Set(incoming.map(session => session.id))

  const survivors = previous.filter(
    session =>
      !incomingIds.has(session.id) &&
      (keep.has(session.id) || (session._lineage_root_id != null && keep.has(session._lineage_root_id)))
  )

  return survivors.length ? [...survivors, ...incoming] : incoming
}

export const $connection = atom<HermesConnection | null>(null)
export const $gatewayState = atom('idle')
export const $sessions = atom<SessionInfo[]>([])
export const $sessionsTotal = atom<number>(0)
export const $sessionsLoading = atom(true)
export const $workingSessionIds = atom<string[]>([])
export const $activeSessionId = atom<string | null>(null)
export const $selectedStoredSessionId = atom<string | null>(null)
export const $messages = atom<ChatMessage[]>([])
export const $freshDraftReady = atom(false)
export const $busy = atom(false)
export const $awaitingResponse = atom(false)
export const $currentModel = atom('')
export const $currentProvider = atom('')
export const $currentReasoningEffort = atom('')
export const $currentServiceTier = atom('')
export const $currentFastMode = atom(false)
// Effective approval-bypass state mirrored from the gateway (session.info).
// Persistence lives in the backend config (approvals.mode), so this is a plain
// reflection of the truth the gateway reports rather than its own store.
export const $yoloActive = atom(false)
export const $currentCwd = atom(getRememberedWorkspaceCwd())
export const $currentBranch = atom('')
export const $currentUsage = atom<UsageStats>({
  calls: 0,
  input: 0,
  output: 0,
  total: 0
})
export const $sessionStartedAt = atom<number | null>(null)
export const $turnStartedAt = atom<number | null>(null)
export const $introPersonality = atom('')
export const $currentPersonality = atom('')
export const $availablePersonalities = atom<string[]>([])
export const $introSeed = atom(0)
export const $contextSuggestions = atom<ContextSuggestion[]>([])
export const $modelPickerOpen = atom(false)

export const setConnection = (next: Updater<HermesConnection | null>) => updateAtom($connection, next)
export const setGatewayState = (next: Updater<string>) => updateAtom($gatewayState, next)
export const setSessions = (next: Updater<SessionInfo[]>) => updateAtom($sessions, next)
export const setSessionsTotal = (next: Updater<number>) => updateAtom($sessionsTotal, next)
export const setSessionsLoading = (next: Updater<boolean>) => updateAtom($sessionsLoading, next)
export const setWorkingSessionIds = (next: Updater<string[]>) => updateAtom($workingSessionIds, next)
export const setActiveSessionId = (next: Updater<string | null>) => updateAtom($activeSessionId, next)
export const setSelectedStoredSessionId = (next: Updater<string | null>) => updateAtom($selectedStoredSessionId, next)
export const setMessages = (next: Updater<ChatMessage[]>) => updateAtom($messages, next)
export const setFreshDraftReady = (next: Updater<boolean>) => updateAtom($freshDraftReady, next)
export const setBusy = (next: Updater<boolean>) => updateAtom($busy, next)
export const setAwaitingResponse = (next: Updater<boolean>) => updateAtom($awaitingResponse, next)
export const setCurrentModel = (next: Updater<string>) => updateAtom($currentModel, next)
export const setCurrentProvider = (next: Updater<string>) => updateAtom($currentProvider, next)
export const setCurrentReasoningEffort = (next: Updater<string>) => updateAtom($currentReasoningEffort, next)
export const setCurrentServiceTier = (next: Updater<string>) => updateAtom($currentServiceTier, next)
export const setCurrentFastMode = (next: Updater<boolean>) => updateAtom($currentFastMode, next)
export const setYoloActive = (next: Updater<boolean>) => updateAtom($yoloActive, next)

export const setCurrentCwd = (next: Updater<string>) => {
  updateAtom($currentCwd, next)
  // Keep localStorage in sync with the atom: a real folder is remembered, an
  // empty cwd clears the key (|| null → removeItem).
  persistString(WORKSPACE_CWD_KEY, $currentCwd.get().trim() || null)
}

export const setCurrentBranch = (next: Updater<string>) => updateAtom($currentBranch, next)
export const setCurrentUsage = (next: Updater<UsageStats>) => updateAtom($currentUsage, next)
export const setSessionStartedAt = (next: Updater<number | null>) => updateAtom($sessionStartedAt, next)
export const setTurnStartedAt = (next: Updater<number | null>) => updateAtom($turnStartedAt, next)
export const setIntroPersonality = (next: Updater<string>) => updateAtom($introPersonality, next)
export const setCurrentPersonality = (next: Updater<string>) => updateAtom($currentPersonality, next)
export const setAvailablePersonalities = (next: Updater<string[]>) => updateAtom($availablePersonalities, next)
export const setIntroSeed = (next: Updater<number>) => updateAtom($introSeed, next)
export const setContextSuggestions = (next: Updater<ContextSuggestion[]>) => updateAtom($contextSuggestions, next)
export const setModelPickerOpen = (next: Updater<boolean>) => updateAtom($modelPickerOpen, next)

// Watchdog tracking — when does a "working" session count as stuck?
// Long-running tool calls (LLM inference, long shell commands, web fetches)
// can take a few minutes legitimately. We allow 8 minutes of complete
// silence on the stream before clearing the working flag; in practice this
// catches gateway hangs and dropped streams without false-positive-clearing
// real long turns.
const SESSION_WATCHDOG_TIMEOUT_MS = 8 * 60 * 1000
const sessionWatchdogTimers = new Map<string, ReturnType<typeof setTimeout>>()

function armSessionWatchdog(sessionId: string) {
  const existing = sessionWatchdogTimers.get(sessionId)

  if (existing) {
    clearTimeout(existing)
  }

  const timer = setTimeout(() => {
    sessionWatchdogTimers.delete(sessionId)

    // Re-check the latest state at fire-time. If the user already navigated
    // away or the session genuinely finished, the timer is a no-op.
    if ($workingSessionIds.get().includes(sessionId)) {
      setWorkingSessionIds(current => current.filter(id => id !== sessionId))
    }
  }, SESSION_WATCHDOG_TIMEOUT_MS)

  sessionWatchdogTimers.set(sessionId, timer)
}

function clearSessionWatchdog(sessionId: string) {
  const existing = sessionWatchdogTimers.get(sessionId)

  if (existing) {
    clearTimeout(existing)
    sessionWatchdogTimers.delete(sessionId)
  }
}

/** Call when a streaming event for a session lands. Refreshes the watchdog
 *  so the session keeps its "working" status as long as data keeps coming. */
export function noteSessionActivity(sessionId: string | null | undefined) {
  if (!sessionId || !$workingSessionIds.get().includes(sessionId)) {
    return
  }

  armSessionWatchdog(sessionId)
}

// Toggle an id's membership in a string-set atom, no-op when unchanged (keeps
// the same array reference so subscribers don't churn).
const toggleMembership = (set: (next: Updater<string[]>) => void, id: string, on: boolean) =>
  set(current => {
    const present = current.includes(id)

    if (on) {
      return present ? current : [...current, id]
    }

    return present ? current.filter(x => x !== id) : current
  })

// Stored session ids with a blocking prompt (clarify) waiting on the user.
// Separate from $workingSessionIds: a session can be "working" (turn running)
// AND need input. The sidebar row reads this for a persistent indicator that,
// unlike a toast, survives window blur / alt-tab.
export const $attentionSessionIds = atom<string[]>([])
export const setAttentionSessionIds = (next: Updater<string[]>) => updateAtom($attentionSessionIds, next)

export function setSessionAttention(sessionId: string | null | undefined, needsInput: boolean) {
  if (sessionId) {
    toggleMembership(setAttentionSessionIds, sessionId, needsInput)
  }
}

export function setSessionWorking(sessionId: string | null | undefined, working: boolean) {
  if (!sessionId) {
    return
  }

  toggleMembership(setWorkingSessionIds, sessionId, working)

  // Bookend the watchdog: arm on enter, disarm on leave. A later
  // noteSessionActivity() from a streaming event refreshes the timer.
  if (working) {
    armSessionWatchdog(sessionId)
  } else {
    clearSessionWatchdog(sessionId)
  }
}
