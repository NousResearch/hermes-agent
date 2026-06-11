import { atom } from 'nanostores'

import type { ContextSuggestion } from '@/app/types'
import type { HermesConnection } from '@/global'
import type { ChatMessage } from '@/lib/chat-messages'
import { persistString, storedString } from '@/lib/storage'
import type { SessionInfo, SessionPresenceRecord, UsageStats } from '@/types/hermes'

type Updater<T> = T | ((current: T) => T)

const WORKSPACE_CWD_KEY = 'hermes.desktop.workspace-cwd'

// Cached copy of Settings → Sessions → Default project directory. The main
// process persists this in project-dir.json, but the renderer must also honor it
// when seeding $currentCwd — otherwise PR #37586's sticky localStorage home dir
// wins and new sessions ignore the user's explicit picker choice.
let configuredDefaultProjectDir = ''

export const getRememberedWorkspaceCwd = (): string => storedString(WORKSPACE_CWD_KEY)?.trim() || ''

export const getConfiguredDefaultProjectDir = (): string => configuredDefaultProjectDir

export async function syncConfiguredDefaultProjectDir(): Promise<string> {
  const settings = window.hermesDesktop?.settings?.getDefaultProjectDir

  if (!settings) {
    configuredDefaultProjectDir = ''

    return ''
  }

  const { dir } = await settings()
  configuredDefaultProjectDir = dir?.trim() || ''

  return configuredDefaultProjectDir
}

/** Align the renderer workspace with the main-process default (home dir when
 *  packaged, optional Settings override). Clears stale install-dir paths that
 *  PR #37586's localStorage stickiness can preserve across the #37536 fix. */
export async function ensureDefaultWorkspaceCwd(): Promise<void> {
  const sanitize = window.hermesDesktop?.sanitizeWorkspaceCwd

  if (!sanitize) {
    return
  }

  await syncConfiguredDefaultProjectDir()
  const configured = getConfiguredDefaultProjectDir()

  const seedLiveCwd = (cwd: string) => {
    if (cwd && !$activeSessionId.get()) {
      setCurrentCwd(cwd)
    }
  }

  if (configured) {
    const { cwd } = await sanitize(configured)
    seedLiveCwd(cwd)

    return
  }

  const { cwd } = await sanitize(getRememberedWorkspaceCwd())
  seedLiveCwd(cwd)
}

export function applyConfiguredDefaultProjectDir(dir: null | string | undefined): void {
  configuredDefaultProjectDir = dir?.trim() || ''

  // Cache only — new chats read this via workspaceCwdForNewSession(). Do not
  // rewrite the live workspace (or localStorage) while a session is active.
  if (configuredDefaultProjectDir && !$activeSessionId.get()) {
    setCurrentCwd(configuredDefaultProjectDir)
  }
}

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

export const sessionAliasIds = (session: Pick<SessionInfo, '_lineage_ids' | '_lineage_root_id' | 'id'>): string[] => {
  const aliases = [session.id, session._lineage_root_id, ...(session._lineage_ids ?? [])].filter(
    (id): id is string => typeof id === 'string' && id.length > 0
  )

  return [...new Set(aliases)]
}

function dedupeSessionsByAlias(sessions: SessionInfo[]): SessionInfo[] {
  const aliasIndex = new Map<string, number>()
  const deduped: SessionInfo[] = []

  for (const session of sessions) {
    const aliases = sessionAliasIds(session)
    const existingIndex = aliases.map(id => aliasIndex.get(id)).find(index => index !== undefined)

    if (existingIndex !== undefined) {
      const current = deduped[existingIndex]

      if (
        aliases.length > sessionAliasIds(current).length ||
        (aliases.length === sessionAliasIds(current).length && session.last_active > current.last_active)
      ) {
        deduped[existingIndex] = session
      }

      for (const id of aliases) {
        aliasIndex.set(id, existingIndex)
      }

      continue
    }

    for (const id of aliases) {
      aliasIndex.set(id, deduped.length)
    }

    deduped.push(session)
  }

  return deduped.length === sessions.length ? sessions : deduped
}

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
 *  its live `id`, `_lineage_root_id`, or any historical `_lineage_ids` alias.
 *  Optimistic deletes/archives
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

  const dedupedIncoming = dedupeSessionsByAlias(incoming)
  const incomingAliases = new Set(dedupedIncoming.flatMap(sessionAliasIds))

  const survivors = previous.filter(
    session =>
      !sessionAliasIds(session).some(id => incomingAliases.has(id)) &&
      sessionAliasIds(session).some(id => keep.has(id))
  )

  return survivors.length ? [...survivors, ...dedupedIncoming] : dedupedIncoming
}

export function reconcileLiveSessionKey(previousSessionId: string | null | undefined, liveSessionId: string) {
  const previousId = previousSessionId?.trim() || ''
  const nextId = liveSessionId.trim()

  if (!previousId || !nextId || previousId === nextId) {
    return
  }

  setSessions(current => {
    const existingLiveIndex = current.findIndex(session => sessionAliasIds(session).includes(nextId))
    const previousIndex = current.findIndex(session => sessionAliasIds(session).includes(previousId))

    if (existingLiveIndex !== -1) {
      if (previousIndex === -1 || previousIndex === existingLiveIndex) {
        return current
      }

      return current.filter((_, index) => index !== previousIndex)
    }

    if (previousIndex === -1) {
      return current
    }

    const previous = current[previousIndex]
    const lineageIds = [...new Set([...(previous._lineage_ids ?? []), previous.id, nextId].filter(Boolean))]

    const next: SessionInfo = {
      ...previous,
      id: nextId,
      _lineage_root_id: previous._lineage_root_id ?? previous.id,
      _lineage_ids: lineageIds,
      ended_at: null,
      is_active: true,
      last_active: Math.max(previous.last_active ?? 0, Date.now() / 1000)
    }

    const nextSessions = current.slice()
    nextSessions[previousIndex] = next

    return nextSessions
  })
}

export const $connection = atom<HermesConnection | null>(null)
export const $gatewayState = atom('idle')
export const $sessions = atom<SessionInfo[]>([])
export const $sessionsTotal = atom<number>(0)
// Cron-job sessions (source === 'cron') are fetched as their own list so the
// scheduler's always-newest sessions never crowd recents out of the page
// budget. Powers the collapsed "Cron jobs" sidebar section.
export const $cronSessions = atom<SessionInfo[]>([])
// Max cron sessions fetched for the sidebar section (single bounded page). When
// the fetch returns exactly this many rows we know more exist, so the section
// badge renders "N+". Lives here so the controller (fetch) and sidebar (badge)
// share one source of truth without a circular import.
export const CRON_SECTION_LIMIT = 50
// Messaging-platform sessions (telegram/discord/...) are fetched as their own
// slice — separate from local recents — so each platform renders a
// self-managed sidebar section and never interleaves with (or buries) local
// chats in the recents page. One combined fetch seeds every platform; a
// platform that exceeds this cap gets its own per-platform "load more".
export const $messagingSessions = atom<SessionInfo[]>([])
export const MESSAGING_SECTION_LIMIT = 100
// Exact per-platform conversation totals, keyed by source id. Empty until a
// per-platform "load more" fetch resolves it (the combined seed fetch only
// knows the aggregate), so sections fall back to their loaded count.
export const $messagingPlatformTotals = atom<Record<string, number>>({})
// True when the combined seed fetch hit MESSAGING_SECTION_LIMIT, so at least
// one platform may have more rows on disk than were loaded.
export const $messagingTruncated = atom<boolean>(false)
// Listable conversation count per profile (children excluded), keyed by profile
// name. Lets the sidebar scope its "Load more" footer to the active profile so a
// huge default profile doesn't keep "Load more" visible while browsing a small
// one. Empty for single-profile users (fall back to $sessionsTotal).
export const $sessionProfileTotals = atom<Record<string, number>>({})
// Archived conversations, fetched as their own slice (archived='only') for the
// sidebar's collapsed Archived section. Rows load lazily on first expand; the
// boot refresh only resolves the TOTAL (cheap limit-1 probe) so the section
// header can show an honest count without paying for rows nobody opened.
export const $archivedSessions = atom<SessionInfo[]>([])
export const $archivedSessionsTotal = atom<number>(0)
export const $archivedSessionsLoading = atom(false)
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
export const DEFAULT_DESKTOP_YOLO_ACTIVE = true
// Desktop new-chat default. The backend approval bypass remains session-scoped;
// this preference decides whether desktop applies that bypass to new sessions.
export const $desktopYoloDefault = atom(DEFAULT_DESKTOP_YOLO_ACTIVE)
// Effective approval-bypass state mirrored from the gateway (session.info).
// Persistence lives in the backend config (approvals.mode), so this is a plain
// reflection of the truth the gateway reports rather than its own store.
export const $yoloActive = atom(false)
// Live sessions discovered across devices/clients (session.presence_list).
export const $sessionPresence = atom<SessionPresenceRecord[]>([])
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
// Transient gateway lifecycle status for the ACTIVE session (auto-compression
// progress, background-process notices). Mirrors `status.update` events and is
// cleared by the next stream activity, so it only shows while nothing else
// (deltas, tool events) is moving.
export interface SessionActivityStatus {
  kind: string
  text: string
}
export const $sessionActivityStatus = atom<SessionActivityStatus | null>(null)
// THIS device's resolved name (config → MeshBoard → Tailscale → hostname),
// captured from the FIRST gateway.ready frame — the primary local gateway
// connects at boot before any remote backend can exist. Used as the
// sender_device on prompts sent to REMOTE gateways (channels Phase 2b).
export const $localDeviceName = atom('')

// One viewer device in a session's channel roster (deduped, with a live-client
// count when the same device watches from more than one window).
export interface SessionParticipant {
  device: string
  count: number
}

// Channel presence (channels Phase 3): who is currently viewing each session,
// keyed by session id. Fed by `session.participants` gateway events; the header
// renders co-viewer chips for the active session (filtering out THIS device).
// Empty/solo-local sessions simply hold no entry, so the chip row is absent
// with no mesh/tailnet involved.
export const $sessionParticipants = atom<Record<string, SessionParticipant[]>>({})

export const setConnection = (next: Updater<HermesConnection | null>) => updateAtom($connection, next)
export const setGatewayState = (next: Updater<string>) => updateAtom($gatewayState, next)
export const setSessions = (next: Updater<SessionInfo[]>) => updateAtom($sessions, next)
export const setSessionsTotal = (next: Updater<number>) => updateAtom($sessionsTotal, next)
export const setCronSessions = (next: Updater<SessionInfo[]>) => updateAtom($cronSessions, next)
export const setMessagingSessions = (next: Updater<SessionInfo[]>) => updateAtom($messagingSessions, next)
export const setMessagingPlatformTotals = (next: Updater<Record<string, number>>) =>
  updateAtom($messagingPlatformTotals, next)
export const setMessagingTruncated = (next: Updater<boolean>) => updateAtom($messagingTruncated, next)
export const setSessionProfileTotals = (next: Updater<Record<string, number>>) =>
  updateAtom($sessionProfileTotals, next)
export const setArchivedSessions = (next: Updater<SessionInfo[]>) => updateAtom($archivedSessions, next)
export const setArchivedSessionsTotal = (next: Updater<number>) => updateAtom($archivedSessionsTotal, next)
export const setArchivedSessionsLoading = (next: Updater<boolean>) => updateAtom($archivedSessionsLoading, next)

/** Shift one profile's listable-count by `delta`, clamped at zero. Only known
 *  keys move: when the aggregator hasn't reported a profile yet, the sidebar
 *  already falls back to the loaded row count, so inventing an entry here
 *  would replace an honest fallback with a guess. */
export const adjustSessionProfileTotal = (profileKey: string, delta: number) =>
  $sessionProfileTotals.set(
    profileKey in $sessionProfileTotals.get()
      ? {
          ...$sessionProfileTotals.get(),
          [profileKey]: Math.max(0, ($sessionProfileTotals.get()[profileKey] ?? 0) + delta)
        }
      : $sessionProfileTotals.get()
  )

/** Same contract as {@link adjustSessionProfileTotal} for a messaging
 *  platform's resolved total: adjust only once a per-platform fetch has
 *  established the real number. */
export const adjustMessagingPlatformTotal = (sourceId: string, delta: number) =>
  $messagingPlatformTotals.set(
    sourceId in $messagingPlatformTotals.get()
      ? {
          ...$messagingPlatformTotals.get(),
          [sourceId]: Math.max(0, ($messagingPlatformTotals.get()[sourceId] ?? 0) + delta)
        }
      : $messagingPlatformTotals.get()
  )
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
export const setDesktopYoloDefaultActive = (next: Updater<boolean>) => updateAtom($desktopYoloDefault, next)
export const setYoloActive = (next: Updater<boolean>) => updateAtom($yoloActive, next)
export const setSessionPresence = (next: Updater<SessionPresenceRecord[]>) => updateAtom($sessionPresence, next)

export const setCurrentCwd = (next: Updater<string>) => {
  updateAtom($currentCwd, next)
  // Keep localStorage in sync with the atom: a real folder is remembered, an
  // empty cwd clears the key (|| null → removeItem).
  persistString(WORKSPACE_CWD_KEY, $currentCwd.get().trim() || null)
}

/** Workspace for a brand-new chat. Explicit Settings override wins; otherwise
 *  fall back to the sticky last-used folder, then whatever is already live. */
export const workspaceCwdForNewSession = (): string =>
  getConfiguredDefaultProjectDir() || getRememberedWorkspaceCwd() || $currentCwd.get().trim()

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
export const setSessionActivityStatus = (next: Updater<SessionActivityStatus | null>) =>
  updateAtom($sessionActivityStatus, next)
export const setLocalDeviceName = (next: Updater<string>) => updateAtom($localDeviceName, next)

// Replace the channel roster for one session. An empty roster drops the key so
// the map doesn't accumulate stale entries as the user moves between sessions.
export const setSessionParticipants = (sessionId: string, participants: SessionParticipant[]) => {
  if (!sessionId) {
    return
  }

  const current = $sessionParticipants.get()

  if (participants.length === 0) {
    if (!(sessionId in current)) {
      return
    }

    const next = { ...current }
    delete next[sessionId]
    $sessionParticipants.set(next)

    return
  }

  $sessionParticipants.set({ ...current, [sessionId]: participants })
}

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

// A session's "working" flag clears the instant its turn ends, but the
// cross-profile aggregator (listSessions with min_messages=1) only sees the
// just-persisted first turn a beat later. The active chat is shielded from that
// race by sessionsToKeep(), but a brand-new session that finished *while you
// were viewing a different chat* is, at the next refresh, neither working,
// pinned, nor active — so mergeSessionPage() evicts it. Nothing re-fetches
// afterward, so it stays gone until the app restarts. (Repro: start a new chat,
// then click another session before the first reply lands.)
//
// To bridge that window we keep a session in the merge keep-set for a short
// grace period after its turn settles, giving the aggregator time to catch up.
// Entries auto-expire, so this never accumulates and can't resurrect a deleted
// session (mergeSessionPage only revives rows still present in the in-memory
// list, which optimistic delete/archive already drops).
const SESSION_SETTLE_GRACE_MS = 30 * 1000
const settledSessionExpiry = new Map<string, number>()

function markSessionSettled(sessionId: string) {
  settledSessionExpiry.set(sessionId, Date.now() + SESSION_SETTLE_GRACE_MS)
}

function clearSessionSettled(sessionId: string) {
  settledSessionExpiry.delete(sessionId)
}

/** Shield a row from the next refresh's page-merge eviction for the settle
 *  grace window. A just-restored (unarchived) conversation usually has old
 *  activity timestamps, so it sits outside the recency page the next refresh
 *  fetches — without a grace entry, mergeSessionPage() would evict the row the
 *  user just brought back while they're looking at it. */
export function shieldSessionFromMerge(sessionId: string) {
  if (sessionId) {
    markSessionSettled(sessionId)
  }
}

/** Stored ids of sessions whose turn ended within the grace window. Prunes
 *  expired entries as it reads, so it stays bounded without a timer. */
export function getRecentlySettledSessionIds(now: number = Date.now()): string[] {
  const live: string[] = []

  for (const [id, expiry] of settledSessionExpiry) {
    if (expiry > now) {
      live.push(id)
    } else {
      settledSessionExpiry.delete(id)
    }
  }

  return live
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

  const wasWorking = $workingSessionIds.get().includes(sessionId)

  toggleMembership(setWorkingSessionIds, sessionId, working)

  // Bookend the watchdog: arm on enter, disarm on leave. A later
  // noteSessionActivity() from a streaming event refreshes the timer.
  if (working) {
    clearSessionSettled(sessionId)
    armSessionWatchdog(sessionId)
  } else {
    clearSessionWatchdog(sessionId)

    // Only grant grace on a real working→idle transition (updateSessionState
    // re-asserts `false` on every state tick, which must not keep extending the
    // window). This keeps the just-finished session visible long enough for the
    // aggregator to return its now-persisted row.
    if (wasWorking) {
      markSessionSettled(sessionId)
    }
  }
}
