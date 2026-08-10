import type { ConnectionState } from '@hermes/shared'
import { atom, computed } from 'nanostores'

import { lastVisibleMessageIsUser } from '@/app/chat/thread-loading'
import type { ContextSuggestion } from '@/app/types'
import type { HermesConnection } from '@/global'
import type { ChatMessage } from '@/lib/chat-messages'
import { persistBoolean, persistString, storedBoolean, storedString } from '@/lib/storage'
import { syncCronModelImpactConnection } from '@/store/cron-model-impact-scope'
import type { SessionInfo, UsageStats } from '@/types/hermes'

type Updater<T> = T | ((current: T) => T)
export type ComposerModelSource = '' | 'default' | 'manual'

const WORKSPACE_CWD_KEY = 'hermes.desktop.workspace-cwd'

// The composer's model/effort/fast is sticky UI state, NOT the profile default
// (that lives in Settings → Model). Persisting it in localStorage makes a pick
// follow across Cmd+N and app restarts instead of snapping back to the default.
// It's deliberately global (not per-profile): a profile switch force-reseeds to
// that profile's default, while within a profile new chats keep your last pick.
const COMPOSER_MODEL_KEY = 'hermes.desktop.composer.model'
const COMPOSER_PROVIDER_KEY = 'hermes.desktop.composer.provider'
const COMPOSER_MODEL_SOURCE_KEY = 'hermes.desktop.composer.model-source'
const COMPOSER_EFFORT_KEY = 'hermes.desktop.composer.reasoning-effort'
const COMPOSER_FAST_KEY = 'hermes.desktop.composer.fast'

// The last chat the user had open, so a relaunch lands back on it instead of an
// empty new-chat. Stored (not runtime) id — the route is keyed by stored id.
//
// Scoped per profile with an explicit namespace (`.profile.<encoded>`) and
// encodeURIComponent so a profile name carrying `/` or other reserved chars
// cannot collide or leak across keys. Legacy global (unsuffixed) keys are
// discarded on first read to prevent cross-profile bleed — ownership of the old
// global values is unknowable, and guessing the owning profile is exactly the
// cross-profile corruption this storage boundary prevents (#67709).
const LAST_SESSION_KEY = 'hermes.desktop.lastSessionId'
const LAST_ROUTE_KEY = 'hermes.desktop.lastRoute'

function profileNavigationKey(base: string, profile: string): string {
  const key = profile.trim() || 'default'

  return `${base}.profile.${encodeURIComponent(key)}`
}

// Discard legacy global keys once per tick. A module-level flag avoids
// redundant synchronous localStorage reads on every get/set call within
// the same synchronous block. The flag resets on cross-window `storage`
// events, which are the only way another window can recontaminate between
// ticks.
let legacyDiscardNeeded = true

if (typeof window !== 'undefined') {
  window.addEventListener('storage', e => {
    if (e.key === LAST_SESSION_KEY || e.key === LAST_ROUTE_KEY) {
      legacyDiscardNeeded = true
    }
  })
}

function discardLegacyRememberedNavigation(): void {
  if (!legacyDiscardNeeded) {
    return
  }

  legacyDiscardNeeded = false

  // Ownership of the old global values is unknowable. Never migrate them into
  // a profile: guessing is exactly the cross-profile corruption this storage
  // boundary prevents.
  if (storedString(LAST_SESSION_KEY) !== null) {
    persistString(LAST_SESSION_KEY, null)
  }

  if (storedString(LAST_ROUTE_KEY) !== null) {
    persistString(LAST_ROUTE_KEY, null)
  }
}

/** @internal Reset the legacy-discard flag for tests. */
export function _resetLegacyDiscardForTests(): void {
  legacyDiscardNeeded = true
}

export function getRememberedSessionId(profile: string): null | string {
  discardLegacyRememberedNavigation()

  return storedString(profileNavigationKey(LAST_SESSION_KEY, profile))
}

export function setRememberedSessionId(id: null | string, profile: string): void {
  discardLegacyRememberedNavigation()
  persistString(profileNavigationKey(LAST_SESSION_KEY, profile), id)
}

export function sessionBelongsToProfile(
  sessions: readonly Pick<SessionInfo, '_lineage_root_id' | 'id' | 'profile'>[],
  storedSessionId: string,
  profile: string
): boolean {
  const key = profile.trim() || 'default'

  return sessions.some(session => {
    const owner = (session.profile ?? '').trim() || 'default'

    return owner === key && sessionMatchesStoredId(session, storedSessionId)
  })
}

/**
 * The profile a routed session belongs to, for keying the remembered id.
 *
 * Prefer the owning profile recorded on the session row (the cross-profile
 * aggregator tags each row), so the session is remembered under ITS profile
 * even while a different one is live. Falls back to the active gateway profile
 * for a session not yet in the in-memory list.
 */
export function rememberedSessionProfile(
  sessions: readonly SessionInfo[],
  sessionId: null | string,
  activeProfile: null | string
): string {
  if (sessionId) {
    const owner = sessions.find(session => sessionMatchesStoredId(session, sessionId))?.profile?.trim()

    if (owner) {
      return owner
    }
  }

  return (activeProfile ?? '').trim() || 'default'
}

// The last non-overlay route (a page like /skills, or a session route), so a
// relaunch lands back where you were instead of a bare new-chat.
//
// Scoped per profile for the same reason the remembered session id is: a single
// global key remembered ONE route across every profile, and a session route
// carries a session id in its path. Restoring under profile B would navigate to
// a session owned by profile A — the remembered-id scoping above is bypassed
// entirely, because the route is preferred over the id on cold start
// (#67603 family). Legacy global values are discarded on first read.

export function getRememberedRoute(profile: string): null | string {
  discardLegacyRememberedNavigation()

  return storedString(profileNavigationKey(LAST_ROUTE_KEY, profile))
}

export function setRememberedRoute(path: null | string, profile: string): void {
  discardLegacyRememberedNavigation()
  persistString(profileNavigationKey(LAST_ROUTE_KEY, profile), path)
}

let configuredDefaultProjectDir = ''

function workspaceCwdKey(connection: HermesConnection | null = $connection.get()): string {
  if (connection?.mode !== 'remote') {
    return WORKSPACE_CWD_KEY
  }

  const base = encodeURIComponent(connection.baseUrl || 'remote')
  const profile = encodeURIComponent(connection.profile || 'default')

  return `${WORKSPACE_CWD_KEY}.remote.${base}.${profile}`
}

export const getRememberedWorkspaceCwd = (): string => storedString(workspaceCwdKey())?.trim() || ''
export type NewChatWorkspaceTarget = null | string | undefined

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

  // Transient: each source below is already remembered or comes from config, so
  // persisting would only promote a configured default into the per-backend
  // memory of what the user picked.
  const seedLiveCwd = (cwd: string) => {
    if (cwd && !$activeSessionId.get()) {
      setCurrentCwdTransient(cwd)
    }
  }

  const remembered = getRememberedWorkspaceCwd()

  if ($connection.get()?.mode === 'remote') {
    seedLiveCwd(remembered)

    return
  }

  if (configured) {
    const { cwd } = await sanitize(configured)
    seedLiveCwd(cwd)

    return
  }

  if (remembered) {
    const { cwd } = await sanitize(remembered)
    seedLiveCwd(cwd)
  }
}

export function applyConfiguredDefaultProjectDir(dir: null | string | undefined): void {
  configuredDefaultProjectDir = dir?.trim() || ''
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

/** True when a stored/lineage id resolves to this session — it matches either
 *  the live id or the stable lineage root (see sessionPinId). The one place the
 *  "same conversation across compression" test lives. */
export const sessionMatchesStoredId = (
  session: Pick<SessionInfo, '_lineage_root_id' | 'id'>,
  storedSessionId: string
): boolean => session.id === storedSessionId || session._lineage_root_id === storedSessionId

// Alias lookup, memoized per sessions-list reference. `lineageAliases` runs
// per cached session state per status projection per message delta — an
// O(sessions) scan there multiplies out to states × sessions × ~30Hz per busy
// session, which is what made a populated recents list drag every stream. The
// list is replaced wholesale (never mutated), so its reference is the cache key.
type LineageRow = Pick<SessionInfo, '_lineage_root_id' | 'id'>
const lineageIndexBySessions = new WeakMap<readonly LineageRow[], Map<string, string[]>>()

function lineageIndex(sessions: readonly LineageRow[]): Map<string, string[]> {
  const cached = lineageIndexBySessions.get(sessions)

  if (cached) {
    return cached
  }

  const index = new Map<string, string[]>()

  const add = (key: string, value: string) => {
    const bucket = index.get(key)

    if (!bucket) {
      index.set(key, [value])
    } else if (!bucket.includes(value)) {
      bucket.push(value)
    }
  }

  for (const session of sessions) {
    add(session.id, session.id)

    if (session._lineage_root_id) {
      add(session.id, session._lineage_root_id)
      add(session._lineage_root_id, session.id)
      add(session._lineage_root_id, session._lineage_root_id)
    }
  }

  lineageIndexBySessions.set(sessions, index)

  return index
}

/** Every id one conversation answers to: the id we were handed, plus the live
 *  id and lineage root of each session it resolves to.
 *
 *  Status sets are published under a session's CURRENT stored id, but a sidebar
 *  row, a persisted tile, and the route can each hold a different tip of the
 *  same lineage after a compression. Publishing every alias lets those surfaces
 *  keep using a plain membership test instead of each re-deriving lineage —
 *  and getting it wrong, which reads as a running session going idle mid-turn. */
export function lineageAliases(storedId: string, sessions: readonly LineageRow[]): string[] {
  // Every key is in its own bucket by construction, so the bucket IS the
  // alias set. Copied so no caller can mutate the shared index.
  return lineageIndex(sessions).get(storedId)?.slice() ?? [storedId]
}

/** True when two ids name the same conversation across compression tip rotation. */
export function idsShareLineage(
  a: string,
  b: string,
  sessions: readonly Pick<SessionInfo, '_lineage_root_id' | 'id'>[]
): boolean {
  if (a === b) {
    return true
  }

  return sessions.some(session => sessionMatchesStoredId(session, a) && sessionMatchesStoredId(session, b))
}

/**
 * Whether a composer draft/queue key should move from `fromKey` onto `toKey`.
 *
 * Only same-conversation rekeys are allowed (compression tip → lineage root).
 * A session-switch window where the route already points at B while the store
 * selection still holds A must NOT migrate — that would re-home Session A's
 * queued prompts onto B and auto-drain them into the wrong chat.
 */
export function shouldMigrateComposerScope(
  fromKey: string | null | undefined,
  toKey: string | null | undefined,
  sessions: readonly Pick<SessionInfo, '_lineage_root_id' | 'id'>[]
): boolean {
  const from = fromKey?.trim()
  const to = toKey?.trim()

  if (!from || !to || from === to) {
    return false
  }

  return idsShareLineage(from, to, sessions)
}

/**
 * Stable composer + `/queue` scope for a selected stored session.
 *
 * Same durability rule as {@link sessionPinId}: prefer the lineage root so
 * auto-compression tip rotation does not remount the composer onto an empty
 * draft/queue key mid-keystroke. Falls back to the live id when the row is
 * not in the in-memory list yet.
 */
export function resolveComposerSessionKey(
  selectedSessionId: string | null | undefined,
  sessions: readonly Pick<SessionInfo, '_lineage_root_id' | 'id'>[]
): string | null {
  if (!selectedSessionId) {
    return null
  }

  const row = sessions.find(session => sessionMatchesStoredId(session, selectedSessionId))

  return row ? sessionPinId(row) : selectedSessionId
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
 *  either its live `id` or its `_lineage_root_id`. Optimistic deletes/archives
 *  drop the row from `previous` (and unpin it), so a removed session can't be
 *  resurrected here. */
export function mergeSessionPage(
  previous: SessionInfo[],
  incoming: SessionInfo[],
  keepIds: Iterable<string>
): SessionInfo[] {
  const keep = keepIds instanceof Set ? keepIds : new Set(keepIds)

  // Carry a known title onto a row that arrives title-less, so a freshly
  // submitted session (e.g. a branch draft) holds its placeholder instead of
  // flashing its raw message preview in the gap between persist and the async
  // auto-titler. A real clear sets the local title null first, so this never
  // masks one.
  const prevById = new Map(previous.map(session => [session.id, session]))
  // Tip rotation changes the live id — carry activity/title across the lineage
  // root so a mid-turn refresh can't drop a touchSessionActivity bump.
  const prevByLineage = new Map(previous.map(session => [session._lineage_root_id ?? session.id, session]))

  const merged = incoming.map(session => {
    const prev = prevById.get(session.id) ?? prevByLineage.get(session._lineage_root_id ?? session.id)
    // User-send stamps last_active before the DB flushes the user row
    // (last_active = MAX(messages.timestamp)). Keep the fresher of the two.
    const last_active = Math.max(prev?.last_active ?? 0, session.last_active ?? 0)
    const title = session.title?.trim() ? session.title : prev?.title?.trim() ? prev.title : session.title

    return last_active === session.last_active && title === session.title ? session : { ...session, last_active, title }
  })

  if (keep.size === 0) {
    return merged
  }

  const incomingIds = new Set(merged.map(session => session.id))

  // Deduplicate by compression lineage: when auto-compression rotates the tip
  // id (old #4 → new #5), the incoming page carries the new tip but the
  // previous list still holds the old one.  Without lineage-level dedup both
  // rows survive as separate sidebar entries (fixes #43483).
  const incomingLineageKeys = new Set(merged.map(session => session._lineage_root_id ?? session.id))

  const survivors = previous.filter(
    session =>
      !incomingIds.has(session.id) &&
      !incomingLineageKeys.has(session._lineage_root_id ?? session.id) &&
      (keep.has(session.id) || (session._lineage_root_id != null && keep.has(session._lineage_root_id)))
  )

  return survivors.length ? [...survivors, ...merged] : merged
}

/** Raise a session in recents on user send (before stream / turn resolve). */
export function touchSessionActivity(
  sessionId: string | null | undefined,
  options?: { at?: number; preview?: string }
): void {
  const id = sessionId?.trim()

  if (!id) {
    return
  }

  const at = options?.at ?? Date.now() / 1000
  const preview = options?.preview?.trim().slice(0, 200) || undefined

  setSessions(prev => {
    let changed = false

    const next = prev.map(session => {
      if (!sessionMatchesStoredId(session, id)) {
        return session
      }

      const last_active = Math.max(session.last_active ?? 0, at)

      if (last_active === session.last_active && (!preview || preview === session.preview)) {
        return session
      }

      changed = true

      return preview ? { ...session, last_active, preview } : { ...session, last_active }
    })

    return changed ? next : prev
  })
}

export const $connection = atom<HermesConnection | null>(null)
export const $gatewayState = atom<ConnectionState>('idle')
export const $sessions = atom<SessionInfo[]>([])
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
// Whether a profile's last session page was CAPPED by the request limit, keyed
// by profile name — i.e. more rows exist on disk than were loaded. Replaces the
// old exact per-profile totals: rendering `loaded/total` in the sidebar cost a
// COUNT(*) per profile DB on every refresh and only ever confused people, while
// "is there another page?" is what pagination actually needs and comes free
// from the row count the query already returned.
export const $sessionProfilesTruncated = atom<Record<string, boolean>>({})

/** Tokens and spend per profile across ALL its sessions, not just the loaded
 *  page — summed in SQL so a profile group's header total doesn't move when the
 *  window does. Keyed by profile name. */
export interface ProfileUsage {
  cost_usd: number
  tokens: number
}

export const $sessionProfilesUsage = atom<Record<string, ProfileUsage>>({})
export const $sessionsLoading = atom(true)
export const $activeSessionId = atom<string | null>(null)
export const $selectedStoredSessionId = atom<string | null>(null)
export interface ActiveSessionStoredIdRotation {
  nextStoredSessionId: string
  previousStoredSessionId: string
  runtimeSessionId: string
}

// One-shot event for when auto-compression rotates the active runtime's stored
// id. Carrying the runtime + previous id is load-bearing: a bare next id cannot
// tell whether the user has already navigated away while React is waiting to
// run the route-following effect, which lets a background session steal the
// foreground route.
export const $activeSessionStoredIdRotation = atom<ActiveSessionStoredIdRotation | null>(null)
export const $messages = atom<ChatMessage[]>([])

// Streaming-stable derivations of $messages. During a token stream the array
// is replaced ~30×/s; components that only care about coarse facts (is the
// thread empty? is the tail a user message?) subscribe to these instead of
// $messages so per-token flushes don't re-render them — nanostores' `computed`
// only notifies when the derived VALUE changes.
export const $messagesEmpty = computed($messages, messages => messages.length === 0)
export const $lastVisibleMessageIsUser = computed($messages, lastVisibleMessageIsUser)

export const $freshDraftReady = atom(false)
export const $busy = atom(false)
export const $awaitingResponse = atom(false)
// Stored-session id whose most recent resume FAILED terminally (the gateway RPC
// rejected AND the REST transcript fallback also failed), leaving the window
// with no runtime and an empty transcript. Drives use-route-resume's self-heal:
// while this matches the routed session the loader would otherwise latch
// forever (messagesEmpty && !activeSessionId), so the hook re-attempts the
// resume on the next render/focus/reconnect instead of stranding the window.
// Null whenever the active route has a healthy (or in-flight) resume.
export const $resumeFailedSessionId = atom<string | null>(null)
// Stored-session id whose resume has EXHAUSTED its bounded auto-retries (the
// terminal-failure latch above kept failing through all MAX_RESUME_RETRIES
// attempts). Distinct from $resumeFailedSessionId, which is armed *during* the
// backoff window too: this fires only once auto-recovery has given up, so the
// chat view can swap the perpetual loader for an explicit error + manual Retry
// affordance. A fresh resumeSession() (manual Retry, reconnect, reselect)
// clears it and resets the retry counter. Null whenever the active route has a
// healthy, in-flight, or still-auto-retrying resume.
export const $resumeExhaustedSessionId = atom<string | null>(null)
export const $currentModel = atom(storedString(COMPOSER_MODEL_KEY) ?? '')
export const $currentProvider = atom(storedString(COMPOSER_PROVIDER_KEY) ?? '')
export const $currentReasoningEffort = atom(storedString(COMPOSER_EFFORT_KEY) ?? '')
export const $currentServiceTier = atom('')
export const $currentFastMode = atom(storedBoolean(COMPOSER_FAST_KEY, false))
// Effective approval-bypass state mirrored from the gateway (session.info).
// Persistence lives in the backend config (approvals.mode), so this is a plain
// reflection of the truth the gateway reports rather than its own store.
export const $yoloActive = atom(false)
export const $currentCwd = atom(getRememberedWorkspaceCwd())

// Which conversation the live `$currentCwd` is known to describe. Three
// inhabitants, and the difference between the last two is load-bearing:
// a stored-session id (that conversation owns the path), `null` (the fresh-draft
// state, which MATCHES a null selection and therefore reads as OWNED — a draft's
// workspace is immediately usable), and the released marker
// `WORKSPACE_CWD_UNOWNED` below, which matches no selection and so reads as
// owned by nobody. `null` cannot double as the release value precisely because
// it matches: releasing to `null` while a draft is selected would hand the
// leftover path to the draft as its own workspace.
//
// A conversation switch publishes the new stored id immediately, but the new
// workspace only arrives when the resume settles, so for that whole window
// `$currentCwd` still holds the PREVIOUS conversation's folder. Without a way to
// say "this path is not this conversation's yet", workspace-derived surfaces
// treat the leftover path as authoritative and show the old repo's cached Git
// facts under the newly selected chat (#71254).
//
// Ownership, not emptiness, is what makes the switch atomic: clearing the path
// would collapse the workspace panes and drop file-tree state on every switch,
// so the path stays put and is simply marked as not-yet-owned.
export const $workspaceCwdOwner = atom<null | string>(null)

// Terminal execution backend (local | docker | ssh | ...) mirrored from the
// gateway (session.info). Drives attachment upload decisions: container
// backends have their own filesystem, so a dropped host path must be uploaded
// as bytes and staged into a bind-mounted cache dir (#76577).
export const $terminalBackend = atom('')
export const $newChatWorkspaceTarget = atom<NewChatWorkspaceTarget>(undefined)
export const $newChatWorkspaceTargetGeneration = atom(0)
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
export const $sessionPickerOpen = atom(false)

export const setConnection = (next: Updater<HermesConnection | null>) => {
  updateAtom($connection, next)
  syncCronModelImpactConnection($connection.get())
}
export const setGatewayState = (next: Updater<ConnectionState>) => updateAtom($gatewayState, next)
export const setSessions = (next: Updater<SessionInfo[]>) => updateAtom($sessions, next)
export const setCronSessions = (next: Updater<SessionInfo[]>) => updateAtom($cronSessions, next)
export const setMessagingSessions = (next: Updater<SessionInfo[]>) => updateAtom($messagingSessions, next)
export const setMessagingPlatformTotals = (next: Updater<Record<string, number>>) =>
  updateAtom($messagingPlatformTotals, next)
export const setMessagingTruncated = (next: Updater<boolean>) => updateAtom($messagingTruncated, next)
export const setSessionProfilesTruncated = (next: Updater<Record<string, boolean>>) =>
  updateAtom($sessionProfilesTruncated, next)
export const setSessionProfilesUsage = (next: Updater<Record<string, ProfileUsage>>) =>
  updateAtom($sessionProfilesUsage, next)
export const setSessionsLoading = (next: Updater<boolean>) => updateAtom($sessionsLoading, next)
export const setActiveSessionId = (next: Updater<string | null>) => updateAtom($activeSessionId, next)
export const setActiveSessionStoredIdRotation = (next: Updater<ActiveSessionStoredIdRotation | null>) =>
  updateAtom($activeSessionStoredIdRotation, next)

// Transient: a background session finished and the user hasn't opened it since.
// Written by session-states.ts (handleTransition), cleared here on session open.
export const $unreadFinishedSessionIds = atom<string[]>([])

export const markAllSessionsRead = () => {
  if ($unreadFinishedSessionIds.get().length) {
    $unreadFinishedSessionIds.set([])
  }
}

export const setSelectedStoredSessionId = (next: Updater<string | null>) => {
  updateAtom($selectedStoredSessionId, next)
  // Opening a session clears its unread state — the user is now looking at it.
  const id = $selectedStoredSessionId.get()

  if (id && $unreadFinishedSessionIds.get().includes(id)) {
    $unreadFinishedSessionIds.set($unreadFinishedSessionIds.get().filter(x => x !== id))
  }
}

export const setMessages = (next: Updater<ChatMessage[]>) => updateAtom($messages, next)
export const setFreshDraftReady = (next: Updater<boolean>) => updateAtom($freshDraftReady, next)
export const setResumeFailedSessionId = (next: Updater<string | null>) => updateAtom($resumeFailedSessionId, next)
export const setResumeExhaustedSessionId = (next: Updater<string | null>) => updateAtom($resumeExhaustedSessionId, next)
export const setBusy = (next: Updater<boolean>) => updateAtom($busy, next)
export const setAwaitingResponse = (next: Updater<boolean>) => updateAtom($awaitingResponse, next)

export const setCurrentModel = (next: Updater<string>) => {
  updateAtom($currentModel, next)
  persistString(COMPOSER_MODEL_KEY, $currentModel.get() || null)
}

export const setCurrentProvider = (next: Updater<string>) => {
  updateAtom($currentProvider, next)
  persistString(COMPOSER_PROVIDER_KEY, $currentProvider.get() || null)
}

export const getCurrentModelSource = (): ComposerModelSource => {
  const source = storedString(COMPOSER_MODEL_SOURCE_KEY)

  return source === 'default' || source === 'manual' ? source : ''
}

// Reactive mirror of the persisted source so UI (the composer pill's
// override badge) can subscribe. The getter above stays storage-backed —
// it's read cross-window, where this atom wouldn't see writes.
export const $currentModelSource = atom<ComposerModelSource>(getCurrentModelSource())

export const setCurrentModelSource = (source: ComposerModelSource) => {
  persistString(COMPOSER_MODEL_SOURCE_KEY, source || null)
  $currentModelSource.set(source)
}

// Monotonic intent token for async default refreshes. A profile/config request
// may start before the user opens the picker and finish after their click; the
// token lets that older response stand down even when the selected value is
// unchanged (value comparisons alone cannot detect re-selecting the same row).
let composerSelectionGeneration = 0

export const getComposerSelectionGeneration = (): number => composerSelectionGeneration

export const markComposerSelectionManual = (): void => {
  composerSelectionGeneration += 1
  setCurrentModelSource('manual')
}

export const setCurrentReasoningEffort = (next: Updater<string>) => {
  updateAtom($currentReasoningEffort, next)
  persistString(COMPOSER_EFFORT_KEY, $currentReasoningEffort.get() || null)
}

// The profile's `agent.reasoning_effort`, mirrored from config so surfaces that
// need to render or apply "the default" resolve the user's configured level
// instead of assuming DEFAULT_REASONING_EFFORT (lib/reasoning-effort). Empty
// until config loads, and re-seeded on every profile switch by useHermesConfig.
export const $defaultReasoningEffort = atom('')

export const setDefaultReasoningEffort = (next: string) => updateAtom($defaultReasoningEffort, next)

export const setCurrentServiceTier = (next: Updater<string>) => updateAtom($currentServiceTier, next)

export const setCurrentFastMode = (next: Updater<boolean>) => {
  updateAtom($currentFastMode, next)
  persistBoolean(COMPOSER_FAST_KEY, $currentFastMode.get())
}

export const setYoloActive = (next: Updater<boolean>) => updateAtom($yoloActive, next)

/** Move the live workspace AND remember it as this backend's workspace.
 *
 *  Only for a path the user chose — a folder pick, a project/worktree entry, an
 *  explicit workspace target. The remembered value is where a new chat starts on
 *  a remote backend, so writing it from a path the user merely *looked at* makes
 *  every new chat land in the last session's folder (#77496, #80213). To follow
 *  a conversation's cwd, use `setCurrentCwdTransient`.
 */
export const setCurrentCwd = (next: Updater<string>) => {
  updateAtom($currentCwd, next)
  persistString(workspaceCwdKey(), $currentCwd.get().trim() || null)
}

export const setTerminalBackend = (next: Updater<string>) => updateAtom($terminalBackend, next)

/** Move the live workspace without claiming it as the user's chosen one.
 *
 *  For paths that come from a conversation rather than from the user: resume
 *  settling, a warm switch, the agent relocating mid-turn, detaching a draft.
 */
export const setCurrentCwdTransient = (next: Updater<string>) => updateAtom($currentCwd, next)

// Released-ownership marker: the live path belongs to no conversation. `null`
// cannot serve as the release value because it MATCHES a fresh draft (whose
// selected id is also null), which would declare a leftover path to be the
// draft's own workspace — #71254, one selection over. Kept here beside the atom
// and the comparison so a release site cannot reinvent a subtly different value.
const WORKSPACE_CWD_UNOWNED = 'desktop:workspace-cwd-unowned'

/** Mark the live workspace as belonging to `storedSessionId`.
 *
 *  Call this wherever a cwd is established for a conversation (resume settling,
 *  a warm switch, an explicit folder pick). Until it is called for the newly
 *  selected conversation, primary workspace-derived selectors hide the previous
 *  conversation's cached facts rather than publishing them (#71254).
 */
export const setWorkspaceCwdOwner = (storedSessionId: null | string) => updateAtom($workspaceCwdOwner, storedSessionId)

/** Declare that no conversation owns the live workspace path.
 *
 *  For a conversation whose workspace is not known yet: the path on screen is
 *  provably still the previous conversation's, so workspace-derived surfaces must
 *  hide it rather than adopt it. The path itself is deliberately left alone —
 *  clearing it would collapse the workspace/review panes and drop file-tree
 *  state on every switch.
 */
export const releaseWorkspaceCwdOwner = () => updateAtom($workspaceCwdOwner, WORKSPACE_CWD_UNOWNED)

/** Commit `cwd` as the workspace of the conversation the user is looking at.
 *
 *  The single primitive for "this path IS the selected conversation's" — a folder
 *  pick, a project entry, the agent relocating itself. Prefer it over a bare
 *  `setCurrentCwdTransient`, which moves the path while leaving ownership naming
 *  whatever held it before; workspace-derived slices then stay hidden even though
 *  the path is correct (#71254).
 */
export const commitWorkspaceCwdForSelectedSession = (cwd: string) => {
  setCurrentCwdTransient(cwd)
  setWorkspaceCwdOwner($selectedStoredSessionId.get())
}

/** True when `$currentCwd` is known to describe the selected conversation. */
export const workspaceCwdBelongsToSelectedSession = (): boolean =>
  ($workspaceCwdOwner.get() ?? null) === ($selectedStoredSessionId.get() ?? null)

export const setNewChatWorkspaceTarget = (next: NewChatWorkspaceTarget): number => {
  const generation = $newChatWorkspaceTargetGeneration.get() + 1
  $newChatWorkspaceTarget.set(next)
  $newChatWorkspaceTargetGeneration.set(generation)

  return generation
}

export const workspaceCwdForNewSession = (): string => {
  if ($connection.get()?.mode === 'remote') {
    return getRememberedWorkspaceCwd()
  }

  // A bare new chat starts DETACHED — no inherited cwd, so the composer's coding
  // rail (which keys off $currentCwd) shows no branch and the first message runs
  // in the gateway's default rather than silently in the last repo you touched.
  // Only an explicit default-project-dir setting pre-attaches. Entering a
  // project/worktree attaches its cwd directly (startSessionInWorkspace), so the
  // "remember where I was when I'm in a project" case is unaffected.
  return getConfiguredDefaultProjectDir()
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
export const setSessionPickerOpen = (next: Updater<boolean>) => updateAtom($sessionPickerOpen, next)
